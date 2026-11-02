"""Resume verified directory relocations without exposing file contents.

Only after every source file has a SHA256-identical destination is the source
replaced with a link. Journals are outside the relocated directories.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .format import atomic_json


def signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev]


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def relocate(source, destination, journal, log=print, replace_source=True):
    source, destination, journal = map(Path, (source, destination, journal))
    quarantine = source.with_name(source.name + '.verified-relocation')
    state = json.loads(journal.read_text()) if journal.exists() else {'files': {}}
    if source.is_symlink():
        if source.resolve() != destination.resolve() or not state.get('verified'):
            raise ValueError('unexpected source link or missing verification journal')
        if quarantine.exists():
            shutil.rmtree(quarantine)
        return
    if quarantine.exists():
        raise ValueError('unexpected relocation quarantine')
    paths = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('source tree contains links; relocate the actual directory explicitly')
    files = [p for p in paths if p.is_file()]
    destination.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in files)
    done = 0
    expected = (json.loads((source / 'checksums.json').read_text())
                if (source / 'checksums.json').is_file() else {})
    for src in files:
        rel = str(src.relative_to(source))
        dst = destination / rel
        before = signature(src)
        previous = state['files'].get(rel)
        if (previous and before == previous['source'] and dst.is_file()
                and signature(dst) == previous['destination']):
            done += before[0]
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            digest = file_hash(src)
            if file_hash(dst) != digest:
                raise ValueError(f'existing destination differs: {dst}')
        else:
            partial = dst.with_name(dst.name + '.relocation-partial')
            h = hashlib.sha256()
            tick = time.monotonic()
            copied = 0
            with src.open('rb') as inp, partial.open('wb') as out:
                for block in iter(lambda: inp.read(16 << 20), b''):
                    out.write(block)
                    h.update(block)
                    copied += len(block)
                    if time.monotonic() - tick >= 30:
                        log(json.dumps({'phase': 'copying', 'source': str(source), 'file': rel,
                                        'bytes': done + copied, 'total_bytes': total}), flush=True)
                        tick = time.monotonic()
                out.flush()
                os.fsync(out.fileno())
            digest = h.hexdigest()
            if file_hash(partial) != digest:
                raise IOError('destination checksum mismatch')
            shutil.copystat(src, partial)
            partial.rename(dst)
        if signature(src) != before or (rel in expected and expected[rel] != digest):
            raise ValueError('source changed or disagrees with published checksum')
        state['files'][rel] = dict(source=before, destination=signature(dst), sha256=digest)
        atomic_json(journal, state)
        done += before[0]
        log(json.dumps({'phase': 'verified_file', 'source': str(source), 'file': rel,
                        'bytes': done, 'total_bytes': total}), flush=True)
    if set(state['files']) != {str(p.relative_to(source)) for p in files}:
        raise ValueError('source file list changed')
    for src in files:
        if signature(src) != state['files'][str(src.relative_to(source))]['source']:
            raise ValueError('source changed after copy')
    state['verified'] = True
    atomic_json(journal, state)
    if not replace_source:
        return
    source.rename(quarantine)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except BaseException:
        quarantine.rename(source)
        raise
    shutil.rmtree(quarantine)
    log(json.dumps({'phase': 'relocated', 'source': str(source), 'destination': str(destination),
                    'bytes': total}), flush=True)


def run(plan):
    root = Path(plan['journal'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i, operation in enumerate(plan['operations']):
            atomic_json(root / 'status.json', {'phase': 'relocating', 'operation': i, **operation})
            relocate(operation['source'], operation['destination'], root / f'{i:02d}.json')
        atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        run(plan)
    except BaseException as exc:
        atomic_json(Path(plan['journal']) / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
