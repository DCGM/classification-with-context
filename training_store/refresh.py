"""Refresh text embeddings and preserve raw sparse stats using sequential LMDB cursors.

Image bytes, document layout, labels, sampling catalogue and pagination are
copied from a verified V1 store. No OCR or page text is accessed. A temporary
on-disk key index routes sequential source reads to the new NVMe shards.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import lmdb
import numpy as np

from page_sequence.features import FeaturePipeline
from embeddings.text.page_stats import PageStats
from .build import code_signature, validate
from .format import (DOCUMENT, RECORD, SPARSE_RECORD, SPARSE_VERSION,
                     atomic_json, digest)


def status(root, phase, **values):
    data = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), **values)
    atomic_json(root / 'status.json', data)
    print(json.dumps(data), flush=True)


def sync_arrays(arrays):
    for array in arrays:
        array.flush()


def make_plan(base, output, sources):
    old = json.loads((base / 'manifest.json').read_text())
    if old['version'] != 1 or not old['complete']:
        raise ValueError('refresh requires a published V1 input')
    features = FeaturePipeline.from_config(sources)
    if features.state_dict()['stats_metadata'] != old['feature_state']['stats_metadata']:
        raise ValueError('raw statistics schema changed')
    plan = {**old, 'version': SPARSE_VERSION, 'record_bytes': SPARSE_RECORD.itemsize,
            'record_dtype': SPARSE_RECORD.descr, 'complete': False,
            'stats_storage': 'raw_sparse_uint16_float16_document_order',
            'feature_state': features.state_dict(), 'source_config': sources,
            'reference_config': sources, 'sources': features.source_provenance(),
            'reference_sources': features.source_provenance(),
            'base_manifest_sha256': digest(base / 'manifest.json'),
            'code_sha256': code_signature(), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    plan.pop('shards', None)
    plan.pop('build_id', None)
    import hashlib
    plan['build_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def copy_metadata(base, output, plan):
    names = ['catalogue.sqlite', 'documents.npy', 'pagination-candidates-v1.npy',
             'pagination-candidates-v1.json']
    for name in names:
        src, dst = base / name, output / name
        if not src.exists():
            if name.startswith('pagination'):
                continue
            raise FileNotFoundError(src)
        expected = (plan['index_sha256'] if name == 'catalogue.sqlite' else
                    plan['documents_sha256'] if name == 'documents.npy' else digest(src))
        if not dst.exists():
            partial = dst.with_name(dst.name + '.partial')
            shutil.copyfile(src, partial)
            if digest(partial) != expected:
                raise IOError('metadata copy checksum mismatch')
            partial.rename(dst)
        elif digest(dst) != expected:
            raise ValueError('metadata changed during resume')


def copy_images(base, output, plan):
    checksums = json.loads((base / 'checksums.json').read_text())
    for shard, count in enumerate(plan['shard_pages']):
        marker = output / 'state' / f'base-{shard:06d}.json'
        if marker.exists():
            continue
        source = base / 'shards' / f'{shard:06d}.bin'
        partial = output / 'shards' / f'{shard:06d}.base-partial'
        import hashlib
        h = hashlib.sha256()
        copied = 0
        with source.open('rb') as inp, partial.open('wb') as out:
            while raw := inp.read(RECORD.itemsize * 4096):
                h.update(raw)
                records = np.frombuffer(raw, RECORD)
                target = np.zeros(len(records), SPARSE_RECORD)
                for name in ('image', 'row', 'document', 'order', 'weak', 'gold', 'placement'):
                    target[name] = records[name]
                target['availability'] = records['availability'] & 1
                out.write(target.tobytes())
                copied += len(records)
            out.flush()
            os.fsync(out.fileno())
        if copied != count or h.hexdigest() != checksums[f'shards/{shard:06d}.bin']:
            raise ValueError('base shard is corrupt')
        partial.rename(output / 'shards' / f'{shard:06d}.bin')
        atomic_json(marker, {'pages': count})
        status(output, 'copying_image_records', completed_shards=shard + 1,
               total_shards=len(plan['shard_pages']))


def key_index(output):
    path = output / 'scratch/locations.sqlite'
    marker = output / 'state/key-index.json'
    if marker.exists():
        return
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('ATTACH DATABASE ? AS corpus', (str(output / 'catalogue.sqlite'),))
    db.execute("CREATE TABLE locations AS SELECT d.library || '_' || p.page AS key, "
               'p.id AS row_id,p.document AS document FROM corpus.pages p '
               'JOIN corpus.documents d ON d.id=p.document')
    db.execute('CREATE INDEX by_key ON locations(key,row_id)')
    db.commit()
    count = db.execute('SELECT count(*) FROM locations').fetchone()[0]
    db.close()
    atomic_json(marker, {'rows': count})


def array_file(path, dtype, shape):
    if path.exists():
        a = np.load(path, mmap_mode='r+', allow_pickle=False)
        if a.dtype != np.dtype(dtype) or a.shape != shape:
            raise ValueError('scratch array layout mismatch')
        return a
    a = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
    a[:] = 0
    a.flush()
    return a


def check_sparse(raw):
    if len(raw) % 4:
        raise ValueError('invalid sparse statistics byte length')
    n = len(raw) // 4
    index = np.frombuffer(raw, '<u2', n)
    values = np.frombuffer(raw, '<f2', n, offset=2 * n)
    if (np.any(index >= len(PageStats.features)) or np.any(index[1:] <= index[:-1])
            or not np.isfinite(values).all()):
        raise ValueError('invalid sparse statistics indices or values')


def ingest(output, plan, docs, kind, owners, offsets=None, sizes=None):
    """Merge each LMDB's sorted cursor with the on-disk page-key index.

    Public-first source precedence is preserved. Replaying the inclusive
    checkpoint key is idempotent, including duplicate keys shared by documents.
    """
    fds = {}
    payload = (output / 'scratch/raw-stats.bin').open('ab') if kind == 'stats' else None
    arrays = [owners] if kind == 'text' else [owners, offsets, sizes]
    db = sqlite3.connect(f'file:{output}/scratch/locations.sqlite?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-32768')
    field_offset = SPARSE_RECORD.fields['text'][1]
    try:
        for owner, (scope, path) in enumerate(plan['source_config']['sources'][kind]['stores'].items(), 1):
            marker = output / 'state' / f'{kind}-{scope}.json'
            state = json.loads(marker.read_text()) if marker.exists() else {}
            if state.get('complete'):
                continue
            last = state.get('last_key', '')
            env = lmdb.open(path, readonly=True, lock=False, readahead=True, create=False)
            matched, visited = state.get('matched', 0), 0
            tick = time.monotonic()
            with env.begin() as txn:
                if kind == 'stats':
                    meta = txn.get(b'__metadata__')
                    if meta is None or json.loads(meta) != PageStats().metadata():
                        raise ValueError('incompatible raw statistics source')
                cursor = txn.cursor()
                valid = cursor.set_range(last.encode()) if last else cursor.first()
                source_key = cursor.key() if valid else None
                for key, row, did in db.execute(
                        'SELECT key,row_id,document FROM locations WHERE key>=? ORDER BY key,row_id', (last,)):
                    if not docs[did]['length'] or (owners[row] and owners[row] < owner):
                        continue
                    wanted = key.encode()
                    while valid and source_key < wanted:
                        valid = cursor.next()
                        source_key = cursor.key() if valid else None
                    if not valid:
                        break
                    if source_key == wanted:
                        raw = cursor.value()
                        if kind == 'stats':
                            check_sparse(raw)
                            offsets[row], sizes[row] = payload.tell(), len(raw)
                            payload.write(raw)
                        else:
                            vector = np.frombuffer(raw, '<f2')
                            if vector.shape != (768,) or not np.isfinite(vector).all():
                                raise ValueError('invalid replacement text embedding')
                            shard = int(docs[did]['shard'])
                            if shard not in fds:
                                fds[shard] = os.open(output / 'shards' / f'{shard:06d}.bin', os.O_RDWR)
                            position = (int(docs[did]['offset']) + row - int(docs[did]['first']))
                            if os.pwrite(fds[shard], raw, position * SPARSE_RECORD.itemsize + field_offset) != len(raw):
                                raise IOError('short text embedding write')
                        matched += int(owners[row] == 0)
                        owners[row] = owner
                    visited += 1
                    if time.monotonic() - tick >= 30:
                        if payload:
                            payload.flush()
                            os.fsync(payload.fileno())
                        for fd in fds.values():
                            os.fsync(fd)
                        sync_arrays(arrays)
                        atomic_json(marker, {'last_key': key, 'matched': matched})
                        status(output, f'ingesting_{kind}', source=scope, matched=matched)
                        tick = time.monotonic()
                cursor.close()
            env.close()
            if payload:
                payload.flush()
                os.fsync(payload.fileno())
            for fd in fds.values():
                os.fsync(fd)
            sync_arrays(arrays)
            atomic_json(marker, {'complete': True, 'matched': matched})
            status(output, f'ingested_{kind}', source=scope, matched=matched)
    finally:
        for fd in fds.values():
            os.close(fd)
        if payload:
            payload.close()
        db.close()


def finalize(output, plan, text_owner, stats_owner, offsets, sizes):
    reports = []
    raw_fd = os.open(output / 'scratch/raw-stats.bin', os.O_RDONLY)
    try:
        for shard, count in enumerate(plan['shard_pages']):
            path = output / 'shards' / f'{shard:06d}.bin'
            stats_path = path.with_name(f'{shard:06d}.stats.bin')
            done = path.with_suffix('.done.json')
            if done.exists():
                report = json.loads(done.read_text())
                if (report['build_id'] != plan['build_id'] or digest(path) != report['sha256']
                        or digest(stats_path) != report['stats_sha256']):
                    raise ValueError('completed sparse shard changed')
                reports.append(report)
                continue
            records = np.memmap(path, dtype=SPARSE_RECORD, mode='r+', shape=(count,))
            position = 0
            present = np.zeros(3, np.int64)
            with stats_path.open('wb') as out:
                for start in range(0, count, 512):
                    block = records[start:start + 512].copy()
                    for rec in block:
                        row = int(rec['row'])
                        size = int(sizes[row])
                        raw = os.pread(raw_fd, size, int(offsets[row])) if size else b''
                        if len(raw) != size:
                            raise IOError('short scratch statistics read')
                        rec['stats_offset'], rec['stats_bytes'] = position, size
                        position += size
                        out.write(raw)
                        rec['availability'] = (rec['availability'] & 1) | (2 if text_owner[row] else 0) | (4 if stats_owner[row] else 0)
                    records[start:start + len(block)] = block
                    for i in range(3):
                        present[i] += np.count_nonzero(block['availability'] & (1 << i))
                out.flush()
                os.fsync(out.fileno())
            records.flush()
            del records
            with path.open('rb') as f:
                os.fsync(f.fileno())
            report = dict(shard=shard, pages=count, build_id=plan['build_id'], sha256=digest(path),
                          stats_bytes=position, stats_sha256=digest(stats_path),
                          present=dict(zip(('image', 'text', 'stats'), map(int, present))))
            atomic_json(done, report)
            reports.append(report)
            status(output, 'finalizing_document_order', completed_shards=len(reports),
                   total_shards=len(plan['shard_pages']))
    finally:
        os.close(raw_fd)
    return reports


def run(base, output, sources, sample_pages=2048):
    base, output = Path(base).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('shards', 'state', 'scratch'):
        (output / name).mkdir(exist_ok=True)
    with (output / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'manifest.json').exists():
            return
        proposed = make_plan(base, output, sources)
        plan_path = output / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            for key in ('source_config', 'sources', 'base_manifest_sha256', 'code_sha256'):
                if proposed[key] != plan[key]:
                    raise ValueError(f'resume input changed: {key}')
        else:
            plan = proposed
            headroom = min(10 << 30, max(16 << 20, plan['pages'] * 256))
            required = (plan['pages'] * (SPARSE_RECORD.itemsize + 600)
                        + (base / 'catalogue.sqlite').stat().st_size + headroom)
            if shutil.disk_usage(output).free < required:
                raise ValueError('insufficient space for new store and bounded scratch files')
            atomic_json(plan_path, plan)
        status(output, 'copying_catalogue')
        copy_metadata(base, output, plan)
        docs = np.load(output / 'documents.npy', mmap_mode='r', allow_pickle=False)
        if docs.dtype != DOCUMENT:
            raise ValueError('document table layout mismatch')
        copy_images(base, output, plan)
        status(output, 'building_key_index')
        key_index(output)
        n = int(np.max(docs['first'].astype(np.uint64) + docs['length'])) + 1
        owner_text = array_file(output / 'scratch/text-owner.npy', 'u1', (n,))
        owner_stats = array_file(output / 'scratch/stats-owner.npy', 'u1', (n,))
        offsets = array_file(output / 'scratch/stats-offset.npy', '<u8', (n,))
        sizes = array_file(output / 'scratch/stats-size.npy', '<u4', (n,))
        ingest(output, plan, docs, 'text', owner_text)
        ingest(output, plan, docs, 'stats', owner_stats, offsets, sizes)
        reports = finalize(output, plan, owner_text, owner_stats, offsets, sizes)
        if FeaturePipeline.from_config(sources).source_provenance() != plan['sources']:
            raise ValueError('source changed during build')
        status(output, 'validating', sample_pages=sample_pages)
        validation = validate(output, plan, sample_pages)
        atomic_json(output / 'validation-report.json', validation)
        checksums = {f"shards/{r['shard']:06d}.bin": r['sha256'] for r in reports}
        checksums.update({f"shards/{r['shard']:06d}.stats.bin": r['stats_sha256'] for r in reports})
        for name in ('catalogue.sqlite', 'documents.npy', 'pagination-candidates-v1.npy', 'pagination-candidates-v1.json'):
            if (output / name).exists():
                checksums[name] = digest(output / name)
        atomic_json(output / 'checksums.json', checksums)
        atomic_json(output / 'manifest.json', plan | {'complete': True, 'shards': reports})
        del owner_text, owner_stats, offsets, sizes
        shutil.rmtree(output / 'scratch')
        (output / 'failure.json').unlink(missing_ok=True)
        status(output, 'complete', pages=plan['pages'], validation=validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--validation-pages', type=int, default=2048)
    args = parser.parse_args()
    try:
        run(args.base, args.output, json.loads(Path(args.sources).read_text()), args.validation_pages)
    except BaseException as exc:
        path = Path(args.output)
        if path.exists() and not (path / 'manifest.json').exists():
            atomic_json(path / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
