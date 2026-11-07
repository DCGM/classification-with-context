"""Replace the image embeddings of a published sparse (V2) store, keeping everything else bit for bit.

Text vectors, raw sparse statistics, labels, document layout, catalogue and pagination are copied from
the base store; only the image field and its availability bit change. Image vectors come from the new
public/private image LMDBs (keys: image path below the public/private root, as `LmdbVectors` reads them),
merged with a sorted on-disk key index. Text and statistics LMDBs are not needed: their provenance is
taken from the base manifest. Every non-image byte of every shard is verified against the base store, and
sampled image vectors are checked against the new LMDBs. Resumable; the store is published atomically.

Example:
    python -m training_store.refresh_images --base ~/data/classification-with-context/document-store-ocr-lines-sparse-v2 \\
        --output ~/data/classification-with-context/document-store-ocr-lines-sparse-v2-dino-orig \\
        --image-public ~/data/classification-with-context/digiknihovna.data/public/image/vitb14_dino2_orig/lmdb \\
        --image-private ~/data/classification-with-context/digiknihovna.data/private/image/vitb14_dino2_orig/lmdb
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import lmdb
import numpy as np

from page_sequence.features import LmdbVectors
from .build import code_signature
from .format import DOCUMENT, SPARSE_RECORD, SPARSE_VERSION, atomic_json, digest
from .refresh import array_file, copy_metadata, status

IMAGE_ROOTS = {'public': 'public', 'public.256': 'public', 'private': 'private', 'private.256': 'private'}


def image_provenance(stores):
    source = LmdbVectors(stores, 'image', 768)
    result = {}
    for label, path in source.stores.items():
        path = Path(path)
        stat = (path / 'data.mdb').stat()
        meta = path.parent / 'meta.json'
        result[label] = {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'meta_sha256': hashlib.sha256(meta.read_bytes()).hexdigest() if meta.exists() else None}
    return result


def make_plan(base, stores):
    old = json.loads((base / 'manifest.json').read_text())
    if old['version'] != SPARSE_VERSION or not old['complete']:
        raise ValueError('image refresh requires a published sparse (V2) store')
    if set(stores) != {'public', 'private'}:
        raise ValueError('public and private image stores required')
    config = copy.deepcopy(old['source_config'])
    image = config['sources']['image']
    if image.get('dim', 768) != 768 or image.get('dtype', '<f2') != '<f2':
        raise ValueError('image field layout is fixed at 768 float16 values')
    image['stores'] = {label: str(Path(path).resolve()) for label, path in stores.items()}
    provenance = copy.deepcopy(old['reference_sources'])
    provenance['image']['stores'] = image_provenance(image['stores'])
    plan = {**old, 'complete': False, 'source_config': config, 'reference_config': config,
            'sources': provenance, 'reference_sources': provenance,
            'base_manifest_sha256': digest(base / 'manifest.json'), 'base_build_id': old['build_id'],
            'replaced_modality': 'image', 'code_sha256': code_signature(),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for key in ('shards', 'build_id'):
        plan.pop(key, None)
    plan['build_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def copy_shards(base, output, plan):
    """Copy every shard with its image field cleared; the statistics companions unchanged."""
    checksums = json.loads((base / 'checksums.json').read_text())
    for shard, count in enumerate(plan['shard_pages']):
        marker = output / 'state' / f'base-{shard:06d}.json'
        if marker.exists():
            continue
        name = f'{shard:06d}.bin'
        stats = f'{shard:06d}.stats.bin'
        partial = output / 'shards' / (name + '.partial')
        h = hashlib.sha256()
        copied = 0
        with (base / 'shards' / name).open('rb') as inp, partial.open('wb') as out:
            while raw := inp.read(SPARSE_RECORD.itemsize * 4096):
                h.update(raw)
                records = np.frombuffer(raw, SPARSE_RECORD).copy()
                records['image'] = 0
                records['availability'] &= np.uint8(0xFE)
                out.write(records.tobytes())
                copied += len(records)
            out.flush()
            os.fsync(out.fileno())
        if copied != count or h.hexdigest() != checksums[f'shards/{name}']:
            raise ValueError(f'base shard {shard} is corrupt')
        stats_partial = output / 'shards' / (stats + '.partial')
        shutil.copyfile(base / 'shards' / stats, stats_partial)
        if digest(stats_partial) != checksums[f'shards/{stats}']:
            raise ValueError(f'base statistics shard {shard} is corrupt')
        partial.rename(output / 'shards' / name)
        stats_partial.rename(output / 'shards' / stats)
        atomic_json(marker, {'pages': count})
        status(output, 'copying_base_records', completed_shards=shard + 1, total_shards=len(plan['shard_pages']))


def key_index(output):
    """Sorted (scope, key, row) for every page with an image path; key as `LmdbVectors` derives it."""
    path = output / 'scratch/image-locations.sqlite'
    marker = output / 'state/image-key-index.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE locations (scope TEXT, key TEXT, row_id INTEGER, document INTEGER)')
    corpus = sqlite3.connect(f'file:{output}/catalogue.sqlite?mode=ro', uri=True)
    counts = {'public': 0, 'private': 0, 'without_path': 0}
    rows = corpus.execute("SELECT path, id, document FROM pages")
    while batch := rows.fetchmany(100_000):
        values = []
        for image_path, row, document in batch:
            if not image_path:
                counts['without_path'] += 1
                continue
            root, key = image_path.split('/', 1)
            scope = IMAGE_ROOTS.get(root)
            if scope is None:
                raise ValueError('unsupported image path root')
            counts[scope] += 1
            values.append((scope, key, row, document))
        db.executemany('INSERT INTO locations VALUES (?,?,?,?)', values)
    corpus.close()
    db.execute('CREATE INDEX by_key ON locations(scope, key, row_id)')
    db.commit()
    db.close()
    atomic_json(marker, counts)
    return counts


def ingest(output, plan, docs, owners):
    """Write each new image vector into its record. Public-first, as the live reader resolves paths."""
    db = sqlite3.connect(f'file:{output}/scratch/image-locations.sqlite?mode=ro', uri=True)
    field = SPARSE_RECORD.fields['image'][1]
    fds = {}
    try:
        for owner, (scope, path) in enumerate(plan['source_config']['sources']['image']['stores'].items(), 1):
            marker = output / 'state' / f'image-{scope}.json'
            state = json.loads(marker.read_text()) if marker.exists() else {}
            if state.get('complete'):
                continue
            last, matched = state.get('last_key', ''), state.get('matched', 0)
            env = lmdb.open(path, readonly=True, lock=False, readahead=True, create=False)
            tick = time.monotonic()
            with env.begin() as txn:
                cursor = txn.cursor()
                valid = cursor.set_range(last.encode()) if last else cursor.first()
                source_key = cursor.key() if valid else None
                for key, row, did in db.execute('SELECT key, row_id, document FROM locations '
                                                'WHERE scope=? AND key>=? ORDER BY key, row_id', (scope, last)):
                    if not docs[did]['length']:
                        continue
                    wanted = key.encode()
                    while valid and source_key < wanted:
                        valid = cursor.next()
                        source_key = cursor.key() if valid else None
                    if not valid:
                        break
                    if source_key != wanted:
                        continue
                    raw = cursor.value()
                    vector = np.frombuffer(raw, '<f2')
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError('invalid image embedding')
                    shard = int(docs[did]['shard'])
                    if shard not in fds:
                        fds[shard] = os.open(output / 'shards' / f'{shard:06d}.bin', os.O_RDWR)
                    position = int(docs[did]['offset']) + row - int(docs[did]['first'])
                    if os.pwrite(fds[shard], raw, position * SPARSE_RECORD.itemsize + field) != len(raw):
                        raise IOError('short image embedding write')
                    matched += int(owners[row] == 0)
                    owners[row] = owner
                    if time.monotonic() - tick >= 30:
                        for fd in fds.values():
                            os.fsync(fd)
                        owners.flush()
                        atomic_json(marker, {'last_key': key, 'matched': matched})
                        status(output, 'ingesting_image', source=scope, matched=matched)
                        tick = time.monotonic()
                cursor.close()
            env.close()
            for fd in fds.values():
                os.fsync(fd)
            owners.flush()
            atomic_json(marker, {'complete': True, 'matched': matched})
            status(output, 'ingested_image', source=scope, matched=matched)
    finally:
        for fd in fds.values():
            os.close(fd)
        db.close()


def finalize(base, output, plan, owners):
    """Set the image availability bit; verify every non-image byte against the base shard."""
    old = {r['shard']: r for r in json.loads((base / 'manifest.json').read_text())['shards']}
    reports, changed = [], {'image_present_before': 0, 'image_present_after': 0, 'gained': 0, 'lost': 0}
    for shard, count in enumerate(plan['shard_pages']):
        path = output / 'shards' / f'{shard:06d}.bin'
        done = path.with_suffix('.done.json')
        if done.exists():
            report = json.loads(done.read_text())
            if report['build_id'] != plan['build_id'] or digest(path) != report['sha256']:
                raise ValueError('completed shard changed')
            reports.append(report)
            for k in changed:
                changed[k] += report['image_change'][k]
            continue
        records = np.memmap(path, dtype=SPARSE_RECORD, mode='r+', shape=(count,))
        before = np.memmap(base / 'shards' / path.name, dtype=SPARSE_RECORD, mode='r', shape=(count,))
        present = np.zeros(3, np.int64)
        delta = dict.fromkeys(changed, 0)
        for start in range(0, count, 65536):
            block = np.array(records[start:start + 65536])
            reference = np.array(before[start:start + 65536])
            has = owners[block['row'].astype(np.int64)] != 0
            block['availability'] = (block['availability'] & np.uint8(0xFE)) | has.astype(np.uint8)
            others = [n for n in SPARSE_RECORD.names if n not in ('image', 'availability')]
            if any(not np.array_equal(block[n], reference[n]) for n in others) or not np.array_equal(
                    block['availability'] & 0xFE, reference['availability'] & 0xFE):
                raise ValueError(f'non-image content of shard {shard} differs from the base store')
            if np.any(block['image'][~has]):
                raise ValueError('image vector without an image source')
            records[start:start + len(block)] = block
            had = (reference['availability'] & 1) != 0
            delta['image_present_before'] += int(had.sum())
            delta['image_present_after'] += int(has.sum())
            delta['gained'] += int((has & ~had).sum())
            delta['lost'] += int((had & ~has).sum())
            for i in range(3):
                present[i] += np.count_nonzero(block['availability'] & (1 << i))
        records.flush()
        del records, before
        with path.open('rb') as f:
            os.fsync(f.fileno())
        stats_path = path.with_name(f'{shard:06d}.stats.bin')
        report = dict(shard=shard, pages=count, build_id=plan['build_id'], sha256=digest(path),
                      stats_bytes=old[shard]['stats_bytes'], stats_sha256=old[shard]['stats_sha256'],
                      present=dict(zip(('image', 'text', 'stats'), map(int, present))), image_change=delta)
        if stats_path.stat().st_size != report['stats_bytes']:
            raise ValueError('statistics shard size changed')
        atomic_json(done, report)
        reports.append(report)
        for k in changed:
            changed[k] += delta[k]
        status(output, 'finalizing', completed_shards=len(reports), total_shards=len(plan['shard_pages']))
    return reports, changed


def validate_images(output, plan, docs, sample_pages):
    """Sampled records against the new LMDBs, read the way training's live reader reads them."""
    from page_sequence.index import PageIndex
    stores = plan['source_config']['sources']['image']['stores']
    source = LmdbVectors(stores, 'image', 768)
    index = PageIndex(output / 'catalogue.sqlite')
    ids = np.flatnonzero(docs['length'])
    lengths = docs['length'][ids].astype(np.int64)
    cumulative = np.cumsum(lengths)
    draws = np.random.default_rng(29).integers(0, int(cumulative[-1]), size=sample_pages)
    pos = np.searchsorted(cumulative, draws, side='right')
    rows = docs['first'][ids][pos].astype(np.int64) + draws - np.concatenate(([0], cumulative[:-1]))[pos]
    checked = present = 0
    for start in range(0, len(rows), 256):
        chunk = [int(r) for r in rows[start:start + 256]]
        expected, has = source.read(index.pages(chunk))
        for row, vector, available in zip(chunk, expected, has):
            record = read_record(output, docs, ids, row)
            if bool(record['availability'] & 1) != bool(available):
                raise ValueError('image presence differs from the new LMDBs')
            if not np.array_equal(record['image'], vector.astype(np.float16)):
                raise ValueError('image vector differs from the new LMDBs')
            checked += 1
            present += int(available)
    return {'sample_pages': checked, 'with_image': present}


def read_record(output, docs, ids, row):
    position = int(np.searchsorted(docs['first'][ids], row, side='right')) - 1
    d = docs[ids[position]]
    offset = int(d['offset']) + row - int(d['first'])
    with (output / 'shards' / f"{int(d['shard']):06d}.bin").open('rb') as f:
        f.seek(offset * SPARSE_RECORD.itemsize)
        record = np.frombuffer(f.read(SPARSE_RECORD.itemsize), SPARSE_RECORD)[0]
    if int(record['row']) != row:
        raise ValueError('record/row mapping mismatch')
    return record


def run(base, output, stores, sample_pages=2048):
    base, output = Path(base).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('shards', 'state', 'scratch'):
        (output / name).mkdir(exist_ok=True)
    with (output / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'manifest.json').exists():
            return
        proposed = make_plan(base, stores)
        plan_path = output / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            for key in ('source_config', 'sources', 'base_manifest_sha256', 'code_sha256'):
                if proposed[key] != plan[key]:
                    raise ValueError(f'resume input changed: {key}')
        else:
            plan = proposed
            required = sum(plan['shard_pages']) * (SPARSE_RECORD.itemsize + 200) + (10 << 30)
            if shutil.disk_usage(output).free < required:
                raise ValueError('insufficient space for the new store')
            atomic_json(plan_path, plan)
        status(output, 'copying_catalogue')
        copy_metadata(base, output, plan)
        docs = np.load(output / 'documents.npy', mmap_mode='r', allow_pickle=False)
        if docs.dtype != DOCUMENT:
            raise ValueError('document table layout mismatch')
        copy_shards(base, output, plan)
        status(output, 'building_key_index')
        locations = key_index(output)
        n = int(np.max(docs['first'].astype(np.uint64) + docs['length'])) + 1
        owners = array_file(output / 'scratch/image-owner.npy', 'u1', (n,))
        ingest(output, plan, docs, owners)
        reports, change = finalize(base, output, plan, owners)
        if image_provenance(plan['source_config']['sources']['image']['stores']) != plan['sources']['image']['stores']:
            raise ValueError('image source changed during build')
        status(output, 'validating', sample_pages=sample_pages)
        validation = dict(validate_images(output, plan, docs, sample_pages), image_change=change,
                          image_paths=locations, non_image_bytes='verified identical to the base store')
        atomic_json(output / 'validation-report.json', validation)
        checksums = json.loads((base / 'checksums.json').read_text())
        checksums.update({f"shards/{r['shard']:06d}.bin": r['sha256'] for r in reports})
        atomic_json(output / 'checksums.json', checksums)
        atomic_json(output / 'manifest.json', plan | {'complete': True, 'shards': reports})
        del owners
        shutil.rmtree(output / 'scratch')
        status(output, 'complete', pages=plan['pages'], validation=validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Published sparse (V2) store')
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-public', required=True, help='Public image LMDB directory')
    parser.add_argument('--image-private', required=True, help='Private image LMDB directory')
    parser.add_argument('--validation-pages', type=int, default=2048)
    args = parser.parse_args()
    run(args.base, args.output, {'public': args.image_public, 'private': args.image_private},
        args.validation_pages)
