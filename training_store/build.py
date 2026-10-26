"""Resumable bounded-memory conversion. Publishes manifest only after validation.

python -m training_store.build --config EXPERIMENT --sources SOURCES --output DIRECTORY
Sources is an optional replacement FeaturePipeline configuration (vectors only).
Completed shards are immutable. An interrupted partial shard is rebuilt on resume.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from page_sequence.features import FeaturePipeline
from page_sequence.schema import IGNORE, LABEL2ID, PAGE_TYPES, PLACEMENTS, Page
from .format import DOCUMENT, RECORD, VERSION, atomic_json, digest
from .reader import TrainingStore


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    return db


def signatures(config):
    return FeaturePipeline.from_config(config).source_provenance()


def code_signature():
    paths = list(Path(__file__).parent.glob('*.py')) + [
        Path(__file__).parents[1] / 'page_sequence/features.py',
        Path(__file__).parents[1] / 'embeddings/text/page_stats/page_stats.py']
    return {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in sorted(paths)}


def plan_store(output, config, source_config, shard_bytes, pilot_documents):
    output = Path(output)
    db = readonly(config['index'])
    maxid = db.execute('SELECT MAX(id) FROM documents').fetchone()[0]
    if maxid >= 2**32:
        raise ValueError('document ID outside uint32')
    docs = np.zeros(maxid + 1, DOCUMENT)
    # Only document IDs are collected; page metadata remains disk-backed.
    selected = None
    if pilot_documents:
        rng = np.random.default_rng(17)
        selected = set()
        strata = db.execute('SELECT DISTINCT library,split FROM documents ORDER BY library,split').fetchall()
        for library, split in strata:
            ids = np.fromiter((r[0] for r in db.execute(
                'SELECT id FROM documents WHERE library=? AND split=? ORDER BY id', (library, split))), np.int64)
            n = min(len(ids), max(1, pilot_documents // len(strata)))
            selected.update(int(x) for x in rng.choice(ids, n, replace=False))
        # Explicitly include long documents and curated pages for correctness coverage.
        for row in db.execute('SELECT id FROM documents ORDER BY length DESC LIMIT 8'):
            selected.add(row[0])
        for label in PAGE_TYPES:
            row = db.execute("SELECT row_id FROM anchors WHERE source='annotated' AND label=? LIMIT 1",
                             (LABEL2ID[label],)).fetchone()
            if row:
                selected.add(db.execute('SELECT document FROM pages WHERE id=?', row).fetchone()[0])
    shard, offset, total = 0, 0, 0
    counts = []
    for did, first, length in db.execute(
            'SELECT id,first,length FROM documents ORDER BY library,periodical,volume,doc,id'):
        if selected is not None and did not in selected:
            continue
        if not 0 < length < 2**32:
            raise ValueError('invalid document length')
        if offset and (offset + length) * RECORD.itemsize > shard_bytes:
            counts.append(offset)
            shard += 1
            offset = 0
        docs[did] = first, length, shard, offset
        offset += length
        total += length
    counts.append(offset)
    db.close()
    required = total * RECORD.itemsize + Path(config['index']).stat().st_size + (10 << 30)
    if shutil.disk_usage(output).free < required:
        raise ValueError(f'insufficient free space: require {required} bytes including catalogue/headroom')
    index_sha = digest(config['index'])
    # A full independent catalogue also preserves sampling anchors and all exclusion audits.
    tmp = output / 'catalogue.sqlite.partial'
    shutil.copyfile(config['index'], tmp)
    if digest(tmp) != index_sha:
        raise ValueError('catalogue copy mismatch')
    os.replace(tmp, output / 'catalogue.sqlite')
    np.save(output / 'documents.npy', docs, allow_pickle=False)
    fp = FeaturePipeline.from_config(source_config)
    if fp.dims != {'image': 768, 'text': 768, 'stats': 147}:
        raise ValueError('unsupported feature dimensions')
    plan = {'version': VERSION, 'record_bytes': RECORD.itemsize, 'record_dtype': RECORD.descr,
            'pages': total, 'documents': int(np.count_nonzero(docs['length'])),
            'shard_pages': counts, 'source_config': source_config,
            'reference_config': config['features'], 'sources': signatures(source_config),
            'reference_sources': signatures(config['features']), 'feature_state': fp.state_dict(),
            'index_sha256': index_sha, 'documents_sha256': digest(output / 'documents.npy'),
            'code_sha256': code_signature(), 'pilot': bool(pilot_documents),
            'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
            'missing_label': IGNORE, 'availability_bits': {'image': 1, 'text': 2, 'stats': 4},
            'dense_storage': 'original_fp16',
            'stats_storage': 'grouped_normalized_block_scaled_fp16',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    plan['build_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic_json(output / 'plan.json', plan)
    return plan


def page_blocks(db, docs, shard, block_pages):
    ids = np.flatnonzero((docs['length'] > 0) & (docs['shard'] == shard))
    ids = ids[np.argsort(docs['offset'][ids])]
    pending = []
    for did in ids:
        library, doc = db.execute('SELECT library,doc FROM documents WHERE id=?', (int(did),)).fetchone()
        count = 0
        for row, page, order, path, weak, gold, side in db.execute(
                'SELECT id,page,ord,path,db_type,annotated_type,placement FROM pages '
                'WHERE document=? ORDER BY id', (int(did),)):
            if row != int(docs[did]['first']) + count or not -(2**31) <= order < 2**31:
                raise ValueError('invalid page order or noncontiguous logical rows')
            pending.append((row, int(did), Page(library, doc, page, order, path, weak, gold, side)))
            count += 1
            if len(pending) == block_pages:
                yield pending
                pending = []
        if count != docs[did]['length']:
            raise ValueError('document length mismatch')
    if pending:
        yield pending


def convert_block(block, pipeline):
    pages = [x[2] for x in block]
    records = np.zeros(len(block), RECORD)
    for i, (row, did, p) in enumerate(block):
        records['row'][i], records['document'][i], records['order'][i] = row, did, p.order
        records['weak'][i] = LABEL2ID.get(p.db_type, IGNORE)
        records['gold'][i] = LABEL2ID.get(p.annotated_type, IGNORE)
        records['placement'][i] = PLACEMENTS.index(p.placement) if p.placement in PLACEMENTS else IGNORE
    for bit, name in enumerate(('image', 'text', 'stats')):
        values, present = pipeline.sources[name].read(pages)
        if name == 'stats':
            values, names = pipeline.stats.view(values, grouped=True, normalize=pipeline.normalization or True)
            if tuple(names) != pipeline.stats_names:
                raise ValueError('statistics order changed')
            values /= np.sqrt(len(names))
        values[~present] = 0
        values = values.astype('<f2')
        if not np.isfinite(values).all():
            raise ValueError('FP16 conversion produced nonfinite values')
        records[name] = values
        records['availability'] |= present.astype(np.uint8) << bit
    return records


def build_shard(output, shard, block_pages):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    docs = np.load(output / 'documents.npy', mmap_mode='r', allow_pickle=False)
    final = output / 'shards' / f'{shard:06d}.bin'
    done = final.with_suffix('.done.json')
    if done.exists():
        report = json.loads(done.read_text())
        if (report['build_id'] != plan['build_id'] or not final.exists()
                or final.stat().st_size != plan['shard_pages'][shard] * RECORD.itemsize
                or digest(final) != report['sha256']):
            raise ValueError('completed shard changed; refusing resume')
        return report
    pipeline = FeaturePipeline.from_config(plan['source_config'])
    db = readonly(output / 'catalogue.sqlite')
    partial = final.with_suffix('.partial')
    h = hashlib.sha256()
    counts = np.zeros(3, np.int64)
    n = 0
    started = time.monotonic()
    last_progress = started
    with open(partial, 'wb') as f:
        for block in page_blocks(db, docs, shard, block_pages):
            records = convert_block(block, pipeline)
            raw = records.tobytes()
            f.write(raw)
            h.update(raw)
            for bit in range(3):
                counts[bit] += np.count_nonzero(records['availability'] & (1 << bit))
            n += len(records)
            now = time.monotonic()
            if now - last_progress >= 15:
                atomic_json(final.with_suffix('.progress.json'), {
                    'shard': shard, 'pages_written': n, 'pages': plan['shard_pages'][shard],
                    'seconds': now - started, 'pages_per_second': n / (now - started)})
                last_progress = now
        f.flush()
        os.fsync(f.fileno())
    db.close()
    if n != plan['shard_pages'][shard]:
        raise ValueError('shard length mismatch')
    os.replace(partial, final)
    report = {'shard': shard, 'pages': n, 'sha256': h.hexdigest(), 'build_id': plan['build_id'],
              'present': dict(zip(('image', 'text', 'stats'), map(int, counts))),
              'seconds': time.monotonic() - started}
    atomic_json(done, report)
    return report


def validate(output, plan, sample_pages=4096):
    # Reader requires a published manifest. Validate a temporary view through the same
    # implementation without publishing a manifest that other processes could consume.
    reader = TrainingStore.__new__(TrainingStore)
    reader.path, reader.manifest = Path(output), plan
    reader.documents = np.load(reader.path / 'documents.npy', mmap_mode='r', allow_pickle=False)
    reader.ids = np.flatnonzero(reader.documents['length'])
    reader.first = reader.documents['first'][reader.ids]
    reader._pid, reader._fds = None, {}
    rng = np.random.default_rng(23)
    # Uniform page sampling without creating a full list of pages.
    lengths = reader.documents['length'][reader.ids].astype(np.int64)
    cumulative = np.cumsum(lengths)
    draws = rng.integers(0, int(cumulative[-1]), size=sample_pages)
    pos = np.searchsorted(cumulative, draws, side='right')
    before = np.concatenate(([0], cumulative[:-1]))
    rows = (reader.first[pos] + draws - before[pos]).astype(np.int64)
    # Add both ends of long spans and all represented library/split groups via documents.
    db = readonly(reader.path / 'catalogue.sqlite')
    reference = FeaturePipeline.from_config(plan['reference_config'])
    max_error = {name: 0.0 for name in ('image', 'text', 'stats')}
    counts = {name: 0 for name in max_error}
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        records = reader.read_rows(chunk)
        pages = []
        for row in chunk:
            r = db.execute('SELECT d.library,d.doc,p.page,p.ord,p.path,p.db_type,p.annotated_type,p.placement,'
                           'p.document FROM pages p JOIN documents d ON d.id=p.document WHERE p.id=?',
                           (int(row),)).fetchone()
            p = Page(*r[:8])
            pages.append(p)
            rec = records[len(pages) - 1]
            if (rec['document'] != r[8] or rec['weak'] != LABEL2ID.get(p.db_type, IGNORE)
                    or rec['gold'] != LABEL2ID.get(p.annotated_type, IGNORE)):
                raise ValueError('document or individual label mismatch')
        expected = reference(pages)
        actual = reader.features(records)
        if not np.array_equal(actual['presence'], expected['presence'].astype(np.float16)):
            raise ValueError('modality presence differs from existing reader')
        for bit, name in enumerate(max_error):
            if not np.array_equal(actual[name], expected[name].astype(np.float16)):
                raise ValueError(f'{name} features differ from FP16-rounded current reader')
            counts[name] += int(expected['presence'][:, bit].sum())
            max_error[name] = max(max_error[name], float(np.max(np.abs(
                actual[name].astype(np.float32) - expected[name]))))
        for rec, p in zip(records, pages):
            label, gold, side = p.target()
            got = int(rec['gold'] if gold else rec['weak'])
            if got != label or rec['placement'] != side or rec['order'] != p.order:
                raise ValueError('label/order mismatch')
        # Dense source bytes must be preserved, not a normalized/requantized substitute.
        for name in ('image', 'text'):
            raw, _ = reference.sources[name].read(pages)
            if not np.array_equal(records[name], raw.astype(np.float16)):
                raise ValueError('original dense embedding bytes changed')
    reader.close()
    db.close()
    return {'sample_pages': sample_pages, 'seed': 23, 'present': counts,
            'fp16_rounded_reference_exact': True, 'max_abs_rounding_error': max_error}


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'shards').mkdir(exist_ok=True)
    lock = (output / 'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(Path(args.config).read_text())
    sources = json.loads(Path(args.sources).read_text()) if args.sources else config['features']
    if (output / 'manifest.json').exists():
        print('Store is already published; refusing to overwrite.', flush=True)
        return
    if (output / 'plan.json').exists():
        plan = json.loads((output / 'plan.json').read_text())
        if (plan['source_config'] != sources or plan['code_sha256'] != code_signature()
                or plan['sources'] != signatures(sources)
                or plan['reference_sources'] != signatures(config['features'])
                or plan['index_sha256'] != digest(config['index'])
                or plan['index_sha256'] != digest(output / 'catalogue.sqlite')
                or plan['documents_sha256'] != digest(output / 'documents.npy')):
            raise ValueError('resume inputs, code or catalogue changed')
    else:
        print('Preparing catalogue and document layout', flush=True)
        atomic_json(output / 'status.json', {'phase': 'preparing_catalogue'})
        plan = plan_store(output, config, sources, args.shard_mib << 20, args.pilot_documents)
    print(json.dumps({'phase': 'convert', 'pages': plan['pages'], 'shards': len(plan['shard_pages']),
                      'workers': args.workers}), flush=True)
    atomic_json(output / 'status.json', {'phase': 'converting', 'pages': plan['pages'], 'completed_pages': 0})
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(build_shard, output, i, args.block_pages) for i in range(len(plan['shard_pages']))]
        try:
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(json.dumps(report), flush=True)
                atomic_json(output / 'status.json', {'phase': 'converting', 'pages': plan['pages'],
                            'completed_pages': sum(r['pages'] for r in reports), 'completed_shards': len(reports)})
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if signatures(sources) != plan['sources'] or signatures(config['features']) != plan['reference_sources']:
        raise ValueError('source files changed during conversion')
    atomic_json(output / 'status.json', {'phase': 'validating', 'completed_pages': plan['pages']})
    validation = validate(output, plan, args.validation_pages)
    atomic_json(output / 'validation-report.json', validation)
    atomic_json(output / 'checksums.json', {f"shards/{r['shard']:06d}.bin": r['sha256'] for r in reports} |
                {'catalogue.sqlite': plan['index_sha256'], 'documents.npy': plan['documents_sha256']})
    atomic_json(output / 'manifest.json', plan | {'complete': True, 'shards': sorted(reports, key=lambda r: r['shard'])})
    atomic_json(output / 'status.json', {'phase': 'complete', 'completed_pages': plan['pages'], 'validation': validation})
    print(json.dumps({'phase': 'complete', 'validation': validation}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--sources')
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--block-pages', type=int, default=2048)
    p.add_argument('--shard-mib', type=int, default=2048)
    p.add_argument('--pilot-documents', type=int, default=0)
    p.add_argument('--validation-pages', type=int, default=4096)
    args = p.parse_args()
    if min(args.workers, args.block_pages, args.shard_mib, args.validation_pages) < 1 or args.pilot_documents < 0:
        p.error('positive sizes/workers required')
    try:
        run(args)
    except BaseException as exc:
        path = Path(args.output)
        if path.exists() and not (path / 'manifest.json').exists():
            atomic_json(path / 'failure.json', {'error': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
