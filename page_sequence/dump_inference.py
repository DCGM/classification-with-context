"""Resumable, bounded full native-store inference with independent document context."""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import collate_windows, to_device
from .inference import InferenceConfig, StreamMerger, window_spans
from .loading import worker_init
from .manifests import file_sha256
from .precision import autocast
from .runner import evaluation_dataset, load_checkpoint
from .schema import PAGE_TYPES, PLACEMENTS, Window, IGNORE
from training_store.format import atomic_json
from training_store.reader import TrainingStore


RECORD = np.dtype([
    ('row', '<u8'), ('document', '<u4'), ('position', '<u4'), ('document_length', '<u4'),
    ('order', '<i4'), ('weak', 'i1'), ('gold', 'i1'), ('db_side', 'i1'),
    ('availability', 'u1'), ('pagination_value', '<u4'), ('pagination_system', 'u1'),
    ('type_probs', '<f4', (len(PAGE_TYPES),)), ('placement_probs', '<f4', (3,)),
    ('type', 'u1'), ('placement', 'u1'), ('top2', 'u1'), ('confidence', '<f4'),
    ('margin', '<f4'), ('entropy', '<f4'), ('weak_probability', '<f4'),
    ('gold_probability', '<f4'), ('weak_disagrees', '?'), ('gold_disagrees', '?'),
    ('starts_document_probs', '<f4'), ('pair_mask', '?'),
    ('page_start_probs', '<f4'), ('page_end_probs', '<f4'),
    ('window_count', '<u4'), ('type_window_js', '<f4'),
])


@dataclass(frozen=True)
class Tile:
    document: int
    first: int
    length: int
    shard: int
    start: int
    end: int


class ExportDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, tile):
        window = Window(tuple(range(tile.first + tile.start, tile.first + tile.end)),
                        scan_start=tile.start == 0, scan_end=tile.end == tile.length,
                        document_metadata=True)
        example = self.dataset[window]
        example['tile'] = tile
        return example


def collate(examples):
    result = collate_windows(examples)
    result['tiles'] = [e['tile'] for e in examples]
    return result


class TileBatches:
    def __init__(self, reader, settings, completed, max_shards=None):
        self.reader, self.settings, self.completed = reader, settings, completed
        self.max_shards = max_shards

    def __iter__(self):
        pending = []
        remaining = sorted(set(map(int, self.reader.documents['shard'][self.reader.ids])) - self.completed)
        if self.max_shards is not None:
            remaining = remaining[:self.max_shards]
        for shard in remaining:
            ids = self.reader.ids[self.reader.documents['shard'][self.reader.ids] == shard]
            ids = ids[np.argsort(self.reader.documents['offset'][ids])]
            for docid in ids:
                d = self.reader.documents[docid]
                for lo, hi in window_spans(int(d['length']), self.settings.window_size,
                                           self.settings.stride, self.settings.tail):
                    pending.append(Tile(int(docid), int(d['first']), int(d['length']), shard, lo, hi))
                    if len(pending) == self.settings.batch_size:
                        yield pending
                        pending = []
            # Do not prefetch a second output shard into the same GPU batch.
            if pending:
                yield pending
                pending = []


def pack_result(source, pagination, result):
    n = len(source)
    out = np.zeros(n, RECORD)
    for key in ('row', 'document', 'order', 'weak', 'gold', 'availability'):
        out[key] = source[key]
    out['db_side'] = source['placement']
    out['position'], out['document_length'] = np.arange(n), n
    out['pagination_value'], out['pagination_system'] = pagination['value'], pagination['system']
    for key in ('type_probs', 'placement_probs', 'type', 'placement', 'starts_document_probs',
                'pair_mask', 'page_start_probs', 'page_end_probs', 'window_count', 'type_window_js'):
        out[key] = result[key]
    probs = result['type_probs']
    rank = np.argsort(probs, axis=-1)
    out['top2'] = rank[:, -2]
    out['confidence'] = probs.max(-1)
    out['margin'] = out['confidence'] - np.take_along_axis(probs, rank[:, -2:-1], axis=-1)[:, 0]
    out['entropy'] = -(probs * np.log(np.maximum(probs, 1e-30))).sum(-1)
    for name in ('weak', 'gold'):
        labels = source[name].astype(np.int64)
        valid = labels != IGNORE
        out[name + '_probability'] = np.nan
        out[name + '_probability'][valid] = probs[np.flatnonzero(valid), labels[valid]]
        out[name + '_disagrees'] = valid & (labels != result['type'])
    if (not np.isfinite(result['embeddings']).all()
            or not np.isfinite(result['embeddings'].astype(np.float16)).all()
            or not np.allclose(probs.sum(-1), 1, atol=1e-5)):
        raise ValueError('invalid export probabilities or embeddings')
    return out


def sync_file(path):
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())


def snapshot(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if expected is not None and file_sha256(destination) != expected:
            raise ValueError(f'snapshot mismatch: {destination.name}')
        return
    temporary = destination.with_suffix(destination.suffix + '.copying')
    shutil.copyfile(source, temporary)
    if expected is not None and file_sha256(temporary) != expected:
        raise ValueError(f'source mismatch: {source.name}')
    sync_file(temporary)
    temporary.replace(destination)


@torch.inference_mode()
def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    settings = InferenceConfig(window_size=512, stride=448, batch_size=args.batch_size,
                               tail='regular', merge='center_logits', edge_ramp=32, edge_ignore=0)
    atomic_json(output / 'status.json', dict(status='initializing', pid=os.getpid()))
    model, state = load_checkpoint(args.checkpoint, args.device)
    config, dataset = evaluation_dataset(state, args.training_store)
    if not model.boundary_features or model.scan_borders or not model.auxiliary_edges:
        raise ValueError('five-bit boundary model with auxiliary heads required')
    native = TrainingStore(args.training_store)
    pagination_path = Path(config['pagination']['path'])
    pagination = np.load(pagination_path, mmap_mode='r')
    report = json.loads(pagination_path.with_suffix('.json').read_text())
    contract = dict(format='page_context_dump_v1', checkpoint_sha256=file_sha256(args.checkpoint),
                    checkpoint_step=state['step'], specification=state['specification'],
                    index_sha256=state['provenance']['index_sha256'],
                    store=dataset.source_provenance(), pagination=dataset.pagination_provenance(),
                    inference=asdict(settings), precision=model.compute_precision,
                    processing='complete_bottom_level_documents', document_metadata='provided',
                    pages=native.manifest['pages'], documents=native.manifest['documents'],
                    page_types=PAGE_TYPES, placements=PLACEMENTS, record_dtype=RECORD.descr,
                    embedding_dtype='float16', embedding_size=model.backbone.hidden_size,
                    embedding_normalization='none', position_base=0,
                    code_sha256=file_sha256(__file__),
                    source_sha256={str(p.relative_to(Path(__file__).parent.parent)): file_sha256(p)
                        for package in ('page_sequence', 'training_store', 'embeddings', 'embedding_probe')
                        for p in sorted((Path(__file__).parent.parent / package).glob('*.py'))},
                    metadata='metadata/catalogue.sqlite joins records.row to pages.id; '
                             'documents by id; metadata/raw-pagination.csv preserves original CSV '
                             'pagination (library,doc,page columns 0,3,4; raw number column 9). '
                             'Test rows have unavailable pagination, matching the existing sidecar. '
                             'Missing labels=-100; probabilities for absent labels and first pair=NaN. '
                             'Availability bits image=1,text=2,stats=4; zero=all missing.')
    contract = json.loads(json.dumps(contract))
    plan = output / 'manifest.json'
    if plan.exists():
        if json.loads(plan.read_text()) != contract:
            raise ValueError('resume contract differs: use original code, settings and sources')
    else:
        if shutil.disk_usage(output).free < 60 * (1 << 30):
            raise ValueError('reserve at least 60 GiB for predictions, metadata and embeddings')
        atomic_json(plan, contract)
    del state
    metadata = output / 'metadata'
    metadata.mkdir(exist_ok=True)
    for source, name, checksum in (
            (Path(args.training_store) / 'catalogue.sqlite', 'catalogue.sqlite', contract['index_sha256']),
            (Path(args.training_store) / 'documents.npy', 'documents.npy', native.manifest['documents_sha256']),
            (pagination_path, 'pagination.npy', report['sha256']),
            (report['source'], 'raw-pagination.csv', report['source_sha256'])):
        print(json.dumps(dict(snapshot=name)), flush=True)
        snapshot(source, metadata / name, checksum)
    # Keep the selected weights, input contract and original number filtering policy together.
    snapshot(args.checkpoint, output / 'checkpoint.pt', contract['checkpoint_sha256'])
    atomic_json(metadata / 'pagination.json', report)
    shards = output / 'shards'
    shards.mkdir(exist_ok=True)
    completed = set()
    for marker in sorted(shards.glob('*.done.json')):
        info = json.loads(marker.read_text())
        for name, checksum in info['sha256'].items():
            if file_sha256(shards / name) != checksum:
                raise ValueError(f'completed shard checksum mismatch: {name}')
        completed.add(info['shard'])
    batches = TileBatches(native, settings, completed, args.max_shards)
    options = dict(num_workers=args.workers, collate_fn=collate, pin_memory=False)
    if args.workers:
        options.update(multiprocessing_context='spawn', prefetch_factor=1, worker_init_fn=worker_init)
    loader = DataLoader(ExportDataset(dataset), batch_sampler=batches, **options)
    started = time.monotonic()
    processed = documents = 0
    current_shard = current_doc = None
    merger = records = embeddings = None
    shard_docs = shard_pages = 0

    def publish():
        nonlocal records, embeddings
        if shard_pages != native.manifest['shard_pages'][current_shard]:
            raise ValueError('incomplete output shard')
        names = [f'{current_shard:06d}.records.npy', f'{current_shard:06d}.embeddings.npy']
        records.flush(); embeddings.flush()
        records = embeddings = None
        checksums = {}
        for name in names:
            temporary = shards / (name + '.writing')
            sync_file(temporary)
            checksums[name] = file_sha256(temporary)
            temporary.replace(shards / name)
        atomic_json(shards / f'{current_shard:06d}.done.json',
                    dict(shard=current_shard, pages=shard_pages, documents=shard_docs, sha256=checksums))
        completed.add(current_shard)

    model.eval()
    for batch in loader:
        tiles = batch['tiles']
        gpu = to_device(batch, args.device)
        with autocast(model, args.device):
            prediction = model(gpu['features'], gpu['attention_mask'], return_embeddings=True)
        host = {k: v.detach().float().cpu().numpy() for k, v in prediction.items()}
        del gpu, prediction
        for i, tile in enumerate(tiles):
            if tile.shard != current_shard:
                if current_shard is not None:
                    publish()
                current_shard, shard_docs, shard_pages = tile.shard, 0, 0
                n = native.manifest['shard_pages'][current_shard]
                records = np.lib.format.open_memmap(shards / f'{current_shard:06d}.records.npy.writing',
                                                    mode='w+', dtype=RECORD, shape=(n,))
                embeddings = np.lib.format.open_memmap(shards / f'{current_shard:06d}.embeddings.npy.writing',
                    mode='w+', dtype=np.float16, shape=(n, model.backbone.hidden_size))
            if tile.document != current_doc:
                if merger is not None:
                    raise ValueError('previous document unfinished')
                current_doc = tile.document
                merger = StreamMerger(tile.length, model.num_types, settings, True, True)
            merger.add(tile.start, tile.end, {k: v[i, :tile.end-tile.start] for k, v in host.items()})
            if tile.end == tile.length:
                result = merger.finalize()
                source = native.read_span(tile.document, 0, tile.length)
                offset = int(native.documents[tile.document]['offset'])
                records[offset:offset + tile.length] = pack_result(source, pagination[source['row']], result)
                embeddings[offset:offset + tile.length] = result['embeddings'].astype(np.float16)
                processed += tile.length
                documents += 1
                shard_pages += tile.length
                shard_docs += 1
                merger = None
        elapsed = time.monotonic() - started
        progress = dict(status='running', pid=os.getpid(), completed_shards=sorted(completed),
                        current_shard=current_shard, current_shard_pages=shard_pages,
                        session_pages=processed, session_documents=documents, seconds=elapsed,
                        pages_per_second=processed / max(elapsed, 1e-6), total_pages=contract['pages'])
        atomic_json(output / 'status.json', progress)
        if documents and documents % 1000 < len(tiles):
            print(json.dumps(progress), flush=True)
    if current_shard is not None:
        publish()
    complete = len(completed) == len(native.manifest['shard_pages'])
    atomic_json(output / 'status.json', dict(status='complete' if complete else 'partial',
        completed_shards=sorted(completed), session_pages=processed, session_documents=documents,
        seconds=time.monotonic() - started, total_pages=contract['pages']))
    print(json.dumps(dict(status='complete' if complete else 'partial', pages=processed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-store', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-shards', type=int, help='Process at most this many pending shards for validation')
    args = parser.parse_args()
    if args.workers < 0 or (args.max_shards is not None and args.max_shards < 1):
        parser.error('nonnegative workers and positive max-shards required')
    try:
        run(args)
    except BaseException:
        if Path(args.output).is_dir():
            atomic_json(Path(args.output) / 'failure.json',
                        dict(time=time.time(), pid=os.getpid(), traceback=traceback.format_exc()))
            status_path = Path(args.output) / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status='failed', failure='failure.json')
            atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
