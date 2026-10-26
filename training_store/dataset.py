"""Document-store windows, with explicit, numerically equivalent build-time fallback.

Only atomically completed shards are consumed. Pending shards use the original
vector pipeline rounded to FP16, preserving sampling over the entire catalogue.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from page_sequence.data import make_example, page_edge_targets
from page_sequence.schema import IGNORE, PAGE_TYPES, PLACEMENTS
from .format import DOCUMENT, RECORD, VERSION, digest
from .reader import TrainingStore


def canonical(value):
    return json.loads(json.dumps(value))


class StoreDataset(Dataset):
    def __init__(self, index, features, path, allow_building=False):
        self.index, self.features = index, features
        self.path = Path(path)
        self.allow_building = allow_building
        plan_path = self.path / ('plan.json' if allow_building else 'manifest.json')
        self.plan = json.loads(plan_path.read_text())
        p = self.plan
        if (p['version'] != VERSION or p['record_bytes'] != RECORD.itemsize or p['pilot']
                or tuple(p['page_types']) != PAGE_TYPES or tuple(p['placements']) != PLACEMENTS
                or canonical(p['feature_state']) != canonical(features.state_dict())):
            raise ValueError('store layout, labels or preprocessing do not match training')
        if not allow_building and not p.get('complete'):
            raise ValueError('incomplete training store')
        if Path(index.path).resolve() != (self.path / 'catalogue.sqlite').resolve():
            raise ValueError('store training must use its own catalogue')
        if digest(self.path / 'documents.npy') != p['documents_sha256']:
            raise ValueError('document mapping checksum mismatch')
        if allow_building:
            if canonical(features.source_provenance()) != p['reference_sources']:
                raise ValueError('fallback vector stores differ from audited store sources')
        self.reader = TrainingStore.__new__(TrainingStore)
        self.reader.path, self.reader.manifest = self.path, p
        self.reader.documents = np.load(self.path / 'documents.npy', mmap_mode='r', allow_pickle=False)
        if self.reader.documents.dtype != DOCUMENT:
            raise ValueError('incompatible document table')
        self.reader.ids = np.flatnonzero(self.reader.documents['length'])
        self.reader.first = self.reader.documents['first'][self.reader.ids]
        if np.any(self.reader.first[1:] <= self.reader.first[:-1]):
            raise ValueError('invalid logical document ranges')
        self.reader._pid, self.reader._fds = None, {}
        self._ready = set()

    def __getstate__(self):
        return dict(index=self.index, features=self.features, path=self.path,
                    allow_building=self.allow_building)

    def __setstate__(self, state):
        self.__init__(**state)

    def source_provenance(self):
        return {'kind': 'document_store_fp16', 'path': str(self.path.resolve()),
                'plan_sha256': hashlib.sha256((self.path / 'plan.json').read_bytes()).hexdigest(),
                'build_id': self.plan['build_id'], 'allow_building': self.allow_building,
                'fallback': self.plan['reference_sources'] if self.allow_building else None}

    def _shard_ready(self, shard):
        shard = int(shard)
        if shard in self._ready:
            return True
        marker = self.path / 'shards' / f'{shard:06d}.done.json'
        if not marker.exists():
            if not self.allow_building:
                raise ValueError('published store is missing a completed shard')
            return False
        report = json.loads(marker.read_text())
        final = marker.with_suffix('').with_suffix('.bin')
        expected = self.plan['shard_pages'][shard]
        if (report['build_id'] != self.plan['build_id'] or report['pages'] != expected
                or final.stat().st_size != expected * RECORD.itemsize):
            raise ValueError('completed shard does not match store plan')
        self._ready.add(shard)
        return True

    def __getitem__(self, window):
        if self.allow_building and (self.path / 'failure.json').exists():
            raise RuntimeError('store build failed; inspect failure.json before continuing training')
        rows = np.asarray(window.rows, np.int64)
        pos = np.searchsorted(self.reader.first, rows, side='right') - 1
        if np.any(pos < 0):
            raise KeyError('row outside store')
        docs = self.reader.documents[self.reader.ids[pos]]
        if np.any(rows - docs['first'].astype(np.int64) >= docs['length']):
            raise KeyError('row outside store')
        ready = np.array([self._shard_ready(s) for s in np.unique(docs['shard'])])
        # Whole-window fallback keeps missing modalities and repeated rows identical.
        if not ready.all():
            pages = self.index.pages(window.rows)
            example = make_example(pages, self.features, window)
            example.update(page_edge_targets(self.index, window.rows))
            example['features'] = {k: v.astype(np.float16) for k, v in example['features'].items()}
            example['storage_pages'] = {'nvme': 0, 'fallback': len(rows)}
            return example
        records = self.reader.read_rows(rows)
        gold = records['gold'] != IGNORE
        pair_mask = np.ones(len(rows), bool)
        pair_mask[0] = False
        starts = np.r_[0, records['document'][1:] != records['document'][:-1]].astype(np.float32)
        return {'features': self.reader.features(records),
                'type': np.where(gold, records['gold'], records['weak']).astype(np.int64),
                'gold': gold, 'placement': records['placement'].astype(np.int64),
                'starts': starts, 'pair_mask': pair_mask,
                'page_start': (rows == docs['first']).astype(np.float32),
                'page_end': (rows == docs['first'] + docs['length'] - 1).astype(np.float32),
                'rows': window.rows, 'source': window.source,
                'storage_pages': {'nvme': len(rows), 'fallback': 0}}
