"""Bounded pread reader: no corpus-sized feature array or private worker cache."""
import json
import os
from pathlib import Path

import numpy as np

from .format import DOCUMENT, RECORD, SPARSE_VERSION, SPARSE_READ_RECORD, record_dtype


class TrainingStore:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        record_dtype(self.manifest)
        if not self.manifest['complete']:
            raise ValueError('incomplete or incompatible training store')
        self.documents = np.load(self.path / 'documents.npy', mmap_mode='r', allow_pickle=False)
        if self.documents.dtype != DOCUMENT:
            raise ValueError('incompatible document table')
        self.ids = np.flatnonzero(self.documents['length'])
        self.first = self.documents['first'][self.ids]
        if np.any(self.first[1:] <= self.first[:-1]):
            raise ValueError('document logical ranges must increase with ID')
        self._pid, self._fds = None, {}

    def __getstate__(self):
        return {'path': str(self.path)}

    def __setstate__(self, state):
        self.__init__(state['path'])

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def _read(self, shard, offset, length):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard = int(shard)
        if shard not in self._fds:
            self._fds[shard] = os.open(self.path / 'shards' / f'{shard:06d}.bin', os.O_RDONLY)
        dtype = record_dtype(self.manifest)
        size = int(length) * dtype.itemsize
        position = int(offset) * dtype.itemsize
        raw = os.pread(self._fds[shard], size, position)
        if len(raw) != size:
            raise IOError('short training-store read')
        records = np.frombuffer(raw, dtype)
        if self.manifest['version'] != SPARSE_VERSION:
            return records
        out = np.empty(length, SPARSE_READ_RECORD)
        for name in RECORD.names:
            if name != 'stats':
                out[name] = records[name]
        if not length:
            return out
        key = ('stats', shard)
        if key not in self._fds:
            self._fds[key] = os.open(self.path / 'shards' / f'{shard:06d}.stats.bin', os.O_RDONLY)
        begin = int(records['stats_offset'][0])
        end = int(records['stats_offset'][-1]) + int(records['stats_bytes'][-1])
        if end < begin or np.any(records['stats_offset'][1:] !=
                                 records['stats_offset'][:-1] + records['stats_bytes'][:-1]):
            raise ValueError('noncontiguous sparse statistics')
        payload = os.pread(self._fds[key], end - begin, begin)
        if len(payload) != end - begin:
            raise IOError('short sparse statistics read')
        for i, row in enumerate(records):
            at, size = int(row['stats_offset']) - begin, int(row['stats_bytes'])
            out['stats'][i] = payload[at:at + size]
        return out

    def read_span(self, document, start, length):
        if document < 0 or document >= len(self.documents):
            raise KeyError(document)
        d = self.documents[document]
        if not d['length'] or start < 0 or length < 0 or start + length > int(d['length']):
            raise ValueError('span outside retained document')
        return self._read(d['shard'], int(d['offset']) + start, length)

    def read_rows(self, rows):
        """Coalesce consecutive requested rows; preserve order and repeated occurrences."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError('one-dimensional rows required')
        if not len(rows):
            return np.empty(0, self.read_dtype)
        unique, restore = np.unique(rows, return_inverse=True)
        positions = np.searchsorted(self.first, unique, side='right') - 1
        if np.any(positions < 0):
            raise KeyError('unknown page row')
        docids = self.ids[positions]
        docs = self.documents[docids]
        offsets = unique - docs['first'].astype(np.int64)
        if np.any(offsets >= docs['length']):
            raise KeyError('unknown page row')
        cuts = np.flatnonzero((docids[1:] != docids[:-1]) | (unique[1:] != unique[:-1] + 1)) + 1
        out = np.empty(len(unique), self.read_dtype)
        for part in np.split(np.arange(len(unique)), cuts):
            a, b = int(part[0]), int(part[-1]) + 1
            out[a:b] = self.read_span(int(docids[a]), int(offsets[a]), b - a)
        if not np.array_equal(out['row'], unique):
            raise ValueError('row mapping disagrees with shard')
        return out[restore]

    @property
    def read_dtype(self):
        return SPARSE_READ_RECORD if self.manifest['version'] == SPARSE_VERSION else RECORD

    def features(self, records):
        """All outputs FP16. Dense norms accumulate in FP64, matching the source pipeline."""
        result = {}
        for name in ('image', 'text'):
            x = records[name].astype(np.float32)
            norm = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
            result[name] = (x / np.maximum(norm, 1e-12)).astype(np.float32).astype(np.float16)
        if self.manifest['version'] == SPARSE_VERSION:
            from embeddings.text.page_stats import PageStats, PageStatsNormalization
            if not hasattr(self, '_stats_preprocessing'):
                normalization = self.manifest['feature_state']['normalization']
                normalization = (PageStatsNormalization.from_dict(normalization)
                                 if isinstance(normalization, dict) else True)
                self._stats_preprocessing = PageStats(), normalization
            stats, normalization = self._stats_preprocessing
            names = self.manifest['feature_state']['stats_names']
            values = np.empty((len(records), len(names)), np.float16)
            # Bound the temporary full 3,324-dimensional decoded array.
            for start in range(0, len(records), 512):
                raw = stats.decode(list(records['stats'][start:start + 512]))
                viewed, actual_names = stats.view(raw, grouped=True, normalize=normalization)
                if list(actual_names) != list(names):
                    raise ValueError('statistics view differs from model schema')
                viewed /= np.sqrt(len(names))
                values[start:start + len(raw)] = viewed.astype(np.float16)
            result['stats'] = values
        else:
            result['stats'] = records['stats'].copy()
        result['presence'] = np.stack([
            (records['availability'] & (1 << i)) != 0 for i in range(3)
        ], axis=-1).astype(np.float16)
        for i, name in enumerate(('image', 'text', 'stats')):
            result[name][result['presence'][:, i] == 0] = 0
        return result
