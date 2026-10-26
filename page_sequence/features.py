"""Lazy worker-local vector I/O and fixed per-page preprocessing. Never opens text/OCR stores."""
import json
import os
from pathlib import Path

import lmdb
import numpy as np

from embeddings.text.page_stats import PageStats, PageStatsNormalization


class LmdbVectors:
    """One dense modality or sparse page-statistics modality.

    stores maps public/private to vector LMDB directories. Text/stats try both stores;
    image paths select their store. Every configured store must exist; a missing key is allowed.
    """
    def __init__(self, stores, kind, dim, dtype='<f2'):
        if kind not in ('text', 'image', 'stats'):
            raise ValueError(kind)
        if kind == 'stats' and dim != len(PageStats.features):
            raise ValueError('stats dimension differs from the raw extraction schema')
        if dim < 1 or not stores or set(stores) - {'public', 'private'}:
            raise ValueError('positive dimension and public/private vector stores required')
        self.stores, self.kind, self.dim, self.dtype = dict(stores), kind, dim, dtype
        self._stats = PageStats() if kind == 'stats' else None
        self._pid, self._envs = None, {}

    def __getstate__(self):
        return self.__dict__ | {'_pid': None, '_envs': {}}

    def _open(self):
        if self._pid != os.getpid():
            self._envs = {}
            self._pid = os.getpid()
        if not self._envs:
            for source, path in self.stores.items():
                self._envs[source] = lmdb.open(str(path), readonly=True, lock=False,
                                              readahead=False, create=False, max_readers=256)
                if self.kind == 'stats':
                    with self._envs[source].begin() as txn:
                        raw = txn.get(b'__metadata__')
                    if raw is None:
                        raise ValueError('stats store lacks schema metadata')
                    meta = json.loads(raw)
                    expected = self._stats.metadata()
                    if any(meta.get(k) != expected[k] for k in ('model', 'version', 'features', 'storage')):
                        raise ValueError('incompatible stats LMDB schema')

    def read(self, pages):
        self._open()
        values = np.zeros((len(pages), self.dim), np.float32)
        present = np.zeros(len(pages), bool)
        for source, env in self._envs.items():
            with env.begin() as txn:
                for i, page in enumerate(pages):
                    if present[i]:
                        continue
                    key = page.key
                    if self.kind == 'image':
                        if not page.path:
                            continue
                        root, key = page.path.split('/', 1)
                        root = {'public.256': 'public', 'private.256': 'private'}.get(root, root)
                        if root not in ('public', 'private'):
                            raise ValueError('unsupported image path root')
                        if root != source:
                            continue
                    raw = txn.get(key.encode('utf-8'))
                    if raw is None:
                        continue
                    if self.kind == 'stats':
                        if len(raw) % 4:
                            raise ValueError('malformed sparse statistics vector')
                        n = len(raw) // 4
                        indices = np.frombuffer(raw, '<u2', n)
                        if np.any(indices >= self.dim) or len(np.unique(indices)) != n:
                            raise ValueError('malformed sparse statistics indices')
                        vector = self._stats.decode([raw])[0]
                    else:
                        vector = np.frombuffer(raw, self.dtype)
                    if vector.shape != (self.dim,) or not np.isfinite(vector).all():
                        raise ValueError('invalid vector dimensions or nonfinite vector')
                    values[i], present[i] = vector, True
        return values, present


class FeaturePipeline:
    """Sources implement read(pages)->(raw float matrix, presence bool vector).

    Dense L2 normalization and all-grouped bounded stats are independent of window composition.
    Saved schema includes every normalization choice; model sees only these arrays/presence flags.
    """
    def __init__(self, sources, normalization=None):
        if set(sources) != {'image', 'text', 'stats'}:
            raise ValueError('initial experiment requires image, text, stats sources')
        self.sources = sources
        self.stats = PageStats()
        self.normalization = normalization
        self.stats_names = tuple(self.stats.view(np.zeros((0, len(self.stats.features)), np.float32),
                                                grouped=True, normalize=True)[1])

    @classmethod
    def from_config(cls, config):
        sources = {name: LmdbVectors(**spec) for name, spec in config['sources'].items()}
        normalization = config.get('normalization')
        if normalization is not None:
            normalization = PageStatsNormalization.from_dict(normalization)
        return cls(sources, normalization)

    @property
    def dims(self):
        return {'image': self.sources['image'].dim, 'text': self.sources['text'].dim,
                'stats': len(self.stats_names)}

    def state_dict(self):
        return {'stats_names': self.stats_names, 'stats_metadata': self.stats.metadata(),
                'normalization': self.normalization.to_dict() if self.normalization else 'fixed_v1',
                'dense_normalization': 'l2_float32_eps_1e-12', 'stats_block_scale': '1/sqrt(dim)',
                'coordinate_normalization': 'clip_page_fractions_0_1',
                'modality_order': ['image', 'text', 'stats'], 'dims': self.dims}

    def source_provenance(self):
        """Cheap immutable-run signature; no scan/hash of large vector LMDB contents."""
        import hashlib
        result = {}
        for name, source in self.sources.items():
            stores = {}
            for label, path in source.stores.items():
                path = Path(path)
                data = path / 'data.mdb' if path.is_dir() else path
                stat = data.stat()
                meta = path.parent / 'meta.json'
                stores[label] = {'path': str(path.resolve()), 'bytes': stat.st_size,
                                 'mtime_ns': stat.st_mtime_ns,
                                 'meta_sha256': hashlib.sha256(meta.read_bytes()).hexdigest() if meta.exists() else None}
            result[name] = {'stores': stores, 'kind': source.kind, 'dim': source.dim, 'dtype': source.dtype}
        return result

    def __call__(self, pages):
        result, flags = {}, []
        for name in ('image', 'text', 'stats'):
            x, present = self.sources[name].read(pages)
            x = np.asarray(x, np.float32)
            if x.shape != (len(pages), self.sources[name].dim) or not np.isfinite(x).all():
                raise ValueError('invalid feature array')
            if name == 'stats':
                x, names = self.stats.view(x, grouped=True, normalize=self.normalization or True)
                if tuple(names) != self.stats_names:
                    raise ValueError('stats feature order mismatch')
                x /= np.sqrt(len(names))
            else:
                # Compute norms in float64 to avoid overflow on finite extreme input values.
                norms = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
                x = (x / np.maximum(norms, 1e-12)).astype(np.float32)
            x[~present] = 0
            result[name] = x
            flags.append(np.asarray(present, bool))
        result['presence'] = np.stack(flags, axis=-1).astype(np.float32)
        return result
