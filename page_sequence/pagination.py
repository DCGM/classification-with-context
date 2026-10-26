"""Optional noisy DB pagination candidates; never inferred bracket contents or document lengths."""
import csv
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
from torch.utils.data import Dataset

from .manifests import file_sha256

RECORD = np.dtype([('value', '<u4'), ('system', 'u1')])
POLICY = 'db_candidate_single_v1'
ROMAN = re.compile(r'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', re.I)


def candidate(text):
    """Single Arabic/Roman value, or explicitly printed part of `55 [58]`.

    Unbracketed DB values are noisy candidates, not proof of visible printing.
    Multiple numbers, ranges, suffixes and malformed brackets are unavailable.
    """
    text = text.strip()
    correction = re.fullmatch(r'([^\[\]]+)\s+\[[^\[\]]+\]', text)
    if correction:
        text = correction[1].strip()
    if re.fullmatch(r'[0-9]{1,9}', text):
        value = int(text)
        return (value, 1) if value > 0 else (0, 0)
    if text and ROMAN.fullmatch(text):
        digits = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        values = [digits[c] for c in text.upper()]
        return sum(-a if a < b else a for a, b in zip(values, values[1:] + [0])), 2
    return 0, 0


def encode(records, cap=10000):
    """[log magnitude, odd parity, Arabic, Roman, availability], all in [0,1]."""
    if cap <= 1:
        raise ValueError('pagination cap must exceed one')
    value, system = records['value'], records['system']
    valid = (value > 0) & np.isin(system, [1, 2])
    result = np.zeros((len(records), 5), np.float32)
    result[:, 0] = np.log1p(np.minimum(value, cap).astype(np.float64)) / math.log1p(cap)
    result[:, 1] = value % 2
    result[:, 2] = system == 1
    result[:, 3] = system == 2
    result[:, 4] = valid
    result[~valid] = 0
    return result.astype(np.float16)


class PaginationFeatures:
    """Inference on caller-provided pagination; missing candidates remain unavailable."""
    def __init__(self, features, cap=10000):
        self.features, self.cap = features, cap

    def __call__(self, pages):
        result = self.features(pages)
        records = np.array([candidate(p.pagenum) for p in pages], RECORD)
        result['pagination'] = encode(records, self.cap)
        return result


class PaginationDataset(Dataset):
    """Append row-aligned mmap features to any existing window dataset."""
    def __init__(self, dataset, path, cap=10000):
        self.dataset, self.path, self.cap = dataset, str(Path(path).resolve()), cap
        self.index, self.features = dataset.index, dataset.features
        self.model_dims = dict(self.features.dims, pagination=5)
        self.report = json.loads(Path(self.path).with_suffix('.json').read_text())
        if self.report['policy'] != POLICY or self.report['index_sha256'] != self.index.fingerprint():
            raise ValueError('pagination policy/catalogue mismatch')
        if file_sha256(self.path) != self.report['sha256']:
            raise ValueError('pagination sidecar checksum mismatch')
        self.records = np.load(self.path, mmap_mode='r', allow_pickle=False)
        if self.records.dtype != RECORD or len(self.records) != self.report['rows']:
            raise ValueError('invalid pagination sidecar layout')
        if hasattr(dataset, 'plan'):
            self.plan = dataset.plan

    def __getstate__(self):
        return self.dataset, self.path, self.cap

    def __setstate__(self, state):
        # Parent checked identity; workers reopen the immutable mmap without rehashing 10GB catalogue.
        self.dataset, self.path, self.cap = state
        self.index, self.features = self.dataset.index, self.dataset.features
        self.model_dims = dict(self.features.dims, pagination=5)
        self.records = np.load(self.path, mmap_mode='r', allow_pickle=False)

    def source_provenance(self):
        return self.dataset.source_provenance()

    def pagination_provenance(self):
        return dict(self.report, path=self.path, cap=self.cap)

    def __getitem__(self, window):
        result = self.dataset[window]
        rows = np.asarray(result['rows'], np.int64)
        if np.any(rows < 1) or np.any(rows >= len(self.records)):
            raise ValueError('pagination row outside catalogue')
        result['features']['pagination'] = encode(self.records[rows], self.cap)
        return result


def build(index, source, output):
    """Stream the training-context CSV with a bounded document mapping cache.

    Test pagination stays unused. Every retained fit/dev row must match exactly once.
    Output is published only after complete coverage and checksums succeed.
    """
    index, source, output = map(Path, (index, source, output))
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    db = sqlite3.connect(index.as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-65536')
    total = db.execute('SELECT MAX(id) FROM pages').fetchone()[0] + 1
    expected = db.execute("SELECT SUM(length) FROM documents WHERE split IN ('fit','dev')").fetchone()[0]
    temporary = output.with_suffix('.building.npy')
    records = np.lib.format.open_memmap(temporary, mode='w+', dtype=RECORD, shape=(total,))
    seen = np.zeros(total, bool)
    counts = Counter()
    previous, mapping = None, {}
    cache, cache_pages = OrderedDict(), 0
    started = time.monotonic()
    with source.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) != 13:
                raise ValueError('expected 13-column CSV')
            key = row[0], row[3]
            if key != previous:
                previous = key
                if key in cache:
                    mapping = cache.pop(key)
                    cache_pages -= len(mapping)
                    counts['document_cache_hits'] += 1
                else:
                    doc = db.execute('SELECT id,split FROM documents WHERE library=? AND doc=?', key).fetchone()
                    mapping = (dict(db.execute('SELECT page,id FROM pages WHERE document=?', (doc[0],)))
                               if doc and doc[1] in ('fit', 'dev') else {})
                    counts['document_cache_misses'] += 1
                # Bound both document count and total retained page identities.
                if len(mapping) <= 262144:
                    while cache and (len(cache) >= 1024 or cache_pages + len(mapping) > 262144):
                        _, evicted = cache.popitem(last=False)
                        cache_pages -= len(evicted)
                    cache[key] = mapping
                    cache_pages += len(mapping)
            counts['csv_rows'] += 1
            target = mapping.get(row[4])
            if target is None:
                counts['excluded_rows'] += 1
                continue
            if seen[target]:
                raise ValueError('duplicate pagination page row')
            seen[target] = True
            value, system = candidate(row[9])
            records[target] = value, system
            counts['matched'] += 1
            counts['arabic' if system == 1 else 'roman' if system == 2 else 'unavailable'] += 1
            if counts['matched'] % 1000000 == 0:
                print(json.dumps(dict(counts, seconds=round(time.monotonic() - started))), flush=True)
    if counts['matched'] != expected:
        raise ValueError(f'pagination coverage mismatch: {counts["matched"]} != {expected}')
    records.flush()
    del records
    report = {'policy': POLICY, 'rows': total, 'counts': dict(counts),
              'index_sha256': file_sha256(index), 'source': str(source),
              'source_sha256': file_sha256(source), 'sha256': file_sha256(temporary),
              'seconds': time.monotonic() - started,
              'interpretation': 'noisy DB candidates; visible printing not verified; test rows unavailable'}
    temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.index, args.source, args.output)
