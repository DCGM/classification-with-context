"""Window materialization, task masks and padding. No backbone-specific behavior."""
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import IGNORE


class WindowDataset(Dataset):
    """Accepts Window keys (or integers if a fixed manifest is supplied)."""
    def __init__(self, index, features, windows=None):
        self.index, self.features, self.windows = index, features, windows

    def __len__(self):
        if self.windows is None:
            raise TypeError('dynamic window dataset has no fixed length')
        return len(self.windows)

    def __getitem__(self, window):
        if isinstance(window, int):
            window = self.windows[window]
        pages = self.index.pages(window.rows)
        return make_example(pages, self.features, window)


class PageListDataset(Dataset):
    """Inference over caller-ordered metadata, including scans with unknown document identities."""
    def __init__(self, pages, features):
        self.pages, self.features = pages, features

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, window):
        return make_example([self.pages[i] for i in window.rows], self.features, window)


def make_example(pages, features, window):
    if not pages:
        raise ValueError('empty window')
    labels, gold, side = zip(*(p.target() for p in pages))
    starts = np.array([0] + [a.doc_key != b.doc_key for a, b in zip(pages, pages[1:])], np.float32)
    pair_mask = np.ones(len(pages), bool)
    pair_mask[0] = False
    return {'features': features(pages), 'type': np.array(labels, np.int64),
            'gold': np.array(gold, bool), 'placement': np.array(side, np.int64),
            'starts': starts, 'pair_mask': pair_mask, 'rows': window.rows, 'source': window.source}


def collate_windows(examples):
    """Right pad; repeated target occurrences in one optimizer batch get reciprocal weight."""
    if not examples:
        raise ValueError('empty batch')
    b, l = len(examples), max(len(e['rows']) for e in examples)
    features = {name: torch.zeros(b, l, x.shape[-1], dtype=torch.from_numpy(x).dtype)
                for name, x in examples[0]['features'].items()}
    result = {'features': features, 'attention_mask': torch.zeros(b, l, dtype=torch.bool),
              'type': torch.full((b, l), IGNORE, dtype=torch.long),
              'gold': torch.zeros(b, l, dtype=torch.bool),
              'placement': torch.full((b, l), IGNORE, dtype=torch.long),
              'starts': torch.zeros(b, l), 'pair_mask': torch.zeros(b, l, dtype=torch.bool),
              'weight': torch.zeros(b, l), 'pair_weight': torch.zeros(b, l),
              'rows': [e['rows'] for e in examples],
              'storage_pages': {name: sum(e.get('storage_pages', {}).get(name, 0) for e in examples)
                                for name in ('nvme', 'fallback')}}
    occurrences = Counter(row for e in examples for row in e['rows'])
    # The same start page paired with a different synthetic predecessor is a different task target.
    pairs = Counter((a, c) for e in examples for a, c in zip(e['rows'], e['rows'][1:]))
    for i, e in enumerate(examples):
        n = len(e['rows'])
        for name, x in e['features'].items():
            features[name][i, :n] = torch.from_numpy(x)
        result['attention_mask'][i, :n] = True
        for name in ('type', 'gold', 'placement', 'starts', 'pair_mask'):
            result[name][i, :n] = torch.from_numpy(e[name])
        result['weight'][i, :n] = torch.tensor([1 / occurrences[r] for r in e['rows']])
        result['pair_weight'][i, 1:n] = torch.tensor([1 / pairs[p] for p in zip(e['rows'], e['rows'][1:])])
    return result


def to_device(batch, device, feature_dtype=None):
    return {k: ({n: x.to(device=device, dtype=feature_dtype) for n, x in v.items()} if k == 'features'
                else v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
