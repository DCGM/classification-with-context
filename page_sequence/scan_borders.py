"""Known scan borders: token-aware tiling and page-aligned dataset metadata."""
import numpy as np

from .inference import window_spans


def scan_windows(n, size, stride):
    """Tile START + pages + END; return page spans and their visible markers.

    Window size and stride count network positions, never additional page slots.
    """
    if n < 1 or size < 3:
        raise ValueError('nonempty scans and at least three network positions required')
    result = []
    for lo, hi in window_spans(n + 2, size, stride):
        start, end = max(1, lo) - 1, min(n + 1, hi) - 1
        result.append((start, end, lo == 0, hi == n + 2))
    return result


class ScanBorderDataset:
    """Border flags bypass modality fusion/dropout and never become prediction targets."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, window):
        result = self.dataset[window]
        borders = np.zeros((len(window.rows), 2), np.float32)
        borders[0, 0] = window.scan_start
        borders[-1, 1] = window.scan_end
        result['features'] = dict(result['features'], scan_borders=borders)
        return result
