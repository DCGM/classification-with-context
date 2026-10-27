"""Known scan borders: token-aware tiling and page-aligned dataset metadata."""
import numpy as np

from .inference import window_spans


def scan_windows(n, size, stride, border_tokens=True):
    """Tile START + pages + END; return page spans and their visible markers.

    Window size and stride count network positions, never additional page slots.
    """
    if not border_tokens:
        return [(lo, hi, lo == 0, hi == n) for lo, hi in window_spans(n, size, stride)]
    if n < 1 or size < 3:
        raise ValueError('nonempty scans and at least three network positions required')
    result = []
    for lo, hi in window_spans(n + 2, size, stride):
        start, end = max(1, lo) - 1, min(n + 1, hi) - 1
        result.append((start, end, lo == 0, hi == n + 2))
    return result


def encode_boundary_features(n, scan_start=False, scan_end=False,
                             document_starts=None, document_ends=None):
    """[scan start, scan end, metadata provided, document start, document end].

    Absent metadata is never inferred from training targets. Supplied metadata
    must describe both tasks on every page; partial metadata is unsupported.
    """
    if n < 1:
        raise ValueError('nonempty page sequence required')
    values = np.zeros((n, 5), np.float32)
    values[0, 0], values[-1, 1] = scan_start, scan_end
    if (document_starts is None) != (document_ends is None):
        raise ValueError('complete start and end metadata must be supplied together')
    if document_starts is not None:
        starts, ends = np.asarray(document_starts), np.asarray(document_ends)
        if (starts.shape != (n,) or ends.shape != (n,)
                or not np.isin(starts, [0, 1]).all() or not np.isin(ends, [0, 1]).all()):
            raise ValueError('binary document metadata required for every page')
        values[:, 2], values[:, 3], values[:, 4] = 1, starts, ends
    return values


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


class BoundaryFeatureDataset(ScanBorderDataset):
    """First-stage experiment: scan endpoints known, archive metadata absent."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_dims = dict(getattr(dataset, 'model_dims', dataset.features.dims), boundaries=5)

    def __getitem__(self, window):
        result = self.dataset[window]
        values = encode_boundary_features(len(window.rows), window.scan_start, window.scan_end)
        result['features'] = dict(result['features'], boundaries=values)
        return result
