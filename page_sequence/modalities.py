"""Permanent information ablations; pagination and boundary inputs are independent."""
import numpy as np

MODALITIES = ('image', 'text', 'stats')


def enabled_modalities(names=None):
    if names is None:
        return MODALITIES
    if isinstance(names, str):
        raise ValueError('enabled_modalities must be a nonempty list of modality names')
    names = tuple(names)
    if not names or len(set(names)) != len(names) or set(names) - set(MODALITIES):
        raise ValueError('enabled_modalities must be a nonempty unique subset of image/text/stats')
    return tuple(name for name in MODALITIES if name in names)


class ModalitySubsetDataset:
    """Mask before augmentation, so dropout rescue cannot re-enable excluded inputs.

    All feature dimensions and model parameter counts stay unchanged. Masked
    inputs carry no availability bit and cannot reveal natural missingness.
    """
    def __init__(self, dataset, names):
        self.dataset, self.enabled = dataset, enabled_modalities(names)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, window):
        example = self.dataset[window]
        features = dict(example['features'])
        flags = features['presence'].copy()
        for i, name in enumerate(MODALITIES):
            if name not in self.enabled:
                features[name] = np.zeros_like(features[name])
                flags[..., i] = 0
        features['presence'] = flags
        return dict(example, features=features)
