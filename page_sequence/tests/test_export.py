"""Portable inference exports must not retain training pickle or store paths."""
import torch
import pytest

from page_sequence.export import export_checkpoint
from page_sequence.schema import PAGE_TYPES, PLACEMENTS


def test_export_loads_with_weights_only_and_preserves_inference_contract(tmp_path):
    source, output = tmp_path / 'training.pt', tmp_path / 'context.pt'
    weight = torch.arange(6).reshape(2, 3).float()
    state = {'format': 1, 'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
             'specification': {'dims': {'image': 768}}, 'features': {'stats_names': ('a', 'b')},
             'model': {'weight': weight}, 'config': {'inference': {'window_size': 128, 'stride': 64},
                                                     'index': '/data/not-needed.sqlite'},
             'optimizer': {'unused': 1}, 'rng': {'unused': 2}}
    torch.save(state, source)
    export_checkpoint(source, output)
    exported = torch.load(output, weights_only=True)
    assert exported['format'] == 'metakat_context_v1'
    assert exported['features']['stats_names'] == ['a', 'b']
    assert exported['inference'] == {'window_size': 128, 'stride': 64}
    assert not {'config', 'optimizer', 'rng'} & exported.keys()
    torch.testing.assert_close(exported['model']['weight'], weight)


def test_export_rejects_wrong_label_order_without_overwriting_destination(tmp_path):
    source, output = tmp_path / 'training.pt', tmp_path / 'context.pt'
    torch.save({'format': 1, 'page_types': tuple(reversed(PAGE_TYPES)), 'placements': PLACEMENTS}, source)
    output.write_bytes(b'previous export')
    with pytest.raises(ValueError, match='schema/label'):
        export_checkpoint(source, output)
    assert output.read_bytes() == b'previous export'


def test_export_pagination_contract_without_training_paths(tmp_path):
    from page_sequence.pagination import POLICY
    source, output = tmp_path / 'training.pt', tmp_path / 'context.pt'
    state = {'format': 1, 'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
             'specification': {'dims': {'image': 768, 'pagination': 5}}, 'features': {},
             'model': {'weight': torch.ones(2)},
             'config': {'pagination': {'path': '/private/training/pagination.npy', 'cap': 20000}},
             'provenance': {'pagination': {'policy': POLICY, 'path': '/private/training/pagination.npy'}}}
    torch.save(state, source)
    export_checkpoint(source, output)
    exported = torch.load(output, weights_only=True)
    assert exported['format'] == 'metakat_context_v2'
    assert exported['pagination'] == {'policy': POLICY, 'cap': 20000}
    assert 'private' not in repr(exported)
    state['provenance']['pagination']['policy'] = 'other'
    torch.save(state, source)
    with pytest.raises(ValueError, match='pagination'):
        export_checkpoint(source, output)
