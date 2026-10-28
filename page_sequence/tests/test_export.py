"""Portable exports preserve all heads and reject obsolete deployment architectures."""
import torch
import pytest

from page_sequence.export import export_checkpoint
from page_sequence.models import BertBackbone, PageTagger, from_specification
from page_sequence.pagination import POLICY
from page_sequence.schema import PAGE_TYPES, PLACEMENTS


def checkpoint():
    backbone = BertBackbone.create(pretrained=False, config={
        'hidden_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
        'intermediate_size': 16, 'max_position_embeddings': 32, 'vocab_size': 1,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.})
    backbone.initial_scale = .02
    model = PageTagger(backbone, {'image': 4, 'text': 3, 'stats': 2, 'pagination': 5, 'boundaries': 5},
                       'separate_mlp', auxiliary_edges=True).eval()
    state = {'format': 1, 'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
             'specification': model.specification(), 'features': {'stats_names': ('a', 'b')},
             'model': model.state_dict(),
             'config': {'inference': {'window_size': 16, 'stride': 8},
                        'pagination': {'path': '/private/pagination.npy', 'cap': 20000}},
             'provenance': {'pagination': {'policy': POLICY, 'path': '/private/pagination.npy'}},
             'optimizer': {'unused': 1}, 'rng': {'unused': 2}}
    return state, model


def test_export_preserves_all_five_heads_and_inference_without_training_paths(tmp_path):
    source, output = tmp_path / 'training.pt', tmp_path / 'context.pt'
    state, model = checkpoint()
    torch.save(state, source)
    export_checkpoint(source, output)
    exported = torch.load(output, weights_only=True)
    assert exported['format'] == 'metakat_context_v4'
    assert exported['specification']['auxiliary_edges'] is True
    assert exported['features']['stats_names'] == ['a', 'b']
    assert exported['inference']['window_size'] == 512
    assert exported['inference']['overlap'] == 64
    assert exported['inference']['overlap_merge'] == {
        'method': 'center_logits', 'ramp_fraction_of_overlap': .75,
        'ignore_fraction_of_ramp': .25, 'power': 3}
    assert 'stride' not in exported['inference']
    assert exported['inference']['tail'] == 'regular'
    assert exported['pagination'] == {'policy': POLICY, 'cap': 20000}
    assert not {'config', 'optimizer', 'rng'} & exported.keys()
    assert 'private' not in repr(exported)
    loaded = from_specification(exported['specification']).eval()
    loaded.load_state_dict(exported['model'], strict=True)
    for n in (1, 7, 16):
        features = {k: torch.randn(1, n, d) for k, d in model.dims.items()}
        features['presence'] = torch.ones(1, n, 3)
        attention = torch.ones(1, n, dtype=torch.bool)
        with torch.inference_mode():
            expected, actual = model(features, attention), loaded(features, attention)
        assert {'page_start_logits', 'page_end_logits'} <= actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize('change', ['labels', 'no_heads', 'missing_head', 'fusion', 'backbone', 'pagination', 'policy', 'no_boundaries'])
def test_export_rejects_unsupported_models_without_overwriting_destination(tmp_path, change):
    source, output = tmp_path / 'training.pt', tmp_path / 'context.pt'
    state, _ = checkpoint()
    if change == 'labels':
        state['page_types'] = tuple(reversed(PAGE_TYPES))
    elif change == 'no_heads':
        state['specification'].pop('auxiliary_edges')
    elif change == 'missing_head':
        state['model'].pop('page_end_head.weight')
    elif change in ('fusion', 'backbone'):
        if change == 'fusion':
            state['specification']['fusion'] = 'linear'
        else:
            state['specification']['backbone']['kind'] = 'page'
    elif change == 'pagination':
        state['specification']['dims'].pop('pagination')
    elif change == 'no_boundaries':
        state['specification']['dims'].pop('boundaries')
    else:
        state['provenance']['pagination']['policy'] = 'other'
    torch.save(state, source)
    output.write_bytes(b'previous export')
    with pytest.raises((ValueError, RuntimeError)):
        export_checkpoint(source, output)
    assert output.read_bytes() == b'previous export'
