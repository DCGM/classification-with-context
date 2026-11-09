import pytest
import torch

from page_sequence.models import BertBackbone, PageTagger, from_specification

DIMS = {'image': 8, 'text': 8, 'stats': 4, 'presence': 3, 'pagination': 5, 'boundaries': 5}


def model(**kwargs):
    torch.manual_seed(3)
    backbone = BertBackbone.create(pretrained=False, config={
        'vocab_size': 8, 'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 4,
        'intermediate_size': 32, 'max_position_embeddings': 64,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.})
    return PageTagger(backbone, DIMS, 'separate_mlp', **kwargs).eval()


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    features = {k: torch.rand(2, 6, d, generator=g) for k, d in DIMS.items()}
    return features, torch.ones(2, 6, dtype=torch.bool)


def outputs(m, features, mask):
    return {k: v for k, v in m(features, mask).items() if k != 'pair_mask'}


@pytest.mark.parametrize('disabled, feature, columns', [
    (['document_metadata'], 'boundaries', slice(2, 5)),
    (['scan_flags'], 'boundaries', slice(0, 2)),
    (['page_numbers'], 'pagination', slice(None))])
def test_a_disabled_input_has_no_effect_and_the_rest_still_does(disabled, feature, columns):
    m = model(disabled_inputs=disabled)
    features, mask = batch(0)
    changed = dict(features, **{feature: features[feature].clone()})
    changed[feature][..., columns] = torch.rand_like(changed[feature][..., columns])
    for key, value in outputs(m, features, mask).items():
        torch.testing.assert_close(value, outputs(m, changed, mask)[key], rtol=0, atol=0)
    other = dict(features, image=torch.rand_like(features['image']))
    assert not torch.equal(outputs(m, features, mask)['type_logits'], outputs(m, other, mask)['type_logits'])
    if feature == 'boundaries':  # the other bits of the same feature still reach the model
        kept = [i for i in range(5) if i not in range(5)[columns]]
        other = dict(features, boundaries=features['boundaries'].clone())
        other['boundaries'][..., kept] = 1 - other['boundaries'][..., kept]
        assert not torch.equal(outputs(m, features, mask)['type_logits'], outputs(m, other, mask)['type_logits'])
    assert torch.equal(features[feature], batch(0)[0][feature])  # the caller's tensors are not modified


def test_all_signals_off_specification_and_validation():
    m = model(disabled_inputs=['page_numbers', 'scan_flags', 'document_metadata'], auxiliary_edges=False)
    assert m.disabled_inputs == ('document_metadata', 'scan_flags', 'page_numbers')
    restored = from_specification(m.specification())
    restored.load_state_dict(m.state_dict())
    features, mask = batch(1)
    for key, value in outputs(m.eval(), features, mask).items():
        torch.testing.assert_close(value, outputs(restored.eval(), features, mask)[key])
    assert 'disabled_inputs' not in model().specification()
    with pytest.raises(ValueError):
        model(disabled_inputs=['page_numbrs'])


def test_scan_flags_omitted_for_an_evaluation_only():
    from page_sequence.evaluate_conditions import CONDITIONS, without_scan_flags
    m, blind = model(), model(disabled_inputs=['scan_flags'])
    blind.load_state_dict(m.state_dict())
    features, mask = batch(2)
    normal = outputs(m, features, mask)
    with without_scan_flags(m, True):
        assert m.disabled_inputs == ('scan_flags',)
        for key, value in outputs(m, features, mask).items():
            torch.testing.assert_close(value, outputs(blind, features, mask)[key], rtol=0, atol=0)
    assert m.disabled_inputs == ()
    for key, value in outputs(m, features, mask).items():
        torch.testing.assert_close(value, normal[key], rtol=0, atol=0)
    with without_scan_flags(blind, True):
        assert blind.disabled_inputs == ('scan_flags',)
    assert sum(not scan for _, _, scan in CONDITIONS.values()) == 4
