import copy
import json
from pathlib import Path

import pytest
import torch

from page_sequence.augmentation import modality_dropout, pagination_dropout, scattered_dropout
from page_sequence.runner import load_checkpoint, rng_state, restore_rng, train
from page_sequence.smoke import prepare_fixture
from page_sequence.tests.test_pagination import sidecar


def test_dropout_missing_representation_rates_and_rng():
    features = {'pagination': torch.ones(10000, 32, 5, dtype=torch.float16),
                'presence': torch.ones(10000, 32, 3)}
    features['pagination'][:, 0] = 0  # naturally missing page
    before = rng_state()
    assert pagination_dropout(features) is features
    torch.testing.assert_close(before['torch'], rng_state()['torch'])
    a = pagination_dropout(features, .2, .2)
    restore_rng(before)
    b = pagination_dropout(features, .2, .2)
    torch.testing.assert_close(a['pagination'], b['pagination'])
    x = a['pagination'][:, 1:]
    assert abs(float(x[..., 4].mean()) - .64) < .015
    assert abs(float((x[..., 4].sum(-1) == 0).float().mean()) - .2) < .015
    assert (x == x[..., 4:5]).all()  # all five components dropped together
    assert not a['pagination'][:, 0].any()
    assert features['pagination'][:, 1:].all()  # original batch unmodified
    assert a['presence'] is features['presence']
    assert not pagination_dropout(features, 1, 0)['pagination'].any()
    assert not pagination_dropout(features, 0, 1)['pagination'].any()


@pytest.mark.parametrize('p', [-.1, 1.1, float('nan'), float('inf')])
def test_reject_invalid_probability(p):
    with pytest.raises(ValueError, match='probabilities'):
        pagination_dropout({}, p, 0)


def test_dropout_resume_matches_uninterrupted_and_final_ablation(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=4)
    path = tmp_path / 'pagination.npy'
    sidecar(config, path)
    config.update(pagination={'path': str(path)},
                  modality_dropout={'all_probability': .7, 'two_probability': .21,
                                    'single_probability': .09},
                  pagination_dropout={'window_probability': .2, 'page_probability': .2},
                  pagination_robustness_evaluation=True,
                  scattered_dropout={'modality_probability': .1, 'page_probability': .05},
                  robustness_evaluation=True,
                  context_evaluation={'enabled': True, 'sample_pages': 20, 'final_sample_pages': 20})
    train(config)
    original, _ = load_checkpoint(Path(config['run_dir']) / 'latest.pt')
    resumed_config = copy.deepcopy(config)
    resumed_config['run_dir'] = str(tmp_path / 'resumed')
    train(resumed_config, stop_after=2)
    checkpoint = Path(resumed_config['run_dir']) / 'latest.pt'
    train(resumed_config, resume=checkpoint)
    resumed, _ = load_checkpoint(checkpoint)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    available = json.loads((Path(config['run_dir']) / 'final-context-evaluation.json').read_text())
    missing = json.loads((Path(config['run_dir']) / 'final-context-without-pagination.json').read_text())
    assert missing['complete'] and missing['pagination'] == 'omitted'
    assert missing['panel_sha256'] == available['panel_sha256']
    assert missing['checkpoint_step'] == available['checkpoint_step']
    reports = list(Path(config['run_dir']).glob('final-robustness-*.json'))
    assert len(reports) == 9
    paired = json.loads((Path(config['run_dir']) / 'final-robustness-normal_paired.json').read_text())
    scattered = json.loads((Path(config['run_dir']) / 'final-robustness-scattered.json').read_text())
    assert paired['corruption']['mask_sha256'] == scattered['corruption']['mask_sha256']
    assert all(json.loads(p.read_text())['complete'] for p in reports if 'summary' not in p.name)


def test_modality_subsets_rates_window_consistency_and_rng():
    features = {name: torch.ones(30000, 4, 5) for name in ('image', 'text', 'stats', 'pagination')}
    features['presence'] = torch.ones(30000, 4, 3)
    before = rng_state()
    assert modality_dropout(features) is features
    torch.testing.assert_close(before['torch'], rng_state()['torch'])
    a = modality_dropout(features, .7, .21, .09)
    restore_rng(before)
    b = modality_dropout(features, .7, .21, .09)
    torch.testing.assert_close(a['presence'], b['presence'], rtol=0, atol=0)
    flags = a['presence'][:, 0]
    assert (a['presence'] == flags[:, None]).all()
    for count, expected in [(3, .7), (2, .21), (1, .09)]:
        assert abs(float((flags.sum(-1) == count).float().mean()) - expected) < .01
    for i, name in enumerate(('image', 'text', 'stats')):
        assert abs(float(((flags.sum(-1) == 1) & (flags[:, i] == 1)).float().mean()) - .03) < .005
        assert (a[name] == a['presence'][..., i:i + 1]).all()
        assert features[name].all()
    assert features['presence'].all()
    assert a['pagination'] is features['pagination']


def test_modality_natural_missingness_never_loses_last_signal():
    flags = torch.tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]]).repeat(100, 1, 1)
    features = {name: torch.ones(100, 4, 2) for name in ('image', 'text', 'stats')}
    features['presence'] = flags
    a = modality_dropout(features, 0, 0, 1)
    torch.testing.assert_close(a['presence'], flags)


@pytest.mark.parametrize('probabilities', [(1, .1, 0), (-.1, .5, .6), (float('nan'), 0, 0)])
def test_modality_reject_invalid_distribution(probabilities):
    with pytest.raises(ValueError, match='probabilities'):
        modality_dropout({}, *probabilities)


def test_scattered_rates_complete_pages_and_no_rescue():
    features = {name: torch.ones(1000, 64, 5) for name in ('image', 'text', 'stats', 'pagination')}
    features['presence'] = torch.ones(1000, 64, 3)
    before = rng_state()
    assert scattered_dropout(features) is features
    torch.testing.assert_close(before['torch'], rng_state()['torch'])
    a = scattered_dropout(features, .1, .05)
    restore_rng(before)
    b = scattered_dropout(features, .1, .05)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert features[key].all()
    missing = a['pagination'][..., 4] == 0
    assert abs(float(missing.float().mean()) - .05) < .005
    for i, key in enumerate(('image', 'text', 'stats')):
        assert not a[key][missing].any()
        assert (a[key] == a['presence'][..., i:i + 1]).all()
    assert abs(float(a['presence'][~missing].mean()) - .9) < .01
    assert all(not x.any() for x in scattered_dropout(features, 0, 1).values())
    single = dict(features, presence=features['presence'] * torch.tensor([1, 0, 0]))
    dropped = scattered_dropout(single, 1, 0)
    assert not dropped['presence'].any()  # no rescue of the last modality
    assert dropped['pagination'].all()  # modality-only failures do not remove pagination


@pytest.mark.parametrize('p', [-.1, 1.1, float('nan'), float('inf')])
def test_scattered_reject_invalid_probability(p):
    with pytest.raises(ValueError, match='probabilities'):
        scattered_dropout({}, p, 0)
