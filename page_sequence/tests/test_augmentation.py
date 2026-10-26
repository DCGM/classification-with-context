import copy
import json
from pathlib import Path

import pytest
import torch

from page_sequence.augmentation import pagination_dropout
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
                  pagination_dropout={'window_probability': .2, 'page_probability': .2},
                  pagination_robustness_evaluation=True,
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
