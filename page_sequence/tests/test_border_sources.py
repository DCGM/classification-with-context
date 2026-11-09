import numpy as np
import pytest

from page_sequence.evaluation import border_result
from page_sequence.inference import deployment_inference
from page_sequence.manifests import load_streams
from page_sequence.runner import evaluate_streams
from page_sequence.tests.test_context_training import model_for, setup_data


def test_border_scores_from_page_edge_heads():
    result = {'page_start_probs': np.array([.9, .7, .2, .6], np.float32),
              'page_end_probs': np.array([.8, .9, .4, .1], np.float32),
              'starts_document_probs': np.array([np.nan, .1, .1, .1], np.float32),
              'pair_mask': np.zeros(4, bool), 'boundary_enabled': False}
    assert border_result(result, 'pair') is result
    for source, expected in (('start', [.7, .2, .6]), ('end', [.8, .9, .4]), ('both', [.7, .2, .4])):
        scored = border_result(result, source)
        np.testing.assert_allclose(scored['starts_document_probs'][1:], expected)
        assert np.isnan(scored['starts_document_probs'][0])
        assert scored['boundary_enabled'] and not scored['pair_mask'][0] and scored['pair_mask'][1:].all()
    # Both heads must fire: the end head alone (pair 2) or the start head alone (pair 3) is not enough.
    assert (border_result(result, 'both')['starts_document_probs'][1:] >= .5).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        border_result(result, 'middle')


@pytest.mark.parametrize('window', [1, 8])
def test_one_prediction_pass_gives_every_source(tmp_path, window):
    config, index, dataset = setup_data(tmp_path)
    model = model_for(dataset).eval()
    streams = load_streams(config['dev_manifest'], index, 'dev')
    inference = deployment_inference(window, 16)
    single = evaluate_streams(model, dataset, streams, inference, 'cpu')
    reports = evaluate_streams(model, dataset, streams, inference, 'cpu', border_sources=['pair', 'start', 'both'])
    for key in ('curated_type', 'placement_db_agreement', 'starts'):
        assert reports['pair'][key] == single[key]
    assert reports['start']['curated_type'] == single['curated_type']
    assert reports['start']['starts']['all']['n'] == sum(s['pages'] - 1 for s in single['streams'])
    assert reports['both']['starts']['all']['tp'] <= reports['start']['starts']['all']['tp']
