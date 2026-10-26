import copy

import numpy as np
import torch

from page_sequence.robustness import corrupt_examples
from page_sequence.context_evaluation import evaluate_contexts
from page_sequence.data import WindowDataset
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.models import PageBackbone, PageTagger
from page_sequence.smoke import prepare_fixture


def test_fixed_corruption_all_signals_targets_and_repeated_identities():
    example = {'rows': tuple(range(1000)), 'type': np.arange(1000),
               'features': {k: np.ones((1000, 5)) for k in ('image', 'text', 'stats', 'pagination')}}
    example['features']['presence'] = np.ones((1000, 3))
    original = copy.deepcopy(example)
    a, b = [copy.deepcopy(example)], [copy.deepcopy(example)]
    before = torch.get_rng_state().clone()
    ra = corrupt_examples(a, page_probability=.5, apply=False)
    rb = corrupt_examples(b, page_probability=.5)
    assert torch.equal(before, torch.get_rng_state())
    assert ra['mask_sha256'] == rb['mask_sha256']
    assert (a[0]['diagnostic_mask'] == b[0]['diagnostic_mask']).all()
    for k in example['features']:
        np.testing.assert_array_equal(a[0]['features'][k], original['features'][k])
        assert not b[0]['features'][k][b[0]['diagnostic_mask']].any()
    np.testing.assert_array_equal(b[0]['type'], original['type'])
    repeated = [copy.deepcopy(example), copy.deepcopy(example)]
    corrupt_examples(repeated)
    np.testing.assert_array_equal(repeated[0]['features']['image'], repeated[1]['features']['image'])


def test_evaluation_group_partition_and_unmodified_dataset(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = WindowDataset(index, features)
    from page_sequence.manifests import load_streams
    blocks = [tuple(load_streams(config['dev_manifest'], index, 'dev')[0]['rows'][:16])]
    model = PageTagger(PageBackbone(16), features.dims, fusion='separate_mlp')
    state = torch.get_rng_state().clone()
    for apply in (False, True):
        report = evaluate_contexts(model, dataset, blocks, [1, 8], corruption={'page_probability': .5, 'apply': apply})
        assert report['complete'] and model.training
        for scores in report['lengths'].values():
            a, b = (scores['groups'][k] for k in ('masked_pages', 'retained_pages'))
            assert a['pages'] + b['pages'] == scores['pages']
            for name in ('curated_type', 'weak_type_agreement', 'placement_db_agreement'):
                assert a[name]['n'] + b[name]['n'] == scores[name]['n']
            if scores['starts']:
                assert sum(x['starts']['n'] if x['starts'] else 0 for x in (a, b)) == scores['starts']['n']
    assert torch.equal(state, torch.get_rng_state())
