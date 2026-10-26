import copy
import pickle

import numpy as np
import pytest
import torch

from page_sequence.cardinality import PairCardinality
from page_sequence.checkpoint_migration import compact_sampler, protected_digest, migrate_run
from page_sequence.index import PageIndex
from page_sequence.sampling import WindowSampler
from page_sequence.smoke import prepare_fixture


def test_cardinality_fixed_memory_accuracy_duplicates_and_resume():
    counter = PairCardinality()
    assert counter.estimate() == 0
    before = torch.get_rng_state().clone()
    size = len(pickle.dumps(counter.state_dict()))
    for i in range(200000):
        counter.add((i, i + 1))
        counter.add((i, i + 1))
    assert len(pickle.dumps(counter.state_dict())) == size
    assert abs(counter.estimate() / 200000 - 1) < .03
    saved = counter.state_dict()
    restored = PairCardinality(saved)
    for i in range(200000, 220000):
        counter.add((i, 0))
        restored.add((i, 0))
    assert counter.state_dict() == restored.state_dict()
    assert counter.estimate() == restored.estimate()
    assert saved['registers'] != counter.state_dict()['registers']  # independent snapshot
    assert torch.equal(before, torch.get_rng_state())


def test_legacy_sampler_migration_preserves_draws_and_independent_snapshots(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    original = WindowSampler(index, 128, variable_lengths=True)
    for _ in range(12):
        original.draw()
    legacy = original.state_dict()
    del legacy['pair_cardinality']
    legacy['seen_pairs'] = {(i, i + 1) for i in range(10000)}
    expected = copy.deepcopy(legacy)
    resumed = WindowSampler(index, 128, variable_lengths=True)
    resumed.load_state_dict(legacy)
    assert legacy == expected  # migration must not mutate the loaded checkpoint
    state = resumed.state_dict()
    assert 'seen_pairs' not in state
    assert len(state['pair_cardinality']['registers']) == 16384
    assert [original.draw() for _ in range(30)] == [resumed.draw() for _ in range(30)]
    resumed.coverage_order.reverse()
    resumed.counters['source:coverage'] += 1
    resumed.seen_volumes.add(('test', 'test'))
    assert state['coverage_order'] != resumed.coverage_order
    assert state['counters'] != resumed.counters
    assert state['seen_volumes'] != resumed.seen_volumes
    assert np.isfinite(resumed.pair_cardinality.estimate())


def test_migration_preserves_training_state_and_rejects_unknown_source(tmp_path):
    state = {'model': {'w': torch.arange(8).float()}, 'optimizer': {'momentum': torch.ones(8)},
             'rng': {'torch': torch.get_rng_state()}, 'config': {'seed': 17},
             'provenance': {'code_sha256': 'old', 'index_sha256': 'immutable'},
             'sampler': {'rng': {'seed': 123}, 'coverage_order': [3, 2, 1],
                         'seen_pairs': {(1, 2), (2, 3)}, 'seen_volumes': {('lib', 'vol')},
                         'counters': {'unique_join_pairs': 2, 'processed_pages': 20}}}
    before = protected_digest(state)
    compact_sampler(state)
    state['provenance']['code_sha256'] = 'new'
    assert protected_digest(state) == before
    state['model']['w'][0] = 100
    assert protected_digest(state) != before
    with pytest.raises((ValueError, FileNotFoundError)):
        migrate_run(tmp_path, tmp_path, tmp_path)
    assert not (tmp_path / 'memory-fix-originals').exists()
