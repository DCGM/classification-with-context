from collections import OrderedDict
from types import SimpleNamespace

import pytest

from page_sequence.index import PageIndex, build_index
from page_sequence.sampling import WindowSampler
from page_sequence.schema import Page, join_stratum


class SqlReferenceSampler(WindowSampler):
    def _compatible(self, a, b):
        if a.id == b.id or a.copy_group == b.copy_group:
            return False
        return not self.index.db.execute(
            'SELECT 1 FROM pages a JOIN pages b ON a.page=b.page '
            'WHERE a.document=? AND b.document=? LIMIT 1', (a.id, b.id)).fetchone()

    def _cross_volume_pool(self, doc):
        return [d for d in self.periodicals.get((doc.library, doc.periodical), [])
                if join_stratum(doc, d) == 'cross_volume'] if doc.periodical else []


def index_fixture(tmp_path):
    docs, pages = [], []
    for i in range(10):
        docs.append(dict(library='a' if i < 9 else 'b', doc=str(i), split='fit',
                         volume=str(i // 2), periodical='p',
                         copy_group='copy' if i in (4, 5) else str(i)))
        for j in range(12):
            key = 'shared' if j == 3 and i in (0, 1, 9) else f'p{i}-{j}'
            pages.append(Page(docs[-1]['library'], str(i), key, j,
                              db_type='TitlePage' if j == 0 else 'NormalPage',
                              annotated_type='TitlePage' if j == 0 and i % 2 else ''))
    build_index(tmp_path / 'index', docs, pages)
    return PageIndex(tmp_path / 'index')


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_cached_sampling_preserves_windows_counters_and_rng(tmp_path, stage):
    index = index_fixture(tmp_path)
    old = SqlReferenceSampler(index, window_size=32, seed=73, stage=stage)
    new = WindowSampler(index, window_size=32, seed=73, stage=stage)
    for _ in range(1000):
        assert new.draw() == old.draw()
    assert new.counters == old.counters
    assert new.rng.bit_generator.state == old.rng.bit_generator.state
    state = new.state_dict()
    resumed = WindowSampler(index, window_size=32, seed=999)
    resumed.load_state_dict(state)
    for _ in range(100):
        assert resumed.draw() == new.draw()


def test_shared_uuid_and_copy_exclusions_match_sql_across_libraries(tmp_path):
    index = index_fixture(tmp_path)
    old, new = SqlReferenceSampler(index), WindowSampler(index)
    for a in new.docs:
        for b in new.docs:
            assert new._compatible(a, b) == old._compatible(a, b)
    assert not new._compatible(new.docs[0], new.docs[9])
    assert not new._compatible(new.docs[4], new.docs[5])


def test_page_cache_is_bounded_and_reuses_document_reads():
    calls = []
    def execute(sql, args):
        calls.append(args)
        return ((f'{args[0]}-{i}',) for i in range(600))
    sampler = object.__new__(WindowSampler)
    sampler.index = SimpleNamespace(db=SimpleNamespace(execute=execute))
    sampler._page_sets, sampler._cached_pages = OrderedDict(), 0
    for i in range(200):
        doc = SimpleNamespace(id=i)
        assert sampler._page_set(doc) is sampler._page_set(doc)
        assert sampler._cached_pages <= 65536
        assert len(sampler._page_sets) <= 128
    assert len(calls) == 200
    sampler._page_set(SimpleNamespace(id=0))
    assert len(calls) == 201  # evicted entries reload exactly


def periodical_fixture(tmp_path):
    docs, pages = [], []
    for i in range(40):
        library = 'a' if i < 30 else 'b'
        periodical = '' if i % 11 == 0 else 'p' if i % 3 else 'q'
        volume = '' if i % 7 == 0 else str(i % 5)
        docs.append(dict(library=library, doc=str(i), split='fit', volume=volume, periodical=periodical,
                         copy_group=str(i)))
        for j in range(6):
            pages.append(Page(library, str(i), f'p{i}-{j}', j,
                              db_type='TitlePage' if j == 0 else 'NormalPage',
                              annotated_type='TitlePage' if j == 0 and i % 4 == 0 else ''))
    build_index(tmp_path / 'index', docs, pages)
    return PageIndex(tmp_path / 'index')


def test_cross_volume_pool_equals_join_stratum_definition(tmp_path):
    index = periodical_fixture(tmp_path)
    old, new = SqlReferenceSampler(index), WindowSampler(index)
    assert any(len(old._cross_volume_pool(d)) for d in new.docs)
    for doc in new.docs:
        pool = new._cross_volume_pool(doc)
        assert [pool[i] for i in range(len(pool))] == old._cross_volume_pool(doc)


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_scan_sampling_with_vectorized_pools_preserves_windows(tmp_path, stage):
    from page_sequence.scan_sampling import ScanBatchSampler

    class Reference(ScanBatchSampler):
        _cross_volume_pool = SqlReferenceSampler._cross_volume_pool

    index = periodical_fixture(tmp_path)
    settings = dict(scan_batches={'metadata_probability': .5, 'border_tokens': False, 'tail': 'regular'})
    old = Reference(index, 16, 31, stage=stage, **settings)
    new = ScanBatchSampler(index, 16, 31, stage=stage, **settings)
    for _ in range(1000):
        assert new.draw() == old.draw()
    assert new.counters == old.counters
    assert new.rng.bit_generator.state == old.rng.bit_generator.state
