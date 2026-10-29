import argparse
import random
import time
from types import SimpleNamespace

import lmdb
import pytest
import torch

from embeddings.run_state import append_progress, check_or_write_meta, load_progress
from embeddings.text import inference
from embeddings.text.inference import (
    FIXED_META_FIELDS,
    PROGRESS_HEADER,
    DevicePool,
    EncoderSpec,
    last_token_pool,
    make_batches,
    mean_pool,
    run,
)
from embeddings.text.qwen3.infer_qwen3 import make_text_transform


def test_make_batches_respects_token_budget_and_batch_size():
    lengths = [10, 500, 30, 200, 20, 400]
    batches = make_batches(lengths, max_tokens_per_batch=1000, max_batch_size=3)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) * max(lengths[i] for i in batch) <= 1000 or len(batch) == 1
        assert lengths[batch[0]] == max(lengths[i] for i in batch)


def test_progress_resumes_after_last_complete_line(tmp_path):
    path = tmp_path / 'progress.tsv'
    assert load_progress(path, PROGRESS_HEADER) == (None, 0, 0)
    append_progress(path, ('t', 0, 'a_1', 'a_5', 5, 0, 0, 5, '1.0'), PROGRESS_HEADER)
    append_progress(path, ('t', 1, 'a_6', 'b_2', 5, 1, 0, 10, '1.0'), PROGRESS_HEADER)
    with path.open('a', encoding='utf-8') as f:
        f.write('t\t2\tb_3\tb_9')  # truncated by a kill
    assert path.read_text(encoding='utf-8').startswith('\t'.join(PROGRESS_HEADER))
    assert load_progress(path, PROGRESS_HEADER) == (b'b_2', 10, 2)


def test_meta_mismatch_refuses_resume(tmp_path):
    meta = {'model': 'm', 'model_commit_hash': 'x', 'pooling': 'mean', 'dtype': 'float16', 'dim': 768,
            'max_length': 8192, 'empty_text': 'e', 'source_lmdb': '/s', 'normalization': 'ocr_lines',
            'max_chars': 4000, 'key_list': None, 'instruction': None}
    check_or_write_meta(tmp_path / 'meta.json', meta, FIXED_META_FIELDS)
    check_or_write_meta(tmp_path / 'meta.json', dict(meta), FIXED_META_FIELDS)
    with pytest.raises(SystemExit, match='max_length'):
        check_or_write_meta(tmp_path / 'meta.json', {**meta, 'max_length': 512}, FIXED_META_FIELDS)
    with pytest.raises(SystemExit, match='normalization'):
        check_or_write_meta(tmp_path / 'meta.json', {**meta, 'normalization': 'none'}, FIXED_META_FIELDS)
    fixed = ('model', 'instruction')
    with pytest.raises(SystemExit, match='instruction'):
        check_or_write_meta(tmp_path / 'meta.json', {**meta, 'instruction': 'Classify'}, fixed)


def test_pooling_ignores_right_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [float('nan'), 9.0]],
                           [[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 3.0], [2.0, 4.0]]
    assert last_token_pool(hidden, mask).tolist() == [[3.0, 5.0], [2.0, 4.0]]


def test_qwen_instruction_prefix():
    assert make_text_transform(argparse.Namespace(instruction=None)) is None
    transform = make_text_transform(argparse.Namespace(instruction='Classify the page type'))
    assert transform('Obsah') == 'Instruct: Classify the page type\nQuery:Obsah'


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[2 + ord(c) % 50 for c in text] + [1] for text in texts]}


class FakeEncoder(torch.nn.Module):
    """Embeds tokens and sleeps a random while, so replicas finish chunks out of order."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = torch.nn.Embedding(52, 8)
        self.config = SimpleNamespace(hidden_size=8, max_position_embeddings=512, _commit_hash='fake')

    @property
    def dtype(self):
        return self.embedding.weight.dtype

    def forward(self, input_ids, attention_mask):
        time.sleep(random.uniform(0, 0.01))
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


FAKE_SPEC = EncoderSpec(description='test', default_model='fake', pooling='mean',
                        load=lambda model_id, attn, dtype: (FakeTokenizer(), FakeEncoder()), pool=mean_pool)


@pytest.fixture
def source_lmdb(tmp_path):
    path = tmp_path / 'source'
    with lmdb.open(str(path), map_size=2 ** 24) as env, env.begin(write=True) as txn:
        for i in range(37):
            txn.put(f'lib_{i:03d}'.encode(), ('' if i % 7 == 0 else f'page {i} ' * (i % 5 + 1)).encode())
    return path


def read_output(output_root):
    with lmdb.open(str(output_root / 'fake' / 'lmdb'), readonly=True, lock=False) as env, env.begin() as txn:
        return {key: bytes(value) for key, value in txn.cursor()}


def run_fake(source_lmdb, output_root, *extra):
    run(FAKE_SPEC, ['--source-lmdb', str(source_lmdb), '--output-root', str(output_root), '--chunk-size', '3',
                    '--max-batch-size', '2', '--compute-dtype', 'float32', '--attn-implementation', 'eager',
                    '--max-length', '64', '--map-size-gb', '1', *extra])


def test_several_devices_commit_in_key_order_and_match_one_device(source_lmdb, tmp_path, monkeypatch):
    monkeypatch.setattr(inference, 'select_devices', lambda n_gpu: [torch.device('cpu')])
    run_fake(source_lmdb, tmp_path / 'one')

    monkeypatch.setattr(inference, 'select_devices', lambda n_gpu: [torch.device('cpu')] * 3)
    run_fake(source_lmdb, tmp_path / 'three', '--limit', '10')
    run_fake(source_lmdb, tmp_path / 'three')  # resumes with three replicas

    expected = read_output(tmp_path / 'one')
    assert len(expected) == 37
    assert read_output(tmp_path / 'three') == expected
    rows = [line.split('\t') for line in
            (tmp_path / 'three' / 'fake' / 'progress.tsv').read_text(encoding='utf-8').splitlines()[1:]]
    assert [int(row[1]) for row in rows] == list(range(len(rows)))
    assert [row[2] for row in rows] == sorted(row[2] for row in rows)
    assert int(rows[-1][7]) == 37


def test_device_pool_reports_worker_errors():
    def embed(model, device, chunk):
        raise ValueError(f'bad chunk {chunk}')

    future = DevicePool([(None, torch.device('cpu'))], embed).submit('c0')
    with pytest.raises(ValueError, match='bad chunk c0'):
        future.result(timeout=5)


def test_normalization_reaches_the_tokenizer_and_is_part_of_the_resume_check(tmp_path, monkeypatch):
    seen = []

    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            return super().__call__(texts, **kwargs)

    spec = EncoderSpec(description='test', default_model='fake', pooling='mean', pool=mean_pool,
                       load=lambda model_id, attn, dtype: (RecordingTokenizer(), FakeEncoder()))
    source = tmp_path / 'source'
    with lmdb.open(str(source), map_size=2 ** 24) as env, env.begin(write=True) as txn:
        txn.put(b'lib_1', b'a  b\n\n  \nc \n')
        txn.put(b'lib_2', b'\n \n\t\n')
    monkeypatch.setattr(inference, 'select_devices', lambda n_gpu: [torch.device('cpu')])
    arguments = ['--source-lmdb', str(source), '--output-root', str(tmp_path / 'out'), '--compute-dtype', 'float32',
                 '--attn-implementation', 'eager', '--max-length', '64', '--map-size-gb', '1']
    run(spec, arguments)
    assert seen == ['a b c', '']
    with pytest.raises(SystemExit, match='normalization'):
        run(spec, [*arguments, '--normalization', 'none'])


def test_key_list_embeds_only_those_pages_in_key_order_and_resumes(source_lmdb, tmp_path, monkeypatch):
    monkeypatch.setattr(inference, 'select_devices', lambda n_gpu: [torch.device('cpu')])
    keys = tmp_path / 'keys.txt'
    keys.write_text('lib_030\nlib_002\nlib_015\nlib_021\n', encoding='utf-8')
    run_fake(source_lmdb, tmp_path / 'listed', '--key-list', str(keys), '--limit', '2')
    run_fake(source_lmdb, tmp_path / 'listed', '--key-list', str(keys))
    assert sorted(read_output(tmp_path / 'listed')) == [b'lib_002', b'lib_015', b'lib_021', b'lib_030']
    everything = tmp_path / 'everything'
    run_fake(source_lmdb, everything)
    expected = read_output(everything)
    assert all(read_output(tmp_path / 'listed')[key] == expected[key] for key in expected if key in
               read_output(tmp_path / 'listed'))
    keys.write_text('lib_030\nlib_999\n', encoding='utf-8')
    with pytest.raises(KeyError, match='lib_999'):
        run_fake(source_lmdb, tmp_path / 'missing', '--key-list', str(keys))
