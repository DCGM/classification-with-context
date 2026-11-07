"""Window-size experiments: anchor windows cut consecutively, coverage scans scanned with overlap."""
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from page_sequence.context_training import piece_spans, scan_context
from page_sequence.data import collate_windows
from page_sequence.inference import window_spans
from page_sequence.runner import load_checkpoint, train
from page_sequence.scan_sampling import ScanBatchSampler
from page_sequence.schema import Window
from page_sequence.tests.test_context_training import setup_data


@pytest.mark.parametrize('size', [256, 128, 64, 32, 16, 8, 2])
def test_coverage_tiles_together_scan_the_whole_scan_with_overlap(size):
    for n in [1, 7, 255, 256, 511, 512, 513, 700, 768, 769, 1024, 1100, 2047, 2048, 2300]:
        pieces = []
        for lo, hi in window_spans(n, 512, 256, 'regular'):
            spans = piece_spans(hi - lo, size, True, hi == n, 256, size // 2)
            assert all(0 <= a < b <= hi - lo for a, b in spans)
            pieces += [(lo + a, lo + b) for a, b in spans]
        assert pieces == window_spans(n, size, size // 2, 'regular')
        assert {p for a, b in pieces for p in range(a + 1, b)} == set(range(1, n))


@pytest.mark.parametrize('size', [256, 64, 8, 3])
def test_anchor_windows_are_cut_consecutively(size):
    for n in [1, 5, 300, 512]:
        spans = piece_spans(n, size, False, False, 256, max(1, size // 2))
        assert [p for a, b in spans for p in range(a, b)] == list(range(n))
        assert all(b - a == size for a, b in spans[:-1])


def test_scan_context_keeps_pages_and_weights_each_page_once(tmp_path):
    _, index, ds = setup_data(tmp_path)
    d = index.documents('fit')[0]
    rows = tuple(range(d.first, d.first + 12))
    # A scan of 12 pages tiled by parents of 8 at stride 4 (regular tail), plus an anchor window.
    tiles = [Window(rows[lo:hi], 'coverage', lo == 0, hi == 12) for lo, hi in window_spans(12, 8, 4, 'regular')]
    anchor = Window(rows[2:9], 'interesting')
    batch = collate_windows([ds[w] for w in tiles + [anchor]])
    pieces = scan_context(batch, 4, parent_stride=4, coverage_stride=2)
    assert [tuple(r) for r in pieces['rows'][:5]] == [rows[a:b] for a, b in window_spans(12, 4, 2, 'regular')]
    assert [tuple(r) for r in pieces['rows'][5:]] == [rows[2:6], rows[6:9]]
    total = Counter()
    for piece, weight in zip(pieces['rows'], pieces['weight']):
        for r, w in zip(piece, weight.tolist()):
            total[r] += w
    assert all(abs(w - 1) < 1e-6 for w in total.values())
    assert not pieces['pair_mask'][:, 0].any() and not pieces['pair_weight'][:, 0].any()
    real = pieces['attention_mask']
    assert int(real.sum()) == sum(len(p) for p in pieces['rows'])
    # Features and targets travel with their pages; the scan flags are not invented at cuts.
    for i, piece in enumerate(pieces['rows']):
        b, offset = next((b, tile.rows.index(piece[0])) for b, tile in enumerate(tiles + [anchor])
                         if set(piece) <= set(tile.rows))
        n = len(piece)
        for name, x in batch['features'].items():
            torch.testing.assert_close(pieces['features'][name][i, :n], x[b, offset:offset + n])
        torch.testing.assert_close(pieces['type'][i, :n], batch['type'][b, offset:offset + n])
    boundaries = pieces['features']['boundaries']
    assert boundaries[..., 0].sum() == 1 and boundaries[..., 1].sum() == 1


def test_regular_tail_draws_and_coverage_training_resume(tmp_path):
    config, index, _ = setup_data(tmp_path)
    scan = {'stride': 6, 'border_tokens': False, 'metadata_probability': .5, 'tail': 'regular'}
    sampler = ScanBatchSampler(index, 12, seed=3, variable_lengths=True, scan_batches=scan)
    for stage in ('pretrain', 'finetune'):
        sampler.set_stage(stage)
        windows = [sampler.draw() for _ in range(300)]
    assert any(w.source == 'coverage' and len(w.rows) < 12 and not w.scan_start for w in windows)
    config.update(window_size=4, sampling_window_size=12, context_microbatch_size=16, context_coverage_stride=2,
                  auxiliary_edges=True, scan_batches=scan, pretrain_steps=2, finetune_steps=2,
                  validate_every=2, document_metadata_evaluation=True)
    config['inference'] = dict(window_size=4, stride=3, edge_ramp=1, batch_size=16, edge_power=3, tail='regular')
    train(config, stop_after=2)
    run = Path(config['run_dir'])
    train(config, resume=run / 'latest.pt')
    for name in ['pretrain_final.pt', 'finetune_init.pt', 'finetune_final.pt']:
        assert (run / name).exists()
    _, state = load_checkpoint(run / 'latest.pt')
    assert state['sampler']['window_size'] == 12 and state['sampler']['scan_batches']['tail'] == 'regular'
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert records[-1]['context_partition'] == dict(parent_size=12, context_size=4, microbatch=16,
                                                    parent_stride=6, coverage_stride=2)
    assert all(r['loss']['starts'] > 0 for r in records)


def test_coverage_scanning_rejects_misaligned_settings(tmp_path):
    config, _, _ = setup_data(tmp_path)
    config.update(window_size=4, sampling_window_size=12, context_microbatch_size=16, context_coverage_stride=4,
                  scan_batches={'stride': 6, 'border_tokens': False, 'tail': 'regular'})
    with pytest.raises(ValueError, match='coverage scanning'):
        train(config)
    config.update(context_coverage_stride=2, scan_batches={'stride': 6, 'border_tokens': False})
    with pytest.raises(ValueError, match='coverage scanning'):
        train(config)
