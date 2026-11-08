"""Paired comparison of selected checkpoints on the full dev streams, with a bootstrap over streams.

`evaluate` runs a run's selected checkpoint (`finetune_best.pt`) on every dev stream of its manifest with all
its modalities, in the `none` condition of `page_sequence.evaluate_conditions` (no page numbers, no document
boundaries) and the deployed MetaKat merging scaled to its window, exactly as `tools/modality_overview.py`.
Besides the full report it keeps, per stream, the confusion matrices of the curated page type and the DB side
and the boundary counts, so metrics can be recomputed for any resample of streams.

`compare` resamples the streams (paired: the same resample for both models) and reports, for each metric, the
difference B - A with a percentile confidence interval and the fraction of resamples in which B is not better.

Examples:
    python tools/compare_checkpoints.py evaluate --output /home/ikohut/tmp/checkpoint-comparison \\
        --store /home/ikohut/tmp/document-store-ocr-lines-sparse-v2 --device cuda RUN_DIR [RUN_DIR ...]
    python tools/compare_checkpoints.py compare /home/ikohut/tmp/checkpoint-comparison/A.json \\
        /home/ikohut/tmp/checkpoint-comparison/B.json
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TYPE_KEYS = ('curated_type', 'placement_db_agreement')


def evaluate_run(run, output, store=None, device='cuda', limit_streams=None):
    from page_sequence.evaluate_conditions import WithoutPagination
    from page_sequence.evaluation import Evaluator
    from page_sequence.inference import InferenceConfig, deployment_inference
    from page_sequence.runner import evaluation_dataset, file_sha256, load_checkpoint, load_streams
    from page_sequence.runner import predict_stream
    import time
    run = Path(run)
    path = output / f'{run.name}.json'
    if path.exists():
        return path
    checkpoint = run / 'finetune_best.pt'
    model, state = load_checkpoint(checkpoint, device)
    config, dataset = evaluation_dataset(state, store)
    saved = InferenceConfig(**config['inference'])
    inference = deployment_inference(saved.window_size, saved.batch_size, saved.start_threshold)
    streams = load_streams(config['dev_manifest'], dataset.index, 'dev')[:limit_streams]
    data = WithoutPagination(dataset)
    evaluator = Evaluator(inference.start_threshold)
    per_stream = []
    start = time.perf_counter()
    for stream in streams:
        before = {k: evaluator.confusions[k].copy() for k in TYPE_KEYS}
        result = predict_stream(model, data, stream['rows'], inference, device)
        documents = {(d.library, d.doc): d for d in dataset.index.documents_for_rows(stream['rows'])}
        evaluator.add(stream['id'], dataset.index.pages(stream['rows']), result, documents,
                      stream['censored_left'], stream['censored_right'])
        starts = evaluator.streams[-1]['starts']
        entry = {'id': stream['id'], 'starts': {k: starts[k] for k in ('tp', 'fp', 'fn')}}
        for k in TYPE_KEYS:
            delta = evaluator.confusions[k] - before[k]
            nonzero = np.nonzero(delta)
            entry[k] = [[int(i), int(j), int(delta[i, j])] for i, j in zip(*nonzero)]
        per_stream.append(entry)
    report = evaluator.report()
    report.update(seconds=time.perf_counter() - start, condition='none', inference=asdict(inference),
                  inference_policy='deployment', checkpoint=str(checkpoint),
                  checkpoint_sha256=file_sha256(checkpoint), checkpoint_step=state['step'],
                  manifest=config['dev_manifest'], manifest_sha256=file_sha256(config['dev_manifest']),
                  modalities=config.get('enabled_modalities'), per_stream=per_stream)
    report.pop('streams')
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, allow_nan=False) + '\n')
    return path


def load(path):
    report = json.loads(Path(path).read_text())
    sizes = {k: len(report[k]['confusion']) for k in TYPE_KEYS}
    ids = [s['id'] for s in report['per_stream']]
    confusions = {k: np.zeros((len(ids), sizes[k], sizes[k]), np.int64) for k in TYPE_KEYS}
    starts = np.zeros((len(ids), 3), np.int64)
    for i, s in enumerate(report['per_stream']):
        for k in TYPE_KEYS:
            for a, b, c in s[k]:
                confusions[k][i, a, b] = c
        starts[i] = [s['starts'][k] for k in ('tp', 'fp', 'fn')]
    for k in TYPE_KEYS:  # the per-stream counts reproduce the report exactly
        assert confusions[k].sum(0).tolist() == report[k]['confusion']
    assert starts.sum(0).tolist() == [report['starts']['all'][k] for k in ('tp', 'fp', 'fn')]
    return report, ids, confusions, starts


def metrics(confusions, starts, weights):
    """Metrics of the streams weighted by their bootstrap counts (weights: resamples x streams)."""
    out = {}
    for k, name in (('curated_type', 'type'), ('placement_db_agreement', 'side')):
        cm = np.einsum('rs,sij->rij', weights, confusions[k]).astype(float)
        support, predicted, tp = cm.sum(2), cm.sum(1), np.diagonal(cm, axis1=1, axis2=2)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp),
                       where=precision + recall > 0)
        n = support.sum(1)
        out[f'{name}_accuracy'] = tp.sum(1) / n
        out[f'{name}_weighted_f1'] = (f1 * support).sum(1) / n
        present = (support + predicted) > 0
        out[f'{name}_macro_f1'] = (f1 * present).sum(1) / present.sum(1)
    tp, fp, fn = (weights @ starts).T.astype(float)
    out['boundary_precision'] = tp / (tp + fp)
    out['boundary_recall'] = tp / (tp + fn)
    out['boundary_f1'] = 2 * tp / (2 * tp + fp + fn)
    return out


def compare(a, b, resamples=10000, seed=0):
    ra, ids_a, ca, sa = load(a)
    rb, ids_b, cb, sb = load(b)
    if ids_a != ids_b or ra['manifest_sha256'] != rb['manifest_sha256']:
        raise ValueError('the reports cover different streams')
    if ra['inference_policy'] != rb['inference_policy'] or ra['condition'] != rb['condition']:
        raise ValueError('the reports use different evaluation settings')
    rng = np.random.default_rng(seed)
    n = len(ids_a)
    weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(resamples)])
    full = np.ones((1, n), np.int64)
    pa, pb = metrics(ca, sa, full), metrics(cb, sb, full)
    ba, bb = metrics(ca, sa, weights), metrics(cb, sb, weights)
    for k in ('type_weighted_f1', 'type_macro_f1', 'side_weighted_f1', 'boundary_f1'):
        report_key = {'type_weighted_f1': ('curated_type', 'weighted_f1'),
                      'type_macro_f1': ('curated_type', 'macro_f1'),
                      'side_weighted_f1': ('placement_db_agreement', 'weighted_f1')}.get(k)
        if report_key:
            assert abs(pa[k][0] - ra[report_key[0]][report_key[1]]) < 1e-9
    rows = {}
    for k in pa:
        d = bb[k] - ba[k]
        rows[k] = {'a': float(pa[k][0]), 'b': float(pb[k][0]), 'difference': float(pb[k][0] - pa[k][0]),
                   'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                   'p_b_not_better': float((d <= 0).mean()), 'p_b_not_worse': float((d >= 0).mean())}
    return {'a': {'checkpoint': ra['checkpoint'], 'step': ra['checkpoint_step']},
            'b': {'checkpoint': rb['checkpoint'], 'step': rb['checkpoint_step']},
            'streams': n, 'resamples': resamples, 'metrics': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('evaluate')
    run.add_argument('runs', nargs='+')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--store')
    run.add_argument('--device', default='cuda')
    run.add_argument('--limit-streams', type=int)
    cmp = sub.add_parser('compare')
    cmp.add_argument('a')
    cmp.add_argument('b')
    cmp.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.command == 'evaluate':
        import torch
        torch.set_num_threads(4)
        for r in args.runs:
            print(evaluate_run(r, args.output, args.store, args.device, args.limit_streams), flush=True)
    else:
        print(json.dumps(compare(args.a, args.b, args.resamples), indent=2))


if __name__ == '__main__':
    main()
