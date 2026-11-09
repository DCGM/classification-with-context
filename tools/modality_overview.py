"""Modality overview: every model on every combination of main signals it can handle.

`evaluate` runs the selected checkpoint (`finetune_best.pt`) of each given run on the full dev streams of its
manifest, once per nonempty subset of the run's enabled modalities (image/text/stats), in the `none` condition
of `page_sequence.evaluate_conditions`: no page numbers and no document boundaries, only the scan endpoints.
Windows are merged with the policy deployed in MetaKat, scaled to each model's window (overlap W/8, regular tail,
cubic centre weights, ramp 3/4 and ignored edge 3/16 of the overlap), not with the checkpoint's validation settings.
All checkpoints are evaluated in one process from one copy of the native store (`--store`), so the catalogue is
hashed once; `evaluation_dataset` still checks each checkpoint's store identity. Reports go to
`<output>/<run name>/<subset>/none.json`; existing reports are kept, so the command can be repeated as runs finish.

`table` writes the Markdown overview: one section per combination (STI, ST, SI, TI, S, T, I) listing every model
that can handle it, with precision, recall and F1 of the curated page type and of the DB side (left/right/single),
both support-weighted averages over the classes (as in the README's page type evaluation), and of document starts.
The r22 context-window series (`-b64parent512-` ... `-r22`, evaluated with `--subset image+text+stats` only) gets a
section of its own, ordered by window, instead of rows in the combination sections; every run with border reports of
the page edge heads gets a row in the head comparison.

Each checkpoint runs once per combination: all its outputs are stored in `<output>/<run>/<subset>/none-predictions/`
(`page_sequence.stored_predictions`) and every report is derived from them, so new statistics need no GPU.

Examples:
    python tools/modality_overview.py evaluate --output /home/ikohut/tmp/modality-overview-deployment \\
        --store /home/ikohut/tmp/document-store-ocr-lines-sparse-v2 --device cuda RUN_DIR [RUN_DIR ...]
    python tools/modality_overview.py table --output /home/ikohut/tmp/modality-overview-deployment \\
        --comparisons /home/ikohut/tmp/checkpoint-comparison-windows > docs/modality-overview.md
"""
import argparse
import functools
import gc
import json
import os
from pathlib import Path
import re
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ABBREVIATIONS = {'stats': 'S', 'text': 'T', 'image': 'I'}
SECTIONS = ('STI', 'ST', 'SI', 'TI', 'S', 'T', 'I')
NAMES = {'S': 'statistics', 'T': 'text', 'I': 'image'}


WINDOW_SERIES = re.compile(r'-b64parent512-w\d+-.*-r22$')
WINDOW_SECTION = [
    '## Context window: r22 series, statistics + text + image', '',
    'The same STI model trained with context windows from 512 pages down to a few (`tools/prepare_window_series.py`). '
    'Every window trains on the same sampled data: the sampler draws 64 windows of 512 pages per update (complete '
    'scans tiled without overlap), and each is cut into consecutive pieces of W pages, so page pairs across a cut, '
    'about 1/W of them, boundaries included, are not trained. Fixed 16,000 pretraining and 4,000 fine-tuning '
    'updates without early stopping, input dropout seeded per update, the best checkpoint of each phase selected '
    'on dev. One seed per window. Evaluated as above, with all three signals; the merging overlap scales with W '
    '(W/8, at least one page). The W1 models of r21 (cut the same way, W1 sections above) complete the series.', '']


COMPARED = (('type_weighted_f1', 'Type F1'), ('side_weighted_f1', 'Side F1'), ('boundary_precision', 'Border P'),
            ('boundary_recall', 'Border R'), ('boundary_f1', 'Border F1'))


def comparison_section(directory):
    """Differences of every window against the W512 reference, with 95% bootstrap intervals."""
    rows = []
    for path in directory.glob('*-vs-*.json'):
        result = json.loads(path.read_text())
        a, b = (re.search(r'-w(\d+)-', result[k]['checkpoint']) for k in ('a', 'b'))
        if a and int(a.group(1)) == 512 and '-dinoorig-' not in result['a']['checkpoint'] + result['b']['checkpoint']:
            rows.append((int(b.group(1)), result))
    if not rows:
        return []
    streams, resamples = rows[0][1]['streams'], rows[0][1]['resamples']
    lines = ['### Paired comparison with W512', '',
             f'Difference of each window to W512 in points, with the 95% interval of a paired bootstrap over the '
             f'{streams} dev streams ({resamples:,} resamples, `tools/compare_checkpoints.py`); bold where the '
             'interval excludes zero. The bootstrap covers the choice of dev streams, not the training randomness '
             '(one seed per window).', '',
             '| W | ' + ' | '.join(label for _, label in COMPARED) + ' |', '|---|' + '---:|' * len(COMPARED)]
    for window, result in sorted(rows, key=lambda r: -r[0]):
        cells = []
        for key, _ in COMPARED:
            m = result['metrics'][key]
            lo, hi = (100 * x for x in m['ci95'])
            text = f'{100 * m["difference"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
            cells.append(f'**{text}**' if lo > 0 or hi < 0 else text)
        lines.append(f'| {window} | ' + ' | '.join(cells) + ' |')
    return lines + ['']


BORDER_NAMES = {'pair': 'Pair', 'start': 'Start', 'end': 'End', 'both': 'Both'}
BORDER_SECTION = [
    '## Borders from the page edge heads', '',
    'The same evaluation of document starts, scored by different heads of the same models, on every adjacent '
    'page pair inside a scan (the scan start and end are never scored). **Pair**: the pairwise border head (as '
    'above). **Start**: the page-start head on the later page of each pair, i.e. the start head itself on every '
    'page but the first of the scan (is this page a document start?). **End**: the page-end head on the earlier '
    'page, i.e. the end head itself on every page but the last (is this page a document end?). **Both**: a border '
    'only where both fire (the smaller of the two probabilities at threshold 0.5). AP: average precision of the '
    'scores. The W1 model has no pairwise head; its page edge heads give it borders.', '']


def abbreviation(modalities):
    return ''.join(letter for name, letter in ABBREVIATIONS.items() if name in modalities)


def cache_fingerprints():
    """Hash each catalogue once per process; the key changes whenever the file does."""
    from page_sequence.index import PageIndex
    original = PageIndex.fingerprint

    @functools.lru_cache(maxsize=None)
    def cached(path, size, mtime_ns):
        return original(type('Catalogue', (), {'path': path})())

    def fingerprint(self):
        stat = os.stat(self.path)
        return cached(str(Path(self.path).resolve()), stat.st_size, stat.st_mtime_ns)
    PageIndex.fingerprint = fingerprint


def raise_open_file_limit():
    """Each checkpoint's store reader keeps its shard files open (three per shard), and they add up."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return soft, hard


def evaluate(args):
    import torch
    raise_open_file_limit()
    from page_sequence.evaluate_conditions import evaluate_checkpoint
    cache_fingerprints()
    for run in map(Path, args.runs):
        checkpoint = run / 'finetune_best.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'{checkpoint} is missing; the run has not selected a fine-tuning checkpoint')
        config = json.loads((run / 'config.json').read_text())
        print(run.name, flush=True)
        evaluate_checkpoint(checkpoint, config['dev_manifest'], args.output / run.name, ['none'], args.subsets or ['all'],
                            training_store=args.store, device=args.device, limit_streams=args.limit_streams,
                            inference='deployment', border_sources=tuple(BORDER_NAMES))
        gc.collect()
        torch.cuda.empty_cache()


def model_label(run, report):
    tag = re.search(r'-(r\d+)$', run).group(1)
    window = report['inference']['window_size']
    return f'{tag} {abbreviation(report["checkpoint_modalities"])} W{window}'


def scores(report):
    def triple(metrics, prefix):
        if not metrics or metrics.get(prefix + 'f1') is None:
            return [None] * 3
        return [metrics[prefix + key] for key in ('precision', 'recall', 'f1')]
    starts = report['starts']['all'] if report.get('boundary_enabled', True) else None
    return (triple(report['curated_type'], 'weighted_') + triple(report['placement_db_agreement'], 'weighted_')
            + triple(starts, ''))


def table(args):
    rows = {section: [] for section in SECTIONS}
    windows = []
    supports = set()
    for path in sorted(args.output.glob('*/*/none.json')):
        report = json.loads(path.read_text())
        if report.get('inference_policy') != 'deployment':
            raise ValueError(f'{path} was not evaluated with the deployed merging policy')
        run = path.parents[1].name
        if WINDOW_SERIES.search(run):
            label = model_label(run, report) + (' (original DINOv2 image features)' if '-dinoorig-' in run else '')
            windows.append((report['inference']['window_size'], '-dinoorig-' in run, label, scores(report)))
        else:
            rows[abbreviation(report['modalities'])].append((model_label(run, report), scores(report), run))
        supports.add((report['curated_type']['n'], report['placement_db_agreement']['n'],
                      report['starts']['all'].get('positive') if report.get('boundary_enabled', True) else None,
                      report['streams_evaluated']))
    curated, sides, _, streams = max(supports, key=lambda s: (s[3], s[0]))
    starts = max(s[2] for s in supports if s[2] is not None)
    fmt = lambda x: '–' if x is None else f'{100 * x:.1f}'
    lines = [
        '# Modality overview: all models on every combination of main signals they can handle', '',
        f'Selected checkpoint (`finetune_best.pt`) of each run on the {streams} full dev streams, without page '
        'numbers and without document boundaries; only the scan start and end are given. Each section is one '
        'combination of the main signals (S statistics, T text, I image): the other signals are masked as in '
        'the modality ablations (vectors and presence zero), and every model trained with at least these '
        'signals is listed. A model is named by its experiment, the signals it was trained with and its '
        'context window. Overlapping windows are merged as deployed in MetaKat, scaled to the window: overlap W/8 '
        '(64 at W512), regular tail, centre-weighted cubic logits with the ramp ending at 3/4 of the overlap and '
        '3/16 of the overlap ignored at artificial window edges.', '',
        f'- **Page type**: the {curated:,} curated dev pages. Precision, recall and F1 are support-weighted '
        'averages over the page types, as in the README\'s page type evaluation.',
        f'- **Side**: the {sides:,} dev pages with a DB placement (left/right/single), weighted the same way.',
        f'- **Border**: the {starts:,} true document starts; precision, recall and F1 of the predicted starts '
        '(threshold 0.5).' + (' Models without pairwise boundary prediction show none.' if any(
            values[8] is None for section in rows.values() for _, values, _ in section) else ''), '',
        'All values in %. Generated by `tools/modality_overview.py table` from the reports in '
        f'`{args.output}`.', '']
    for section in SECTIONS:
        if not rows[section]:
            continue
        lines += [f'## {section}: ' + ' + '.join(NAMES[c] for c in section), '',
                  '| Model | Type P | Type R | **Type F1** | Side P | Side R | **Side F1** | Border P | Border R | '
                  '**Border F1** |', '|---|' + '---:|' * 9]
        for label, values, _ in sorted(rows[section], key=lambda r: -(r[1][2] or 0)):
            cells = [fmt(v) if i % 3 != 2 else f'**{fmt(v)}**' for i, v in enumerate(values)]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if windows:
        lines += WINDOW_SECTION + [
            '| Model | Type P | Type R | **Type F1** | Side P | Side R | **Side F1** | Border P | Border R | '
            '**Border F1** |', '|---|' + '---:|' * 9]
        for _, _, label, values in sorted(windows, key=lambda r: (-r[0], r[1])):
            cells = [fmt(v) if i % 3 != 2 else f'**{fmt(v)}**' for i, v in enumerate(values)]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if args.comparisons:
        lines += comparison_section(args.comparisons)
    heads = border_heads(args.output)
    if heads:
        lines += BORDER_SECTION + [
            '| Model | ' + ' | '.join(f'{name} P | {name} R | **{name} F1** | {name} AP'
                                     for name in BORDER_NAMES.values()) + ' |',
            '|---|' + '---:|' * (4 * len(BORDER_NAMES))]
        for _, _, label, values in sorted(heads, key=lambda r: (-r[0], r[1])):
            cells = [fmt(v) if i % 4 != 2 else f'**{fmt(v)}**' for i, v in enumerate(values)]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    print('\n'.join(lines))


def border_heads(output):
    """Border precision, recall and F1 of each source, for every run evaluated with the page edge heads."""
    result = []
    for directory in sorted({p.parent for p in output.glob('*/*/none-border-*.json')}):
        reports = {source: json.loads((directory / (
            'none.json' if source == 'pair' else f'none-border-{source}.json')).read_text()) for source in BORDER_NAMES}
        run, pair = directory.parent.name, reports['pair']
        label = model_label(run, pair) + (' (original DINOv2 image features)' if '-dinoorig-' in run else '')
        values = []
        for source in BORDER_NAMES:
            starts = reports[source]['starts']['all'] if reports[source].get('boundary_enabled', True) else None
            values += ([starts[k] for k in ('precision', 'recall', 'f1', 'average_precision')] if starts
                       else [None] * 4)
        result.append((pair['inference']['window_size'], '-dinoorig-' in run, label, values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('evaluate')
    run.add_argument('runs', nargs='+', help='Run directories with config.json and finetune_best.pt')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--store', help='Native store copy for all checkpoints (identity is checked)')
    run.add_argument('--device', default='cuda')
    run.add_argument('--limit-streams', type=int, help='Only the first streams, for testing')
    run.add_argument('--subset', dest='subsets', action='append',
                     help='A modality subset, e.g. image+text+stats; repeatable (default: all nonempty subsets)')
    show = commands.add_parser('table')
    show.add_argument('--output', type=Path, required=True)
    show.add_argument('--comparisons', type=Path,
                      help='Paired comparisons of the window series against W512 (`compare_checkpoints.py compare` '
                           'outputs named <A>-vs-<B>.json)')
    args = parser.parse_args()
    evaluate(args) if args.command == 'evaluate' else table(args)


if __name__ == '__main__':
    main()
