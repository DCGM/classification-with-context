"""Markdown section with the full evaluation of the context system on the dev streams, by optional input.

Reads the per-condition reports of page_sequence.evaluate_conditions (none, page_numbers, boundaries, both).
"""
import json
from pathlib import Path

CONDITIONS = (('none', 'No extra input'), ('page_numbers', 'Page numbers'), ('boundaries', 'Document boundaries'),
              ('both', 'Page numbers and boundaries'))
STRATA = {'same_volume': 'same volume (adjacent issues or parts)',
          'cross_volume': 'different volumes of one periodical', 'unrestricted': 'unrelated documents'}
TOP_CONFUSIONS = 12


def percent(value):
    return '-' if value is None else f'{100 * value:.1f}'


def class_table(per_class, first_column):
    rows = [f'| {first_column} | Support | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for name, m in per_class.items():
        if m['support'] or m['precision']:
            rows.append(f'| {name} | {m["support"]} | {percent(m["precision"])} | {percent(m["recall"])} | '
                        f'{percent(m["f1"])} |')
    return rows


def confusion_lines(matrix, names):
    """The matrix (rows reference, columns predicted; classes that occur) and the most frequent mistakes."""
    keep = [i for i in range(len(names)) if sum(matrix[i]) or sum(row[i] for row in matrix)]
    header = '| Reference \\ predicted | ' + ' | '.join(str(k + 1) for k in range(len(keep))) + ' |'
    lines = [header, '|---|' + '---:|' * len(keep)]
    for k, i in enumerate(keep):
        cells = [('**' + str(matrix[i][j]) + '**' if i == j and matrix[i][j] else str(matrix[i][j]) if matrix[i][j]
                  else '·') for j in keep]
        lines.append(f'| {k + 1} {names[i]} | ' + ' | '.join(cells) + ' |')
    mistakes = sorted(((matrix[i][j], names[i], names[j]) for i in range(len(names)) for j in range(len(names))
                       if i != j and matrix[i][j]), reverse=True)[:TOP_CONFUSIONS]
    return lines, mistakes


def render_context_section(directory: Path):
    reports = {name: json.loads((directory / f'{name}.json').read_text()) for name, _ in CONDITIONS}
    base = reports['both']
    streams, pages = len(base['streams']), sum(s['pages'] for s in base['streams'])
    out = ['# The context system on its own: the labeled dev set', '',
           'The page-type model that MetaKat now runs (the exported W512 fine-tuning checkpoint, step '
           f'{base["checkpoint_step"]}, `finetune_best.pt`, SHA-256 `{base["checkpoint_sha256"][:16]}…`), '
           'evaluated by the training pipeline on its native features: the DINOv2, mmBERT and page-statistics '
           f'vectors of the training store, not the MetaKat extraction. {streams} dev streams ({pages:,} pages) '
           'are each scanned whole with 512-page windows, stride 256 and centre-weighted merging, as in '
           'deployment; the scan endpoints are inputs in every case. The deployment export gives the same '
           'decisions as this checkpoint (see `metakat-boundary5-inference-parity.md`).', '',
           '- **Page type:** the curated (annotated) pages only, so the reference is a human label.',
           '- **Side:** the DB placement (`left`, `right`, `single`), on the pages that have one.',
           '- **Document starts:** the true document boundaries of the streams, every page after the first of '
           'a stream; the decision threshold is 0.5. AP is average precision over the start probability.',
           '- **Optional inputs:** the *page numbers* (pagination features; absent means all zero) and the '
           '*document boundaries* (the true starts and ends of every document, given as input). The model was '
           'trained to work without either, with the boundaries supplied for half of its training scans.',
           '- The dev split also chose this checkpoint (curated page-type F1 without extra input), so it is '
           'slightly optimistic. The test split has no pagination, so it cannot be used for all four cases.',
           '- Percentages; confusion matrices have the reference in the rows and the prediction in the columns, '
           'with the correct cells in bold.', '']
    out += ['## Summary', '',
            '| Input | Page type: pages | accuracy | weighted F1 | macro F1 | Side: pages | accuracy | '
            'weighted F1 | Starts: precision | recall | F1 | AP |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, title in CONDITIONS:
        r = reports[name]
        t, s, b = r['curated_type'], r['placement_db_agreement'], r['starts']['all']
        out.append(f'| {title} | {t["n"]} | {percent(t["accuracy"])} | {percent(t["weighted_f1"])} | '
                   f'{percent(t["macro_f1"])} | {s["n"]} | {percent(s["accuracy"])} | {percent(s["weighted_f1"])} '
                   f'| {percent(b["precision"])} | {percent(b["recall"])} | {percent(b["f1"])} | '
                   f'{percent(b["average_precision"])} |')
    out += ['', f'Documents recovered exactly (start and end both right), of {base["exact_spans"]["true"]}: ' +
            ', '.join(f'{title.lower()} {percent(reports[name]["exact_spans"]["recall"])}%' for name, title in CONDITIONS) + '.', '']
    for name, title in CONDITIONS:
        r = reports[name]
        out += [f'## {title}', '',
                f'Page numbers {"supplied" if r["page_numbers"] else "absent"}, document boundaries '
                f'{"supplied" if r["document_boundaries"] else "absent"}.', '']
        t = r['curated_type']
        out += ['### Page type', '',
                f'{t["n"]} curated pages: accuracy {percent(t["accuracy"])}, weighted F1 {percent(t["weighted_f1"])}, '
                f'macro F1 {percent(t["macro_f1"])}.', ''] + class_table(t['per_class'], 'Page type')
        names = list(t['per_class'])
        matrix, mistakes = confusion_lines(t['confusion'], names)
        out += ['', 'Confusion matrix:', ''] + matrix + ['', 'Most frequent mistakes (reference → predicted): ' +
                ', '.join(f'{a} → {b} ({n})' for n, a, b in mistakes), '']
        s = r['placement_db_agreement']
        out += ['### Side', '', f'{s["n"]} pages with a DB placement: accuracy {percent(s["accuracy"])}, weighted '
                f'F1 {percent(s["weighted_f1"])}.', ''] + class_table(s['per_class'], 'Side')
        side_names = list(s['per_class'])
        matrix, mistakes = confusion_lines(s['confusion'], side_names)
        out += ['', 'Confusion matrix:', ''] + matrix + ['', 'Most frequent mistakes (reference → predicted): ' +
                ', '.join(f'{a} → {b} ({n})' for n, a, b in mistakes), '']
        b = r['starts']['all']
        negatives = b['n'] - b['positive']
        out += ['### Document starts', '',
                f'{b["n"]:,} page transitions, {b["positive"]} of them true document starts. Precision '
                f'{percent(b["precision"])}, recall {percent(b["recall"])}, F1 {percent(b["f1"])}, AP '
                f'{percent(b["average_precision"])}, {b["false_splits_per_1000_negatives"]:.2f} false splits per '
                f'1,000 non-starts. Documents recovered exactly: {r["exact_spans"]["tp"]} of '
                f'{r["exact_spans"]["true"]} ({percent(r["exact_spans"]["recall"])}%), with '
                f'{r["exact_spans"]["predicted"]} predicted.', '',
                'Confusion matrix:', '',
                '| Reference \\ predicted | not a start | start |', '|---|---:|---:|',
                f'| not a start | **{negatives - b["fp"]}** | {b["fp"]} |',
                f'| start | {b["fn"]} | **{b["tp"]}** |', '',
                'Recall of the true starts by what the neighbouring documents are:', '',
                '| Boundary between | True starts | Found | Recall |', '|---|---:|---:|---:|']
        for stratum, m in r['join_detection'].items():
            out.append(f'| {STRATA.get(stratum, stratum)} | {m["n"]} | {m["tp"]} | {percent(m["recall"])} |')
        out.append('')
    return '\n'.join(out) + '\n'
