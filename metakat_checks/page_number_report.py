"""Markdown section with the page-number engine of MetaKat scored against the DB page numbers.

The reference is the DB page number of each page (the manifest's `page_number`, the cataloguing label), the
prediction is `pageNumber.text` of the MetaKat JSON of a variant, a page counting as right only when the label
is identical (whitespace aside). The pages are split into the situations of the NDK pagination rules, by the shape of
the reference label and its place in the document:

- normal numbers: a plain numeral (`12`), printed or filled in a gap of a running sequence;
- bracketed numbers at the beginning: `[3]`, before the first plain numeral of the document;
- bracketed numbers in the middle: `[3]`, after the first plain numeral of the document;
- bracketed numbers with a letter suffix: `[12a]`, `[1b]`, the inserted leaves and the unnumbered leaves before 1.

Everything else (roman numerals, signatures, recto/verso, double labels) is only in the total.
"""
import json
from pathlib import Path
import re

SITUATIONS = (('all', 'All pages with a DB page number'), ('normal', 'Normal numbers (`12`)'),
              ('bracket_beginning', 'Bracketed numbers at the beginning (`[3]`, before the first plain number)'),
              ('bracket_middle', 'Bracketed numbers in the middle (`[3]`, after the first plain number)'),
              ('bracket_suffix', 'Bracketed numbers with a letter suffix (`[12a]`, `[1b]`)'))
PLAIN, BRACKET, SUFFIX = re.compile(r'\d+'), re.compile(r'\[\d+\]'), re.compile(r'\[\d*[a-z]+\]')


def normalize(text):
    return re.sub(r'\s+', '', text or '')


NUMBER = re.compile(r'\[?(\d+|[ivxlcdm]+)(?![a-z])', re.I)


def base_number(label):
    """The number of a label in any convention: `12`, `[12]`, `12r`, `[12v]` and `[12a]` are all 12.

    Roman numerals compare by their letters, case aside. Codes (FC, G9, A 1r) and labels with two numbers
    have no base."""
    label = normalize(label)
    if not label or ',' in label or re.search(r'.\[|\].', label):
        return None
    match = re.fullmatch(r'\[?(\d+|[ivxlcdm]+)(?:[rv]|[a-z]*)\]?', label, re.I)
    return match[1].lower() if match else None


def situations(labels):
    """The situation of each label of one document, in page order (None for the other kinds of label)."""
    result, plain_seen = [], False
    for label in labels:
        label = normalize(label)
        if PLAIN.fullmatch(label):
            result.append('normal')
            plain_seen = True
        elif BRACKET.fullmatch(label):
            result.append('bracket_middle' if plain_seen else 'bracket_beginning')
        elif SUFFIX.fullmatch(label):
            result.append('bracket_suffix')
        else:
            result.append(None)
    return result


def load_page_numbers(root, variants, ready=None):
    """Rows (set, batch, reference label, situation, {variant: predicted label}) of every page."""
    rows = []
    for set_name in ('evaluation', 'consistency'):
        for batch in sorted(p for p in (root / set_name).iterdir() if p.is_dir() and (p / 'READY.json').is_file()):
            if ready is not None and batch.name not in ready.get(set_name, ()):
                continue
            pages = json.loads((batch / 'manifest.json').read_text())['pages']
            predicted = {}
            for variant, (engine, file_name) in variants.items():
                result = json.loads((batch.with_name(f'{batch.name}.{engine}.results') / file_name).read_text())
                predicted[variant] = {p['id']: (p.get('pageNumber') or {}).get('text') or ''
                                      for p in result['elements'] if p['type'] == 'page'}
            kinds = {}
            for document in {p['document'] for p in pages}:
                members = [p for p in pages if p['document'] == document]
                kinds.update(zip((p['page'] for p in members), situations([p['page_number'] for p in members])))
            for page in pages:
                rows.append({'set': set_name, 'batch': batch.name, 'reference': normalize(page['page_number']),
                             'situation': kinds[page['page']],
                             'predicted': {v: normalize(predicted[v][page['page']]) for v in variants}})
    return rows


def score(rows, variants):
    """right: identical label; convention: the same number in another convention (12 for [12], 12r for [12a], ...);
    missing: no number predicted; wrong: another number."""
    result = {}
    for v in variants:
        right = sum(r['predicted'][v] == r['reference'] for r in rows)
        convention = sum(
            r['predicted'][v] != r['reference'] and bool(r['predicted'][v])
            and base_number(r['predicted'][v]) is not None
            and base_number(r['predicted'][v]) == base_number(r['reference'])
            for r in rows)
        missing = sum(not r['predicted'][v] for r in rows)
        result[v] = {'right': right, 'convention': convention, 'missing': missing,
                     'wrong': len(rows) - right - convention - missing}
    return result


def report(rows, variants):
    result = {}
    for set_name in ('evaluation', 'consistency'):
        labelled = [r for r in rows if r['set'] == set_name and r['reference']]
        unlabelled = [r for r in rows if r['set'] == set_name and not r['reference']]
        entry = {'unlabelled': len(unlabelled),
                 'unlabelled_predicted': {v: sum(bool(r['predicted'][v]) for r in unlabelled) for v in variants},
                 'situations': {}}
        for key, _ in SITUATIONS:
            members = labelled if key == 'all' else [r for r in labelled if r['situation'] == key]
            entry['situations'][key] = {'n': len(members), **score(members, variants)}
        result[set_name] = entry
    return result


def percent(count, total):
    return '-' if not total else f'{100 * count / total:.1f}'


def render_page_number_section(root, variants, ready=None):
    rows = load_page_numbers(root, variants, ready)
    scores = report(rows, variants)
    names = list(variants)
    out = ['# The page-number engine against the DB page numbers', '',
           'The page-number stage of MetaKat, scored against the DB page numbers, the cataloguing labels, with the '
           'DB label as the truth. A page is right only when the predicted label is identical to the DB label '
           '(`12`, `[3]` and `[12a]` are different labels). *Other convention* is a label with the right number written '
           'in another convention that the standard also allows or that the DB uses (`12` for `[12]`, `[12a]` for `12v`, '
           '`12r` for `12`); it is not counted as right, and shows how much of the error is the convention, not the '
           'number. *None* is a page without a predicted number, *wrong* a different number. The numbers come from the same MetaKat runs as the page types. The engine versions '
           'differ in the page-number logic that lives in the code, not in the model, so the **old code** column '
           'is a run with the code from before the page-number changes (the filling of gaps, bracketed and '
           'suffixed labels from the document context) and the **new code** columns are the current code.', '',
           'The pages are split into the situations of the NDK pagination rules by the DB label: *normal numbers* '
           '(a plain numeral, printed or filled into a gap), *bracketed numbers at the beginning* (before the first '
           'plain number of the document), *bracketed numbers in the middle* (after it) and *bracketed numbers with '
           'a letter suffix* (inserted leaves, unnumbered leaves before page 1). Other labels (roman numerals, '
           'signatures, recto/verso) are only in the total. Percentages.', '']
    for set_name, entry in scores.items():
        if not any(r['set'] == set_name for r in rows):
            continue
        held = 'evaluation (dev documents)' if set_name == 'evaluation' else 'consistency (fit documents)'
        out += [f'## {held.capitalize()}', '',
                '| Situation | Pages | ' + ' | '.join(f'{n}: right | other convention | none | wrong' for n in names) + ' |',
                '|---|---:|' + '---:|---:|---:|---:|' * len(names)]
        for key, title in SITUATIONS:
            s = entry['situations'][key]
            cells = [f'{percent(s[v]["right"], s["n"])} | {percent(s[v]["convention"], s["n"])} | '
                     f'{percent(s[v]["missing"], s["n"])} | {percent(s[v]["wrong"], s["n"])}' for v in names]
            out.append(f'| {title} | {s["n"]} | ' + ' | '.join(cells) + ' |')
        out += ['', f'{entry["unlabelled"]} pages have no DB page number and are not scored. Pages where the '
                    'variant predicts a number anyway: ' +
                ', '.join(f'{n} {entry["unlabelled_predicted"][n]}' for n in names) + '.', '']
    return '\n'.join(out) + '\n'
