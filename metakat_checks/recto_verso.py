"""Which prepared batches are to be run in the recto/verso mode of the MetaKat page-number engine.

Monograph rule 1.1.8 of the NDK numbers the sheets of some books `1r, 1v, 2r, 2v`. Whether a book is described that
way is the commissioner's choice and cannot be read off the scans, so the engine setting
`definition:page_number:core:postprocess:ndk:recto_verso` is chosen per batch from the DB page numbers.

A document is a recto/verso document when at least MINIMUM_PAGES of its DB page numbers are plain or bracketed
sheet labels (`12r`, `[12v]`, `IIr`) and they are at least MINIMUM_SHARE of the numbered pages. A batch is run in
the mode when any of its documents is (the alternation test of the engine is conservative, so the other
documents of such a batch are not changed by it). The result is written to the batch's manifest.json as

    "recto_verso": {"batch": true, "documents": {"<document uuid>": true, ...}, "rule": "..."}

and `engine_settings(manifest)` gives the `--set` pairs for metakat.process_batch.

Usage: python -m metakat_checks.recto_verso --root <set directory> [--root ...]   (adds or refreshes the field)
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

MINIMUM_PAGES = 6
MINIMUM_SHARE = 0.3
SHEET_LABEL = re.compile(r'\[?(\d+|[ivxlcdm]+)[rv]\]?', re.I)
SETTING = 'definition:page_number:core:postprocess:ndk:recto_verso'
RULE = (f'document: at least {MINIMUM_PAGES} DB page numbers like 12r, [12v], IIr and at least '
        f'{MINIMUM_SHARE:.0%} of its numbered pages; batch: any document')


def is_sheet_label(label):
    return bool(SHEET_LABEL.fullmatch((label or '').strip()))


def recto_verso_documents(pages):
    """{document: True/False} from the manifest pages (each with `document` and the DB `page_number`)."""
    labels = defaultdict(list)
    for page in pages:
        if (page.get('page_number') or '').strip():
            labels[page['document']].append(page['page_number'])
    documents = {page['document']: False for page in pages}
    for document, numbers in labels.items():
        sheets = sum(map(is_sheet_label, numbers))
        documents[document] = sheets >= MINIMUM_PAGES and sheets / len(numbers) >= MINIMUM_SHARE
    return documents


def recto_verso_field(pages):
    documents = recto_verso_documents(pages)
    return {'batch': any(documents.values()), 'documents': documents, 'rule': RULE}


def engine_settings(manifest):
    """The `--set PATH VALUE` pairs of a prepared batch, from its manifest."""
    return [(SETTING, 'true')] if (manifest.get('recto_verso') or {}).get('batch') else []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', type=Path, action='append', required=True,
                        help='A directory of prepared batches (evaluation, consistency)')
    args = parser.parse_args()
    for root in args.root:
        flagged = []
        manifests = sorted(root.glob('*/manifest.json'))
        for path in manifests:
            manifest = json.loads(path.read_text())
            manifest['recto_verso'] = recto_verso_field(manifest['pages'])
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
            temporary.replace(path)
            if manifest['recto_verso']['batch']:
                flagged.append(path.parent.name)
        print(f'{root}: {len(flagged)} of {len(manifests)} batches in recto/verso mode: {flagged}')


if __name__ == '__main__':
    main()
