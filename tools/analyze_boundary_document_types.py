"""Describe true/missed boundaries using cached DB hierarchy, never OCR/text."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np


def document_kind(document):
    kind = document['record_type']
    if kind == 'monographunit':
        return {'monographmulti': 'multipart_part', 'convolute': 'bound_with_part'}.get(
            document['parent_type'], 'monograph_unit_other_parent')
    return {'monograph': 'single_monograph', 'periodicalitem': 'periodical_issue'}.get(kind, kind)


def relationship(a, b):
    if a['kind'] == b['kind'] == 'periodical_issue':
        if a['library'] == b['library']:
            if a['volume'] and a['volume'] == b['volume']:
                return 'periodical_same_volume'
            if a['periodical'] and a['periodical'] == b['periodical']:
                return 'periodical_same_title_other_or_unknown_volume'
        return ('periodical_different_titles' if a['periodical'] and b['periodical']
                else 'periodical_relation_unknown')
    if a['kind'] == b['kind'] == 'multipart_part':
        return ('multipart_same_work' if (a['parent_library'], a['parent']) ==
                (b['parent_library'], b['parent']) else 'multipart_different_works')
    if a['kind'] == b['kind'] == 'bound_with_part':
        return ('bound_with_same_collection' if (a['parent_library'], a['parent']) ==
                (b['parent_library'], b['parent']) else 'bound_with_different_collections')
    return ' + '.join(sorted([a['kind'], b['kind']]))


def run(args):
    metadata = json.loads((Path(args.scores) / 'metadata.json').read_text())
    examples, ids = [], set()
    for path in sorted(Path(args.scores).glob('*.npz')):
        with np.load(path) as values:
            mask = values['truth']
            for left, right, unknown, supplied in zip(values['previous_rows'][mask], values['rows'][mask],
                                                       values['unknown'][mask], values['supplied'][mask]):
                examples.append(dict(stream=int(path.stem), left=int(left), right=int(right),
                                     unknown=float(unknown), supplied=float(supplied)))
                ids.update([int(left), int(right)])
    db = sqlite3.connect(Path(args.catalogue).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    hierarchy = sqlite3.connect(Path(args.hierarchy).resolve().as_uri() + '?mode=ro', uri=True)
    hierarchy.row_factory = sqlite3.Row
    rows, documents = {}, {}
    ids = sorted(ids)
    for start in range(0, len(ids), 512):
        chunk = ids[start:start + 512]
        query = ('SELECT p.id AS row_id,d.* FROM pages p JOIN documents d ON d.id=p.document '
                 'WHERE p.id IN (' + ','.join('?' for _ in chunk) + ')')
        for value in db.execute(query, chunk):
            rows[value['row_id']] = value['id']
            documents[value['id']] = dict(value)
    for d in documents.values():
        value = hierarchy.execute('SELECT * FROM records WHERE library=? AND id=?',
                                  (d['library'], d['doc'])).fetchone()
        if value is None:
            raise ValueError('document missing from hierarchy cache')
        d.update(record_type=value['kind'], parent=value['parent'],
                 parent_library=value['parent_library'] or value['library'], parent_type=None)
        if d['parent']:
            parent = hierarchy.execute('SELECT kind FROM records WHERE library=? AND id=?',
                                       (d['parent_library'], d['parent'])).fetchone()
            d['parent_type'] = parent[0] if parent else None
        d['kind'] = document_kind(d)
    thresholds = [.5, .6, .6372565627098083, .65]
    totals, groups = Counter(), {mode: defaultdict(Counter) for mode in ('unknown', 'supplied')}
    pairs = {mode: defaultdict(Counter) for mode in groups}
    for example in examples:
        a, b = (documents[rows[example[k]]] for k in ('left', 'right'))
        assert a['id'] != b['id']
        rel = relationship(a, b)
        pair = ' + '.join(sorted([a['kind'], b['kind']]))
        example.update(left_document=a['id'], right_document=b['id'], relationship=rel, pair=pair)
        totals['boundaries'] += 1
        for mode in groups:
            for counter in (groups[mode][rel], pairs[mode][pair]):
                counter['total'] += 1
                for t in thresholds:
                    counter[str(t)] += int(example[mode] < t)
    assert totals['boundaries'] == metadata['expected_unknown']['positive']
    for mode in groups:
        assert sum(v['0.5'] for v in groups[mode].values()) == metadata['expected_' + mode]['fn']
    report = dict(model=Path(metadata['checkpoint']).parent.name, split='dev', thresholds=thresholds,
                  checkpoint_step=metadata['checkpoint_step'],
                  checkpoint_sha256=metadata['checkpoint_sha256'],
                  probabilities='raw merged pair-boundary scores, before downstream recovery',
                  true_boundaries=len(examples), documents=len(documents),
                  document_types=dict(Counter(d['kind'] for d in documents.values())),
                  pairs=pairs, relationships=groups)
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    (output / 'document-types.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'true-boundaries-by-document-type.csv').open('w', newline='') as f:
        fields = ['stream', 'left', 'right', 'left_document', 'right_document', 'pair',
                  'relationship', 'unknown', 'supplied']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(examples)
    (output / 'document-type-identities.json').write_text(json.dumps(documents, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--catalogue', required=True)
    parser.add_argument('--hierarchy', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
