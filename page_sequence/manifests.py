"""Frozen scan manifests; validation and deterministic sibling/shuffled evaluation recipes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .index import PageIndex


def load_streams(path, index, split, allow_repeated_pages=False):
    docs = {d.id: d for d in index.documents(split)}
    seen, ids = set(), set()
    streams = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if s['split'] != split or s['id'] in ids or not s['rows']:
                raise ValueError('invalid stream split/id/empty rows')
            ids.add(s['id'])
            previous_doc, previous_row = None, None
            for row in s['rows']:
                if not allow_repeated_pages and row in seen:
                    raise ValueError('repeated canonical evaluation page; use a separate stress manifest')
                r = index.db.execute('SELECT document FROM pages WHERE id=?', (row,)).fetchone()
                if r is None or r[0] not in docs:
                    raise ValueError('stream crosses splits or contains an unknown row')
                d = docs[r[0]]
                if previous_doc is not None:
                    if d.id == previous_doc.id and row != previous_row + 1:
                        raise ValueError('noncontiguous/reordered document pages')
                    if d.id != previous_doc.id:
                        if previous_row != previous_doc.first + previous_doc.length - 1 or row != d.first:
                            raise ValueError('internal join of arbitrary document fragments')
                        if d.copy_group == previous_doc.copy_group or index.db.execute(
                            'SELECT 1 FROM pages a JOIN pages b ON a.page=b.page '
                            'WHERE a.document=? AND b.document=? LIMIT 1', (d.id, previous_doc.id)).fetchone():
                            raise ValueError('join between copies')
                previous_doc, previous_row = d, row
                seen.add(row)
            first = docs[index.db.execute('SELECT document FROM pages WHERE id=?', (s['rows'][0],)).fetchone()[0]]
            last = previous_doc
            # Derive censoring, never trust a caller to claim a cropped span is complete.
            s['censored_left'] = s['rows'][0] != first.first
            s['censored_right'] = s['rows'][-1] != last.first + last.length - 1
            streams.append(s)
    if not streams:
        raise ValueError('empty stream manifest')
    return streams


def write_streams(index, split, path, target_pages=2048, seed=17, recipe='siblings'):
    if split not in ('dev', 'test') or recipe not in ('siblings', 'shuffled') or target_pages < 1:
        raise ValueError('invalid evaluation manifest settings')
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for d in index.documents(split):
        groups[(d.library, d.volume) if recipe == 'siblings' and d.volume else ('doc', d.id)].append(d)
    runs = list(groups.values())
    rng.shuffle(runs)
    ordered = []
    for run in runs:
        rng.shuffle(run)
        ordered.extend(run)
    streams, rows, previous = [], [], None
    for d in ordered:
        copies = previous is not None and (previous.copy_group == d.copy_group or index.db.execute(
            'SELECT 1 FROM pages a JOIN pages b ON a.page=b.page WHERE a.document=? AND b.document=? LIMIT 1',
            (previous.id, d.id)).fetchone())
        if rows and (len(rows) >= target_pages or copies):
            streams.append(rows)
            rows = []
        rows.extend(range(d.first, d.first + d.length))
        previous = d
    if rows:
        streams.append(rows)
    with Path(path).open('x') as f:
        for i, rows in enumerate(streams):
            f.write(json.dumps({'id': f'{split}-{recipe}-{seed}-{i}', 'split': split, 'rows': rows,
                                'recipe': recipe, 'seed': seed}) + '\n')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-pages', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--recipe', choices=['siblings', 'shuffled'], default='siblings')
    args = parser.parse_args()
    index = PageIndex(args.index)
    write_streams(index, args.split, args.output, args.target_pages, args.seed, args.recipe)
    print(json.dumps({'streams': len(load_streams(args.output, index, args.split)),
                      'sha256': file_sha256(args.output)}))


if __name__ == '__main__':
    main()
