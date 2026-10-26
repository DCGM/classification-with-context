"""Bounded metadata/vector checks before a corpus run; never reads page text/OCR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .features import FeaturePipeline
from .index import PageIndex


def check(config, sample_pages=8192):
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    before = features.source_provenance()
    maximum = index.db.execute('SELECT MAX(id) FROM pages').fetchone()[0]
    rows = random.Random(config['seed']).sample(range(1, maximum + 1), min(sample_pages, maximum))
    counts = Counter()
    for offset in range(0, len(rows), 128):
        ids = rows[offset:offset + 128]
        marks = ','.join('?' for _ in ids)
        # Test data is unnecessary for training preflight; preserve the held-out partition.
        eligible = [r[0] for r in index.db.execute(
            f'SELECT p.id FROM pages p JOIN documents d ON d.id=p.document '
            f"WHERE p.id IN ({marks}) AND d.split IN ('fit','dev')", ids)]
        if not eligible:
            continue
        pages = index.pages(eligible)
        vectors = features(pages)
        for page, presence in zip(pages, vectors['presence']):
            root = page.path.split('/', 1)[0]
            counts['sampled_pages'] += 1
            counts[root + ':sampled_pages'] += 1
            for name, available in zip(('image', 'text', 'stats'), presence):
                counts[root + ':' + name + '_present'] += int(available)
                counts[name + '_missing'] += int(not available)
    if features.source_provenance() != before:
        raise ValueError('vector stores changed during preflight')
    report = {'counts': dict(counts), 'feature_schema': features.state_dict(),
              'vector_sources': before,
              'documents': dict(index.db.execute('SELECT split,COUNT(*) FROM documents GROUP BY split')),
              'pages': dict(index.db.execute('SELECT split,SUM(length) FROM documents GROUP BY split'))}
    report['passed'] = counts['sampled_pages'] > 0 and not any(counts[name + '_missing'] for name in ('image', 'text', 'stats'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-pages', type=int, default=8192)
    args = parser.parse_args()
    if args.sample_pages < 1:
        parser.error('positive sample size required')
    report = check(json.loads(Path(args.config).read_text()), args.sample_pages)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'counts': report['counts']}), flush=True)
    if not report['passed']:
        raise SystemExit('feature availability preflight failed')


if __name__ == '__main__':
    main()
