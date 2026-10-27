"""Bounded-memory document-length summaries from the training catalogue; no vectors needed."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3


WINDOWS = (32, 64, 128, 256, 512, 1024)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def quantile(histogram, fraction, page_weighted=False):
    """Nearest-rank quantile, optionally sampling a retained page rather than a document."""
    total = sum(n * (length if page_weighted else 1) for length, n in histogram.items())
    if not total:
        return None
    rank = max(1, math.ceil(fraction * total))
    cumulative = 0
    for length, count in sorted(histogram.items()):
        cumulative += count * (length if page_weighted else 1)
        if cumulative >= rank:
            return length


def summarize(histogram):
    documents = sum(histogram.values())
    pages = sum(length * count for length, count in histogram.items())
    if not documents:
        raise ValueError('no retained documents')
    return {
        'documents': documents,
        'pages': pages,
        'mean_pages': pages / documents,
        'min_pages': min(histogram),
        'max_pages': max(histogram),
        'document_quantiles': {f'p{round(q * 100)}': quantile(histogram, q) for q in QUANTILES},
        'page_weighted_document_quantiles': {
            f'p{round(q * 100)}': quantile(histogram, q, True) for q in QUANTILES},
        'windows': {str(w): {
            'documents_fitting': sum(n for length, n in histogram.items() if length <= w),
            'documents_fitting_percent': 100 * sum(n for length, n in histogram.items() if length <= w) / documents,
            'pages_in_fitting_documents': sum(length * n for length, n in histogram.items() if length <= w),
            'pages_in_fitting_documents_percent': 100 * sum(length * n for length, n in histogram.items() if length <= w) / pages,
        } for w in WINDOWS},
        'length_histogram': {str(length): n for length, n in sorted(histogram.items())},
    }


def catalogue_statistics(path):
    path = Path(path).resolve()
    groups = defaultdict(Counter)
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-4096')
        db.execute('PRAGMA mmap_size=0')
        db.execute('PRAGMA temp_store=FILE')
        # Aggregate only the document table; never load page rows or document identities.
        for split, length, count in db.execute(
                'SELECT split,length,COUNT(*) FROM documents GROUP BY split,length'):
            if length is None or length < 1:
                raise ValueError('catalogue contains an empty/invalid document')
            groups[split][length] += count
            groups['all'][length] += count
    return {
        'catalogue': str(path),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'length_definition': 'Retained ordered page records per bottom-level catalogue document, not printed pagination or original index span.',
        'quantile_method': 'Nearest rank; page-weighted quantiles weight each document length by its retained page count.',
        'groups': {name: summarize(histogram) for name, histogram in sorted(groups.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = catalogue_statistics(args.index)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, group in result['groups'].items():
        compact = {key: value for key, value in group.items() if key != 'length_histogram'}
        print(json.dumps({'split': name, **compact}))


if __name__ == '__main__':
    main()
