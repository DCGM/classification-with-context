"""How the DB numbers the pages by page type: the evidence for the type policy of the MetaKat NDK pagination.

Samples random FIT documents of the page-sequence index and reads their pages from the library metadata DB (the
tunnel on localhost:5888, read-only). Prints, per DB page type, the shape of the page numbers (none, plain `12`,
bracketed `[12]`, letter `[12a]`, special codes) and where in the document the pages are, then the pages after the
last plain number (does `[N+1]` continue the numbering?) and before the first one (letters or bracketed numbers?).

Usage: python -m metakat_checks.page_number_conventions [--documents 6000] [--seed 1]
"""
import argparse
from collections import Counter, defaultdict
import re
import sqlite3

import psycopg2

from embedding_probe.fetch_ocr_xml import DEFAULT_DB_URL

INDEX = 'file:/home/ikohut/tmp/page-sequence-data/index.sqlite?mode=ro'
PLAIN, BRACKET, LETTER = re.compile(r'\d+'), re.compile(r'\[\d+\]'), re.compile(r'\[\d*[a-z]+\]')


def shape(label):
    label = (label or '').strip()
    return ('none' if not label else 'plain' if PLAIN.fullmatch(label) else 'bracket_number'
            if BRACKET.fullmatch(label) else 'letter' if LETTER.fullmatch(label) else 'special')


def tail_class(label, last, k):
    label = (label or '').strip()
    if not label:
        return 'none'
    if re.fullmatch(rf'\[{last + k}\]', label):
        return 'bracket_continuation'
    if re.fullmatch(rf'{last + k}', label):
        return 'plain_continuation'
    if re.fullmatch(rf'\[{last}[a-z]+\]', label):
        return 'letter_after_last'
    return 'other_' + shape(label)


def show(title, table, minimum=15, top=4):
    print(f'\n{title}')
    for kind, counter in sorted(table.items(), key=lambda item: -sum(item[1].values())):
        total = sum(counter.values())
        if total >= minimum:
            print(f'{str(kind):22}{total:7}  ' + '  '.join(f'{k}={v / total:.0%}' for k, v in counter.most_common(top)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--documents', type=int, default=6000)
    parser.add_argument('--index', default=INDEX)
    args = parser.parse_args()
    index = sqlite3.connect(args.index, uri=True)
    documents = index.execute("SELECT library,doc FROM documents WHERE split='fit' AND length BETWEEN 10 AND 600 "
                              'ORDER BY random() LIMIT ?', (args.documents,)).fetchall()
    cursor = psycopg2.connect(DEFAULT_DB_URL, connect_timeout=5).cursor()
    overall, position = defaultdict(Counter), defaultdict(Counter)
    tail, lead = defaultdict(Counter), defaultdict(Counter)
    for library, doc in documents:
        cursor.execute('SELECT page_type,number FROM meta_records WHERE parent_id=%s AND parent_library=%s '
                       'ORDER BY "order"', (doc, library))
        rows = cursor.fetchall()
        for i, (kind, number) in enumerate(rows):
            overall[kind][shape(number)] += 1
            position[kind]['start' if i < 4 else 'end' if i >= len(rows) - 4 else 'middle'] += 1
        plain = [i for i, (_, number) in enumerate(rows) if PLAIN.fullmatch((number or '').strip())]
        if len(plain) < 4:
            continue
        last, first = plain[-1], plain[0]
        for k, i in enumerate(range(last + 1, min(len(rows), last + 9)), 1):
            tail[rows[i][0]][tail_class(rows[i][1], int(rows[last][1]), k)] += 1
        for i in range(min(first, 8)):
            lead[rows[i][0]][shape(rows[i][1])] += 1
    show('All pages: shape of the page number by DB page type', overall, 12)
    show('Position in the document (first / last four pages, else middle)', position, 100, 3)
    show('Tail: up to eight pages after the last plain number', tail)
    show('Lead: up to eight pages before the first plain number', lead)


if __name__ == '__main__':
    main()
