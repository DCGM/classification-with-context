"""Explicit little-endian packed records; identifiers are never model inputs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

VERSION = 1
RECORD = np.dtype([
    ('image', '<f2', (768,)), ('text', '<f2', (768,)), ('stats', '<f2', (147,)),
    ('row', '<u8'), ('document', '<u4'), ('order', '<i4'),
    ('weak', 'i1'), ('gold', 'i1'), ('placement', 'i1'), ('availability', 'u1'),
], align=False)
DOCUMENT = np.dtype([('first', '<u8'), ('length', '<u4'), ('shard', '<u4'), ('offset', '<u8')])
assert RECORD.itemsize == 3386

# V2 keeps raw sparse statistics in a document-ordered companion file.
SPARSE_VERSION = 2
SPARSE_RECORD = np.dtype([
    ('image', '<f2', (768,)), ('text', '<f2', (768,)),
    ('stats_offset', '<u8'), ('stats_bytes', '<u4'),
    *RECORD.descr[3:],
], align=False)
# In-memory reads hold only the requested sparse byte strings, never a corpus array.
SPARSE_READ_RECORD = np.dtype([
    ('image', '<f2', (768,)), ('text', '<f2', (768,)), ('stats', object),
    *RECORD.descr[3:],
])


def record_dtype(manifest):
    version = manifest['version']
    dtype = {VERSION: RECORD, SPARSE_VERSION: SPARSE_RECORD}.get(version)
    if dtype is None or manifest['record_bytes'] != dtype.itemsize:
        raise ValueError('incompatible training store layout')
    return dtype


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
