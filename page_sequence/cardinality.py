"""Fixed-memory diagnostics; no training RNG or influence on sampling decisions."""
import hashlib
import math
import struct

import numpy as np


class PairCardinality:
    """HyperLogLog: 16 KiB registers, approximately 0.81% relative standard error.

    Stable 64-bit hashes of ordered document IDs; no Python pair history retained.
    """
    precision = 14
    version = 1

    def __init__(self, state=None):
        size = 1 << self.precision
        if state is None:
            self.registers = np.zeros(size, np.uint8)
        else:
            if (state.get('version') != self.version or state.get('precision') != self.precision
                    or len(state['registers']) != size):
                raise ValueError('incompatible pair cardinality state')
            self.registers = np.frombuffer(state['registers'], np.uint8).copy()
            if np.any(self.registers > 65 - self.precision):
                raise ValueError('invalid pair cardinality registers')
        self.counts = np.bincount(self.registers, minlength=66 - self.precision)
        self.weights = np.exp2(-np.arange(len(self.counts), dtype=np.float64))

    def add(self, pair):
        value = int.from_bytes(hashlib.blake2b(struct.pack('<QQ', *pair), digest_size=8,
                                             person=b'page-pairs-v1').digest(), 'little')
        index = value & ((1 << self.precision) - 1)
        remainder = value >> self.precision
        rank = 65 - self.precision - remainder.bit_length()
        if rank > self.registers[index]:
            self.counts[self.registers[index]] -= 1
            self.counts[rank] += 1
            self.registers[index] = rank

    def estimate(self):
        m = len(self.registers)
        estimate = (.7213 / (1 + 1.079 / m)) * m * m / np.dot(self.counts, self.weights)
        empty = int(self.counts[0])
        if empty and estimate <= 2.5 * m:
            estimate = m * math.log(m / empty)
        return round(estimate)

    def state_dict(self):
        return {'version': self.version, 'precision': self.precision, 'registers': self.registers.tobytes()}
