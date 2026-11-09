"""Drop the per-update sampler counters from the metrics.jsonl of finished runs.

Training wrote the cumulative sampler counters into every per-update line of `metrics.jsonl` (about 48 KB per
line); they are kept at the validation steps in `validation.jsonl` and in the checkpoints, and training now
writes them only there. For each run directory this removes the `sampler` key from every metrics line, verifies
that every new line equals the old one without it and that the validation records hold the counters, and replaces
the file atomically with its original modification time. Runs written to within the last hour, runs without
validation records and files without the key are left alone.

Example:
    python tools/strip_metrics_sampler.py /home/ikohut/tmp/page-sequence-runs/*
"""
import argparse
import json
import os
from pathlib import Path
import time


def strip(run, quiet_seconds=3600):
    metrics, validation = run / 'metrics.jsonl', run / 'validation.jsonl'
    if not metrics.exists():
        return 'no metrics.jsonl'
    if time.time() - max(p.stat().st_mtime for p in run.iterdir()) < quiet_seconds:
        return 'active (written within the last hour)'
    if not validation.exists():
        return 'no validation.jsonl'
    validated = {}
    for line in validation.read_text().splitlines():
        record = json.loads(line)
        if 'sampler' not in record:
            return 'validation records without sampler counters'
        validated[record['step']] = record['sampler']
    before = metrics.stat()
    temporary = metrics.with_name('metrics.jsonl.stripping')
    stripped = lines = 0
    with metrics.open() as source, temporary.open('w') as target:
        for line in source:
            lines += 1
            try:
                record = json.loads(line)
            except ValueError:
                target.write(line)
                continue
            if 'sampler' not in record:
                target.write(line)
                continue
            counters = record.pop('sampler')
            if record.get('step') in validated and validated[record['step']] != counters:
                temporary.unlink()
                return f'counters at step {record["step"]} differ from the validation record'
            new = json.dumps(record, allow_nan=False) + '\n'
            if json.loads(new) != record:
                temporary.unlink()
                return 'a line does not survive re-serialization'
            target.write(new)
            stripped += 1
        target.flush()
        os.fsync(target.fileno())
    if not stripped:
        temporary.unlink()
        return 'nothing to strip'
    if sum(1 for _ in temporary.open()) != lines:
        temporary.unlink()
        return 'line count changed'
    os.utime(temporary, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(temporary, metrics)
    return f'stripped {stripped} lines: {before.st_size / 1e6:.0f} MB -> {metrics.stat().st_size / 1e6:.0f} MB'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for run in args.runs:
        if run.is_dir():
            print(f'{run.name}: {strip(run)}', flush=True)
