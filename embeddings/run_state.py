"""Run bookkeeping shared by the embedding inference scripts (text and image).

An output directory holds ``meta.json`` (settings a resumed run must repeat), ``progress.tsv`` (one line per
committed chunk; the last line's key is where a resumed run continues), ``log.txt`` and ``run.lock``.
"""
import fcntl
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path, level: str) -> None:
    log_formatter = logging.Formatter('EMBEDDINGS - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    root_logger = logging.getLogger()
    root_logger.handlers = []
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding='utf-8')):
        handler.setFormatter(log_formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(level)


def check_or_write_meta(meta_path: Path, meta: dict, fixed: Tuple[str, ...]) -> None:
    """Writes meta.json on the first run; on resume refuses settings that would mix embeddings."""
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        mismatched = {k: (existing.get(k), meta[k]) for k in fixed if existing.get(k) != meta[k]}
        if mismatched:
            raise SystemExit(f'{meta_path} does not match this run (existing, requested): {mismatched}. '
                             f'Use a different --output-root/--model-name or delete the output directory.')
        return
    temp_path = meta_path.with_name(meta_path.name + '.tmp')
    temp_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    temp_path.replace(meta_path)


def load_progress(progress_path: Path, header: Tuple[str, ...]) -> Tuple[Optional[bytes], int, int]:
    """Returns (last committed key, total pages done, next chunk index) from the progress file."""
    if not progress_path.exists():
        return None, 0, 0
    last_fields = None
    with progress_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                continue  # a truncated last line left by a killed run; that chunk is redone
            fields = line.rstrip('\n').split('\t')
            if fields[0] == header[0] or len(fields) != len(header):
                continue
            last_fields = fields
    if last_fields is None:
        return None, 0, 0
    row = dict(zip(header, last_fields))
    return row['last_key'].encode('utf-8'), int(row['total_done']), int(row['chunk_index']) + 1


def append_progress(progress_path: Path, values: Tuple, header: Tuple[str, ...]) -> None:
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with progress_path.open('a', encoding='utf-8') as pf:
        if write_header:
            pf.write('\t'.join(header) + '\n')
        pf.write('\t'.join(str(v) for v in values) + '\n')
        pf.flush()
        os.fsync(pf.fileno())


def acquire_run_lock(output_dir: Path):
    """Holds output_dir/run.lock for the life of the process; two runs would corrupt the resume point."""
    lock_file = (output_dir / 'run.lock').open('w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'Another run is already writing to {output_dir} (holds run.lock).')
    return lock_file


class StopRequest:
    """Turns the first SIGINT/SIGTERM into a stop after the chunks in progress; a second one aborts."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, _frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        logger.warning(f'Signal {signum} received, stopping after the chunks already on the GPUs '
                       f'(repeat to abort).')
