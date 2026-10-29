"""Which prepared batches have finished results, so reports can be made while a run is still going."""
import json
from pathlib import Path


def result_ready(folder: Path, name: str = 'metakat.json') -> bool:
    """The results file exists and, where the folder records its run (cli-run.json), the run completed.

    metakat.json is written before the PDF and the MODS, and not atomically: a run in progress is not ready."""
    if not (folder / name).is_file():
        return False
    record = folder / 'cli-run.json'
    return not record.is_file() or json.loads(record.read_text()).get('state') == 'completed'


def prepared_batches(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / 'READY.json').is_file())


def ready_batches(root: Path, variants):
    """(ready, all) batch names; variants: [(engine, results file), ...] that all have to be ready."""
    everything = prepared_batches(root)
    ready = [b for b in everything
             if all(result_ready(b.with_name(f'{b.name}.{engine}.results'), name) for engine, name in variants)]
    return [b.name for b in ready], [b.name for b in everything]
