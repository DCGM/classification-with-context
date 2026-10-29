"""Run prepared public batches sequentially through the local MetaKat CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from metakat_checks.recto_verso import engine_settings


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def images_alto_input(source):
    """Carry scan identity/order and asset geometry, without any supplied labels."""
    fields = ('type', 'id', 'batch_id', 'batch_index', 'imageDim', 'altoDim')
    pages = source['elements']
    if any(page.get('type') != 'page' for page in pages):
        raise ValueError('Prepared input must contain only pages')
    result = {key: source[key] for key in ('batch_id', 'page_to_image_mapping', 'page_to_alto_mapping')}
    result['elements'] = [{key: page[key] for key in fields if key in page} for page in pages]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--metakat-root', type=Path, required=True)
    parser.add_argument('--metakat-python', type=Path, required=True)
    parser.add_argument('--engine-config', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--engine-name', default='chapter')
    parser.add_argument('--engine-version', default='v1.7.0')
    parser.add_argument('--skip-batch', action='append', default=[])
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    batches = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / 'READY.json').is_file())
    unknown = set(args.skip_batch) - {p.name for p in batches}
    if unknown:
        raise ValueError(f'Unknown skipped batches: {sorted(unknown)}')
    status = {'started': now(), 'pid': os.getpid(), 'state': 'running', 'input_mode': 'images_alto_only',
              'engine_config': str(args.engine_config.resolve()), 'batches': {}}
    env = dict(os.environ, PYTHONPATH=str(args.metakat_root.resolve()), PYTHONUNBUFFERED='1')
    status_path = args.work / 'status.json'
    for batch in batches:
        output = batch.with_name(f'{batch.name}.{args.engine_name}-{args.engine_version}.results')
        if batch.name in args.skip_batch:
            if not all((output / name).exists() for name in ('metakat.json', 'metakat.pdf', 'mods-overview.txt')):
                raise ValueError(f'Skipped batch has missing outputs: {batch.name}')
            status['batches'][batch.name] = {'state': 'previously_completed'}
            continue
        output.mkdir(exist_ok=True)
        record_path = output / 'cli-run.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['state'] == 'completed' and all((output / name).exists() for name in
                                                     ('metakat.json', 'metakat.pdf', 'mods-overview.txt')):
                status['batches'][batch.name] = record
                continue
        payload = images_alto_input(json.loads((batch / 'metakat.json').read_text()))
        write_json(output / 'input.images-alto.json', payload)
        command = [str(args.metakat_python), '-u', '-m', 'metakat.process_batch',
                   '--batch-dir', str(batch), '--metakat-json', str(output / 'input.images-alto.json'),
                   '--engine-config', str(args.engine_config), '--engine-name', args.engine_name,
                   '--engine-version', args.engine_version, '--output-metakat-json', str(output / 'metakat.json'),
                   '--output-metakat-pdf', str(output / 'metakat.pdf'), '--pdf-downscale',
                   '--output-mods-dir', str(output / 'mods'),
                   '--output-mods-overview', str(output / 'mods-overview.txt')]
        settings = engine_settings(json.loads((batch / 'manifest.json').read_text()))
        for path, value in settings:
            command += ['--set', path, value]
        record = {'state': 'running', 'started': now(), 'pages': len(payload['elements']), 'command': command,
                  'engine_settings': dict(settings)}
        status['batches'][batch.name] = record
        status['current_batch'] = batch.name
        write_json(record_path, record)
        write_json(status_path, status)
        print(f'{now()} START {batch.name}: {record["pages"]} pages', flush=True)
        try:
            with (output / 'run.log').open('w') as log:
                result = subprocess.run(command, cwd=args.metakat_root, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            record['returncode'] = result.returncode
            record['state'] = 'completed' if result.returncode == 0 else 'failed'
        except Exception as error:
            record.update(state='failed', error=str(error))
        record['finished'] = now()
        write_json(record_path, record)
        write_json(status_path, status)
        print(f'{now()} {record["state"].upper()} {batch.name}', flush=True)
    status.pop('current_batch', None)
    failures = [name for name, record in status['batches'].items() if record['state'] == 'failed']
    status.update(state='failed' if failures else 'completed', finished=now(), failed_batches=failures)
    write_json(status_path, status)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
