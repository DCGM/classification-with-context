"""Local CLI batch runner with persistent paid-request accounting and fail-stop.

Run with the MetaKat Python environment. A first batch is run separately for
coverage review; --all continues the remaining selected batches without retries.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types')
SOURCE = Path('/home/ikohut/Projects/MetaKat.worker')
ENGINE = 'chapter.openai.gpt-6-luna.medium'
VERSION = 'v1.7.0'
CONFIG = Path('/mnt/kolosus/data/metakat_worker/src_engines') / f'{ENGINE}.{VERSION}/engine_config.json'
STATE = Path('/home/ikohut/tmp/luna-chapter-v1.7.0')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextlib.contextmanager
def ledger():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'guard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STATE / 'guard.json'
        data = json.loads(path.read_text()) if path.exists() else dict(failures=0, stopped=False, calls=[])
        try:
            yield data
        finally:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(path)


def install_guard():
    from openai import OpenAI
    from metakat.common.vlm import client as vlm

    original_init = vlm.VLMClient.__init__

    def initialize(self, config, client=None):
        if config.max_attempts != 1:
            raise RuntimeError('Paid run requires max_attempts=1')
        if client is None:
            client = OpenAI(base_url=config.api_url, api_key=config.resolve_api_key(),
                            timeout=config.timeout, max_retries=0)
        original_init(self, config, client)

    def request(self, prompts, images=(), json_schema=None, request_id=''):
        with ledger() as data:
            if data['stopped']:
                raise RuntimeError('Persistent paid-request guard is stopped; diagnose before resuming')
            number = len(data['calls'])
            data['calls'].append(dict(started=now(), state='in_flight', request_id=request_id,
                                      model=self.config.model, images=len(images), batch=os.environ['LUNA_BATCH']))
        response = None
        deterministic = False
        try:
            build = self._responses_request if self.config.api == 'responses' else self._completions_request
            payload, send, reply_text = build(prompts, images, json_schema)
            response = send(**payload)
            if self.config.api == 'completions':
                choice = response.choices[0]
                if choice.finish_reason in ('length', 'content_filter') or getattr(choice.message, 'refusal', None):
                    deterministic = True
                    raise RuntimeError(f'Provider stopped response: {choice.finish_reason}, refusal={bool(getattr(choice.message, "refusal", None))}')
            result = vlm.parse_reply(reply_text(response), json_schema)
        except Exception as error:
            status = getattr(error, 'status_code', None)
            deterministic |= status in (400, 401, 402, 403, 404, 413, 422)
            with ledger() as data:
                data['failures'] += 1
                data['stopped'] |= deterministic or data['failures'] >= 2
                data['calls'][number].update(state='failed', finished=now(),
                                             error_type=type(error).__name__, http_status=status,
                                             deterministic=deterministic, usage=usage(response))
            # RuntimeError deliberately propagates through engines that swallow VLMResponseError.
            raise RuntimeError(f'Paid request failed ({type(error).__name__}); inspect run.log and guard.json') from error
        with ledger() as data:
            data['calls'][number].update(state='completed', finished=now(), usage=usage(response))
        out = Path(os.environ['LUNA_OUTPUT']) / 'vlm_replies'
        out.mkdir(exist_ok=True)
        (out / f'{number:04d}.json').write_text(json.dumps(dict(request_id=request_id, images=len(images), reply=result), ensure_ascii=False, indent=2))
        return result

    vlm.VLMClient.__init__ = initialize
    vlm.VLMClient.request_json = request


def usage(response):
    value = getattr(response, 'usage', None)
    return value.model_dump() if value is not None else None


def selected():
    for name in ('evaluation', 'consistency'):
        for batch in sorted((ROOT / name).iterdir()):
            if not (batch / 'READY.json').is_file():
                continue
            manifest = json.loads((batch / 'manifest.json').read_text())
            if any(d['kind'] not in ('monograph', 'monographunit') for d in manifest['documents']):
                continue
            old = batch.with_name(batch.name + '.chapter-v1.7.0.results')
            result = old / 'metakat.json'
            if not result.exists():
                continue
            toc = sum((p.get('pageType') or {}).get('text') == 'tableOfContents'
                      for p in json.loads(result.read_text()).get('elements', []) if p.get('type') == 'page')
            if toc:
                yield batch, old, manifest, toc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--only')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--cli', action='store_true')
    args, cli_args = parser.parse_known_args()
    if args.cli:
        install_guard()
        sys.argv = ['metakat.process_batch', *cli_args]
        runpy.run_module('metakat.process_batch', run_name='__main__')
        return
    batches = list(selected())
    if args.list:
        print(json.dumps([dict(set=b.parent.name, batch=b.name, pages=len(m['pages']), toc=t)
                          for b, _, m, t in batches], indent=2))
        return
    if not args.all and not args.only:
        parser.error('Choose --only for coverage review, or --all after review')
    key_file = Path('/home/ikohut/Projects/MetaKat.worker/metakat/worker/docapi/.openrouter_api_key_naki-tools')
    env = {**os.environ, 'PYTHONPATH': str(SOURCE), 'OPENROUTER_API_KEY': key_file.read_text().strip()}
    for batch, old, manifest, toc in batches:
        if args.only and batch.name != args.only:
            continue
        with ledger() as data:
            if data['stopped'] or data['failures']:
                raise RuntimeError('Previous paid failure requires diagnosis; refusing automatic continuation')
        out = batch.with_name(f'{batch.name}.{ENGINE}-{VERSION}.results')
        status = out / 'cli-run.json'
        if status.exists():
            record = json.loads(status.read_text())
            if record['state'] == 'completed':
                continue
            raise RuntimeError(f'Existing unfinished/failed batch requires diagnosis: {out}')
        out.mkdir(exist_ok=True)
        shutil.copy2(old / 'input.images-alto.json', out / 'input.images-alto.json')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--cli',
                   '--batch-dir', str(batch), '--metakat-json', str(out / 'input.images-alto.json'),
                   '--engine-config', str(CONFIG), '--engine-name', ENGINE, '--engine-version', VERSION,
                   '--output-metakat-json', str(out / 'metakat.json'),
                   '--output-metakat-pdf', str(out / 'metakat.pdf'), '--pdf-downscale',
                   '--output-mods-dir', str(out / 'mods'), '--output-mods-overview', str(out / 'mods-overview.txt')]
        record = dict(state='running', started=now(), pages=len(manifest['pages']), reference_toc_pages=toc,
                      command=command, metakat_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
        status.write_text(json.dumps(record, indent=2))
        print(f'{now()} START {batch.parent.name}/{batch.name}', flush=True)
        with (out / 'run.log').open('w') as log:
            code = subprocess.run(command, cwd=SOURCE, env={**env, 'LUNA_BATCH': str(batch), 'LUNA_OUTPUT': str(out)}, stdout=log, stderr=subprocess.STDOUT).returncode
        record.update(state='completed' if code == 0 else 'failed', returncode=code, finished=now())
        status.write_text(json.dumps(record, indent=2))
        print(f'{now()} {record["state"]} {batch.parent.name}/{batch.name}', flush=True)
        if code:
            raise RuntimeError(f'Batch failed; no automatic retry: {out}')


if __name__ == '__main__':
    main()
