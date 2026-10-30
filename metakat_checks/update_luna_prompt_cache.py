"""Turn on explicit prompt caching of the v1.7 Luna engines, locally and in DocAPI.

Every `vlm` section of the three Luna definitions gets `prompt_cache: explicit`: the
system prompt, the same in every request since the page data moved to the user prompt,
is cached and read back by the following requests. Nothing else changes. Without
--apply the patches are only prepared. Curl consumes the admin key file directly; its
value never enters Python, command arguments, logs or generated artifacts.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from update_engine_descriptions import API, ROOT, request

PROMPT_CACHE = 'explicit'
NAMES = [f'{kind}.openai.gpt-6-luna.medium' for kind in ('monograph', 'periodical', 'chapter')]
WORK = Path('/home/ikohut/tmp/metakat-luna-prompt-cache')


def updated(definition):
    definition = copy.deepcopy(definition)
    sections = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'vlm' and isinstance(child, dict):
                    child['prompt_cache'] = PROMPT_CACHE
                    sections.append(f'{path}.vlm')
                else:
                    visit(child, f'{path}.{key}')

    visit(definition, '')
    if not sections:
        raise ValueError('No vlm section in the definition')
    return definition, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    before = request('/v1/engines?active=true&show_definition=true', WORK / 'before.json')['data']
    changes = []
    for name in NAMES:
        remote = next(x for x in before if x['name'] == name and x['version'] == 'v1.7.0')
        path = ROOT / f'{name}.v1.7.0.engine.json'
        bundle = ROOT / f'{name}.v1.7.0'
        local = json.loads(path.read_text())
        if local['definition'] != remote['definition']:
            raise ValueError(f'Definition mismatch before the update: {name}')
        if json.loads((bundle / 'engine_config.json').read_text()) != local['definition']:
            raise ValueError(f'Bundle config differs from the definition: {name}')
        definition, sections = updated(remote['definition'])
        payload = WORK / f'{name}.patch.json'
        payload.write_text(json.dumps({'definition': definition}, ensure_ascii=False, indent=2) + '\n')
        changes.append({'name': name, 'sections': sections, 'definition': definition})
        if args.apply and definition != remote['definition']:
            request(f'/v1/admin/engines/{name}/v1.7.0', WORK / f'{name}.response.json', patch=payload)
            local['definition'] = definition
            path.write_text(json.dumps(local, ensure_ascii=False, indent=2) + '\n')
            (bundle / 'engine_config.json').write_text(json.dumps(definition, ensure_ascii=False, indent=2) + '\n')
            (bundle / 'engine_config.yaml').write_text(yaml.safe_dump(definition, allow_unicode=True, sort_keys=False))
            check = ROOT / f'{name}.v1.7.0.upload-check.json'
            report = json.loads(check.read_text())
            report['descriptor_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            check.write_text(json.dumps(report, indent=2) + '\n')
        print(('Updated' if args.apply else 'Prepared') + f': {name} {", ".join(sections)}', flush=True)
    if args.apply:
        after = request('/v1/engines?active=true&show_definition=true', WORK / 'after.json')['data']
        for change in changes:
            old = next(x for x in before if x['name'] == change['name'] and x['version'] == 'v1.7.0')
            new = next(x for x in after if x['id'] == old['id'])
            assert new['definition'] == change['definition'], change['name']
            for key in ('id', 'name', 'version', 'description', 'default', 'active', 'files_updated', 'created_date'):
                assert new.get(key) == old.get(key), (change['name'], key)
        reports = [json.loads((ROOT / f'{name}.v1.7.0.upload-check.json').read_text()) for name in NAMES]
        (ROOT / 'luna-v1.7.0-upload-check.json').write_text(json.dumps(reports, indent=2) + '\n')
        (WORK / 'verification.json').write_text(json.dumps({
            'verified_utc': datetime.now(timezone.utc).isoformat(), 'api': API,
            'prompt_cache': PROMPT_CACHE, 'engines': len(changes),
            'definition_only_changes': True, 'local_descriptors_match_api': True}, indent=2) + '\n')
        print(f'Verified {len(changes)} definition updates', flush=True)


if __name__ == '__main__':
    main()
