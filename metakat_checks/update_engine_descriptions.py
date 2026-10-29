"""Describe active v1.7 engines in English/Czech, locally and in DocAPI.

Only descriptions are patched. Curl consumes the admin key file directly;
its value never enters Python, command arguments, logs or generated artifacts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/kolosus/data/metakat_worker/src_engines')
API = 'https://metakat.smart.lib.cas.cz'
KEY_FILE = '/home/ikohut/Projects/MetaKat/metakat/worker/docapi/.docapi_admin_key'
WORK = Path('/home/ikohut/tmp/metakat-engine-descriptions')

BOOK_EN = ('Intended scan batches: one standalone monograph; several unrelated standalone monographs; '
           'or one or more parts of a single multipart monograph. Keep pages in scan order. '
           'Do not use for periodicals, two multipart works, or a mixture of multipart and standalone books.')
BOOK_CS = ('Určené skenovací dávky: jedna samostatná monografie; několik vzájemně nesouvisejících samostatných '
           'monografií; nebo jeden či více dílů jedné vícedílné monografie. Zachovejte pořadí snímků. '
           'Nepoužívejte pro periodika, dvě vícedílná díla ani směs dílů vícedílného díla a samostatných knih.')
PERIODICAL_EN = ('Intended scan batches: one or more consecutive issues/volumes of ONE periodical title, '
                 'in scan order, with the issues of each volume together. Do not mix different periodical '
                 'titles or include monographs. Creates issue, volume and periodical-title hierarchy.')
PERIODICAL_CS = ('Určené skenovací dávky: jedno či více po sobě následujících čísel nebo ročníků JEDNOHO '
                 'titulu periodika v pořadí snímků, s čísly každého ročníku pohromadě. Nekombinujte různé '
                 'tituly periodik ani nepřidávejte monografie. Vytváří hierarchii čísel, ročníků a titulu periodika.')
COMMON_EN = ('MetaKat v1.7.0: page-number detection, contextual page type, left/right/single placement and '
             'document-boundary inference (512-page window, 64-page overlap), followed by bibliographic metadata extraction.')
COMMON_CS = ('MetaKat v1.7.0: rozpoznávání čísel stran, kontextová klasifikace typů stran, určení levé/pravé/samostatné '
             'strany a hranic dokumentů (okno 512 stran, překryv 64 stran), následované extrakcí bibliografických metadat.')


def description(kind, luna):
    en = [COMMON_EN, PERIODICAL_EN if kind == 'periodical' else BOOK_EN]
    cs = [COMMON_CS, PERIODICAL_CS if kind == 'periodical' else BOOK_CS]
    if kind == 'chapter':
        en.append('Additionally reads book tables of contents (TOCs), locates chapter destination titles and '
                  'aligns TOC entries to chapter pages separately within each detected book/part. '
                  'Not intended for article/review TOCs or article metadata extraction.')
        cs.append('Navíc čte knižní obsahy (TOC), vyhledává nadpisy kapitol na cílových stranách a propojuje položky '
                  'obsahu se stranami kapitol samostatně v každé rozpoznané knize či dílu. '
                  'Není určen pro obsahy článků a recenzí ani pro extrakci jejich metadat.')
    if luna:
        en.append('Title-page metadata is read by OpenAI GPT-6 Luna with medium reasoning via OpenRouter, '
                  'using Zero Data Retention (ZDR). ' +
                  ('TOCs use the same VLM; destination-page analysis remains local YOLO with fuzzy chapter alignment.'
                   if kind == 'chapter' else 'This engine does not extract TOC entries or chapters.'))
        cs.append('Metadata titulních stran čte OpenAI GPT-6 Luna s úrovní uvažování medium přes OpenRouter '
                  'v režimu Zero Data Retention (ZDR). ' +
                  ('Stejný VLM čte obsahy; analýza cílových stran zůstává lokální pomocí YOLO s fuzzy přiřazením kapitol.'
                   if kind == 'chapter' else 'Tento engine neextrahuje položky obsahu ani kapitoly.'))
        en.append('Estimated OpenRouter budget in USD: $0.003–$0.005 per title-page image' +
                  ('; $0.002–$0.003 per TOC image.' if kind == 'chapter' else '; TOC extraction is not run.') +
                  ' Estimates are based on the local Luna medium trial (2026-10-29)' +
                  ('; title-page cost for periodicals is extrapolated from book title pages.' if kind == 'periodical' else '.') +
                  ' Multipage spans are requested together. Actual cost varies with image size, OCR, output length, '
                  'reasoning and retries.' + (' Long/dense TOCs may cost more.' if kind == 'chapter' else '') +
                  ' Other scan pages use local models without VLM API charges.')
        cs.append('Orientační rozpočet OpenRouter v USD: 0,003–0,005 USD za snímek titulní strany' +
                  ('; 0,002–0,003 USD za snímek obsahu.' if kind == 'chapter' else '; extrakce obsahu se neprovádí.') +
                  ' Odhady vycházejí z místního testu Luna medium ze dne 29. 10. 2026' +
                  ('; cena titulních stran periodik je odvozena z knižních titulních stran.' if kind == 'periodical' else '.') +
                  ' Vícestránkové celky se zpracovávají v jednom požadavku. Skutečná cena závisí na velikosti obrázků, '
                  'OCR, délce výstupu, uvažování a opakováních.' +
                  (' Dlouhé či husté obsahy mohou stát více.' if kind == 'chapter' else '') + ' '
                  'Ostatní snímky zpracovávají lokální modely bez poplatků za VLM API.')
    else:
        en.append('Bibliographic metadata' + (' and TOC extraction use' if kind == 'chapter' else ' uses') +
                  ' local YOLO/ALTO models; no paid VLM requests. ' +
                  ('Chapter destinations are resolved by fuzzy alignment.' if kind == 'chapter' else
                   'This engine does not extract TOC entries or chapters.'))
        cs.append('Extrakce bibliografických metadat' + (' a obsahu využívá' if kind == 'chapter' else ' využívá') +
                  ' lokální modely YOLO/ALTO bez placených požadavků na VLM. ' +
                  ('Cílové strany kapitol se určují pomocí fuzzy přiřazení.' if kind == 'chapter' else
                   'Tento engine neextrahuje položky obsahu ani kapitoly.'))
    return 'EN\n' + '\n\n'.join(en) + '\n\nCS\n' + '\n\n'.join(cs)


def request(path, output, patch=None):
    command = ['curl', '-4', '--silent', '--show-error', '--fail-with-body', '--retry', '0',
               '--connect-timeout', '30', '--max-time', '60',
               '--variable', f'docapi_key@{KEY_FILE}', '--expand-header', 'X-API-Key: {{docapi_key:trim}}']
    if patch:
        command += ['--request', 'PATCH', '--header', 'Content-Type: application/json', '--data-binary', f'@{patch}']
    command += ['--output', str(output), API + path]
    subprocess.run(command, check=True)
    response = json.loads(output.read_text())
    if response['status'] not in (200, 201):
        raise RuntimeError(f'API response: {response.get("code")}')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    before = request('/v1/engines?active=true&show_definition=true', WORK / 'before.json')['data']
    changes = []
    for remote in before:
        if not remote.get('active') or remote['version'] != 'v1.7.0':
            continue
        name = remote['name']
        kind = name.split('.')[0]
        if kind not in ('monograph', 'periodical', 'chapter'):
            raise ValueError(f'Unknown engine: {name}')
        luna = name != kind
        if luna and name != f'{kind}.openai.gpt-6-luna.medium':
            raise ValueError(f'Unknown VLM variant: {name}')
        path = ROOT / f'{name}.v1.7.0.engine.json'
        local = json.loads(path.read_text())
        if local['definition'] != remote['definition']:
            raise ValueError(f'Definition mismatch before description update: {name}')
        text = description(kind, luna)
        payload = WORK / f'{name}.patch.json'
        payload.write_text(json.dumps({'description': text}, ensure_ascii=False, indent=2) + '\n')
        changes.append({'name': name, 'old_description': remote['description'], 'description': text})
        if args.apply:
            request(f'/v1/admin/engines/{name}/v1.7.0', WORK / f'{name}.response.json', patch=payload)
            local['description'] = text
            path.write_text(json.dumps(local, ensure_ascii=False, indent=2) + '\n')
            if luna:
                check = ROOT / f'{name}.v1.7.0.upload-check.json'
                report = json.loads(check.read_text())
                report['descriptor_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                check.write_text(json.dumps(report, indent=2) + '\n')
        print(('Updated' if args.apply else 'Prepared') + ': ' + name, flush=True)
    (WORK / 'changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
    if args.apply:
        after = request('/v1/engines?active=true&show_definition=true', WORK / 'after.json')['data']
        for change in changes:
            old = next(x for x in before if x['name'] == change['name'] and x['version'] == 'v1.7.0')
            new = next(x for x in after if x['id'] == old['id'])
            assert new['description'] == change['description']
            for key in ('id', 'name', 'version', 'definition', 'default', 'active', 'files_updated', 'created_date'):
                assert new.get(key) == old.get(key), (change['name'], key)
        reports = [json.loads((ROOT / f'{kind}.openai.gpt-6-luna.medium.v1.7.0.upload-check.json').read_text())
                   for kind in ('monograph', 'periodical', 'chapter')]
        (ROOT / 'luna-v1.7.0-upload-check.json').write_text(json.dumps(reports, indent=2) + '\n')
        (WORK / 'verification.json').write_text(json.dumps({'verified_utc': datetime.now(timezone.utc).isoformat(),
            'engines': len(changes), 'description_only_changes': True, 'local_descriptors_match_api': True}, indent=2) + '\n')
        print(f'Verified {len(changes)} description-only updates', flush=True)


if __name__ == '__main__':
    main()
