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
KEY_FILE = '/home/ikohut/Projects/MetaKat.worker/metakat/worker/docapi/.docapi_admin_key'
WORK = Path('/home/ikohut/tmp/metakat-engine-descriptions')

BOOK_EN = ('Use for one standalone book, several unrelated standalone books, or parts of one multipart work. '
           'Keep pages in scan order. Do not mix standalone books with multipart works, combine different '
           'multipart works, or include journals or newspapers.')
BOOK_CS = ('Použijte pro jednu samostatnou knihu, několik vzájemně nesouvisejících samostatných knih nebo '
           'díly jednoho vícedílného díla. Zachovejte pořadí snímků. Nekombinujte samostatné knihy s díly '
           'vícedílného díla, různá vícedílná díla ani knihy s časopisy či novinami.')
PERIODICAL_EN = ('Use for consecutive issues or volumes of a single journal, newspaper or other periodical. '
                 'Keep pages in scan order, with the issues of each volume together. '
                 'Do not mix different periodical titles or include books.')
PERIODICAL_CS = ('Použijte pro po sobě následující čísla nebo ročníky jednoho časopisu, novin či jiného '
                 'periodika. Zachovejte pořadí snímků a čísla každého ročníku ponechte pohromadě. '
                 'Nekombinujte různé tituly periodik ani nepřidávejte knihy.')
SUMMARIES = {
    'monograph': (
        'Identifies page numbers and page types, separates books or parts within the scan, and extracts '
        'bibliographic details such as titles, authors and publication information.',
        'Rozpoznává čísla a typy stran, odděluje knihy či díly v dávce a získává bibliografické údaje, '
        'například název, autory a údaje o vydání.'),
    'periodical': (
        'Identifies page numbers and page types, separates issues and extracts their bibliographic details. '
        'Groups issues into volumes under their periodical title.',
        'Rozpoznává čísla a typy stran, odděluje jednotlivá čísla periodika a získává jejich bibliografické '
        'údaje. Seskupuje čísla do ročníků pod společný titul periodika.'),
    'chapter': (
        'Extracts book details and a table of contents linked to the chapter pages. Also identifies '
        'page numbers, page types and separate books or parts within the scan.',
        'Získává bibliografické údaje o knihách a obsah propojený se stranami kapitol. Zároveň rozpoznává '
        'čísla a typy stran a odděluje jednotlivé knihy či díly v dávce.'),
}


def description(kind, luna):
    en = [SUMMARIES[kind][0], PERIODICAL_EN if kind == 'periodical' else BOOK_EN]
    cs = [SUMMARIES[kind][1], PERIODICAL_CS if kind == 'periodical' else BOOK_CS]
    if kind == 'chapter':
        en.append('Processes the contents of each book or part separately. Not intended for article or '
                  'review contents in journals, or for extracting article details.')
        cs.append('Obsah zpracovává samostatně pro každou knihu či díl. Není určen pro časopisecké obsahy '
                  'článků a recenzí ani pro získávání údajů o jednotlivých článcích.')
    else:
        en.append('Does not extract tables of contents or chapters.')
        cs.append('Nezpracovává obsahy ani kapitoly.')
    if luna:
        en.insert(0, 'IMPORTANT — PRIVATE DATA: Selected page images and recognized text are sent to an '
                  'external AI service via OpenRouter. Requests use Zero Data Retention (ZDR). '
                  'Do not submit private data unless you understand this processing and have decided '
                  'for yourself that ZDR is sufficient for your requirements.')
        cs.insert(0, 'DŮLEŽITÉ — NEVEŘEJNÁ DATA: Vybrané snímky stran a rozpoznaný text se odesílají '
                  'externí službě AI přes OpenRouter. Požadavky používají politiku Zero Data Retention (ZDR). '
                  'Neveřejná data odesílejte pouze tehdy, pokud tomuto zpracování rozumíte a sami jste '
                  'posoudili, že ZDR splňuje vaše požadavky na ochranu dat.')
        en.append('Approximate external processing fees: USD 0.003–0.005 per title-page image' +
                  (' and USD 0.002–0.003 per table-of-contents image.' if kind == 'chapter' else '.') +
                  ' Only pages sent for external processing incur these fees. Actual costs may vary' +
                  (', especially for long or dense tables of contents.' if kind == 'chapter' else '.'))
        cs.append('Orientační poplatky za externí zpracování: 0,003–0,005 USD za snímek titulní strany' +
                  (' a 0,002–0,003 USD za snímek obsahu.' if kind == 'chapter' else '.') +
                  ' Tyto poplatky se vztahují pouze na snímky odeslané k externímu zpracování. Skutečná cena '
                  'se může lišit' + (', zejména u dlouhých či hustě popsaných obsahů.' if kind == 'chapter' else '.'))
    else:
        en.append('Processing does not use an external AI service or incur external AI usage fees.')
        cs.append('Zpracování nevyužívá externí službu AI a není spojeno s poplatky za její použití.')
    return 'CS\n' + '\n\n'.join(cs) + '\n\nEN\n' + '\n\n'.join(en)


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
        changed = remote['description'] != text
        if args.apply and changed:
            request(f'/v1/admin/engines/{name}/v1.7.0', WORK / f'{name}.response.json', patch=payload)
            local['description'] = text
            path.write_text(json.dumps(local, ensure_ascii=False, indent=2) + '\n')
            if luna:
                check = ROOT / f'{name}.v1.7.0.upload-check.json'
                report = json.loads(check.read_text())
                report['descriptor_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                check.write_text(json.dumps(report, indent=2) + '\n')
        print(('Updated' if args.apply and changed else 'Unchanged' if args.apply else 'Prepared') + ': ' + name, flush=True)
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
