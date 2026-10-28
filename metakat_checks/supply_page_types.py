"""Create separate annotated-type inputs using the exact training target policy.

Run with the MetaKat environment. Original metakat.json inputs remain prediction
inputs; the new variant fixes known targets externally and leaves ignored ones absent.
"""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID, uuid5

# Support invocation by file path with the separate MetaKat Python environment.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from page_sequence.schema import IGNORE, PAGE_TYPES, Page

OUTPUT_NAME = 'metakat.with-page-types.json'


def training_target(record):
    page=Page(library=record['library'],doc=record['document'],page=record['page'],
              order=record.get('ord',0),db_type=record.get('db_type') or '',
              annotated_type=record.get('annotated_type') or '')
    index,gold,_=page.target()
    if gold and index==IGNORE:
        raise ValueError(f'unsupported curated page type: {page.annotated_type}')
    if index==IGNORE:return None,'missing'
    label=PAGE_TYPES[index]
    # Archive labels and MetaKat's NDK spelling differ; FlyLeaf is an exception
    # to lowercasing the first character.
    text='flyleaf' if label=='FlyLeaf' else label[0].lower()+label[1:]
    return text,'human' if gold else 'db'


def with_page_types(payload,records):
    result=copy.deepcopy(payload)
    by_id={r['page']:r for r in records}
    pages=[p for p in result['elements'] if p['type']=='page']
    if len(by_id)!=len(records) or {p['id'] for p in pages}!=set(by_id) or len(pages)!=len(records):
        raise ValueError('input and reference page identities differ or repeat')
    counts=Counter()
    for page in pages:
        if page.get('pageType') is not None:
            raise ValueError('base prediction input already contains page types')
        text,source=training_target(by_id[page['id']]);counts[source]+=1
        if text is None:continue
        page['pageType']={'text':text,'confidence':1.,'annotated':True,
            'id':str(uuid5(UUID(result['batch_id']),page['id']+'/training-page-type'))}
    return result,dict(counts)


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    parser.add_argument('--metakat-root',type=Path,default=Path('/home/ikohut/Projects/MetaKat/.claude/worktrees/schema-shared-base'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.metakat_root))
    from metakat.schemas.base_objects import MetakatIO,PageType
    from metakat.process_batch import init_io
    # Fail before writing any batch if the two vocabularies no longer agree.
    for label in PAGE_TYPES:
        PageType('flyleaf' if label=='FlyLeaf' else label[0].lower()+label[1:])
    report={'verified_at':datetime.datetime.now().astimezone().isoformat(),
            'file':OUTPUT_NAME,'policy':'curated_over_db_same_as_Page.target',
            'annotation_meaning':'externally fixed value; DB labels are not human verification',
            'sets':{},'batches':[]}
    for set_name in ('consistency','evaluation'):
        batches=sorted(p for p in (args.root/set_name).iterdir() if p.is_dir())
        if len(batches)!=50:raise ValueError(f'{set_name} must have 50 prepared batches')
        totals=Counter()
        for batch in batches:
            if not (batch/'READY.json').exists():raise ValueError(f'incomplete batch: {batch.name}')
            original=(batch/'metakat.json').read_bytes()
            payload=json.loads(original)
            records=json.loads((batch/'manifest.json').read_text())['pages']
            expected,counts=with_page_types(payload,records)
            target=batch/OUTPUT_NAME
            if args.check_only:
                actual=json.loads(target.read_text())
                if actual!=expected:raise ValueError(f'supplied types differ from training targets: {batch.name}')
            else:actual=expected
            model=MetakatIO.model_validate(actual)
            cli_model,_=init_io(str(batch),metakat_data=actual)
            if model.elements!=cli_model.elements:raise ValueError('CLI input loader changed page values')
            if not args.check_only:atomic_json(target,actual)
            if (batch/'metakat.json').read_bytes()!=original:raise ValueError('original input changed')
            totals.update(counts);totals['batches']+=1;totals['pages']+=len(records)
            if counts.get('missing',0)==0:totals['fully_supplied_batches']+=1
            if counts.get('human',0)+counts.get('db',0)==0:totals['batches_without_known_types']+=1
            report['batches'].append({'set':set_name,'batch':batch.name,'counts':counts,
                'base_input_sha256':hashlib.sha256(original).hexdigest(),
                'supplied_input_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        report['sets'][set_name]=dict(totals)
    report['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report_name='supplied-page-types.validation.json' if args.check_only else 'supplied-page-types.json'
    atomic_json(args.root/report_name,report)
    print(json.dumps(report['sets'],indent=2))


if __name__=='__main__':main()
