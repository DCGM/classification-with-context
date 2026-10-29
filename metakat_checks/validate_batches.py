"""Validate prepared scan batches with MetaKat's own schema and CLI input loader.

Run with the MetaKat Python environment. No models, worker or remote API run.
"""
import argparse
import datetime
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    parser.add_argument('--metakat-root',type=Path,default=Path('/home/ikohut/Projects/MetaKat.worker'))
    parser.add_argument('--checksums',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.metakat_root))
    from metakat.schemas.base_objects import MetakatIO, MetakatPage
    from metakat.process_batch import init_io
    seen_docs=set();seen_pages=set();result={'sets':{},'checksums_verified':args.checksums,'verified_at':datetime.datetime.now().astimezone().isoformat()}
    for set_name in ('consistency','evaluation'):
        batches=sorted(p for p in (args.root/set_name).iterdir() if p.is_dir())
        assert len(batches)==50,(set_name,len(batches))
        groups=Counter();n_pages=n_docs=gold=0;largest=0
        for batch in batches:
            ready=json.loads((batch/'READY.json').read_text())
            manifest=json.loads((batch/'manifest.json').read_text())
            payload=json.loads((batch/'metakat.json').read_text())
            io=MetakatIO.model_validate(payload)
            # Exercise the same initial-IO handoff as `python -m metakat.process_batch`.
            cli_io,_=init_io(str(batch),metakat_data=payload)
            assert [p.id for p in cli_io.elements]==[p.id for p in io.elements]
            pages=manifest['pages'];docs=manifest['documents'];group=manifest['group']
            assert group in ('issues','multipart','monographs','single')
            assert batch.name.startswith(group+'_')
            assert ready['public_verified'] and ready['pagination_matches_native']
            assert 1<=len(pages)<=400 and len(pages)==len(io.elements)==ready['pages']
            if group=='single':assert len(docs)==1
            assert group=='single' or 2<=len(docs)<=3
            if group=='issues':assert all(d['kind']=='periodicalitem' for d in docs)
            if group=='multipart':assert all(d['kind']=='monographunit' for d in docs)
            if group=='monographs':assert all(d['kind']=='monograph' for d in docs)
            assert len(list((batch/'images').glob('*.jpg')))==len(pages)
            assert len(list((batch/'alto').glob('*.xml')))==len(pages)
            assert all(d['split']==('dev' if set_name=='evaluation' else 'fit') for d in docs)
            for d in docs:
                assert d['doc'] not in seen_docs,('reused document',d['doc'])
                seen_docs.add(d['doc'])
                chunk=[p for p in pages if p['document']==d['doc']]
                assert len(chunk)==d['length']
                assert [p['row_id'] for p in chunk]==list(range(d['first'],d['first']+d['length']))
            for i,(record,page) in enumerate(zip(pages,io.elements)):
                assert isinstance(page,MetakatPage) and str(page.id)==record['page']
                assert record['page'] not in seen_pages,('reused page',record['page'])
                seen_pages.add(record['page'])
                assert page.batch_index==i and record['batch_index']==i
                assert page.pageType is None and page.side is None and page.startsDocument is None
                assert page.pageNumber.annotated is True and page.pageNumber.text==record['page_number']
                assert record['document_start']==(i==0 or record['document']!=pages[i-1]['document'])
                assert record['document_end']==(i+1==len(pages) or record['document']!=pages[i+1]['document'])
                image_path=Path(io.page_to_image_mapping[page.id]);alto_path=Path(io.page_to_alto_mapping[page.id])
                assert not image_path.is_absolute() and '..' not in image_path.parts
                assert not alto_path.is_absolute() and '..' not in alto_path.parts
                assert image_path.stem==alto_path.stem==f'{i+1:06d}_{page.id}'
                with Image.open(batch/image_path) as im:
                    assert im.size==(page.imageDim.width,page.imageDim.height)
                    im.verify()
                tree=ET.parse(batch/alto_path).getroot()
                assert tree.tag.rsplit('}',1)[-1].lower()=='alto'
                if args.checksums:
                    assert checksum(batch/image_path)==record['image_sha256']
                    assert checksum(batch/alto_path)==record['alto_sha256']
                gold+=bool(record['annotated_type'])
            groups[group]+=1;n_pages+=len(pages);n_docs+=len(docs);largest=max(largest,len(pages))
            print(json.dumps({'validated':set_name+'/'+batch.name,'pages':len(pages)}),flush=True)
        result['sets'][set_name]=dict(batches=len(batches),groups=dict(groups),documents=n_docs,
                                     pages=n_pages,human_labelled_pages=gold,largest_batch=largest)
    (args.root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
