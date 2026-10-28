from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4
import json
import zipfile

from PIL import Image

from metakat_checks.prepare_batches import Preparation, consecutive_windows,rank


def test_related_windows_never_bridge_missing_siblings_and_respect_budget():
    docs={k:dict(doc=k,length=n) for k,n in [('a',200),('b',210),('c',90)]}
    assert consecutive_windows(['a','missing','b'],docs)==[]
    windows=consecutive_windows(['a','b','c'],docs,400)
    assert [[d['doc'] for d in w] for w in windows]==[['b','c']]
    assert all(sum(d['length'] for d in w)<=400 for w in windows)
    assert rank('parent',17)==rank('parent',17) and rank('parent',17)!=rank('parent',18)


def test_add_rejects_page_copies_and_keeps_reference_labels_out_of_inputs(tmp_path):
    p=object.__new__(Preparation);p.args=SimpleNamespace(seed=17,max_pages=400)
    p.plans=[];p.used_docs=set();p.used_pages=set();p.work=tmp_path;p.status=Mock()
    a={'doc':'a','library':'lib','length':1,'kind':'monograph','pages':[{'page':'shared'}]}
    b={'doc':'b','library':'lib','length':1,'kind':'monograph','pages':[{'page':'shared'}]}
    assert p.add('consistency','single',[a])
    assert not p.add('evaluation','single',[b])
    assert p.plans[0]['pages'][0]['document_start'] and p.plans[0]['pages'][0]['document_end']


def test_materialized_batch_has_ordered_assets_fixed_pagination_and_no_label_leak(tmp_path):
    source=tmp_path/'source';source.mkdir()
    image=source/'page.jpg';Image.new('RGB',(100,200),'white').save(image)
    archive=source/'alto.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('page.xml','<alto><Layout><Page WIDTH="100" HEIGHT="200"/></Layout></alto>')
    page_id=str(uuid4());doc_id=str(uuid4())
    page={'page':page_id,'library':'public','document':doc_id,'document_index':0,
          'document_start':True,'document_end':True,'image_source':str(image),
          'alto_archive':str(archive),'alto_member':'page.xml','page_number':'',
          'annotated_type':'TitlePage','row_id':1}
    job=object.__new__(Preparation)
    job.args=SimpleNamespace(copy_workers=2,seed=17,max_pages=400,catalogue=Path('catalogue.sqlite'))
    job.output=tmp_path/'out';job.status=Mock();job.failures={}
    job.plans=[{'name':'single_001_public','group':'single','set':'evaluation',
               'documents':[{'doc':doc_id}],'pages':[page]}]
    # Summary expects both sets; use a second independent batch for consistency.
    second=dict(job.plans[0],name='single_001_other',set='consistency')
    job.plans.append(second)
    job.materialize()
    path=job.output/'evaluation/single_001_public'
    payload=json.loads((path/'metakat.json').read_text())
    p=payload['elements'][0]
    assert p['pageNumber']['annotated'] is True and p['pageNumber']['text']==''
    assert 'pageType' not in p and 'startsDocument' not in p
    assert p['batch_index']==0 and p['imageDim']=={'width':100,'height':200}
    assert p['altoDim']=={'width':100.,'height':200.}
    assert (path/payload['page_to_image_mapping'][page_id]).is_file()
    assert json.loads((path/'manifest.json').read_text())['pages'][0]['annotated_type']=='TitlePage'
    assert (path/'READY.json').exists()
