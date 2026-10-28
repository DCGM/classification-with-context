from copy import deepcopy
from uuid import uuid4

import pytest

from metakat_checks.supply_page_types import training_target,with_page_types
from page_sequence.schema import PAGE_TYPES


def record(**extra):
    return dict(library='mzk',document=str(uuid4()),page=str(uuid4()),ord=0,
                db_type='NormalPage',annotated_type='',**extra)


def test_training_priority_and_ndk_mapping():
    r=record();r['annotated_type']='FrontCover'
    assert training_target(r)==('frontCover','human')
    r['annotated_type']='';assert training_target(r)==('normalPage','db')
    r['db_type']='FlyLeaf';assert training_target(r)==('flyleaf','db')
    for label in PAGE_TYPES:
        r['db_type']=label
        assert training_target(r)[0]==('flyleaf' if label=='FlyLeaf' else label[0].lower()+label[1:])


def test_missing_and_unsupported_targets_follow_training_mask():
    r=record();r['db_type']=''
    assert training_target(r)==(None,'missing')
    r['db_type']='UnsupportedDBLabel'
    assert training_target(r)==(None,'missing')
    r['db_type']='NormalPage';r['annotated_type']='UnsupportedHumanLabel'
    with pytest.raises(ValueError,match='curated'):training_target(r)


def test_variant_is_deterministic_and_only_adds_known_page_types():
    records=[record(),record(),record()]
    records[0]['annotated_type']='TitlePage'
    records[2]['db_type']=''
    payload={'batch_id':str(uuid4()),'elements':[dict(type='page',id=r['page'],batch_index=i,
              pageNumber={'text':'[12]','confidence':1.,'annotated':True}) for i,r in enumerate(records)]}
    original=deepcopy(payload)
    variant,counts=with_page_types(payload,records)
    assert payload==original
    assert counts=={'human':1,'db':1,'missing':1}
    assert variant==with_page_types(payload,records)[0]
    assert variant['elements'][0]['pageType']['text']=='titlePage'
    assert variant['elements'][1]['pageType']['text']=='normalPage'
    assert all(p['pageType']['annotated'] is True for p in variant['elements'][:2])
    assert 'pageType' not in variant['elements'][2]
    for p in variant['elements']:p.pop('pageType',None)
    assert variant==original


def test_identity_mismatches_or_existing_predictions_are_rejected():
    r=record();payload={'batch_id':str(uuid4()),'elements':[{'type':'page','id':r['page']}]}
    with pytest.raises(ValueError):with_page_types(payload,[r,r])
    payload['elements'][0]['pageType']={'text':'blank','confidence':.8}
    with pytest.raises(ValueError,match='already contains'):with_page_types(payload,[r])
