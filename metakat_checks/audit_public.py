"""Recheck prepared batch identities/access/order using read-only DB metadata."""
import argparse
import datetime
import json
from pathlib import Path

import psycopg2
from embedding_probe.fetch_ocr_xml import DEFAULT_DB_URL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    args=parser.parse_args()
    wanted=set();documents=set();batches=[]
    for name in ('consistency','evaluation'):
        for path in sorted((args.root/name).glob('*/manifest.json')):
            b=json.loads(path.read_text());batches.append(b)
            for d in b['documents']:documents.add((d['library'],d['doc']))
            for p in b['pages']:wanted.add((p['library'],p['page']))
    assert len(batches)==100,'preparation is incomplete'
    selected=wanted|documents
    seen={};ordered=0
    with psycopg2.connect(DEFAULT_DB_URL,connect_timeout=10) as db:
        db.set_session(readonly=True)
        with db.cursor() as q:
            q.execute('SET statement_timeout=60000')
            keys=sorted(selected)
            for lo in range(0,len(keys),500):
                chunk=keys[lo:lo+500]
                q.execute('SELECT library,id::text,public FROM meta_records WHERE id=ANY(%s::uuid[])',([u for _,u in chunk],))
                seen.update(((lib,u),public) for lib,u,public in q.fetchall() if (lib,u) in selected)
            assert all(seen.get(key) is True for key in selected),'missing or nonpublic selected record'
            for b in batches:
                evidence=b['ordering_evidence']
                if not evidence:continue
                q.execute('''SELECT id::text FROM meta_records WHERE parent_id=%s::uuid
                    AND COALESCE(NULLIF(parent_library,''),library)=%s AND record_type=%s
                    AND in_library IS DISTINCT FROM false ORDER BY "order",id''',
                    (evidence['parent'],evidence['parent_library'],b['documents'][0]['kind']))
                siblings=[r[0] for r in q.fetchall()]
                positions=[siblings.index(d['doc']) for d in b['documents']]
                assert positions==list(range(positions[0],positions[0]+len(positions))),b['name']
                ordered+=1
    report=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),batches=len(batches),
                public_pages=len(wanted),public_documents=len(documents),verified_related_sequences=ordered)
    (args.root/'public_access_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
