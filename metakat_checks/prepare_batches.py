"""Prepare bounded public scan batches without running MetaKat or its API.

Metadata comes from the native catalogue and read-only PostgreSQL. Assets come
only from the public Matylda archive. Every selected page must currently be public.
"""
import argparse
import csv
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid5, NAMESPACE_URL
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import psycopg2
from PIL import Image

from embedding_probe.fetch_ocr_xml import DEFAULT_DB_URL
from metakat_checks.recto_verso import recto_verso_field
from page_sequence.pagination import candidate

SHORT_PAGES = 40  # boundaries batches: longest document
BOOK_PAGES = (3, SHORT_PAGES)  # boundaries batches: shortest and longest book
BOUNDARY_DOCS = (10, 40)  # boundaries batches: fewest and most documents
KINDS = ('monograph','monographunit','periodicalitem','sheetmusic','graphic','map','manuscript')
PUBLIC_ROOT = Path('/mnt/matylda0/ihradis/digiknihovna_public')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def rank(value, seed):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{value}'.encode(),digest_size=8).digest(),'big') % 2**63


def consecutive_windows(siblings, valid, max_pages=400):
    """Real adjacent siblings only, two or three complete bottom-level documents."""
    result=[]
    for start in range(len(siblings)-1):
        group=[]
        for sibling in siblings[start:start+3]:
            if sibling not in valid: break
            group.append(valid[sibling])
            if sum(d['length'] for d in group)>max_pages: break
            if len(group)>=2: result.append(list(group))
    return result


class Preparation:
    def __init__(self,args):
        self.args=args; self.output=args.output;self.work=args.work
        self.work.mkdir(parents=True,exist_ok=True);self.output.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.work/'candidates.sqlite',uri=True)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA cache_size=-8192');self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.create_function('sample_rank',2,rank,deterministic=True)
        self.db.execute('ATTACH DATABASE ? AS cat',(args.catalogue.as_uri()+'?mode=ro',))
        self.db.execute('ATTACH DATABASE ? AS hierarchy',(args.hierarchy.as_uri()+'?mode=ro',))
        self.pg=psycopg2.connect(args.db_url,connect_timeout=10);self.pg.set_session(readonly=True)
        with self.pg.cursor() as q:q.execute('SET statement_timeout=60000')
        self.pagination=np.load(args.pagination,mmap_mode='r',allow_pickle=False)
        self.failures=Counter();self.metadata={};self.image_listing=OrderedDict();self.used_docs=set();self.used_pages=set()
        self.plans=json.loads((self.work/'plan.json').read_text()) if (self.work/'plan.json').exists() else []
        for batch in self.plans:
            self.used_docs.update(d['doc'] for d in batch['documents'])
            self.used_pages.update(p['page'] for p in batch['pages'])
        self.status('initializing')

    def status(self,phase,**extra):
        atomic_json(self.work/'status.json',dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),
            phase=phase,batches_planned=len(self.plans),groups=dict(Counter(p['set']+'/'+p['group'] for p in self.plans)),
            rejection_counts=dict(self.failures),**extra))
        print(json.dumps({'phase':phase,'batches':len(self.plans),**extra}),flush=True)

    def build_candidates(self):
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='candidates'").fetchone():
            self.add_annotation_counts()
            return
        self.status('indexing_document_candidates')
        self.db.execute('''CREATE TABLE candidates AS
            SELECT d.id,d.library,d.doc,d.split,d.volume,d.periodical,d.copy_group,d.first,d.length,
                   h.kind,h.parent,COALESCE(NULLIF(h.parent_library,''),h.library) AS parent_library
            FROM cat.documents d JOIN hierarchy.records h ON h.library=d.library AND h.id=d.doc
            JOIN cat.pages p ON p.id=d.first
            WHERE d.split IN ('fit','dev') AND d.length BETWEEN 1 AND ?
              AND p.path LIKE 'public.256/%' AND h.kind IN (?,?,?,?,?,?,?)''',
            (self.args.max_pages,*KINDS))
        self.db.execute('CREATE UNIQUE INDEX candidate_key ON candidates(library,doc)')
        self.db.execute('CREATE INDEX candidate_family ON candidates(kind,split,library,parent)')
        self.db.commit()
        self.add_annotation_counts()
        self.status('candidate_index_ready',candidates=self.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0])

    def add_annotation_counts(self):
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS candidate_id ON candidates(id)')
        self.db.execute('CREATE TABLE IF NOT EXISTS preparation_meta(key TEXT PRIMARY KEY,value TEXT)')
        if self.db.execute("SELECT 1 FROM preparation_meta WHERE key='gold_counts_ready'").fetchone():return
        if 'gold_count' not in {r[1] for r in self.db.execute('PRAGMA table_info(candidates)')}:
            self.db.execute('ALTER TABLE candidates ADD COLUMN gold_count INTEGER NOT NULL DEFAULT 0')
        # Every curated dev label is in the ordinary validation panel. Build a
        # bounded disk-backed set of its rows, without loading CSV dumps or images.
        self.db.execute('CREATE TEMP TABLE panel_rows(row_id INTEGER PRIMARY KEY)')
        with self.args.dev_manifest.open() as stream:
            for line in stream:
                rows=json.loads(line)['rows']
                self.db.executemany('INSERT OR IGNORE INTO panel_rows VALUES (?)',((r,) for r in rows))
        counts=list(self.db.execute("""SELECT p.document,COUNT(*) FROM panel_rows r
            CROSS JOIN cat.pages p ON p.id=r.row_id WHERE p.annotated_type!='' GROUP BY p.document"""))
        self.db.executemany('UPDATE candidates SET gold_count=? WHERE id=?',((n,d) for d,n in counts))
        self.db.execute("INSERT INTO preparation_meta VALUES ('gold_counts_ready','1')")
        self.db.commit()

    def query(self,sql,params):
        with self.pg.cursor() as q:
            q.execute(sql,params);return q.fetchall()

    def reject(self,reason):
        self.failures[reason]+=1
        return None

    def document(self,d):
        key=(d['library'],d['doc'])
        if key in self.metadata:return self.metadata[key]
        result=self._document(dict(d))
        self.metadata[key]=result
        return result

    def image_present(self,path):
        # One directory listing replaces hundreds of remote SSHFS stat calls.
        owner=path.parent
        if owner not in self.image_listing:
            names=set()
            with os.scandir(owner) as entries:
                for entry in entries:
                    names.add(entry.name)
                    if len(names)>200000:return False
            while self.image_listing and (len(self.image_listing)>=4 or
                    sum(map(len,self.image_listing.values()))+len(names)>200000):
                self.image_listing.popitem(last=False)
            self.image_listing[owner]=names
        else:self.image_listing.move_to_end(owner)
        return path.name in self.image_listing[owner]

    def _document(self,d):
        record=self.query('''SELECT public,record_type,"order",number,title
            FROM meta_records WHERE library=%s AND id=%s::uuid''',(d['library'],d['doc']))
        if not record or record[0][0] is not True:return self.reject('document_not_public')
        if record[0][1]!=d['kind']:return self.reject('document_kind_changed')
        native=[dict(r) for r in self.db.execute('''SELECT id AS row_id,page,ord,path,db_type,
            annotated_type,placement FROM cat.pages WHERE document=? ORDER BY id''',(d['id'],))]
        if len(native)!=d['length'] or any(not p['path'].startswith('public.256/') for p in native):
            return self.reject('mixed_or_changed_catalogue')
        if any(b['ord']!=a['ord']+1 for a,b in zip(native,native[1:])):
            return self.reject('catalogue_page_gap')
        records=self.query('''SELECT id::text,public,"order",number,image_path FROM meta_records
            WHERE library=%s AND parent_id=%s::uuid AND record_type='page'
              AND in_library IS DISTINCT FROM false ORDER BY "order",id''',(d['library'],d['doc']))
        if [p[0] for p in records]!=[p['page'] for p in native]:
            return self.reject('incomplete_or_changed_page_sequence')
        if any(p[1] is not True for p in records):return self.reject('page_not_public')
        archives={}
        try:
            for page,source in zip(native,records):
                _,_,order,number,image_path=source
                if not image_path or not image_path.startswith(str(PUBLIC_ROOT)+'/'):
                    return self.reject('source_not_public_archive')
                path=Path(image_path)
                if '..' in path.parts or not self.image_present(path):
                    return self.reject('image_missing')
                raw='' if number in (None,'not_found') else number
                old=self.pagination[page['row_id']]
                if candidate(raw)!=(int(old['value']),int(old['system'])):
                    return self.reject('pagination_changed_since_dump')
                owner=path.parent
                if not owner.name.endswith('.images'):return self.reject('unsupported_source_layout')
                archive=owner.with_name(owner.name[:-7]+'.alto_xml.zip')
                if archive not in archives:
                    if not archive.is_file():return self.reject('alto_archive_missing')
                    # Avoid loading enormous periodical ZIP directories into RAM.
                    if archive.stat().st_size>2*1024**3:return self.reject('alto_archive_over_2gib')
                    z=zipfile.ZipFile(archive);archives[archive]=z
                z=archives[archive]
                member=page['page']+'.xml'
                try:info=z.getinfo(member)
                except KeyError:
                    matches=[i for i in z.infolist() if Path(i.filename).name==member]
                    if len(matches)!=1:return self.reject('alto_member_missing_or_ambiguous')
                    info=matches[0]
                if not 0<info.file_size<=32*1024**2:return self.reject('alto_member_invalid_size')
                page.update(image_source=str(path),alto_archive=str(archive),alto_member=info.filename,
                            page_number=raw,db_order=order)
        except (OSError,zipfile.BadZipFile):
            return self.reject('source_archive_unreadable')
        finally:
            for z in archives.values():z.close()
        d.update(title=record[0][4],number=record[0][3],sibling_order=record[0][2],pages=native)
        return d

    def available(self,documents):
        ids=[d['doc'] for d in documents]
        pages=[p['page'] for d in documents for p in d['pages']]
        return len(set(ids))==len(ids) and not self.used_docs.intersection(ids) and len(set(pages))==len(pages) and not self.used_pages.intersection(pages)

    def add(self,set_name,group,documents,order_evidence=None,suffix=None):
        if not self.available(documents):return False
        index=1+sum(p['set']==set_name and p['group']==group for p in self.plans)
        suffix=suffix or ((documents[0]['library']+'_'+documents[0]['doc'][:8]) if group!='monographs' else 'mixed')
        name=f'{group}_{index:03d}_{suffix}'
        batch={'set':set_name,'group':group,'name':name,'seed':self.args.seed,
               'documents':[{k:v for k,v in d.items() if k!='pages'} for d in documents],
               'pages':[], 'ordering_evidence':order_evidence}
        for doc_index,d in enumerate(documents):
            for i,p in enumerate(d['pages']):
                batch['pages'].append(dict(p,library=d['library'],document=d['doc'],document_index=doc_index,
                    document_start=i==0,document_end=i==len(d['pages'])-1))
        assert len(batch['pages'])<=self.args.max_pages
        self.plans.append(batch);self.used_docs.update(d['doc'] for d in documents)
        self.used_pages.update(p['page'] for p in batch['pages'])
        atomic_json(self.work/'plan.json',self.plans)
        self.status('batch_planned',set=set_name,batch=name,pages=len(batch['pages']),documents=len(documents))
        return True

    def related(self,set_name,kind,group,target):
        if sum(p['set']==set_name and p['group']==group for p in self.plans)>=target:return
        split='dev' if set_name=='evaluation' else 'fit'
        groups=self.db.execute('''SELECT library,parent,parent_library,COUNT(*) n FROM candidates
            WHERE split=? AND kind=? AND parent IS NOT NULL AND parent!=''
            GROUP BY library,parent,parent_library HAVING COUNT(*)>=2
            ORDER BY (SUM(gold_count)>0 AND ?='dev') DESC,sample_rank(library||parent,?)''',(split,kind,split,self.args.seed))
        for g in groups:
            if sum(p['set']==set_name and p['group']==group for p in self.plans)>=target:break
            if group=='issues' and not self.db.execute('SELECT 1 FROM hierarchy.records WHERE library=? AND id=? AND kind=\'periodicalvolume\'',(g['parent_library'],g['parent'])).fetchone():continue
            if group=='multipart' and not self.db.execute('SELECT 1 FROM hierarchy.records WHERE library=? AND id=? AND kind=\'monographmulti\'',(g['parent_library'],g['parent'])).fetchone():continue
            siblings=self.query('''SELECT library,id::text,"order",public FROM meta_records
                WHERE COALESCE(NULLIF(parent_library,''),library)=%s AND parent_id=%s::uuid AND record_type=%s
                  AND in_library IS DISTINCT FROM false ORDER BY "order",id''',(g['parent_library'],g['parent'],kind))
            if any(r[2] is None for r in siblings) or len({r[2] for r in siblings})!=len(siblings):
                self.failures['sibling_order_missing_or_ambiguous']+=1;continue
            valid={}
            for lib,doc,order,public in siblings:
                if public is not True or doc in self.used_docs:continue
                c=self.db.execute('SELECT * FROM candidates WHERE library=? AND doc=? AND split=?',(lib,doc,split)).fetchone()
                if c is not None:valid[doc]=dict(c)
            windows=consecutive_windows([r[1] for r in siblings],valid,self.args.max_pages)
            windows.sort(key=lambda w:(-sum(d['gold_count'] for d in w) if set_name=='evaluation' else 0,rank('|'.join(d['doc'] for d in w),self.args.seed)))
            for window in windows:
                docs=[self.document(d) for d in window]
                if all(d is not None for d in docs) and self.add(set_name,group,docs,{
                    'source':'read-only meta_records sibling order','parent':g['parent'],
                    'parent_library':g['parent_library'],'orders':[d['sibling_order'] for d in docs]}):break

    def random_documents(self,set_name,multi,target):
        split='dev' if set_name=='evaluation' else 'fit';group='monographs' if multi else 'single'
        if sum(p['set']==set_name and p['group']==group for p in self.plans)>=target:return
        kind_clause="AND kind='monograph'" if multi else ''
        rows=self.db.execute(f'''SELECT * FROM candidates WHERE split=? {kind_clause}
            ORDER BY (gold_count>0 AND ?='dev') DESC,sample_rank(library||doc,?)''',(split,split,self.args.seed+(10 if multi else 20)))
        pending=[]
        for row in rows:
            if sum(p['set']==set_name and p['group']==group for p in self.plans)>=target:break
            if row['doc'] in self.used_docs:continue
            if multi and row['length']>self.args.max_pages-20:continue
            d=self.document(row)
            if d is None or not self.available([d]):continue
            if not multi:
                self.add(set_name,group,[d]);continue
            if pending and (sum(x['length'] for x in pending)+d['length']>self.args.max_pages):continue
            if not self.available(pending+[d]):continue
            pending.append(d)
            desired=2+(rank(pending[0]['doc'],self.args.seed)%2 if pending[0]['length']<=self.args.max_pages//3 else 0)
            if len(pending)>=desired:
                if self.add(set_name,group,pending):pending=[]
        if multi and len(pending)>=2:self.add(set_name,group,pending)

    def boundary_batches(self,set_name,per_kind):
        """Batches of many short complete documents, so that their document boundaries are close together.

        Taken from the catalogue in a fixed seeded order, like page_sequence.manifests: documents of a volume
        are grouped by (library, volume) and their order inside it does not matter. Documents with annotated
        pages come first. A document that the public archive cannot supply is skipped, as in plan().
        Kinds: issues (many short issues of one volume), books (unrelated short monographs) and mixed
        (runs of 3 to 5 issues of a volume between unrelated books)."""
        split='dev' if set_name=='evaluation' else 'fit';fewest,most=BOUNDARY_DOCS
        used_copies={d.get('copy_group') for b in self.plans for d in b['documents']}-{None,''}
        count=lambda kind:sum(p['set']==set_name and p['group']=='boundaries' and p['name'].split('_')[2]==kind for p in self.plans)
        short=f"split=? AND length BETWEEN {BOOK_PAGES[0]} AND {SHORT_PAGES}"
        order="ORDER BY (gold_count>0 AND ?='dev') DESC,sample_rank(library||doc,?)"
        def volumes(salt):
            rows=self.db.execute(f"""SELECT library,volume FROM candidates WHERE {short} AND kind='periodicalitem' AND volume!=''
                GROUP BY library,volume HAVING COUNT(*)>=? ORDER BY (SUM(gold_count)>0 AND ?='dev') DESC,sample_rank(library||volume,?)""",
                (split,fewest,split,self.args.seed+salt)).fetchall()
            return (self.db.execute(f"SELECT * FROM candidates WHERE {short} AND kind='periodicalitem' AND library=? AND volume=? ORDER BY first",
                                    (split,r['library'],r['volume'])) for r in rows)
        def books(salt):
            return self.db.execute(f"SELECT * FROM candidates WHERE {short} AND kind='monograph' {order}",(split,split,self.args.seed+salt))
        def fill(rows,pending,max_docs,reserve=0):
            for row in rows:
                if len(pending)>=max_docs or sum(d['length'] for d in pending)>self.args.max_pages-reserve-BOOK_PAGES[0]:break
                if row['doc'] in self.used_docs or row['copy_group'] in used_copies:continue
                if sum(d['length'] for d in pending)+row['length']>self.args.max_pages-reserve:continue
                d=self.document(row)
                if d is not None and self.available(pending+[d]):pending.append(d)
            return pending
        for rows in volumes(30):
            if count('issues')>=per_kind:break
            docs=fill(rows,[],most)
            if len(docs)>=fewest:self.add(set_name,'boundaries',docs,suffix=f"issues_{docs[0]['library']}_{docs[0]['doc'][:8]}")
        pool=books(31)
        while count('books')<per_kind:
            docs=fill(pool,[],most)
            if len(docs)<fewest or not self.add(set_name,'boundaries',docs,suffix='books'):break
        pool,groups=books(32),volumes(33)
        while count('mixed')<per_kind:
            docs=[]
            for rows in groups:
                before=len(docs);fill(rows,docs,before+3+rank(str(before),self.args.seed)%3,reserve=SHORT_PAGES)
                if len(docs)-before<3:del docs[before:];continue
                fill(pool,docs,len(docs)+1+rank(docs[-1]['doc'],self.args.seed)%3)
                if len(docs)>=most or sum(d['length'] for d in docs)>self.args.max_pages-SHORT_PAGES:break
            if len(docs)<fewest or not self.add(set_name,'boundaries',docs,suffix='mixed'):break

    def plan(self):
        self.build_candidates()
        for set_name in ('evaluation','consistency'):
            self.related(set_name,'periodicalitem','issues',10)
            self.related(set_name,'monographunit','multipart',10)
            self.random_documents(set_name,True,10)
            singles=50-sum(p['set']==set_name and p['group']!='single' for p in self.plans)
            self.random_documents(set_name,False,singles)
            count=sum(p['set']==set_name for p in self.plans)
            if count!=50:raise RuntimeError(f'{set_name} has {count}/50 candidates; see rejection report')
        self.status('planning_complete')

    def copy_image(self,source,destination):
        if not destination.exists():
            temporary=destination.with_suffix('.jpg.part')
            digest=hashlib.sha256()
            with Path(source).open('rb') as src,temporary.open('wb') as dst:
                for chunk in iter(lambda:src.read(1024**2),b''):
                    dst.write(chunk);digest.update(chunk)
            temporary.replace(destination)
            checksum=digest.hexdigest()
        else:checksum=file_sha256(destination)
        with Image.open(destination) as decoded:
            dimensions=dict(width=decoded.width,height=decoded.height)
            decoded.verify()
        return dimensions,checksum

    def materialize(self):
        for batch in self.plans:
            destination=self.output/batch['set']/batch['name']
            if (destination/'READY.json').exists():continue
            self.status('copying_batch',set=batch['set'],batch=batch['name'])
            images=destination/'images';alto=destination/'alto';images.mkdir(parents=True,exist_ok=True);alto.mkdir(exist_ok=True)
            batch_id=uuid5(NAMESPACE_URL,'metakat-page-types/'+batch['set']+'/'+batch['name'])
            elements=[];image_map={};alto_map={};pages=[]
            archives={}
            pool=ThreadPoolExecutor(max_workers=self.args.copy_workers)
            futures=[pool.submit(self.copy_image,p['image_source'],
                     images/f'{i+1:06d}_{p["page"]}.jpg') for i,p in enumerate(batch['pages'])]
            try:
                for i,p in enumerate(batch['pages']):
                    stem=f'{i+1:06d}_{p["page"]}';image_name=stem+'.jpg';xml_name=stem+'.xml'
                    image=images/image_name;xml=alto/xml_name
                    dimensions,image_checksum=futures[i].result()
                    if not xml.exists():
                        archive=p['alto_archive']
                        if archive not in archives:
                            for z in archives.values():z.close()
                            archives={archive:zipfile.ZipFile(archive)}
                        raw=archives[archive].read(p['alto_member'])
                        tree=ET.fromstring(raw)
                        if tree.tag.rsplit('}',1)[-1].lower()!='alto':raise ValueError('source member is not ALTO')
                        tmp=xml.with_suffix('.xml.part');tmp.write_bytes(raw);tmp.replace(xml)
                    tree=ET.parse(xml).getroot()
                    page_node=next((node for node in tree.iter() if node.tag.rsplit('}',1)[-1]=='Page'),None)
                    alto_dimensions=(dict(width=float(page_node.get('WIDTH')),height=float(page_node.get('HEIGHT')))
                                     if page_node is not None else None)
                    number_id=uuid5(batch_id,p['page']+'/db-pagination')
                    elements.append({'type':'page','id':p['page'],'batch_id':str(batch_id),'batch_index':i,
                        'imageDim':dimensions,'altoDim':alto_dimensions,'pageNumber':{'text':p['page_number'],'confidence':1.,
                        'id':str(number_id),'annotated':True}})
                    image_map[p['page']]='images/'+image_name;alto_map[p['page']]='alto/'+xml_name
                    if (i+1)%50==0:self.status('copying_batch',set=batch['set'],batch=batch['name'],copied_pages=i+1,total_pages=len(batch['pages']))
                    pages.append(dict(p,batch_index=i,image=image_map[p['page']],alto=alto_map[p['page']],
                        image_sha256=image_checksum,
                        alto_sha256=file_sha256(xml)))
            finally:
                pool.shutdown(wait=True,cancel_futures=True)
                for z in archives.values():z.close()
            atomic_json(destination/'metakat.json',{'batch_id':str(batch_id),'elements':elements,
                'page_to_image_mapping':image_map,'page_to_alto_mapping':alto_map})
            manifest=dict(batch,pages=pages,recto_verso=recto_verso_field(pages))
            atomic_json(destination/'manifest.json',manifest)
            atomic_json(destination/'READY.json',{'pages':len(pages),'documents':len(batch['documents']),
                'group':batch['group'],'set':batch['set'],'public_verified':True,
                'pagination_matches_native':True,'ordered_complete_documents':True})
        summary={'seed':self.args.seed,'max_batch_pages':self.args.max_pages,'sets':{},
                 'rejections':dict(self.failures),'source_catalogue':str(self.args.catalogue)}
        for name in ('consistency','evaluation'):
            batches=[b for b in self.plans if b['set']==name]
            summary['sets'][name]={'batches':len(batches),'groups':dict(Counter(b['group'] for b in batches)),
                 'documents':sum(len(b['documents']) for b in batches),'pages':sum(len(b['pages']) for b in batches),
                 'largest_batch':max(len(b['pages']) for b in batches),
                 'human_labelled_pages':sum(bool(p['annotated_type']) for b in batches for p in b['pages'])}
        summary['preparation_code_sha256']=file_sha256(Path(__file__))
        for name in ('consistency','evaluation'):
            with (self.output/name/'index.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['batch','group','documents','pages','human_labelled_pages','libraries','document_types','first_title'])
                for b in sorted((b for b in self.plans if b['set']==name),key=lambda b:b['name']):
                    writer.writerow([b['name'],b['group'],len(b['documents']),len(b['pages']),
                        sum(bool(p['annotated_type']) for p in b['pages']),
                        ';'.join(sorted({d.get('library','') for d in b['documents']})),
                        ';'.join(d.get('kind','') for d in b['documents']),b['documents'][0].get('title','')])
        atomic_json(self.output/'summary.json',summary);self.status('complete',summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue',type=Path,default=Path('/home/ikohut/tmp/document-store-fp16-v1/catalogue.sqlite'))
    parser.add_argument('--hierarchy',type=Path,default=Path('/home/ikohut/tmp/page-sequence-data/hierarchy.sqlite'))
    parser.add_argument('--dev-manifest',type=Path,default=Path('/home/ikohut/tmp/page-sequence-data/dev.jsonl'))
    parser.add_argument('--pagination',type=Path,default=Path('/home/ikohut/tmp/pagination-experiment-w128/pagination.npy'))
    parser.add_argument('--output',type=Path,default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    parser.add_argument('--work',type=Path,default=Path('/home/ikohut/tmp/metakat-page-type-batches'))
    parser.add_argument('--db-url',default=os.environ.get('DATABASE_URL',DEFAULT_DB_URL))
    parser.add_argument('--max-pages',type=int,default=400);parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--copy-workers',type=int,default=4)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--boundaries-per-kind',type=int,default=0,
                        help='only add this many boundaries batches of each kind (issues, books, mixed) to the evaluation set')
    args=parser.parse_args()
    if not 1<=args.copy_workers<=8:parser.error('copy-workers must be between 1 and 8')
    if args.max_pages<2:parser.error('max-pages must allow at least two pages')
    job=Preparation(args)
    try:
        if args.boundaries_per_kind:
            job.build_candidates();job.boundary_batches('evaluation',args.boundaries_per_kind)
        else:job.plan()
        if not args.plan_only:job.materialize()
    except Exception as error:
        job.status('failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:job.pg.close();job.db.close()


if __name__=='__main__':main()
