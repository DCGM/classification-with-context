"""Group cached DINO-vector differences by reference page type, not predictions."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from training_store.reader import TrainingStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','work','store','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = TrainingStore(args.store)
    groups = defaultdict(list)
    records = []
    try:
        for path in sorted((args.work/'fresh').glob('*.features.npz')):
            name = path.name.removesuffix('.features.npz')
            manifest = json.loads((args.root/name/'manifest.json').read_text())
            native = store.features(store.read_rows([p['row_id'] for p in manifest['pages']]))['image'].astype(np.float64)
            with np.load(path) as data:
                full = data['image'].astype(np.float64)
            with np.load(args.work/'exact-thumbnail'/(name+'.image.npz')) as data:
                thumb = data['image'].astype(np.float64)
            def cosine(values):
                norms=np.linalg.norm(values,axis=-1)*np.linalg.norm(native,axis=-1)
                return np.divide((values*native).sum(-1),norms,out=np.full(len(norms),np.nan),where=norms>0)
            a,b=cosine(full),cosine(thumb)
            for i,page in enumerate(manifest['pages']):
                label=page['annotated_type'] or page['db_type'] or 'Unlabelled'
                record={'batch':name,'position':i,'page':page['page'],'page_type':label,
                        'label_source':'human' if page['annotated_type'] else 'DB' if page['db_type'] else 'none',
                        'full_cosine':float(a[i]),'thumbnail_cosine':float(b[i])}
                groups[label].append(record)
                records.append(record)
        summary=[]
        for label,items in groups.items():
            a=np.array([v['full_cosine'] for v in items]);b=np.array([v['thumbnail_cosine'] for v in items])
            summary.append({'page_type':label,'pages':len(items),'full_mean':float(np.nanmean(a)),
                            'thumbnail_mean':float(np.nanmean(b)),'thumbnail_min':float(np.nanmin(b)),
                            'thumbnail_below_099':int((b<.99).sum()),'thumbnail_below_098':int((b<.98).sum())})
        summary.sort(key=lambda r:r['thumbnail_mean'])
        report={'pages':len(records),'label_policy':'human annotation, otherwise DB label; unlabelled kept separate',
                'threshold_note':'0.99/0.98 are descriptive similarity cutoffs, not classification errors',
                'summary':summary,'worst_pages':sorted(records,key=lambda r:r['thumbnail_cosine'])[:20]}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        with (args.output/'pages.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        print(json.dumps(report,indent=2))
    finally:
        store.close()


if __name__=='__main__':
    main()
