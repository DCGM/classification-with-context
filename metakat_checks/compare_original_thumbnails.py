"""Check copied original JPEGs against the native DINO vectors and reconstructions."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_store.reader import TrainingStore


def image_info(path):
    with Image.open(path) as image:
        return {'size':list(image.size), 'mode':image.mode,
                'quantization':getattr(image,'quantization',None),
                'sampling':getattr(image,'layer',None),
                'pixels':np.asarray(image.convert('RGB'))}


def main():
    from metakat.page_type.engines.core.context.dino import DinoEncoder
    from metakat.page_type.engines.core.context.alto_stats import normalize_dense
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('originals','reconstructed','batches','store','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    samples=json.loads((args.originals/'samples.json').read_text())['samples']
    expected={line.split('  ',1)[1]:line.split('  ',1)[0]
              for line in (args.originals/'SHA256SUMS').read_text().splitlines()}
    manifests={}
    for sample in samples:
        path=args.originals/sample['relative_path']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected[sample['relative_path']]:
                raise ValueError('Copy checksum mismatch: '+str(path))
        if sample['batch'] not in manifests:
            manifests[sample['batch']]=json.loads((args.batches/sample['batch']/'manifest.json').read_text())
    # Determine JPEG quantization-table matches independently of the decoder.
    quality_tables={}
    for quality in range(1,101):
        data=BytesIO()
        Image.new('RGB',(16,16),'white').save(data,format='JPEG',quality=quality)
        data.seek(0)
        with Image.open(data) as image:quality_tables[quality]=image.quantization
    torch.set_num_threads(2)
    store=TrainingStore(args.store)
    native=store.read_rows([p['row_id'] for p in samples])
    encoder=DinoEncoder(args.checkpoint,'cuda',batch_size=32)
    values,present=encoder.encode([str(args.originals/s['relative_path']) for s in samples])
    normalized=normalize_dense(values,present).astype(np.float64)
    reference=store.features(native)['image'].astype(np.float64)
    cosine=(normalized*reference).sum(-1)/(np.linalg.norm(normalized,axis=-1)*np.linalg.norm(reference,axis=-1))
    report={'pages':len(samples),'copied_sha256_verified':True,'samples':[]}
    groups=defaultdict(list)
    for i,sample in enumerate(samples):
        original=args.originals/sample['relative_path']
        manifest=manifests[sample['batch']]
        index=next(i for i,p in enumerate(manifest['pages']) if p['page']==sample['page'])
        created=args.reconstructed/f"{sample['batch']}-{index:04d}.jpg"
        a,b=image_info(original),image_info(created)
        scores={key:value for key,value in sample.items() if key not in ('full_resolution_image',)}
        scores.update(original_size=a['size'],reconstructed_size=b['size'],original_mode=a['mode'],
                      reconstructed_mode=b['mode'],quantization_identical=a['quantization']==b['quantization'],
                      original_sampling=a['sampling'],reconstructed_sampling=b['sampling'],
                      matching_standard_quality=[q for q,t in quality_tables.items() if a['quantization']==t],
                      cosine=float(cosine[i]),
                      raw_vector_max_abs=float(np.max(np.abs(values[i].astype(np.float64)-native['image'][i].astype(np.float64)))),
                      raw_vector_identical=bool(np.array_equal(values[i].astype(np.float16),native['image'][i])))
        if a['pixels'].shape==b['pixels'].shape:
            diff=np.abs(a['pixels'].astype(float)-b['pixels'].astype(float))
            scores.update(pixel_mean_abs=float(diff.mean()),pixel_max_abs=float(diff.max()))
        report['samples'].append(scores)
        groups[sample['page_type']].append(i)
    report['summary']={'mean_cosine':float(cosine.mean()),'min_cosine':float(cosine.min()),
        'raw_vectors_identical':sum(p['raw_vector_identical'] for p in report['samples']),
        'quantization_identical':sum(p['quantization_identical'] for p in report['samples']),
        'dimensions_identical':sum(p['original_size']==p['reconstructed_size'] for p in report['samples']),
        'original_long_edges':dict(Counter(max(p['original_size']) for p in report['samples'])),
        'quality_table_matches':dict(Counter(str(p['matching_standard_quality']) for p in report['samples']))}
    report['by_type']={kind:{'pages':len(ids),'mean_cosine':float(cosine[ids].mean()),
                            'min_cosine':float(cosine[ids].min())} for kind,ids in groups.items()}
    np.savez_compressed(args.output/'original-vectors.npz',image=values,normalized=normalized)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':report['summary'],'by_type':report['by_type']},indent=2))
    store.close()


if __name__=='__main__':
    main()
