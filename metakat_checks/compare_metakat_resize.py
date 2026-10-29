"""Compare MetaKat's actual resize utility with copied original thumbnails."""
import argparse
from collections import Counter
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_store.reader import TrainingStore


def main():
    from metakat.page_type.engines.core.context.dino import DinoEncoder
    from metakat.page_type.engines.core.context.alto_stats import normalize_dense
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('originals','resize-script','store','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    samples=json.loads((args.originals/'samples.json').read_text())['samples']
    spec=importlib.util.spec_from_file_location('metakat_resize_utility',args.resize_script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    store=TrainingStore(args.store)
    native=store.features(store.read_rows([s['row_id'] for s in samples]))['image'].astype(np.float64)
    torch.set_num_threads(2)
    encoder=DinoEncoder(args.checkpoint,'cuda',batch_size=32)
    report={'resize_script':str(args.resize_script),'max_size':256,'samples':[],'variants':{}}
    resized=[];originals=[]
    for sample in samples:
        image=cv2.imread(sample['full_resolution_image'])
        if image is None:raise ValueError('Cannot decode source image')
        resized.append(module.resize_image_with_max_size(image,256))
        with Image.open(args.originals/sample['relative_path']) as im:
            originals.append({'size':im.size,'quantization':im.quantization,'pixels':np.asarray(im.convert('RGB'))})
    variants=[('script_default',None,None),('same_resize_quality90',None,90)]
    variants += [(f'thumbnail_q{quality}_then_q90',quality,90) for quality in (85,80,75,70)]
    variants += [(f'source_q{quality}_then_resize_q90',quality,90) for quality in (80,75)]
    for variant,first_quality,final_quality in variants:
        params=[] if final_quality is None else [cv2.IMWRITE_JPEG_QUALITY,final_quality]
        folder=args.output/variant;folder.mkdir(exist_ok=True)
        paths=[];matched_size=matched_tables=matched_pixels=0;deltas=Counter();pixel_errors=[]
        for index,(sample,image,original) in enumerate(zip(samples,resized,originals)):
            path=folder/f'{index:04d}.jpg'
            if first_quality is not None:
                source=(cv2.imread(sample['full_resolution_image']) if variant.startswith('source_') else image)
                success,encoded=cv2.imencode('.jpg',source,[cv2.IMWRITE_JPEG_QUALITY,first_quality])
                if not success:raise IOError('First JPEG encoding failed')
                image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                if variant.startswith('source_'):
                    image=module.resize_image_with_max_size(image,256)
            if not cv2.imwrite(str(path),image,params):raise IOError('JPEG write failed')
            paths.append(str(path))
            with Image.open(path) as im:
                size=im.size;tables=im.quantization;pixels=np.asarray(im.convert('RGB'))
            size_matches=size==original['size'];table_matches=tables==original['quantization']
            pixel_matches=size_matches and np.array_equal(pixels,original['pixels'])
            if size_matches:pixel_errors.append(float(np.abs(pixels.astype(float)-original['pixels'].astype(float)).mean()))
            matched_size+=size_matches;matched_tables+=table_matches;matched_pixels+=pixel_matches
            deltas[str(tuple(a-b for a,b in zip(original['size'],size)))]+=1
            report['samples'].append({'variant':variant,'page':sample['page'],'page_type':sample['page_type'],
                                      'original_size':list(original['size']),'generated_size':list(size),
                                      'quantization_matches':table_matches,'pixels_identical':pixel_matches})
        values,present=encoder.encode(paths);values=normalize_dense(values,present).astype(np.float64)
        cosine=(values*native).sum(-1)/(np.linalg.norm(values,axis=-1)*np.linalg.norm(native,axis=-1))
        by_type={}
        for kind in ('Spine','Edge','NormalPage','TitlePage'):
            mask=np.array([s['page_type']==kind for s in samples])
            if mask.any():by_type[kind]={'pages':int(mask.sum()),'mean_cosine':float(cosine[mask].mean()),'min_cosine':float(cosine[mask].min())}
        data=BytesIO();Image.new('RGB',(16,16)).save(data,format='JPEG',quality=95);data.seek(0)
        with Image.open(data) as im:standard95=im.quantization
        with Image.open(paths[0]) as im:matches95=im.quantization==standard95
        report['variants'][variant]={'pages':len(samples),'matching_dimensions':matched_size,
            'matching_quantization_tables':matched_tables,'identical_decoded_pixels':matched_pixels,
            'dimension_deltas':dict(deltas),'first_jpeg_has_standard_quality95_tables':matches95,
            'mean_pixel_error_dimension_matched':float(np.mean(pixel_errors)),
            'mean_cosine':float(cosine.mean()),'min_cosine':float(cosine.min()),'by_type':by_type}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(variant,json.dumps(report['variants'][variant]),flush=True)
    store.close()


if __name__=='__main__':
    main()
