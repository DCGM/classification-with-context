"""Run original dump encoder functions and MetaKat on identical public inputs.

Run in MetaKat's environment with its checkout on PYTHONPATH. Models are shared
between calls; the DINO checkpoint loader sources are separately byte-compared.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    import metakat.page_type
    from metakat.engine_config import load_config_file, prepare_engine_config
    from metakat.page_type.engines.core.context.dino import DinoEncoder
    from metakat.page_type.engines.core.context.mmbert import MmbertEncoder
    from metakat.page_type.engines.core.context.alto_stats import AltoStatsEncoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Expose only the old training/nets packages alongside current engines.
    metakat.page_type.__path__.append(str(args.original_root/'metakat/page_type'))
    original_image = importlib.import_module('metakat.page_type.training.image_embeddings.infer_dinov2_vitb14')
    original_text = importlib.import_module('metakat.page_type.training.text_embeddings.embedding_inference')
    core = prepare_engine_config(load_config_file(args.config), base_dir=args.config.parent)['page_type']['core']
    meta = json.loads((args.batch/'metakat.json').read_text())
    pages = sorted(meta['elements'], key=lambda p:p['batch_index'])
    paths = [args.batch/meta['page_to_image_mapping'][p['id']] for p in pages]
    ocr = [args.batch/meta['page_to_alto_mapping'][p['id']] for p in pages]
    torch.set_num_threads(2)
    image = DinoEncoder(core['image_checkpoint_path'], 'cuda', **core['image'])
    current, present = image.encode(paths)
    original_pixels = [original_image.load_image(str(path), 224) for path in paths]
    identical_pixels = all(np.array_equal(old, image.load_image(path)) for old,path in zip(original_pixels, paths))
    reference = []
    for start in range(0,len(paths),image.batch_size):
        pixels = torch.from_numpy(np.stack(original_pixels[start:start+image.batch_size]))
        vectors = original_image.embed(image.model, pixels, image.device, torch.float16, 'cls',
                                       image.model.config.num_register_tokens)
        reference.append(vectors.astype(np.float16).astype(np.float32))
    reference = np.concatenate(reference)
    image_result = {'pixels_identical': identical_pixels, 'vectors_identical': np.array_equal(current,reference),
                    'max_abs': float(np.max(np.abs(current-reference)))}
    old_loader = args.original_root/'metakat/page_type/training/image_embeddings/dinov2_checkpoint.py'
    current_loader = Path(importlib.import_module('metakat.page_type.engines.core.context.dinov2_checkpoint').__file__)
    image_result['checkpoint_loader_sources_identical'] = old_loader.read_bytes()==current_loader.read_bytes()
    del image
    torch.cuda.empty_cache()
    text = MmbertEncoder(core['text_model_dir'], 'cuda', **core['text'])
    texts = list(AltoStatsEncoder.iter_texts(ocr))
    prepared = [value[:text.max_chars] if value.strip() else '' for value in texts]
    current, present = text.encode(texts)
    reference = []
    for start in range(0,len(texts),text.batch_size):
        values = prepared[start:start+text.batch_size]
        ids = text.tokenizer(values, add_special_tokens=True, truncation=False,
                             return_attention_mask=False)['input_ids']
        ids = [v if len(v)<=text.max_length else v[:text.max_length-1]+v[-1:] for v in ids]
        chunk = original_text.Chunk([str(i).encode() for i in range(len(values))], ids, 0, 0)
        vectors = original_text.embed_chunk(text.model, original_text.mean_pool, chunk,
                    text.tokenizer.pad_token_id, text.device, torch.bfloat16,
                    text.max_tokens_per_batch, text.batch_size)
        reference.append(vectors.astype(np.float16).astype(np.float32))
    reference = np.concatenate(reference)
    report = {'batch':str(args.batch),'pages':len(pages),
              'scope':'same inputs and controlled matching batch groups; source encoders use shared loaded models',
              'image':image_result,'text':{'vectors_identical':np.array_equal(current,reference),
                                         'max_abs':float(np.max(np.abs(current-reference)))},
              'original_source':{'image':original_image.__file__, 'text':original_text.__file__}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
