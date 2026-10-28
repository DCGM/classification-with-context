"""Extract bounded public-batch features with the actual MetaKat encoders.

Run with MetaKat's Python and PYTHONPATH pointing to its development checkout.
Only vectors and model outputs are saved, never OCR text or source images.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    from metakat.engine_config import load_config_file, prepare_engine_config
    from metakat.page_type.engines.core.context.context import PageTypeCoreEngineContext, scan_features
    from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', action='append', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    config = prepare_engine_config(load_config_file(args.config), base_dir=args.config.parent)
    engine = PageTypeCoreEngineContext(config['page_type']['core'])
    for name in args.batch:
        batch = args.root/name
        meta = json.loads((batch.with_name(name+'.results')/'metakat.json').read_text())
        pages = sorted((p for p in meta['elements'] if p['type']=='page'), key=lambda p:p['batch_index'])
        inputs = [PageTypeInput(image_path=str(batch/meta['page_to_image_mapping'][p['id']]),
                                ocr_path=str(batch/meta['page_to_alto_mapping'][p['id']]),
                                page_number=(p.get('pageNumber') or {}).get('text')) for p in pages]
        features = engine.features(inputs)
        np.savez_compressed(args.output/(name+'.features.npz'), **features)
        np.savez_compressed(args.output/(name+'.metakat.npz'),
                            **scan_features(engine.model, features, engine.inference, engine.device))
        print('EXTRACTED',name,len(pages),flush=True)


if __name__ == '__main__':
    main()
