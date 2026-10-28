"""Separate encoder/store differences from contextual-model implementation differences."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from page_sequence.inference import InferenceConfig, StreamMerger, window_spans
from page_sequence.pagination import RECORD, candidate, encode
from page_sequence.runner import load_checkpoint
from page_sequence.scan_borders import encode_boundary_features
from training_store.reader import TrainingStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'fresh', 'checkpoint', 'store', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.checkpoint, 'cuda')
    model.eval()
    store = TrainingStore(args.store)
    cfg = InferenceConfig(window_size=512, stride=448, edge_ramp=48, edge_ignore=12,
                          edge_power=3, batch_size=8, tail='regular')
    def infer(features):
        n = len(features['presence'])
        merger = StreamMerger(n, model.num_types, cfg)
        for lo, hi in window_spans(n, cfg.window_size, cfg.stride, cfg.tail):
            tensors = {key: torch.tensor(value[None, lo:hi], device='cuda', dtype=torch.float32)
                       for key, value in features.items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                result = model(tensors, torch.ones((1, hi-lo), dtype=torch.bool, device='cuda'))
            merger.add(lo, hi, {key: value[0].float().cpu().numpy() for key, value in result.items()})
        return merger.finalize()
    report = []
    for path in sorted(args.fresh.glob('*.features.npz')):
        name = path.name.removesuffix('.features.npz')
        manifest = json.loads((args.root/name/'manifest.json').read_text())
        output = json.loads((args.root/(name+'.results')/'metakat.json').read_text())
        by_id = {p['id']:p for p in output['elements'] if p['type']=='page'}
        native = store.features(store.read_rows([p['row_id'] for p in manifest['pages']]))
        records = np.array([candidate((by_id[p['page']].get('pageNumber') or {}).get('text',''))
                            for p in manifest['pages']], RECORD)
        native['pagination'] = encode(records)
        native['boundaries'] = encode_boundary_features(len(records), True, True)
        with np.load(path) as data:
            fresh = dict(data)
        with np.load(args.fresh/(name+'.metakat.npz')) as data:
            metakat = dict(data)
        entry = {'batch':name, 'pages':len(records), 'features':{}, 'hybrids':{}}
        for key in fresh:
            a,b = native[key].astype(np.float64), fresh[key].astype(np.float64)
            metrics = {'max_abs':float(np.max(np.abs(a-b))), 'mean_abs':float(np.mean(np.abs(a-b)))}
            if key in ('image','text','stats'):
                norms = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
                valid=norms>0
                cosine=(a*b).sum(-1)[valid]/norms[valid]
                metrics.update(mean_cosine=float(cosine.mean()) if len(cosine) else None,
                               min_cosine=float(cosine.min()) if len(cosine) else None,
                               native_missing=int((~np.any(a,axis=-1)).sum()),
                               fresh_missing=int((~np.any(b,axis=-1)).sum()))
            entry['features'][key]=metrics
        for variant in ('fresh', 'native', 'image', 'text', 'stats', 'presence'):
            features = fresh if variant=='fresh' else native.copy()
            if variant not in ('fresh','native'):
                features[variant]=fresh[variant]
            predicted = infer(features)
            metrics = {key:int(np.count_nonzero(predicted[key]!=metakat[key])) for key in ('type','placement','starts')}
            metrics['max_probability_error'] = {key:float(np.nanmax(np.abs(predicted[key]-metakat[key])))
                for key in ('type_probs','placement_probs','starts_document_probs','page_start_probs','page_end_probs')}
            entry['hybrids'][variant]=metrics
        report.append(entry)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
    store.close()


if __name__=='__main__':
    main()
