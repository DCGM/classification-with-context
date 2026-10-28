"""Compare the portable MetaKat engine against training-repo inference on fixed features.

No image/OCR data or feature encoders are loaded. Run with the context environment;
the second half runs independently with MetaKat's Python and checkout.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def worker(args):
    from metakat.page_type.engines.core.context.context import scan_features
    from metakat.page_type.engines.core.context.context import PageTypeCoreEngineContext
    from types import SimpleNamespace
    from unittest.mock import patch
    state = torch.load(args.output / 'context.pt', weights_only=True, map_location='cpu')
    dims = state['specification']['dims']
    with patch('metakat.page_type.engines.core.context.context.DinoEncoder', return_value=SimpleNamespace(dim=dims['image'])), \
         patch('metakat.page_type.engines.core.context.context.MmbertEncoder', return_value=SimpleNamespace(dim=dims['text'])):
        engine = PageTypeCoreEngineContext({'name': 'context', 'device': 'cpu',
            'checkpoint_path': str(args.output / 'context.pt'),
            'image_checkpoint_path': 'unused', 'text_model_dir': 'unused'})
    model, cfg = engine.model, engine.inference
    for path in sorted(args.output.glob('features-*.npz')):
        with np.load(path) as data:
            features = dict(data)
        np.savez(args.output / path.name.replace('features-', 'metakat-'),
                 **scan_features(model, features, cfg))


def compare(args):
    from page_sequence.export import export_checkpoint
    from page_sequence.models import from_specification
    from page_sequence.inference import InferenceConfig, StreamMerger, window_spans
    from page_sequence.scan_borders import encode_boundary_features
    args.output.mkdir(parents=True, exist_ok=True)
    export_checkpoint(args.checkpoint, args.output / 'context.pt')
    state = torch.load(args.output / 'context.pt', weights_only=True, map_location='cpu')
    model = from_specification(state['specification']).eval()
    model.load_state_dict(state['model'], strict=True)
    settings = dict(state['inference'])
    settings['stride'] = settings['window_size'] - settings.pop('overlap')
    cfg = InferenceConfig(**settings)
    rng = np.random.default_rng(17)
    for n in (1, 17, 512, 1024):
        base = {k: rng.normal(size=(n, d)).astype(np.float32) for k, d in model.dims.items()}
        base['presence'] = rng.integers(0, 2, size=(n, 3)).astype(np.float32)
        # Include fully missing pages; preserve the separate boundary signal.
        for i, name in enumerate(('image', 'text', 'stats')):
            base[name] *= base['presence'][:, i:i + 1]
        base['pagination'][:] = 0
        for supplied in (False, True):
            metadata = {}
            if supplied:
                starts, ends = np.zeros(n, bool), np.zeros(n, bool)
                starts[0], ends[-1] = True, True
                metadata = dict(document_starts=starts, document_ends=ends)
            features = dict(base, boundaries=encode_boundary_features(n, True, True, **metadata))
            name = f'{n}-{int(supplied)}.npz'
            np.savez(args.output / ('features-' + name), **features)
            merger = StreamMerger(n, model.num_types, cfg)
            # Separate calls per window also check MetaKat's padded window batching.
            for lo, hi in window_spans(n, cfg.window_size, cfg.stride, cfg.tail):
                with torch.inference_mode():
                    output = model({k: torch.from_numpy(v[None, lo:hi]) for k, v in features.items()},
                                   torch.ones((1, hi - lo), dtype=torch.bool))
                merger.add(lo, hi, {k: v[0].float().numpy() for k, v in output.items()})
            np.savez(args.output / ('reference-' + name), **merger.finalize())
    import os
    env = dict(os.environ, PYTHONPATH=str(args.metakat))
    subprocess.run([str(args.metakat_python), str(Path(__file__).resolve()), '--worker',
                    '--output', str(args.output)], env=env, cwd=args.metakat, check=True)
    report = {'artifact': str(args.output / 'context.pt'), 'inference': state['inference'], 'cases': []}
    for path in sorted(args.output.glob('reference-*.npz')):
        with np.load(path) as reference, np.load(args.output / path.name.replace('reference-', 'metakat-')) as actual:
            differences = {}
            assert set(reference.files) == set(actual.files)
            for key in reference.files:
                if reference[key].dtype.kind in 'bui':
                    np.testing.assert_array_equal(reference[key], actual[key])
                else:
                    np.testing.assert_allclose(reference[key], actual[key], atol=2e-6, rtol=1e-5)
                    finite = np.isfinite(reference[key])
                    differences[key] = float(np.max(np.abs(reference[key][finite] - actual[key][finite]), initial=0))
            report['cases'].append({'case': path.stem.removeprefix('reference-'), 'max_absolute_error': differences})
    (args.output / 'parity-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metakat', type=Path)
    parser.add_argument('--metakat-python', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.worker:
        worker(args)
    else:
        if any(x is None for x in (args.checkpoint, args.metakat, args.metakat_python)):
            parser.error('--checkpoint, --metakat and --metakat-python required')
        compare(args)


if __name__ == '__main__':
    main()
