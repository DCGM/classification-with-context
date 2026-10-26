"""Export a trusted local training checkpoint as a portable MetaKat inference artifact.

No optimizer, sampler, RNG, paths to data stores or training dependencies are
needed at inference. Input checkpoints use pickle: export only your own files.
"""
import argparse
import json
from pathlib import Path

import torch

from .schema import PAGE_TYPES, PLACEMENTS


def export_checkpoint(checkpoint, output):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (state['format'] != 1 or tuple(state['page_types']) != PAGE_TYPES
            or tuple(state['placements']) != PLACEMENTS):
        raise ValueError('Checkpoint schema/label mismatch')
    if state['config'].get('pagination'):
        raise ValueError('MetaKat context_v1 does not support pagination; use the page_sequence inference API')
    # JSON round-trip drops Python-specific containers from metadata.
    metadata = json.loads(json.dumps({
        'format': 'metakat_context_v1',
        'specification': state['specification'], 'features': state['features'],
        'page_types': state['page_types'], 'placements': state['placements'],
        'inference': state['config'].get('inference', {}),
    }))
    artifact = dict(metadata, model={k: v.detach().cpu().clone() for k, v in state['model'].items()})
    output = Path(output)
    temporary = output.with_suffix(output.suffix + '.writing')
    torch.save(artifact, temporary)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
