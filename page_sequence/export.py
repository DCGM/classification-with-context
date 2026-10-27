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
    spec = state['specification']
    if (spec.get('auxiliary_edges') is not True or spec.get('backbone', {}).get('kind') != 'bert'
            or spec.get('fusion') != 'separate_mlp' or spec.get('dims', {}).get('pagination') != 5
            or set(spec.get('dims', {})) != {'image', 'text', 'stats', 'pagination'}
            or spec.get('num_types') != len(PAGE_TYPES)):
        raise ValueError('MetaKat requires pagination-aware BERT with separate MLPs and auxiliary start/end heads')
    from .models import from_specification
    from_specification(spec).load_state_dict(state['model'], strict=True)
    from .pagination import POLICY, encode, RECORD
    import numpy as np
    pagination = state['config'].get('pagination')
    if (not isinstance(pagination, dict)
            or state['provenance']['pagination']['policy'] != POLICY):
        raise ValueError('Unsupported pagination feature contract')
    cap = pagination.get('cap', 10000)
    encode(np.zeros(0, RECORD), cap)  # Validate the exact inference normalization.
    pagination_contract = {'policy': POLICY, 'cap': cap}
    # JSON round-trip drops Python-specific containers from metadata.
    specification = dict(state['specification'])
    metadata = json.loads(json.dumps({
        'format': 'metakat_context_v3',
        'specification': specification, 'features': state['features'],
        'page_types': state['page_types'], 'placements': state['placements'],
        'inference': state['config'].get('inference', {}),
    }))
    metadata['pagination'] = pagination_contract
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
