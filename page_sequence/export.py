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
    pagination = state['config'].get('pagination')
    has_pagination = 'pagination' in state['specification']['dims']
    if bool(pagination) != has_pagination:
        raise ValueError('Pagination configuration/model dimensions mismatch')
    pagination_contract = None
    if has_pagination:
        from .pagination import POLICY, encode, RECORD
        import numpy as np
        if (state['specification']['dims']['pagination'] != 5
                or state['provenance']['pagination']['policy'] != POLICY):
            raise ValueError('Unsupported pagination feature contract')
        cap = pagination.get('cap', 10000)
        encode(np.zeros(0, RECORD), cap)  # Validate the exact inference normalization.
        pagination_contract = {'policy': POLICY, 'cap': cap}
    # JSON round-trip drops Python-specific containers from metadata.
    metadata = json.loads(json.dumps({
        'format': 'metakat_context_v2' if has_pagination else 'metakat_context_v1',
        'specification': state['specification'], 'features': state['features'],
        'page_types': state['page_types'], 'placements': state['placements'],
        'inference': state['config'].get('inference', {}),
    }))
    if pagination_contract is not None:
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
