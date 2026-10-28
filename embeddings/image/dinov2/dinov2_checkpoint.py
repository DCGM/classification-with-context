"""Load a DINOv2-with-registers backbone exported by LightlyTrain into Hugging Face transformers.

LightlyTrain's exported_models/exported_last.pt (method "dinov2", model
"dinov2/vitb14") is a plain state dict in the original facebookresearch/dinov2
layout: cls_token, register_tokens, pos_embed, patch_embed.proj, blocks.N with a
fused attn.qkv, LayerScale ls1/ls2.gamma and an MLP fc1/fc2, and a final norm.
It maps one-to-one onto transformers' Dinov2WithRegistersModel once qkv is split
into query/key/value; the load is strict, so any unexpected or missing weight
fails instead of being silently initialized.
"""

import re
from pathlib import Path
from typing import Dict

import torch
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

ARCHITECTURES = {
    # hidden size, layers, heads for the ViT sizes DINOv2 ships
    'vits14': (384, 12, 6),
    'vitb14': (768, 12, 12),
    'vitl14': (1024, 24, 16),
}


def convert_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Renames original DINOv2 keys to Dinov2WithRegistersModel keys, splitting the fused qkv."""
    converted = {}
    direct = {
        'cls_token': 'embeddings.cls_token',
        'mask_token': 'embeddings.mask_token',
        'register_tokens': 'embeddings.register_tokens',
        'pos_embed': 'embeddings.position_embeddings',
        'patch_embed.proj.weight': 'embeddings.patch_embeddings.projection.weight',
        'patch_embed.proj.bias': 'embeddings.patch_embeddings.projection.bias',
        'norm.weight': 'layernorm.weight',
        'norm.bias': 'layernorm.bias',
    }
    block_parts = {
        'norm1': 'norm1', 'norm2': 'norm2',
        'attn.proj': 'attention.output.dense',
        'ls1.gamma': 'layer_scale1.lambda1', 'ls2.gamma': 'layer_scale2.lambda1',
        'mlp.fc1': 'mlp.fc1', 'mlp.fc2': 'mlp.fc2',
    }
    for key, value in state.items():
        if key in direct:
            converted[direct[key]] = value
            continue
        match = re.fullmatch(r'blocks\.(\d+)\.(.+?)(?:\.(weight|bias))?', key)
        if match is None:
            raise KeyError(f'Unexpected DINOv2 weight {key!r}')
        layer, part, kind = match.groups()
        prefix = f'encoder.layer.{layer}'
        if part == 'attn.qkv':
            for name, chunk in zip(('query', 'key', 'value'), value.chunk(3, dim=0)):
                converted[f'{prefix}.attention.attention.{name}.{kind}'] = chunk.clone()
        elif part in block_parts:
            converted[f'{prefix}.{block_parts[part]}' + (f'.{kind}' if kind else '')] = value
        else:
            raise KeyError(f'Unexpected DINOv2 weight {key!r}')
    return converted


def load_dinov2_with_registers(checkpoint: Path, architecture: str = 'vitb14',
                               attn_implementation: str = 'sdpa') -> Dinov2WithRegistersModel:
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for wrapper in ('state_dict', 'model'):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    hidden, layers, heads = ARCHITECTURES[architecture]
    num_positions = state['pos_embed'].shape[1] - 1
    grid = int(round(num_positions ** 0.5))
    config = Dinov2WithRegistersConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, mlp_ratio=4, patch_size=14,
        image_size=grid * 14, num_register_tokens=state['register_tokens'].shape[1], layerscale_value=1.0,
        use_swiglu_ffn=False, attn_implementation=attn_implementation)
    model = Dinov2WithRegistersModel(config)
    model.load_state_dict(convert_state_dict(state), strict=True)
    return model
