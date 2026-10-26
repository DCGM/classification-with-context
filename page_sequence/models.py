"""Fusion and task heads shared by all backbones; Transformers is isolated to BertBackbone."""
import torch
from torch import nn

from .schema import PAGE_TYPES, PLACEMENTS


class BertBackbone(nn.Module):
    """Page embeddings enter BERT's normal position/type embedding + LayerNorm path once."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.max_length = model.config.max_position_embeddings
        # Measure checkpoint content scale before removing the unused vocabulary table.
        with torch.no_grad():
            w = model.get_input_embeddings().weight.float()
            self.initial_scale = float((w - w.mean(dim=-1, keepdim=True)).square().mean().sqrt())
        # inputs_embeds bypasses this table. Replacing rather than freezing removes ~7.8M params.
        model.set_input_embeddings(nn.Embedding(1, self.hidden_size))
        model.get_input_embeddings().weight.requires_grad_(False)
        model.config.vocab_size = 1

    @classmethod
    def create(cls, pretrained=True, checkpoint='google/bert_uncased_L-4_H-256_A-4',
               revision=None, config=None, local_files_only=False):
        from transformers import BertConfig, BertModel
        if config is not None:
            if pretrained:
                raise ValueError('use checkpoint/revision for pretrained weights, config for offline/random construction')
            model = BertModel(BertConfig.from_dict(config), add_pooling_layer=False)
        elif pretrained:
            model = BertModel.from_pretrained(checkpoint, revision=revision, add_pooling_layer=False,
                                               local_files_only=local_files_only)
        else:
            cfg = BertConfig.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
            model = BertModel(cfg, add_pooling_layer=False)
        return cls(model)

    def forward(self, x, attention_mask):
        if x.shape[1] > self.max_length:
            raise ValueError('window exceeds backbone position limit')
        return self.model(inputs_embeds=x, attention_mask=attention_mask,
                          token_type_ids=torch.zeros_like(attention_mask, dtype=torch.long),
                          return_dict=True).last_hidden_state

    def specification(self):
        return {'kind': 'bert', 'config': self.model.config.to_dict(), 'initial_scale': self.initial_scale}


class PageBackbone(nn.Module):
    """Page-wise baseline; same fusion and adjacent-pair heads, no context mixing."""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size, self.initial_scale = hidden_size, .02
        self.layers = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))

    def forward(self, x, attention_mask):
        return self.layers(x)

    def specification(self):
        return {'kind': 'page', 'hidden_size': self.hidden_size}


class Fusion(nn.Module):
    def __init__(self, dims, hidden_size, scale, kind='linear'):
        super().__init__()
        if kind not in ('linear', 'joint_mlp', 'separate_mlp'):
            raise ValueError(kind)
        self.kind = kind
        self.names = ('image', 'text', 'stats') + (('pagination',) if 'pagination' in dims else ())
        self.branches = nn.ModuleDict()
        if kind == 'separate_mlp':
            for name in self.names:
                self.branches[name] = nn.Sequential(nn.Linear(dims[name], hidden_size), nn.GELU(),
                                                    nn.LayerNorm(hidden_size))
            width = len(self.names) * hidden_size + 3
        else:
            width = sum(dims.values()) + 3
        self.projection = (nn.Sequential(nn.Linear(width, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
                           if kind == 'joint_mlp' else nn.Linear(width, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, features):
        flags = features['presence']
        parts = []
        for i, name in enumerate(self.names):
            x = self.branches[name](features[name]) if self.kind == 'separate_mlp' else features[name]
            # Mask after biased/nonlinear branch operations, including future custom feature sources.
            presence = features['pagination'][..., 4:5] if name == 'pagination' else flags[..., i:i + 1]
            parts.append(x * presence)
        return self.scale * self.norm(self.projection(torch.cat(parts + [flags], dim=-1)))


class PageTagger(nn.Module):
    def __init__(self, backbone, dims, fusion='linear', num_types=len(PAGE_TYPES)):
        super().__init__()
        self.backbone, self.dims, self.fusion_kind, self.num_types = backbone, dims, fusion, num_types
        h = backbone.hidden_size
        self.fusion = Fusion(dims, h, backbone.initial_scale, fusion)
        self.type_head = nn.Linear(h, num_types)
        self.placement_head = nn.Linear(h, len(PLACEMENTS))
        self.start_head = nn.Sequential(nn.Linear(4 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, features, attention_mask):
        if not attention_mask.any(dim=-1).all():
            raise ValueError('empty sequence in batch')
        h = self.backbone(self.fusion(features), attention_mask)
        a, b = h[:, :-1], h[:, 1:]
        starts = self.start_head(torch.cat((a, b, (a - b).abs(), a * b), dim=-1)).squeeze(-1)
        # Position zero has no predecessor. Its zero is a placeholder, never a negative target.
        starts = torch.cat((h.new_zeros(h.shape[0], 1), starts), dim=1)
        pair_mask = attention_mask.clone()
        pair_mask[:, 0] = False
        pair_mask[:, 1:] &= attention_mask[:, :-1]
        return {'type_logits': self.type_head(h), 'placement_logits': self.placement_head(h),
                'starts_document_logits': starts, 'pair_mask': pair_mask}

    def specification(self):
        return {'backbone': self.backbone.specification(), 'dims': self.dims,
                'fusion': self.fusion_kind, 'num_types': self.num_types}


def from_specification(spec):
    b = spec['backbone']
    backbone = (BertBackbone.create(pretrained=False, config=b['config']) if b['kind'] == 'bert'
                else PageBackbone(b['hidden_size']) if b['kind'] == 'page' else None)
    if backbone is None:
        raise ValueError('unknown backbone')
    if 'initial_scale' in b:
        backbone.initial_scale = b['initial_scale']
    return PageTagger(backbone, spec['dims'], spec['fusion'], spec['num_types'])
