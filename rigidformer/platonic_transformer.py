from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum, tensor
from torch.nn import Module, Linear, RMSNorm

import einx
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import masked_mean

from rigidformer.rotary_3d import RotaryEmbedding3D, apply_rotary_pos_emb

# constants

LinearNoBias = partial(Linear, bias = False)

# platonic symmetries

TETRAHEDRON_ROTATIONS = tensor([
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
    [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
    [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
    [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]],
    [[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]],
    [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]],
    [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
    [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]],
    [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
    [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]],
])

# helpers

def exists(v):
    return v is not None

# classes

class SwiGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LinearNoBias(dim, dim_inner * 2),
        SwiGLU(),
        LinearNoBias(dim_inner, dim)
    )

class PlatonicAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.to_qkv = LinearNoBias(dim, dim_inner * 4)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.split_heads = Rearrange('... n (h d) -> ... h n d', h = heads)
        self.merge_heads = Rearrange('... h n d -> ... n (h d)')

        # This experimental encoder may use a reduced head width. Make its
        # non-paper rotary width explicit instead of asking RotaryEmbedding3D
        # to silently truncate it.

        rotary_dim = (dim_head // 6) * 6
        assert rotary_dim > 0, 'attention head dimension must be at least 6 for 3D RoPE'
        self.rope = RotaryEmbedding3D(rotary_dim)

    def forward(
        self,
        x,
        pos_g,
        mask = None
    ):
        # qkv and gates
        
        q, k, v, gates = self.to_qkv(x).chunk(4, dim = -1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        # frame-dependent rope
        
        pos_emb = self.rope(pos_g)
        pos_emb = rearrange(pos_emb, '... g n d -> ... g 1 n d')
        
        q = apply_rotary_pos_emb(pos_emb, q)
        k = apply_rotary_pos_emb(pos_emb, k)

        # attention
        
        sim = einsum('... g h i d, ... g h j d -> ... g h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = einx.where('... j, ... g h i j, -> ... g h i j', mask, sim, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum('... g h i j, ... g h j d -> ... g h i d', attn, v)
        out = self.merge_heads(out)

        out = out * gates.sigmoid()
        return self.to_out(out)

class PlatonicTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        depth = 2,
        heads = 4,
        dim_head = 32,
        ff_mult = 4,
        align_canonical_frames = False,
        final_norm = True
    ):
        super().__init__()
        self.depth = depth

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                PlatonicAttention(dim, heads = heads, dim_head = dim_head),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

        self.to_out = LinearNoBias(dim, dim_out) if dim != dim_out else nn.Identity()

        self.register_buffer('rotations', TETRAHEDRON_ROTATIONS, persistent = False)

    def forward(
        self,
        features,
        pos,
        mask = None
    ):
        g = self.rotations.shape[0]
        
        # lift
        
        features = repeat(features, '... n d -> ... g n d', g = g)

        pos_g = einsum('... n c, g r c -> ... g n r', pos, self.rotations)

        # layers
        
        for attn_norm, attn, ff_norm, ff in self.layers:
            features = attn(attn_norm(features), pos_g, mask = mask) + features
            features = ff(ff_norm(features)) + features

        features = self.norm(features)

        # pool over points
        
        if exists(mask):
            mask = repeat(mask, '... n -> ... g n', g = g)
            
        pooled = masked_mean(features, mask, dim = -2)

        # pool over group
        
        pooled = reduce(pooled, '... g d -> ... d', 'mean')

        return self.to_out(pooled)
