from __future__ import annotations

import torch
from torch import einsum, stack, cat
from torch.nn import Module
from einops import rearrange

class RotaryEmbedding3D(Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()

        assert dim >= 6, '3D RoPE dimension must be at least 6'
        assert dim % 6 == 0, '3D RoPE dimension must be divisible by 6 (3 axes x 2 channels per rotary pair)'

        coord_dim = dim // 3
        inv_freq = omega ** (-torch.arange(0, coord_dim, 2).float() / coord_dim)

        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim
        self.coord_dim = coord_dim
    
    @property
    def device(self):
        return self.inv_freq.device

    def forward(self, pos):
        assert pos.shape[-1] == 3, '3D RoPE positions must have exactly three coordinates'

        # The paper assigns 32 channels to each coordinate in the 96D main
        # configuration: 16 log-spaced frequencies, each repeated over one
        # adjacent even-odd rotary pair.

        pos = pos.to(dtype = self.inv_freq.dtype)
        freqs = einsum('... p, f -> ... p f', pos, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim = -1)
        return rearrange(freqs, '... p f -> ... (p f)')

def rotate_pairs(x):
    assert x.shape[-1] % 2 == 0

    even, odd = x[..., 0::2], x[..., 1::2]
    return stack((-odd, even), dim = -1).flatten(-2)

def apply_rotary_pos_emb(pos_emb, t):
    rope_dim = pos_emb.shape[-1]
    assert rope_dim % 2 == 0, 'rotary embedding dimension must be even'
    assert rope_dim <= t.shape[-1], 'rotary embedding cannot exceed the query/key head dimension'

    t_rope, t_pass = t[..., :rope_dim], t[..., rope_dim:]

    # Evaluate trigonometric functions in at least fp32, then restore the
    # query/key dtype so mixed-precision attention does not silently upcast.

    trig_dtype = torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype
    pos_emb = pos_emb.to(dtype = trig_dtype)
    cos, sin = pos_emb.cos().to(t.dtype), pos_emb.sin().to(t.dtype)

    t_rope = t_rope * cos + rotate_pairs(t_rope) * sin
    return cat((t_rope, t_pass), dim = -1)
