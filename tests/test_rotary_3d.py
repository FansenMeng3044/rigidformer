import pytest
import torch

from rigidformer import Rigidformer
from rigidformer.rotary_3d import RotaryEmbedding3D, apply_rotary_pos_emb


def test_paper_arope_dimension_frequency_layout_and_passthrough():
    torch.manual_seed(0)

    rope = RotaryEmbedding3D(96)
    pos = torch.randn(2, 5, 3)
    phases = rope(pos)

    assert rope.dim == 96
    assert rope.coord_dim == 32
    assert rope.inv_freq.numel() == 16
    assert phases.shape == (2, 5, 96)

    # Each frequency is repeated over one adjacent even-odd rotary pair.
    phase_pairs = phases.reshape(2, 5, 48, 2)
    assert torch.equal(phase_pairs[..., 0], phase_pairs[..., 1])

    q = torch.randn(2, 6, 5, 128)
    rotated = apply_rotary_pos_emb(phases[:, None], q)

    assert rotated.shape == q.shape
    assert torch.equal(rotated[..., 96:], q[..., 96:])


def test_paper_arope_common_translation_preserves_qk_dot_product():
    torch.manual_seed(0)

    rope = RotaryEmbedding3D(96)
    query_pos = torch.randn(2, 4, 3)
    key_pos = torch.randn(2, 4, 3)
    translation = torch.randn(2, 1, 3)

    queries = torch.randn(2, 3, 4, 128)
    keys = torch.randn(2, 3, 4, 128)

    def rotated_dot(q_pos, k_pos):
        q = apply_rotary_pos_emb(rope(q_pos)[:, None], queries)
        k = apply_rotary_pos_emb(rope(k_pos)[:, None], keys)
        return (q * k).sum(dim = -1)

    before = rotated_dot(query_pos, key_pos)
    after = rotated_dot(query_pos + translation, key_pos + translation)

    assert torch.allclose(before, after, atol = 1e-5, rtol = 1e-5)


def test_paper_arope_anchor_order_invariance_and_unpositioned_registers():
    torch.manual_seed(0)

    model = Rigidformer(
        dim = 32,
        dim_head = 128,
        heads = 1,
        num_register_tokens = 16,
        object_self_attn_depth = 1,
        anchor_cross_attn_depth = 1,
        object_hidden_layers = (1,),
        pointnet_vertex_dim = 32,
        pointnet_num_samples = (8, 8, 8),
        anchor_avp_dim = 16
    )

    anchors = torch.randn(2, 3, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    anchor_phase, object_phase, phase_with_registers = model._build_arope_embeddings(anchors)
    _, permuted_object_phase, _ = model._build_arope_embeddings(anchors[:, :, permutation])

    assert anchor_phase.shape == (2, 3, 4, 96)
    assert object_phase.shape == (2, 3, 96)
    assert phase_with_registers.shape == (2, 19, 96)
    assert torch.allclose(object_phase, permuted_object_phase, atol = 1e-7, rtol = 1e-7)
    assert torch.count_nonzero(phase_with_registers[:, :16]) == 0

    register_q = torch.randn(2, 6, 16, 128)
    rotated_register_q = apply_rotary_pos_emb(phase_with_registers[:, None, :16], register_q)
    assert torch.equal(register_q, rotated_register_q)


def test_paper_arope_mixed_precision_keeps_query_dtype_and_gradients():
    rope = RotaryEmbedding3D(96)
    pos = torch.randn(2, 5, 3)
    q = torch.randn(2, 2, 5, 128, dtype = torch.float16, requires_grad = True)

    rotated = apply_rotary_pos_emb(rope(pos)[:, None], q)
    rotated.float().sum().backward()

    assert rotated.dtype == q.dtype
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_paper_arope_rejects_implicit_or_invalid_dimensions():
    with pytest.raises(AssertionError, match = 'divisible by 6'):
        RotaryEmbedding3D(128)

    with pytest.raises(AssertionError, match = 'must not exceed'):
        Rigidformer(dim = 32, dim_head = 64)
