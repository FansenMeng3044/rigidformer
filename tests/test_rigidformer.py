import torch

import pytest
param = pytest.mark.parametrize

@param('fps', (False, True))
@param('test_rand_steps', (False, True))
@param('attn_residual_block_size', (2, 4))
@param('variable_object_lens', (False, True))
@param('variable_point_lens', (False, True))
@param('anchor_self_attn', (False, True))
@param('use_platonic_transformer', (False, True))
def test_rigidformer(
    fps,
    test_rand_steps,
    attn_residual_block_size,
    variable_object_lens,
    variable_point_lens,
    anchor_self_attn,
    use_platonic_transformer
):
    from rigidformer.rigidformer import Rigidformer, RigidformerRolloutWrapper, PointNet

    object_pos = torch.randn(1, 2, 64, 3)
    object_pos_prev = torch.randn(1, 2, 64, 3)
    object_pos_next = torch.randn(1, 2, 64, 3)
    vertex_properties = torch.randn(1, 2, 3)

    anchor_indices = torch.randint(0, 64, (1, 2, 4))

    delta_times = torch.randn(1)

    rigidformer = Rigidformer(
        32,
        dim_head = 6,
        arope_dim = 6,
        heads = 4,
        num_register_tokens = 2,
        object_self_attn_depth = 2,
        anchor_cross_attn_depth = 2,
        object_hidden_layers = (0, 2),
        attn_residual_block_size = attn_residual_block_size,
        anchor_self_attn = anchor_self_attn,
        pointnet_vertex_dim = 32,
        pointnet_num_samples = (8, 8, 8),
        anchor_avp_dim = 16,
        use_platonic_transformer = use_platonic_transformer
    )

    kwargs = dict()
    if not fps:
        kwargs.update(anchor_indices = anchor_indices)

    if variable_object_lens:
        kwargs.update(object_lens = torch.tensor([1]))

    if variable_point_lens:
        kwargs.update(object_point_lens = torch.randint(48, 65, (1, 2)))

    loss, loss_breakdown = rigidformer(
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_pos = object_pos,
        object_pos_prev = object_pos_prev,
        object_pos_next = object_pos_next,
        object_first_frame_pos = object_pos_prev,
        **kwargs
    )

    loss.backward()

    rollout_wrapper = RigidformerRolloutWrapper(rigidformer)

    if test_rand_steps:
        delta_times_input = rollout_wrapper.rand_steps(
            delta_times = delta_times,
            num_rand_substeps = 3,
            max_step_weight = 3
        )
        assert delta_times_input.shape == (1, 3)
        assert torch.allclose(delta_times_input.sum(dim = -1), delta_times)

        num_steps = None
    else:
        delta_times_input = delta_times
        num_steps = 2

    object_positions = rollout_wrapper(
        delta_times = delta_times_input,
        num_steps = num_steps,
        vertex_properties = vertex_properties,
        object_positions = [object_pos_prev, object_pos],
        **kwargs
    )

    assert len(object_positions) == (5 if test_rand_steps else 4)

    last_position = object_positions[-1]

    assert last_position.shape == (1, 2, 64, 3)

def test_pointnet():
    from rigidformer.rigidformer import PointNet

    features = torch.randn(2, 2, 64, 16)
    pos = torch.randn(2, 2, 64, 3)

    net = PointNet(dim = 16, dim_out = 32)
    out = net(features, pos)

    assert out.shape == (2, 2, 32)

def test_paper_hierarchical_pointnet():
    from rigidformer import PaperHierarchicalPointNet

    torch.manual_seed(0)
    features = torch.randn(2, 2, 64, 12)
    pos = torch.randn(2, 2, 64, 3)
    mask = torch.ones(2, 2, 64, dtype = torch.bool)
    mask[0, 0, 51:] = False

    net = PaperHierarchicalPointNet(
        dim = 12,
        dim_out = 32,
        vertex_dim = 64,
        num_samples = (8, 8, 8)
    ).eval()

    level_sizes = []
    hooks = [
        layer.register_forward_hook(lambda _module, _inputs, output: level_sizes.append(output[0].shape[-2]))
        for layer in net.hierarchy
    ]

    object_tokens_1, vertex_features_1 = net(features, pos, mask = mask, reference_pos = pos)
    object_tokens_2, vertex_features_2 = net(features, pos, mask = mask, reference_pos = pos)

    for hook in hooks:
        hook.remove()

    assert object_tokens_1.shape == (2, 2, 32)
    assert vertex_features_1.shape == (2, 2, 64, 64)
    assert level_sizes[:3] == [32, 16, 8]
    assert torch.equal(object_tokens_1, object_tokens_2)
    assert torch.equal(vertex_features_1, vertex_features_2)

    features_with_bad_padding = features.clone()
    pos_with_bad_padding = pos.clone()
    features_with_bad_padding[0, 0, 51:] = 1e6
    pos_with_bad_padding[0, 0, 51:] = 1e6

    object_tokens_with_bad_padding, _ = net(
        features_with_bad_padding,
        pos_with_bad_padding,
        mask = mask,
        reference_pos = pos_with_bad_padding
    )
    assert torch.allclose(object_tokens_1[0, 0], object_tokens_with_bad_padding[0, 0])

    (object_tokens_1.sum() + vertex_features_1.sum()).backward()

def test_deterministic_fps():
    from rigidformer import deterministic_farthest_point_sample

    torch.manual_seed(0)
    pos = torch.randn(2, 3, 64, 3)
    first = deterministic_farthest_point_sample(pos, 8)
    second = deterministic_farthest_point_sample(pos, 8)

    assert torch.equal(first, second)

def test_reference_frame_is_required():
    from rigidformer import Rigidformer

    model = Rigidformer(
        dim = 32,
        dim_head = 8,
        arope_dim = 6,
        heads = 2,
        object_self_attn_depth = 2,
        anchor_cross_attn_depth = 2,
        object_hidden_layers = (0, 2),
        pointnet_vertex_dim = 32,
        pointnet_num_samples = (8, 8, 8),
        anchor_avp_dim = 16
    )

    with pytest.raises(AssertionError, match = 'object_first_frame_pos must be provided'):
        model(
            delta_times = torch.ones(1),
            vertex_properties = torch.randn(1, 2, 3),
            object_pos = torch.randn(1, 2, 64, 3),
            object_pos_prev = torch.randn(1, 2, 64, 3)
        )

def test_paper_predictor_dimensions_and_zero_init():
    from rigidformer import Rigidformer

    model = Rigidformer(
        dim = 32,
        dim_head = 8,
        arope_dim = 6,
        heads = 2,
        object_self_attn_depth = 2,
        anchor_cross_attn_depth = 2,
        object_hidden_layers = (0, 2)
    )

    assert model.hierarchical_encoder.vertex_dim == 1024
    assert model.anchor_avp.proj_in.in_features == 1024
    assert model.anchor_avp.proj_in.out_features == 256
    assert model.anchor_avp.proj_out.in_features == 256
    assert model.anchor_avp.proj_out.out_features == 256
    assert model.anchor_query_fuse.net[0].in_features == 271
    assert torch.count_nonzero(model.anchor_avp.proj_out.weight) == 0
    assert torch.count_nonzero(model.anchor_avp.proj_out.bias) == 0

def test_paper_swiglu_uses_silu_and_full_2_5x_hidden_width():
    from torch.nn import functional as F
    from rigidformer.rigidformer import SwiGluFeedforward

    torch.manual_seed(0)

    dim = 16
    ff = SwiGluFeedforward(dim = dim, expansion_factor = 2.5)
    tokens = torch.randn(2, 3, dim, requires_grad = True)

    projected, gates = ff.proj_in(tokens).chunk(2, dim = -1)
    expected = ff.proj_out(projected * F.silu(gates))
    actual = ff(tokens)

    assert ff.dim_inner == 40
    assert ff.proj_in.in_features == 16
    assert ff.proj_in.out_features == 80
    assert ff.proj_out.in_features == 40
    assert ff.proj_out.out_features == 16
    assert torch.equal(actual, expected)

    actual.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()

def test_block_attention_residual_matches_paper_equations():
    from rigidformer import BlockAttentionResidual

    torch.manual_seed(0)

    dim = 8
    residual = BlockAttentionResidual(dim)
    blocks = [
        torch.randn(2, 3, dim, requires_grad = True),
        torch.randn(2, 3, dim, requires_grad = True)
    ]
    partial_block = torch.randn(2, 3, dim, requires_grad = True)
    values = torch.stack([*blocks, partial_block], dim = 0)

    # All pseudo-queries must be zero-initialized, making the initial depth
    # distribution uniform regardless of the RMS-normalized keys.

    assert torch.count_nonzero(residual.query) == 0
    assert torch.allclose(residual(blocks, partial_block), values.mean(dim = 0))

    with torch.no_grad():
        residual.query.copy_(torch.linspace(-.4, .4, dim))

    keys = residual.key_rmsnorm(values)
    logits = torch.einsum('d,sbnd->sbn', residual.query, keys)
    weights = logits.softmax(dim = 0)
    expected = (weights[..., None] * values).sum(dim = 0)
    actual = residual(blocks, partial_block)

    assert torch.allclose(weights.sum(dim = 0), torch.ones_like(weights[0]))
    assert torch.allclose(actual, expected)

    actual.square().mean().backward()

    assert residual.query.grad is not None
    assert torch.isfinite(residual.query.grad).all()
    assert all(source.grad is not None for source in [*blocks, partial_block])
    assert all(torch.isfinite(source.grad).all() for source in [*blocks, partial_block])

def test_rigidformer_block_attnres_uses_sublayers_and_paper_block_size():
    from rigidformer import BlockAttentionResidual, Rigidformer

    model = Rigidformer(
        dim = 24,
        dim_head = 8,
        arope_dim = 6,
        heads = 2,
        num_register_tokens = 2,
        object_self_attn_depth = 4,
        anchor_cross_attn_depth = 1,
        object_hidden_layers = (4,),
        attn_residual_block_size = 4,
        pointnet_vertex_dim = 32,
        pointnet_num_samples = (4, 4, 4),
        anchor_avp_dim = 16
    ).eval()

    residuals = [
        residual
        for layer in model.self_attn_layers
        for residual in layer[-2:]
    ]
    residuals.append(model.object_final_attn_residual)

    assert len(residuals) == 9
    assert all(isinstance(residual, BlockAttentionResidual) for residual in residuals)
    assert all(torch.count_nonzero(residual.query) == 0 for residual in residuals)

    source_counts = []
    hooks = [
        residual.register_forward_pre_hook(
            lambda _module, inputs: source_counts.append(
                len(inputs[0]) + int(inputs[1] is not None)
            )
        )
        for residual in residuals
    ]

    object_pos = torch.randn(1, 2, 32, 3)
    object_pos_prev = torch.randn(1, 2, 32, 3)

    with torch.no_grad():
        prediction = model(
            delta_times = torch.ones(1),
            vertex_properties = torch.randn(1, 2, 3),
            object_pos = object_pos,
            object_pos_prev = object_pos_prev,
            object_first_frame_pos = object_pos_prev,
            anchor_indices = torch.randint(0, 32, (1, 2, 4))
        )

    for hook in hooks:
        hook.remove()

    # Four Transformer layers contain eight AttnRes layers. S=4 yields two
    # completed transformation blocks; the final read sees b_0, b_1, and b_2.

    assert source_counts == [1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert prediction.anchor_acc.shape == (1, 2, 4, 3)
    assert torch.isfinite(prediction.anchor_acc).all()

@param('use_linear_attn', (False, True))
@param('variable_point_lens', (False, True))
def test_pointnet_linear_attn(
    use_linear_attn,
    variable_point_lens
):
    from rigidformer.rigidformer import PointNet

    features = torch.randn(2, 2, 64, 16)
    pos = torch.randn(2, 2, 64, 3)

    net = PointNet(
        dim = 16,
        dim_out = 32,
        use_linear_attn = use_linear_attn,
        linear_attn_dim_head = 16,
        linear_attn_heads = 4
    )

    kwargs = dict()
    if variable_point_lens:
        from torch_einops_utils import lens_to_mask
        point_lens = torch.tensor([[32, 64], [64, 50]])
        kwargs['mask'] = lens_to_mask(point_lens, max_len = 64)

    out = net(features, pos, **kwargs)

    assert out.shape == (2, 2, 32)

    out.sum().backward()

def test_platonic_transformer():
    from rigidformer.platonic_transformer import PlatonicTransformer

    features = torch.randn(2, 2, 64, 16)
    pos = torch.randn(2, 2, 64, 3)

    net = PlatonicTransformer(dim = 16, dim_out = 32)
    out = net(features, pos)

    assert out.shape == (2, 2, 32)
    out.sum().backward()

def test_platonic_transformer_invariance():
    from rigidformer.platonic_transformer import PlatonicTransformer
    from torch_einops_utils import lens_to_mask
    from scipy.spatial.transform import Rotation

    # check continuous rotation invariance

    net = PlatonicTransformer(dim = 16, dim_out = 32).eval()

    features = torch.randn(2, 2, 64, 16)
    pos = torch.randn(2, 2, 64, 3)

    # variable length points

    point_lens = torch.tensor([[32, 64], [64, 50]])
    mask = lens_to_mask(point_lens, max_len = 64)

    with torch.no_grad():
        out1 = net(features, pos, mask = mask)

        # apply random 3d rotation from the discrete platonic group

        from rigidformer.platonic_transformer import TETRAHEDRON_ROTATIONS
        rot = TETRAHEDRON_ROTATIONS[torch.randint(0, 12, (1,)).item()]
        pos_rotated = pos @ rot.T

        out2 = net(features, pos_rotated, mask = mask)

    assert torch.allclose(out1, out2, atol = 1e-4)
