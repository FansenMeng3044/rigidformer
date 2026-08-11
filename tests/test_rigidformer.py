import torch

import pytest
param = pytest.mark.parametrize

@param('fps', (False, True))
@param('test_rand_steps', (False, True))
@param('attn_residual_learned_pooling', (False, True))
@param('variable_object_lens', (False, True))
@param('variable_point_lens', (False, True))
@param('anchor_self_attn', (False, True))
@param('use_platonic_transformer', (False, True))
def test_rigidformer(
    fps,
    test_rand_steps,
    attn_residual_learned_pooling,
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
        heads = 4,
        num_register_tokens = 2,
        object_self_attn_depth = 2,
        anchor_cross_attn_depth = 2,
        object_hidden_layers = (0, 2),
        attn_residual_learned_pooling = attn_residual_learned_pooling,
        anchor_self_attn = anchor_self_attn,
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
