from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_paper_rotation_augmentation_uses_one_discrete_angle_for_entire_batch():
    from rigidformer import apply_rigidformer_rotation_augmentation

    positions = torch.tensor([
        [1., 0., 2.],
        [0., 1., -3.],
        [2., 1., 4.]
    ]).reshape(1, 1, 1, 3, 3).expand(2, 8, 4, -1, -1).clone()

    augmentation = apply_rigidformer_rotation_augmentation(
        positions,
        selected_angle_degrees = 90,
        apply_rotation = True
    )
    expected = positions[..., [1, 0, 2]].clone()
    expected[..., 0] *= -1.

    assert bool(augmentation.applied)
    assert augmentation.angle_degrees.item() == 90
    assert torch.allclose(augmentation.object_positions, expected, atol = 1e-6)
    assert torch.allclose(
        torch.cdist(augmentation.object_positions, augmentation.object_positions),
        torch.cdist(positions, positions),
        atol = 1e-6
    )
    assert torch.equal(augmentation.object_positions[..., 2], positions[..., 2])


def test_paper_rotation_augmentation_samples_only_disclosed_angles_and_can_skip():
    from rigidformer import apply_rigidformer_rotation_augmentation

    generator = torch.Generator().manual_seed(0)
    positions = torch.randn(1, 8, 2, 4, 3)
    sampled_angles = []

    for _ in range(100):
        augmentation = apply_rigidformer_rotation_augmentation(
            positions,
            probability = 1.,
            generator = generator
        )
        sampled_angles.append(augmentation.angle_degrees.item())

    assert all(5 <= angle <= 355 and angle % 5 == 0 for angle in sampled_angles)

    skipped = apply_rigidformer_rotation_augmentation(
        positions,
        probability = 0.,
        generator = generator
    )
    assert not bool(skipped.applied)
    assert skipped.angle_degrees.item() == 0
    assert torch.equal(skipped.object_positions, positions)
    assert torch.equal(skipped.rotation_matrix, torch.eye(3))


def test_paper_object_permutation_keeps_every_object_field_aligned():
    from rigidformer import apply_rigidformer_object_permutation_augmentation

    object_ids = torch.tensor([10, 20, 30])
    sample = dict(
        object_positions = object_ids[None, :, None, None].expand(8, -1, 2, 3).clone(),
        object_velocities = object_ids[None, :, None].expand(8, -1, 3).clone(),
        vertex_properties = object_ids[:, None].expand(-1, 4).clone(),
        physics_parameters = object_ids[:, None].expand(-1, 2).clone(),
        anchor_indices = object_ids[:, None].clone(),
        object_point_lens = object_ids.clone(),
        pointnet_fps_indices = (
            object_ids[:, None].clone(),
            object_ids[:, None].clone() + 1
        ),
        physical_dt = torch.tensor(.1),
        step_code = torch.tensor(5)
    )
    permutation = torch.tensor([2, 0, 1])
    expected_ids = object_ids[permutation]
    augmented = apply_rigidformer_object_permutation_augmentation(
        sample,
        permutation = permutation
    )

    assert torch.equal(augmented['object_positions'][0, :, 0, 0], expected_ids)
    assert torch.equal(augmented['object_velocities'][0, :, 0], expected_ids)
    assert torch.equal(augmented['vertex_properties'][:, 0], expected_ids)
    assert torch.equal(augmented['physics_parameters'][:, 0], expected_ids)
    assert torch.equal(augmented['anchor_indices'][:, 0], expected_ids)
    assert torch.equal(augmented['object_point_lens'], expected_ids)
    assert torch.equal(augmented['pointnet_fps_indices'][0][:, 0], expected_ids)
    assert torch.equal(augmented['pointnet_fps_indices'][1][:, 0], expected_ids + 1)
    assert torch.equal(augmented['physical_dt'], sample['physical_dt'])
    assert torch.equal(augmented['step_code'], sample['step_code'])

    unchanged = apply_rigidformer_object_permutation_augmentation(
        sample,
        probability = 0.,
        generator = torch.Generator().manual_seed(0)
    )
    assert torch.equal(unchanged['object_positions'], sample['object_positions'])
    assert torch.equal(unchanged['vertex_properties'], sample['vertex_properties'])


def test_training_window_sampler_uses_one_stride_for_all_eight_frames():
    from rigidformer import sample_rigidformer_training_windows

    trajectories = torch.arange(2 * 80, dtype = torch.float32).reshape(2, 80, 1, 1, 1)
    trajectories = trajectories.expand(-1, -1, -1, -1, 3).clone()

    window = sample_rigidformer_training_windows(
        trajectories,
        base_physical_dt = torch.tensor([.01, .02]),
        selected_step_codes = torch.tensor([1, 10]),
        start_indices = torch.tensor([3, 5])
    )

    assert window.object_positions.shape == (2, 8, 1, 1, 3)
    assert torch.equal(
        window.object_positions[0, :, 0, 0, 0],
        torch.arange(3, 11, dtype = torch.float32)
    )
    assert torch.equal(
        window.object_positions[1, :, 0, 0, 0],
        torch.arange(85, 156, 10, dtype = torch.float32)
    )
    assert torch.equal(window.step_code, torch.tensor([1, 10]))
    assert torch.allclose(window.physical_dt, torch.tensor([.01, .2]))


def test_training_window_sampler_draws_near_uniform_paper_steps():
    from rigidformer import sample_rigidformer_training_windows

    generator = torch.Generator().manual_seed(0)
    trajectories = torch.zeros(6000, 71, 1, 1, 3)
    window = sample_rigidformer_training_windows(
        trajectories,
        base_physical_dt = 1. / 60.,
        generator = generator
    )

    counts = torch.stack([(window.step_code == step).sum() for step in (1, 5, 10)])
    assert torch.all((counts - 2000).abs() < 150)
    assert torch.all(window.start_indices >= 0)
    assert torch.all(
        window.start_indices + 7 * window.step_code < trajectories.shape[1]
    )


class _RecordingDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.25))
        self.calls = []

    def forward(
        self,
        *,
        object_pos,
        object_pos_prev,
        object_pos_next,
        object_first_frame_pos,
        anchor_indices,
        **kwargs
    ):
        self.calls.append(dict(
            previous = object_pos_prev,
            current = object_pos,
            previous_gt = kwargs['object_pos_prev_gt'],
            current_gt = kwargs['object_pos_gt'],
            reference = object_first_frame_pos,
            anchor_indices = anchor_indices,
            physical_dt = kwargs['physical_dt'],
            step_code = kwargs['step_code']
        ))

        predicted_positions = 2. * object_pos - object_pos_prev + self.offset
        step_loss = (predicted_positions - object_pos_next).square().mean()
        losses = SimpleNamespace(
            acceleration = step_loss * .5,
            position = step_loss * .05
        )

        if anchor_indices is None:
            anchor_indices = torch.tensor(
                [[[0, 1, 2, 3]]],
                device = object_pos.device
            )

        return (
            step_loss,
            losses,
            SimpleNamespace(object_pos_next = predicted_positions),
            SimpleNamespace(anchor_indices = anchor_indices)
        )


def test_sequence_wrapper_rotates_only_during_training():
    from rigidformer import RigidformerSequenceTrainingWrapper

    positions = torch.zeros(1, 8, 1, 4, 3)
    positions[..., 0] = 1.
    batch = dict(
        object_positions = positions,
        physical_dt = torch.tensor([.1]),
        step_code = torch.tensor([5]),
        vertex_properties = torch.zeros(1, 1, 3)
    )

    training_dynamics = _RecordingDynamics()
    training_wrapper = RigidformerSequenceTrainingWrapper(
        training_dynamics,
        rotation_probability = 1.
    )
    training_output = training_wrapper(**batch)

    assert not torch.allclose(training_output.target_positions, positions)
    assert torch.equal(training_output.target_positions[..., 2], positions[..., 2])

    evaluation_dynamics = _RecordingDynamics()
    evaluation_wrapper = RigidformerSequenceTrainingWrapper(
        evaluation_dynamics,
        rotation_probability = 1.
    ).eval()
    evaluation_output = evaluation_wrapper(**batch)

    assert torch.equal(evaluation_output.target_positions, positions)


def test_sequence_wrapper_reuses_explicit_rotation_across_micro_batches():
    from rigidformer import RigidformerSequenceTrainingWrapper

    positions = torch.zeros(1, 8, 1, 4, 3)
    positions[..., 0] = 1.
    batch = dict(
        object_positions = positions,
        physical_dt = torch.tensor([.1]),
        step_code = torch.tensor([5]),
        vertex_properties = torch.zeros(1, 1, 3),
        rotation_apply = torch.tensor(True),
        rotation_angle_degrees = torch.tensor(45)
    )
    wrapper = RigidformerSequenceTrainingWrapper(_RecordingDynamics())

    first = wrapper(**batch)
    second = wrapper(**batch)

    assert torch.equal(first.target_positions, second.target_positions)
    assert not torch.equal(first.target_positions, positions)


def test_t8_training_is_closed_loop_reuses_reference_and_anchors_and_averages_time():
    from rigidformer import RigidformerSequenceTrainingWrapper

    dynamics = _RecordingDynamics()
    wrapper = RigidformerSequenceTrainingWrapper(
        dynamics,
        rotation_augmentation = False
    )

    positions = torch.arange(8, dtype = torch.float32).reshape(1, 8, 1, 1, 1)
    positions = positions.expand(-1, -1, -1, 4, 3).clone()
    output = wrapper(
        object_positions = positions,
        physical_dt = torch.tensor([.1]),
        step_code = torch.tensor([5]),
        vertex_properties = torch.zeros(1, 1, 3)
    )

    assert len(dynamics.calls) == 6
    assert output.rollout_positions.shape == positions.shape
    assert output.per_step_total_losses.shape == (6,)
    assert output.per_step_losses.acceleration.shape == (6,)
    assert torch.allclose(output.loss, output.per_step_total_losses.mean())
    assert torch.allclose(
        output.losses.acceleration,
        output.per_step_losses.acceleration.mean()
    )
    assert torch.allclose(
        output.losses.position,
        output.per_step_losses.position.mean()
    )

    first_prediction = output.rollout_positions[:, 2]
    assert torch.allclose(dynamics.calls[1]['current'], first_prediction)
    assert not torch.allclose(dynamics.calls[1]['current'], positions[:, 2])

    for step_index, call in enumerate(dynamics.calls):
        assert torch.equal(call['reference'], positions[:, 0])
        assert torch.equal(call['previous_gt'], positions[:, step_index])
        assert torch.equal(call['current_gt'], positions[:, step_index + 1])
        assert torch.equal(call['physical_dt'], torch.tensor([.1]))
        assert torch.equal(call['step_code'], torch.tensor([5]))

    assert not torch.equal(dynamics.calls[1]['current'], dynamics.calls[1]['current_gt'])

    assert dynamics.calls[0]['anchor_indices'] is None
    for call in dynamics.calls[1:]:
        assert torch.equal(call['anchor_indices'], output.anchor_indices)

    output.loss.backward()
    assert dynamics.offset.grad is not None
    assert torch.isfinite(dynamics.offset.grad)
    assert dynamics.offset.grad.abs() > 0


def test_real_rigidformer_supports_t8_training_and_full_bptt():
    from rigidformer import Rigidformer, RigidformerSequenceTrainingWrapper

    torch.manual_seed(0)
    model = Rigidformer(
        dim = 24,
        dim_head = 6,
        arope_dim = 6,
        heads = 4,
        num_register_tokens = 2,
        object_self_attn_depth = 1,
        anchor_cross_attn_depth = 1,
        object_hidden_layers = (1,),
        num_anchors = 4,
        pointnet_vertex_dim = 32,
        pointnet_num_samples = (4, 4, 4),
        anchor_avp_dim = 16
    )
    wrapper = RigidformerSequenceTrainingWrapper(
        model,
        rotation_augmentation = False
    )

    initial = torch.randn(1, 1, 16, 3)
    velocity = torch.randn(1, 1, 16, 3) * .01
    positions = torch.stack([
        initial + time * velocity
        for time in range(8)
    ], dim = 1)

    output = wrapper(
        object_positions = positions,
        physical_dt = torch.tensor([.1]),
        step_code = torch.tensor([5]),
        vertex_properties = torch.randn(1, 1, 3)
    )

    assert output.rollout_positions.shape == positions.shape
    assert output.anchor_indices.shape == (1, 1, 4)
    assert torch.isfinite(output.loss)
    assert torch.allclose(
        output.loss,
        output.losses.acceleration + 10. * output.losses.position
    )

    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ('optimizer_step', 'expected_multiplier'),
    (
        (0, .1),
        (50, .55),
        (100, 1.),
        (3000, .01)
    )
)
def test_paper_learning_rate_schedule(optimizer_step, expected_multiplier):
    from rigidformer import rigidformer_learning_rate_multiplier

    multiplier = rigidformer_learning_rate_multiplier(
        optimizer_step,
        steps_per_epoch = 10
    )
    assert multiplier == pytest.approx(expected_multiplier)


def test_paper_optimizer_scheduler_and_training_step():
    from rigidformer import (
        RigidformerSequenceTrainingWrapper,
        build_rigidformer_optimizer_and_scheduler,
        rigidformer_training_step
    )

    dynamics = _RecordingDynamics()
    wrapper = RigidformerSequenceTrainingWrapper(
        dynamics,
        rotation_augmentation = False
    )
    optimizer, scheduler = build_rigidformer_optimizer_and_scheduler(
        wrapper,
        steps_per_epoch = 10
    )
    batch = dict(
        object_positions = torch.zeros(1, 8, 1, 4, 3),
        physical_dt = torch.tensor([.1]),
        step_code = torch.tensor([5]),
        vertex_properties = torch.zeros(1, 1, 3)
    )

    assert optimizer.defaults['betas'] == (.9, .999)
    assert optimizer.defaults['weight_decay'] == .01
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-5)

    output, gradient_norm = rigidformer_training_step(
        wrapper,
        batch,
        optimizer,
        gradient_clip_norm = 1.,
        scheduler = scheduler
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(gradient_norm)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1.09e-5)
