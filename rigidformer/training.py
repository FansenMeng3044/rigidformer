from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from math import cos, pi

import torch
from torch import nn, Tensor

from rigidformer.rigidformer import Losses


RigidformerTrainingWindow = namedtuple(
    'RigidformerTrainingWindow',
    (
        'object_positions',
        'delta_times',
        'step_sizes',
        'start_indices'
    )
)

RigidformerSequenceTrainingOutput = namedtuple(
    'RigidformerSequenceTrainingOutput',
    (
        'loss',
        'losses',
        'per_step_total_losses',
        'per_step_losses',
        'rollout_positions',
        'target_positions',
        'anchor_indices'
    )
)


@dataclass(frozen = True)
class RigidformerTrainingConfig:
    """Training settings disclosed in Appendix C of the paper."""

    sequence_length: int = 8
    step_sizes: tuple[int, ...] = (1, 5, 10)
    epochs: int = 300
    batch_size_per_process: int = 18
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 10
    warmup_start_factor: float = .1
    weight_decay: float = .01
    betas: tuple[float, float] = (.9, .999)
    gradient_clip_norm: float = 1.

    def __post_init__(self):
        assert self.sequence_length >= 3
        assert len(self.step_sizes) > 0
        assert all(isinstance(step, int) and step > 0 for step in self.step_sizes)
        assert len(set(self.step_sizes)) == len(self.step_sizes)
        assert self.epochs > self.warmup_epochs >= 0
        assert self.batch_size_per_process > 0
        assert self.learning_rate > self.min_learning_rate >= 0.
        assert 0. < self.warmup_start_factor <= 1.
        assert self.weight_decay >= 0.
        assert all(0. <= beta < 1. for beta in self.betas)
        assert self.gradient_clip_norm > 0.


def sample_rigidformer_training_windows(
    trajectories: Tensor,
    base_delta_times: float | Tensor,
    *,
    sequence_length = 8,
    step_sizes: tuple[int, ...] = (1, 5, 10),
    selected_step_sizes: Tensor | None = None,
    start_indices: Tensor | None = None,
    generator: torch.Generator | None = None
):
    """Sample batched T-state windows from native-rate trajectories.

    `trajectories` has shape (batch, frames, objects, points, 3). One stride is
    selected uniformly per batch element and held fixed across its whole
    sequence. For T=8 this returns two warmup states and six supervised rollout
    targets.
    """

    assert trajectories.ndim == 5 and trajectories.shape[-1] == 3
    assert sequence_length >= 3
    assert len(step_sizes) > 0
    assert all(isinstance(step, int) and step > 0 for step in step_sizes)
    assert len(set(step_sizes)) == len(step_sizes)

    batch, num_frames = trajectories.shape[:2]
    assert batch > 0
    device = trajectories.device

    step_options = torch.tensor(step_sizes, device = device, dtype = torch.long)

    if selected_step_sizes is None:
        option_indices = torch.randint(
            len(step_sizes),
            (batch,),
            device = device,
            generator = generator
        )
        selected_step_sizes = step_options[option_indices]
    else:
        selected_step_sizes = torch.as_tensor(
            selected_step_sizes,
            device = device,
            dtype = torch.long
        )
        assert selected_step_sizes.shape == (batch,)
        valid_steps = (selected_step_sizes[..., None] == step_options).any(dim = -1)
        assert torch.all(valid_steps), 'selected step sizes must come from step_sizes'

    max_start_indices = (
        num_frames - 1 - (sequence_length - 1) * selected_step_sizes
    )
    assert torch.all(max_start_indices >= 0), (
        'trajectory is too short for the selected step size and sequence length'
    )

    if start_indices is None:
        uniform = torch.rand(batch, device = device, generator = generator)
        start_indices = torch.floor(
            uniform * (max_start_indices + 1)
        ).long()
    else:
        start_indices = torch.as_tensor(
            start_indices,
            device = device,
            dtype = torch.long
        )
        assert start_indices.shape == (batch,)
        assert torch.all(start_indices >= 0)
        assert torch.all(start_indices <= max_start_indices)

    sequence_offsets = torch.arange(
        sequence_length,
        device = device,
        dtype = torch.long
    )
    frame_indices = (
        start_indices[:, None] +
        selected_step_sizes[:, None] * sequence_offsets[None, :]
    )
    batch_indices = torch.arange(batch, device = device)[:, None]
    object_positions = trajectories[batch_indices, frame_indices]

    base_delta_times = torch.as_tensor(
        base_delta_times,
        device = device,
        dtype = trajectories.dtype
    )

    if base_delta_times.ndim == 0:
        base_delta_times = base_delta_times.expand(batch)
    else:
        assert base_delta_times.shape == (batch,)

    assert torch.all(torch.isfinite(base_delta_times))
    assert torch.all(base_delta_times > 0), 'base_delta_times must be positive'

    delta_times = base_delta_times * selected_step_sizes.to(trajectories.dtype)

    return RigidformerTrainingWindow(
        object_positions = object_positions,
        delta_times = delta_times,
        step_sizes = selected_step_sizes,
        start_indices = start_indices
    )


class RigidformerSequenceTrainingWrapper(nn.Module):
    """Closed-loop T=8 training with two warmup states and no scheduled sampling.

    The paper discloses T=8 but not whether T counts states or targets. This
    implementation treats it as eight states: x_0 and x_1 are observed, while
    x_2 ... x_7 are supervised autoregressive predictions. Model predictions
    are always fed back after warmup and gradients remain connected through the
    full six-step rollout. The acceleration target remains the paper's
    three-frame ground-truth finite difference even after rollout inputs become
    predicted states.
    """

    def __init__(
        self,
        rigidformer: nn.Module,
        sequence_length = 8
    ):
        super().__init__()
        assert sequence_length >= 3
        self.rigidformer = rigidformer
        self.sequence_length = sequence_length

    def forward(
        self,
        object_positions: Tensor,
        delta_times: Tensor,
        *,
        vertex_properties: Tensor,
        anchor_indices = None,
        pointnet_fps_indices = None,
        object_point_lens = None,
        object_lens = None
    ):
        assert object_positions.ndim == 5 and object_positions.shape[-1] == 3
        assert object_positions.shape[1] == self.sequence_length
        assert delta_times.shape == (object_positions.shape[0],)
        assert torch.all(torch.isfinite(delta_times))
        assert torch.all(delta_times > 0)

        reference_positions = object_positions[:, 0]
        rollout_positions = [object_positions[:, 0], object_positions[:, 1]]

        step_total_losses = []
        step_acceleration_losses = []
        step_position_losses = []

        supervised_steps = zip(
            object_positions[:, :-2].unbind(dim = 1),
            object_positions[:, 1:-1].unbind(dim = 1),
            object_positions[:, 2:].unbind(dim = 1)
        )

        for gt_positions_prev, gt_positions, target_positions in supervised_steps:
            object_pos_prev, object_pos = rollout_positions[-2:]

            (
                step_loss,
                step_losses,
                prediction,
                intermediates
            ) = self.rigidformer(
                delta_times = delta_times,
                vertex_properties = vertex_properties,
                object_pos = object_pos,
                object_pos_prev = object_pos_prev,
                object_pos_next = target_positions,
                object_pos_prev_gt = gt_positions_prev,
                object_pos_gt = gt_positions,
                object_first_frame_pos = reference_positions,
                anchor_indices = anchor_indices,
                pointnet_fps_indices = pointnet_fps_indices,
                object_point_lens = object_point_lens,
                object_lens = object_lens,
                return_predictions_with_loss = True,
                return_intermediates = True
            )

            if anchor_indices is None:
                anchor_indices = intermediates.anchor_indices

            rollout_positions.append(prediction.object_pos_next)
            step_total_losses.append(step_loss)
            step_acceleration_losses.append(step_losses.acceleration)
            step_position_losses.append(step_losses.position)

        per_step_total_losses = torch.stack(step_total_losses)
        per_step_losses = Losses(
            acceleration = torch.stack(step_acceleration_losses),
            position = torch.stack(step_position_losses)
        )
        losses = Losses(
            acceleration = per_step_losses.acceleration.mean(),
            position = per_step_losses.position.mean()
        )

        return RigidformerSequenceTrainingOutput(
            loss = per_step_total_losses.mean(),
            losses = losses,
            per_step_total_losses = per_step_total_losses,
            per_step_losses = per_step_losses,
            rollout_positions = torch.stack(rollout_positions, dim = 1),
            target_positions = object_positions,
            anchor_indices = anchor_indices
        )


def rigidformer_learning_rate_multiplier(
    optimizer_step,
    *,
    steps_per_epoch,
    config = RigidformerTrainingConfig()
):
    """10-epoch linear warmup followed by cosine decay to the paper minimum."""

    assert steps_per_epoch > 0
    assert optimizer_step >= 0

    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    if warmup_steps > 0 and optimizer_step < warmup_steps:
        progress = optimizer_step / warmup_steps
        return config.warmup_start_factor + (
            1. - config.warmup_start_factor
        ) * progress

    decay_steps = total_steps - warmup_steps
    decay_progress = min(
        max(optimizer_step - warmup_steps, 0) / decay_steps,
        1.
    )
    min_factor = config.min_learning_rate / config.learning_rate
    cosine_factor = .5 * (1. + cos(pi * decay_progress))

    return min_factor + (1. - min_factor) * cosine_factor


def build_rigidformer_optimizer_and_scheduler(
    model: nn.Module,
    *,
    steps_per_epoch,
    config = RigidformerTrainingConfig()
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        betas = config.betas,
        weight_decay = config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda step: rigidformer_learning_rate_multiplier(
            step,
            steps_per_epoch = steps_per_epoch,
            config = config
        )
    )

    return optimizer, scheduler


def rigidformer_training_step(
    training_wrapper: RigidformerSequenceTrainingWrapper,
    batch,
    optimizer,
    *,
    gradient_clip_norm = 1.,
    scheduler = None
):
    """One standard training update with the paper's gradient clipping."""

    assert gradient_clip_norm > 0.

    optimizer.zero_grad(set_to_none = True)
    output = training_wrapper(**batch)
    output.loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        training_wrapper.rigidformer.parameters(),
        gradient_clip_norm
    )
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return output, gradient_norm
