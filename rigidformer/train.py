from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rigidformer.isaac_movi import IsaacMoviHDF5Dataset
from rigidformer.rigidformer import Rigidformer
from rigidformer.training import (
    RigidformerSequenceTrainingWrapper,
    RigidformerTrainingConfig,
    apply_rigidformer_object_permutation_augmentation,
    build_rigidformer_optimizer_and_scheduler
)


PAPER_MOVI_SAMPLING_PROTOCOL = 'one-random-window-per-trajectory-per-epoch'
GLOBAL_VALID_OBJECT_LOSS_PROTOCOL = 'global-valid-object-mean-v1'
MODEL_ARCHITECTURE_PROTOCOL = 'parameter-matched-inferred-v2'


class TrajectoryArchiveDataset(Dataset):
    """Random T-state windows from a trajectory archive.

    The archive must contain `positions` with shape
    `(trajectories, frames, objects, points, 3)` and `props` with shape
    `(trajectories, objects, 3)` in paper order `[mass, friction, restitution]`.
    Variable-size archives additionally contain integer `object_lens` with
    shape `(trajectories,)` and integer `point_lens` with shape
    `(trajectories, objects)`. Valid objects and points occupy prefixes; padded
    object entries have point length zero.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        base_physical_dt: float,
        sequence_length = 8,
        step_codes = (1, 5, 10),
        object_permutation_probability = .5,
        require_length_metadata = False
    ):
        path = Path(path)
        assert path.exists(), f'trajectory archive does not exist: {path}'
        assert base_physical_dt > 0.
        assert sequence_length >= 3
        assert len(step_codes) > 0
        assert all(isinstance(step, int) and step > 0 for step in step_codes)
        assert len(set(step_codes)) == len(step_codes)
        assert 0. <= object_permutation_probability <= 1.

        if path.is_dir():
            positions_path = path / 'positions.npy'
            properties_path = path / 'props.npy'
            object_lens_path = path / 'object_lens.npy'
            point_lens_path = path / 'point_lens.npy'
            assert positions_path.is_file() and properties_path.is_file(), (
                'archive directory must contain positions.npy and props.npy'
            )
            has_object_lens = object_lens_path.is_file()
            has_point_lens = point_lens_path.is_file()
            assert has_object_lens == has_point_lens, (
                'archive directory must contain both object_lens.npy and '
                'point_lens.npy, or neither'
            )
            self.positions = np.load(
                positions_path,
                mmap_mode = 'r',
                allow_pickle = False
            )
            self.properties = np.load(
                properties_path,
                mmap_mode = 'r',
                allow_pickle = False
            )
            object_lens = np.load(
                object_lens_path,
                mmap_mode = 'r',
                allow_pickle = False
            ) if has_object_lens else None
            point_lens = np.load(
                point_lens_path,
                mmap_mode = 'r',
                allow_pickle = False
            ) if has_point_lens else None
        else:
            assert path.suffix == '.npz', 'file archive must use the .npz suffix'
            with np.load(path, allow_pickle = False) as archive:
                assert 'positions' in archive and 'props' in archive
                has_object_lens = 'object_lens' in archive
                has_point_lens = 'point_lens' in archive
                assert has_object_lens == has_point_lens, (
                    'archive must contain both object_lens and point_lens, '
                    'or neither'
                )
                self.positions = np.asarray(archive['positions'], dtype = np.float32)
                self.properties = np.asarray(archive['props'], dtype = np.float32)
                object_lens = (
                    np.asarray(archive['object_lens'])
                    if has_object_lens else None
                )
                point_lens = (
                    np.asarray(archive['point_lens'])
                    if has_point_lens else None
                )

        assert self.positions.ndim == 5 and self.positions.shape[-1] == 3
        assert self.properties.ndim == 3 and self.properties.shape[-1] == 3
        assert self.positions.dtype == np.float32
        assert self.properties.dtype == np.float32
        assert self.positions.shape[0] == self.properties.shape[0]
        assert self.positions.shape[2] == self.properties.shape[1]
        assert self.positions.shape[0] > 0

        num_trajectories, _, max_objects, max_points, _ = self.positions.shape
        assert max_points >= 4, (
            'every rigid object needs at least four points for the paper\'s '
            'four-anchor Kabsch projection'
        )
        self.has_length_metadata = object_lens is not None
        assert self.has_length_metadata or not require_length_metadata, (
            'length metadata is required: provide object_lens and point_lens'
        )

        if not self.has_length_metadata:
            object_lens = np.full(
                (num_trajectories,), max_objects, dtype = np.int64
            )
            point_lens = np.full(
                (num_trajectories, max_objects), max_points, dtype = np.int64
            )
        else:
            assert np.issubdtype(object_lens.dtype, np.integer)
            assert np.issubdtype(point_lens.dtype, np.integer)
            assert object_lens.shape == (num_trajectories,)
            assert point_lens.shape == (num_trajectories, max_objects)
            assert np.all((object_lens >= 1) & (object_lens <= max_objects))
            object_indices = np.arange(max_objects)[None, :]
            valid_objects = object_indices < object_lens[:, None]
            assert np.all(
                (point_lens[valid_objects] >= 4) &
                (point_lens[valid_objects] <= max_points)
            ), 'every valid rigid object needs at least four points for four anchors'
            assert np.all(point_lens[~valid_objects] == 0), (
                'padded object entries in point_lens must be zero'
            )

        self.object_lens = object_lens
        self.point_lens = point_lens

        max_step_code = max(step_codes)
        required_frames = 1 + (sequence_length - 1) * max_step_code
        assert self.positions.shape[1] >= required_frames, (
            f'trajectories need at least {required_frames} frames for '
            f'T={sequence_length} and s={max_step_code}'
        )
        self.base_physical_dt = float(base_physical_dt)
        self.sequence_length = sequence_length
        self.step_codes = tuple(step_codes)
        self.object_permutation_probability = object_permutation_probability
        self.sampling_protocol = PAPER_MOVI_SAMPLING_PROTOCOL

    def __len__(self):
        # One shuffled visit to each training scene per epoch. Each visit draws
        # a fresh random T-state window below. Multiplying this length by an
        # arbitrary number changes the optimizer-step meaning of "300 epochs".
        return self.positions.shape[0]

    def __getitem__(self, index):
        trajectory_index = index
        trajectory = self.positions[trajectory_index]
        num_objects = int(self.object_lens[trajectory_index])
        point_lens = np.array(
            self.point_lens[trajectory_index, :num_objects],
            dtype = np.int64,
            copy = True
        )
        max_sample_points = int(point_lens.max())
        properties = self.properties[trajectory_index, :num_objects]
        step_code = self.step_codes[
            torch.randint(len(self.step_codes), ()).item()
        ]
        max_start = (
            trajectory.shape[0] - 1 -
            (self.sequence_length - 1) * step_code
        )
        start = torch.randint(max_start + 1, ()).item()
        frame_indices = start + np.arange(self.sequence_length) * step_code

        sample = dict(
            object_positions = torch.from_numpy(
                np.array(
                    trajectory[
                        frame_indices,
                        :num_objects,
                        :max_sample_points
                    ],
                    copy = True,
                    order = 'C'
                )
            ),
            vertex_properties = torch.from_numpy(
                np.array(properties, copy = True, order = 'C')
            ),
            physical_dt = torch.tensor(
                self.base_physical_dt * step_code,
                dtype = torch.float32
            ),
            step_code = torch.tensor(step_code, dtype = torch.long),
            object_lens = torch.tensor(num_objects, dtype = torch.long),
            object_point_lens = torch.from_numpy(point_lens)
        )

        return apply_rigidformer_object_permutation_augmentation(
            sample,
            probability = self.object_permutation_probability
        )


def collate_rigidformer_trajectory_batch(samples):
    """Dynamically pad a trajectory batch and preserve both length masks.

    Invalid source padding is never copied into the returned tensors. This is
    important for simulator archives whose backing arrays use arbitrary
    sentinel values outside the valid object and point prefixes.
    """

    assert len(samples) > 0
    required_keys = {
        'object_positions',
        'vertex_properties',
        'physical_dt',
        'step_code',
        'object_lens',
        'object_point_lens'
    }
    assert all(required_keys <= sample.keys() for sample in samples)

    first_positions = samples[0]['object_positions']
    first_properties = samples[0]['vertex_properties']
    has_anchor_indices = ['anchor_indices' in sample for sample in samples]
    assert all(has_anchor_indices) or not any(has_anchor_indices), (
        'anchor_indices must be present in every sample or no samples'
    )
    assert first_positions.ndim == 4 and first_positions.shape[-1] == 3
    assert first_properties.ndim == 2 and first_properties.shape[-1] == 3

    batch_size = len(samples)
    sequence_length = first_positions.shape[0]
    object_lens = torch.stack([
        sample['object_lens'] for sample in samples
    ]).to(dtype = torch.long)
    assert object_lens.shape == (batch_size,)
    assert torch.all(object_lens > 0)
    max_objects = int(object_lens.max().item())
    max_points = max(
        int(sample['object_point_lens'].max().item())
        for sample in samples
    )

    positions = first_positions.new_zeros(
        (batch_size, sequence_length, max_objects, max_points, 3)
    )
    properties = first_properties.new_zeros((batch_size, max_objects, 3))
    point_lens = torch.zeros(
        (batch_size, max_objects),
        dtype = torch.long,
        device = object_lens.device
    )
    anchor_indices = None

    if all(has_anchor_indices):
        first_anchor_indices = samples[0]['anchor_indices']
        assert first_anchor_indices.ndim == 2
        num_anchors = first_anchor_indices.shape[-1]
        anchor_indices = torch.zeros(
            (batch_size, max_objects, num_anchors),
            dtype = torch.long,
            device = object_lens.device
        )

    for batch_index, sample in enumerate(samples):
        sample_positions = sample['object_positions']
        sample_properties = sample['vertex_properties']
        sample_point_lens = sample['object_point_lens'].to(dtype = torch.long)
        num_objects = int(object_lens[batch_index].item())

        assert sample_positions.ndim == 4
        assert sample_positions.shape[0] == sequence_length
        assert sample_positions.shape[1] == num_objects
        assert sample_positions.shape[-1] == 3
        assert sample_positions.dtype == first_positions.dtype
        assert sample_positions.device == first_positions.device
        assert sample_properties.shape == (num_objects, 3)
        assert sample_properties.dtype == first_properties.dtype
        assert sample_properties.device == first_properties.device
        assert sample_point_lens.shape == (num_objects,)
        assert sample_point_lens.device == object_lens.device
        assert torch.all(sample_point_lens >= 4), (
            'every valid rigid object needs at least four points'
        )
        assert torch.all(sample_point_lens <= sample_positions.shape[2])

        if anchor_indices is not None:
            sample_anchor_indices = sample['anchor_indices'].to(dtype = torch.long)
            assert sample_anchor_indices.shape == (num_objects, num_anchors)
            assert sample_anchor_indices.device == object_lens.device
            assert torch.all(sample_anchor_indices >= 0)
            assert torch.all(sample_anchor_indices < sample_point_lens[:, None])
            anchor_indices[batch_index, :num_objects] = sample_anchor_indices

        properties[batch_index, :num_objects] = sample_properties
        point_lens[batch_index, :num_objects] = sample_point_lens
        for object_index, num_points_tensor in enumerate(sample_point_lens):
            num_points = int(num_points_tensor.item())
            positions[
                batch_index,
                :,
                object_index,
                :num_points
            ] = sample_positions[:, object_index, :num_points]

    batch = dict(
        object_positions = positions,
        vertex_properties = properties,
        physical_dt = torch.stack([
            sample['physical_dt'] for sample in samples
        ]),
        step_code = torch.stack([
            sample['step_code'] for sample in samples
        ]),
        object_lens = object_lens,
        object_point_lens = point_lens
    )

    if anchor_indices is not None:
        batch['anchor_indices'] = anchor_indices

    return batch


def parse_args(argv = None):
    parser = argparse.ArgumentParser(
        description = '300-epoch single-node DDP training for RigidFormer'
    )

    # data and run control

    parser.add_argument('--train-data', type = Path, required = True)
    parser.add_argument('--output-dir', type = Path, default = Path('runs/rigidformer'))
    parser.add_argument('--resume', type = str, default = None, help = "checkpoint path or 'auto'")
    parser.add_argument('--epochs', type = int, default = 300)
    parser.add_argument('--batch-size-per-process', type = int, default = 18)
    parser.add_argument('--workers', type = int, default = 8)
    parser.add_argument(
        '--base-physical-dt',
        type = float,
        default = None,
        help = (
            'saved-frame interval for .npz/.npy archives; HDF5 reads '
            'record_dt_s from its schema and rejects a conflicting value'
        )
    )
    parser.add_argument(
        '--require-length-metadata',
        action = 'store_true',
        help = 'reject archives without object_lens and point_lens'
    )
    parser.add_argument('--sequence-length', type = int, default = 8)
    parser.add_argument('--step-codes', type = int, nargs = '+', default = (1, 5, 10))
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--save-every', type = int, default = 5)
    parser.add_argument('--log-every', type = int, default = 20)
    parser.add_argument('--amp', choices = ('none', 'bf16', 'fp16'), default = 'none')
    parser.add_argument('--device', choices = ('auto', 'cuda', 'cpu'), default = 'auto')
    parser.add_argument('--ddp-timeout-minutes', type = int, default = 10)

    # optimizer and paper schedule

    parser.add_argument('--learning-rate', type = float, default = 1e-4)
    parser.add_argument('--min-learning-rate', type = float, default = 1e-6)
    parser.add_argument('--warmup-epochs', type = int, default = 10)
    parser.add_argument('--warmup-start-factor', type = float, default = .1)
    parser.add_argument('--weight-decay', type = float, default = .01)
    parser.add_argument('--gradient-clip-norm', type = float, default = 1.)

    # disclosed main model defaults; overrides make smoke tests possible

    parser.add_argument('--dim', type = int, default = 768)
    parser.add_argument('--dim-head', type = int, default = 128)
    parser.add_argument('--arope-dim', type = int, default = 96)
    parser.add_argument('--heads', type = int, default = 6)
    parser.add_argument('--dropout', type = float, default = .1)
    parser.add_argument('--num-register-tokens', type = int, default = 16)
    parser.add_argument('--object-self-attn-depth', type = int, default = 4)
    parser.add_argument('--anchor-cross-attn-depth', type = int, default = 4)
    parser.add_argument('--object-hidden-layers', type = int, nargs = '+', default = (0, 1, 2, 4))
    parser.add_argument('--num-anchors', type = int, default = 4)
    parser.add_argument('--pointnet-vertex-dim', type = int, default = 1024)
    parser.add_argument('--pointnet-num-samples', type = int, nargs = 3, default = (32, 32, 32))
    parser.add_argument('--pointnet-local-mlp-depth', type = int, default = 4)
    parser.add_argument('--anchor-avp-dim', type = int, default = 256)
    parser.add_argument('--anchor-query-hidden-dim', type = int, default = None)
    parser.add_argument('--anchor-query-hidden-depth', type = int, default = 2)
    parser.add_argument('--anchor-predictor-ff-depth', type = int, default = 6)

    args = parser.parse_args(argv)
    assert args.epochs > args.warmup_epochs >= 0
    assert args.batch_size_per_process > 0
    assert args.workers >= 0
    assert args.base_physical_dt is None or args.base_physical_dt > 0.
    assert args.save_every > 0
    assert args.log_every > 0
    assert args.ddp_timeout_minutes > 0
    assert args.pointnet_local_mlp_depth >= 2
    assert args.anchor_query_hidden_dim is None or args.anchor_query_hidden_dim > 0
    assert args.anchor_query_hidden_depth >= 1
    assert args.anchor_predictor_ff_depth >= 1
    return args


def distributed_context(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1

    if args.device == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_type = args.device

    if device_type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA was requested but is unavailable'
        assert local_rank < torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    if distributed:
        backend = 'nccl' if device.type == 'cuda' else 'gloo'
        dist.init_process_group(
            backend = backend,
            init_method = 'env://',
            timeout = timedelta(minutes = args.ddp_timeout_minutes),
            device_id = device if device.type == 'cuda' else None
        )
        assert dist.get_world_size() == world_size
        assert dist.get_rank() == rank

    return distributed, rank, local_rank, world_size, device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def validate_paper_scene_epoch_sharding(num_trajectories, world_size):
    assert num_trajectories > 0
    assert world_size > 0
    assert num_trajectories % world_size == 0, (
        'the paper scene-epoch sampling protocol requires the number of '
        'trajectories to be divisible by world size; otherwise '
        'DistributedSampler would duplicate or discard scenes'
    )


def build_training_data_loader(
    dataset,
    *,
    sampler,
    batch_size_per_process,
    workers,
    device,
    loader_generator
):
    """Build the paper scene-epoch loader without discarding its tail batch."""

    return DataLoader(
        dataset,
        batch_size = batch_size_per_process,
        shuffle = sampler is None,
        sampler = sampler,
        num_workers = workers,
        pin_memory = device.type == 'cuda',
        drop_last = False,
        persistent_workers = False,
        worker_init_fn = seed_worker,
        generator = loader_generator,
        collate_fn = collate_rigidformer_trajectory_batch
    )


def build_training_dataset(args):
    """Build either the legacy dense archive or strict Isaac-MOVi reader."""

    data_path = Path(args.train_data)
    if data_path.suffix.lower() in ('.h5', '.hdf5'):
        dataset = IsaacMoviHDF5Dataset(
            data_path,
            split = 'train',
            sequence_length = args.sequence_length,
            step_codes = tuple(args.step_codes),
            object_permutation_probability = .5,
            enforce_paper_split = True
        )
        if args.base_physical_dt is not None:
            assert np.isclose(
                args.base_physical_dt,
                dataset.record_dt_s,
                rtol = 0.,
                atol = 1e-12
            ), (
                '--base-physical-dt conflicts with HDF5 record_dt_s; do not '
                'pass the PhysX internal base_physics_dt_s here'
            )
        assert args.num_anchors == dataset.num_anchors, (
            'the supplied Isaac-MOVi-A cache contains four anchors per object'
        )
        return dataset

    assert args.base_physical_dt is not None, (
        '--base-physical-dt is required for .npz and .npy trajectory archives'
    )
    dataset = TrajectoryArchiveDataset(
        data_path,
        base_physical_dt = args.base_physical_dt,
        sequence_length = args.sequence_length,
        step_codes = tuple(args.step_codes),
        object_permutation_probability = .5,
        require_length_metadata = args.require_length_metadata
    )
    dataset.dataset_protocol = 'dense-world-point-trajectory-archive-v1'
    dataset.dataset_format = 'npy-directory' if data_path.is_dir() else 'npz'
    dataset.physical_dt_source = 'cli-base-physical-dt-times-step-code'
    dataset.split_protocol = None
    return dataset


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking = device.type == 'cuda')

    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)

    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]

    return value


def build_model(args):
    return Rigidformer(
        dim = args.dim,
        dim_head = args.dim_head,
        arope_dim = args.arope_dim,
        heads = args.heads,
        dropout = args.dropout,
        num_register_tokens = args.num_register_tokens,
        object_self_attn_depth = args.object_self_attn_depth,
        anchor_cross_attn_depth = args.anchor_cross_attn_depth,
        object_hidden_layers = tuple(args.object_hidden_layers),
        num_anchors = args.num_anchors,
        vertex_properties_dim = 3,
        pointnet_vertex_dim = args.pointnet_vertex_dim,
        pointnet_ratios = (1., .5, .25, .125),
        pointnet_num_samples = tuple(args.pointnet_num_samples),
        pointnet_local_mlp_depth = args.pointnet_local_mlp_depth,
        anchor_avp_dim = args.anchor_avp_dim,
        anchor_query_hidden_dim = args.anchor_query_hidden_dim,
        anchor_query_hidden_depth = args.anchor_query_hidden_depth,
        anchor_predictor_ff_depth = args.anchor_predictor_ff_depth
    )


def capture_rng_state(loader_generator):
    state = dict(
        python = random.getstate(),
        numpy = np.random.get_state(),
        torch = torch.get_rng_state(),
        loader_generator = loader_generator.get_state()
    )

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state, loader_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    loader_generator.set_state(state['loader_generator'])

    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def collect_rank_rng_states(distributed, world_size, loader_generator):
    local_state = capture_rng_state(loader_generator)

    if not distributed:
        return [local_state]

    states = [None] * world_size
    dist.all_gather_object(states, local_state)
    return states


def save_checkpoint_atomic(path, state):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def reduce_epoch_metrics(metric_sums, distributed):
    if distributed:
        dist.all_reduce(metric_sums, op = dist.ReduceOp.SUM)

    denominator = metric_sums[3].clamp(min = 1.)
    return dict(
        loss = (metric_sums[0] / denominator).item(),
        acceleration_loss = (metric_sums[1] / denominator).item(),
        position_loss = (metric_sums[2] / denominator).item()
    )


def build_valid_object_step_statistics(
    output,
    object_lens,
    *,
    distributed,
    world_size
):
    """Build globally-correct DDP loss scaling and detached metrics.

    The model loss is a local mean over valid anchors, objects, and rollout
    steps. Anchor count and rollout length are fixed across ranks, so the only
    rank-varying denominator is the number of valid objects. PyTorch DDP
    averages gradients across ranks; multiplying each local mean by

        world_size * local_valid_objects / global_valid_objects

    makes that averaged gradient exactly equal to the gradient of one masked
    mean over the global batch. Detached metric numerators use the same valid
    object weighting so step and epoch logs report that global objective too.
    """

    assert object_lens.ndim == 1
    assert object_lens.dtype in (torch.int32, torch.int64)
    assert torch.all(object_lens > 0)
    assert world_size > 0
    assert distributed == (world_size > 1)

    local_valid_objects = object_lens.sum(dtype = torch.float64)
    local_metric_sums = torch.stack((
        output.loss.detach().double() * local_valid_objects,
        output.losses.acceleration.detach().double() * local_valid_objects,
        output.losses.position.detach().double() * local_valid_objects,
        local_valid_objects
    ))
    global_step_metric_sums = local_metric_sums.clone()

    if distributed:
        assert dist.is_initialized()
        dist.all_reduce(global_step_metric_sums, op = dist.ReduceOp.SUM)

    global_valid_objects = global_step_metric_sums[3]
    assert torch.isfinite(global_step_metric_sums).all()
    assert global_valid_objects > 0

    ddp_loss_scale = (
        world_size * local_valid_objects / global_valid_objects
    ).to(dtype = output.loss.dtype)

    return local_metric_sums, global_step_metric_sums, ddp_loss_scale


def run_training(args):
    distributed = False

    try:
        distributed, rank, local_rank, world_size, device = distributed_context(args)
        is_main = rank == 0
        seed_everything(args.seed)

        if is_main:
            args.output_dir.mkdir(parents = True, exist_ok = True)
        if distributed:
            dist.barrier()

        training_config = RigidformerTrainingConfig(
            sequence_length = args.sequence_length,
            step_codes = tuple(args.step_codes),
            epochs = args.epochs,
            batch_size_per_process = args.batch_size_per_process,
            learning_rate = args.learning_rate,
            min_learning_rate = args.min_learning_rate,
            warmup_epochs = args.warmup_epochs,
            warmup_start_factor = args.warmup_start_factor,
            weight_decay = args.weight_decay,
            gradient_clip_norm = args.gradient_clip_norm
        )
        dataset = build_training_dataset(args)
        validate_paper_scene_epoch_sharding(len(dataset), world_size)
        sampler = DistributedSampler(
            dataset,
            num_replicas = world_size,
            rank = rank,
            shuffle = True,
            seed = args.seed,
            drop_last = False
        ) if distributed else None
        loader_generator = torch.Generator().manual_seed(args.seed + rank)
        loader = build_training_data_loader(
            dataset,
            sampler = sampler,
            batch_size_per_process = args.batch_size_per_process,
            workers = args.workers,
            device = device,
            loader_generator = loader_generator
        )
        assert len(loader) > 0, (
            'trajectory archive must contain at least one training scene'
        )

        rigidformer = build_model(args).to(device)
        training_model = RigidformerSequenceTrainingWrapper(
            rigidformer,
            sequence_length = args.sequence_length,
            rotation_augmentation = True,
            rotation_probability = .5
        ).to(device)
        optimizer, scheduler = build_rigidformer_optimizer_and_scheduler(
            training_model,
            steps_per_epoch = len(loader),
            config = training_config
        )

        amp_dtype = dict(
            none = torch.float32,
            bf16 = torch.bfloat16,
            fp16 = torch.float16
        )[args.amp]
        amp_enabled = args.amp != 'none'
        assert not amp_enabled or device.type == 'cuda', 'AMP requires CUDA'
        scaler = torch.amp.GradScaler(
            'cuda',
            enabled = args.amp == 'fp16'
        )

        start_epoch = 0
        global_step = 0
        resume_path = None

        if args.resume == 'auto':
            candidate = args.output_dir / 'latest.pt'
            resume_path = candidate if candidate.exists() else None
        elif args.resume is not None:
            resume_path = Path(args.resume)

        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location = 'cpu', weights_only = False)
            assert checkpoint.get('sampling_protocol') == dataset.sampling_protocol, (
                'checkpoint predates or uses a different epoch sampling '
                'protocol; restarting is required for comparable scheduling'
            )
            assert checkpoint.get('dataset_protocol') == dataset.dataset_protocol, (
                'checkpoint uses a different dataset interpretation; '
                'restarting is required'
            )
            assert checkpoint.get('loss_reduction_protocol') == GLOBAL_VALID_OBJECT_LOSS_PROTOCOL, (
                'checkpoint predates global valid-object DDP loss reduction; '
                'restarting is required to avoid changing the objective mid-run'
            )
            assert checkpoint.get('model_architecture_protocol') == MODEL_ARCHITECTURE_PROTOCOL, (
                'checkpoint predates or uses a different parameter-matched '
                'architecture; restarting is required'
            )
            assert checkpoint.get('steps_per_epoch') == len(loader), (
                'checkpoint optimizer schedule has a different steps_per_epoch'
            )
            assert checkpoint['world_size'] == world_size, (
                'exact DDP resume requires the same world size'
            )
            training_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['next_epoch']
            global_step = checkpoint['global_step']
            restore_rng_state(checkpoint['rng_by_rank'][rank], loader_generator)

        ddp_model: nn.Module = training_model
        if distributed:
            ddp_model = DistributedDataParallel(
                training_model,
                device_ids = [local_rank] if device.type == 'cuda' else None,
                output_device = local_rank if device.type == 'cuda' else None,
                broadcast_buffers = False,
                find_unused_parameters = False
            )

        if is_main:
            trajectories_per_process = len(dataset) // world_size
            last_batch_size_per_process = (
                trajectories_per_process % args.batch_size_per_process or
                args.batch_size_per_process
            )
            run_summary = dict(
                world_size = world_size,
                device = str(device),
                epochs = args.epochs,
                start_epoch = start_epoch,
                sampling_protocol = dataset.sampling_protocol,
                dataset_protocol = dataset.dataset_protocol,
                dataset_format = dataset.dataset_format,
                split_protocol = dataset.split_protocol,
                physical_dt_source = dataset.physical_dt_source,
                loss_reduction_protocol = GLOBAL_VALID_OBJECT_LOSS_PROTOCOL,
                model_architecture_protocol = MODEL_ARCHITECTURE_PROTOCOL,
                trajectories_per_epoch = len(dataset),
                steps_per_epoch = len(loader),
                total_optimizer_steps = args.epochs * len(loader),
                warmup_optimizer_steps = args.warmup_epochs * len(loader),
                batch_size_per_process = args.batch_size_per_process,
                global_batch_size = args.batch_size_per_process * world_size,
                last_batch_size_per_process = last_batch_size_per_process,
                last_global_batch_size = last_batch_size_per_process * world_size,
                parameters = sum(parameter.numel() for parameter in rigidformer.parameters()),
                amp = args.amp,
                length_metadata = dataset.has_length_metadata
            )
            print(json.dumps(run_summary), flush = True)

        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            ddp_model.train()
            metric_sums = torch.zeros(4, device = device, dtype = torch.float64)

            for batch_index, batch in enumerate(loader):
                batch = move_to_device(batch, device)
                optimizer.zero_grad(set_to_none = True)

                with torch.autocast(
                    device_type = device.type,
                    dtype = amp_dtype,
                    enabled = amp_enabled
                ):
                    output = ddp_model(**batch)

                assert torch.isfinite(output.loss), 'non-finite training loss'
                (
                    local_step_metric_sums,
                    global_step_metric_sums,
                    ddp_loss_scale
                ) = build_valid_object_step_statistics(
                    output,
                    batch['object_lens'],
                    distributed = distributed,
                    world_size = world_size
                )
                loss_for_backward = output.loss * ddp_loss_scale
                assert torch.isfinite(loss_for_backward), 'non-finite scaled DDP loss'

                scaler.scale(loss_for_backward).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    rigidformer.parameters(),
                    training_config.gradient_clip_norm
                )
                assert torch.isfinite(gradient_norm), 'non-finite gradient norm'
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                metric_sums += local_step_metric_sums
                global_step_metrics = reduce_epoch_metrics(
                    global_step_metric_sums,
                    distributed = False
                )

                if is_main and global_step % args.log_every == 0:
                    step_log = dict(
                        epoch = epoch + 1,
                        batch = batch_index + 1,
                        global_step = global_step,
                        loss = global_step_metrics['loss'],
                        acceleration_loss = global_step_metrics['acceleration_loss'],
                        position_loss = global_step_metrics['position_loss'],
                        gradient_norm = gradient_norm.detach().item(),
                        learning_rate = optimizer.param_groups[0]['lr']
                    )
                    print(json.dumps(step_log), flush = True)

            epoch_metrics = reduce_epoch_metrics(metric_sums, distributed)

            if is_main:
                epoch_log = dict(
                    epoch = epoch + 1,
                    global_step = global_step,
                    learning_rate = optimizer.param_groups[0]['lr'],
                    **epoch_metrics
                )
                print(json.dumps(epoch_log), flush = True)

            should_save = (
                (epoch + 1) % args.save_every == 0 or
                (epoch + 1) == args.epochs
            )

            if should_save:
                rng_by_rank = collect_rank_rng_states(
                    distributed,
                    world_size,
                    loader_generator
                )

                if is_main:
                    checkpoint = dict(
                        next_epoch = epoch + 1,
                        global_step = global_step,
                        world_size = world_size,
                        sampling_protocol = dataset.sampling_protocol,
                        dataset_protocol = dataset.dataset_protocol,
                        dataset_format = dataset.dataset_format,
                        split_protocol = dataset.split_protocol,
                        physical_dt_source = dataset.physical_dt_source,
                        loss_reduction_protocol = GLOBAL_VALID_OBJECT_LOSS_PROTOCOL,
                        model_architecture_protocol = MODEL_ARCHITECTURE_PROTOCOL,
                        steps_per_epoch = len(loader),
                        model = training_model.state_dict(),
                        optimizer = optimizer.state_dict(),
                        scheduler = scheduler.state_dict(),
                        scaler = scaler.state_dict(),
                        rng_by_rank = rng_by_rank,
                        training_config = asdict(training_config),
                        arguments = vars(args)
                    )
                    save_checkpoint_atomic(
                        args.output_dir / 'latest.pt',
                        checkpoint
                    )

                if distributed:
                    dist.barrier()

        return dict(
            epoch = args.epochs,
            global_step = global_step,
            checkpoint = args.output_dir / 'latest.pt'
        ) if is_main else None

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main(argv = None):
    args = parse_args(argv)
    return run_training(args)


if __name__ == '__main__':
    main()
