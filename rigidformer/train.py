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

from rigidformer.rigidformer import Rigidformer
from rigidformer.training import (
    RigidformerSequenceTrainingWrapper,
    RigidformerTrainingConfig,
    apply_rigidformer_object_permutation_augmentation,
    build_rigidformer_optimizer_and_scheduler
)


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
        samples_per_trajectory = 16,
        object_permutation_probability = .5,
        require_length_metadata = False
    ):
        path = Path(path)
        assert path.exists(), f'trajectory archive does not exist: {path}'
        assert base_physical_dt > 0.
        assert sequence_length >= 3
        assert samples_per_trajectory > 0
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
                (point_lens[valid_objects] >= 1) &
                (point_lens[valid_objects] <= max_points)
            )
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
        self.samples_per_trajectory = samples_per_trajectory
        self.object_permutation_probability = object_permutation_probability

    def __len__(self):
        return self.positions.shape[0] * self.samples_per_trajectory

    def __getitem__(self, index):
        trajectory_index = index % self.positions.shape[0]
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
        assert torch.all(sample_point_lens > 0)
        assert torch.all(sample_point_lens <= sample_positions.shape[2])

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

    return dict(
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
    parser.add_argument('--samples-per-trajectory', type = int, default = 16)
    parser.add_argument('--base-physical-dt', type = float, required = True)
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
    parser.add_argument('--anchor-avp-dim', type = int, default = 256)

    args = parser.parse_args(argv)
    assert args.epochs > args.warmup_epochs >= 0
    assert args.batch_size_per_process > 0
    assert args.workers >= 0
    assert args.samples_per_trajectory > 0
    assert args.save_every > 0
    assert args.log_every > 0
    assert args.ddp_timeout_minutes > 0
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
        anchor_avp_dim = args.anchor_avp_dim
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
        dataset = TrajectoryArchiveDataset(
            args.train_data,
            base_physical_dt = args.base_physical_dt,
            sequence_length = args.sequence_length,
            step_codes = tuple(args.step_codes),
            samples_per_trajectory = args.samples_per_trajectory,
            object_permutation_probability = .5,
            require_length_metadata = args.require_length_metadata
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas = world_size,
            rank = rank,
            shuffle = True,
            seed = args.seed,
            drop_last = True
        ) if distributed else None
        loader_generator = torch.Generator().manual_seed(args.seed + rank)
        loader = DataLoader(
            dataset,
            batch_size = args.batch_size_per_process,
            shuffle = sampler is None,
            sampler = sampler,
            num_workers = args.workers,
            pin_memory = device.type == 'cuda',
            drop_last = True,
            persistent_workers = False,
            worker_init_fn = seed_worker,
            generator = loader_generator,
            collate_fn = collate_rigidformer_trajectory_batch
        )
        assert len(loader) > 0, (
            'dataset is too small for batch_size_per_process with drop_last=True'
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
            run_summary = dict(
                world_size = world_size,
                device = str(device),
                epochs = args.epochs,
                start_epoch = start_epoch,
                steps_per_epoch = len(loader),
                batch_size_per_process = args.batch_size_per_process,
                global_batch_size = args.batch_size_per_process * world_size,
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
                scaler.scale(output.loss).backward()
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

                metric_sums += torch.stack((
                    output.loss.detach().double(),
                    output.losses.acceleration.detach().double(),
                    output.losses.position.detach().double(),
                    torch.ones((), device = device, dtype = torch.float64)
                ))

                if is_main and global_step % args.log_every == 0:
                    step_log = dict(
                        epoch = epoch + 1,
                        batch = batch_index + 1,
                        global_step = global_step,
                        loss = output.loss.detach().item(),
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
