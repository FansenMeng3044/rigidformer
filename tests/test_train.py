from pathlib import Path

import numpy as np
import torch


def write_archive(path: Path, *, frames = 8, objects = 1, points = 16):
    positions = np.random.default_rng(0).normal(
        size = (1, frames, objects, points, 3)
    ).astype(np.float32)
    properties = np.array([[[1., .4, .2]]] * objects, dtype = np.float32)
    properties = properties.reshape(1, objects, 3)
    np.savez(path, positions = positions, props = properties)
    return positions, properties


def variable_archive_arrays(*, frames = 8, max_objects = 3, max_points = 16):
    rng = np.random.default_rng(1)
    positions = np.full(
        (2, frames, max_objects, max_points, 3),
        10_000.,
        dtype = np.float32
    )
    properties = np.full((2, max_objects, 3), 20_000., dtype = np.float32)
    object_lens = np.array([3, 1], dtype = np.int64)
    point_lens = np.array([
        [16, 11, 7],
        [13, 0, 0]
    ], dtype = np.int64)

    for trajectory_index, num_objects in enumerate(object_lens):
        properties[trajectory_index, :num_objects] = rng.uniform(
            .1, 2., size = (num_objects, 3)
        )
        for object_index in range(num_objects):
            num_points = point_lens[trajectory_index, object_index]
            positions[
                trajectory_index,
                :,
                object_index,
                :num_points
            ] = rng.normal(size = (frames, num_points, 3))

    return positions, properties, object_lens, point_lens


def write_variable_archive(path: Path, *, directory = False):
    arrays = variable_archive_arrays()
    positions, properties, object_lens, point_lens = arrays

    if directory:
        path.mkdir()
        np.save(path / 'positions.npy', positions)
        np.save(path / 'props.npy', properties)
        np.save(path / 'object_lens.npy', object_lens)
        np.save(path / 'point_lens.npy', point_lens)
    else:
        np.savez(
            path,
            positions = positions,
            props = properties,
            object_lens = object_lens,
            point_lens = point_lens
        )

    return arrays


def test_ddp_training_entry_has_paper_run_defaults():
    from rigidformer.train import parse_args

    args = parse_args([
        '--train-data', 'train.npz',
        '--base-physical-dt', str(1. / 60.)
    ])

    assert args.epochs == 300
    assert args.batch_size_per_process == 18
    assert args.sequence_length == 8
    assert tuple(args.step_codes) == (1, 5, 10)
    assert args.dim == 768
    assert args.dim_head == 128
    assert args.arope_dim == 96
    assert args.num_register_tokens == 16
    assert args.pointnet_vertex_dim == 1024
    assert args.ddp_timeout_minutes == 10


def test_ddp_process_group_is_bound_to_local_cuda_device(monkeypatch):
    from types import SimpleNamespace
    from rigidformer.train import distributed_context

    calls = {}
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        'set_device',
        lambda device: calls.update(set_device = device)
    )
    monkeypatch.setattr(
        torch.distributed,
        'init_process_group',
        lambda **kwargs: calls.update(init = kwargs)
    )
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    context = distributed_context(SimpleNamespace(
        device = 'cuda',
        ddp_timeout_minutes = 10
    ))

    assert context[:4] == (True, 0, 1, 2)
    assert context[4] == torch.device('cuda', 1)
    assert calls['set_device'] == 1
    assert calls['init']['backend'] == 'nccl'
    assert calls['init']['device_id'] == torch.device('cuda', 1)
    assert calls['init']['timeout'].total_seconds() == 600


def test_trajectory_archive_dataset_returns_separate_physical_dt_and_step_code(tmp_path):
    from rigidformer.train import TrajectoryArchiveDataset

    path = tmp_path / 'train.npz'
    positions, properties = write_archive(path, frames = 36, objects = 1)
    dataset = TrajectoryArchiveDataset(
        path,
        base_physical_dt = 1. / 60.,
        sequence_length = 8,
        step_codes = (5,),
        samples_per_trajectory = 2,
        object_permutation_probability = 0.
    )
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample['object_positions'].shape == (8, 1, 16, 3)
    assert torch.equal(sample['vertex_properties'], torch.from_numpy(properties[0]))
    assert sample['physical_dt'] == torch.tensor(5. / 60.)
    assert sample['step_code'] == 5

    first_frame = torch.where(
        torch.tensor([
            np.array_equal(frame, sample['object_positions'][0].numpy())
            for frame in positions[0]
        ])
    )[0].item()
    assert torch.equal(
        sample['object_positions'],
        torch.from_numpy(positions[0, first_frame:first_frame + 36:5])
    )


def test_trajectory_archive_dataset_memory_maps_npy_directory(tmp_path):
    from rigidformer.train import TrajectoryArchiveDataset

    npy_directory = tmp_path / 'arrays'
    npy_directory.mkdir()
    positions = np.zeros((1, 8, 1, 16, 3), dtype = np.float32)
    properties = np.array([[[1., .4, .2]]], dtype = np.float32)
    np.save(npy_directory / 'positions.npy', positions)
    np.save(npy_directory / 'props.npy', properties)
    dataset = TrajectoryArchiveDataset(
        npy_directory,
        base_physical_dt = .1,
        step_codes = (1,),
        samples_per_trajectory = 1,
        object_permutation_probability = 0.
    )

    assert isinstance(dataset.positions, np.memmap)
    assert isinstance(dataset.properties, np.memmap)
    assert dataset[0]['object_positions'].shape == (8, 1, 16, 3)


def test_variable_archive_dataset_trims_to_trajectory_lengths(tmp_path):
    from rigidformer.train import TrajectoryArchiveDataset

    path = tmp_path / 'train.npz'
    positions, properties, object_lens, point_lens = write_variable_archive(path)
    dataset = TrajectoryArchiveDataset(
        path,
        base_physical_dt = .1,
        step_codes = (1,),
        samples_per_trajectory = 1,
        object_permutation_probability = 0.,
        require_length_metadata = True
    )
    first = dataset[0]
    second = dataset[1]

    assert dataset.has_length_metadata
    assert first['object_positions'].shape == (8, 3, 16, 3)
    assert first['object_lens'] == 3
    assert torch.equal(first['object_point_lens'], torch.tensor([16, 11, 7]))
    assert torch.equal(
        first['vertex_properties'],
        torch.from_numpy(properties[0, :object_lens[0]])
    )
    assert second['object_positions'].shape == (8, 1, 13, 3)
    assert second['object_lens'] == 1
    assert torch.equal(second['object_point_lens'], torch.tensor([13]))
    assert torch.equal(
        second['object_positions'],
        torch.from_numpy(positions[1, :, :1, :13])
    )


def test_variable_batch_collate_zeroes_all_invalid_archive_padding(tmp_path):
    from rigidformer.train import (
        TrajectoryArchiveDataset,
        collate_rigidformer_trajectory_batch
    )

    path = tmp_path / 'train.npz'
    write_variable_archive(path)
    dataset = TrajectoryArchiveDataset(
        path,
        base_physical_dt = .1,
        step_codes = (1,),
        samples_per_trajectory = 1,
        object_permutation_probability = 0.,
        require_length_metadata = True
    )
    batch = collate_rigidformer_trajectory_batch([dataset[0], dataset[1]])

    assert batch['object_positions'].shape == (2, 8, 3, 16, 3)
    assert batch['vertex_properties'].shape == (2, 3, 3)
    assert torch.equal(batch['object_lens'], torch.tensor([3, 1]))
    assert torch.equal(
        batch['object_point_lens'],
        torch.tensor([[16, 11, 7], [13, 0, 0]])
    )
    assert torch.count_nonzero(batch['object_positions'][0, :, 1, 11:]) == 0
    assert torch.count_nonzero(batch['object_positions'][0, :, 2, 7:]) == 0
    assert torch.count_nonzero(batch['object_positions'][1, :, 0, 13:]) == 0
    assert torch.count_nonzero(batch['object_positions'][1, :, 1:]) == 0
    assert torch.count_nonzero(batch['vertex_properties'][1, 1:]) == 0
    assert not torch.any(batch['object_positions'] == 10_000.)
    assert not torch.any(batch['vertex_properties'] == 20_000.)


def test_variable_archive_directory_memory_maps_length_metadata(tmp_path):
    from rigidformer.train import TrajectoryArchiveDataset

    path = tmp_path / 'arrays'
    write_variable_archive(path, directory = True)
    dataset = TrajectoryArchiveDataset(
        path,
        base_physical_dt = .1,
        step_codes = (1,),
        samples_per_trajectory = 1,
        object_permutation_probability = 0.,
        require_length_metadata = True
    )

    assert isinstance(dataset.positions, np.memmap)
    assert isinstance(dataset.properties, np.memmap)
    assert isinstance(dataset.object_lens, np.memmap)
    assert isinstance(dataset.point_lens, np.memmap)


def test_variable_archive_rejects_incomplete_or_non_prefix_metadata(tmp_path):
    import pytest
    from rigidformer.train import TrajectoryArchiveDataset

    positions, properties, object_lens, point_lens = variable_archive_arrays()
    incomplete_path = tmp_path / 'incomplete.npz'
    np.savez(
        incomplete_path,
        positions = positions,
        props = properties,
        object_lens = object_lens
    )
    with pytest.raises(AssertionError, match = 'both object_lens and point_lens'):
        TrajectoryArchiveDataset(
            incomplete_path,
            base_physical_dt = .1,
            step_codes = (1,)
        )

    point_lens[1, 1] = 4
    invalid_path = tmp_path / 'invalid.npz'
    np.savez(
        invalid_path,
        positions = positions,
        props = properties,
        object_lens = object_lens,
        point_lens = point_lens
    )
    with pytest.raises(AssertionError, match = 'padded object entries'):
        TrajectoryArchiveDataset(
            invalid_path,
            base_physical_dt = .1,
            step_codes = (1,)
        )


def test_single_process_training_entry_writes_resumable_checkpoint(tmp_path):
    from rigidformer.train import main

    data_path = tmp_path / 'train.npz'
    output_dir = tmp_path / 'run'
    write_variable_archive(data_path)
    arguments = [
        '--train-data', str(data_path),
        '--output-dir', str(output_dir),
        '--base-physical-dt', '.1',
        '--step-codes', '1',
        '--epochs', '1',
        '--warmup-epochs', '0',
        '--batch-size-per-process', '2',
        '--samples-per-trajectory', '1',
        '--require-length-metadata',
        '--workers', '0',
        '--save-every', '1',
        '--log-every', '1',
        '--device', 'cpu',
        '--dim', '24',
        '--dim-head', '6',
        '--arope-dim', '6',
        '--heads', '4',
        '--num-register-tokens', '2',
        '--object-self-attn-depth', '1',
        '--anchor-cross-attn-depth', '1',
        '--object-hidden-layers', '1',
        '--num-anchors', '4',
        '--pointnet-vertex-dim', '32',
        '--pointnet-num-samples', '4', '4', '4',
        '--anchor-avp-dim', '16'
    ]
    result = main(arguments)

    checkpoint_path = output_dir / 'latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)

    assert result['epoch'] == 1
    assert result['global_step'] == 1
    assert result['checkpoint'] == checkpoint_path
    assert checkpoint['next_epoch'] == 1
    assert checkpoint['global_step'] == 1
    assert checkpoint['world_size'] == 1
    assert checkpoint['training_config']['epochs'] == 1
    assert checkpoint['model']
    assert checkpoint['optimizer']['state']
    assert checkpoint['scheduler']
    assert len(checkpoint['rng_by_rank']) == 1

    resumed = main([*arguments, '--resume', 'auto'])
    assert resumed['epoch'] == 1
    assert resumed['global_step'] == 1
