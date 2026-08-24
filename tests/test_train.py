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


def test_single_process_training_entry_writes_resumable_checkpoint(tmp_path):
    from rigidformer.train import main

    data_path = tmp_path / 'train.npz'
    output_dir = tmp_path / 'run'
    write_archive(data_path)
    arguments = [
        '--train-data', str(data_path),
        '--output-dir', str(output_dir),
        '--base-physical-dt', '.1',
        '--step-codes', '1',
        '--epochs', '1',
        '--warmup-epochs', '0',
        '--batch-size-per-process', '1',
        '--samples-per-trajectory', '1',
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
