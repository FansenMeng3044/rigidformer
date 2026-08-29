from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rigidformer.training import (
    apply_rigidformer_object_permutation_augmentation
)


PAPER_ISAAC_MOVI_SPLIT_PROTOCOL = (
    'isaac-movi-paper-counts-960-120-120-v1'
)
ISAAC_MOVI_DATASET_PROTOCOL = (
    'rigidformer-isaac-hdf5-v1-world-points-record-dt-cached-anchors-v1'
)


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def resolve_isaac_movi_paper_splits(train, validation, test):
    """Resolve a fixed 960/120/120 protocol without random re-splitting.

    The supplied Isaac-MOVi-A archive stores 960 training and 240 validation
    scene IDs. Its validation IDs are in a frozen generation order. We retain
    the first 120 as validation and assign the final 120 to test. Archives that
    already contain 960/120/120 use those lists unchanged.
    """

    raw = {
        'train': np.asarray(train, dtype = np.int64),
        'validation': np.asarray(validation, dtype = np.int64),
        'test': np.asarray(test, dtype = np.int64)
    }
    assert all(ids.ndim == 1 for ids in raw.values())

    if tuple(len(raw[name]) for name in ('train', 'validation', 'test')) == (
        960, 120, 120
    ):
        resolved = raw
    else:
        assert tuple(
            len(raw[name]) for name in ('train', 'validation', 'test')
        ) == (960, 240, 0), (
            'paper-count Isaac-MOVi-A expects either stored 960/120/120 '
            'splits or the supplied archive\'s stored 960/240/0 split'
        )
        resolved = {
            'train': raw['train'],
            'validation': raw['validation'][:120],
            'test': raw['validation'][120:]
        }

    concatenated = np.concatenate(tuple(resolved.values()))
    assert concatenated.shape == (1200,)
    assert np.unique(concatenated).shape == (1200,), (
        'Isaac-MOVi-A splits must be disjoint and contain 1200 unique scenes'
    )
    return resolved


def quaternion_wxyz_to_rotation_matrix(quaternion):
    """Convert normalized wxyz quaternions to row-major rotation matrices."""

    quaternion = np.asarray(quaternion, dtype = np.float32)
    assert quaternion.shape[-1] == 4
    norms = np.linalg.norm(quaternion, axis = -1)
    assert np.all(np.isfinite(norms))
    assert np.allclose(norms, 1., rtol = 1e-4, atol = 1e-5), (
        'quaternion_wxyz must contain normalized quaternions'
    )
    quaternion = quaternion / norms[..., None]
    w, x, y, z = np.moveaxis(quaternion, -1, 0)

    rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype = np.float32)
    rotation[..., 0, 0] = 1. - 2. * (y * y + z * z)
    rotation[..., 0, 1] = 2. * (x * y - z * w)
    rotation[..., 0, 2] = 2. * (x * z + y * w)
    rotation[..., 1, 0] = 2. * (x * y + z * w)
    rotation[..., 1, 1] = 1. - 2. * (x * x + z * z)
    rotation[..., 1, 2] = 2. * (y * z - x * w)
    rotation[..., 2, 0] = 2. * (x * z - y * w)
    rotation[..., 2, 1] = 2. * (y * z + x * w)
    rotation[..., 2, 2] = 1. - 2. * (x * x + y * y)
    return rotation


class IsaacMoviHDF5Dataset(Dataset):
    """Lazy, worker-safe reader for the Isaac-MOVi-A HDF5 contract.

    Each dataset index corresponds to exactly one scene visit per epoch. A
    fresh step code and valid T-state window are sampled on every visit, which
    preserves the paper's optimizer-step meaning for 300 epochs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split = 'train',
        sequence_length = 8,
        step_codes = (1, 5, 10),
        object_permutation_probability = .5,
        enforce_paper_split = True
    ):
        self.path = Path(path).resolve()
        assert self.path.is_file(), f'HDF5 archive does not exist: {self.path}'
        assert self.path.suffix.lower() in ('.h5', '.hdf5')
        assert split in ('train', 'validation', 'test')
        assert sequence_length >= 3
        assert len(step_codes) > 0
        assert all(isinstance(step, int) and step > 0 for step in step_codes)
        assert len(set(step_codes)) == len(step_codes)
        assert 0. <= object_permutation_probability <= 1.

        with h5py.File(self.path, 'r') as archive:
            self._validate_root(archive, sequence_length, step_codes)
            stored_splits = {
                name: np.asarray(archive[f'splits/{name}'][:], dtype = np.int64)
                for name in ('train', 'validation', 'test')
            }
            if enforce_paper_split:
                splits = resolve_isaac_movi_paper_splits(**stored_splits)
                self.split_protocol = PAPER_ISAAC_MOVI_SPLIT_PROTOCOL
            else:
                splits = stored_splits
                self.split_protocol = 'stored-splits-v1'

            self.scene_ids = np.array(splits[split], dtype = np.int64, copy = True)
            assert len(self.scene_ids) > 0, f'{split} split is empty'
            assert np.unique(self.scene_ids).shape == self.scene_ids.shape
            for scene_id in self.scene_ids:
                assert f'{int(scene_id):06d}' in archive['scenes'], (
                    f'split references missing scene {int(scene_id):06d}'
                )

            self.record_dt_s = float(archive.attrs['record_dt_s'])
            self.base_physics_dt_s = float(archive.attrs['base_physics_dt_s'])

        self.split = split
        self.sequence_length = int(sequence_length)
        self.step_codes = tuple(step_codes)
        self.object_permutation_probability = float(
            object_permutation_probability
        )
        self.num_anchors = 4
        self.has_length_metadata = True
        self.sampling_protocol = 'one-random-window-per-trajectory-per-epoch'
        self.dataset_protocol = ISAAC_MOVI_DATASET_PROTOCOL
        self.dataset_format = 'isaac-movi-hdf5'
        self.physical_dt_source = 'record_dt_s-times-step-code'
        self._archive = None
        self._archive_pid = None

    @staticmethod
    def _validate_root(archive, sequence_length, step_codes):
        attrs = archive.attrs
        expected_text = {
            'schema_version': 'rigidformer-isaac-hdf5-v1',
            'dataset_name': 'Isaac-MOVi-A',
            'props_layout': 'mass_kg,dynamic_friction,restitution',
            'friction_definition': 'dynamic_friction',
            'quaternion_order': 'wxyz',
            'up_axis': 'Z',
            'length_unit': 'meter',
            'mass_unit': 'kilogram',
            'time_unit': 'second',
            'step_code_semantics': 'frame_stride_multiplier'
        }
        for key, expected in expected_text.items():
            assert key in attrs and _as_text(attrs[key]) == expected, (
                f'invalid {key}: expected {expected!r}'
            )

        assert int(attrs.get('write_complete', 0)) == 1, (
            'HDF5 archive is not marked write_complete'
        )
        assert int(attrs['sequence_length']) == sequence_length, (
            'requested sequence length disagrees with the HDF5 contract'
        )
        assert float(attrs['record_dt_s']) > 0.
        assert float(attrs['base_physics_dt_s']) > 0.
        assert float(attrs['record_dt_s']) >= float(attrs['base_physics_dt_s'])
        allowed_steps = set(
            np.asarray(attrs['allowed_step_codes'], dtype = np.int64).tolist()
        )
        assert set(step_codes) <= allowed_steps, (
            f'requested step codes {tuple(step_codes)} are not a subset of '
            f'the archive contract {tuple(sorted(allowed_steps))}'
        )
        assert 'scenes' in archive and 'splits' in archive

    def _get_archive(self):
        pid = os.getpid()
        if self._archive is None or self._archive_pid != pid:
            self.close()
            self._archive = h5py.File(self.path, 'r', swmr = True)
            self._archive_pid = pid
        return self._archive

    def close(self):
        archive = getattr(self, '_archive', None)
        if archive is not None:
            archive.close()
        self._archive = None
        self._archive_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_archive'] = None
        state['_archive_pid'] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        scene_id = int(self.scene_ids[index])
        scene = self._get_archive()['scenes'][f'{scene_id:06d}']
        num_frames = int(scene.attrs['num_frames'])
        num_objects = int(scene.attrs['num_objects'])
        assert int(scene.attrs['scene_id']) == scene_id
        assert num_frames > 0 and num_objects > 0

        step_code = self.step_codes[
            torch.randint(len(self.step_codes), ()).item()
        ]
        required_span = (self.sequence_length - 1) * step_code
        assert num_frames > required_span, (
            f'scene {scene_id:06d} has too few frames for T='
            f'{self.sequence_length}, s={step_code}'
        )
        start = torch.randint(num_frames - required_span, ()).item()
        frame_indices = start + np.arange(
            self.sequence_length,
            dtype = np.int64
        ) * step_code

        objects = scene['objects']
        states = scene['states']
        point_lens = np.asarray(objects['point_lens'][:], dtype = np.int64)
        assert point_lens.shape == (num_objects,)
        assert np.all(point_lens >= 4)
        max_points = int(point_lens.max())

        points_local = np.asarray(
            objects['points_local'][:, :max_points],
            dtype = np.float32
        )
        properties = np.asarray(objects['props'][:], dtype = np.float32)
        is_dynamic = np.asarray(objects['is_dynamic'][:], dtype = np.bool_)
        anchor_indices = np.asarray(
            scene['cache/anchor_indices'][:],
            dtype = np.int64
        )
        translations = np.asarray(
            states['translation_world'][frame_indices, :num_objects],
            dtype = np.float32
        )
        quaternions = np.asarray(
            states['quaternion_wxyz'][frame_indices, :num_objects],
            dtype = np.float32
        )
        time_s = np.asarray(scene['time_s'][:], dtype = np.float64)
        ground_plane = np.asarray(
            scene['environment/ground_plane'][:],
            dtype = np.float32
        )

        assert points_local.shape == (num_objects, max_points, 3)
        assert properties.shape == (num_objects, 3)
        assert np.all(np.isfinite(points_local))
        assert np.all(np.isfinite(properties))
        assert np.all(properties[:, 0] > 0.), 'mass must be positive'
        assert np.all((properties[:, 1:] >= 0.) & (properties[:, 1:] <= 1.))
        assert np.all(is_dynamic), 'RigidFormer training expects dynamic objects'
        assert anchor_indices.shape == (num_objects, self.num_anchors)
        assert np.all(anchor_indices >= 0)
        assert np.all(anchor_indices < point_lens[:, None])
        assert all(
            len(np.unique(indices)) == self.num_anchors
            for indices in anchor_indices
        ), 'each object must have four distinct cached anchors'
        for object_index, indices in enumerate(anchor_indices):
            anchors = points_local[object_index, indices]
            assert np.linalg.matrix_rank(anchors - anchors.mean(0)) >= 2, (
                'cached anchors must not be collinear'
            )
        assert np.array_equal(ground_plane, np.array([0., 0., 1., 0.])), (
            'the current RigidFormer ground feature requires the z=0 plane'
        )
        expected_times = np.arange(num_frames, dtype = np.float64) * self.record_dt_s
        assert time_s.shape == (num_frames,)
        assert np.allclose(time_s, expected_times, rtol = 0., atol = 1e-12)

        rotations = quaternion_wxyz_to_rotation_matrix(quaternions)
        world_positions = np.einsum(
            'toij,opj->topi',
            rotations,
            points_local,
            optimize = True
        )
        world_positions += translations[:, :, None, :]
        assert world_positions.dtype == np.float32
        assert np.all(np.isfinite(world_positions))

        sample = dict(
            object_positions = torch.from_numpy(
                np.ascontiguousarray(world_positions)
            ),
            vertex_properties = torch.from_numpy(
                np.ascontiguousarray(properties)
            ),
            physical_dt = torch.tensor(
                self.record_dt_s * step_code,
                dtype = torch.float32
            ),
            step_code = torch.tensor(step_code, dtype = torch.long),
            object_lens = torch.tensor(num_objects, dtype = torch.long),
            object_point_lens = torch.from_numpy(point_lens.copy()),
            anchor_indices = torch.from_numpy(anchor_indices.copy())
        )
        return apply_rigidformer_object_permutation_augmentation(
            sample,
            probability = self.object_permutation_probability
        )
