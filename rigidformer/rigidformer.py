from __future__ import annotations
from math import log
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, cdist, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, Linear, Parameter

import einx
from einops import einsum, rearrange, repeat, pack, reduce
from einops.layers.torch import Rearrange, Reduce

from torch_einops_utils import pack_with_inverse, maybe, pad_left_at_dim, lens_to_mask, masked_mean, pad_right_ndim_to_and_expand_as, batched_index_select

from x_mlps_pytorch import MLP

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from rigidformer.rotary_3d import RotaryEmbedding3D, apply_rotary_pos_emb
from rigidformer.knn import exact_knn_indices

import roma

# constants

INF = float('inf')

Predictions = namedtuple('Predictions', ('anchor_acc', 'object_pos_next'))

Intermediates = namedtuple('Intermediates', ('anchor_indices',))

Losses = namedtuple('Losses', ('acceleration', 'position'))

RigidformerRolloutStepSchedule = namedtuple(
    'RigidformerRolloutStepSchedule',
    ('physical_dt', 'step_code')
)

AnchorLossTerms = namedtuple(
    'AnchorLossTerms',
    ('raw_position', 'rigid_position', 'raw_acceleration', 'rigid_acceleration')
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def last(arr):
    return arr[-1]

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def masked_max(t, mask = None, dim = -2):
    """max pool while returning zeros for fully-masked rows"""

    if not exists(mask):
        return t.amax(dim = dim)

    expanded_mask = pad_right_ndim_to_and_expand_as(mask, t)
    mask_value = -torch.finfo(t.dtype).max
    pooled = t.masked_fill(~expanded_mask, mask_value).amax(dim = dim)
    has_value = mask.any(dim = -1)
    return torch.where(has_value[..., None], pooled, torch.zeros_like(pooled))

def reduce_anchor_smooth_l1(
    residual,
    object_mask = None
):
    """Paper C.2 reduction: sum xyz, average valid anchors/objects/samples."""

    assert residual.ndim == 4 and residual.shape[-1] == 3

    elementwise_loss = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        reduction = 'none',
        beta = 1.
    )
    per_anchor_loss = elementwise_loss.sum(dim = -1)

    if not exists(object_mask):
        return per_anchor_loss.mean()

    assert object_mask.shape == residual.shape[:2]
    anchor_mask = repeat(
        object_mask,
        'b no -> b no na',
        na = residual.shape[-2]
    )

    return masked_mean(per_anchor_loss, anchor_mask)

def rigidformer_anchor_losses(
    pred_acc,
    pred_anchor_pos_next,
    pred_anchor_pos_next_rigid,
    anchor_pos_next,
    anchor_pos,
    anchor_pos_prev,
    physical_dt_squared,
    object_mask = None,
    anchor_pos_gt = None,
    anchor_pos_prev_gt = None
):
    """Compute Eqs. 8-11 with separate rollout and GT histories.

    `anchor_pos` and `anchor_pos_prev` are the states consumed by the model and
    define the Verlet base for rigid predicted acceleration. Equations 10-11
    require the acceleration target to use three ground-truth states, supplied
    through `anchor_pos_gt`, `anchor_pos_prev_gt`, and `anchor_pos_next`.
    """

    assert physical_dt_squared.ndim == 1
    assert physical_dt_squared.shape[0] == pred_acc.shape[0]
    assert torch.all(physical_dt_squared > 0), 'physical_dt_squared must be positive'

    has_gt_current = exists(anchor_pos_gt)
    has_gt_previous = exists(anchor_pos_prev_gt)
    assert has_gt_current == has_gt_previous, (
        'ground-truth current and previous anchor positions must be supplied together'
    )

    anchor_pos_gt = default(anchor_pos_gt, anchor_pos)
    anchor_pos_prev_gt = default(anchor_pos_prev_gt, anchor_pos_prev)
    assert anchor_pos_gt.shape == anchor_pos_next.shape
    assert anchor_pos_prev_gt.shape == anchor_pos_next.shape

    physical_dt_squared = rearrange(physical_dt_squared, 'b -> b 1 1 1')
    prediction_verlet_base = 2 * anchor_pos - anchor_pos_prev
    ground_truth_verlet_base = 2 * anchor_pos_gt - anchor_pos_prev_gt

    target_acc = (
        anchor_pos_next - ground_truth_verlet_base
    ) / physical_dt_squared
    pred_acc_rigid = (
        pred_anchor_pos_next_rigid - prediction_verlet_base
    ) / physical_dt_squared

    # Appendix C.2 specifies normalizing multi-step residuals by delta-t^2
    # before SmoothL1. This keeps every configured integration step on the
    # same acceleration scale.

    raw_position_residual = (
        pred_anchor_pos_next - anchor_pos_next
    ) / physical_dt_squared
    rigid_position_residual = (
        pred_anchor_pos_next_rigid - anchor_pos_next
    ) / physical_dt_squared

    return AnchorLossTerms(
        raw_position = reduce_anchor_smooth_l1(
            raw_position_residual,
            object_mask
        ),
        rigid_position = reduce_anchor_smooth_l1(
            rigid_position_residual,
            object_mask
        ),
        raw_acceleration = reduce_anchor_smooth_l1(
            pred_acc - target_acc,
            object_mask
        ),
        rigid_acceleration = reduce_anchor_smooth_l1(
            pred_acc_rigid - target_acc,
            object_mask
        )
    )

# nearest neighbor displacement - accounts for ground plane

@torch.no_grad()
def nearest_neighbor_displacement(
    object_pos,     # (b no n 3)
    mask = None,    # (b no n)
    ground_z = 0.
):
    """for each vertex, displacement vector to the closest point on another object or the ground plane"""

    _, num_objects, num_points, _ = object_pos.shape
    total_points = num_objects * num_points

    # ground plane as default nearest surface - displacement is purely in z

    ground_z_disp = rearrange(ground_z - object_pos[..., 2], '... -> ... 1')
    ground_disp = F.pad(ground_z_disp, (2, 0))

    # flatten all points and compute pairwise distances per object against all points

    all_pos = rearrange(object_pos, 'b no n p -> b (no n) p')
    dists = cdist(object_pos, rearrange(all_pos, 'b m p -> b 1 m p'))  # (b, no, n, total_points)

    # mask out same-object points with block diagonal

    self_mask = torch.eye(num_objects, device = object_pos.device, dtype = torch.bool)
    self_mask = repeat(self_mask, 'i j -> 1 i 1 (j n)', n = num_points)
    dists.masked_fill_(self_mask, INF)

    # mask out invalid points

    if exists(mask):
        packed_mask = rearrange(mask, 'b no n -> b (no n)')
        dists = einx.where('b m, b no n m, -> b no n m', packed_mask, dists, INF)

    # concat ground distance and find nearest

    dists = cat((dists, ground_z_disp.abs()), dim = -1)
    other_dist, other_idx = dists.min(dim = -1)

    # get object displacement (clamp idx to safely avoid out of bounds if ground is nearest)

    safe_idx = rearrange(other_idx.clamp(max = total_points - 1), 'b no n -> b (no n)')
    safe_idx = pad_right_ndim_to_and_expand_as(safe_idx, all_pos)
    other_pos = all_pos.gather(1, safe_idx)
    other_disp = rearrange(other_pos, 'b (no n) p -> b no n p', no = num_objects) - object_pos

    # use ground displacement where ground was closest

    is_ground = other_idx == total_points
    return einx.where('b no n, b no n p, b no n p -> b no n p', is_ground, ground_disp, other_disp)

# naive fps

@torch.no_grad()
def naive_farthest_point_sample(
    positions,  # (... n d)
    num_points,
    mask = None # (... n)
):
    positions, inverse_pack = pack_with_inverse(positions, '* n p')
    device, batch, max_num_points, d = positions.device, *positions.shape

    (mask), (_) = maybe(pack_with_inverse, default = (None, None))(mask, '* n')

    num_points = min(num_points, max_num_points)

    if num_points <= 0:
        return inverse_pack(torch.empty((batch, 0), device = device, dtype = torch.long), '* na')

    sampled = torch.empty((batch, num_points), device = device, dtype = torch.long)

    # first one is random

    if exists(mask):
        first_rand_point = rearrange(mask.float().multinomial(1), '... 1 -> ...')
    else:
        first_rand_point = torch.randint(0, max_num_points, (batch,), device = device)

    sampled[:, 0] = first_rand_point

    # iterate through remaining, picking the farthest point from the remaining

    for i in range(num_points - 1):
        is_first = i == 0
        next_i = i + 1

        last_pos = batched_index_select(positions, sampled[:, i:next_i], dim = 1)

        next_distance = cdist(last_pos, positions)[:, 0]

        if is_first:
            min_distances = next_distance
        else:
            min_distances = torch.minimum(min_distances, next_distance)

        if exists(mask):
            min_distances.masked_fill_(~mask, -1.)

        sampled[:, next_i] = min_distances.argmax(dim = -1)

    return inverse_pack(sampled, '* na')

@torch.no_grad()
def deterministic_farthest_point_sample(
    positions,  # (... n d)
    num_points,
    mask = None # (... n)
):
    """Rigid-transform-invariant FPS seeded by the point farthest from the centroid.

    Exact distance ties are resolved by the input index. For fully permutation-stable
    datasets, precompute and store canonical FPS indices in the dataset.
    """

    positions, inverse_pack = pack_with_inverse(positions, '* n p')
    device, batch, max_num_points, _ = positions.device, *positions.shape

    if exists(mask):
        mask, _ = pack_with_inverse(mask, '* n')
    else:
        mask = torch.ones((batch, max_num_points), device = device, dtype = torch.bool)

    num_points = min(num_points, max_num_points)

    if num_points <= 0:
        empty = torch.empty((batch, 0), device = device, dtype = torch.long)
        return inverse_pack(empty, '* na')

    sampled = torch.zeros((batch, num_points), device = device, dtype = torch.long)

    mask_f = mask.to(positions.dtype)
    centroid = einsum(positions, mask_f, 'b n d, b n -> b d')
    centroid = centroid / mask_f.sum(dim = -1, keepdim = True).clamp(min = 1.)

    distance_to_centroid = (positions - centroid[:, None]).square().sum(dim = -1)
    distance_to_centroid.masked_fill_(~mask, -1.)
    sampled[:, 0] = distance_to_centroid.argmax(dim = -1)

    min_distances = torch.full(
        (batch, max_num_points),
        torch.finfo(positions.dtype).max,
        device = device,
        dtype = positions.dtype
    )

    for sample_index in range(num_points):
        last_pos = batched_index_select(
            positions,
            sampled[:, sample_index:sample_index + 1],
            dim = 1
        )
        next_distance = (positions - last_pos).square().sum(dim = -1)
        min_distances = torch.minimum(min_distances, next_distance)
        min_distances.masked_fill_(~mask, -1.)

        if sample_index + 1 < num_points:
            sampled[:, sample_index + 1] = min_distances.argmax(dim = -1)

    return inverse_pack(sampled, '* na')

# pointnet++

class PointNetSetAbstract(Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        num_points,
        num_samples,
        mlp_hidden_dim = None,
        use_linear_attn = False,
        linear_attn_dim_head = 16,
        linear_attn_heads = 4
    ):
        super().__init__()
        self.num_points = num_points
        self.num_samples = num_samples

        mlp_hidden_dim = default(mlp_hidden_dim, dim_out)

        self.mlp = MLP(dim + 3, dim_out, mlp_hidden_dim)

        self.linear_attn = TaylorSeriesLinearAttn(
            dim = dim_out,
            dim_head = linear_attn_dim_head,
            heads = linear_attn_heads,
            prenorm = True
        ) if use_linear_attn else None

    def forward(
        self,
        features, # (... n d)
        pos,      # (... n 3)
        mask = None
    ):
        pos, inverse_pack_pos = pack_with_inverse(pos, '* n p')
        features, inverse_pack_features = pack_with_inverse(features, '* n d')

        batch, n, _ = pos.shape
        _, _, dim = features.shape

        (packed_mask), (_) = maybe(pack_with_inverse, default = (None, None))(mask, '* n')

        # global pool

        if not exists(self.num_points) or self.num_points >= n:
            new_pos = masked_mean(pos, packed_mask, dim = -2, keepdim = True)

            grouped_pos = einx.subtract('b n p, b 1 p -> b 1 n p', pos, new_pos)
            grouped_features = repeat(features, 'b n d -> b 1 n d')

            grouped_features = cat((grouped_pos, grouped_features), dim = -1)

            new_features = self.mlp(grouped_features)

            if exists(self.linear_attn):
                attn_input, inverse_pack = pack_with_inverse(new_features, '* n d')
                attn_mask = packed_mask if exists(mask) else None

                attn_out = self.linear_attn(attn_input, mask = attn_mask)
                new_features = new_features + inverse_pack(attn_out)

            if exists(mask):
                mask_value = -torch.finfo(new_features.dtype).max
                new_features = einx.where('b n, b 1 n d, -> b 1 n d', packed_mask, new_features, mask_value)

            new_features = reduce(new_features, 'b 1 n d -> b 1 d', 'max')

            return inverse_pack_features(new_features, '* n d'), inverse_pack_pos(new_pos, '* n p')

        # fps

        sampled_indices = naive_farthest_point_sample(pos, self.num_points, mask = mask)

        new_pos = batched_index_select(pos, sampled_indices, dim = 1)

        # Exact GPU KNN on CUDA, with a CPU fallback for development/tests.

        knn_indices = exact_knn_indices(
            new_pos,
            pos,
            self.num_samples,
            support_mask = packed_mask
        )

        knn_indices_packed = rearrange(knn_indices, 'b m k -> b (m k)')

        grouped_pos = batched_index_select(pos, knn_indices_packed, dim = 1)
        grouped_pos = rearrange(grouped_pos, 'b (m k) p -> b m k p', m = self.num_points)
        grouped_pos = einx.subtract('b m k p, b m p -> b m k p', grouped_pos, new_pos)

        grouped_features = batched_index_select(features, knn_indices_packed, dim = 1)
        grouped_features = rearrange(grouped_features, 'b (m k) d -> b m k d', m = self.num_points)

        grouped_features = cat((grouped_pos, grouped_features), dim = -1)

        new_features = self.mlp(grouped_features)

        if exists(self.linear_attn):
            attn_input, inverse_pack = pack_with_inverse(new_features, '* k d')
            attn_out = self.linear_attn(attn_input)
            new_features = new_features + inverse_pack(attn_out)

        new_features = reduce(new_features, 'b m k d -> b m d', 'max')

        return inverse_pack_features(new_features, '* n d'), inverse_pack_pos(new_pos, '* n p')

class PointNet(Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        num_points: tuple[int | None, ...] = (128, 32, None),
        num_samples: tuple[int | None, ...] = (32, 16, None),
        expansion_factor: int = 2,
        use_linear_attn = False,
        linear_attn_dim_head = 16,
        linear_attn_heads = 4
    ):
        super().__init__()
        assert len(num_points) == len(num_samples)

        self.layers = ModuleList([])

        num_layers = len(num_points)
        dim_in = dim

        for ind, (layer_num_points, layer_num_samples) in enumerate(zip(num_points, num_samples)):
            is_last = ind == (num_layers - 1)

            dim_out_layer = dim_out if is_last else int(dim_in * expansion_factor)

            self.layers.append(PointNetSetAbstract(
                dim = dim_in,
                dim_out = dim_out_layer,
                num_points = layer_num_points,
                num_samples = layer_num_samples,
                use_linear_attn = use_linear_attn,
                linear_attn_dim_head = linear_attn_dim_head,
                linear_attn_heads = linear_attn_heads
            ))

            dim_in = dim_out_layer

    def forward(
        self,
        features,  # (... n d)
        pos,       # (... n 3)
        mask = None
    ):
        for layer in self.layers:
            features, pos = layer(features, pos, mask = mask)
            mask = None

        features = rearrange(features, '... 1 d -> ... d')
        return features

class SharedConv1dVertexBackbone(Module):
    """Shared per-object Conv1d MLP producing paper-width vertex features.

    RigidFormer specifies a 1024-channel Conv1d MLP backbone but does not
    publish its intermediate channel widths. The 1/4 and 1/2 widths below are
    explicit reproduction assumptions and remain configurable.
    """

    def __init__(
        self,
        dim_in,
        dim_out = 1024,
        hidden_dims: tuple[int, ...] | None = None
    ):
        super().__init__()

        hidden_dims = default(hidden_dims, (
            max(dim_out // 4, 8),
            max(dim_out // 2, 16)
        ))

        dims = (dim_in, *hidden_dims, dim_out)
        layers = []

        for layer_index, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Conv1d(layer_dim_in, layer_dim_out, 1))

            if layer_index < (len(dims) - 2):
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, features, mask = None):
        leading_shape = features.shape[:-2]
        num_points, dim = features.shape[-2:]

        features = features.reshape(-1, num_points, dim).transpose(1, 2)
        features = self.net(features).transpose(1, 2)
        features = features.reshape(*leading_shape, num_points, -1)

        if exists(mask):
            features = features.masked_fill(~mask[..., None], 0.)

        return features

class PaperPointNetSetAbstraction(Module):
    """One deterministic FPS + KNN hierarchy level for the paper-aligned encoder."""

    def __init__(
        self,
        dim,
        num_samples = 32
    ):
        super().__init__()
        self.num_samples = num_samples

        self.local_mlp = nn.Sequential(
            nn.Conv2d(dim + 3, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(
        self,
        features,       # (... n d)
        positions,      # (... n 3), current-frame geometry for local offsets
        sampling_pos,   # (... n 3), reference geometry for stable FPS and KNN
        target_lens,    # (...), desired number of centers for each object
        mask = None,    # (... n)
        center_indices = None
    ):
        leading_shape = features.shape[:-2]
        num_support, dim = features.shape[-2:]
        flat_batch = features.numel() // (num_support * dim)

        features = features.reshape(flat_batch, num_support, dim)
        positions = positions.reshape(flat_batch, num_support, 3)
        sampling_pos = sampling_pos.reshape(flat_batch, num_support, 3)
        target_lens = target_lens.reshape(flat_batch)

        if exists(mask):
            mask = mask.reshape(flat_batch, num_support)
        else:
            mask = torch.ones(
                (flat_batch, num_support),
                device = features.device,
                dtype = torch.bool
            )

        max_centers = max(int(target_lens.max().item()), 1)

        if not exists(center_indices):
            center_indices = deterministic_farthest_point_sample(
                sampling_pos,
                max_centers,
                mask = mask
            )
        else:
            center_indices = center_indices.reshape(flat_batch, -1)
            assert center_indices.shape[-1] >= max_centers
            center_indices = center_indices[:, :max_centers]

        center_mask = lens_to_mask(target_lens, max_len = max_centers)
        center_pos = batched_index_select(positions, center_indices, dim = 1)
        center_sampling_pos = batched_index_select(sampling_pos, center_indices, dim = 1)

        # Exact device-native KNN: CUDA tensors remain on GPU and supports are
        # streamed in chunks instead of allocating a full distance matrix.

        num_neighbors = min(self.num_samples, num_support)
        neighbor_indices = exact_knn_indices(
            center_sampling_pos,
            sampling_pos,
            num_neighbors,
            support_mask = mask
        )
        packed_neighbor_indices = rearrange(neighbor_indices, 'b c k -> b (c k)')

        grouped_features = batched_index_select(features, packed_neighbor_indices, dim = 1)
        grouped_features = rearrange(
            grouped_features,
            'b (c k) d -> b c k d',
            c = max_centers,
            k = num_neighbors
        )

        grouped_positions = batched_index_select(positions, packed_neighbor_indices, dim = 1)
        grouped_positions = rearrange(
            grouped_positions,
            'b (c k) p -> b c k p',
            c = max_centers,
            k = num_neighbors
        )
        relative_positions = grouped_positions - center_pos[:, :, None, :]

        grouped_mask = batched_index_select(mask, packed_neighbor_indices, dim = 1)
        grouped_mask = rearrange(
            grouped_mask,
            'b (c k) -> b c k',
            c = max_centers,
            k = num_neighbors
        )

        local_features = cat((grouped_features, relative_positions), dim = -1)
        local_features = rearrange(local_features, 'b c k d -> b d c k')
        local_features = self.local_mlp(local_features)
        local_features = rearrange(local_features, 'b d c k -> b c k d')

        mask_value = -torch.finfo(local_features.dtype).max
        local_features = local_features.masked_fill(~grouped_mask[..., None], mask_value)
        center_features = local_features.amax(dim = -2)

        has_neighbor = grouped_mask.any(dim = -1)
        center_features = torch.where(
            has_neighbor[..., None],
            center_features,
            torch.zeros_like(center_features)
        )
        center_features = center_features.masked_fill(~center_mask[..., None], 0.)
        center_pos = center_pos.masked_fill(~center_mask[..., None], 0.)
        center_sampling_pos = center_sampling_pos.masked_fill(~center_mask[..., None], 0.)

        center_features = center_features.reshape(*leading_shape, max_centers, dim)
        center_pos = center_pos.reshape(*leading_shape, max_centers, 3)
        center_sampling_pos = center_sampling_pos.reshape(*leading_shape, max_centers, 3)
        center_mask = center_mask.reshape(*leading_shape, max_centers)

        return center_features, center_pos, center_sampling_pos, center_mask

class PaperHierarchicalPointNet(Module):
    """Paper-aligned four-scale PointNet encoder.

    Confirmed paper dimensions:
      - 12D per-point state input
      - 1024D per-vertex backbone feature
      - scales 100%, 50%, 25%, 12.5%
      - 768D object token in the main model

    KNN count, intermediate Conv1d widths, activation, and fusion normalization
    are not disclosed by the paper and are explicit configurable assumptions.
    """

    def __init__(
        self,
        *,
        dim,
        dim_out,
        vertex_dim = 1024,
        ratios: tuple[float, ...] = (1., .5, .25, .125),
        num_samples: tuple[int, ...] = (32, 32, 32),
        backbone_hidden_dims: tuple[int, ...] | None = None
    ):
        super().__init__()
        assert ratios[0] == 1.
        assert len(ratios) == 4
        assert len(num_samples) == 3
        assert all(0. < ratio <= 1. for ratio in ratios)

        self.vertex_dim = vertex_dim
        self.ratios = ratios

        self.vertex_backbone = SharedConv1dVertexBackbone(
            dim_in = dim,
            dim_out = vertex_dim,
            hidden_dims = backbone_hidden_dims
        )

        self.hierarchy = ModuleList([
            PaperPointNetSetAbstraction(vertex_dim, one_num_samples)
            for one_num_samples in num_samples
        ])

        self.fuse = nn.Sequential(
            nn.RMSNorm(vertex_dim * len(ratios)),
            Linear(vertex_dim * len(ratios), dim_out)
        )

    def forward(
        self,
        features,          # (... n 12)
        pos,               # (... n 3)
        mask = None,       # (... n)
        reference_pos = None,
        fps_indices = None
    ):
        reference_pos = default(reference_pos, pos)

        if not exists(mask):
            mask = torch.ones(features.shape[:-1], device = features.device, dtype = torch.bool)

        original_lens = mask.sum(dim = -1)
        vertex_features = self.vertex_backbone(features, mask = mask)

        level_features = vertex_features
        level_pos = pos
        level_reference_pos = reference_pos
        level_mask = mask

        descriptors = [masked_max(level_features, level_mask)]
        fps_indices = default(fps_indices, (None,) * len(self.hierarchy))
        assert len(fps_indices) == len(self.hierarchy)

        for ratio, layer, one_level_indices in zip(
            self.ratios[1:],
            self.hierarchy,
            fps_indices
        ):
            target_lens = torch.ceil(original_lens.float() * ratio).long()
            target_lens = torch.where(
                original_lens > 0,
                target_lens.clamp(min = 1),
                torch.zeros_like(target_lens)
            )

            level_features, level_pos, level_reference_pos, level_mask = layer(
                level_features,
                level_pos,
                level_reference_pos,
                target_lens,
                mask = level_mask,
                center_indices = one_level_indices
            )
            descriptors.append(masked_max(level_features, level_mask))

        object_tokens = self.fuse(cat(descriptors, dim = -1))
        return object_tokens, vertex_features

# anchor vertex pooling

# basically a weighted aggregation with the l1norm on the negative exponentiated euclidean distance from anchor to object positions
# the learned sigma seems like a weak point in the scheme. seems like it should be scene dependent?

class AnchorVertexPool(Module):
    def __init__(
        self,
        init_sigma = 1.,
        learned_sigma = False
    ):
        super().__init__()

        log_sigma = log(init_sigma)

        self.log_sigma = nn.Parameter(tensor(log_sigma), requires_grad = learned_sigma)

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def forward(
        self,
        object_tokens,  # (b no n d)
        object_pos,     # (b no n p)
        anchor_indices, # (b no na)
        mask = None     # (b no n)
    ):

        anchor_pos = batched_index_select(object_pos, anchor_indices, dim = 2)

        object_pos, inverse_pack = pack_with_inverse(object_pos, '* n p')
        packed_anchor_pos, _ = pack_with_inverse(anchor_pos, '* n p')

        distance = cdist(packed_anchor_pos, object_pos)

        weights = (-distance / self.sigma).exp()

        packed_mask, _ = maybe(pack_with_inverse, default = (None, None))(mask, '* n')

        if exists(packed_mask):
            weights = einx.where('b n, b na n, -> b na n', packed_mask, weights, 0.)

        weights = l1norm(weights)

        weights = inverse_pack(weights)

        # aggregate

        anchor_tokens = einsum(object_tokens, weights, 'b no n d, b no na n -> b no na d')

        return anchor_tokens, anchor_pos

class AVPProjection(Module):
    """Paper-specified 1024 -> 256 -> 256 SiLU AVP projection."""

    def __init__(
        self,
        dim_in,
        dim_out = 256
    ):
        super().__init__()
        self.proj_in = Linear(dim_in, dim_out)
        self.proj_out = Linear(dim_out, dim_out)
        self.act = nn.SiLU()

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, features):
        return self.proj_out(self.act(self.proj_in(features)))

class AnchorQueryProjection(Module):
    """Project the paper's 271D anchor-state + AVP input to model width.

    The paper does not disclose this MLP's hidden width. Using model width is a
    minimal explicit assumption.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            Linear(dim_in, dim_out),
            nn.SiLU(),
            Linear(dim_out, dim_out)
        )

    def forward(self, features):
        return self.net(features)

# film

class FiLM(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma_beta = Linear(dim_cond, dim * 2, bias = False)
        nn.init.zeros_(self.to_gamma_beta.weight)

    def forward(
        self,
        tokens,
        cond
    ):
        normed = self.norm(tokens)

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)

        scaled = einx.multiply('b n d, b d', normed, gamma + 1.)
        return einx.add('b n d, b d', scaled, beta)

# block attention residuals

class BlockAttentionResidual(Module):
    """Exact inter-block AttnRes operator from Chen et al. (2026).

    Completed block sums and the current intra-block partial sum are values.
    A layer-specific pseudo-query attends to their RMS-normalized keys with a
    softmax over depth. The query starts at zero so training begins with the
    paper's uniform depth average.
    """

    def __init__(self, dim):
        super().__init__()
        self.query = Parameter(torch.zeros(dim))
        self.key_rmsnorm = nn.RMSNorm(dim)

    def forward(
        self,
        blocks: list[Tensor],
        partial_block: Tensor | None = None
    ):
        sources = [*blocks]

        if exists(partial_block):
            sources.append(partial_block)

        assert len(sources) > 0, 'Block AttnRes requires at least one depth source'

        values = stack(sources, dim = 0)
        keys = self.key_rmsnorm(values)

        logits = einsum(self.query, keys, 'd, s ... d -> s ...')
        weights = logits.softmax(dim = 0)

        return (weights[..., None] * values).sum(dim = 0)

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True,
        dropout = 0.
    ):
        super().__init__()
        assert 0. <= dropout < 1.

        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_queries_gates = Linear(dim, dim_inner * 2, bias = False)
        self.to_keys_values = Linear(dim, dim_inner * 2, bias = False)

        self.to_out = Linear(dim_inner, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # qk rmsnorm

        self.has_qk_rmsnorm = qk_rmsnorm

        self.qk_rmsnorm = nn.RMSNorm(dim_head, elementwise_affine = False)
        self.qk_rmsnorm_scales = nn.Parameter(torch.ones(2, heads, dim_head))

    def forward(
        self,
        tokens,
        context = None,
        rotary_pos_emb = None,
        context_rotary_pos_emb = None,
        mask = None
    ):

        context = default(context, tokens)

        queries, gates, keys, values = (
            *self.to_queries_gates(tokens).chunk(2, dim = -1),
            *self.to_keys_values(context).chunk(2, dim = -1)
        )

        queries, keys, values = (self.split_heads(t) for t in (queries, keys, values))

        if self.has_qk_rmsnorm:
            queries, keys = tuple(self.qk_rmsnorm(t) for t in (queries, keys))
            queries, keys = tuple(einx.multiply('b h n d, h d', t, scale) for t, scale in zip((queries, keys), self.qk_rmsnorm_scales))

        if exists(rotary_pos_emb):
            context_rotary_pos_emb = default(context_rotary_pos_emb, rotary_pos_emb)

            queries = apply_rotary_pos_emb(rotary_pos_emb, queries)
            keys = apply_rotary_pos_emb(context_rotary_pos_emb, keys)

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j') * self.scale

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        out = out * gates.sigmoid()
        return self.to_out(out)

class SwiGluFeedforward(Module):
    """Paper-configured SwiGLU feed-forward network.

    The reported 2.5x expansion denotes the gated hidden width itself, so the
    main D=768 model uses 1920 hidden channels before projecting back to D.
    """

    def __init__(
        self,
        dim,
        expansion_factor = 4.,
        dropout = 0.
    ):
        super().__init__()
        assert expansion_factor > 0.
        assert 0. <= dropout < 1.

        dim_inner = int(dim * expansion_factor)
        self.dim_inner = dim_inner

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens
    ):
        hiddens, gates = self.proj_in(tokens).chunk(2, dim = -1)

        hiddens = hiddens * F.silu(gates)

        return self.dropout(self.proj_out(hiddens))

# main class

class Rigidformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 128,
        heads = 6,
        ff_expansion = 2.5,
        dropout = .1,
        num_register_tokens = 16,
        object_self_attn_depth = 4,
        anchor_cross_attn_depth = 4,
        anchor_self_attn = False,
        num_anchors = 4,
        object_hidden_layers: tuple[int, ...] = (0, 1, 2, 4),  # the hidden object layer outputs that the anchor decoder cross attends to
        learned_object_hidden_layers = False, # learned softmax pooling over object decoder depths
        attn_residual_block_size = 4, # counts self-attention and FFN as separate AttnRes layers
        pos_loss_weight = 10.,
        acc_loss_weight = 1.,
        arope_dim = 96,
        arope_base = 10_000,
        anchor_vertex_pool_kwargs: dict = dict(
            learned_sigma = True
        ),
        vertex_properties_dim = 3,  # paper physics properties: [mass, friction, restitution]
        hierarchical_encoder: Module | None = None,
        pointnet_vertex_dim = 1024,
        pointnet_ratios: tuple[float, ...] = (1., .5, .25, .125),
        pointnet_num_samples: tuple[int, ...] = (32, 32, 32),
        pointnet_backbone_hidden_dims: tuple[int, ...] | None = None,
        anchor_avp_dim = 256,
        use_platonic_transformer = False,
        platonic_transformer_kwargs: dict = dict(
            depth = 2,
            heads = 4,
            dim_head = 32
        )
    ):
        super().__init__()

        assert 0. <= dropout < 1.

        self.vertex_properties_dim = vertex_properties_dim

        vertex_state_dim = 3 + 3 + 3 + vertex_properties_dim

        # vertex encoder

        self.uses_paper_pointnet = not exists(hierarchical_encoder) and not use_platonic_transformer

        if self.uses_paper_pointnet:
            self.vertex_encoder = None
            hierarchical_encoder = PaperHierarchicalPointNet(
                dim = vertex_state_dim,
                dim_out = dim,
                vertex_dim = pointnet_vertex_dim,
                ratios = pointnet_ratios,
                num_samples = pointnet_num_samples,
                backbone_hidden_dims = pointnet_backbone_hidden_dims
            )
            vertex_feature_dim = pointnet_vertex_dim
        else:
            # Backwards-compatible custom/experimental encoder path.
            self.vertex_encoder = MLP(vertex_state_dim, dim * 2, dim)
            vertex_feature_dim = dim

            if not exists(hierarchical_encoder):
                from rigidformer.platonic_transformer import PlatonicTransformer
                hierarchical_encoder = PlatonicTransformer(dim = dim, dim_out = dim, **platonic_transformer_kwargs)

        self.hierarchical_encoder = hierarchical_encoder

        # embedding

        self.anchor_vertex_pool = AnchorVertexPool(**anchor_vertex_pool_kwargs)

        # Paper predictor input: 15D anchor state (including [mass, friction, restitution])
        # concatenated with the 256D AVP feature, i.e. 271D in the main model.

        anchor_state_dim = 3 + vertex_state_dim
        self.anchor_avp = AVPProjection(vertex_feature_dim, anchor_avp_dim)
        self.anchor_query_fuse = AnchorQueryProjection(
            anchor_state_dim + anchor_avp_dim,
            dim
        )

        # rotary embeddings

        assert arope_dim <= dim_head, 'ARoPE dimension must not exceed the attention head dimension'
        assert arope_dim % 6 == 0, 'ARoPE dimension must be divisible by 6 for three-axis rotary pairs'

        self.arope_dim = arope_dim
        self.rope_3d = RotaryEmbedding3D(arope_dim, omega = arope_base)

        # object self attention related

        assert isinstance(attn_residual_block_size, int) and attn_residual_block_size > 0
        self.attn_residual_block_size = attn_residual_block_size

        layers = ModuleList([])

        for _ in range(object_self_attn_depth):
            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                dropout = dropout
            )

            ff = SwiGluFeedforward(
                dim = dim,
                expansion_factor = ff_expansion,
                dropout = dropout
            )

            attn_film = FiLM(dim, 2)

            # AttnRes defines attention and MLP as separate layers, each with
            # its own RMSNorm and zero-initialized pseudo-query.

            attn_residual = BlockAttentionResidual(dim)
            ff_residual = BlockAttentionResidual(dim)

            layers.append(ModuleList([attn_film, attn, ff, attn_residual, ff_residual]))

        self.self_attn_layers = layers
        self.object_final_attn_residual = BlockAttentionResidual(dim)

        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # anchor related

        self.num_anchors = num_anchors # if anchor_indices not passed in, will do naive fps

        self.learned_object_hidden_layers = learned_object_hidden_layers
        self.object_hidden_layers = object_hidden_layers

        if not learned_object_hidden_layers:
            assert object_self_attn_depth in object_hidden_layers, f'`object_hidden_layers` should attend to the output of the object transformer ({object_self_attn_depth})'
            assert all([0 <= l <= object_self_attn_depth for l in object_hidden_layers])
            assert len(object_hidden_layers) == anchor_cross_attn_depth, 'length of `object_hidden_layers` must be equal to the depth of the anchor cross attention transformer'

        layers = ModuleList([])

        for _ in range(anchor_cross_attn_depth):

            self_attn_film = FiLM(dim, 2) if anchor_self_attn else None
            self_attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                dropout = dropout
            ) if anchor_self_attn else None

            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                dropout = dropout
            )

            attn_film = FiLM(dim, 2)

            context_attn_residual = BlockAttentionResidual(dim) if learned_object_hidden_layers else None

            layers.append(ModuleList([self_attn_film, self_attn, attn_film, attn, context_attn_residual]))

        self.cross_attn_layers = layers

        # fuse the parallel multi-scale cross attention outputs (appendix G)

        self.cross_attn_fuse = nn.Sequential(
            nn.RMSNorm(anchor_cross_attn_depth * dim),
            Linear(anchor_cross_attn_depth * dim, dim, bias = False)
        )

        self.to_acc_pred = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, 3, bias = False)
        )

        self.pos_loss_weight = pos_loss_weight
        self.acc_loss_weight = acc_loss_weight

    def _build_arope_embeddings(self, anchor_pos):
        """Build paper ARoPE phases for anchors, objects, and registers."""

        anchor_rotary_pos_emb = self.rope_3d(anchor_pos)
        object_rotary_pos_emb = anchor_rotary_pos_emb.mean(dim = -2)

        # The paper defines register tokens as an unpositioned global
        # workspace. Zero phase is exactly the identity RoPE transform.

        object_rotary_pos_emb_with_registers = pad_left_at_dim(
            object_rotary_pos_emb,
            self.num_register_tokens,
            dim = -2,
            value = 0.
        )

        return anchor_rotary_pos_emb, object_rotary_pos_emb, object_rotary_pos_emb_with_registers

    def forward(
        self,
        *,
        physical_dt,                    # (b), elapsed simulator time in seconds
        step_code,                      # (b), dimensionless FiLM integration code s
        vertex_properties,              # (b no n 3) or (b no 3): [mass, friction, restitution]
        object_pos,                     # (b no n 3)
        object_pos_prev = None,         # (b no n 3)
        object_pos_next = None,         # (b no n 3)
        object_pos_prev_gt = None,      # (b no n 3), loss-only GT history
        object_pos_gt = None,           # (b no n 3), loss-only GT current state
        object_first_frame_pos = None,  # (b no n 3)
        anchor_indices = None,          # (b no na)
        pointnet_fps_indices = None,    # 3-tuple of nested hierarchy indices
        object_point_lens = None,       # (b no)
        object_lens = None,             # (b)
        return_predictions_with_loss = False,
        return_intermediates = False
    ):
        batch, max_num_objects = object_pos.shape[:2]

        assert exists(vertex_properties), 'vertex_properties must be passed in'
        assert vertex_properties.ndim in (3, 4), (
            'vertex_properties must have shape (batch, objects, properties) or '
            '(batch, objects, points, properties)'
        )
        assert vertex_properties.shape[:2] == (batch, max_num_objects)
        assert vertex_properties.shape[-1] == self.vertex_properties_dim, (
            f'vertex_properties must have last dimension {self.vertex_properties_dim}; '
            'the paper configuration expects [mass, friction, restitution]'
        )
        if vertex_properties.ndim == 4:
            assert vertex_properties.shape[2] == object_pos.shape[-2]

        object_mask = lens_to_mask(object_lens, max_len = max_num_objects) if exists(object_lens) else None
        object_point_mask = lens_to_mask(object_point_lens, max_len = object_pos.shape[-2]) if exists(object_point_lens) else None

        combined_mask = None
        if exists(object_mask) and exists(object_point_mask):
            combined_mask = einx.logical_and('b no, b no n -> b no n', object_mask, object_point_mask)
        elif exists(object_mask):
            combined_mask = repeat(object_mask, 'b no -> b no n', n = object_pos.shape[-2])
        elif exists(object_point_mask):
            combined_mask = object_point_mask

        assert exists(object_first_frame_pos), (
            'object_first_frame_pos must be provided; using an all-zero reference '
            'degenerates the Kabsch projection and collapses each rigid object'
        )

        # deterministic FPS on the reference geometry unless canonical indices
        # are supplied by the dataset

        if not exists(anchor_indices):
            anchor_indices = deterministic_farthest_point_sample(
                object_first_frame_pos,
                self.num_anchors,
                mask = combined_mask
            )

        num_anchors = anchor_indices.shape[-1]

        # validate inputs

        if exists(object_pos_prev):
            anchor_pos_prev = batched_index_select(object_pos_prev, anchor_indices, dim = 2)

        # construct vertex and object tokens

        assert exists(object_pos_prev), 'object_pos_prev must be provided'

        # Paper Sec. 3.1 uses the per-step position increment as a discrete
        # velocity surrogate; it is deliberately not divided by physical_dt.
        discrete_velocity = object_pos - object_pos_prev

        reference_offset = object_pos - object_first_frame_pos

        if vertex_properties.ndim == 3: # (b, no, d_attr)
            vertex_properties = repeat(vertex_properties, 'b no d -> b no n d', n = object_pos.shape[-2])

        # nearest neighbor displacement to other object or ground plane - section 3.1 of paper

        nearest_neighbor_disp = nearest_neighbor_displacement(object_pos, mask = combined_mask)

        vertex_features = cat((
            nearest_neighbor_disp,
            discrete_velocity,
            reference_offset,
            vertex_properties
        ), dim = -1)

        # paper-aligned four-scale PointNet, or backwards-compatible custom encoder

        if self.uses_paper_pointnet:
            object_tokens, vertex_tokens = self.hierarchical_encoder(
                vertex_features,
                object_pos,
                mask = combined_mask,
                reference_pos = object_first_frame_pos,
                fps_indices = pointnet_fps_indices
            )
        else:
            vertex_tokens = self.vertex_encoder(vertex_features)
            encoder_kwargs = dict(mask = combined_mask) if exists(combined_mask) else dict()
            object_tokens = self.hierarchical_encoder(vertex_tokens, object_pos, **encoder_kwargs)

        if object_tokens.ndim == 4 and object_tokens.shape[-2] == 1:
            object_tokens = rearrange(object_tokens, 'b no 1 d -> b no d')

        assert object_tokens.ndim == 3, 'hierarchical encoder must output a single token per object, i.e. (batch, num_objects, dim)'

        # Paper predictor input: absolute anchor position + 12D state + 256D AVP.

        pooled_vertex_tokens, anchor_pos = self.anchor_vertex_pool(
            vertex_tokens,
            object_pos,
            anchor_indices,
            mask = combined_mask
        )
        avp_features = self.anchor_avp(pooled_vertex_tokens)

        anchor_vertex_state = batched_index_select(vertex_features, anchor_indices, dim = 2)
        anchor_state = cat((anchor_pos, anchor_vertex_state), dim = -1)
        anchor_tokens = self.anchor_query_fuse(cat((anchor_state, avp_features), dim = -1))

        # Keep dimensional physical time separate from the paper's FiLM code.
        # physical_dt^2 is used only by Verlet and the objective. FiLM receives
        # the dimensionless c = (s, s^2), where s is the sampled step code.

        assert physical_dt.shape == (batch,)
        assert step_code.shape == (batch,)
        assert physical_dt.device == object_pos.device
        assert step_code.device == object_pos.device

        physical_dt = physical_dt.float()
        step_code = step_code.float()
        assert torch.all(torch.isfinite(physical_dt))
        assert torch.all(torch.isfinite(step_code))
        assert torch.all(physical_dt > 0), 'physical_dt must be strictly positive'
        assert torch.all(step_code > 0), 'step_code must be strictly positive'

        physical_dt_squared = physical_dt.pow(2)
        time_cond = stack((step_code, step_code.pow(2)), dim = -1)

        # register tokens

        registers = repeat(self.register_tokens, 'r d -> b r d', b = batch)

        object_tokens, inverse_pack_registers = pack_with_inverse((registers, object_tokens), 'b * d')

        # Paper ARoPE: 96 rotary channels (32 per axis) in a 128D head.

        anchor_rope, object_rotary_pos_emb, object_rotary_pos_emb_with_registers = self._build_arope_embeddings(anchor_pos)

        anchor_rope = rearrange(anchor_rope, 'b ... f -> b 1 ... f')
        object_rotary_pos_emb = rearrange(object_rotary_pos_emb, 'b ... f -> b 1 ... f')
        object_rotary_pos_emb_with_registers = rearrange(object_rotary_pos_emb_with_registers, 'b ... f -> b 1 ... f')

        # handle the "ARoPE" for anchors

        anchor_rotary_pos_emb = rearrange(anchor_rope, 'b h no na f -> b h (no na) f')

        # object self attention

        # Attention Residuals counts self-attention and FFN as individual
        # layers. With the paper setting of four Transformer layers and S=4,
        # these eight sublayers form two blocks. The embedding is b_0;
        # transformation outputs are summed only within their current block.

        attn_residual_blocks = [object_tokens]
        partial_attn_residual_block = None
        sublayers_in_partial_block = 0

        object_hiddens = [object_tokens]

        object_mask_with_registers = pad_left_at_dim(object_mask, self.num_register_tokens, value = True) if exists(object_mask) else None

        for layer_index, (attn_film, attn, ff, attn_residual, ff_residual) in enumerate(self.self_attn_layers):

            attn_input = attn_residual(
                attn_residual_blocks,
                partial_attn_residual_block
            )

            # The next layer's pre-attention aggregate is the AttnRes analogue
            # of the preceding Transformer block's output.

            if layer_index > 0:
                object_hiddens.append(attn_input)

            attn_output = attn(
                attn_input,
                rotary_pos_emb = object_rotary_pos_emb_with_registers,
                mask = object_mask_with_registers
            )

            partial_attn_residual_block = attn_output if not exists(partial_attn_residual_block) else partial_attn_residual_block + attn_output
            sublayers_in_partial_block += 1

            if sublayers_in_partial_block == self.attn_residual_block_size:
                attn_residual_blocks.append(partial_attn_residual_block)
                partial_attn_residual_block = None
                sublayers_in_partial_block = 0

            ff_input = ff_residual(
                attn_residual_blocks,
                partial_attn_residual_block
            )
            ff_input = attn_film(ff_input, time_cond)

            ff_output = ff(ff_input)

            partial_attn_residual_block = ff_output if not exists(partial_attn_residual_block) else partial_attn_residual_block + ff_output
            sublayers_in_partial_block += 1

            if sublayers_in_partial_block == self.attn_residual_block_size:
                attn_residual_blocks.append(partial_attn_residual_block)
                partial_attn_residual_block = None
                sublayers_in_partial_block = 0

        object_tokens = self.object_final_attn_residual(
            attn_residual_blocks,
            partial_attn_residual_block
        )
        object_hiddens.append(object_tokens)

        # anchor cross attention - parallel multi-scale cross attention (appendix G)

        anchor_tokens, inverse_pack_objects_num_anchors = pack_with_inverse(anchor_tokens, 'b * d')

        anchor_mask = repeat(object_mask, 'b no -> b (no na)', na = num_anchors) if exists(object_mask) else None

        anchor_outputs = []

        for ind, (self_attn_film, self_attn, attn_film, attn, context_attn_residual) in enumerate(self.cross_attn_layers):

            if exists(self_attn):
                filmed_self = self_attn_film(anchor_tokens, time_cond)
                anchor_tokens = self_attn(filmed_self, rotary_pos_emb = anchor_rotary_pos_emb, mask = anchor_mask) + anchor_tokens

            if self.learned_object_hidden_layers:
                object_context = context_attn_residual(object_hiddens)
            else:
                object_layer_index = self.object_hidden_layers[ind]
                object_context = object_hiddens[object_layer_index]

            _, object_context = inverse_pack_registers(object_context) # remove register tokens

            anchor_output = attn(anchor_tokens, rotary_pos_emb = anchor_rotary_pos_emb, context_rotary_pos_emb = object_rotary_pos_emb, context = object_context, mask = object_mask) + anchor_tokens
            anchor_output = attn_film(anchor_output, time_cond)

            anchor_outputs.append(anchor_output)

        anchor_tokens = self.cross_attn_fuse(cat(anchor_outputs, dim = -1))

        anchor_tokens = inverse_pack_objects_num_anchors(anchor_tokens)

        pred_acc = self.to_acc_pred(anchor_tokens)

        assert exists(anchor_pos) == exists(anchor_pos_prev)

        # early return prediction if ground truth not passed in

        return_loss = exists(object_pos_next)
        has_gt_current = exists(object_pos_gt)
        has_gt_previous = exists(object_pos_prev_gt)

        assert not return_predictions_with_loss or return_loss, (
            '`return_predictions_with_loss` requires `object_pos_next`'
        )
        assert has_gt_current == has_gt_previous, (
            '`object_pos_gt` and `object_pos_prev_gt` must be supplied together'
        )
        assert not has_gt_current or return_loss, (
            'ground-truth history is only valid when `object_pos_next` is supplied'
        )

        if return_loss:
            object_pos_gt = default(object_pos_gt, object_pos)
            object_pos_prev_gt = default(object_pos_prev_gt, object_pos_prev)

            assert object_pos_gt.shape == object_pos_next.shape
            assert object_pos_prev_gt.shape == object_pos_next.shape

            anchor_pos_next = batched_index_select(object_pos_next, anchor_indices, dim = 2)
            anchor_pos_gt = batched_index_select(object_pos_gt, anchor_indices, dim = 2)
            anchor_pos_prev_gt = batched_index_select(object_pos_prev_gt, anchor_indices, dim = 2)

        # verlet, then differentiable kabsch aligning reference anchors to predicted (paper 3.2)

        pred_anchor_pos_next = 2 * anchor_pos - anchor_pos_prev + einx.multiply('b ..., b', pred_acc, physical_dt_squared)

        object_pos_ref = object_first_frame_pos
        anchor_pos_ref = batched_index_select(object_first_frame_pos, anchor_indices, dim = 2)

        R, T = roma.rigid_points_registration(anchor_pos_ref, pred_anchor_pos_next)

        pred_anchor_pos_next_rigid = einx.add('b no na c, b no c', einsum(anchor_pos_ref, R, 'b no na c1, b no c2 c1 -> b no na c2'), T)
        rigid_object_pos_next = einx.add('b no c, b no n c', T, einsum(object_pos_ref, R, 'b no n c1, b no c2 c1 -> b no n c2'))

        predictions = Predictions(pred_acc, rigid_object_pos_next)

        if not return_loss:
            if not return_intermediates:
                return predictions

            return predictions, Intermediates(anchor_indices)

        # Paper objective (Sec. 3.4 and Appendix C.2): four anchor-level
        # SmoothL1 terms, before/after Kabsch for position and acceleration.

        anchor_loss_terms = rigidformer_anchor_losses(
            pred_acc = pred_acc,
            pred_anchor_pos_next = pred_anchor_pos_next,
            pred_anchor_pos_next_rigid = pred_anchor_pos_next_rigid,
            anchor_pos_next = anchor_pos_next,
            anchor_pos = anchor_pos,
            anchor_pos_prev = anchor_pos_prev,
            physical_dt_squared = physical_dt_squared,
            object_mask = object_mask,
            anchor_pos_gt = anchor_pos_gt,
            anchor_pos_prev_gt = anchor_pos_prev_gt
        )

        pos_loss = (
            anchor_loss_terms.raw_position +
            anchor_loss_terms.rigid_position
        )
        acc_loss = (
            anchor_loss_terms.raw_acceleration +
            anchor_loss_terms.rigid_acceleration
        )

        total_loss = (
            acc_loss * self.acc_loss_weight +
            pos_loss * self.pos_loss_weight
        )

        ret = (total_loss, Losses(acc_loss, pos_loss))

        if return_predictions_with_loss:
            ret = (*ret, predictions)

        if not return_intermediates:
            return ret

        return *ret, Intermediates(anchor_indices)

# rollout wrapper, for inference but also for training

class RigidformerRolloutWrapper(Module):
    def __init__(
        self,
        rigidformer: Rigidformer,
        cache_anchor_indices = True
    ):
        super().__init__()

        self.rigidformer = rigidformer
        self.cache_anchor_indices = cache_anchor_indices

    def rand_steps(
        self,
        physical_dt, # (b)
        step_code,   # (b)
        *,
        num_rand_substeps,
        max_step_weight = 2
    ):
        assert physical_dt.shape == step_code.shape
        batch, device = physical_dt.shape[0], physical_dt.device
        assert step_code.device == device

        # returns times broken up into random substeps, for consistency training

        rand_step_weights = torch.randint(1, max_step_weight, (batch, num_rand_substeps), device = device)

        normalized_weights = l1norm(rand_step_weights.float())

        return RigidformerRolloutStepSchedule(
            physical_dt = einx.multiply(
                'b n, b',
                normalized_weights,
                physical_dt
            ),
            step_code = einx.multiply(
                'b n, b',
                normalized_weights,
                step_code
            )
        )

    def forward(
        self,
        physical_dt, # (b) | (b steps)
        step_code,   # (b) | (b steps)
        *,
        vertex_properties,              # (b no n d_attr) or (b no d_attr)
        object_positions: list[Tensor], # must be at least 2
        num_steps = None,
        anchor_indices = None,          # (b no na)
        pointnet_fps_indices = None,    # 3-tuple of nested hierarchy indices
        object_point_lens = None,       # (b no)
        object_lens = None,             # (b)
        return_intermediates = False
    ):

        # Either fixed physical times and FiLM codes for num_steps, or one
        # value of each per rollout step.

        assert (
            (exists(num_steps) and physical_dt.ndim == step_code.ndim == 1) or
            (not exists(num_steps) and physical_dt.ndim == step_code.ndim == 2)
        )
        assert physical_dt.shape == step_code.shape

        if physical_dt.ndim == 1:
            physical_dt = repeat(physical_dt, 'b -> b steps', steps = num_steps)
            step_code = repeat(step_code, 'b -> b steps', steps = num_steps)

        # validate the object initial positions and make a shallow copy

        assert len(object_positions) >= 2, 'object position history must be at least 2'
        object_positions = object_positions.copy()

        # for the reference vector feature

        object_first_frame_pos = first(object_positions)

        # Iterate with physical integration time and FiLM code kept separate.

        step_schedule = zip(
            physical_dt.unbind(dim = -1),
            step_code.unbind(dim = -1)
        )

        for one_physical_dt, one_step_code in step_schedule:

            *_, object_pos_prev, object_pos = object_positions

            one_step_pred, intermediates = self.rigidformer(
                physical_dt = one_physical_dt,
                step_code = one_step_code,
                object_pos = object_pos,
                object_pos_prev = object_pos_prev,
                object_first_frame_pos = object_first_frame_pos,
                vertex_properties = vertex_properties,
                anchor_indices = anchor_indices,
                pointnet_fps_indices = pointnet_fps_indices,
                object_point_lens = object_point_lens,
                object_lens = object_lens,
                return_intermediates = True
            )

            # anchor indices are generated via FPS on first step, then reused for all subsequent steps

            if not exists(anchor_indices):
                anchor_indices = intermediates.anchor_indices

            object_positions.append(one_step_pred.object_pos_next)

        if not return_intermediates:
            return object_positions

        return object_positions, Intermediates(anchor_indices)
