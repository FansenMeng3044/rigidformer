from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class ExactKNNResult(NamedTuple):
    indices: Tensor
    squared_distances: Tensor
    valid: Tensor


def _expand_group_ids(
    group_ids: Tensor,
    *,
    batch: int,
    num_points: int,
    device: torch.device
) -> Tensor:
    assert group_ids.device == device

    if group_ids.ndim == 1:
        assert group_ids.shape == (num_points,)
        return group_ids[None, :].expand(batch, -1)

    assert group_ids.shape == (batch, num_points)
    return group_ids


@torch.no_grad()
def exact_masked_knn(
    queries: Tensor,
    supports: Tensor,
    num_neighbors: int,
    *,
    query_mask: Tensor | None = None,
    support_mask: Tensor | None = None,
    query_group_ids: Tensor | None = None,
    support_group_ids: Tensor | None = None,
    exclude_same_group: bool = False,
    query_chunk_size: int = 1024,
    support_chunk_size: int = 1024
) -> ExactKNNResult:
    """Exact, device-native KNN with bounded pairwise-distance memory.

    Both query and support axes are streamed, so the largest temporary distance
    tensor is ``(batch, query_chunk_size, support_chunk_size)``. Masks remove
    padding from both sides. Optional group ids can exclude every support in a
    query's own group, which is useful for inter-object contact features.

    Neighbor selection is intentionally non-differentiable. Callers that need
    gradients should gather the selected support coordinates outside this
    function; gradients then flow through the selected, piecewise-constant
    neighborhood.
    """

    assert queries.ndim == supports.ndim == 3
    assert queries.shape[0] == supports.shape[0]
    assert queries.shape[-1] == supports.shape[-1]
    assert queries.device == supports.device
    assert queries.dtype == supports.dtype
    assert torch.is_floating_point(queries)
    assert isinstance(num_neighbors, int) and num_neighbors > 0
    assert isinstance(query_chunk_size, int) and query_chunk_size > 0
    assert isinstance(support_chunk_size, int) and support_chunk_size > 0

    batch, num_queries, _ = queries.shape
    num_supports = supports.shape[1]
    assert num_queries > 0
    assert num_supports > 0
    num_neighbors = min(num_neighbors, num_supports)

    if query_mask is None:
        query_mask = torch.ones(
            (batch, num_queries),
            dtype = torch.bool,
            device = queries.device
        )
    else:
        assert query_mask.shape == (batch, num_queries)
        assert query_mask.dtype == torch.bool
        assert query_mask.device == queries.device

    if support_mask is None:
        support_mask = torch.ones(
            (batch, num_supports),
            dtype = torch.bool,
            device = supports.device
        )
        has_explicit_support_mask = False
    else:
        assert support_mask.shape == (batch, num_supports)
        assert support_mask.dtype == torch.bool
        assert support_mask.device == supports.device
        has_explicit_support_mask = True

    if exclude_same_group:
        assert query_group_ids is not None and support_group_ids is not None
        query_group_ids = _expand_group_ids(
            query_group_ids,
            batch = batch,
            num_points = num_queries,
            device = queries.device
        )
        support_group_ids = _expand_group_ids(
            support_group_ids,
            batch = batch,
            num_points = num_supports,
            device = supports.device
        )

    # Float32 accumulation avoids fp16/bfloat16 overflow while retaining fp64
    # when explicitly requested for numerical reference checks.

    distance_dtype = (
        torch.float32
        if queries.dtype in (torch.float16, torch.bfloat16)
        else queries.dtype
    )
    queries_for_distance = queries.to(distance_dtype)
    supports_for_distance = supports.to(distance_dtype)

    all_distances = torch.full(
        (batch, num_queries, num_neighbors),
        torch.inf,
        device = queries.device,
        dtype = distance_dtype
    )

    # If fewer than K valid supports exist, point the unused slots at padding.
    # This preserves the legacy PointNet contract: gathering support_mask at
    # returned indices keeps those slots invalid. With no padding, index zero
    # is a safe sentinel and ``valid`` remains the source of truth.

    if has_explicit_support_mask:
        fallback_indices = (~support_mask).to(torch.long).argmax(dim = -1)
    else:
        fallback_indices = torch.zeros(
            (batch,),
            device = queries.device,
            dtype = torch.long
        )

    all_indices = fallback_indices[:, None, None].expand(
        batch,
        num_queries,
        num_neighbors
    ).clone()

    for query_start in range(0, num_queries, query_chunk_size):
        query_end = min(query_start + query_chunk_size, num_queries)
        query_chunk = queries_for_distance[:, query_start:query_end]
        query_chunk_mask = query_mask[:, query_start:query_end]
        query_norm = query_chunk.square().sum(dim = -1, keepdim = True)

        chunk_num_queries = query_end - query_start
        best_distances = torch.full(
            (batch, chunk_num_queries, num_neighbors),
            torch.inf,
            device = queries.device,
            dtype = distance_dtype
        )
        best_indices = fallback_indices[:, None, None].expand(
            batch,
            chunk_num_queries,
            num_neighbors
        ).clone()

        for support_start in range(0, num_supports, support_chunk_size):
            support_end = min(support_start + support_chunk_size, num_supports)
            support_chunk = supports_for_distance[:, support_start:support_end]
            support_norm = support_chunk.square().sum(dim = -1)[:, None, :]
            squared_distances = query_norm + support_norm - 2. * torch.bmm(
                query_chunk,
                support_chunk.transpose(1, 2)
            )
            squared_distances.clamp_min_(0.)

            pair_mask = query_chunk_mask[:, :, None]
            pair_mask = pair_mask & support_mask[:, None, support_start:support_end]

            if exclude_same_group:
                query_groups = query_group_ids[:, query_start:query_end, None]
                support_groups = support_group_ids[:, None, support_start:support_end]
                pair_mask = pair_mask & (query_groups != support_groups)

            squared_distances.masked_fill_(~pair_mask, torch.inf)

            if num_neighbors == 1:
                # Strict comparison plus ascending chunk traversal gives a
                # deterministic lowest-index winner for equal distances.
                candidate_distances, candidate_local_indices = squared_distances.min(
                    dim = -1,
                    keepdim = True
                )
                candidate_indices = candidate_local_indices + support_start
                improves = candidate_distances < best_distances
                best_distances = torch.where(
                    improves,
                    candidate_distances,
                    best_distances
                )
                best_indices = torch.where(
                    improves,
                    candidate_indices,
                    best_indices
                )
                continue

            support_indices = torch.arange(
                support_start,
                support_end,
                device = queries.device,
                dtype = torch.long
            )
            support_indices = support_indices.view(1, 1, -1).expand(
                batch,
                chunk_num_queries,
                -1
            )

            candidate_distances = torch.cat(
                (best_distances, squared_distances),
                dim = -1
            )
            candidate_indices = torch.cat(
                (best_indices, support_indices),
                dim = -1
            )
            best_distances, selected = candidate_distances.topk(
                num_neighbors,
                dim = -1,
                largest = False,
                sorted = True
            )
            best_indices = candidate_indices.gather(-1, selected)

        chunk_valid = torch.isfinite(best_distances)
        best_indices = torch.where(
            chunk_valid,
            best_indices,
            fallback_indices[:, None, None]
        )
        all_distances[:, query_start:query_end] = best_distances
        all_indices[:, query_start:query_end] = best_indices

    valid = torch.isfinite(all_distances) & query_mask[:, :, None]
    return ExactKNNResult(all_indices, all_distances, valid)


@torch.no_grad()
def exact_knn_indices(
    queries: Tensor,
    supports: Tensor,
    num_neighbors: int,
    *,
    support_mask: Tensor | None = None,
    support_chunk_size: int = 1024,
    query_chunk_size: int = 1024
) -> Tensor:
    """Backward-compatible index-only wrapper around :func:`exact_masked_knn`."""

    return exact_masked_knn(
        queries,
        supports,
        num_neighbors,
        support_mask = support_mask,
        query_chunk_size = query_chunk_size,
        support_chunk_size = support_chunk_size
    ).indices
