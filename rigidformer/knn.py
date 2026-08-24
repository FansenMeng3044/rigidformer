from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def exact_knn_indices(
    queries: Tensor,
    supports: Tensor,
    num_neighbors: int,
    *,
    support_mask: Tensor | None = None,
    support_chunk_size = 1024
):
    """Exact device-native KNN without materializing the full distance matrix.

    CUDA inputs execute entirely with CUDA tensor operations (`bmm` and
    `topk`). Supports are streamed in chunks, bounding temporary distance
    memory at `(batch, queries, support_chunk_size)`. CPU execution is retained
    as a numerically equivalent development and test fallback.
    """

    assert queries.ndim == supports.ndim == 3
    assert queries.shape[0] == supports.shape[0]
    assert queries.shape[-1] == supports.shape[-1]
    assert queries.device == supports.device
    assert queries.dtype == supports.dtype
    assert torch.is_floating_point(queries)
    assert torch.is_floating_point(supports)
    assert isinstance(num_neighbors, int) and num_neighbors > 0
    assert isinstance(support_chunk_size, int) and support_chunk_size > 0

    batch, num_queries, _ = queries.shape
    num_supports = supports.shape[1]
    assert num_queries > 0
    assert num_supports > 0
    num_neighbors = min(num_neighbors, num_supports)

    if support_mask is not None:
        assert support_mask.shape == (batch, num_supports)
        assert support_mask.dtype == torch.bool
        assert support_mask.device == supports.device

    # Float32 accumulation avoids half-precision overflow and matches the
    # precision used by common CUDA KNN extensions for geometric distances.

    distance_dtype = (
        torch.float32
        if queries.dtype in (torch.float16, torch.bfloat16)
        else queries.dtype
    )
    queries_for_distance = queries.to(distance_dtype)
    supports_for_distance = supports.to(distance_dtype)
    query_norm = queries_for_distance.square().sum(dim = -1, keepdim = True)

    best_distances = torch.full(
        (batch, num_queries, num_neighbors),
        torch.inf,
        device = queries.device,
        dtype = distance_dtype
    )
    if support_mask is None:
        fallback_indices = torch.zeros(
            (batch,),
            device = queries.device,
            dtype = torch.long
        )
    else:
        # When a padded cloud has fewer than K valid points, sentinel slots
        # must reference padding so the caller's gathered mask stays false.
        fallback_indices = (~support_mask).to(torch.long).argmax(dim = -1)

    best_indices = fallback_indices[:, None, None].expand(
        batch,
        num_queries,
        num_neighbors
    ).clone()

    for chunk_start in range(0, num_supports, support_chunk_size):
        chunk_end = min(chunk_start + support_chunk_size, num_supports)
        support_chunk = supports_for_distance[:, chunk_start:chunk_end]
        support_norm = support_chunk.square().sum(dim = -1)[:, None, :]
        squared_distances = query_norm + support_norm - 2. * torch.bmm(
            queries_for_distance,
            support_chunk.transpose(1, 2)
        )
        squared_distances.clamp_min_(0.)

        if support_mask is not None:
            chunk_mask = support_mask[:, None, chunk_start:chunk_end]
            squared_distances.masked_fill_(~chunk_mask, torch.inf)

        chunk_indices = torch.arange(
            chunk_start,
            chunk_end,
            device = queries.device,
            dtype = torch.long
        )
        chunk_indices = chunk_indices.view(1, 1, -1).expand(
            batch,
            num_queries,
            -1
        )

        candidate_distances = torch.cat((best_distances, squared_distances), dim = -1)
        candidate_indices = torch.cat((best_indices, chunk_indices), dim = -1)
        best_distances, selected = candidate_distances.topk(
            num_neighbors,
            dim = -1,
            largest = False,
            sorted = True
        )
        best_indices = candidate_indices.gather(-1, selected)

    return best_indices
