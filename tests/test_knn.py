import pytest
import torch


def reference_knn(queries, supports, num_neighbors, support_mask = None):
    distances = torch.cdist(queries, supports)

    if support_mask is not None:
        distances.masked_fill_(~support_mask[:, None, :], torch.inf)

    return distances.topk(
        num_neighbors,
        dim = -1,
        largest = False,
        sorted = True
    ).indices


@pytest.mark.parametrize('support_chunk_size', (3, 7, 64))
def test_exact_knn_matches_full_distance_reference_with_mask(support_chunk_size):
    from rigidformer import exact_knn_indices

    torch.manual_seed(0)
    queries = torch.randn(2, 5, 3)
    supports = torch.randn(2, 23, 3)
    support_mask = torch.ones(2, 23, dtype = torch.bool)
    support_mask[0, 17:] = False
    support_mask[1, 19:] = False

    actual = exact_knn_indices(
        queries,
        supports,
        5,
        support_mask = support_mask,
        support_chunk_size = support_chunk_size
    )
    expected = reference_knn(queries, supports, 5, support_mask)

    assert actual.device == queries.device
    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)


def test_exact_knn_padding_slots_remain_masked_when_fewer_than_k_are_valid():
    from rigidformer import exact_knn_indices

    queries = torch.randn(1, 3, 3)
    supports = torch.randn(1, 8, 3)
    support_mask = torch.tensor([[True, True, False, False, False, False, False, False]])
    indices = exact_knn_indices(
        queries,
        supports,
        5,
        support_mask = support_mask,
        support_chunk_size = 3
    )
    gathered_mask = support_mask.gather(1, indices.reshape(1, -1)).reshape_as(indices)

    assert torch.equal(gathered_mask.sum(dim = -1), torch.full((1, 3), 2))


def test_exact_knn_does_not_call_full_torch_cdist(monkeypatch):
    from rigidformer import exact_knn_indices

    monkeypatch.setattr(
        torch,
        'cdist',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('cdist called'))
    )
    indices = exact_knn_indices(
        torch.randn(2, 4, 3),
        torch.randn(2, 11, 3),
        3,
        support_chunk_size = 5
    )

    assert indices.shape == (2, 4, 3)


def test_exact_masked_knn_matches_dense_group_aware_reference():
    from rigidformer import exact_masked_knn

    torch.manual_seed(1)
    queries = torch.randn(2, 7, 3, dtype = torch.float64)
    supports = torch.randn(2, 13, 3, dtype = torch.float64)
    query_mask = torch.tensor([
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, False]
    ])
    support_mask = torch.tensor([
        [True] * 11 + [False] * 2,
        [True] * 12 + [False]
    ])
    query_groups = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    support_groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    actual = exact_masked_knn(
        queries,
        supports,
        3,
        query_mask = query_mask,
        support_mask = support_mask,
        query_group_ids = query_groups,
        support_group_ids = support_groups,
        exclude_same_group = True,
        query_chunk_size = 2,
        support_chunk_size = 4
    )

    expected_distances = torch.cdist(queries, supports).square()
    expected_pair_mask = query_mask[:, :, None] & support_mask[:, None, :]
    expected_pair_mask &= query_groups[None, :, None] != support_groups[None, None, :]
    expected_distances.masked_fill_(~expected_pair_mask, torch.inf)
    expected_distances, expected_indices = expected_distances.topk(
        3,
        dim = -1,
        largest = False,
        sorted = True
    )
    expected_valid = torch.isfinite(expected_distances)

    assert torch.equal(actual.valid, expected_valid)
    assert torch.equal(actual.indices[expected_valid], expected_indices[expected_valid])
    assert torch.allclose(
        actual.squared_distances[expected_valid],
        expected_distances[expected_valid],
        atol = 1e-12,
        rtol = 1e-12
    )


def test_exact_masked_knn_bounds_both_temporary_distance_axes(monkeypatch):
    from rigidformer import exact_masked_knn

    original_bmm = torch.bmm
    observed_shapes = []

    def recording_bmm(left, right):
        observed_shapes.append((left.shape, right.shape))
        return original_bmm(left, right)

    monkeypatch.setattr(torch, 'bmm', recording_bmm)
    result = exact_masked_knn(
        torch.randn(2, 9, 3),
        torch.randn(2, 17, 3),
        4,
        query_chunk_size = 3,
        support_chunk_size = 5
    )

    assert result.indices.shape == (2, 9, 4)
    assert len(observed_shapes) == 12
    assert all(left[1] <= 3 and right[2] <= 5 for left, right in observed_shapes)


def test_exact_masked_knn_k1_ties_choose_lowest_support_index():
    from rigidformer import exact_masked_knn

    result = exact_masked_knn(
        torch.tensor([[[0., 0., 0.]]]),
        torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]]),
        1,
        query_group_ids = torch.tensor([0]),
        support_group_ids = torch.tensor([1, 2]),
        exclude_same_group = True,
        query_chunk_size = 1,
        support_chunk_size = 1
    )

    assert result.valid.item()
    assert result.indices.item() == 0
    assert result.squared_distances.item() == 1.


def _dense_nearest_neighbor_displacement(object_pos, mask, ground_z):
    expected = torch.zeros_like(object_pos)
    batch, num_objects, num_points, _ = object_pos.shape

    for batch_index in range(batch):
        for object_index in range(num_objects):
            for point_index in range(num_points):
                if not mask[batch_index, object_index, point_index]:
                    continue

                query = object_pos[batch_index, object_index, point_index]
                best_disp = torch.tensor(
                    [0., 0., ground_z[batch_index] - query[2]],
                    dtype = object_pos.dtype,
                    device = object_pos.device
                )
                best_squared_distance = best_disp.square().sum()
                best_is_ground = True

                for support_object_index in range(num_objects):
                    if support_object_index == object_index:
                        continue

                    for support_point_index in range(num_points):
                        if not mask[batch_index, support_object_index, support_point_index]:
                            continue

                        candidate_disp = (
                            object_pos[batch_index, support_object_index, support_point_index]
                            - query
                        )
                        candidate_squared_distance = candidate_disp.square().sum()
                        if (
                            candidate_squared_distance < best_squared_distance
                            or (
                                best_is_ground
                                and candidate_squared_distance == best_squared_distance
                            )
                        ):
                            best_disp = candidate_disp
                            best_squared_distance = candidate_squared_distance
                            best_is_ground = False

                expected[batch_index, object_index, point_index] = best_disp

    return expected


def test_contact_knn_matches_dense_reference_with_padding_and_ground(monkeypatch):
    import rigidformer.rigidformer as rigidformer_module

    monkeypatch.setattr(
        rigidformer_module,
        'cdist',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('cdist called'))
    )
    object_pos = torch.tensor([
        [
            [[0., 0., 2.], [0., 1., 2.], [1e5, 1e5, 1e5]],
            [[.5, 0., 2.], [.5, 1., 2.], [-1e5, -1e5, -1e5]],
            [[5., 0., .2], [5., 1., .2], [5., 2., .2]]
        ],
        [
            [[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]],
            [[10., 0., 1.], [11., 0., 1.], [12., 0., 1.]],
            [[20., 0., 1.], [21., 0., 1.], [22., 0., 1.]]
        ]
    ])
    mask = torch.tensor([
        [[True, True, False], [True, True, False], [True, True, True]],
        [[True, True, True], [False, False, False], [False, False, False]]
    ])
    ground_z = torch.tensor([0., -.5])

    actual = rigidformer_module.nearest_neighbor_displacement(
        object_pos,
        mask = mask,
        ground_z = ground_z,
        query_chunk_size = 2,
        support_chunk_size = 3
    )
    expected = _dense_nearest_neighbor_displacement(object_pos, mask, ground_z)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[~mask]) == 0


def test_contact_knn_retains_piecewise_neighbor_gradient():
    from rigidformer.rigidformer import nearest_neighbor_displacement

    object_pos = torch.tensor(
        [[[[0., 0., 2.]], [[1., 0., 2.]]]],
        requires_grad = True
    )
    displacement = nearest_neighbor_displacement(
        object_pos,
        query_chunk_size = 1,
        support_chunk_size = 1
    )
    displacement.square().sum().backward()

    assert displacement.requires_grad
    assert object_pos.grad is not None
    assert torch.equal(
        object_pos.grad,
        torch.tensor([[[[-4., 0., 0.]], [[4., 0., 0.]]]])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA is unavailable')
def test_exact_knn_matches_reference_on_cuda():
    from rigidformer import exact_knn_indices

    torch.manual_seed(0)
    queries = torch.randn(4, 31, 3, device = 'cuda')
    supports = torch.randn(4, 257, 3, device = 'cuda')
    support_mask = torch.ones(4, 257, device = 'cuda', dtype = torch.bool)
    support_mask[0, 211:] = False

    actual = exact_knn_indices(
        queries,
        supports,
        16,
        support_mask = support_mask,
        support_chunk_size = 64
    )
    expected = reference_knn(queries, supports, 16, support_mask)

    assert actual.is_cuda
    assert torch.equal(actual, expected)
