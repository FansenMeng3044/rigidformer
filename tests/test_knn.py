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
