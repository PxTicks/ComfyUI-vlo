from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


def _point_mask(frames: int, height: int, width: int, y: int, x: int) -> torch.Tensor:
    mask = torch.zeros(frames, height, width)
    mask[:, y, x] = 1.0
    return mask


def _grow(nodes_module, mask, expand):
    return nodes_module.vloGrowMask.execute(mask, expand=expand).result[0]


def test_grow_is_a_square_of_the_requested_radius(nodes_module) -> None:
    out = _grow(nodes_module, _point_mask(1, 32, 32, 16, 16), 4)

    expected = torch.zeros(1, 32, 32)
    expected[:, 12:21, 12:21] = 1.0
    assert torch.equal(out, expected)


def test_grow_reaches_the_diagonal_corner(nodes_module) -> None:
    # The latent-cell guarantee is Chebyshev distance: a round or tapered kernel
    # of the same radius would miss this corner.
    out = _grow(nodes_module, _point_mask(1, 32, 32, 16, 16), 8)

    assert out[0, 8, 8] == 1.0
    assert out[0, 24, 24] == 1.0
    assert out[0, 7, 8] == 0.0


def test_grow_matches_stock_grow_mask_without_tapered_corners(nodes_module) -> None:
    pytest.importorskip("scipy")
    import numpy as np
    import scipy.ndimage

    torch.manual_seed(0)
    mask = (torch.rand(3, 40, 56) > 0.97).float() * torch.rand(3, 40, 56)

    expected = []
    for frame in mask:
        output = frame.numpy()
        for _ in range(5):
            output = scipy.ndimage.grey_dilation(output, footprint=np.ones((3, 3)))
        expected.append(torch.from_numpy(output))

    assert torch.allclose(_grow(nodes_module, mask, 5), torch.stack(expected))


def test_grow_keeps_soft_values_and_never_exceeds_the_peak(nodes_module) -> None:
    mask = torch.zeros(1, 16, 16)
    mask[0, 8, 8] = 0.4
    mask[0, 8, 9] = 0.7

    out = _grow(nodes_module, mask, 2)

    assert out.max().item() == pytest.approx(0.7)
    assert out[0, 8, 6].item() == pytest.approx(0.4)
    assert out[0, 8, 11].item() == pytest.approx(0.7)


def test_grow_handles_masks_past_one_device_chunk(nodes_module) -> None:
    frames = nodes_module.mask_grow._GROW_CHUNK_FRAMES * 2 + 3
    mask = torch.zeros(frames, 12, 12)
    for index in range(frames):
        mask[index, index % 12, 6] = 1.0

    out = _grow(nodes_module, mask, 1)

    assert out.shape == (frames, 12, 12)
    for index in range(frames):
        row = index % 12
        assert out[index, max(0, row - 1) : row + 2, 5:8].min() == 1.0
        assert out[index].sum() == len(range(max(0, row - 1), min(12, row + 2))) * 3


def test_zero_expand_passes_the_mask_through(nodes_module) -> None:
    mask = _point_mask(2, 8, 8, 3, 3)
    assert torch.equal(_grow(nodes_module, mask, 0), mask)


def test_single_frame_2d_mask_gains_a_batch_axis(nodes_module) -> None:
    out = _grow(nodes_module, torch.zeros(8, 8), 2)
    assert out.shape == (1, 8, 8)
