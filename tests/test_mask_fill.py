from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


def _fill(nodes_module, image, mask, **kwargs):
    return nodes_module.vloFillMaskedArea.execute(image, mask, **kwargs).result[0]


def _outpaint_frame(height=48, width=80, colour=(0.6, 0.3, 0.2)):
    """A coloured patch on black, and the mask of the black around it."""
    image = torch.zeros(1, height, width, 3)
    mask = torch.ones(1, height, width)
    image[:, 13:35, 21:59] = torch.tensor(colour)
    mask[:, 13:35, 21:59] = 0.0
    return image, mask


def test_known_pixels_come_back_exactly(nodes_module) -> None:
    torch.manual_seed(0)
    image = torch.rand(2, 33, 47, 3)
    mask = (torch.rand(2, 33, 47) > 0.7).float()

    out = _fill(nodes_module, image, mask)

    kept = mask == 0
    assert torch.equal(out[kept], image[kept])


def test_the_black_around_an_outpaint_takes_the_patch_colour(nodes_module) -> None:
    image, mask = _outpaint_frame()

    out = _fill(nodes_module, image, mask)

    assert torch.allclose(out, torch.tensor([0.6, 0.3, 0.2]).expand_as(out), atol=1e-5)


def test_fill_stays_within_the_range_of_the_known_colours(nodes_module) -> None:
    # Pull-push only ever averages known colours, so it cannot overshoot them.
    image = torch.zeros(1, 40, 64, 3)
    image[:, :, :32] = torch.tensor([0.9, 0.1, 0.1])
    image[:, :, 32:] = torch.tensor([0.1, 0.1, 0.9])
    mask = torch.zeros(1, 40, 64)
    mask[:, 10:30, 16:48] = 1.0
    image[mask.bool()] = 0.0

    out = _fill(nodes_module, image, mask)

    filled = out[mask.bool()]
    assert filled[:, 0].min() >= 0.1 - 1e-5 and filled[:, 0].max() <= 0.9 + 1e-5
    assert filled[:, 2].min() >= 0.1 - 1e-5 and filled[:, 2].max() <= 0.9 + 1e-5
    # Red side stays red-ish, blue side blue-ish: the fill follows its surroundings.
    assert out[0, 20, 18, 0] > out[0, 20, 18, 2]
    assert out[0, 20, 45, 2] > out[0, 20, 45, 0]


def test_values_at_or_below_the_threshold_are_kept(nodes_module) -> None:
    image, mask = _outpaint_frame()
    mask[:, 0, 0] = 0.1  # raised black from a decoded mask video, not mask
    image[:, 0, 0] = torch.tensor([0.0, 0.0, 0.0])

    out = _fill(nodes_module, image, mask)

    assert torch.equal(out[0, 0, 0], torch.zeros(3))
    assert out[0, 47, 79].min() > 0.1  # the real black is still filled


def test_threshold_is_adjustable(nodes_module) -> None:
    image, mask = _outpaint_frame()
    mask[:, 0, 0] = 0.1

    out = _fill(nodes_module, image, mask, threshold=0.05)

    assert torch.allclose(out[0, 0, 0], torch.tensor([0.6, 0.3, 0.2]), atol=1e-5)


def test_one_mask_applies_to_every_frame(nodes_module) -> None:
    image, mask = _outpaint_frame()
    frames = image.expand(5, -1, -1, -1).clone()

    out = _fill(nodes_module, frames, mask)

    assert out.shape == frames.shape
    assert torch.allclose(out, torch.tensor([0.6, 0.3, 0.2]).expand_as(out), atol=1e-5)


def test_frames_with_nothing_to_fill_or_nothing_to_fill_from_are_untouched(
    nodes_module,
) -> None:
    torch.manual_seed(1)
    image = torch.rand(3, 16, 16, 3)
    mask = torch.zeros(3, 16, 16)
    mask[1] = 1.0  # all masked: nothing to extend from
    mask[2, 4:8, 4:8] = 1.0

    out = _fill(nodes_module, image, mask)

    assert torch.equal(out[0], image[0])
    assert torch.equal(out[1], image[1])
    assert not torch.equal(out[2], image[2])


def test_more_frames_than_one_chunk(nodes_module) -> None:
    image, mask = _outpaint_frame(height=17, width=23)
    frames = image.expand(37, -1, -1, -1).clone()
    masks = mask.expand(37, -1, -1).clone()
    masks[30] = 0.0  # one frame with no hole, in the last chunk

    out = _fill(nodes_module, frames, masks)

    assert torch.equal(out[30], frames[30])
    assert torch.allclose(out[0], torch.tensor([0.6, 0.3, 0.2]).expand_as(out[0]), atol=1e-5)
    assert torch.allclose(out[36], out[0])


def test_mismatched_mask_is_rejected(nodes_module) -> None:
    image = torch.zeros(4, 16, 16, 3)
    with pytest.raises(ValueError, match="Resize the mask"):
        _fill(nodes_module, image, torch.zeros(4, 8, 8))
    with pytest.raises(ValueError, match="one mask per frame"):
        _fill(nodes_module, image, torch.zeros(3, 16, 16))
