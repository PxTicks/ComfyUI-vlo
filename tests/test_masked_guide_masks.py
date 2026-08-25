"""Mask geometry, token pooling and the strength -> noise-augmentation map."""

from __future__ import annotations

import pytest
import torch

from minimax_h3_harness import masked_guide_module


@pytest.fixture(scope="module")
def masks():
    return masked_guide_module("masks")


def test_area_pooling_keeps_partial_coverage(masks):
    """A token half covered by the mask is worth half, not a conservative 1.0."""
    mask = torch.zeros(4, 4)
    mask[:2] = 1.0                       # top half of a 2x2 token grid's top row
    pooled = masks.pool_mask_to_tokens(mask, 1, 1, "average")
    assert pooled.shape == (1, 1)
    assert pooled.item() == pytest.approx(0.5)
    assert masks.pool_mask_to_tokens(mask, 1, 1, "max").item() == pytest.approx(1.0)
    assert masks.pool_mask_to_tokens(mask, 1, 1, "min").item() == pytest.approx(0.0)


def test_pooling_rejects_unknown_modes(masks):
    with pytest.raises(ValueError, match="unknown pooling mode"):
        masks.pool_mask_to_tokens(torch.zeros(4, 4), 2, 2, "median")


def test_pooled_grid_is_not_transposed(masks):
    """An x/y transpose would still make plausible video; catch it here instead."""
    mask = torch.zeros(64, 128)
    mask[:, :64] = 1.0                   # left half of a 2:1 canvas
    grid = masks.guide_token_strengths(mask, width=128, height=64,
                                       token_t=1, token_h=2, token_w=4).reshape(2, 4)
    assert torch.equal(grid, torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 2, dtype=torch.float64))


def test_row_order_is_t_then_h_then_w(masks):
    """The flat vector must follow patchify_video's (t, h, w) row order."""
    token_h, token_w = 2, 3
    mask = torch.zeros(token_h * 32, token_w * 32)
    for h in range(token_h):
        for w in range(token_w):
            mask[h * 32:(h + 1) * 32, w * 32:(w + 1) * 32] = (h * token_w + w) / 5.0
    flat = masks.guide_token_strengths(mask, width=token_w * 32, height=token_h * 32,
                                       token_t=2, token_h=token_h, token_w=token_w)
    one_frame = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float64) / 5.0
    assert flat.shape == (2 * token_h * token_w,)
    assert torch.allclose(flat[:6], one_frame, atol=1.0 / 510.0)
    assert torch.equal(flat[:6], flat[6:])  # a still guide shares one map across time


def test_strength_and_gamma_shape_the_map(masks):
    mask = torch.full((32, 32), 0.5)
    kwargs = dict(width=32, height=32, token_t=1, token_h=1, token_w=1)
    plain = masks.guide_token_strengths(mask, **kwargs).item()
    assert plain == pytest.approx(0.5, abs=1.0 / 510.0)
    assert masks.guide_token_strengths(mask, strength=0.5, **kwargs).item() == pytest.approx(0.25, abs=1.0 / 510.0)
    assert masks.guide_token_strengths(mask, gamma=2.0, **kwargs).item() == pytest.approx(0.25, abs=1.0 / 510.0)
    assert masks.guide_token_strengths(mask, gamma=0.5, **kwargs).item() > plain


def test_quantization_keeps_the_endpoints_exact(masks):
    s = torch.tensor([0.0, 0.001, 0.5, 0.999, 1.0], dtype=torch.float64)
    q = masks.quantize_strengths(s)
    assert q[0].item() == 0.0 and q[-1].item() == 1.0
    assert q.unique().numel() <= 5
    assert (q * 255).round().eq(q * 255).all()


def test_strength_map_pins_both_endpoints(masks):
    """s = 1 must land exactly on the stock coefficient; s = 0 exactly on the floor."""
    s = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    aug = masks.strengths_to_aug(s, a_max=0.999, a_min=0.0)
    assert aug[-1].item() == 0.999
    assert aug[0].item() == 0.0
    assert torch.equal(aug, aug.sort().values)          # monotone in the mask
    floored = masks.strengths_to_aug(s, a_max=0.999, a_min=0.3)
    assert floored[0].item() == 0.3 and floored[-1].item() == 0.999


def test_condition_timestep_never_drops_below_the_video_timestep(masks):
    aug = torch.tensor([0.0, 0.4, 0.999], dtype=torch.float64)
    rows_t = masks.aug_to_cond_timestep(aug, t_v=0.6)
    assert rows_t.tolist() == [0.6, 0.6, 0.999]
    # a fully open guide reproduces core's scalar seg_t["cond"] bit for bit -- which is
    # why the whole strength -> timestep chain stays in float64 (float32 would land on
    # 0.9990000128746033 instead, splitting the modulation table in two)
    stock = masks.strengths_to_aug(torch.ones(1, dtype=torch.float64), a_max=0.999)
    assert float(masks.aug_to_cond_timestep(stock, 0.6)[0]) == max(0.6, 0.999)
    assert float(masks.aug_to_cond_timestep(torch.tensor([0.999], dtype=torch.float32), 0.6)[0]) != 0.999


def test_mask_must_frame_the_same_crop_as_the_image(masks):
    image = torch.zeros(1, 64, 128, 3)                  # [B, H, W, C]
    masks.check_mask_matches_image(torch.zeros(1, 32, 64), image)   # same aspect, fine
    with pytest.raises(ValueError, match="aspect ratio"):
        masks.check_mask_matches_image(torch.zeros(1, 128, 64), image)


def test_resize_uses_the_guides_cover_crop(masks):
    """A 1:1 mask on a 2:1 canvas must be centre-cropped, exactly like the guide image."""
    mask = torch.zeros(64, 64)
    mask[:, :32] = 1.0
    resized = masks.resize_mask_to_canvas(mask, 128, 32)
    assert resized.shape == (1, 32, 128)
    assert resized[0, :, :60].min().item() == pytest.approx(1.0)
    assert resized[0, :, 68:].max().item() == pytest.approx(0.0)


def test_resize_keeps_the_whole_mask_batch(masks):
    """Collapsing to mask 0 here is what made the pixel-fill baseline fill every
    image in a batch by the first image's mask."""
    masks_in = torch.stack([torch.ones(16, 16), torch.zeros(16, 16)])
    resized = masks.resize_mask_to_canvas(masks_in, 32, 32, crop="disabled")
    assert resized.shape == (2, 32, 32)
    assert float(resized[0].mean()) == pytest.approx(1.0)
    assert float(resized[1].mean()) == pytest.approx(0.0)


def test_single_mask_refuses_to_pick_from_a_batch(masks):
    assert masks.single_mask(torch.ones(1, 8, 8)).shape == (8, 8)
    assert masks.single_mask(torch.ones(8, 8)).shape == (8, 8)
    with pytest.raises(ValueError, match="carries 2 masks"):
        masks.single_mask(torch.ones(2, 8, 8))


def test_guide_strengths_refuse_a_mask_batch(masks):
    with pytest.raises(ValueError, match="guide mask carries 3 masks"):
        masks.guide_token_strengths(torch.ones(3, 32, 32), width=32, height=32,
                                    token_t=1, token_h=1, token_w=1)
