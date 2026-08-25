"""Condition-row alignment: the easiest bug to introduce, and the quietest one.

A misaligned strength vector still produces plausible video, so every mapping
between guide masks, patchified condition rows and `PackedLayout`'s `cond`
segments is asserted here rather than trusted.
"""

from __future__ import annotations

import pytest
import torch

from minimax_h3_harness import guide_payload, h3_model_module, masked_guide_module


@pytest.fixture(scope="module")
def fork():
    return masked_guide_module("masked_h3_forward")


@pytest.fixture(scope="module")
def core():
    return h3_model_module()


def _indexed_latent(token_t, token_h, token_w):
    """A latent whose every 2x2 patch carries its own flat row index."""
    latent = torch.zeros(1, 24, token_t, token_h * 2, token_w * 2)
    for t in range(token_t):
        for h in range(token_h):
            for w in range(token_w):
                latent[0, :, t, h * 2:h * 2 + 2, w * 2:w * 2 + 2] = (
                    (t * token_h + h) * token_w + w)
    return latent


def test_patchify_row_order_is_t_then_h_then_w(core):
    """Pins the ordering the whole feature's flattening depends on."""
    token_t, token_h, token_w = 2, 2, 3
    rows = core.patchify_video(_indexed_latent(token_t, token_h, token_w), (1, 2, 2))
    assert rows.shape[0] == token_t * token_h * token_w
    assert torch.equal(rows[:, 0], torch.arange(rows.shape[0], dtype=torch.float32))
    assert (rows == rows[:, :1]).all()   # every element of a patch shares its row


def test_cond_rows_land_where_the_layout_says(core):
    """Guide rows fill the layout's `cond` segments, references fill `ref_img`."""
    guide = torch.zeros(1, 24, 1, 4, 6)
    ref = torch.zeros(1, 24, 1, 2, 2)
    layout = core.PackedLayout(3, 2, 4, 6, 5,
                               keyframes=[{"resolved_frame_index": 0, "latent": guide}],
                               refs=[{"kind": "image", "latent": ref, "latent_h": 2, "latent_w": 2}])
    kinds = [(kind, b - a) for a, b, kind in layout.segments]
    assert ("cond", 2 * 3) in kinds
    assert ("ref_img", 1 * 1) in kinds


def test_plan_covers_every_condition_row_in_order(fork):
    """Keyframes first, then references -- the order model_base packs them in."""
    payload = guide_payload(torch.zeros(2 * 3, dtype=torch.float64), lat_h=4, lat_w=6)
    ref_latent = torch.zeros(1, 24, 1, 2, 2)
    payload["refs"] = [{"kind": "image", "latent": ref_latent}]
    payload["cond_video_latents"] = payload["cond_video_latents"] + [ref_latent]

    plan = fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)
    assert plan.aug_rows.shape == (2 * 3 + 1,)
    assert torch.equal(plan.aug_rows[:6], torch.zeros(6, dtype=torch.float64))  # masked guide
    assert plan.aug_rows[-1].item() == 0.999                                    # reference untouched
    assert len(plan.segment_rows_t) == 1                                        # one cond segment


def test_multiple_guides_keep_their_own_strengths(fork):
    """Nothing may hard-code "the first keyframe is the masked one"."""
    unmasked = {"resolved_frame_index": 40, "latent": torch.zeros(1, 24, 1, 4, 6)}
    masked_late = dict(unmasked)
    payload = guide_payload(torch.zeros(6, dtype=torch.float64),
                            extra_keyframes=(unmasked, masked_late))
    payload["keyframes"][2] = {
        "resolved_frame_index": 80, "latent": torch.zeros(1, 24, 1, 4, 6),
        fork.MASKED_GUIDE_KEY: {"strengths": torch.full((6,), 0.5, dtype=torch.float64)},
    }
    payload["cond_video_latents"] = [kf["latent"] for kf in payload["keyframes"]]

    plan = fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)
    assert plan.aug_rows.shape == (18,)
    assert plan.aug_rows[:6].tolist() == [0.0] * 6                 # guide 0, mask 0
    assert plan.aug_rows[6:12].tolist() == [0.999] * 6             # guide 1, unmasked
    assert plan.aug_rows[12:].unique().tolist() == [pytest.approx(0.4995)]  # guide 2, mask 0.5
    assert plan.segment_rows_t[1] is None                          # unmasked guide stays scalar
    assert plan.segment_rows_t[0] is not None and plan.segment_rows_t[2] is not None


def test_wrong_length_strengths_raise_rather_than_misalign(fork):
    payload = guide_payload(torch.zeros(5, dtype=torch.float64), lat_h=4, lat_w=6)
    with pytest.raises(RuntimeError, match="row alignment failed"):
        fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)


def test_payload_drift_between_keyframes_and_cond_latents_raises(fork):
    payload = guide_payload(torch.zeros(6, dtype=torch.float64))
    payload["cond_video_latents"] = payload["cond_video_latents"] * 2
    with pytest.raises(RuntimeError, match="row alignment failed"):
        fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)


def test_guide_mask_rows_match_the_cond_segment_width(fork, core):
    """The invariant of section 29: |mask rows| == |cond positional rows| == |patch rows|."""
    latent = torch.zeros(1, 24, 1, 4, 6)
    layout = core.PackedLayout(3, 2, 4, 6, 5,
                               keyframes=[{"resolved_frame_index": 0, "latent": latent}])
    cond = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    patch_rows = core.patchify_video(latent, (1, 2, 2)).shape[0]
    strengths = torch.zeros(patch_rows, dtype=torch.float64)
    plan = fork.build_cond_row_plan(guide_payload(strengths), t_v=0.5, vis_aug=0.999)
    assert cond[0][1] - cond[0][0] == patch_rows == plan.segment_rows_t[0].shape[0]


def test_has_masked_guides_only_fires_on_real_specs(fork):
    assert not fork.has_masked_guides(None)
    assert not fork.has_masked_guides({})
    assert not fork.has_masked_guides(guide_payload())
    assert fork.has_masked_guides(guide_payload(torch.zeros(6, dtype=torch.float64)))


def test_min_aug_is_clamped_into_the_usable_range(fork):
    """A floor above the stock coefficient would invert the mask's meaning."""
    payload = guide_payload(torch.zeros(6, dtype=torch.float64), min_aug=5.0)
    plan = fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)
    assert plan.aug_rows.max().item() == 0.999
