"""Cutting a masked video into guide clips: the plan, and what the node builds from it."""

from __future__ import annotations

import pytest
import torch

from minimax_h3_harness import comfyui_on_path, masked_guide_module


@pytest.fixture(scope="module")
def clips():
    comfyui_on_path()
    return masked_guide_module("clips")


@pytest.fixture(scope="module")
def node_module():
    comfyui_on_path()
    return masked_guide_module("nodes")


@pytest.fixture(scope="module")
def forward_module():
    return masked_guide_module("masked_h3_forward")


# --- the clip-length ladder ------------------------------------------------


@pytest.mark.parametrize("length,expected", [
    (0, 0), (1, 1), (2, 1), (4, 1), (5, 5), (6, 5), (21, 5), (22, 22),
    (23, 22), (38, 22), (39, 39), (40, 39), (56, 56), (124, 124),
])
def test_runs_round_down_to_a_length_h3_accepts(clips, length, expected):
    assert clips.permissible_clip_length(length) == expected


def test_every_permissible_length_is_one_core_would_not_truncate(clips):
    """Core's own rule: sub-5 batches become a still, the rest are cropped to 17k + 5."""
    for length in range(1, 200):
        n = clips.permissible_clip_length(length)
        assert n == 1 or n % 17 == 5
        assert n <= length


def test_latent_time_tokens_cover_the_clip_exactly(clips):
    # FRAME_PER_TOKEN is (1, 4, 4, 4, 4), so 5 tokens span 17 frames
    assert clips.frame_groups(2) == [(0, 1), (1, 5)]
    assert clips.frames_in_latent_t(2) == 5
    assert clips.frames_in_latent_t(7) == 22
    assert clips.frames_in_latent_t(12) == 39


# --- frame selection ------------------------------------------------------


def _keep(*flags):
    return torch.tensor([bool(f) for f in flags])


def test_frames_whose_mask_is_empty_are_dropped(clips):
    masks = torch.stack([torch.ones(8, 8), torch.zeros(8, 8), torch.full((8, 8), 0.01)])
    assert clips.frame_keep_flags(masks).tolist() == [True, False, True]


def test_min_coverage_also_drops_barely_visible_frames(clips):
    masks = torch.stack([torch.ones(8, 8), torch.full((8, 8), 0.05)])
    assert clips.frame_keep_flags(masks, min_coverage=0.1).tolist() == [True, False]


# --- the plan -------------------------------------------------------------


def test_one_guide_per_run_of_kept_frames(clips):
    keep = _keep(*([1] * 22 + [0] * 3 + [1] * 6))
    assert clips.plan_video_guides(keep, frame_idx=0, frame_count=124) == [
        (0, 0, 22), (25, 25, 5)]


def test_an_empty_frame_in_the_middle_splits_the_run(clips):
    """Deliberate in V1: a single dropped frame cuts a 40 frame run into two short ones."""
    keep = _keep(*([1] * 20 + [0] + [1] * 19))
    assert clips.plan_video_guides(keep, frame_idx=0, frame_count=124) == [
        (0, 0, 5), (21, 21, 5)]


def test_a_run_too_short_for_a_clip_still_anchors_a_single_frame(clips):
    assert clips.plan_video_guides(_keep(0, 1, 1, 1, 0), frame_idx=0, frame_count=124) == [
        (1, 1, 1)]


def test_frame_idx_offsets_the_whole_guide_video(clips):
    keep = _keep(*([1] * 22))
    assert clips.plan_video_guides(keep, frame_idx=30, frame_count=124) == [(0, 30, 22)]


def test_frames_past_the_end_of_the_target_are_dropped_before_rounding(clips):
    """Clipping first is what keeps a chunk from ever overflowing the target video."""
    keep = _keep(*([1] * 40))
    # target has 30 frames left, so 30 frames survive and round down to 22
    assert clips.plan_video_guides(keep, frame_idx=94, frame_count=124) == [(0, 94, 22)]


def test_frames_before_the_start_of_the_target_are_dropped(clips):
    keep = _keep(*([1] * 30))
    assert clips.plan_video_guides(keep, frame_idx=-10, frame_count=124) == [(10, 0, 5)]


def test_center_alignment_keeps_the_middle_of_the_run(clips):
    keep = _keep(*([1] * 30))
    assert clips.plan_video_guides(keep, frame_idx=0, frame_count=124, align="start") == [(0, 0, 22)]
    assert clips.plan_video_guides(keep, frame_idx=0, frame_count=124, align="center") == [(4, 4, 22)]


def test_an_unknown_alignment_is_refused(clips):
    with pytest.raises(ValueError, match="chunk alignment"):
        clips.plan_video_guides(_keep(1), frame_idx=0, frame_count=124, align="end")


# --- pooling a time-varying mask onto the token grid ----------------------


def _strengths(clips, masks, **kwargs):
    params = dict(token_t=2, token_h=1, token_w=2)
    params.update(kwargs)
    return clips.clip_token_strengths(clips.masks_to_canvas(masks, 64, 32), **params)


def test_each_latent_token_pools_only_the_frames_it_was_encoded_from(clips):
    """Token 0 covers frame 0 alone; token 1 covers frames 1-4."""
    masks = torch.zeros(5, 32, 64)
    masks[0] = 1.0
    s = _strengths(clips, masks)
    assert s.shape == (4,)                      # token_t * token_h * token_w, (t, h, w) order
    assert s[:2].tolist() == [1.0, 1.0]         # frame 0 -> token 0
    assert s[2:].tolist() == [0.0, 0.0]         # frames 1-4 were empty


def test_average_time_pooling_weights_a_token_by_how_many_frames_carry_the_mask(clips):
    masks = torch.zeros(5, 32, 64)
    masks[1] = 1.0                              # 1 of the 4 frames behind token 1
    s = _strengths(clips, masks, time_pooling="average")
    assert s[2].item() == pytest.approx(0.25, abs=1.0 / 510.0)


def test_max_time_pooling_takes_the_union_across_the_frames(clips):
    masks = torch.zeros(5, 32, 64)
    masks[1, :, :32] = 1.0                      # left half in one frame
    masks[3, :, 32:] = 1.0                      # right half in another
    s = _strengths(clips, masks, time_pooling="max")
    assert s[2:].tolist() == [1.0, 1.0]


def test_a_mask_count_that_does_not_match_the_clip_is_refused(clips):
    with pytest.raises(ValueError, match="5 pixel frames, but 4 masks"):
        _strengths(clips, torch.ones(4, 32, 64))


def test_an_unknown_time_pooling_is_refused(clips):
    with pytest.raises(ValueError, match="time pooling"):
        _strengths(clips, torch.ones(5, 32, 64), time_pooling="median")


# --- the node -------------------------------------------------------------


class _Vae:
    """Stands in for the H3 video VAE: 16x spatial, H3's own temporal grid."""

    def __init__(self, clips):
        self._clips = clips

    def encode(self, frames):
        n, h, w, _ = frames.shape
        latent_t = next(t for t in range(1, n + 2) if self._clips.frames_in_latent_t(t) == n)
        return torch.zeros(1, 24, latent_t, h // 16, w // 16)


def _av_latent(width=128, height=64, latent_t=37, audio_t=5):
    import comfy.nested_tensor

    video = torch.zeros(1, 24, latent_t, height // 16, width // 16)
    audio = torch.zeros(1, 32, 2, audio_t)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _add(node_module, clips, video, mask, **kwargs):
    params = dict(positive=[[torch.zeros(1, 4, 16), {}]], latent=_av_latent(), vae=_Vae(clips),
                  video=video, mask=mask, frame_idx=0)
    params.update(kwargs)
    return node_module.vloMiniMaxH3AddMaskedGuidesFromVideo.execute(**params)


def _video_and_mask(n, kept):
    """`n` frames whose masks are full for the frame indices in `kept` and empty elsewhere."""
    masks = torch.zeros(n, 64, 128)
    masks[list(kept)] = 1.0
    return torch.zeros(n, 64, 128, 3), masks


def test_a_masked_video_becomes_several_guide_clips(node_module, clips, forward_module):
    video, masks = _video_and_mask(60, list(range(0, 22)) + list(range(30, 40)))
    positive, plan = _add(node_module, clips, video, masks).result
    keyframes = positive[0][1]["minimax_keyframes"]
    assert [kf["resolved_frame_index"] for kf in keyframes] == [0, 30]
    assert [kf["latent"].shape[2] for kf in keyframes] == [7, 2]   # 22 and 5 frames
    assert ("2 masked guide clip(s) covering 27 of 60 frames (28 dropped by the mask, "
            "0 outside the target video, 5 to clip-length rounding)") in plan
    assert "condition tokens riding every sampling step: 72" in plan


def test_each_clip_carries_strengths_for_its_own_condition_rows(node_module, clips, forward_module):
    video, masks = _video_and_mask(22, range(22))
    positive, _ = _add(node_module, clips, video, masks).result
    keyframe = positive[0][1]["minimax_keyframes"][0]
    spec = keyframe[forward_module.MASKED_GUIDE_KEY]
    latent = keyframe["latent"]
    rows = latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)
    assert spec["strengths"].numel() == rows
    assert (spec["token_t"], spec["token_h"], spec["token_w"]) == (7, 2, 4)
    assert torch.equal(spec["strengths"], torch.ones(rows, dtype=torch.float64))


def test_the_clips_it_builds_line_up_with_the_forward_pass(node_module, clips, forward_module):
    """The row-alignment check the forked forward runs is the real contract here."""
    video, masks = _video_and_mask(30, range(30))
    positive, _ = _add(node_module, clips, video, masks).result
    keyframes = positive[0][1]["minimax_keyframes"]
    payload = {"keyframes": keyframes,
               "cond_video_latents": [kf["latent"] for kf in keyframes]}
    plan = forward_module.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)
    assert plan.aug_rows.numel() == sum(
        forward_module._latent_cond_rows(kf["latent"]) for kf in keyframes)


def test_guides_are_appended_to_whatever_is_already_on_the_conditioning(node_module, clips):
    video, masks = _video_and_mask(22, range(22))
    first = _add(node_module, clips, video, masks).result[0]
    second = _add(node_module, clips, video, masks, positive=first, frame_idx=40).result[0]
    assert [kf["resolved_frame_index"] for kf in second[0][1]["minimax_keyframes"]] == [0, 40]


def test_a_single_mask_is_broadcast_across_the_whole_clip(node_module, clips):
    video = torch.zeros(22, 64, 128, 3)
    positive, _ = _add(node_module, clips, video, torch.ones(1, 64, 128)).result
    assert len(positive[0][1]["minimax_keyframes"]) == 1


def test_a_mask_count_it_cannot_pair_up_is_refused(node_module, clips):
    with pytest.raises(ValueError, match="22 guide frames and 3 masks"):
        _add(node_module, clips, torch.zeros(22, 64, 128, 3), torch.ones(3, 64, 128))


def test_mismatched_mask_framing_is_refused(node_module, clips):
    video, _ = _video_and_mask(22, range(22))
    with pytest.raises(ValueError, match="aspect ratio"):
        _add(node_module, clips, video, torch.ones(22, 128, 64))


def test_a_video_that_guides_nothing_is_refused_rather_than_adding_no_guides(node_module, clips):
    video = torch.zeros(22, 64, 128, 3)
    with pytest.raises(ValueError, match="no guide clips survive"):
        _add(node_module, clips, video, torch.zeros(22, 64, 128))


def test_negative_frame_idx_counts_from_the_end_of_the_target(node_module, clips):
    video, masks = _video_and_mask(22, range(22))
    positive, _ = _add(node_module, clips, video, masks, frame_idx=-22).result
    # latent_t 37 -> 7 * 17 + 1 + 4 = 124 pixel frames
    assert positive[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 102


# --- the canvas crop happens before the keep decision, not after ----------


def test_coverage_is_measured_after_the_canvas_crop(clips):
    """A subject sitting in a band the cover-crop discards must not pass min_coverage.

    Judged on the raw mask it looks well covered; judged on the canvas it is empty,
    and it is the canvas that the token pooling will see."""
    masks = torch.zeros(5, 64, 64)          # 1:1 source
    masks[:, :12, :] = 1.0                  # only in the top band
    canvas = clips.masks_to_canvas(masks, 128, 64)   # 2:1 canvas -> crops top/bottom
    assert float(clips.frame_coverage(masks)[0]) > 0.1        # raw: looks covered
    assert float(clips.frame_coverage(canvas)[0]) == 0.0      # canvas: nothing left
    assert clips.frame_keep_flags(canvas).tolist() == [False] * 5


def test_a_frame_cropped_empty_never_becomes_an_all_noise_guide(node_module, clips):
    """The whole point of the keep decision: an all-zero guide is not an absent
    guide, it is a segment of pure-noise condition tokens riding every step."""
    video = torch.zeros(5, 64, 64, 3)
    masks = torch.zeros(5, 64, 64)
    masks[:, :12, :] = 1.0
    with pytest.raises(ValueError, match="no guide clips survive"):
        _add(node_module, clips, video, masks, latent=_av_latent(width=128, height=64))


def test_a_subject_the_crop_keeps_still_guides(node_module, clips, forward_module):
    """The mirror case, so the fix cannot pass by simply dropping everything."""
    video = torch.zeros(5, 64, 64, 3)
    masks = torch.zeros(5, 64, 64)
    masks[:, 16:48, :] = 1.0                # exactly the band a 2:1 crop keeps
    positive, _ = _add(node_module, clips, video, masks,
                       latent=_av_latent(width=128, height=64)).result
    spec = positive[0][1]["minimax_keyframes"][0][forward_module.MASKED_GUIDE_KEY]
    assert float(spec["strengths"].min()) == 1.0
