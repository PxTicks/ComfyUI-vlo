from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


def _video_latent(mask_frames: torch.Tensor, channels: int = 4) -> dict:
    """A plain video latent whose mask is stored the way SetLatentNoiseMask stores it."""
    frames, height, width = mask_frames.shape
    return {
        "samples": torch.zeros(1, channels, frames, height, width),
        "noise_mask": mask_frames.reshape(frames, 1, height, width).clone(),
    }


def _feather(nodes_module, latent, **kwargs):
    return nodes_module.vloFeatherVideoLatentMask.execute(latent=latent, **kwargs).result[0]


def _frames(output) -> torch.Tensor:
    mask = output["noise_mask"]
    return mask.reshape(-1, *mask.shape[-2:])


def _timeline(output, row: int = 0, col: int = 0) -> list[float]:
    return [round(value, 4) for value in _frames(output)[:, row, col].tolist()]


def test_schema_defaults_to_an_unheld_outer_feather(nodes_module) -> None:
    schema = nodes_module.vloFeatherVideoLatentMask.GET_SCHEMA()
    inputs = {input_spec.id: input_spec.as_dict() for input_spec in schema.inputs}

    assert inputs["mode"]["default"] == "outer"
    assert inputs["curve"]["default"] == "cosine"
    assert inputs["lead_ramp"]["default"] == inputs["tail_ramp"]["default"] == 2
    assert inputs["lead_hold"]["default"] == inputs["tail_hold"]["default"] == 0
    assert inputs["floor"]["default"] == 0.0
    assert inputs["original_latent"]["optional"] is True


def test_outer_ramp_frames_borrow_the_nearest_masked_frames_shape(nodes_module) -> None:
    mask = torch.zeros(10, 4, 4)
    mask[4:7, :2, :2] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), lead_ramp=2, tail_ramp=2, curve="linear"
    )
    frames = _frames(output)

    # Inside the shape the masked frames stay solid and the ramp decays outward.
    assert _timeline(output) == [0.0, 0.0, 0.3333, 0.6667, 1.0, 1.0, 1.0, 0.6667, 0.3333, 0.0]
    # Outside it nothing changes, so the fade keeps the shape it fades.
    assert torch.equal(frames[:, 2:, :], torch.zeros(10, 2, 4))
    assert torch.equal(frames[:, :, 2:], torch.zeros(10, 4, 2))


@pytest.mark.parametrize("mode", ["outer", "centered", "inner"])
@pytest.mark.parametrize("curve", ["cosine", "linear", "smoothstep", "exponential"])
@pytest.mark.parametrize(
    ("lead_ramp", "tail_ramp", "lead_hold", "tail_hold"),
    [(3, 3, 0, 0), (2, 5, 0, 0), (4, 1, 1, 2)],
)
def test_a_binary_timeline_feathers_exactly_like_the_audio_feather(
    nodes_module, mode, curve, lead_ramp, tail_ramp, lead_hold, tail_hold
) -> None:
    """Wherever the audio feather succeeds, the two streams get the same ramp.

    The runs are wide enough for every erosion here and far enough apart that
    the holds never merge them, which is where the two methods could differ.
    """
    timeline = torch.zeros(40)
    timeline[0:10] = 1.0  # touches the clip start, which is not a seam
    timeline[18:30] = 1.0
    timeline[34:40] = 1.0  # touches the clip end

    kernel = nodes_module.temporal_feather._feather_kernel(
        curve,
        lead_hold=lead_hold,
        lead_ramp=lead_ramp,
        tail_hold=tail_hold,
        tail_ramp=tail_ramp,
        max_offset=timeline.numel() - 1,
    )
    audio = nodes_module.audio_masks._feather_audio_mask(
        timeline.view(1, -1),
        time_axis=1,
        mode=mode,
        kernel=kernel,
        lead_ramp=lead_ramp,
        tail_ramp=tail_ramp,
        floor=0.0,
    ).view(-1)
    video = nodes_module.latent_masks._feather_video_mask(
        timeline.view(-1, 1, 1),
        mode=mode,
        curve=curve,
        lead_ramp=lead_ramp,
        tail_ramp=tail_ramp,
        lead_hold=lead_hold,
        tail_hold=tail_hold,
        floor=0.0,
    ).view(-1)

    assert torch.allclose(video, audio, atol=1e-6)


def test_inner_mode_never_erases_a_run_shorter_than_the_ramp(nodes_module) -> None:
    """A moving mask's edge pixels are covered for a frame or two.

    The audio feather's erosion would reject such a run outright, which per pixel
    would reject almost every moving mask. The video feather lets it peak lower.
    """
    mask = torch.zeros(10, 1, 1)
    mask[5] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), mode="inner", lead_ramp=3, tail_ramp=3,
        curve="linear",
    )

    # Depth 1 from both edges weighs (3 - 1 + 1) / 4 along the ramp: 1 - 0.75.
    assert _timeline(output) == [0.0] * 5 + [0.25] + [0.0] * 4

    kernel = nodes_module.temporal_feather._feather_kernel(
        "linear", lead_hold=0, lead_ramp=3, tail_hold=0, tail_ramp=3, max_offset=9
    )
    with pytest.raises(ValueError, match="erasing 1 masked step"):
        nodes_module.audio_masks._feather_audio_mask(
            mask.view(1, -1), time_axis=1, mode="inner", kernel=kernel,
            lead_ramp=3, tail_ramp=3, floor=0.0,
        )


def test_inner_mode_never_touches_a_preserved_frame(nodes_module) -> None:
    mask = torch.zeros(12, 2, 2)
    mask[3:9] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), mode="inner", lead_ramp=2, tail_ramp=2,
        curve="linear",
    )

    assert _timeline(output) == [
        0.0, 0.0, 0.0, 0.3333, 0.6667, 1.0, 1.0, 0.6667, 0.3333, 0.0, 0.0, 0.0,
    ]


def test_each_region_fades_on_its_own_schedule(nodes_module) -> None:
    """One region stopping while another carries on still gets its fade.

    A per-frame coverage test would see every frame as covered and fade nothing.
    """
    mask = torch.zeros(10, 1, 2)
    mask[:, 0, 0] = 1.0  # covers the whole clip
    mask[:5, 0, 1] = 1.0  # stops halfway

    output = _feather(
        nodes_module, _video_latent(mask), lead_ramp=2, tail_ramp=2, curve="linear"
    )

    assert _timeline(output, 0, 0) == [1.0] * 10
    assert _timeline(output, 0, 1) == [1.0] * 5 + [0.6667, 0.3333, 0.0, 0.0, 0.0]


def test_a_run_touching_the_clip_edge_is_not_feathered_there(nodes_module) -> None:
    mask = torch.zeros(8, 1, 1)
    mask[:4] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), mode="inner", lead_ramp=2, tail_ramp=2,
        curve="linear",
    )

    assert _timeline(output) == [1.0, 1.0, 0.6667, 0.3333, 0.0, 0.0, 0.0, 0.0]


def test_neighbouring_runs_merge_instead_of_cancelling(nodes_module) -> None:
    mask = torch.zeros(9, 1, 1)
    mask[2] = 1.0
    mask[6] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), lead_ramp=3, tail_ramp=3, curve="linear"
    )

    # Frame 4 sits two frames from both runs, so it takes one ramp's value, not a sum.
    assert _timeline(output) == [0.5, 0.75, 1.0, 0.75, 0.5, 0.75, 1.0, 0.75, 0.5]


def test_soft_mask_values_carry_into_the_ramp(nodes_module) -> None:
    """Latent masks are bilinear-resized, so their spatial edges are fractional."""
    mask = torch.zeros(6, 1, 1)
    mask[3:] = 0.5

    output = _feather(
        nodes_module, _video_latent(mask), lead_ramp=1, tail_ramp=1, curve="linear"
    )

    assert _timeline(output) == [0.0, 0.0, 0.25, 0.5, 0.5, 0.5]


def test_both_mask_layouts_feather_along_time_and_keep_their_shape(nodes_module) -> None:
    mask = torch.zeros(6, 3, 3)
    mask[3:] = 1.0
    plain = _video_latent(mask)
    five_d = _video_latent(mask)
    five_d["noise_mask"] = mask.reshape(1, 1, 6, 3, 3).clone()

    plain_out = _feather(nodes_module, plain, lead_ramp=2, curve="linear")
    five_d_out = _feather(nodes_module, five_d, lead_ramp=2, curve="linear")

    assert tuple(plain_out["noise_mask"].shape) == (6, 1, 3, 3)
    assert tuple(five_d_out["noise_mask"].shape) == (1, 1, 6, 3, 3)
    assert torch.equal(_frames(plain_out), _frames(five_d_out))
    assert _timeline(plain_out) == [0.0, 0.3333, 0.6667, 1.0, 1.0, 1.0]


def test_a_single_frame_mask_has_no_time_edges(nodes_module) -> None:
    latent = _video_latent(torch.zeros(5, 2, 2))
    single = torch.zeros(1, 1, 2, 2)
    single[..., 0, 0] = 1.0
    latent["noise_mask"] = single

    output = _feather(nodes_module, latent, lead_ramp=3, tail_ramp=3, floor=0.1)

    expected = single.clamp(min=0.1)
    assert torch.equal(output["noise_mask"], expected)


def test_floor_lifts_the_whole_preserved_region(nodes_module) -> None:
    mask = torch.zeros(8, 1, 1)
    mask[4] = 1.0

    output = _feather(
        nodes_module, _video_latent(mask), lead_ramp=1, tail_ramp=1, curve="linear",
        floor=0.2,
    )

    assert _timeline(output) == [0.2, 0.2, 0.2, 0.5, 1.0, 0.5, 0.2, 0.2]


def test_zero_ramps_leave_the_mask_alone(nodes_module) -> None:
    mask = (torch.rand(7, 3, 3) > 0.5).float()

    output = _feather(
        nodes_module, _video_latent(mask), mode="inner", lead_ramp=0, tail_ramp=0
    )

    assert torch.equal(_frames(output), mask)


def test_an_unknown_curve_fails_even_when_the_ramps_are_zero(nodes_module) -> None:
    with pytest.raises(ValueError, match="Unsupported feather curve"):
        _feather(
            nodes_module, _video_latent(torch.zeros(4, 1, 1)), lead_ramp=0,
            tail_ramp=0, curve="sigmoid",
        )


def test_nested_av_latent_feathers_only_the_video_mask(nodes_module) -> None:
    nt = nodes_module.comfy.nested_tensor
    video_mask = torch.zeros(6, 1, 2, 2)
    video_mask[3:] = 1.0
    audio_mask = torch.zeros(1, 32, 2, 40)
    audio_mask[..., 10:20] = 1.0
    latent = {
        "samples": nt.NestedTensor((torch.zeros(1, 4, 6, 2, 2), torch.zeros(1, 32, 2, 40))),
        "noise_mask": nt.NestedTensor((video_mask, audio_mask)),
    }

    output = _feather(nodes_module, latent, lead_ramp=1, tail_ramp=1, curve="linear")
    feathered_video, kept_audio = output["noise_mask"].unbind()

    assert feathered_video.reshape(6, -1)[:, 0].tolist() == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    assert torch.equal(kept_audio, audio_mask)


def test_a_plain_mask_on_a_joint_latent_leaves_audio_fully_generated(nodes_module) -> None:
    nt = nodes_module.comfy.nested_tensor
    video_mask = torch.zeros(6, 1, 2, 2)
    video_mask[3:] = 1.0
    latent = {
        "samples": nt.NestedTensor((torch.zeros(1, 4, 6, 2, 2), torch.zeros(1, 32, 2, 40))),
        "noise_mask": video_mask,
    }

    output = _feather(nodes_module, latent, lead_ramp=1, tail_ramp=1)
    _, audio = output["noise_mask"].unbind()

    # The sampler would fill an absent stream mask with ones; say so explicitly.
    assert torch.equal(audio, torch.ones(1, 1, 2, 40))


def test_a_latent_without_a_video_mask_fails_clearly(nodes_module) -> None:
    latent = {"samples": torch.zeros(1, 4, 6, 2, 2)}
    with pytest.raises(ValueError, match="no video noise mask"):
        _feather(nodes_module, latent)


def test_an_image_latent_has_no_time_axis_to_feather(nodes_module) -> None:
    latent = {"samples": torch.zeros(3, 4, 8, 8), "noise_mask": torch.ones(3, 1, 8, 8)}
    with pytest.raises(ValueError, match="no time axis"):
        _feather(nodes_module, latent)


def test_a_pixel_frame_mask_is_refused_rather_than_ramped_in_the_wrong_units(
    nodes_module,
) -> None:
    latent = _video_latent(torch.zeros(7, 2, 2))
    latent["noise_mask"] = torch.zeros(25, 1, 2, 2)  # source frames, not latent frames
    with pytest.raises(ValueError, match="vlo Mask to Latent Mask"):
        _feather(nodes_module, latent)


def _composited_chain(nodes_module, frames: int = 8):
    """A joint latent past the blank composite, plus the pre-composite latent."""
    nt = nodes_module.comfy.nested_tensor
    video = torch.randn(1, 4, frames, 4, 4)
    audio = torch.randn(1, 32, 2, 40)
    video_mask = torch.zeros(frames, 1, 4, 4)
    video_mask[3:6] = 1.0
    before = {
        "samples": nt.NestedTensor((video, audio)),
        "noise_mask": nt.NestedTensor((video_mask, torch.zeros(1, 32, 2, 40))),
    }
    blank = before.copy()
    blank["samples"] = before["samples"] * 0.0
    after = nodes_module.vloLatentCompositeMasked.execute(
        destination=before, source=blank, force_binary_mask=True
    ).result[0]
    return before, after, video


@pytest.mark.parametrize("mode", ["outer", "centered", "inner"])
def test_every_mode_blends_toward_real_footage_when_the_original_is_supplied(
    nodes_module, mode
) -> None:
    before, after, video = _composited_chain(nodes_module)

    output = _feather(
        nodes_module, after, mode=mode, lead_ramp=2, tail_ramp=2,
        original_latent=before,
    )
    samples, _ = output["samples"].unbind()
    # The mask is uniform within each frame, so one pixel per frame suffices.
    per_frame = output["noise_mask"].unbind()[0].reshape(8, -1)[:, 0]

    for frame, value in enumerate(per_frame.tolist()):
        if value < 1.0:
            # Anything the sampler reads back from the latent must be footage.
            assert torch.equal(samples[:, :, frame], video[:, :, frame]), frame
        else:
            assert torch.equal(samples[:, :, frame], torch.zeros_like(video[:, :, frame]))


def test_inner_ramps_blend_toward_a_blank_latent_without_the_original(nodes_module) -> None:
    _, after, video = _composited_chain(nodes_module)

    output = _feather(nodes_module, after, mode="inner", lead_ramp=1, tail_ramp=1)
    samples, _ = output["samples"].unbind()

    # Frame 3 is now a ramp frame, but the composite already blanked it.
    assert output["noise_mask"].unbind()[0][3].max() < 1.0
    assert torch.equal(samples[:, :, 3], torch.zeros_like(video[:, :, 3]))


def test_outer_ramps_need_no_original_latent(nodes_module) -> None:
    _, after, video = _composited_chain(nodes_module)

    output = _feather(nodes_module, after, mode="outer", lead_ramp=2, tail_ramp=2)
    samples, _ = output["samples"].unbind()

    for frame in (1, 2, 6, 7):  # the outer ramp frames
        assert torch.equal(samples[:, :, frame], video[:, :, frame])


def test_a_mismatched_original_latent_is_rejected(nodes_module) -> None:
    _, after, _ = _composited_chain(nodes_module)
    wrong, _, _ = _composited_chain(nodes_module, frames=9)

    with pytest.raises(ValueError, match="same latent"):
        _feather(nodes_module, after, mode="inner", original_latent=wrong)


def test_the_feather_reaches_minimax_as_distinct_token_timesteps(nodes_module) -> None:
    """The ramp survives H3's token-grid pooling and sets per-token timesteps.

    comfy/ldm/minimax/model.py runs a masked video token at sigma = m * sigma_v,
    so a decaying mask is a genuine per-token denoise strength. This pins that
    contract to the real ComfyUI code, from node output to row timesteps.
    """
    import types

    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base
    import comfy.sampler_helpers
    import comfy.utils

    class _PooledLike(comfy.model_base.MiniMaxH3):
        def __init__(self):  # the real methods, none of the model machinery
            self.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))

    nt = nodes_module.comfy.nested_tensor
    video_shape, audio_shape = (1, 24, 7, 30, 52), (1, 32, 2, 80)
    video_mask = torch.zeros(7, 1, 30, 52)
    video_mask[4:] = 1.0
    latent = {
        "samples": nt.NestedTensor((torch.zeros(video_shape), torch.zeros(audio_shape))),
        "noise_mask": nt.NestedTensor((video_mask, torch.ones(audio_shape))),
    }

    feathered = _feather(nodes_module, latent, lead_ramp=3, tail_ramp=3, curve="linear")

    # comfy/samplers.py reshapes each stream mask to its full latent shape and
    # packs them before the model ever sees them.
    latent_shapes = [video_shape, audio_shape]
    packed, _ = comfy.utils.pack_latents(
        [
            comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
            for mask, shape in zip(feathered["noise_mask"].unbind(), latent_shapes)
        ]
    )
    values = _PooledLike()._denoise_mask_values(packed, latent_shapes)
    assert "denoise_mask" in values, "the model saw a fully-denoised video mask"

    rows = minimax_model.mask_row_values(values["denoise_mask"][0, 0], 7, 30, 52)
    per_frame = rows.reshape(7, -1)
    sigma_video = 0.4  # any mid-sampling video sigma
    frame_timesteps = (1.0 - per_frame * sigma_video).amax(dim=1)

    assert frame_timesteps[:1].tolist() == pytest.approx([1.0])
    assert frame_timesteps[1:4].tolist() == pytest.approx(
        [1.0 - 0.25 * sigma_video, 1.0 - 0.5 * sigma_video, 1.0 - 0.75 * sigma_video],
        abs=1 / 256,
    )
    assert frame_timesteps[4:].tolist() == pytest.approx([1.0 - sigma_video] * 3)
