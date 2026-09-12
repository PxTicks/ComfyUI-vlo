from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


class _MiniMaxAudioVAE:
    __module__ = "comfy.ldm.minimax.audio_vae"

    latents_per_second = 40.0
    output_sample_rate = 32000


class _VAEWrapper:
    def __init__(self, first_stage_model):
        self.first_stage_model = first_stage_model


def _minimax_latent(length: int, active: slice | None) -> dict:
    """A MiniMax [B, C, S, T] audio latent carrying a binary mask over `active`."""
    samples = torch.zeros(1, 32, 2, length)
    mask = torch.zeros(1, 32, 2, length)
    if active is not None:
        mask[..., active] = 1.0
    return {"samples": samples, "noise_mask": mask}


def _timeline(output) -> list[float]:
    return [round(value, 4) for value in output["noise_mask"][0, 0, 0].tolist()]


def _feather(**kwargs):
    node = kwargs.pop("nodes_module").vloFeatherAudioLatentMask
    kwargs.setdefault("audio_vae", _VAEWrapper(_MiniMaxAudioVAE()))
    return node.execute(**kwargs).result[0]


def test_schema_defaults_to_an_asymmetric_outer_feather(nodes_module) -> None:
    schema = nodes_module.vloFeatherAudioLatentMask.GET_SCHEMA()
    inputs = {input_spec.id: input_spec.as_dict() for input_spec in schema.inputs}

    assert inputs["mode"]["default"] == "outer"
    assert inputs["curve"]["default"] == "cosine"
    assert inputs["tail_ramp"]["default"] > inputs["lead_ramp"]["default"] > 0.0
    assert inputs["lead_hold"]["default"] == inputs["tail_hold"]["default"] == 0.0
    assert inputs["floor"]["default"] == 0.0
    assert inputs["audio_vae"]["optional"] is True


def test_outer_feather_keeps_the_core_solid_and_decays_outward(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(8, 12)),
        lead_ramp=0.075,  # 3 latent steps at 40 Hz
        tail_ramp=0.075,
        curve="linear",
    )
    timeline = _timeline(output)

    assert timeline[8:12] == [1.0, 1.0, 1.0, 1.0]
    assert timeline[5:8] == [0.25, 0.5, 0.75]
    assert timeline[12:15] == [0.75, 0.5, 0.25]
    assert timeline[:5] == [0.0] * 5
    assert timeline[15:] == [0.0] * 5


def test_ramp_length_in_steps_yields_that_many_nonzero_rungs(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(10, 12)),
        lead_ramp=0.025,  # exactly one latent step
        tail_ramp=0.0,
        curve="linear",
    )
    timeline = _timeline(output)

    # One rung, genuinely between the core and zero, and zero immediately beyond.
    assert timeline[9] == 0.5
    assert timeline[8] == 0.0
    assert timeline[12] == 0.0


def test_a_ramp_shorter_than_one_latent_step_still_softens_the_seam(
    nodes_module,
) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(12, slice(5, 7)),
        lead_ramp=0.001,
        tail_ramp=0.0,
        curve="linear",
    )

    assert _timeline(output)[4] == 0.5


def test_hold_extends_the_solid_core_before_the_ramp_begins(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(10, 12)),
        lead_ramp=0.05,
        lead_hold=0.05,
        tail_ramp=0.0,
        curve="linear",
    )
    timeline = _timeline(output)

    assert timeline[8:12] == [1.0, 1.0, 1.0, 1.0]
    assert timeline[6:8] == pytest.approx([1 / 3, 2 / 3], abs=1e-4)
    assert timeline[5] == 0.0


def test_centered_mode_straddles_the_original_edge(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(8, 14)),
        lead_ramp=0.1,  # 4 steps, so 2 fall inside the region
        tail_ramp=0.0,
        curve="linear",
        mode="centered",
    )
    timeline = _timeline(output)

    assert timeline[10:14] == [1.0, 1.0, 1.0, 1.0]
    assert timeline[6] < timeline[7] < timeline[8] < timeline[9] < 1.0
    assert timeline[5] == 0.0


def test_inner_mode_never_touches_a_preserved_step(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(8, 14)),
        lead_ramp=0.05,
        tail_ramp=0.05,
        curve="linear",
        mode="inner",
    )
    timeline = _timeline(output)

    assert timeline[:8] == [0.0] * 8
    assert timeline[14:] == [0.0] * 6
    assert 0.0 < timeline[8] < 1.0
    assert timeline[10:12] == [1.0, 1.0]


def test_neighbouring_regions_merge_instead_of_cancelling(nodes_module) -> None:
    latent = _minimax_latent(24, None)
    latent["noise_mask"][..., 6:8] = 1.0
    latent["noise_mask"][..., 12:14] = 1.0

    output = _feather(
        nodes_module=nodes_module,
        audio_latent=latent,
        lead_ramp=0.1,
        tail_ramp=0.1,
        curve="linear",
    )
    timeline = _timeline(output)

    assert timeline[6:8] == [1.0, 1.0]
    assert timeline[12:14] == [1.0, 1.0]
    # Each gap step takes the stronger of the two ramps reaching it, never their
    # sum, so the gap dips in the middle and nothing anywhere exceeds 1.
    assert timeline[8:12] == pytest.approx([0.8, 0.6, 0.6, 0.8], abs=1e-4)


def test_a_region_touching_the_clip_edge_is_not_feathered_there(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(12, slice(0, 4)),
        lead_ramp=0.1,
        tail_ramp=0.05,
        curve="linear",
    )
    timeline = _timeline(output)

    assert timeline[0:4] == [1.0, 1.0, 1.0, 1.0]
    assert 0.0 < timeline[4] < 1.0


def test_smooth_curves_flatten_where_the_ramp_meets_the_core(nodes_module) -> None:
    def first_rung(curve: str) -> float:
        output = _feather(
            nodes_module=nodes_module,
            audio_latent=_minimax_latent(20, slice(10, 12)),
            lead_ramp=0.1,
            tail_ramp=0.0,
            curve=curve,
        )
        return _timeline(output)[9]

    # A flat start means the first step away from the core barely drops.
    assert first_rung("cosine") > first_rung("linear")
    assert first_rung("smoothstep") > first_rung("linear")
    assert first_rung("exponential") < first_rung("linear")


def test_floor_lifts_the_whole_preserved_region(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(12, slice(5, 7)),
        lead_ramp=0.025,
        tail_ramp=0.025,
        floor=0.1,
    )
    timeline = _timeline(output)

    assert min(timeline) == pytest.approx(0.1)
    assert timeline[5:7] == [1.0, 1.0]


def test_zero_durations_leave_the_mask_alone(nodes_module) -> None:
    latent = _minimax_latent(12, slice(4, 8))
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=latent,
        lead_ramp=0.0,
        tail_ramp=0.0,
    )

    assert torch.equal(output["noise_mask"], latent["noise_mask"])


def test_nested_av_latent_feathers_only_the_audio_mask(nodes_module) -> None:
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.zeros(1, 32, 2, 20)
    video_mask = torch.zeros(1, 1, 2, 4, 4)
    audio_mask = torch.zeros(1, 1, 2, 20)
    audio_mask[..., 8:12] = 1.0
    nested = nodes_module.comfy.nested_tensor.NestedTensor
    latent = {
        "samples": nested((video, audio)),
        "noise_mask": nested((video_mask, audio_mask)),
    }

    output = _feather(
        nodes_module=nodes_module,
        audio_latent=latent,
        lead_ramp=0.05,
        tail_ramp=0.05,
        curve="linear",
    )
    output_video_mask, output_audio_mask = output["noise_mask"].unbind()

    assert torch.equal(output_video_mask, video_mask)
    assert tuple(output_audio_mask.shape) == tuple(audio.shape)
    timeline = [round(value, 4) for value in output_audio_mask[0, 0, 0].tolist()]
    assert timeline[8:12] == [1.0, 1.0, 1.0, 1.0]
    assert timeline[6:8] == pytest.approx([1 / 3, 2 / 3], abs=1e-4)
    assert timeline[12:14] == pytest.approx([2 / 3, 1 / 3], abs=1e-4)


def test_a_latent_without_an_audio_mask_fails_clearly(nodes_module) -> None:
    with pytest.raises(ValueError, match="no audio noise mask to feather"):
        _feather(
            nodes_module=nodes_module,
            audio_latent={"samples": torch.zeros(1, 32, 2, 20)},
        )


def test_durations_require_a_resolvable_audio_rate(nodes_module) -> None:
    with pytest.raises(ValueError, match="requires an audio latent rate"):
        latent = _minimax_latent(20, slice(8, 12))
        # A time axis with no rate alongside it: nothing can convert seconds to steps.
        latent["audio_latent_metadata"] = {"time_axis": 3}
        nodes_module.vloFeatherAudioLatentMask.execute(
            audio_latent=latent,
            lead_ramp=0.1,
        )


def test_the_feathered_mask_survives_the_models_token_grid_quantisation(
    nodes_module,
) -> None:
    """MiniMax rounds the mask to 1/256 before deriving per-row timesteps."""
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(8, 12)),
        lead_ramp=0.1,
        tail_ramp=0.1,
    )
    quantised = torch.ceil(output["noise_mask"] * 256.0) / 256.0
    timeline = quantised[0, 0, 0].tolist()

    assert len(set(timeline)) > 3
    assert timeline[8:12] == [1.0, 1.0, 1.0, 1.0]


def test_inner_mode_refuses_to_erode_a_region_away(nodes_module) -> None:
    with pytest.raises(ValueError, match="erasing 2 masked step"):
        _feather(
            nodes_module=nodes_module,
            audio_latent=_minimax_latent(20, slice(10, 12)),  # 2 steps
            lead_ramp=0.1,  # 4 steps, so nothing survives the erosion
            tail_ramp=0.1,
            mode="inner",
        )


def test_a_short_region_cannot_vanish_beside_a_surviving_one(nodes_module) -> None:
    """A whole-mask peak check would miss this: the long region keeps it non-zero."""
    latent = _minimax_latent(50, None)
    latent["noise_mask"][..., 5:7] = 1.0  # too narrow for the ramp
    latent["noise_mask"][..., 30:45] = 1.0  # wide enough to survive

    with pytest.raises(ValueError, match="erasing 2 masked step"):
        _feather(
            nodes_module=nodes_module,
            audio_latent=latent,
            lead_ramp=0.1,
            tail_ramp=0.1,
            mode="inner",
        )


def test_a_region_exactly_wide_enough_survives_the_erosion(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(30, slice(10, 19)),  # 9 steps, eroded by 4 + 4
        lead_ramp=0.1,
        tail_ramp=0.1,
        mode="inner",
        curve="linear",
    )
    timeline = _timeline(output)

    assert timeline[14] == 1.0
    assert timeline[:10] == [0.0] * 10
    assert timeline[19:] == [0.0] * 11


def test_outer_mode_handles_a_ramp_longer_than_the_clip(nodes_module) -> None:
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(10, 12)),
        lead_ramp=99.0,
        tail_ramp=99.0,
    )
    timeline = _timeline(output)

    assert timeline[10:12] == [1.0, 1.0]
    assert all(value > 0.0 for value in timeline)


def test_a_ramp_running_off_the_clip_keeps_its_requested_decay_rate(
    nodes_module,
) -> None:
    """A ramp with no room to finish is truncated, never steepened.

    Its length is the curve's denominator, so shortening it to fit the clip would
    change the rate: the same setting would decay differently on a short clip.
    """
    # 2 s at 40 Hz is an 80-step ramp with only 18 steps of clip to run into.
    output = _feather(
        nodes_module=nodes_module,
        audio_latent=_minimax_latent(20, slice(18, 20)),
        lead_ramp=2.0,
        tail_ramp=0.0,
        curve="linear",
    )
    timeline = _timeline(output)

    # Every step follows the 80-step ramp, so the clip start is still near 1.
    assert timeline == pytest.approx(
        [1.0 - (18 - index) / 81 for index in range(18)] + [1.0, 1.0], abs=1e-3
    )
    assert timeline[0] == pytest.approx(1.0 - 18 / 81, abs=1e-3)


def test_a_ramp_that_fits_is_unaffected_by_the_offset_bound(nodes_module) -> None:
    short_clip = _timeline(
        _feather(
            nodes_module=nodes_module,
            audio_latent=_minimax_latent(14, slice(10, 12)),
            lead_ramp=0.1,
            tail_ramp=0.0,
            curve="linear",
        )
    )
    long_clip = _timeline(
        _feather(
            nodes_module=nodes_module,
            audio_latent=_minimax_latent(40, slice(10, 12)),
            lead_ramp=0.1,
            tail_ramp=0.0,
            curve="linear",
        )
    )

    assert short_clip[:14] == pytest.approx(long_clip[:14], abs=1e-6)


def test_an_unknown_curve_fails_even_when_the_ramps_round_to_nothing(
    nodes_module,
) -> None:
    with pytest.raises(ValueError, match="Unsupported feather curve"):
        _feather(
            nodes_module=nodes_module,
            audio_latent=_minimax_latent(20, slice(8, 12)),
            lead_ramp=0.0,
            tail_ramp=0.0,
            curve="gaussian",
        )
