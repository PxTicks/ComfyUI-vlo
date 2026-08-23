from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


class _LTXAudioVAE:
    __module__ = "comfy.ldm.lightricks.vae.audio_vae"

    latent_frequency_bins = 16
    latents_per_second = 25.0


class _MiniMaxAudioVAE:
    __module__ = "comfy.ldm.minimax.audio_vae"

    latents_per_second = 40.0
    output_sample_rate = 32000


class _UnknownAudioVAE:
    pass


class _VAEWrapper:
    def __init__(self, first_stage_model):
        self.first_stage_model = first_stage_model


def _frame_masks(frame_count: int, active_frames: tuple[int, ...]) -> torch.Tensor:
    masks = torch.zeros(frame_count, 4, 4)
    masks[list(active_frames)] = 1.0
    return masks


def test_generic_audio_mask_schema_exposes_automatic_metadata_inputs(
    nodes_module,
) -> None:
    schema = nodes_module.vloSetAudioLatentBinaryMasks.GET_SCHEMA()
    inputs = {input_spec.id: input_spec.as_dict() for input_spec in schema.inputs}

    assert inputs["audio_vae"]["optional"] is True
    assert inputs["layout_override"]["options"] == ["auto", "ltx", "minimax"]
    assert inputs["mask_fps"]["default"] == 0.0
    assert inputs["audio_latent_rate"]["default"] == 0.0


def test_ltx_layout_and_rate_resolve_from_audio_vae(nodes_module) -> None:
    latent = {"samples": torch.zeros(1, 128, 25, 16)}

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        audio_vae=_VAEWrapper(_LTXAudioVAE()),
        masks=_frame_masks(24, (12,)),
        mask_fps=24.0,
    ).result[0]

    noise_mask = output["noise_mask"]
    assert tuple(noise_mask.shape) == (1, 128, 25, 16)
    assert torch.nonzero(noise_mask[0, 0, :, 0]).flatten().tolist() == [12, 13]
    assert output["audio_latent_metadata"] == {
        "time_axis": 2,
        "layout_source": "ltx",
        "layout": "ltx",
        "latents_per_second": 25.0,
    }


def test_minimax_layout_and_rate_resolve_from_audio_vae(nodes_module) -> None:
    latent = {"samples": torch.zeros(1, 32, 2, 40)}

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        audio_vae=_VAEWrapper(_MiniMaxAudioVAE()),
        masks=_frame_masks(24, (12,)),
        mask_fps=24.0,
    ).result[0]

    noise_mask = output["noise_mask"]
    assert tuple(noise_mask.shape) == (1, 32, 2, 40)
    assert torch.nonzero(noise_mask[0, 0, 0]).flatten().tolist() == [20]
    assert torch.equal(noise_mask[:, :, 0], noise_mask[:, :, 1])
    assert output["audio_latent_metadata"] == {
        "time_axis": 3,
        "layout_source": "minimax",
        "layout": "minimax",
        "latents_per_second": 40.0,
    }


def test_latent_metadata_takes_priority_without_a_vae(nodes_module) -> None:
    latent = {
        "samples": torch.zeros(1, 8, 2, 10),
        "audio_latent_metadata": {
            "layout": ["batch", "channels", "stereo", "time"],
            "latents_per_second": 10.0,
        },
    }

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        masks=_frame_masks(5, (2,)),
        mask_fps=5.0,
    ).result[0]

    assert tuple(output["noise_mask"].shape) == (1, 8, 2, 10)
    assert output["audio_latent_metadata"]["time_axis"] == 3
    assert output["audio_latent_metadata"]["latents_per_second"] == 10.0


def test_explicit_overrides_support_an_unknown_vae(nodes_module) -> None:
    latent = {"samples": torch.zeros(1, 6, 2, 12)}

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        audio_vae=_VAEWrapper(_UnknownAudioVAE()),
        masks=_frame_masks(6, (3,)),
        mask_fps=6.0,
        layout_override="minimax",
        audio_latent_rate=12.0,
    ).result[0]

    assert tuple(output["noise_mask"].shape) == (1, 6, 2, 12)
    assert output["audio_latent_metadata"]["time_axis"] == 3
    assert output["audio_latent_metadata"]["latents_per_second"] == 12.0


def test_unknown_layout_fails_instead_of_guessing_from_shape(nodes_module) -> None:
    with pytest.raises(ValueError, match="Could not determine the audio latent time axis"):
        nodes_module.vloSetAudioLatentBinaryMasks.execute(
            audio_latent={"samples": torch.zeros(1, 8, 2, 10)},
            audio_vae=_VAEWrapper(_UnknownAudioVAE()),
            masks=torch.ones(5, 4, 4),
        )


def test_nested_av_latent_updates_only_audio_mask(nodes_module) -> None:
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.zeros(1, 32, 2, 40)
    video_mask = torch.zeros(1, 1, 2, 4, 4)
    audio_mask = torch.ones(1, 1, 2, 40)
    latent = {
        "samples": nodes_module.comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": nodes_module.comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        ),
    }

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        audio_vae=_VAEWrapper(_MiniMaxAudioVAE()),
        masks=_frame_masks(24, (12,)),
        mask_fps=24.0,
    ).result[0]

    output_masks = output["noise_mask"].unbind()
    assert len(output_masks) == 2
    assert torch.equal(output_masks[0], video_mask)
    assert tuple(output_masks[1].shape) == tuple(audio.shape)
    assert torch.nonzero(output_masks[1][0, 0, 0]).flatten().tolist() == [20]
    assert output["audio_latent_metadata"]["audio_stream_index"] == 1


def test_nested_av_latent_defaults_other_streams_to_fully_denoised(
    nodes_module,
) -> None:
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.zeros(1, 32, 2, 40)
    latent = {
        "samples": nodes_module.comfy.nested_tensor.NestedTensor((video, audio))
    }

    output = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        audio_vae=_VAEWrapper(_MiniMaxAudioVAE()),
        masks=torch.zeros(24, 4, 4),
        mask_fps=24.0,
    ).result[0]

    video_output_mask, audio_output_mask = output["noise_mask"].unbind()
    assert tuple(video_output_mask.shape) == (1, 1, 2, 4, 4)
    assert torch.all(video_output_mask == 1)
    assert torch.all(audio_output_mask == 0)


def test_existing_mask_add_and_subtract_modes(nodes_module) -> None:
    samples = torch.zeros(1, 4, 4, 2)
    existing = torch.zeros(1, 1, 4, 1)
    existing[:, :, 1:3] = 1.0
    latent = {"samples": samples, "noise_mask": existing}
    masks = _frame_masks(4, (2,))

    added = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        masks=masks,
        layout_override="ltx",
        existing_mask_mode="add",
    ).result[0]["noise_mask"]
    subtracted = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=latent,
        masks=masks,
        layout_override="ltx",
        existing_mask_mode="subtract",
    ).result[0]["noise_mask"]

    assert added[0, 0, :, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert subtracted[0, 0, :, 0].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_positive_mask_fps_requires_a_resolvable_audio_rate(nodes_module) -> None:
    with pytest.raises(ValueError, match="mask_fps requires an audio latent rate"):
        nodes_module.vloSetAudioLatentBinaryMasks.execute(
            audio_latent={
                "samples": torch.zeros(1, 8, 2, 10),
                "audio_latent_metadata": {"time_axis": 3},
            },
            masks=torch.ones(5, 4, 4),
            mask_fps=5.0,
        )
