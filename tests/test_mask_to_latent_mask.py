from __future__ import annotations

import math

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


class _UniformVideoVae:
    """A fixed-factor causal video VAE, as LTX/Wan/Cosmos expose themselves."""

    crop_input = True

    def spacial_compression_encode(self):
        return self.downscale_ratio[-1]

    def __init__(self, temporal_factor: int, spatial_factor: int = 8):
        self.latent_dim = 3
        self.downscale_ratio = (
            lambda frames, f=temporal_factor: max(0, math.floor((frames + f - 1) / f)),
            spatial_factor,
            spatial_factor,
        )
        self.downscale_index_formula = (temporal_factor, spatial_factor, spatial_factor)


class _MiniMaxModel:
    """The real MiniMax H3 clip geometry from comfy/ldm/minimax/vae.py."""

    vae_ratio_t = 4
    clip_length = 17
    token_drop = 3
    frame_pre_padding = (-17) % 4  # 3
    tokens_chunk_size = math.ceil(17 / 4)  # 5


class _MiniMaxVae:
    """Matches how comfy/sd.py configures the MiniMax H3 video VAE."""

    latent_dim = 3
    crop_input = True
    # Note the index formula advertises 4, which is the intra-clip token ratio,
    # NOT a usable "frame 0 then blocks of 4" grouping.
    downscale_index_formula = (4, 16, 16)
    downscale_ratio = (
        lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1,
        16,
        16,
    )

    def spacial_compression_encode(self):
        return type(self).downscale_ratio[-1]

    def __init__(self):
        self.first_stage_model = _MiniMaxModel()


class _ImageVae:
    latent_dim = 2
    downscale_ratio = 8
    downscale_index_formula = None
    crop_input = True

    def spacial_compression_encode(self):
        return self.downscale_ratio


def _video_latent(frames: int, height: int, width: int, channels: int = 24):
    return {"samples": torch.zeros(1, channels, frames, height, width)}


def _ramp_masks(frame_count: int, size: int = 8) -> torch.Tensor:
    """One distinct flat value per source frame, so grouping is readable."""
    values = torch.arange(frame_count, dtype=torch.float32) / max(1, frame_count)
    return values.view(-1, 1, 1).expand(frame_count, size, size).clone()


def _groups_from_output(out: torch.Tensor, masks: torch.Tensor) -> list[list[int]]:
    """Recover which source frames landed in each latent frame, via max pooling."""
    source_values = masks[:, 0, 0].tolist()
    recovered = []
    for latent_value in out[:, 0, 0].tolist():
        recovered.append(
            [i for i, v in enumerate(source_values) if v == pytest.approx(latent_value)]
        )
    return recovered


# --- temporal mapping ---------------------------------------------------------


def test_uniform_vae_groups_first_frame_alone_then_blocks(nodes_module) -> None:
    groups = nodes_module._vae_temporal_groups(_UniformVideoVae(8), 25)
    assert groups == [(0, 1), (1, 9), (9, 17), (17, 25)]


def test_uniform_vae_factor_four(nodes_module) -> None:
    groups = nodes_module._vae_temporal_groups(_UniformVideoVae(4), 9)
    assert groups == [(0, 1), (1, 5), (5, 9)]


def test_minimax_uses_real_clip_geometry_not_the_index_formula(nodes_module) -> None:
    # 22 frames = 2 clips of 17 (second padded), 10 tokens, last 3 dropped -> 7.
    groups = nodes_module._vae_temporal_groups(_MiniMaxVae(), 22)

    assert groups == [
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17),  # clip 0: pattern 1,4,4,4,4
        (17, 18), (18, 22),                          # clip 1 restarts the pattern
    ]
    # A "frame 0 then blocks of 4" reading (what downscale_index_formula suggests)
    # would have produced 4-frame groups straight through the clip boundary.
    assert groups[5] != (17, 21)


def test_minimax_group_count_matches_its_own_frame_formula(nodes_module) -> None:
    vae = _MiniMaxVae()
    for source_frames in (5, 17, 22, 39, 56):
        groups = nodes_module._vae_temporal_groups(vae, source_frames)
        assert len(groups) == int(type(vae).downscale_ratio[0](source_frames))
        # Every source frame is covered exactly once, in order.
        assert groups[0][0] == 0
        assert groups[-1][1] == source_frames
        assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))


def test_minimax_dropped_tokens_keep_their_mask_coverage(nodes_module) -> None:
    # 17 frames = 1 clip = 5 tokens, but 3 are dropped, so only 2 latents exist.
    # Frames 5-16 have no latent of their own; their coverage must not vanish.
    groups = nodes_module._vae_temporal_groups(_MiniMaxVae(), 17)
    assert groups == [(0, 1), (1, 17)]

    masks = torch.zeros(17, 8, 8)
    masks[12] = 1.0  # painted only inside the dropped region

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(2, 8, 8),
        vae=_MiniMaxVae(),
        masks=masks,
    ).result[0]

    assert out[1].max() == pytest.approx(1.0)


def test_minimax_end_to_end_shape_and_grouping(nodes_module) -> None:
    masks = _ramp_masks(22)

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(7, 8, 12),
        vae=_MiniMaxVae(),
        masks=masks,
    ).result[0]

    assert tuple(out.shape) == (7, 8, 12)
    assert _groups_from_output(out, masks) == [
        [0], [4], [8], [12], [16], [17], [21],
    ]  # max of each group == its last frame, given a rising ramp


def test_non_uniform_formula_without_chunk_geometry_stays_proportional(
    nodes_module,
) -> None:
    class _OddVae:
        latent_dim = 3
        downscale_index_formula = None
        # 1 latent for frame 1, then 5 more every 17 frames.
        downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)

    groups = nodes_module._vae_temporal_groups(_OddVae(), 22)
    assert len(groups) == 7
    assert groups[0] == (0, 1)
    assert groups[-1][1] == 22
    assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))


# --- shape contract -----------------------------------------------------------


def test_output_matches_latent_shape_exactly(nodes_module) -> None:
    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(7, 8, 12),
        vae=_UniformVideoVae(4),
        masks=torch.rand(25, 64, 96),
    ).result[0]

    assert tuple(out.shape) == (7, 8, 12)


def test_frame_count_mismatch_is_always_an_error(nodes_module) -> None:
    with pytest.raises(ValueError, match="but the latent has 5"):
        nodes_module.vloMaskToLatentMask.execute(
            latent=_video_latent(5, 8, 8),
            vae=_UniformVideoVae(4),
            masks=torch.ones(9, 8, 8),
        )


def test_image_latent_bypasses_temporal_grouping(nodes_module) -> None:
    out = nodes_module.vloMaskToLatentMask.execute(
        latent={"samples": torch.zeros(4, 4, 8, 8)},
        vae=_ImageVae(),
        masks=torch.ones(4, 64, 64),
    ).result[0]

    assert tuple(out.shape) == (4, 8, 8)


def test_image_latent_still_validates_the_batch_count(nodes_module) -> None:
    with pytest.raises(ValueError, match="but the latent has 4"):
        nodes_module.vloMaskToLatentMask.execute(
            latent={"samples": torch.zeros(4, 4, 8, 8)},
            vae=_ImageVae(),
            masks=torch.ones(3, 64, 64),
        )


# --- pooling / resize ---------------------------------------------------------


def test_pooling_methods(nodes_module) -> None:
    values = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
    masks = values.view(5, 1, 1).expand(5, 8, 8).clone()

    def run(method):
        return nodes_module.vloMaskToLatentMask.execute(
            latent=_video_latent(2, 8, 8),
            vae=_UniformVideoVae(4),
            masks=masks,
            pooling_method=method,
        ).result[0][1, 0, 0]

    assert run("max") == pytest.approx(0.8)
    assert run("min") == pytest.approx(0.2)
    assert run("mean") == pytest.approx(0.5)


def test_nearest_resize_keeps_a_mask_binary(nodes_module) -> None:
    masks = torch.zeros(9, 32, 32)
    masks[:, 8:24, 8:24] = 1.0

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(3, 4, 4),
        vae=_UniformVideoVae(4),
        masks=masks,
        resize_mode="nearest-exact",
    ).result[0]

    assert set(out.unique().tolist()) <= {0.0, 1.0}


# --- integration with the stock node -----------------------------------------


def test_output_passes_through_set_latent_noise_mask_untouched(nodes_module) -> None:
    import comfy.utils

    latent = _video_latent(7, 8, 12)
    out = nodes_module.vloMaskToLatentMask.execute(
        latent=latent,
        vae=_MiniMaxVae(),
        masks=torch.rand(22, 64, 96),
    ).result[0]

    # What SetLatentNoiseMask stores, then what the sampler reshapes it into.
    noise_mask = out.reshape((-1, 1, out.shape[-2], out.shape[-1]))
    prepared = comfy.utils.reshape_mask(noise_mask, latent["samples"].shape)

    assert tuple(prepared.shape) == tuple(latent["samples"].shape)
    assert torch.allclose(prepared[0, 0], out)


def test_compositor_normalizes_stock_video_noise_mask(nodes_module) -> None:
    destination = _video_latent(3, 2, 2, channels=4)
    destination["noise_mask"] = torch.tensor([0.0, 0.25, 1.0]).view(3, 1, 1, 1).expand(
        3, 1, 2, 2
    )
    source = {"samples": torch.ones_like(destination["samples"])}

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination,
        source=source,
    ).result[0]

    expected = torch.tensor([0.0, 0.25, 1.0]).view(1, 1, 3, 1, 1).expand_as(
        output["samples"]
    )
    assert torch.allclose(output["samples"], expected)
    assert tuple(output["noise_mask"].shape) == (3, 1, 2, 2)


def test_compositor_keeps_preshaped_video_noise_mask_compatible(nodes_module) -> None:
    destination = _video_latent(3, 2, 2, channels=4)
    destination["noise_mask"] = torch.tensor([0.0, 0.25, 1.0]).view(1, 1, 3, 1, 1).expand(
        1, 1, 3, 2, 2
    )
    source = {"samples": torch.ones_like(destination["samples"])}

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination,
        source=source,
    ).result[0]

    expected = torch.tensor([0.0, 0.25, 1.0]).view(1, 1, 3, 1, 1).expand_as(
        output["samples"]
    )
    assert torch.allclose(output["samples"], expected)


# --- encoder-accurate frame counts (non-canonical lengths) --------------------


def test_minimax_noncanonical_lengths_follow_the_real_encoder(nodes_module) -> None:
    """encode_temporal is ceil(F/17) clips x 5 tokens - 3 drop, for every F.

    downscale_ratio disagrees whenever F does not fill a whole clip, so the
    latent's own frame count has to pick the right mapping.
    """
    vae = _MiniMaxVae()
    formula = type(vae).downscale_ratio[0]

    for source_frames in (18, 19, 20, 21, 35, 38):
        real_latents = math.ceil(source_frames / 17) * 5 - 3
        assert int(formula(source_frames)) != real_latents  # the trap

        groups = nodes_module._vae_temporal_groups(vae, source_frames, real_latents)
        assert len(groups) == real_latents
        assert groups[0][0] == 0
        assert groups[-1][1] == source_frames
        # Ranges advance in order and cover every source frame. They may repeat in
        # the padded tail, where several latents encode the same repeated frame.
        assert all(a[0] <= b[0] for a, b in zip(groups, groups[1:]))
        covered = {f for start, end in groups for f in range(start, end)}
        assert covered == set(range(source_frames))


def test_minimax_18_frames_accepts_the_7_frame_latent(nodes_module) -> None:
    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(7, 8, 12),
        vae=_MiniMaxVae(),
        masks=torch.rand(18, 64, 96),
    ).result[0]

    assert tuple(out.shape) == (7, 8, 12)


def test_formula_mapping_still_wins_when_it_is_the_matching_one(nodes_module) -> None:
    # 17 frames really do encode to 2 latents, and chunk geometry agrees.
    groups = nodes_module._vae_temporal_groups(_MiniMaxVae(), 17, 2)
    assert groups == [(0, 1), (1, 17)]


def test_a_latent_matching_neither_mapping_is_still_rejected(nodes_module) -> None:
    with pytest.raises(ValueError, match="but the latent has 4"):
        nodes_module.vloMaskToLatentMask.execute(
            latent=_video_latent(4, 8, 8),
            vae=_MiniMaxVae(),
            masks=torch.ones(18, 8, 8),
        )


# --- spatial crop parity ------------------------------------------------------


def test_mask_is_centre_cropped_exactly_like_the_vae_crops_pixels(nodes_module) -> None:
    vae = _MiniMaxVae()  # 16x spatial
    # 66 -> (66 // 16) * 16 = 64, offset (66 % 16) // 2 = 1.
    masks = torch.zeros(22, 66, 66)
    masks[:, 0, :] = 1.0   # row 0 is dropped by the crop
    masks[:, 65, :] = 1.0  # row 65 is dropped by the crop

    cropped = nodes_module._vae_encode_spatial_crop(vae, masks)
    assert tuple(cropped.shape) == (22, 64, 64)
    assert cropped.max() == 0.0  # both painted rows were outside the crop

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(7, 4, 4), vae=vae, masks=masks
    ).result[0]
    assert out.max() == 0.0


def test_crop_matches_comfy_vae_encode_crop_pixels(nodes_module) -> None:
    """Same geometry as comfy/sd.py vae_encode_crop_pixels, on the same input."""
    ratio = 16
    for size in (64, 66, 70, 79, 80):
        masks = torch.arange(size * size, dtype=torch.float32).view(1, size, size)

        pixels = masks.unsqueeze(-1)  # [B, H, W, C], as VAE.encode receives
        for dim in (1, 2):
            d = pixels.shape[dim]
            kept = (d // ratio) * ratio
            if kept != d:
                pixels = pixels.narrow(dim, (d % ratio) // 2, kept)

        cropped = nodes_module._vae_encode_spatial_crop(_MiniMaxVae(), masks)
        assert torch.equal(cropped, pixels.squeeze(-1))


def test_crop_is_skipped_when_the_vae_does_not_crop(nodes_module) -> None:
    class _NoCropVae(_UniformVideoVae):
        crop_input = False

    masks = torch.zeros(9, 66, 66)
    assert tuple(nodes_module._vae_encode_spatial_crop(_NoCropVae(4), masks).shape) == (
        9,
        66,
        66,
    )


# --- value range --------------------------------------------------------------


def test_bicubic_overshoot_is_clamped_out(nodes_module) -> None:
    masks = torch.zeros(9, 17, 17)
    masks[:, 4:13, 4:13] = 1.0

    # Bare bicubic overshoots on this input, so the node must not pass it through.
    raw = torch.nn.functional.interpolate(
        masks.unsqueeze(1), size=(5, 5), mode="bicubic", align_corners=False
    )
    assert raw.max() > 1.0

    class _NoCropVae(_UniformVideoVae):
        crop_input = False

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_video_latent(3, 5, 5),
        vae=_NoCropVae(4),
        masks=masks,
        resize_mode="bicubic",
    ).result[0]

    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_minimax_padded_tail_repeats_the_last_real_frame(nodes_module) -> None:
    """18 frames = 2 clips; clip 1 holds frame 17 plus 16 repeats of it.

    Both of clip 1's kept tokens therefore encode frame-17 content, so the two
    trailing latents map to the same source frame rather than to a gap.
    """
    groups = nodes_module._vae_temporal_groups(_MiniMaxVae(), 18, 7)
    assert groups == [
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17),
        (17, 18), (17, 18),
    ]


# --- joint AV latents ---------------------------------------------------------


def _av_latent(frames: int, height: int, width: int, audio_len: int = 37):
    import comfy.nested_tensor

    video = torch.zeros(1, 24, frames, height, width)
    audio = torch.zeros(1, 32, 2, audio_len)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def test_joint_av_latent_is_sized_against_its_video_stream(nodes_module) -> None:
    out = nodes_module.vloMaskToLatentMask.execute(
        latent=_av_latent(7, 8, 12),
        vae=_MiniMaxVae(),
        masks=torch.rand(22, 64, 96),
    ).result[0]

    assert tuple(out.shape) == (7, 8, 12)


def test_av_latent_output_leaves_the_audio_stream_unmasked(nodes_module) -> None:
    """A plain video mask on a nested latent: the sampler fills audio with ones."""
    import comfy.utils

    latent = _av_latent(7, 8, 12)
    video, audio = latent["samples"].unbind()

    out = nodes_module.vloMaskToLatentMask.execute(
        latent=latent, vae=_MiniMaxVae(), masks=torch.rand(22, 64, 96)
    ).result[0]

    # What SetLatentNoiseMask stores, then what CFGGuider.inner_sample does with it.
    noise_mask = out.reshape((-1, 1, out.shape[-2], out.shape[-1]))
    assert not noise_mask.is_nested
    denoise_masks = [noise_mask]
    for shape in [audio.shape][len(denoise_masks) - 1 :]:
        denoise_masks.append(torch.ones(shape))

    prepared = [
        comfy.utils.reshape_mask(m, s)
        for m, s in zip(denoise_masks, [video.shape, audio.shape])
    ]
    assert tuple(prepared[0].shape) == tuple(video.shape)
    assert torch.allclose(prepared[0][0, 0], out)
    assert tuple(prepared[1].shape) == tuple(audio.shape)
    assert prepared[1].min() == 1.0  # audio fully denoised, i.e. generated normally


def test_av_latent_still_validates_the_video_frame_count(nodes_module) -> None:
    with pytest.raises(ValueError, match="but the latent has 5"):
        nodes_module.vloMaskToLatentMask.execute(
            latent=_av_latent(5, 8, 12),
            vae=_MiniMaxVae(),
            masks=torch.ones(22, 64, 96),
        )


def test_audio_only_nested_latent_is_rejected_clearly(nodes_module) -> None:
    import comfy.nested_tensor

    latent = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 32, 2, 37),))}
    with pytest.raises(ValueError, match="no video stream"):
        nodes_module.vloMaskToLatentMask.execute(
            latent=latent, vae=_MiniMaxVae(), masks=torch.ones(22, 64, 96)
        )


# --- compositing joint AV latents ---------------------------------------------


def _av_composite_pair(frames: int = 3, audio_len: int = 8):
    """A destination of zeros and a source of ones, as nested (video, audio)."""
    import comfy.nested_tensor

    destination = _av_latent(frames, 2, 2, audio_len=audio_len)
    video, audio = destination["samples"].unbind()
    source = {
        "samples": comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.ones_like(audio))
        )
    }
    return destination, source, video, audio


def _audio_timeline_mask(audio: torch.Tensor, values: list[float]) -> torch.Tensor:
    """An audio noise mask shaped like its stream, varying only along time."""
    timeline = torch.tensor(values).view(1, 1, 1, -1)
    return timeline.expand_as(audio).clone()


def test_compositor_masks_both_streams_of_a_joint_av_latent(nodes_module) -> None:
    import comfy.nested_tensor

    destination, source, video, audio = _av_composite_pair(audio_len=4)
    video_mask = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1, 1).expand(3, 1, 2, 2)
    audio_mask = _audio_timeline_mask(audio, [1.0, 1.0, 0.0, 0.0])
    destination["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, audio_mask)
    )

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination, source=source
    ).result[0]

    assert getattr(output["samples"], "is_nested", False)
    out_video, out_audio = output["samples"].unbind()
    assert torch.allclose(
        out_video,
        torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 3, 1, 1).expand_as(out_video),
    )
    assert torch.allclose(
        out_audio,
        torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 1, 4).expand_as(out_audio),
    )


def test_compositor_gives_an_unmasked_stream_wholly_to_the_source(nodes_module) -> None:
    """A plain video mask covers stream 0; the sampler denoised audio in full."""
    destination, source, _, _ = _av_composite_pair()
    destination["noise_mask"] = (
        torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1).expand(3, 1, 2, 2)
    )

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination, source=source
    ).result[0]

    out_video, out_audio = output["samples"].unbind()
    assert torch.allclose(
        out_video,
        torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1).expand_as(out_video),
    )
    assert torch.allclose(out_audio, torch.ones_like(out_audio))


def test_compositor_drops_mask_entries_past_the_last_stream(nodes_module) -> None:
    """The sampler truncates a nested mask to the stream count, so this does too."""
    import comfy.nested_tensor

    destination, source, _, audio = _av_composite_pair(audio_len=4)
    destination["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (
            torch.zeros(3, 1, 2, 2),
            _audio_timeline_mask(audio, [0.0, 0.0, 0.0, 0.0]),
            torch.ones(3, 1, 2, 2),
        )
    )

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination, source=source
    ).result[0]

    out_video, out_audio = output["samples"].unbind()
    assert torch.allclose(out_video, torch.zeros_like(out_video))
    assert torch.allclose(out_audio, torch.zeros_like(out_audio))


def test_compositor_binarizes_and_clears_a_nested_mask(nodes_module) -> None:
    import comfy.nested_tensor

    destination, source, _, audio = _av_composite_pair(audio_len=4)
    destination["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (
            torch.tensor([0.4, 0.6, 0.5]).view(3, 1, 1, 1).expand(3, 1, 2, 2),
            _audio_timeline_mask(audio, [0.49, 0.51, 0.0, 1.0]),
        )
    )

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination,
        source=source,
        clear_mask=True,
        force_binary_mask=True,
    ).result[0]

    out_video, out_audio = output["samples"].unbind()
    assert torch.allclose(
        out_video,
        torch.tensor([0.0, 1.0, 1.0]).view(1, 1, 3, 1, 1).expand_as(out_video),
    )
    assert torch.allclose(
        out_audio,
        torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 1, 4).expand_as(out_audio),
    )
    assert "noise_mask" not in output


def test_compositor_returns_an_unmasked_joint_latent_untouched(nodes_module) -> None:
    destination, source, video, _ = _av_composite_pair()
    destination["samples"].unbind()[0].fill_(0.25)

    output = nodes_module.vloLatentCompositeMasked.execute(
        destination=destination, source=source
    ).result[0]

    out_video, out_audio = output["samples"].unbind()
    assert torch.allclose(out_video, torch.full_like(out_video, 0.25))
    assert torch.allclose(out_audio, torch.zeros_like(out_audio))
    assert out_video is not video  # a copy, not the caller's tensor


def test_compositor_rejects_a_source_with_different_streams(nodes_module) -> None:
    destination, _, video, _ = _av_composite_pair()
    destination["noise_mask"] = torch.ones(3, 1, 2, 2)
    source = {"samples": torch.ones_like(video)}

    with pytest.raises(ValueError, match="2 latent stream"):
        nodes_module.vloLatentCompositeMasked.execute(
            destination=destination, source=source
        )
