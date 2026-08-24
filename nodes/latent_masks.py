"""Latent-space masking: VAE temporal-group math, compositing and mask conversion."""

from __future__ import annotations

import math
from typing import Any

import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io

from .mask_utils import _normalize_mask_frames


_MASK_POOLING_METHODS = {
    "max": lambda frames: frames.amax(dim=0),
    "mean": lambda frames: frames.mean(dim=0),
    "min": lambda frames: frames.amin(dim=0),
}
_MASK_SPATIAL_RESIZE_MODES = ("bilinear", "nearest-exact", "area", "bicubic")


def _latent_video_stream(samples: Any) -> torch.Tensor:
    """The stream a video mask applies to, unwrapping joint AV latents.

    Joint latents (EmptyMiniMaxH3LatentAV, LTXVConcatAVLatent) carry a
    NestedTensor of (video, audio) rather than a plain tensor. A video mask
    describes the video stream, and the sampler fills any stream the mask does
    not cover with ones, so the audio stream is simply generated as usual.
    """
    if getattr(samples, "is_nested", False):
        for stream in samples.unbind():
            if isinstance(stream, torch.Tensor) and stream.ndim == 5:
                return stream
        raise ValueError(
            "This joint latent has no video stream to mask. Masks describe video, "
            "so use an audio mask node for the audio stream."
        )
    return samples


def _latent_mask_target_shape(samples: torch.Tensor) -> tuple[int, int, int, bool]:
    """Return the (frames, height, width, is_temporal) a noise mask must match."""
    if samples.ndim == 5:
        # [B, C, T, H, W]: T is real latent time, so the VAE mapping applies.
        return int(samples.shape[2]), int(samples.shape[3]), int(samples.shape[4]), True
    if samples.ndim == 4:
        # [B, C, H, W]: a plain batch of image latents, with no temporal axis
        # and therefore no VAE temporal conversion to perform.
        return int(samples.shape[0]), int(samples.shape[2]), int(samples.shape[3]), False

    raise ValueError(
        f"Unsupported latent shape {tuple(samples.shape)}. "
        "Expected [B, C, H, W] or [B, C, T, H, W]."
    )


def _vae_frame_count_formula(vae: Any):
    """The VAE's own pixel-frame-count -> latent-frame-count callable, if it has one."""
    downscale_ratio = getattr(vae, "downscale_ratio", None)
    if isinstance(downscale_ratio, (tuple, list)) and downscale_ratio:
        temporal = downscale_ratio[0]
        if callable(temporal):
            return temporal
    return None


def _formula_temporal_groups(formula, source_frames: int) -> list[tuple[int, int]]:
    """Source-frame ranges per latent frame, read off the VAE's frame-count formula.

    Wherever the formula completes new latent frames, the source frames since the
    previous completion are split evenly among them. Trailing frames that complete
    no further latent frame merge into the last group so their coverage is kept.
    This keeps the VAE's whole mapping instead of collapsing it to one factor, so
    it stays correct for non-uniform formulas.
    """
    groups: list[tuple[int, int]] = []
    previous_frame = 0
    previous_latents = 0
    for frame_count in range(1, source_frames + 1):
        latents = int(formula(frame_count))
        if latents <= previous_latents:
            continue
        new_latents = latents - previous_latents
        span = frame_count - previous_frame
        for index in range(new_latents):
            start = previous_frame + round(index * span / new_latents)
            end = previous_frame + round((index + 1) * span / new_latents)
            if end > start:
                groups.append((start, end))
        previous_frame = frame_count
        previous_latents = latents

    if not groups:
        return [(0, source_frames)]
    if previous_frame < source_frames:
        groups[-1] = (groups[-1][0], source_frames)
    return groups


def _chunked_temporal_groups(vae: Any, source_frames: int) -> list[tuple[int, int]] | None:
    """Exact source-frame ranges for VAEs that encode in fixed temporal chunks.

    Chunked encoders (MiniMax H3) split the video into clips, pre-pad each clip so
    it divides evenly, then trim a fixed number of trailing latents from the whole
    sequence. That restarts the grouping pattern at every clip boundary, which an
    averaged frames-per-latent view cannot express, so the clip geometry is read
    off the model directly.
    """
    model = getattr(vae, "first_stage_model", None)
    if model is None:
        return None

    try:
        frames_per_token = int(model.vae_ratio_t)
        clip_length = int(model.clip_length)
        pre_padding = int(model.frame_pre_padding)
        tokens_per_clip = int(model.tokens_chunk_size)
        dropped_tokens = int(getattr(model, "token_drop", 0))
    except (AttributeError, TypeError, ValueError):
        return None

    if min(frames_per_token, clip_length, tokens_per_clip) < 1:
        return None
    if pre_padding < 0 or dropped_tokens < 0:
        return None

    clip_count = math.ceil(source_frames / clip_length)
    kept_tokens = tokens_per_clip * clip_count - dropped_tokens
    if kept_tokens < 1:
        return None

    groups: list[tuple[int, int]] = []
    for token_index in range(kept_tokens):
        clip_offset = (token_index // tokens_per_clip) * clip_length
        token_in_clip = token_index % tokens_per_clip
        start = clip_offset + max(0, token_in_clip * frames_per_token - pre_padding)
        end = clip_offset + min(
            clip_length, (token_in_clip + 1) * frames_per_token - pre_padding
        )
        # Clamp into the real footage: the encoder pads short clips by repeating
        # the final frame, so tokens over the padding cover that frame.
        start = min(start, source_frames - 1)
        end = min(max(end, start + 1), source_frames)
        groups.append((start, end))

    # Trailing tokens can be trimmed away entirely (MiniMax H3 drops 3), leaving
    # real frames with no latent of their own. Merge their coverage into the last
    # kept group so painted regions are never silently discarded.
    if groups[-1][1] < source_frames:
        groups[-1] = (groups[-1][0], source_frames)
    return groups


def _vae_temporal_groups(
    vae: Any,
    source_frames: int,
    target_frames: int | None = None,
) -> list[tuple[int, int]]:
    """Source-frame ranges feeding each latent frame, derived from the VAE.

    Chunk geometry is the accurate description where a VAE exposes it, but a
    VAE's frame-count formula can disagree with its own encoder at frame counts
    that do not fill a whole chunk (MiniMax H3 reports 2 latents for 18 frames
    while its encoder actually produces 7). Neither source is authoritative on
    its own, so when the latent tells us how many latent frames really exist,
    that count picks between them.
    """
    if source_frames < 1:
        raise ValueError("masks must contain at least one frame.")

    candidates: list[list[tuple[int, int]]] = []
    chunked = _chunked_temporal_groups(vae, source_frames)
    if chunked is not None:
        candidates.append(chunked)
    formula = _vae_frame_count_formula(vae)
    if formula is not None:
        candidates.append(_formula_temporal_groups(formula, source_frames))

    if not candidates:
        # No temporal compression advertised: one latent frame per source frame.
        return [(index, index + 1) for index in range(source_frames)]

    if target_frames is not None:
        for groups in candidates:
            if len(groups) == target_frames:
                return groups

    return candidates[0]


def _vae_encode_spatial_crop(vae: Any, masks: torch.Tensor) -> torch.Tensor:
    """Apply the same centre crop the VAE applies to pixels before encoding.

    ComfyUI's ``vae_encode_crop_pixels`` trims height and width down to a
    multiple of the spatial compression ratio, centred. Skipping it would scale
    the mask from a larger area than the VAE ever saw, shifting the alignment.
    """
    if not getattr(vae, "crop_input", False):
        return masks

    ratio = None
    compression = getattr(vae, "spacial_compression_encode", None)
    if callable(compression):
        try:
            ratio = compression()
        except Exception:  # noqa: BLE001 - a VAE without a usable ratio just skips
            ratio = None
    if ratio is None:
        downscale_ratio = getattr(vae, "downscale_ratio", None)
        if isinstance(downscale_ratio, (tuple, list)) and downscale_ratio:
            ratio = downscale_ratio[-1]
        else:
            ratio = downscale_ratio
    if not isinstance(ratio, int) or ratio < 1:
        return masks

    for dim in (1, 2):
        size = int(masks.shape[dim])
        cropped = (size // ratio) * ratio
        if cropped != size and cropped >= 1:
            masks = masks.narrow(dim, (size % ratio) // 2, cropped)
    return masks


def _pool_masks_over_groups(
    masks: torch.Tensor,
    groups: list[tuple[int, int]],
    *,
    pooling_method: str,
) -> torch.Tensor:
    try:
        pool = _MASK_POOLING_METHODS[pooling_method]
    except KeyError:
        raise ValueError(f"Unsupported pooling method: {pooling_method}") from None

    return torch.stack([pool(masks[start:end]) for start, end in groups], dim=0)


def _resize_masks_spatially(
    masks: torch.Tensor,
    height: int,
    width: int,
    *,
    mode: str,
) -> torch.Tensor:
    if mode not in _MASK_SPATIAL_RESIZE_MODES:
        raise ValueError(f"Unsupported spatial resize mode: {mode}")
    if masks.shape[-2:] == (height, width):
        return masks

    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1),
        size=(height, width),
        mode=mode,
        **kwargs,
    )
    return resized.squeeze(1)


def _latent_streams(samples: Any) -> tuple[list[torch.Tensor], bool]:
    """Every stream of a latent, plus whether it arrived as a joint AV latent."""
    if getattr(samples, "is_nested", False):
        return list(samples.unbind()), True
    return [samples], False


def _clone_latent_samples(samples: Any) -> Any:
    """A detached copy of a latent, nested or not (NestedTensor has no clone)."""
    streams, nested = _latent_streams(samples)
    cloned = [stream.clone() for stream in streams]
    return comfy.nested_tensor.NestedTensor(cloned) if nested else cloned[0]


def _stream_noise_masks(noise_mask: Any, stream_count: int) -> list[Any]:
    """One mask per stream, following the sampler's own nested-mask rules.

    comfy.samplers unbinds a nested mask, drops any entry past the last stream
    and fills the streams the mask does not reach with ones, so those streams
    are denoised in full. A plain mask therefore describes the first stream
    only. Compositing under the same rules reproduces what the sampler did.
    """
    if getattr(noise_mask, "is_nested", False):
        masks = list(noise_mask.unbind())[:stream_count]
    else:
        masks = [noise_mask]
    masks += [None] * (stream_count - len(masks))
    return [mask if isinstance(mask, torch.Tensor) else None for mask in masks]


def _composite_stream(
    dest_samples: torch.Tensor,
    src_samples: torch.Tensor,
    mask: Any,
    *,
    force_binary_mask: bool,
) -> torch.Tensor:
    if mask is None:
        # Unmasked streams are denoised in full, so the source owns them whole.
        return src_samples.clone()

    mask = mask.to(dtype=dest_samples.dtype, device=dest_samples.device)
    mask = comfy.utils.reshape_mask(mask, dest_samples.shape)

    if force_binary_mask:
        mask = (mask >= 0.5).to(dtype=mask.dtype)

    try:
        return src_samples * mask + dest_samples * (1.0 - mask)
    except RuntimeError as e:
        raise ValueError(
            f"Could not composite: destination {tuple(dest_samples.shape)}, "
            f"source {tuple(src_samples.shape)}, mask {tuple(mask.shape)} "
            f"are not broadcast-compatible. Ensure the mask is preshaped for this latent."
        ) from e


class vloLatentCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloLatentCompositeMasked",
            search_aliases=["vlo composite latent", "vlo inpaint latent"],
            display_name="vlo Latent Composite Masked",
            category="latent/composite",
            description=(
                "Composites a source latent into a destination latent using "
                "the destination's existing noise_mask. The mask dictates where the "
                "source replaces the destination. Joint video+audio latents are "
                "composited stream by stream, exactly as the sampler masked them: a "
                "nested mask supplies one mask per stream, a plain mask covers the "
                "first stream only, and any stream the mask does not reach is taken "
                "whole from the source because the sampler denoised it in full."
            ),
            inputs=[
                io.Latent.Input(
                    "destination",
                    tooltip="The destination latent. Should have an existing noise_mask.",
                ),
                io.Latent.Input(
                    "source",
                    tooltip="The source latent patches to composite into the destination.",
                ),
                io.Boolean.Input(
                    "clear_mask",
                    default=False,
                    tooltip="If true, removes the noise_mask from the output latent after compositing.",
                ),
                io.Boolean.Input(
                    "force_binary_mask",
                    default=False,
                    tooltip="If true, applies a 0.5 threshold to the mask to prevent continuous blending at the edges.",
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, destination, source, clear_mask=False, force_binary_mask=False) -> io.NodeOutput:
        output = destination.copy()

        mask = destination.get("noise_mask")
        if mask is None:
            output["samples"] = _clone_latent_samples(destination["samples"])
            return io.NodeOutput(output)

        dest_streams, dest_nested = _latent_streams(destination["samples"])
        src_streams, _ = _latent_streams(source["samples"])
        if len(src_streams) != len(dest_streams):
            raise ValueError(
                f"Could not composite: the destination has {len(dest_streams)} "
                f"latent stream(s) and the source has {len(src_streams)}. Joint "
                "video+audio latents must be composited against a source with the "
                "same streams."
            )

        masks = _stream_noise_masks(mask, len(dest_streams))
        composited = [
            _composite_stream(
                dest_stream,
                src_stream,
                stream_mask,
                force_binary_mask=force_binary_mask,
            )
            for dest_stream, src_stream, stream_mask in zip(
                dest_streams, src_streams, masks
            )
        ]
        output["samples"] = (
            comfy.nested_tensor.NestedTensor(composited)
            if dest_nested
            else composited[0]
        )

        if clear_mask:
            output.pop("noise_mask", None)

        return io.NodeOutput(output)


class vloMaskToLatentMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMaskToLatentMask",
            search_aliases=[
                "vlo preprocess masks",
                "vlo latent noise mask",
                "vlo mask resize latent",
            ],
            display_name="vlo Mask to Latent Mask",
            category="latent/mask",
            description=(
                "Converts a pixel-space mask sequence into a mask shaped exactly like the "
                "supplied latent. The latent supplies the destination frames/height/width "
                "and the VAE supplies the temporal correspondence, so the result can be "
                "attached with the stock SetLatentNoiseMask node. Joint video+audio "
                "latents are sized against their video stream; the audio stream is left "
                "unmasked and generates as usual."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Latent the mask must match. Only its dimensions are read; the "
                        "latent itself is neither modified nor returned. Joint AV latents "
                        "are accepted and sized against their video stream."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "VAE used to encode the video. Supplies the mapping from source "
                        "frames to latent frames."
                    ),
                ),
                io.Mask.Input(
                    "masks",
                    tooltip="Pixel-space mask sequence, one mask per source video frame.",
                ),
                io.Combo.Input(
                    "pooling_method",
                    options=list(_MASK_POOLING_METHODS),
                    default="max",
                    tooltip=(
                        "How the source frames feeding one latent frame are combined. "
                        "'max' keeps anything masked in any frame, 'mean' averages, "
                        "'min' keeps only what is masked in every frame."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(_MASK_SPATIAL_RESIZE_MODES),
                    default="bilinear",
                    advanced=True,
                    tooltip=(
                        "How the mask is resized to the latent's height and width. "
                        "'nearest-exact' keeps hard edges, 'area' averages coverage."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="latent_mask",
                    tooltip=(
                        "Mask shaped [latent_frames, latent_height, latent_width], ready "
                        "for SetLatentNoiseMask."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        vae,
        masks,
        pooling_method="max",
        resize_mode="bilinear",
    ) -> io.NodeOutput:
        samples = _latent_video_stream(latent.get("samples"))
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent['samples'] must be a tensor.")

        frame_count, height, width, is_temporal = _latent_mask_target_shape(samples)
        frames = _normalize_mask_frames(masks)
        source_count = int(frames.shape[0])
        if source_count < 1:
            raise ValueError("masks must contain at least one frame.")

        if is_temporal:
            # 1. Ask the VAE which source frames feed each latent frame, then pool
            #    them. The latent's frame count disambiguates when the VAE's chunk
            #    geometry and its frame formula describe different mappings.
            groups = _vae_temporal_groups(vae, source_count, frame_count)
            frames = _pool_masks_over_groups(
                frames, groups, pooling_method=pooling_method
            )

        # 2. The latent is the validator: a count that does not line up means the
        #    mask belongs to a different video, so resampling it here would produce
        #    a correctly shaped mask with the wrong meaning.
        if frames.shape[0] != frame_count:
            raise ValueError(
                f"{source_count} source mask frame(s) map to {frames.shape[0]} latent "
                f"frame(s), but the latent has {frame_count}. Check that the mask covers "
                "the same video as the latent."
            )

        # 3. Crop exactly as the VAE crops pixels before encoding, then resize to
        #    the latent's exact spatial size.
        frames = _vae_encode_spatial_crop(vae, frames)
        frames = _resize_masks_spatially(frames, height, width, mode=resize_mode)

        # 4. Keep the mask a mask: bicubic overshoots, and the sampler blends with
        #    noise_mask unclamped, so out-of-range values extrapolate instead.
        frames = frames.clamp(0.0, 1.0)
        return io.NodeOutput(frames.contiguous())
