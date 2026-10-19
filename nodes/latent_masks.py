"""Latent-space masking: VAE temporal-group math, compositing and mask conversion."""

from __future__ import annotations

import math
from typing import Any

import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io

from .mask_utils import (
    _default_stream_noise_mask,
    _nested_existing_masks,
    _normalize_mask_frames,
)
from .temporal_feather import (
    _FEATHER_CURVES,
    _FEATHER_MODES,
    _feather_weight,
    _flat_dilate_mask,
    _shift_along,
)


_MASK_POOLING_METHODS = {
    "max": lambda frames: frames.amax(dim=0),
    "mean": lambda frames: frames.mean(dim=0),
    "min": lambda frames: frames.amin(dim=0),
}
_MASK_SPATIAL_RESIZE_MODES = ("bilinear", "nearest-exact", "area", "bicubic", "max")


def _video_stream_index(streams: list[Any]) -> int:
    """Index of the video stream among a latent's streams.

    Joint latents (EmptyMiniMaxH3LatentAV, LTXVConcatAVLatent) carry a
    NestedTensor of (video, audio) rather than a plain tensor. A video mask
    describes the video stream, and the sampler fills any stream the mask does
    not cover with ones, so the audio stream is simply generated as usual.
    """
    for index, stream in enumerate(streams):
        if isinstance(stream, torch.Tensor) and stream.ndim == 5:
            return index
    raise ValueError(
        "This joint latent has no video stream to mask. Masks describe video, "
        "so use an audio mask node for the audio stream."
    )


def _latent_video_stream(samples: Any) -> torch.Tensor:
    """The stream a video mask applies to, unwrapping joint AV latents."""
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        return streams[_video_stream_index(streams)]
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

    if mode == "max":
        # Every pixel lands in some window, so a cell holding any of the mask
        # takes its peak. Interpolating modes sample only near the cell centre
        # and can keep a cell the mask partly covers.
        resized = torch.nn.functional.adaptive_max_pool2d(
            masks.unsqueeze(1), (height, width)
        )
        return resized.squeeze(1)

    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1),
        size=(height, width),
        mode=mode,
        **kwargs,
    )
    return resized.squeeze(1)


def _snap_masks_to_grid(masks: torch.Tensor, grid: int) -> torch.Tensor:
    """Raise every grid x grid block of latent cells to its peak.

    Blocks start at the top-left cell, as DiT patches do (patchify pads the
    bottom and right), so grid=2 makes each MiniMax H3 token wholly kept or
    wholly generated. A partial block at the bottom or right edge is snapped
    over the cells it has.
    """
    if grid <= 1:
        return masks
    height, width = masks.shape[-2:]
    blocks = torch.nn.functional.max_pool2d(
        masks.unsqueeze(1), kernel_size=grid, stride=grid, ceil_mode=True
    )
    snapped = blocks.repeat_interleave(grid, dim=-2).repeat_interleave(grid, dim=-1)
    return snapped[..., :height, :width].squeeze(1)


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
                        "'nearest-exact' keeps hard edges, 'area' averages coverage, "
                        "'max' marks a cell wherever it holds any of the mask. "
                        "'bilinear' samples near each cell's centre, so at 16x it can "
                        "keep a cell the mask covers up to half of."
                    ),
                ),
                io.Int.Input(
                    "snap_grid",
                    default=1,
                    min=1,
                    max=64,
                    advanced=True,
                    tooltip=(
                        "Snap the mask outward to blocks of this many latent cells "
                        "per side, aligned to the top-left. 1 leaves it on the "
                        "latent grid. 2 matches a DiT that patchifies 2x2 latents "
                        "(MiniMax H3: 32 px), so no token is part kept and part "
                        "generated. Pixel size is this times the VAE's spatial "
                        "compression."
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
        snap_grid=1,
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

        # 5. Snap outward to the model's block grid, so a block the mask reaches
        #    is generated whole rather than split between kept and generated.
        frames = _snap_masks_to_grid(frames, int(snap_grid))
        return io.NodeOutput(frames.contiguous())


def _inside_ramp_frames(mode: str, ramp: int) -> int:
    """How many of a ramp's frames fall inside the masked run.

    The split matches the audio feather's erosion, so a mode means the same
    thing on both streams.
    """
    if mode == "outer":
        return 0
    if mode == "centered":
        return ramp // 2
    if mode == "inner":
        return ramp
    raise ValueError(f"Unsupported feather mode: {mode}")


def _run_depths(covered: torch.Tensor, *, cap: int, reverse: bool) -> torch.Tensor:
    """Per pixel, how far each frame sits into its covered run.

    Counted from the run's start, or from its end when `reverse` is set, with
    the edge frame at depth 1 and empty frames at 0. A run that reaches the clip
    edge starts at `cap` there, because the clip boundary is not a seam with
    preserved footage. Depths saturate at `cap`, past which the weight is 1.
    """
    depths = torch.zeros(covered.shape, dtype=torch.long, device=covered.device)
    depth = torch.full(covered.shape[1:], cap, dtype=torch.long, device=covered.device)
    frame_order = range(covered.shape[0] - 1, -1, -1) if reverse else range(covered.shape[0])
    for index in frame_order:
        depth = torch.where(covered[index], (depth + 1).clamp(max=cap), 0)
        depths[index] = depth
    return depths


def _inside_ramp_weights(
    depths: torch.Tensor,
    *,
    inside: int,
    ramp: int,
    curve: str,
) -> torch.Tensor:
    """Ramp weight for each covered frame from its depth into the run.

    Depth d weighs what the audio feather gives the same frame after eroding
    the run by `inside` and ramping back out, so the two agree whenever that
    erosion leaves a core. Every weight is positive, so no covered frame is lost.
    """
    table = [1.0]
    table += [
        _feather_weight(curve, (inside - depth + 1) / (ramp + 1))
        for depth in range(1, inside + 1)
    ]
    table.append(1.0)
    lookup = torch.tensor(table, dtype=torch.float32, device=depths.device)
    return lookup[depths.clamp(max=inside + 1)]


def _feather_video_mask(
    frames: torch.Tensor,
    *,
    mode: str,
    curve: str,
    lead_ramp: int,
    tail_ramp: int,
    lead_hold: int,
    tail_hold: int,
    floor: float,
) -> torch.Tensor:
    """Feather a time-first mask [T, ...] along time, independently per pixel.

    Working per pixel lets each region of the mask fade on its own schedule:
    one region can stop while another carries on, and only the one that stops
    gets a ramp.
    """
    if curve not in _FEATHER_CURVES:
        # Checked up front so a bad curve fails even when every ramp is zero.
        raise ValueError(f"Unsupported feather curve: {curve}")

    length = int(frames.shape[0])
    lead_inside = _inside_ramp_frames(mode, lead_ramp)
    tail_inside = _inside_ramp_frames(mode, tail_ramp)

    # 1. Holds widen every run with solid mask before any ramp begins.
    held = _flat_dilate_mask(
        frames,
        lead_steps=min(lead_hold, length),
        tail_steps=min(tail_hold, length),
        time_axis=0,
    )

    # 2. Inside a run, weigh each frame by its distance from the run's edge. The
    #    audio feather erodes the run instead and fails when a run disappears;
    #    per pixel, that would fail on nearly every moving mask, whose edge
    #    pixels are covered for only a frame or two. Weighting by depth agrees
    #    with the erosion wherever the erosion succeeds, and a run shorter than
    #    the ramp merely peaks lower.
    feathered = held
    if lead_inside or tail_inside:
        covered = held > 0.0
        weights = torch.ones_like(held)
        for inside, ramp, reverse in (
            (lead_inside, lead_ramp, False),
            (tail_inside, tail_ramp, True),
        ):
            if inside:
                depths = _run_depths(covered, cap=inside + 1, reverse=reverse)
                weights = torch.minimum(
                    weights,
                    _inside_ramp_weights(
                        depths, inside=inside, ramp=ramp, curve=curve
                    ).to(weights.dtype),
                )
        feathered = held * weights

    # 3. Outside a run, a frame borrows the nearest covered frame's mask at the
    #    rest of the ramp's weight, so the spatial shape carries into the fade.
    #    Taking the max lets neighbouring runs merge instead of cancelling. The
    #    requested ramp stays the curve's denominator even when the clip is
    #    shorter, so a ramp running off the end keeps its rate.
    for sign, inside, ramp in ((1, lead_inside, lead_ramp), (-1, tail_inside, tail_ramp)):
        for distance in range(1, min(ramp - inside, length - 1) + 1):
            weight = _feather_weight(curve, (inside + distance) / (ramp + 1))
            if weight > 0.0:
                feathered = torch.maximum(
                    feathered, _shift_along(held, sign * distance, 0) * weight
                )

    if floor > 0.0:
        feathered = feathered.clamp(min=floor)
    return feathered.clamp(0.0, 1.0)


def _video_mask_frames(
    mask: torch.Tensor,
    latent_frames: int,
) -> tuple[torch.Tensor, Any]:
    """A time-first view of a video noise mask, plus how to restore its layout.

    comfy.utils.reshape_mask reads a 5-D mask as [B, C, T, H, W] and folds every
    leading axis of anything smaller into time. SetLatentNoiseMask stores
    [T, 1, H, W], so both layouts reach the feather here.
    """
    if mask.ndim == 5:
        frames = mask.movedim(2, 0)
        restore = lambda feathered: feathered.movedim(0, 2)  # noqa: E731
    elif mask.ndim >= 2:
        frames = mask.reshape(-1, *mask.shape[-2:])
        restore = lambda feathered: feathered.reshape(mask.shape)  # noqa: E731
    else:
        raise ValueError(f"Unsupported video noise mask shape {tuple(mask.shape)}.")

    # A single frame is broadcast across the whole clip, so it has no time edges
    # but still takes the floor.
    if frames.shape[0] not in (1, latent_frames):
        raise ValueError(
            f"The video noise mask has {frames.shape[0]} frame(s) but the latent "
            f"has {latent_frames}. Ramps are counted in latent frames, so the mask "
            "must already match the latent: build it with vlo Mask to Latent Mask."
        )
    return frames, restore


def _restore_video_under_ramp(
    video_samples: torch.Tensor,
    feathered: torch.Tensor,
    original_latent,
) -> torch.Tensor | None:
    """Put the pre-composite video back wherever the mask is partial.

    A blank latent composite clears the samples under the binary mask, which a
    fully generated token never reads. A partial token does read its latent,
    weighted by 1 - mask, so an inner or centered ramp inside the cleared region
    would blend toward a blank latent instead of the footage.
    """
    if original_latent is None:
        return None

    full_mask = comfy.utils.reshape_mask(
        feathered.to(device=video_samples.device, dtype=torch.float32),
        video_samples.shape,
    )
    partial = (full_mask > 0.0) & (full_mask < 1.0)
    if not bool(partial.any()):
        return None

    original_streams, _ = _latent_streams(original_latent["samples"])
    original = original_streams[_video_stream_index(original_streams)]
    if tuple(original.shape) != tuple(video_samples.shape):
        raise ValueError(
            f"original_latent's video stream is shaped {tuple(original.shape)}, "
            f"but this latent's is {tuple(video_samples.shape)}. It must be the "
            "same latent, from before the composite."
        )

    restored = video_samples.clone()
    restored[partial] = original.to(
        device=video_samples.device, dtype=video_samples.dtype
    )[partial]
    return restored


class vloFeatherVideoLatentMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloFeatherVideoLatentMask",
            search_aliases=[
                "feather video mask",
                "temporal feather mask",
                "video mask crossfade",
                "soften video latent mask",
            ],
            display_name="vlo Feather Video Latent Mask",
            category="latent/mask",
            description=(
                "Softens the time edges of a video latent noise mask so inpainted "
                "footage eases in and out of the surrounding footage instead of "
                "cutting in. A fractional mask value is a genuine per-token denoise "
                "strength: MiniMax H3 and LTX run those tokens at sigma = mask * "
                "sigma, so the model generates the transition. Each pixel is "
                "feathered on its own, and ramp frames outside the mask borrow the "
                "nearest masked frame's shape. Run this after any blank latent "
                "composite; inner and centered ramps also need original_latent."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Video latent or joint AV latent that already carries a video "
                        "noise mask at latent resolution. A nested audio mask is "
                        "preserved."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(_FEATHER_MODES),
                    default="outer",
                    tooltip=(
                        "Where the ramp sits relative to the masked frames. 'outer' "
                        "keeps them solid and decays into the preserved footage, "
                        "'centered' straddles the edge, and 'inner' keeps the ramp "
                        "inside so no preserved frame is touched. A run shorter than "
                        "an inner ramp is never erased; it just peaks lower."
                    ),
                ),
                io.Int.Input(
                    "lead_ramp",
                    default=2,
                    min=0,
                    max=64,
                    tooltip=(
                        "Latent frames of ramp before each masked run. One latent "
                        "frame is about 4 video frames for MiniMax H3 and 8 for LTX."
                    ),
                ),
                io.Int.Input(
                    "tail_ramp",
                    default=2,
                    min=0,
                    max=64,
                    tooltip="Latent frames of ramp after each masked run.",
                ),
                io.Int.Input(
                    "lead_hold",
                    default=0,
                    min=0,
                    max=64,
                    tooltip=(
                        "Latent frames of fully solid mask added before each run, "
                        "ahead of the ramp."
                    ),
                ),
                io.Int.Input(
                    "tail_hold",
                    default=0,
                    min=0,
                    max=64,
                    tooltip=(
                        "Latent frames of fully solid mask added after each run, "
                        "ahead of the ramp."
                    ),
                ),
                io.Combo.Input(
                    "curve",
                    options=list(_FEATHER_CURVES),
                    default="cosine",
                    tooltip=(
                        "Ramp shape. 'cosine' and 'smoothstep' flatten at both ends, so "
                        "there is no kink where the ramp meets the solid run; 'linear' "
                        "and 'exponential' leave one."
                    ),
                ),
                io.Float.Input(
                    "floor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Minimum mask value everywhere, applied after the ramp. Above "
                        "zero this lightly regenerates all of the preserved footage. "
                        "Zero leaves it untouched."
                    ),
                ),
                io.Latent.Input(
                    "original_latent",
                    optional=True,
                    tooltip=(
                        "The same latent from before any blank latent composite. Its "
                        "video is restored wherever the mask is partial, so ramp "
                        "frames blend toward the footage instead of a blank latent. "
                        "Required for 'inner' and 'centered' after a composite; "
                        "'outer' ramps fall outside the cleared region."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(
        cls,
        latent,
        mode="outer",
        lead_ramp=2,
        tail_ramp=2,
        lead_hold=0,
        tail_hold=0,
        curve="cosine",
        floor=0.0,
        original_latent=None,
    ) -> io.NodeOutput:
        streams, nested = _latent_streams(latent["samples"])
        if not nested and getattr(streams[0], "ndim", None) != 5:
            raise ValueError(
                "Temporal feathering needs a video latent shaped [B, C, T, H, W]; "
                f"got {tuple(streams[0].shape)}, which has no time axis."
            )
        video_index = _video_stream_index(streams)
        video_samples = streams[video_index]

        existing_masks = _nested_existing_masks(latent.get("noise_mask"), len(streams))
        mask = existing_masks[video_index]
        if not isinstance(mask, torch.Tensor):
            raise ValueError(
                "The latent has no video noise mask to feather. Set one first "
                "(for example with vlo Mask to Latent Mask and SetLatentNoiseMask)."
            )

        frames, restore_layout = _video_mask_frames(mask, int(video_samples.shape[2]))
        feathered = _feather_video_mask(
            frames.to(torch.float32),
            mode=mode,
            curve=curve,
            lead_ramp=int(lead_ramp),
            tail_ramp=int(tail_ramp),
            lead_hold=int(lead_hold),
            tail_hold=int(tail_hold),
            floor=float(floor),
        )
        output_dtype = mask.dtype if mask.is_floating_point() else torch.float32
        feathered = restore_layout(feathered).to(dtype=output_dtype).contiguous()

        restored = _restore_video_under_ramp(video_samples, feathered, original_latent)

        output = latent.copy()
        if not nested:
            output["noise_mask"] = feathered
            if restored is not None:
                output["samples"] = restored
            return io.NodeOutput(output)

        existing_masks[video_index] = feathered
        for index, stream in enumerate(streams):
            if existing_masks[index] is None:
                existing_masks[index] = _default_stream_noise_mask(stream)
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(existing_masks)
        if restored is not None:
            rebuilt = list(streams)
            rebuilt[video_index] = restored
            output["samples"] = comfy.nested_tensor.NestedTensor(rebuilt)
        return io.NodeOutput(output)
