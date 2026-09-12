"""Audio-latent binary noise masks: the LTX-specific path and the generic one."""

from __future__ import annotations

import math
import re
from typing import Any

import torch

import comfy.nested_tensor
from comfy_api.latest import io

from .mask_utils import _normalize_mask_frames


def _coerce_existing_audio_mask(
    mask: torch.Tensor | None,
    target_shape: tuple[int, int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None

    coerced = mask.to(device=device, dtype=dtype)
    if coerced.ndim == 3:
        coerced = coerced.unsqueeze(0)

    if coerced.ndim != 4 or coerced.shape[2:] != target_shape[2:]:
        return None

    batch_size, channels, _, _ = target_shape
    if coerced.shape[0] not in (1, batch_size):
        return None
    if coerced.shape[1] not in (1, channels):
        return None

    return coerced.expand(batch_size, channels, target_shape[2], target_shape[3]).clone()


def _build_audio_binary_noise_mask(
    audio_samples: torch.Tensor,
    masks: torch.Tensor,
    *,
    threshold: float,
    resize_mode: str,
) -> torch.Tensor:
    batch_size, channels, frame_count, feature_count = audio_samples.shape
    frame_masks = _normalize_mask_frames(masks).to(
        device=audio_samples.device,
        dtype=torch.float32,
    )

    # 1. Collapse each input frame to a single active/inactive value so only the
    #    temporal extent of the mask affects the audio latent.
    frame_activity = (frame_masks >= threshold).amax(dim=(-2, -1)).to(torch.float32)

    # 2. Resample only along time. Spatial dimensions are intentionally ignored.
    timeline = frame_activity.view(1, 1, -1)
    if resize_mode == "nearest":
        resized_timeline = torch.nn.functional.interpolate(
            timeline,
            size=frame_count,
            mode="nearest",
        )
    elif resize_mode == "linear":
        resized_timeline = torch.nn.functional.interpolate(
            timeline,
            size=frame_count,
            mode="linear",
            align_corners=False,
        )
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    # 3. Broadcast the binary timeline across the non-time audio latent axes.
    binary_timeline = (resized_timeline >= 0.5).to(dtype=audio_samples.dtype)
    return binary_timeline.view(1, 1, frame_count, 1).expand(
        batch_size, channels, frame_count, feature_count
    ).clone()


_AUDIO_LAYOUT_TIME_AXES = {
    "ltx": 2,
    "bctf": 2,
    "time_frequency": 2,
    "minimax": 3,
    "bcst": 3,
    "stereo_time": 3,
}
_AUDIO_METADATA_KEYS = ("audio_latent_metadata", "audio_metadata")


def _audio_metadata_sources(audio_latent, audio_vae) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key in _AUDIO_METADATA_KEYS:
        metadata = audio_latent.get(key)
        if isinstance(metadata, dict):
            sources.append(metadata)
    sources.append(audio_latent)

    first_stage_model = getattr(audio_vae, "first_stage_model", None)
    for owner in (audio_vae, first_stage_model):
        if owner is None:
            continue
        for key in _AUDIO_METADATA_KEYS:
            metadata = getattr(owner, key, None)
            if isinstance(metadata, dict):
                sources.append(metadata)
    return sources


def _metadata_value(sources: list[dict[str, Any]], *keys: str) -> Any | None:
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _time_axis_from_layout(layout: Any) -> int | None:
    if isinstance(layout, dict):
        explicit_axis = layout.get("time_axis")
        if isinstance(explicit_axis, int) and not isinstance(explicit_axis, bool):
            return explicit_axis
        layout = layout.get("axes", layout.get("layout"))

    if isinstance(layout, (list, tuple)):
        axes = [str(axis).strip().lower() for axis in layout]
        for name in ("time", "t"):
            if name in axes:
                return axes.index(name)
        return None

    if not isinstance(layout, str):
        return None

    normalized = layout.strip().lower()
    if normalized in _AUDIO_LAYOUT_TIME_AXES:
        return _AUDIO_LAYOUT_TIME_AXES[normalized]
    compact = re.sub(r"[^a-z]", "", normalized)
    if compact in _AUDIO_LAYOUT_TIME_AXES:
        return _AUDIO_LAYOUT_TIME_AXES[compact]
    if len(compact) >= 3 and compact.count("t") == 1:
        return compact.index("t")
    return None


def _known_audio_architecture(audio_vae) -> str | None:
    first_stage_model = getattr(audio_vae, "first_stage_model", None)
    if first_stage_model is None:
        return None

    type_name = (
        f"{type(first_stage_model).__module__}."
        f"{type(first_stage_model).__qualname__}"
    ).lower()
    if "minimax" in type_name or "minimaxh3audiovae" in type_name:
        return "minimax"
    if "lightricks" in type_name or hasattr(
        first_stage_model, "latent_frequency_bins"
    ):
        return "ltx"
    return None


def _normalize_audio_time_axis(time_axis: int, ndim: int) -> int:
    normalized = time_axis + ndim if time_axis < 0 else time_axis
    if normalized < 2 or normalized >= ndim:
        raise ValueError(
            f"Audio time axis {time_axis} is invalid for a {ndim}D latent; "
            "batch and channel axes cannot be used as time."
        )
    return normalized


def _resolve_audio_time_axis(
    audio_latent,
    audio_vae,
    audio_samples: torch.Tensor,
    layout_override: str,
) -> tuple[int, str]:
    if layout_override != "auto":
        return (
            _normalize_audio_time_axis(
                _AUDIO_LAYOUT_TIME_AXES[layout_override], audio_samples.ndim
            ),
            layout_override,
        )

    sources = _audio_metadata_sources(audio_latent, audio_vae)
    explicit_axis = _metadata_value(sources, "audio_time_axis", "time_axis")
    if isinstance(explicit_axis, int) and not isinstance(explicit_axis, bool):
        return _normalize_audio_time_axis(explicit_axis, audio_samples.ndim), "metadata"

    layout = _metadata_value(sources, "audio_latent_layout", "audio_layout", "layout")
    metadata_axis = _time_axis_from_layout(layout)
    if metadata_axis is not None:
        return _normalize_audio_time_axis(metadata_axis, audio_samples.ndim), "metadata"

    first_stage_model = getattr(audio_vae, "first_stage_model", None)
    for owner in (audio_vae, first_stage_model):
        if owner is None:
            continue
        direct_axis = getattr(owner, "audio_time_axis", None)
        if isinstance(direct_axis, int) and not isinstance(direct_axis, bool):
            return _normalize_audio_time_axis(direct_axis, audio_samples.ndim), "vae"
        vae_axis = _time_axis_from_layout(
            getattr(owner, "audio_latent_layout", None)
        )
        if vae_axis is not None:
            return _normalize_audio_time_axis(vae_axis, audio_samples.ndim), "vae"

    architecture = _known_audio_architecture(audio_vae)
    if architecture is not None:
        return (
            _normalize_audio_time_axis(
                _AUDIO_LAYOUT_TIME_AXES[architecture], audio_samples.ndim
            ),
            architecture,
        )

    raise ValueError(
        "Could not determine the audio latent time axis. Connect its audio VAE, "
        "attach audio_latent_metadata, or select an explicit latent layout."
    )


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        return None
    return converted


def _resolve_audio_latent_rate(
    audio_latent,
    audio_vae,
    *,
    rate_override: float,
    architecture: str,
) -> float | None:
    override = _positive_float(rate_override)
    if override is not None:
        return override

    sources = _audio_metadata_sources(audio_latent, audio_vae)
    metadata_rate = _positive_float(
        _metadata_value(
            sources,
            "audio_latent_rate",
            "audio_latents_per_second",
            "latents_per_second",
        )
    )
    if metadata_rate is not None:
        return metadata_rate

    first_stage_model = getattr(audio_vae, "first_stage_model", None)
    for owner in (first_stage_model, audio_vae):
        if owner is None:
            continue
        direct_rate = _positive_float(getattr(owner, "latents_per_second", None))
        if direct_rate is not None:
            return direct_rate

    if first_stage_model is not None:
        sample_rate = _positive_float(getattr(first_stage_model, "sample_rate", None))
        samples_per_latent = _positive_float(
            getattr(first_stage_model, "samples_per_latent", None)
        )
        if sample_rate is not None and samples_per_latent is not None:
            return sample_rate / samples_per_latent

    if architecture == "ltx":
        return 25.0
    if architecture == "minimax":
        return 40.0
    return None


def _resolve_audio_stream(
    audio_latent,
    samples,
    audio_vae,
) -> tuple[torch.Tensor, int | None, list[torch.Tensor] | None]:
    if isinstance(samples, torch.Tensor):
        return samples, None, None
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "audio_latent['samples'] must be a tensor or a nested AV latent."
        )

    streams = list(samples.unbind())
    stream_index = _metadata_value(
        _audio_metadata_sources(audio_latent, audio_vae), "audio_stream_index"
    )
    if isinstance(stream_index, int) and not isinstance(stream_index, bool):
        normalized = stream_index + len(streams) if stream_index < 0 else stream_index
        if 0 <= normalized < len(streams):
            return streams[normalized], normalized, streams
        raise ValueError(
            f"Audio stream index {stream_index} is invalid for {len(streams)} streams."
        )

    candidates = [index for index, stream in enumerate(streams) if stream.ndim == 4]
    if len(candidates) == 1:
        return streams[candidates[0]], candidates[0], streams
    if len(streams) > 1 and streams[1].ndim >= 3:
        return streams[1], 1, streams
    raise ValueError(
        "Could not identify the audio stream in the nested latent; attach "
        "audio_stream_index metadata."
    )


def _resample_audio_activity(
    frame_activity: torch.Tensor,
    *,
    target_length: int,
    resize_mode: str,
    mask_fps: float,
    audio_latent_rate: float | None,
) -> torch.Tensor:
    if target_length <= 0:
        raise ValueError("Audio latent time dimension must contain at least one frame.")
    if frame_activity.shape[0] == 0:
        raise ValueError("At least one input mask frame is required.")

    source_rate = _positive_float(mask_fps)
    if source_rate is None:
        timeline = frame_activity.view(1, 1, -1)
        if resize_mode == "nearest":
            return torch.nn.functional.interpolate(
                timeline, size=target_length, mode="nearest"
            ).view(-1)
        if resize_mode == "linear":
            return torch.nn.functional.interpolate(
                timeline, size=target_length, mode="linear", align_corners=False
            ).view(-1)
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    if audio_latent_rate is None:
        raise ValueError(
            "mask_fps requires an audio latent rate. Connect the audio VAE, attach "
            "rate metadata, or set an explicit audio latent rate."
        )

    source_positions = (
        torch.arange(target_length, device=frame_activity.device, dtype=torch.float64)
        * source_rate
        / audio_latent_rate
    )
    if resize_mode == "nearest":
        indices = source_positions.round().to(torch.long)
        indices.clamp_(0, frame_activity.shape[0] - 1)
        return frame_activity.index_select(0, indices)
    if resize_mode == "linear":
        lower = source_positions.floor().to(torch.long)
        upper = lower + 1
        lower.clamp_(0, frame_activity.shape[0] - 1)
        upper.clamp_(0, frame_activity.shape[0] - 1)
        fraction = (source_positions - source_positions.floor()).to(frame_activity.dtype)
        return frame_activity[lower] * (1.0 - fraction) + frame_activity[upper] * fraction
    raise ValueError(f"Unsupported resize mode: {resize_mode}")


def _build_generic_audio_binary_noise_mask(
    audio_samples: torch.Tensor,
    masks: torch.Tensor,
    *,
    time_axis: int,
    threshold: float,
    resize_mode: str,
    mask_fps: float,
    audio_latent_rate: float | None,
) -> torch.Tensor:
    frame_masks = _normalize_mask_frames(masks).to(
        device=audio_samples.device, dtype=torch.float32
    )
    frame_activity = (frame_masks >= threshold).amax(dim=(-2, -1)).to(torch.float32)
    resized = _resample_audio_activity(
        frame_activity,
        target_length=int(audio_samples.shape[time_axis]),
        resize_mode=resize_mode,
        mask_fps=mask_fps,
        audio_latent_rate=audio_latent_rate,
    )
    binary_timeline = (resized >= 0.5).to(dtype=audio_samples.dtype)
    view_shape = [1] * audio_samples.ndim
    view_shape[time_axis] = binary_timeline.shape[0]
    return binary_timeline.view(view_shape).expand(tuple(audio_samples.shape)).clone()


def _expand_existing_mask(mask: Any, audio_samples: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(mask, torch.Tensor):
        return None
    expanded = mask.to(device=audio_samples.device, dtype=audio_samples.dtype)
    if expanded.ndim == audio_samples.ndim - 1:
        expanded = expanded.unsqueeze(1)
    if expanded.ndim != audio_samples.ndim:
        return None
    if any(
        current not in (1, target)
        for current, target in zip(expanded.shape, audio_samples.shape)
    ):
        return None
    return expanded.expand(tuple(audio_samples.shape)).clone()


def _combine_audio_masks(
    existing_mask: Any,
    generated_mask: torch.Tensor,
    audio_samples: torch.Tensor,
    existing_mask_mode: str,
) -> torch.Tensor:
    if existing_mask_mode == "overwrite":
        return generated_mask

    existing = _expand_existing_mask(existing_mask, audio_samples)
    if existing is None:
        existing = torch.zeros_like(audio_samples)
    if existing_mask_mode == "add":
        return torch.maximum(existing, generated_mask)
    if existing_mask_mode == "subtract":
        existing[generated_mask > 0] = 0.0
        return existing
    raise ValueError(f"Unsupported existing_mask_mode: {existing_mask_mode}")


def _nested_existing_masks(noise_mask: Any, stream_count: int) -> list[Any]:
    if getattr(noise_mask, "is_nested", False):
        masks = list(noise_mask.unbind())
        return masks[:stream_count] + [None] * max(0, stream_count - len(masks))
    if isinstance(noise_mask, torch.Tensor):
        return [noise_mask] + [None] * max(0, stream_count - 1)
    return [None] * stream_count


def _default_stream_noise_mask(stream: torch.Tensor) -> torch.Tensor:
    shape = (stream.shape[0], 1, *stream.shape[2:])
    return torch.ones(shape, device=stream.device, dtype=stream.dtype)


_FEATHER_CURVES = ("cosine", "linear", "smoothstep", "exponential")
_FEATHER_MODES = ("outer", "centered", "inner")
_EXPONENTIAL_FEATHER_DECAY = 5.0


def _feather_weight(curve: str, position: float) -> float:
    """Ramp weight at `position` in (0, 1), falling from the solid core to zero."""
    if position <= 0.0:
        return 1.0
    if position >= 1.0:
        return 0.0
    if curve == "linear":
        return 1.0 - position
    if curve == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * position))
    if curve == "smoothstep":
        return 1.0 - position * position * (3.0 - 2.0 * position)
    if curve == "exponential":
        tail = math.exp(-_EXPONENTIAL_FEATHER_DECAY)
        return (math.exp(-_EXPONENTIAL_FEATHER_DECAY * position) - tail) / (1.0 - tail)
    raise ValueError(f"Unsupported feather curve: {curve}")


def _feather_kernel(
    curve: str,
    *,
    lead_hold: int,
    lead_ramp: int,
    tail_hold: int,
    tail_ramp: int,
    max_offset: int,
) -> dict[int, float]:
    """Offset -> weight for one masked step. Negative offsets lead, positive trail.

    A ramp of N steps yields exactly N non-zero weights, so the step beyond the
    ramp is genuinely zero rather than the last rung of the ramp.

    `max_offset` bounds which offsets are worth evaluating on a short clip; it
    deliberately does not shorten the ramp itself, because the requested length
    is the curve's denominator. Clamping it there would steepen the decay instead
    of truncating it, and a ramp running off the end of the clip would come out
    at a completely different rate from the same ramp with room to finish.
    """
    if curve not in _FEATHER_CURVES:
        # Checked here rather than only where the ramp is sampled, so a bad curve
        # is caught even when every ramp rounds to zero steps.
        raise ValueError(f"Unsupported feather curve: {curve}")

    kernel = {0: 1.0}
    for sign, hold, ramp in ((-1, lead_hold, lead_ramp), (1, tail_hold, tail_ramp)):
        for distance in range(1, min(hold + ramp, max_offset) + 1):
            if distance <= hold:
                weight = 1.0
            else:
                weight = _feather_weight(curve, (distance - hold) / (ramp + 1))
            if weight > 0.0:
                kernel[sign * distance] = weight
    return kernel


def _shift_along(tensor: torch.Tensor, offset: int, time_axis: int) -> torch.Tensor:
    """tensor[..., i + offset, ...] with replicate padding at both ends.

    Replicate is what keeps a region that runs to the clip edge from being
    feathered there: the clip boundary is not a seam with preserved audio.
    """
    if offset == 0:
        return tensor
    length = tensor.shape[time_axis]
    indices = torch.arange(length, device=tensor.device) + offset
    return tensor.index_select(time_axis, indices.clamp_(0, length - 1))


def _erode_mask(
    mask: torch.Tensor,
    *,
    lead_steps: int,
    tail_steps: int,
    time_axis: int,
) -> torch.Tensor:
    """Pull each masked region's start in by `lead_steps` and its end by `tail_steps`."""
    eroded = mask
    for offset in range(1, lead_steps + 1):
        eroded = torch.minimum(eroded, _shift_along(mask, -offset, time_axis))
    for offset in range(1, tail_steps + 1):
        eroded = torch.minimum(eroded, _shift_along(mask, offset, time_axis))
    return eroded


def _flat_dilate_mask(
    mask: torch.Tensor,
    *,
    lead_steps: int,
    tail_steps: int,
    time_axis: int,
) -> torch.Tensor:
    """The dual of `_erode_mask`: push each region's start and end back out.

    Eroding then flat-dilating by the same amounts is a morphological opening,
    which restores every region wide enough to survive and leaves nothing behind
    for the ones that were not. That difference is how a destroyed region is found.
    """
    dilated = mask
    for offset in range(1, lead_steps + 1):
        dilated = torch.maximum(dilated, _shift_along(mask, offset, time_axis))
    for offset in range(1, tail_steps + 1):
        dilated = torch.maximum(dilated, _shift_along(mask, -offset, time_axis))
    return dilated


def _dilate_mask(
    mask: torch.Tensor,
    kernel: dict[int, float],
    time_axis: int,
) -> torch.Tensor:
    """Grayscale dilation by the ramp kernel: out[i] = max_j mask[i - j] * kernel[j].

    Taking the max is what makes neighbouring regions merge instead of their
    ramps cancelling, and it leaves the solid core untouched because kernel[0] is 1.
    """
    dilated = None
    for offset, weight in kernel.items():
        shifted = _shift_along(mask, -offset, time_axis)
        if weight != 1.0:
            shifted = shifted * weight
        dilated = shifted if dilated is None else torch.maximum(dilated, shifted)
    return dilated


def _feather_steps(
    seconds: float,
    latent_rate: float | None,
    *,
    label: str,
    at_least_one: bool,
) -> int:
    duration = _positive_float(seconds)
    if duration is None:
        return 0
    if latent_rate is None:
        raise ValueError(
            f"{label} requires an audio latent rate. Connect the audio VAE, attach "
            "rate metadata, or set an explicit audio latent rate."
        )
    steps = int(round(duration * latent_rate))
    if at_least_one:
        # A requested ramp shorter than one latent step still gets one, so a small
        # value softens the seam instead of silently doing nothing.
        return max(1, steps)
    return steps


def _reject_eroded_regions(
    mask: torch.Tensor,
    core: torch.Tensor,
    *,
    lead_erode: int,
    tail_erode: int,
    time_axis: int,
    mode: str,
) -> None:
    """Fail when the erosion destroyed any masked region, not just all of them.

    Checking the mask's overall peak would only catch the case where nothing at
    all survives, so a short region alongside a long one would disappear in
    silence and quietly shrink the edit.
    """
    reopened = _flat_dilate_mask(
        core, lead_steps=lead_erode, tail_steps=tail_erode, time_axis=time_axis
    )
    lost = (mask > 0.0) & (reopened <= 0.0)
    if not bool(lost.any()):
        return

    lost_steps = int(
        lost.movedim(time_axis, 0).contiguous().flatten(1).any(dim=1).sum()
    )
    raise ValueError(
        f"The '{mode}' feather of {lead_erode} + {tail_erode} latent steps is wider "
        f"than a masked region it has to fit inside, erasing {lost_steps} masked "
        "step(s) entirely. Shorten the ramp, or use 'outer', which never shrinks "
        "the masked region."
    )


def _feather_audio_mask(
    mask: torch.Tensor,
    *,
    time_axis: int,
    mode: str,
    kernel: dict[int, float],
    lead_ramp: int,
    tail_ramp: int,
    floor: float,
) -> torch.Tensor:
    if mode == "outer":
        lead_erode = tail_erode = 0
    elif mode == "centered":
        lead_erode, tail_erode = lead_ramp // 2, tail_ramp // 2
    elif mode == "inner":
        lead_erode, tail_erode = lead_ramp, tail_ramp
    else:
        raise ValueError(f"Unsupported feather mode: {mode}")

    # Shifting further than the clip only repeats the edge step, so bounding the
    # erosion loops here costs nothing and changes no result.
    length = int(mask.shape[time_axis])
    lead_erode, tail_erode = min(lead_erode, length), min(tail_erode, length)

    core = _erode_mask(
        mask, lead_steps=lead_erode, tail_steps=tail_erode, time_axis=time_axis
    )
    if lead_erode or tail_erode:
        _reject_eroded_regions(
            mask,
            core,
            lead_erode=lead_erode,
            tail_erode=tail_erode,
            time_axis=time_axis,
            mode=mode,
        )
    feathered = _dilate_mask(core, kernel, time_axis)
    if floor > 0.0:
        feathered = feathered.clamp(min=floor)
    return feathered.clamp(0.0, 1.0)


class LTXSetAudioLatentBinaryMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXSetAudioLatentBinaryMasks",
            display_name="LTX Set Audio Latent Binary Masks",
            category="latent/audio",
            description=(
                "Converts a binary mask image or mask video into an audio latent noise mask "
                "by reducing each frame to active/inactive, resizing only along time, and "
                "broadcasting that timeline across the audio latent."
            ),
            inputs=[
                io.Latent.Input(
                    "audio_latent",
                    tooltip="Audio latent whose noise_mask will be set.",
                ),
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "Binary mask image or mask video. Only the temporal activity of each "
                        "frame is used; spatial dimensions are ignored after thresholding."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Per-pixel threshold used when deciding whether a mask frame is active. "
                        "If any pixel in a frame meets this threshold, that frame activates audio masking."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["nearest", "linear"],
                    default="nearest",
                    tooltip=(
                        "How to resize the derived binary timeline to the audio latent length. "
                        "'nearest' preserves hard ranges; 'linear' smooths transitions before the final binary threshold."
                    ),
                ),
                io.Combo.Input(
                    "existing_mask_mode",
                    options=["overwrite", "add", "subtract"],
                    default="overwrite",
                    tooltip=(
                        "How to combine with an existing audio noise mask. "
                        "'overwrite' replaces it, 'add' takes the max, and 'subtract' clears masked regions."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="audio_latent")],
        )

    @classmethod
    def execute(
        cls,
        audio_latent,
        masks,
        threshold,
        resize_mode="nearest",
        existing_mask_mode="overwrite",
    ) -> io.NodeOutput:
        samples = audio_latent["samples"]
        if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
            raise ValueError(
                "audio_latent['samples'] must be a 4D tensor shaped like [B, C, F, S]."
            )

        generated_mask = _build_audio_binary_noise_mask(
            samples,
            masks,
            threshold=threshold,
            resize_mode=resize_mode,
        )

        output = audio_latent.copy()
        if existing_mask_mode == "overwrite":
            output["noise_mask"] = generated_mask
            return io.NodeOutput(output)

        existing_mask = _coerce_existing_audio_mask(
            audio_latent.get("noise_mask"),
            tuple(samples.shape),
            dtype=samples.dtype,
            device=samples.device,
        )
        if existing_mask is None:
            existing_mask = torch.zeros_like(samples)

        if existing_mask_mode == "add":
            output["noise_mask"] = torch.maximum(existing_mask, generated_mask)
        elif existing_mask_mode == "subtract":
            existing_mask[generated_mask > 0] = 0.0
            output["noise_mask"] = existing_mask
        else:
            raise ValueError(f"Unsupported existing_mask_mode: {existing_mask_mode}")

        return io.NodeOutput(output)


class vloSetAudioLatentBinaryMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloSetAudioLatentBinaryMasks",
            search_aliases=[
                "set audio latent mask",
                "audio retake mask",
                "minimax audio mask",
                "ltx audio mask",
            ],
            display_name="vlo Set Audio Latent Binary Masks",
            category="latent/audio",
            description=(
                "Sets a temporal binary noise mask on a standalone audio latent or the "
                "audio stream of a nested AV latent. Resolves layout and latent rate from "
                "latent/VAE metadata, with compatibility layouts for LTX and MiniMax H3."
            ),
            inputs=[
                io.Latent.Input(
                    "audio_latent",
                    tooltip=(
                        "Standalone audio latent or nested AV latent. Nested video masks "
                        "are preserved while only the audio mask is changed."
                    ),
                ),
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "Binary mask frames. Each frame is reduced to active/inactive; "
                        "spatial dimensions do not affect the audio mask."
                    ),
                ),
                io.Float.Input(
                    "mask_fps",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "FPS of the input mask frames. Values above zero map mask "
                        "timestamps to the VAE's audio latent rate (25 Hz for LTX, 40 Hz "
                        "for MiniMax). Zero stretches the complete mask batch to the "
                        "audio latent length."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="A mask frame is active when any pixel meets this threshold.",
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["nearest", "linear"],
                    default="nearest",
                    tooltip=(
                        "Nearest preserves hard frame ranges. Linear interpolates the "
                        "timeline before applying the final binary threshold."
                    ),
                ),
                io.Combo.Input(
                    "existing_mask_mode",
                    options=["overwrite", "add", "subtract"],
                    default="overwrite",
                    tooltip=(
                        "Overwrite replaces the audio mask, add takes the maximum, and "
                        "subtract clears active regions from the existing audio mask."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Audio VAE used to resolve layout and latent rate automatically. "
                        "It may be omitted when the latent carries metadata or overrides "
                        "are supplied."
                    ),
                ),
                io.Combo.Input(
                    "layout_override",
                    options=["auto", "ltx", "minimax"],
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "Auto prefers latent/VAE metadata, then recognizes current LTX "
                        "[B,C,T,F] and MiniMax [B,C,S,T] VAEs."
                    ),
                ),
                io.Float.Input(
                    "audio_latent_rate",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Audio latent steps per second. Zero resolves this from metadata "
                        "or the connected VAE."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="audio_latent")],
        )

    @classmethod
    def execute(
        cls,
        audio_latent,
        masks,
        mask_fps=0.0,
        threshold=0.5,
        resize_mode="nearest",
        existing_mask_mode="overwrite",
        audio_vae=None,
        layout_override="auto",
        audio_latent_rate=0.0,
    ) -> io.NodeOutput:
        samples = audio_latent["samples"]
        audio_samples, stream_index, streams = _resolve_audio_stream(
            audio_latent, samples, audio_vae
        )
        time_axis, architecture = _resolve_audio_time_axis(
            audio_latent,
            audio_vae,
            audio_samples,
            layout_override,
        )
        latent_rate = _resolve_audio_latent_rate(
            audio_latent,
            audio_vae,
            rate_override=audio_latent_rate,
            architecture=architecture,
        )
        generated_mask = _build_generic_audio_binary_noise_mask(
            audio_samples,
            masks,
            time_axis=time_axis,
            threshold=threshold,
            resize_mode=resize_mode,
            mask_fps=mask_fps,
            audio_latent_rate=latent_rate,
        )

        output = audio_latent.copy()
        existing_noise_mask = audio_latent.get("noise_mask")
        if streams is None:
            output["noise_mask"] = _combine_audio_masks(
                existing_noise_mask,
                generated_mask,
                audio_samples,
                existing_mask_mode,
            )
        else:
            existing_masks = _nested_existing_masks(existing_noise_mask, len(streams))
            existing_masks[stream_index] = _combine_audio_masks(
                existing_masks[stream_index],
                generated_mask,
                audio_samples,
                existing_mask_mode,
            )
            for index, stream in enumerate(streams):
                if existing_masks[index] is None:
                    existing_masks[index] = _default_stream_noise_mask(stream)
            output["noise_mask"] = comfy.nested_tensor.NestedTensor(existing_masks)

        existing_metadata = audio_latent.get("audio_latent_metadata")
        resolved_metadata = (
            existing_metadata.copy() if isinstance(existing_metadata, dict) else {}
        )
        resolved_metadata.update(
            {
                "time_axis": time_axis,
                "layout_source": architecture,
            }
        )
        if "layout" not in resolved_metadata:
            resolved_metadata["layout"] = (
                architecture
                if architecture in ("ltx", "minimax")
                else f"time_axis_{time_axis}"
            )
        if latent_rate is not None:
            resolved_metadata["latents_per_second"] = latent_rate
        if stream_index is not None:
            resolved_metadata["audio_stream_index"] = stream_index
        output["audio_latent_metadata"] = resolved_metadata
        return io.NodeOutput(output)


def _restore_audio_under_ramp(
    audio_samples: torch.Tensor,
    feathered: torch.Tensor,
    original_audio_latent,
    audio_vae,
) -> torch.Tensor | None:
    """Put the pre-composite audio back wherever the feather left a partial step.

    A blank latent composite clears the samples under the binary mask, which is
    harmless while the mask is binary: a fully generated step never reads its
    latent image. A ramp does read it, weighted by 1 - mask, so any ramp step
    lying inside the cleared region would blend toward silence. Outer ramps fall
    outside that region and need nothing; inner and centered ramps sit within it.
    """
    if original_audio_latent is None:
        return None

    ramp = (feathered > 0.0) & (feathered < 1.0)
    if not bool(ramp.any()):
        return None

    original_samples, _, _ = _resolve_audio_stream(
        original_audio_latent, original_audio_latent["samples"], audio_vae
    )
    if tuple(original_samples.shape) != tuple(audio_samples.shape):
        raise ValueError(
            "original_audio_latent's audio stream is shaped "
            f"{tuple(original_samples.shape)}, but this latent's is "
            f"{tuple(audio_samples.shape)}. It must be the same latent, from "
            "before the composite."
        )

    restored = audio_samples.clone()
    restored[ramp] = original_samples.to(
        device=audio_samples.device, dtype=audio_samples.dtype
    )[ramp]
    return restored


class vloFeatherAudioLatentMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloFeatherAudioLatentMask",
            search_aliases=[
                "feather audio mask",
                "audio mask crossfade",
                "audio mask tail",
                "soften audio latent mask",
            ],
            display_name="vlo Feather Audio Latent Mask",
            category="latent/audio",
            description=(
                "Softens the time edges of an audio latent noise mask so inpainted "
                "audio joins the surrounding audio without a click. A fractional mask "
                "value is a genuine per-step denoise strength, not just a crossfade: "
                "MiniMax H3 places those rows at sigma = mask * sigma_audio and "
                "conditions them accordingly, so the model generates the transition. "
                "Run this after any blank latent composite, which leaves outer ramps "
                "with the original audio to blend toward. Inner and centered ramps sit "
                "inside the region that composite cleared, so those need "
                "original_audio_latent connected as well."
            ),
            inputs=[
                io.Latent.Input(
                    "audio_latent",
                    tooltip=(
                        "Standalone audio latent or nested AV latent that already "
                        "carries an audio noise mask. Nested video masks are preserved."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(_FEATHER_MODES),
                    default="outer",
                    tooltip=(
                        "Where the ramp sits relative to the masked region. 'outer' "
                        "keeps the region solid and decays outward into the preserved "
                        "audio, 'centered' straddles the edge, and 'inner' keeps the "
                        "ramp inside so no preserved step is touched. The latter two "
                        "shrink the region, so a region narrower than the ramp loses "
                        "steps to it."
                    ),
                ),
                io.Float.Input(
                    "lead_ramp",
                    default=0.15,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Seconds of ramp before each masked region. Rounded to whole "
                        "audio latent steps (25 ms for MiniMax, 40 ms for LTX); a "
                        "non-zero value always gets at least one step."
                    ),
                ),
                io.Float.Input(
                    "tail_ramp",
                    default=0.15,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Seconds of ramp after each masked region. Worth lengthening "
                        "past the lead when a note or phoneme decays for longer than it "
                        "takes to start."
                    ),
                ),
                io.Float.Input(
                    "lead_hold",
                    default=0.1,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Seconds of fully solid mask added before each region, ahead of "
                        "the ramp. Audible onsets lead the visible motion that drew the "
                        "mask, so a hold hands those steps to the model outright."
                    ),
                ),
                io.Float.Input(
                    "tail_hold",
                    default=0.1,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Seconds of fully solid mask added after each region, ahead of "
                        "the ramp. Useful when reverb or a decay tail outlasts the "
                        "masked frames."
                    ),
                ),
                io.Combo.Input(
                    "curve",
                    options=list(_FEATHER_CURVES),
                    default="cosine",
                    tooltip=(
                        "Ramp shape. 'cosine' and 'smoothstep' flatten at both ends, so "
                        "there is no kink where the ramp meets the solid core; 'linear' "
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
                        "zero this lightly denoises the whole preserved region, which "
                        "can even out a level or timbre step but rewrites audio you "
                        "asked to keep. Zero leaves preserved audio untouched."
                    ),
                ),
                io.Latent.Input(
                    "original_audio_latent",
                    optional=True,
                    tooltip=(
                        "The same latent from before any blank latent composite. Its "
                        "audio is restored underneath the ramp, so ramp steps blend "
                        "toward the original audio instead of toward silence. Required "
                        "for 'inner' and 'centered' after a composite, since those "
                        "ramps sit inside the region it cleared; 'outer' ramps fall "
                        "outside it and need nothing."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Audio VAE used to resolve layout and latent rate automatically. "
                        "It may be omitted when the latent carries metadata or overrides "
                        "are supplied."
                    ),
                ),
                io.Combo.Input(
                    "layout_override",
                    options=["auto", "ltx", "minimax"],
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "Auto prefers latent/VAE metadata, then recognizes current LTX "
                        "[B,C,T,F] and MiniMax [B,C,S,T] VAEs."
                    ),
                ),
                io.Float.Input(
                    "audio_latent_rate",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Audio latent steps per second, used to convert the ramp and "
                        "hold seconds. Zero resolves this from metadata or the VAE."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="audio_latent")],
        )

    @classmethod
    def execute(
        cls,
        audio_latent,
        mode="outer",
        lead_ramp=0.15,
        tail_ramp=0.15,
        lead_hold=0.1,
        tail_hold=0.1,
        curve="cosine",
        floor=0.0,
        original_audio_latent=None,
        audio_vae=None,
        layout_override="auto",
        audio_latent_rate=0.0,
    ) -> io.NodeOutput:
        samples = audio_latent["samples"]
        audio_samples, stream_index, streams = _resolve_audio_stream(
            audio_latent, samples, audio_vae
        )
        time_axis, architecture = _resolve_audio_time_axis(
            audio_latent, audio_vae, audio_samples, layout_override
        )
        latent_rate = _resolve_audio_latent_rate(
            audio_latent,
            audio_vae,
            rate_override=audio_latent_rate,
            architecture=architecture,
        )

        existing_noise_mask = audio_latent.get("noise_mask")
        if streams is None:
            existing_masks = None
            audio_mask = existing_noise_mask
        else:
            existing_masks = _nested_existing_masks(existing_noise_mask, len(streams))
            audio_mask = existing_masks[stream_index]

        mask = _expand_existing_mask(audio_mask, audio_samples)
        if mask is None:
            raise ValueError(
                "The audio latent has no audio noise mask to feather. Set one first "
                "(for example with vlo Set Audio Latent Binary Masks)."
            )

        length = int(audio_samples.shape[time_axis])
        steps = {
            "lead_ramp": (lead_ramp, True),
            "tail_ramp": (tail_ramp, True),
            "lead_hold": (lead_hold, False),
            "tail_hold": (tail_hold, False),
        }
        # Kept at the requested length rather than clipped to the clip: the ramp
        # length is the curve's denominator, so clipping would change its rate.
        resolved = {
            name: _feather_steps(
                seconds, latent_rate, label=name, at_least_one=at_least_one
            )
            for name, (seconds, at_least_one) in steps.items()
        }

        kernel = _feather_kernel(
            curve,
            lead_hold=resolved["lead_hold"],
            lead_ramp=resolved["lead_ramp"],
            tail_hold=resolved["tail_hold"],
            tail_ramp=resolved["tail_ramp"],
            max_offset=max(0, length - 1),
        )
        feathered = _feather_audio_mask(
            mask.to(torch.float32),
            time_axis=time_axis,
            mode=mode,
            kernel=kernel,
            lead_ramp=resolved["lead_ramp"],
            tail_ramp=resolved["tail_ramp"],
            floor=float(floor),
        ).to(dtype=audio_samples.dtype)

        restored = _restore_audio_under_ramp(
            audio_samples, feathered, original_audio_latent, audio_vae
        )

        output = audio_latent.copy()
        if streams is None:
            output["noise_mask"] = feathered
            if restored is not None:
                output["samples"] = restored
        else:
            existing_masks[stream_index] = feathered
            for index, stream in enumerate(streams):
                if existing_masks[index] is None:
                    existing_masks[index] = _default_stream_noise_mask(stream)
            output["noise_mask"] = comfy.nested_tensor.NestedTensor(existing_masks)
            if restored is not None:
                rebuilt = list(streams)
                rebuilt[stream_index] = restored
                output["samples"] = comfy.nested_tensor.NestedTensor(rebuilt)
        return io.NodeOutput(output)
