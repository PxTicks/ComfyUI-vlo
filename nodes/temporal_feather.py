"""Temporal feather primitives shared by the audio and video latent mask nodes.

Everything here works along one time axis of a plain tensor, so the audio node
(time on its layout's axis) and the video node (latent frames) share a single
definition of the ramp and its curves.
"""

from __future__ import annotations

import math

import torch


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
    feathered there: the clip boundary is not a seam with preserved content.
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
