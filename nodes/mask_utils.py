"""Mask helpers shared by the audio-latent and video-latent mask nodes."""

from __future__ import annotations

from typing import Any

import torch


def _normalize_mask_frames(masks: torch.Tensor) -> torch.Tensor:
    mask_tensor = masks.float()

    if mask_tensor.ndim == 2:
        return mask_tensor.unsqueeze(0)
    if mask_tensor.ndim == 3:
        return mask_tensor
    if mask_tensor.ndim == 4:
        # Collapse any unexpected channel axis into a single per-frame mask.
        if mask_tensor.shape[1] == 1:
            return mask_tensor[:, 0]
        return mask_tensor.mean(dim=1)

    raise ValueError(
        f"Unsupported mask shape {tuple(mask_tensor.shape)}. "
        "Expected [H, W], [F, H, W], or [F, C, H, W]."
    )


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
