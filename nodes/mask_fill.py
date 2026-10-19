"""Pixel-space fill of a mask's area before VAE encoding.

At full denoise an inpaint never shows the model what lies under the generated
mask: those latents start from pure noise. The VAE encoder still sees it,
though, and its receptive field reaches well past one latent cell, so kept
cells beside a hard edge (the black around an outpaint, a cut-out) encode that
edge and the decoder draws it back as a seam. Filling the area with a smooth
extension of the surrounding colours first means the encoder never sees one.
"""

from __future__ import annotations

import comfy.model_management
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .mask_utils import _normalize_mask_frames


# Bounds device memory: a 720p float32 RGB chunk of this many frames is ~180 MB,
# plus a third again for the pyramid.
_FILL_CHUNK_FRAMES = 16

# Mask MP4s decode their black slightly above 0 after a YUV round trip, so a
# pixel counts as masked only above this. About the frontend's
# MASK_VIDEO_WHITE_THRESHOLD (32 / 255), rounded so a float widget keeps it.
_DEFAULT_FILL_THRESHOLD = 0.125


def _pull_push_fill(pixels: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Fill the unknown pixels of `[N, C, H, W]` from the known ones.

    Pull-push: average the known pixels down a 2x pyramid, then rebuild from the
    coarsest level, keeping each level's known colour where it has some and
    taking the upsampled coarser estimate for the rest. Known pixels come back
    exactly; holes get a smooth extension of whatever surrounds them.
    """
    weight = known.to(pixels.dtype)
    colour = pixels * weight
    levels = [(colour, weight)]

    # 1. Pull: premultiplied colour and coverage, each averaged over 2x2. Odd
    #    sizes pad with zero coverage, which reads as "unknown", not black.
    while colour.shape[-2] > 1 or colour.shape[-1] > 1:
        pad = (0, colour.shape[-1] % 2, 0, colour.shape[-2] % 2)
        colour = F.avg_pool2d(F.pad(colour, pad), 2)
        weight = F.avg_pool2d(F.pad(weight, pad), 2)
        levels.append((colour, weight))

    # 2. Push: a level's colour is coverage x mean known colour, so adding the
    #    uncovered share of the coarser estimate completes it.
    colour, weight = levels[-1]
    estimate = colour / weight.clamp(min=1e-12)
    for colour, weight in reversed(levels[:-1]):
        height, width = colour.shape[-2:]
        upsampled = F.interpolate(
            estimate, scale_factor=2, mode="bilinear", align_corners=False
        )[..., :height, :width]
        estimate = colour + (1.0 - weight) * upsampled
    return estimate


def _fill_masked_pixels(
    images: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Fill `[F, H, W, C]` images wherever `[F or 1, H, W]` masks exceed `threshold`."""
    if masks.shape[0] not in (1, images.shape[0]):
        raise ValueError(
            f"The mask has {masks.shape[0]} frame(s) but the image has "
            f"{images.shape[0]}. Use one mask per frame, or a single mask for all."
        )
    if tuple(masks.shape[-2:]) != tuple(images.shape[1:3]):
        raise ValueError(
            f"The mask is {tuple(masks.shape[-2:])} but the image is "
            f"{tuple(images.shape[1:3])}. Resize the mask to the image first."
        )

    out = images.clone()
    for start in range(0, images.shape[0], _FILL_CHUNK_FRAMES):
        stop = min(start + _FILL_CHUNK_FRAMES, images.shape[0])
        chunk_masks = masks if masks.shape[0] == 1 else masks[start:stop]
        holes = (chunk_masks > threshold).to(device)
        holes = holes.expand(stop - start, *holes.shape[1:]).unsqueeze(1)

        # Frames with nothing to fill, or nothing to fill from, stay as they are.
        hole_counts = holes.flatten(1).sum(dim=1)
        fillable = (hole_counts > 0) & (hole_counts < holes[0].numel())
        if not bool(fillable.any()):
            continue

        # A copy, since .to() returns the caller's tensor when it is already on device.
        pixels = images[start:stop].to(device, copy=True).permute(0, 3, 1, 2)
        filled = _pull_push_fill(pixels[fillable], ~holes[fillable])
        pixels[fillable] = filled
        out[start:stop] = pixels.permute(0, 2, 3, 1).to(out.device)
    return out


class vloFillMaskedArea(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloFillMaskedArea",
            search_aliases=["vlo prefill mask", "vlo inpaint fill", "vlo pull push fill"],
            display_name="vlo Fill Masked Area",
            category="mask",
            description=(
                "Replaces the masked pixels of an image sequence with a smooth "
                "extension of the colours around them. Put it before VAE Encode in "
                "a full-denoise inpaint or outpaint: the model never sees what is "
                "under the generated mask, but the encoder does, and a hard edge "
                "there (black around an outpaint) leaks into the kept latents "
                "beside it as a seam. Do not use it below full denoise, where the "
                "masked content is the starting point."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Frames to fill, at encode resolution."),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Area to fill (1 = fill), one mask per frame or one for all, "
                        "at the image's resolution."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=_DEFAULT_FILL_THRESHOLD,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    advanced=True,
                    tooltip=(
                        "A pixel is filled when its mask value is above this. The "
                        "default sits above the slightly-raised black of a decoded "
                        "mask video, and below a soft edge's partly covered pixels, "
                        "which are usually darkened by what was behind them."
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, mask, threshold=_DEFAULT_FILL_THRESHOLD) -> io.NodeOutput:
        filled = _fill_masked_pixels(
            image.float(),
            _normalize_mask_frames(mask),
            float(threshold),
            comfy.model_management.get_torch_device(),
        )
        return io.NodeOutput(filled.to(comfy.model_management.intermediate_device()))
