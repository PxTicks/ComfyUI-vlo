"""Pixel-space mask growth for inpaint workflows.

ComfyUI's stock GrowMask dilates with a 3x3 scipy footprint once per pixel of
growth, per frame, on the CPU. That is fine for a still, but a 16 px grow over
a 720p clip costs about a quarter of a second per frame, minutes for a long
inpaint. A square grow is separable, so two 1-D max pools do the same work in
a single pass per axis, and they run on the torch device a frame chunk at a
time.
"""

from __future__ import annotations

import comfy.model_management
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .mask_utils import _normalize_mask_frames


# Bounds device memory: a 720p float32 chunk of this many frames is ~60 MB.
_GROW_CHUNK_FRAMES = 16


def _grow_masks(frames: torch.Tensor, expand: int, device: torch.device) -> torch.Tensor:
    """Grayscale-dilate `[F, H, W]` masks by `expand` px with a square kernel.

    Square, not round, because the latent cells a mask is pooled into are
    square: reaching every cell a masked pixel touches is a Chebyshev-distance
    requirement, and a round kernel would need sqrt(2) times the radius to
    cover the diagonal.
    """
    if expand <= 0 or frames.shape[0] == 0:
        return frames

    size = 2 * expand + 1
    out = torch.empty_like(frames)
    for start in range(0, frames.shape[0], _GROW_CHUNK_FRAMES):
        chunk = frames[start : start + _GROW_CHUNK_FRAMES].to(device).unsqueeze(1)
        # max_pool2d pads with -inf, so the frame border never grows the mask.
        chunk = F.max_pool2d(chunk, (1, size), stride=1, padding=(0, expand))
        chunk = F.max_pool2d(chunk, (size, 1), stride=1, padding=(expand, 0))
        out[start : start + _GROW_CHUNK_FRAMES] = chunk.squeeze(1).to(out.device)
    return out


class vloGrowMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloGrowMask",
            search_aliases=["vlo dilate mask", "vlo expand mask", "vlo grow mask"],
            display_name="vlo Grow Mask",
            category="mask",
            description=(
                "Grows a mask sequence by a fixed number of pixels with a square "
                "kernel. Inpaint workflows use it to give a tight segmentation mask a "
                "margin, so object edges, motion blur and shadows outside the mask do "
                "not survive, and so every latent cell a masked pixel touches is "
                "regenerated. Run it at the resolution the video is encoded at, since "
                "the grow is in pixels."
            ),
            inputs=[
                io.Mask.Input("mask", tooltip="Mask sequence, one mask per frame."),
                io.Int.Input(
                    "expand",
                    default=8,
                    min=0,
                    max=512,
                    step=1,
                    tooltip=(
                        "Pixels to grow by in every direction. One latent cell is a "
                        "sensible default: 8 for Wan, 16 for MiniMax H3."
                    ),
                ),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, expand=8) -> io.NodeOutput:
        frames = _normalize_mask_frames(mask)
        grown = _grow_masks(frames, int(expand), comfy.model_management.get_torch_device())
        return io.NodeOutput(grown.to(comfy.model_management.intermediate_device()))
