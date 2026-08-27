"""Experimental per-guide-token masked conditioning for MiniMax H3.

`masked_h3_forward.py` holds a fork of core's `MiniMaxH3Model._forward`; keeping
it in one file is deliberate, so it can be deleted wholesale if the feature ever
lands upstream. Everything else here is ordinary node code.
"""

from __future__ import annotations

from .nodes import (
    vloMiniMaxH3AddMaskedGuide,
    vloMiniMaxH3AddMaskedGuidesFromVideo,
    vloMiniMaxH3GuideTokenMaskPreview,
    vloMiniMaxH3MaskedGuidePixelFill,
    vloMiniMaxH3PatchMaskedGuides,
)

__all__ = [
    "vloMiniMaxH3AddMaskedGuide",
    "vloMiniMaxH3AddMaskedGuidesFromVideo",
    "vloMiniMaxH3PatchMaskedGuides",
    "vloMiniMaxH3GuideTokenMaskPreview",
    "vloMiniMaxH3MaskedGuidePixelFill",
]
