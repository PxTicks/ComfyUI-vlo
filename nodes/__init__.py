"""The vlo ComfyUI node pack.

Each concern lives in its own module; this package is the front door that
registers them with ComfyUI. `routes` is imported for its side effects — the
/api/vlo-memory endpoints register via PromptServer decorators at import time.
"""

from __future__ import annotations

from typing_extensions import override

# `comfy`, `io`, `Input`, `InputImpl` and `Types` are re-exported as module
# attributes on purpose: the integration tests build ComfyUI values off this
# module rather than importing ComfyUI themselves.
import comfy.nested_tensor  # noqa: F401
import comfy.utils  # noqa: F401
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io  # noqa: F401

from . import routes  # noqa: F401  (registers the /api/vlo-memory endpoints)
from .audio_masks import LTXSetAudioLatentBinaryMasks, vloSetAudioLatentBinaryMasks
from .latent_masks import (
    _vae_encode_spatial_crop,
    _vae_temporal_groups,
    vloLatentCompositeMasked,
    vloMaskToLatentMask,
)
from .loaders import (
    vloMemoryLoadAudio,
    vloMemoryLoadAudioBatch,
    vloMemoryLoadImage,
    vloMemoryLoadImageBatch,
    vloMemoryLoadVideo,
    vloMemoryLoadVideoBatch,
)
from .logic_nodes import vloGateNone, vloLogicNot
from .minimax import (
    _get_native_minimax_h3_reference_contract,
    vloMiniMaxH3ReferenceToVideoBatch,
)
from .registry import REGISTRY
from .ttm import vloTimeToMove
from .video_nodes import (
    vloSaveImageWebsocketBMP,
    vloSaveVideoWebsocket,
    vloVideoConvertFps,
)


class vloExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            vloMemoryLoadImage,
            vloMemoryLoadAudio,
            vloMemoryLoadVideo,
            vloMemoryLoadImageBatch,
            vloMemoryLoadAudioBatch,
            vloMemoryLoadVideoBatch,
            vloMiniMaxH3ReferenceToVideoBatch,
            vloVideoConvertFps,
            vloSaveImageWebsocketBMP,
            vloSaveVideoWebsocket,
            LTXSetAudioLatentBinaryMasks,
            vloSetAudioLatentBinaryMasks,
            vloLatentCompositeMasked,
            vloMaskToLatentMask,
            vloGateNone,
            vloLogicNot,
            vloTimeToMove,
        ]


async def comfy_entrypoint() -> vloExtension:
    return vloExtension()
