"""PromptServer plumbing: execution identity, progress and binary websocket events."""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
import torch
from PIL import Image
from server import PromptServer

from comfy.cli_args import args
from comfy_api.latest import Types, io
from comfy_execution.utils import get_executing_context


def _get_client_id() -> str | None:
    client_id = getattr(PromptServer.instance, "client_id", None)
    if isinstance(client_id, str) and client_id.strip():
        return client_id
    return None


def _get_execution_ids() -> tuple[str | None, str | None]:
    context = get_executing_context()
    if context is None:
        return None, None
    return context.node_id, context.prompt_id


def _send_progress_update(
    value: float,
    max_value: float,
    *,
    node_id: str | None,
    prompt_id: str | None,
) -> None:
    client_id = _get_client_id()
    if client_id is None:
        return

    progress = {"value": value, "max": max_value}
    if prompt_id is not None:
        progress["prompt_id"] = prompt_id
    if node_id is not None:
        progress["node"] = node_id

    PromptServer.instance.send_sync("progress", progress, sid=client_id)


def _send_binary_event(event: int, payload: bytes) -> None:
    client_id = _get_client_id()
    if client_id is None:
        return
    PromptServer.instance.send_sync(event, payload, sid=client_id)


def _encode_payload_with_metadata(payload: bytes, metadata: dict[str, Any]) -> bytes:
    metadata_json = json.dumps(metadata).encode("utf-8")
    return struct.pack(">I", len(metadata_json)) + metadata_json + payload


def _tensor_to_pil_rgb_image(image: torch.Tensor) -> Image.Image:
    image_rgb = (
        torch.clamp(image[..., :3] * 255.0, min=0, max=255)
        .to(device=torch.device("cpu"), dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(np.ascontiguousarray(image_rgb), mode="RGB")


def _build_saved_video_metadata(node_cls: type[io.ComfyNode]) -> dict[str, Any] | None:
    if args.disable_metadata:
        return None

    metadata: dict[str, Any] = {}
    if node_cls.hidden.extra_pnginfo is not None:
        metadata.update(node_cls.hidden.extra_pnginfo)
    if node_cls.hidden.prompt is not None:
        metadata["prompt"] = node_cls.hidden.prompt
    return metadata or None


def _resolve_video_content_type(format_value: Types.VideoContainer | str) -> str:
    if isinstance(format_value, Types.VideoContainer):
        container = format_value
    else:
        container = Types.VideoContainer(format_value)

    if container in (Types.VideoContainer.AUTO, Types.VideoContainer.MP4):
        return "video/mp4"
    return "application/octet-stream"
