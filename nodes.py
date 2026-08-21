from __future__ import annotations

import hashlib
import io as stdlib_io
import json
import logging
import math
import os
import struct
from fractions import Fraction
from typing import Any

import av
import folder_paths
import node_helpers
import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence
from protocol import BinaryEventTypes
from server import PromptServer
from typing_extensions import override

import comfy.model_management
from comfy.cli_args import args
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.utils import get_executing_context

from .batch_loader_utils import (
    normalize_memory_batch_flags,
    normalize_memory_batch_values,
)
from .media_registry import (
    MediaItem,
    MediaRegistry,
    MediaRegistryCapacityError,
    MediaTooLargeError,
)


REGISTRY = MediaRegistry()
logger = logging.getLogger(__name__)
_DEFAULT_CONTENT_TYPES = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/wav",
}
_UI_PLACEHOLDER_MEDIA_IDS = frozenset({"loading..."})
_COMFY_MAX_RESOLUTION = 16384
_INPUT_KIND_CONTENT_TYPES: dict[str, list[str]] = {
    "image": ["image"],
    "audio": ["audio", "video"],
    "video": ["video"],
}


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _normalize_kind(raw_kind: Any) -> str | None:
    if not isinstance(raw_kind, str):
        return None
    kind = raw_kind.strip().lower()
    if kind in _DEFAULT_CONTENT_TYPES:
        return kind
    return None


def _media_summary(item: MediaItem) -> dict[str, Any]:
    return {
        "media_id": item.media_id,
        "kind": item.kind,
        "filename": item.filename,
        "content_type": item.content_type,
        "size_bytes": item.size_bytes,
        "created_at": item.created_at,
        "last_accessed_at": item.last_accessed_at,
        "accessed_once": item.accessed_once,
    }


def _get_media_item(media_id: str, *, expected_kind: str | None = None) -> MediaItem:
    normalized_media_id = _normalize_media_id(media_id)
    if normalized_media_id is None:
        expected_label = expected_kind or "media"
        raise ValueError(f"No {expected_label} selected")

    item = REGISTRY.get(normalized_media_id)
    if item is None:
        raise ValueError(f"Unknown media id: {normalized_media_id}")
    if expected_kind is not None and item.kind != expected_kind:
        raise ValueError(
            f"Media id '{normalized_media_id}' has kind '{item.kind}', expected '{expected_kind}'"
        )
    return item


def _normalize_media_id(raw_media_id: Any) -> str | None:
    if not isinstance(raw_media_id, str):
        return None
    media_id = raw_media_id.strip()
    if not media_id:
        return None
    if media_id.lower() in _UI_PLACEHOLDER_MEDIA_IDS:
        return None
    return media_id


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


def _build_memory_output_item(
    item: MediaItem,
    *,
    subfolder: str = "",
) -> dict[str, Any]:
    return {
        "filename": item.filename,
        "subfolder": subfolder,
        "type": "output",
        "content_type": item.content_type,
        "view_url": f"/api/vlo-memory/view/{item.media_id}",
    }


def _list_input_files(content_types: list[str]) -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = [
        filename
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename))
    ]
    return sorted(folder_paths.filter_files_content_types(files, content_types))


def _annotated_filepath_exists(raw_value: Any) -> bool:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return False
    try:
        return folder_paths.exists_annotated_filepath(raw_value)
    except Exception:
        return False


def _should_load_from_filepath(raw_value: Any, *, disable_in_memory: bool) -> bool:
    if disable_in_memory:
        return True

    normalized_media_id = _normalize_media_id(raw_value)
    if normalized_media_id is not None and REGISTRY.get(normalized_media_id, mark_accessed=False) is not None:
        return False

    return _annotated_filepath_exists(raw_value)


def _validate_memory_batch_values(
    raw_values: Any,
    *,
    label: str,
    expected_kind: str,
    disable_in_memory: bool,
) -> bool | str:
    try:
        values = normalize_memory_batch_values(raw_values, label=label)
    except ValueError as exc:
        return str(exc)

    for value in values:
        if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
            if not folder_paths.exists_annotated_filepath(value):
                return f"Invalid {expected_kind} file: {value}"
            continue

        item = REGISTRY.get(value, mark_accessed=False)
        if item is None:
            return f"Invalid {expected_kind} id: {value}"
        if item.kind != expected_kind:
            return (
                f"Media id '{value}' has kind '{item.kind}', "
                f"expected '{expected_kind}'"
            )
    return True


def _fingerprint_memory_batch_values(
    raw_values: Any,
    *,
    label: str,
    expected_kind: str,
    disable_in_memory: bool,
    use_mtime: bool,
) -> tuple[bool, tuple[str | float, ...]]:
    try:
        values = normalize_memory_batch_values(raw_values, label=label)
    except ValueError:
        return disable_in_memory, ("__unset__",)

    fingerprints: list[str | float] = []
    for value in values:
        if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
            fingerprints.append(
                _fingerprint_annotated_filepath(value, use_mtime=use_mtime)
            )
            continue

        item = REGISTRY.get(value, mark_accessed=False)
        if item is None or item.kind != expected_kind:
            fingerprints.append(value)
        else:
            fingerprints.append(hashlib.sha256(item.data).hexdigest())
    return disable_in_memory, tuple(fingerprints)


def _memory_batch_input(
    input_id: str,
    *,
    display_name: str,
    placeholder: str,
) -> io.MultiCombo.Input:
    # ComfyUI does not support remote options on MultiCombo. The bundled web
    # extension replaces this inert stock widget with an ordered selector that
    # reads the live registry/input-folder routes. Keep an empty option set here
    # so object_info never advertises a stale or semantically wrong source.
    return io.MultiCombo.Input(
        input_id,
        options=[],
        display_name=display_name,
        default=[],
        placeholder=placeholder,
        chip=True,
    )


def _fingerprint_annotated_filepath(raw_value: str, *, use_mtime: bool) -> str | float:
    media_path = folder_paths.get_annotated_filepath(raw_value)
    if use_mtime:
        return os.path.getmtime(media_path)

    digest = hashlib.sha256()
    with open(media_path, "rb") as media_file:
        digest.update(media_file.read())
    return digest.hexdigest()


def _load_image_from_filepath(image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = comfy.model_management.intermediate_dtype()
    device = comfy.model_management.intermediate_device()

    components = InputImpl.VideoFromFile(image_path).get_components()
    if components.images.shape[0] > 0:
        alpha = getattr(components, "alpha", None)
        mask = (
            (1.0 - alpha[..., -1]).to(device=device, dtype=dtype)
            if alpha is not None
            else torch.zeros(
                (components.images.shape[0], 64, 64),
                dtype=dtype,
                device=device,
            )
        )
        return components.images.to(device=device, dtype=dtype), mask

    # This fallback keeps animated WebP support for formats PyAV can't decode here.
    img = node_helpers.pillow(Image.open, image_path)
    output_images: list[torch.Tensor] = []
    output_masks: list[torch.Tensor] = []
    width: int | None = None
    height: int | None = None

    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

        if frame.mode == "I":
            frame = frame.point(lambda value: value * (1 / 255))
        rgb_frame = frame.convert("RGB")

        if len(output_images) == 0:
            width, height = rgb_frame.size

        if rgb_frame.size[0] != width or rgb_frame.size[1] != height:
            continue

        image = np.array(rgb_frame).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)[None,]

        if "A" in frame.getbands():
            mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        elif frame.mode == "P" and "transparency" in frame.info:
            mask = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        else:
            mask_tensor = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        output_images.append(image_tensor.to(dtype=dtype))
        output_masks.append(mask_tensor.unsqueeze(0).to(dtype=dtype))

        if img.format == "MPO":
            break

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image.to(device=device, dtype=dtype), output_mask.to(device=device, dtype=dtype)


def _load_image_from_bytes(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    img = node_helpers.pillow(Image.open, stdlib_io.BytesIO(data))
    output_images: list[torch.Tensor] = []
    output_masks: list[torch.Tensor] = []
    width: int | None = None
    height: int | None = None
    dtype = comfy.model_management.intermediate_dtype()

    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

        if frame.mode == "I":
            frame = frame.point(lambda value: value * (1 / 255))
        rgb_frame = frame.convert("RGB")

        if len(output_images) == 0:
            width, height = rgb_frame.size

        if rgb_frame.size[0] != width or rgb_frame.size[1] != height:
            continue

        image = np.array(rgb_frame).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)[None,]

        if "A" in frame.getbands():
            mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        elif frame.mode == "P" and "transparency" in frame.info:
            mask = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        else:
            mask_tensor = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        output_images.append(image_tensor.to(dtype=dtype))
        output_masks.append(mask_tensor.unsqueeze(0).to(dtype=dtype))

        if img.format == "MPO":
            break

    if len(output_images) > 1:
        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
    return output_images[0], output_masks[0]


def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2**15)
    if wav.dtype == torch.int32:
        return wav.float() / (2**31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _load_audio_from_source(source: str | stdlib_io.BytesIO) -> tuple[torch.Tensor, int]:
    with av.open(source) as audio_file:
        if not audio_file.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = audio_file.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames: list[torch.Tensor] = []
        for frame in audio_file.decode(streams=stream.index):
            buffer = torch.from_numpy(frame.to_ndarray())
            if buffer.shape[0] != n_channels:
                buffer = buffer.view(-1, n_channels).t()
            frames.append(buffer)

        if not frames:
            raise ValueError("No audio frames decoded.")

        waveform = torch.cat(frames, dim=1)
        waveform = _f32_pcm(waveform)
        return waveform, sample_rate


def _load_audio_from_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    return _load_audio_from_source(stdlib_io.BytesIO(data))


def _load_audio_from_filepath(audio_path: str) -> tuple[torch.Tensor, int]:
    return _load_audio_from_source(audio_path)


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


def _coerce_positive_fps(value: float) -> Fraction:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"FPS must be a positive finite number, got {value!r}")
    return Fraction(round(float(value) * 1000), 1000)


def _resample_frame_tensor_to_fps(
    images: torch.Tensor,
    *,
    source_fps: Fraction | float,
    target_fps: Fraction | float,
) -> torch.Tensor:
    source_frame_count = int(images.shape[0])
    if source_frame_count <= 0:
        raise ValueError("Video must contain at least one frame to resample FPS.")

    source_frame_rate = Fraction(source_fps)
    if source_frame_rate <= 0:
        raise ValueError(f"Video must have a positive frame rate, got {source_fps!r}")

    target_frame_rate = Fraction(target_fps)
    if target_frame_rate <= 0:
        raise ValueError(f"Target FPS must be positive, got {target_fps!r}")
    if target_frame_rate == source_frame_rate:
        return images

    duration_seconds = source_frame_count / float(source_frame_rate)
    target_frame_count = max(
        1,
        int(math.ceil(duration_seconds * float(target_frame_rate))),
    )
    target_timestamps = torch.arange(target_frame_count, dtype=torch.float64)
    target_timestamps /= float(target_frame_rate)
    source_indices = torch.round(target_timestamps * float(source_frame_rate)).to(
        dtype=torch.long
    )
    source_indices = source_indices.clamp(0, source_frame_count - 1)
    return images.index_select(0, source_indices.to(images.device))


def _resample_video_frames_to_fps(
    video: Input.Video,
    *,
    target_fps: float,
) -> Input.Video:
    components = video.get_components()
    target_frame_rate = _coerce_positive_fps(target_fps)

    # Match the frontend exporter closely: preserve the clip duration coverage by
    # rounding the frame count up to the next target-fps boundary, then sample the
    # nearest source frame for each target timestamp. This duplicates or drops
    # frames, but never blends them, which keeps binary mask mattes crisp.
    resampled_images = _resample_frame_tensor_to_fps(
        components.images,
        source_fps=components.frame_rate,
        target_fps=target_frame_rate,
    )
    if resampled_images is components.images:
        return video
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=resampled_images,
            audio=components.audio,
            frame_rate=target_frame_rate,
            metadata=components.metadata,
        )
    )


@PromptServer.instance.routes.post("/api/vlo-memory/register")
async def register_memory_media(request: web.Request) -> web.Response:
    post_data = await request.post()
    file_field = post_data.get("media")
    if not isinstance(file_field, web.FileField):
        return _json_error(400, "Missing uploaded file field 'media'")

    kind = _normalize_kind(post_data.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    filename = post_data.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        filename = file_field.filename or f"upload.{kind}"

    content_type = post_data.get("content_type")
    if not isinstance(content_type, str) or not content_type.strip():
        content_type = file_field.content_type or _DEFAULT_CONTENT_TYPES[kind]

    client_id = post_data.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        client_id = None

    media_bytes = file_field.file.read()

    try:
        item = REGISTRY.register(
            kind=kind,
            filename=filename,
            content_type=content_type,
            data=media_bytes,
            client_id=client_id,
        )
    except MediaTooLargeError as exc:
        return _json_error(413, str(exc))
    except MediaRegistryCapacityError as exc:
        return _json_error(507, str(exc))
    except ValueError as exc:
        return _json_error(400, str(exc))

    return web.json_response(_media_summary(item))


@PromptServer.instance.routes.get("/api/vlo-memory/options")
async def list_memory_media_options(request: web.Request) -> web.Response:
    kind = _normalize_kind(request.query.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    options = [item.media_id for item in REGISTRY.list_media(kind=kind)]
    return web.json_response(options)


@PromptServer.instance.routes.get("/api/vlo-memory/input-files")
async def list_memory_input_files(request: web.Request) -> web.Response:
    kind = _normalize_kind(request.query.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    content_types = _INPUT_KIND_CONTENT_TYPES.get(kind)
    if content_types is None:
        return _json_error(400, "Unsupported media kind for input folder listing")

    return web.json_response(_list_input_files(content_types))


@PromptServer.instance.routes.get("/api/vlo-memory/view/{media_id}")
async def view_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.get(media_id)
    if item is None:
        logger.warning("vlo memory media not found: media_id=%s", media_id)
        return _json_error(404, "Unknown media id")
    logger.debug(
        "vlo memory media served: media_id=%s filename=%s content_type=%s size_bytes=%s",
        media_id,
        item.filename,
        item.content_type,
        item.size_bytes,
    )
    return web.Response(
        body=item.data,
        content_type=item.content_type,
        headers={"Content-Disposition": f'inline; filename="{item.filename}"'},
    )


@PromptServer.instance.routes.get("/api/vlo-memory/item/{media_id}")
async def get_memory_media_item(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.get(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
    return web.json_response(_media_summary(item))


@PromptServer.instance.routes.delete("/api/vlo-memory/item/{media_id}")
async def delete_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.delete(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
    return web.json_response(_media_summary(item))


class vloMemoryLoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadImage",
            display_name="vlo Memory Load Image",
            category="image",
            inputs=[
                io.Combo.Input(
                    "image",
                    options=_list_input_files(["image"]),
                    upload=io.UploadType.image,
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=image",
                        refresh_button=True,
                    ),
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load the selected image from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
            ],
            outputs=[io.Image.Output(), io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, disable_in_memory=False) -> io.NodeOutput:
        if _should_load_from_filepath(image, disable_in_memory=disable_in_memory):
            image_path = folder_paths.get_annotated_filepath(image)
            output_image, output_mask = _load_image_from_filepath(image_path)
            return io.NodeOutput(output_image, output_mask)

        item = _get_media_item(image, expected_kind="image")
        output_image, output_mask = _load_image_from_bytes(item.data)
        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(cls, image, disable_in_memory=False):
        if _should_load_from_filepath(image, disable_in_memory=disable_in_memory):
            return _fingerprint_annotated_filepath(image, use_mtime=False)

        normalized_image = _normalize_media_id(image)
        if normalized_image is None:
            return "__unset__"
        item = REGISTRY.get(normalized_image, mark_accessed=False)
        if item is None:
            return normalized_image
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, image, disable_in_memory=False):
        if _should_load_from_filepath(image, disable_in_memory=disable_in_memory):
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
            return True

        normalized_image = _normalize_media_id(image)
        if normalized_image is None:
            return True
        if REGISTRY.get(normalized_image, mark_accessed=False) is None:
            return f"Invalid image id: {image}"
        return True


class vloMemoryLoadAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadAudio",
            display_name="vlo Memory Load Audio",
            category="audio",
            inputs=[
                # ComfyUI's native audio upload widget assumes an `audioUI` preview
                # widget that is only auto-injected for built-in audio node classes.
                # Keep this as a remote-backed combo so custom nodes do not crash the
                # frontend during widget initialization.
                io.Combo.Input(
                    "audio",
                    options=_list_input_files(["audio", "video"]),
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=audio",
                        refresh_button=True,
                    ),
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load the selected audio from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, disable_in_memory=False) -> io.NodeOutput:
        if _should_load_from_filepath(audio, disable_in_memory=disable_in_memory):
            audio_path = folder_paths.get_annotated_filepath(audio)
            waveform, sample_rate = _load_audio_from_filepath(audio_path)
            return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})

        item = _get_media_item(audio, expected_kind="audio")
        waveform, sample_rate = _load_audio_from_bytes(item.data)
        return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})

    @classmethod
    def fingerprint_inputs(cls, audio, disable_in_memory=False):
        if _should_load_from_filepath(audio, disable_in_memory=disable_in_memory):
            return _fingerprint_annotated_filepath(audio, use_mtime=False)

        normalized_audio = _normalize_media_id(audio)
        if normalized_audio is None:
            return "__unset__"
        item = REGISTRY.get(normalized_audio, mark_accessed=False)
        if item is None:
            return normalized_audio
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, audio, disable_in_memory=False):
        if _should_load_from_filepath(audio, disable_in_memory=disable_in_memory):
            if not folder_paths.exists_annotated_filepath(audio):
                return f"Invalid audio file: {audio}"
            return True

        normalized_audio = _normalize_media_id(audio)
        if normalized_audio is None:
            return True
        if REGISTRY.get(normalized_audio, mark_accessed=False) is None:
            return f"Invalid audio id: {audio}"
        return True


class vloMemoryLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadVideo",
            display_name="vlo Memory Load Video",
            category="image/video",
            inputs=[
                io.Combo.Input(
                    "file",
                    options=_list_input_files(["video"]),
                    upload=io.UploadType.video,
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=video",
                        refresh_button=True,
                    ),
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load the selected video from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, file, disable_in_memory=False) -> io.NodeOutput:
        if _should_load_from_filepath(file, disable_in_memory=disable_in_memory):
            video_path = folder_paths.get_annotated_filepath(file)
            return io.NodeOutput(InputImpl.VideoFromFile(video_path))

        item = _get_media_item(file, expected_kind="video")
        return io.NodeOutput(InputImpl.VideoFromFile(stdlib_io.BytesIO(item.data)))

    @classmethod
    def fingerprint_inputs(cls, file, disable_in_memory=False):
        if _should_load_from_filepath(file, disable_in_memory=disable_in_memory):
            return _fingerprint_annotated_filepath(file, use_mtime=True)

        normalized_file = _normalize_media_id(file)
        if normalized_file is None:
            return "__unset__"
        item = REGISTRY.get(normalized_file, mark_accessed=False)
        if item is None:
            return normalized_file
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, file, disable_in_memory=False):
        if _should_load_from_filepath(file, disable_in_memory=disable_in_memory):
            if not folder_paths.exists_annotated_filepath(file):
                return f"Invalid video file: {file}"
            return True

        normalized_file = _normalize_media_id(file)
        if normalized_file is None:
            return True
        if REGISTRY.get(normalized_file, mark_accessed=False) is None:
            return f"Invalid video id: {file}"
        return True


class vloMemoryLoadImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadImageBatch",
            display_name="vlo Memory Load Image Batch",
            category="image",
            description=(
                "Loads an ordered collection of images from vlo's in-memory registry "
                "or ComfyUI's input folder. Each output is a Comfy list item, so image "
                "dimensions do not need to match."
            ),
            inputs=[
                _memory_batch_input(
                    "images",
                    display_name="Images",
                    placeholder="Select images in reference order",
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load every selection from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="Ordered image list.",
                    is_output_list=True,
                ),
                io.Mask.Output(
                    display_name="masks",
                    tooltip="Masks in the same order as the image list.",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, images, disable_in_memory=False) -> io.NodeOutput:
        values = normalize_memory_batch_values(images, label="image")
        output_images: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []
        for value in values:
            if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
                image_path = folder_paths.get_annotated_filepath(value)
                image, mask = _load_image_from_filepath(image_path)
            else:
                item = _get_media_item(value, expected_kind="image")
                image, mask = _load_image_from_bytes(item.data)
            output_images.append(image)
            output_masks.append(mask)
        return io.NodeOutput(output_images, output_masks)

    @classmethod
    def fingerprint_inputs(cls, images, disable_in_memory=False):
        return _fingerprint_memory_batch_values(
            images,
            label="image",
            expected_kind="image",
            disable_in_memory=disable_in_memory,
            use_mtime=False,
        )

    @classmethod
    def validate_inputs(cls, images, disable_in_memory=False):
        return _validate_memory_batch_values(
            images,
            label="image",
            expected_kind="image",
            disable_in_memory=disable_in_memory,
        )


class vloMemoryLoadAudioBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadAudioBatch",
            display_name="vlo Memory Load Audio Batch",
            category="audio",
            description=(
                "Loads an ordered collection of audio clips from vlo's in-memory "
                "registry or ComfyUI's input folder as a Comfy list."
            ),
            inputs=[
                _memory_batch_input(
                    "audios",
                    display_name="Audio clips",
                    placeholder="Select audio clips in reference order",
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load every selection from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
            ],
            outputs=[
                io.Audio.Output(
                    display_name="audios",
                    tooltip="Ordered audio list.",
                    is_output_list=True,
                )
            ],
        )

    @classmethod
    def execute(cls, audios, disable_in_memory=False) -> io.NodeOutput:
        values = normalize_memory_batch_values(audios, label="audio clip")
        output: list[dict[str, Any]] = []
        for value in values:
            if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
                audio_path = folder_paths.get_annotated_filepath(value)
                waveform, sample_rate = _load_audio_from_filepath(audio_path)
            else:
                item = _get_media_item(value, expected_kind="audio")
                waveform, sample_rate = _load_audio_from_bytes(item.data)
            output.append(
                {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            )
        return io.NodeOutput(output)

    @classmethod
    def fingerprint_inputs(cls, audios, disable_in_memory=False):
        return _fingerprint_memory_batch_values(
            audios,
            label="audio clip",
            expected_kind="audio",
            disable_in_memory=disable_in_memory,
            use_mtime=False,
        )

    @classmethod
    def validate_inputs(cls, audios, disable_in_memory=False):
        return _validate_memory_batch_values(
            audios,
            label="audio clip",
            expected_kind="audio",
            disable_in_memory=disable_in_memory,
        )


class vloMemoryLoadVideoBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMemoryLoadVideoBatch",
            display_name="vlo Memory Load Video Batch",
            category="image/video",
            description=(
                "Loads an ordered collection of videos from vlo's in-memory registry "
                "or ComfyUI's input folder as a Comfy list."
            ),
            inputs=[
                _memory_batch_input(
                    "files",
                    display_name="Videos",
                    placeholder="Select videos in reference order",
                ),
                io.Boolean.Input(
                    "disable_in_memory",
                    default=False,
                    tooltip=(
                        "When true, load every selection from ComfyUI's normal input "
                        "directory instead of the vlo in-memory registry."
                    ),
                ),
                # Appended last on purpose: workflows saved before this input
                # existed restore widget values by position, so the two
                # original widgets have to keep their slots.
                io.String.Input(
                    "include_audio",
                    default="",
                    tooltip=(
                        "Per-video audio inclusion, as a comma-separated flag list "
                        "in selection order (for example '1,0,1'). Unset videos "
                        "are excluded. Feed the 'use audio' output to a consumer "
                        "that takes a BOOLEAN list, such as the vlo MiniMax H3 "
                        "adapter's use_embedded_video_audio."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="videos",
                    tooltip="Ordered video list.",
                    is_output_list=True,
                ),
                io.Boolean.Output(
                    display_name="use audio",
                    tooltip=(
                        "Audio-inclusion flags in the same order as the video "
                        "list, one per video."
                    ),
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        files,
        disable_in_memory=False,
        include_audio="",
    ) -> io.NodeOutput:
        values = normalize_memory_batch_values(files, label="video")
        audio_flags = normalize_memory_batch_flags(
            include_audio,
            count=len(values),
            label="Video audio inclusion",
        )
        output: list[Input.Video] = []
        for value in values:
            if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
                video_path = folder_paths.get_annotated_filepath(value)
                output.append(InputImpl.VideoFromFile(video_path))
            else:
                item = _get_media_item(value, expected_kind="video")
                output.append(
                    InputImpl.VideoFromFile(stdlib_io.BytesIO(item.data))
                )
        return io.NodeOutput(output, audio_flags)

    @classmethod
    def fingerprint_inputs(cls, files, disable_in_memory=False, include_audio=""):
        changed, fingerprints = _fingerprint_memory_batch_values(
            files,
            label="video",
            expected_kind="video",
            disable_in_memory=disable_in_memory,
            use_mtime=True,
        )
        # The flags are part of what this node delivers, so flipping one has to
        # invalidate the cached execution just like swapping a video does.
        return changed, (*fingerprints, f"audio:{include_audio}")

    @classmethod
    def validate_inputs(cls, files, disable_in_memory=False, include_audio=""):
        result = _validate_memory_batch_values(
            files,
            label="video",
            expected_kind="video",
            disable_in_memory=disable_in_memory,
        )
        if result is not True:
            return result
        try:
            normalize_memory_batch_flags(
                include_audio,
                count=len(normalize_memory_batch_values(files, label="video")),
                label="Video audio inclusion",
            )
        except ValueError as exc:
            return str(exc)
        return True


def _unwrap_list_input(value: Any, *, label: str) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) != 1:
        raise ValueError(f"{label} expects exactly one value, received {len(value)}")
    return value[0]


def _normalize_list_input(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _enforce_reference_limit(values: list[Any], *, label: str, maximum: int) -> None:
    if len(values) > maximum:
        display_label = label if maximum == 1 else f"{label}s"
        raise ValueError(
            f"MiniMax H3 supports at most {maximum} {display_label}; "
            f"received {len(values)}"
        )


def _resolve_per_video_flags(
    raw_flags: Any,
    *,
    count: int,
    label: str,
    default: bool,
) -> list[bool]:
    # `is_input_list=True` means a widget arrives as a one-item list while a
    # connected BOOLEAN list arrives with one entry per video. Broadcasting the
    # single-value case is what lets vlo move from one node-wide toggle to
    # per-upload tickboxes later without a schema change or a node_id bump.
    flags = _normalize_list_input(raw_flags)
    if not flags:
        return [default] * count
    if len(flags) == 1:
        return [bool(flags[0])] * count
    if len(flags) != count:
        raise ValueError(
            f"{label} expects a single value, or one value per reference video; "
            f"received {len(flags)} for {count} videos"
        )
    return [bool(flag) for flag in flags]


def _get_native_minimax_h3_reference_node() -> type[io.ComfyNode]:
    # Keep MiniMax's model stack out of this extension's import path. Besides
    # reducing startup coupling, this lets the other VLO nodes keep working on
    # ComfyUI builds that predate the native H3 node.
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "The native MiniMax H3 Reference to Video node is unavailable. "
            "Update ComfyUI and its Python dependencies before using this adapter."
        ) from exc
    return MiniMaxH3ReferenceToVideo


def _get_native_minimax_h3_reference_contract() -> tuple[str, dict[str, tuple[str, int]]]:
    native_node = _get_native_minimax_h3_reference_node()
    try:
        schema = native_node.GET_SCHEMA()
    except Exception as exc:
        raise RuntimeError(
            "Could not inspect the native MiniMax H3 Reference to Video schema"
        ) from exc

    expected_node_id = "MiniMaxH3ReferenceToVideo"
    if schema.node_id != expected_node_id:
        raise RuntimeError(
            "Incompatible native MiniMax H3 node id: "
            f"expected '{expected_node_id}', got '{schema.node_id}'"
        )

    inputs_by_id = {input_spec.id: input_spec for input_spec in schema.inputs}
    expected_fixed_types = {
        "clip": "CLIP",
        "vae": "VAE",
        "audio_vae": "VAE",
        "prompt": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "ref_image_size": "COMBO",
    }
    for input_id, expected_type in expected_fixed_types.items():
        input_spec = inputs_by_id.get(input_id)
        actual_type = getattr(input_spec, "io_type", None)
        if actual_type != expected_type:
            raise RuntimeError(
                "Incompatible native MiniMax H3 input "
                f"'{input_id}': expected {expected_type}, got {actual_type}"
            )

    expected_reference_types = {
        "ref_images": "IMAGE",
        "ref_videos": "IMAGE",
        "ref_video_audios": "AUDIO",
        "ref_audios": "AUDIO",
    }
    reference_contract: dict[str, tuple[str, int]] = {}
    for input_id, expected_type in expected_reference_types.items():
        input_spec = inputs_by_id.get(input_id)
        if not isinstance(input_spec, io.Autogrow.Input):
            raise RuntimeError(
                f"Incompatible native MiniMax H3 input '{input_id}': expected Autogrow"
            )

        template = input_spec.template
        actual_type = getattr(template.input, "io_type", None)
        prefix = getattr(template, "prefix", None)
        maximum = getattr(template, "max", None)
        if actual_type != expected_type:
            raise RuntimeError(
                "Incompatible native MiniMax H3 reference input "
                f"'{input_id}': expected {expected_type}, got {actual_type}"
            )
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError(
                f"Incompatible native MiniMax H3 input '{input_id}': missing prefix"
            )
        if not isinstance(maximum, int) or maximum < 1:
            raise RuntimeError(
                f"Incompatible native MiniMax H3 input '{input_id}': invalid maximum"
            )
        reference_contract[input_id] = (prefix, maximum)

    output_types = [output.io_type for output in schema.outputs]
    if output_types != ["CONDITIONING", "LATENT"]:
        raise RuntimeError(
            "Incompatible native MiniMax H3 outputs: expected CONDITIONING, LATENT; "
            f"got {', '.join(output_types) or 'none'}"
        )
    return schema.node_id, reference_contract


class vloMiniMaxH3ReferenceToVideoBatch(io.ComfyNode):
    """Adapt VLO's media-list outputs to ComfyUI's native MiniMax H3 node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMiniMaxH3ReferenceToVideoBatch",
            display_name="vlo MiniMax H3 Reference to Video (Batch)",
            category="model/conditioning/minimax",
            description=(
                "Consumes ordered IMAGE, VIDEO, and AUDIO lists and expands to "
                "ComfyUI's native MiniMax H3 Reference to Video node."
            ),
            is_input_list=True,
            enable_expand=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=_COMFY_MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=_COMFY_MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip=(
                        "Frame count at 24 fps (124 is about 5 seconds; the trained "
                        "range is approximately 124-362)."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip=(
                        "Use 'match' to limit each image to the generation pixel area, "
                        "or 'max' for the native 2048px-short-edge reference pipeline."
                    ),
                ),
                io.Image.Input(
                    "ref_images",
                    optional=True,
                    tooltip="Ordered reference image list. Limit follows the native node.",
                ),
                io.Video.Input(
                    "ref_videos",
                    optional=True,
                    tooltip=(
                        "Ordered reference video list. Videos are resampled to the "
                        "native node's required 24 fps. Limit follows the native node."
                    ),
                ),
                io.Boolean.Input(
                    "use_embedded_video_audio",
                    default=False,
                    tooltip=(
                        "Use the audio embedded in each reference video as its "
                        "soundtrack. MiniMax treats a reference video's own sound as "
                        "a separate <Audio N> reference that must be enabled, so this "
                        "is off by default. Connect a BOOLEAN list to set it per "
                        "video; a single value applies to every video."
                    ),
                ),
                io.Audio.Input(
                    "ref_video_audios",
                    optional=True,
                    tooltip=(
                        "Optional ordered soundtrack overrides for the reference videos. "
                        "An override always wins, whether or not embedded audio is "
                        "enabled for that video."
                    ),
                ),
                io.Audio.Input(
                    "ref_audios",
                    optional=True,
                    tooltip=(
                        "Ordered standalone reference audio list. Limit follows the "
                        "native node."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        ref_images=None,
        ref_videos=None,
        use_embedded_video_audio=False,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        native_node_id, reference_contract = (
            _get_native_minimax_h3_reference_contract()
        )
        images = _normalize_list_input(ref_images)
        videos = _normalize_list_input(ref_videos)
        video_audio_overrides = _normalize_list_input(ref_video_audios)
        audios = _normalize_list_input(ref_audios)

        image_prefix, image_max = reference_contract["ref_images"]
        video_prefix, video_max = reference_contract["ref_videos"]
        video_audio_prefix, video_audio_max = reference_contract[
            "ref_video_audios"
        ]
        audio_prefix, audio_max = reference_contract["ref_audios"]
        _enforce_reference_limit(
            images,
            label="reference image",
            maximum=image_max,
        )
        _enforce_reference_limit(
            videos,
            label="reference video",
            maximum=video_max,
        )
        _enforce_reference_limit(
            video_audio_overrides,
            label="reference video soundtrack",
            maximum=video_audio_max,
        )
        _enforce_reference_limit(
            audios,
            label="standalone reference audio",
            maximum=audio_max,
        )
        if len(video_audio_overrides) > len(videos):
            raise ValueError(
                "Reference video soundtrack overrides cannot outnumber reference videos"
            )

        native_images = {
            f"ref_images.{image_prefix}{index}": image
            for index, image in enumerate(images)
        }
        embedded_audio_flags = _resolve_per_video_flags(
            use_embedded_video_audio,
            count=len(videos),
            label="use_embedded_video_audio",
            default=False,
        )

        native_videos: dict[str, torch.Tensor] = {}
        native_video_audios: dict[str, Any] = {}
        for index, video in enumerate(videos):
            components = video.get_components()
            native_videos[
                f"ref_videos.{video_prefix}{index}"
            ] = _resample_frame_tensor_to_fps(
                components.images,
                source_fps=components.frame_rate,
                target_fps=Fraction(24, 1),
            )
            if index < len(video_audio_overrides):
                soundtrack = video_audio_overrides[index]
            elif embedded_audio_flags[index]:
                soundtrack = components.audio
            else:
                soundtrack = None
            if soundtrack is not None:
                native_video_audios[
                    f"ref_video_audios.{video_audio_prefix}{index}"
                ] = soundtrack
        _enforce_reference_limit(
            list(native_video_audios.values()),
            label="reference video soundtrack",
            maximum=video_audio_max,
        )

        native_audios = {
            f"ref_audios.{audio_prefix}{index}": audio
            for index, audio in enumerate(audios)
        }
        graph = GraphBuilder()
        native_graph_node = graph.node(
            native_node_id,
            clip=_unwrap_list_input(clip, label="clip"),
            vae=_unwrap_list_input(vae, label="vae"),
            audio_vae=_unwrap_list_input(audio_vae, label="audio_vae"),
            prompt=_unwrap_list_input(prompt, label="prompt"),
            width=_unwrap_list_input(width, label="width"),
            height=_unwrap_list_input(height, label="height"),
            length=_unwrap_list_input(length, label="length"),
            ref_image_size=_unwrap_list_input(
                ref_image_size,
                label="ref_image_size",
            ),
            **native_images,
            **native_videos,
            **native_video_audios,
            **native_audios,
        )
        return io.NodeOutput(
            native_graph_node.out(0),
            native_graph_node.out(1),
            expand=graph.finalize(),
        )


class vloVideoConvertFps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloVideoConvertFps",
            search_aliases=[
                "convert video fps",
                "resample video fps",
                "retime video fps",
                "change video fps",
            ],
            display_name="vlo Video Convert FPS",
            category="image/video",
            description=(
                "Resamples a video to a target FPS while preserving audio and overall clip "
                "coverage. Frames are duplicated or dropped using nearest-frame temporal "
                "sampling; no frame blending is applied."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip="The source video to retime.",
                ),
                io.Float.Input(
                    "fps",
                    default=25.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "Target frames per second. Duration is preserved approximately by "
                        "duplicating or dropping frames rather than changing playback speed."
                    ),
                ),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, video: Input.Video, fps: float) -> io.NodeOutput:
        return io.NodeOutput(_resample_video_frames_to_fps(video, target_fps=fps))


class vloSaveImageWebsocketBMP(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloSaveImageWebsocketBMP",
            search_aliases=["bmp websocket", "save image websocket bmp"],
            display_name="vlo Save Image Websocket (BMP)",
            category="api/image",
            description=(
                "Streams full-size images to the websocket as BMP payloads. "
                "This avoids PNG encode time at the cost of larger payloads."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The image batch to stream to the websocket as BMP.",
                )
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image) -> io.NodeOutput:
        total_images = int(images.shape[0]) if hasattr(images, "shape") else len(images)
        if total_images <= 0:
            return io.NodeOutput()

        node_id, prompt_id = _get_execution_ids()

        for step, image in enumerate(images, start=1):
            pil_image = _tensor_to_pil_rgb_image(image)
            buffer = stdlib_io.BytesIO()
            pil_image.save(buffer, format="BMP")
            preview_metadata: dict[str, Any] = {"image_type": "image/bmp"}
            if node_id is not None:
                preview_metadata["node_id"] = node_id
            if prompt_id is not None:
                preview_metadata["prompt_id"] = prompt_id
            preview_payload = _encode_payload_with_metadata(
                buffer.getvalue(),
                preview_metadata,
            )
            _send_progress_update(
                step,
                total_images,
                node_id=node_id,
                prompt_id=prompt_id,
            )
            _send_binary_event(
                BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA,
                preview_payload,
            )

        return io.NodeOutput()


class vloSaveVideoWebsocket(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloSaveVideoWebsocket",
            search_aliases=["export video websocket", "save video websocket"],
            display_name="vlo Save Video Websocket",
            category="api/video",
            description=(
                "Stores the input video in vlo memory and emits a websocket result "
                "entry so the frontend can fetch it immediately without saving to disk."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The video to expose to the frontend."),
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip=(
                        "The filename prefix to use for the in-memory video result. "
                        "Formatting tokens follow the same rules as Save Video."
                    ),
                ),
                io.Combo.Input(
                    "format",
                    options=Types.VideoContainer.as_input(),
                    default="auto",
                    tooltip="The container format to use for the emitted video.",
                ),
                io.Combo.Input(
                    "codec",
                    options=Types.VideoCodec.as_input(),
                    default="auto",
                    tooltip="The codec to use for the emitted video.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        format: str,
        codec: str,
    ) -> io.NodeOutput:
        width, height = video.get_dimensions()
        _, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )

        container_format = Types.VideoContainer(format)
        video_codec = Types.VideoCodec(codec)
        file = (
            f"{filename}_{counter:05}_."
            f"{Types.VideoContainer.get_extension(container_format)}"
        )

        buffer = stdlib_io.BytesIO()
        video.save_to(
            buffer,
            format=container_format,
            codec=video_codec,
            metadata=_build_saved_video_metadata(cls),
        )

        item = REGISTRY.register(
            kind="video",
            filename=file,
            content_type=_resolve_video_content_type(container_format),
            data=buffer.getvalue(),
            client_id=_get_client_id(),
        )
        node_id, prompt_id = _get_execution_ids()
        logger.info(
            "Registered vlo websocket video output: media_id=%s filename=%s subfolder=%s content_type=%s size_bytes=%s client_id=%s node_id=%s prompt_id=%s",
            item.media_id,
            item.filename,
            subfolder,
            item.content_type,
            item.size_bytes,
            item.client_id,
            node_id,
            prompt_id,
        )

        return io.NodeOutput(
            ui={
                "videos": [
                    _build_memory_output_item(item, subfolder=subfolder),
                ]
            }
        )


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
                "source replaces the destination."
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
        dest_samples = destination["samples"]
        src_samples = source["samples"]

        output = destination.copy()
        output["samples"] = dest_samples.clone()

        mask = destination.get("noise_mask")
        if mask is None:
            return io.NodeOutput(output)

        mask = mask.to(dtype=dest_samples.dtype, device=dest_samples.device)

        if force_binary_mask:
            mask = (mask >= 0.5).to(dtype=mask.dtype)

        try:
            output["samples"] = src_samples * mask + dest_samples * (1.0 - mask)
        except RuntimeError as e:
            raise ValueError(
                f"Could not composite: destination {tuple(dest_samples.shape)}, "
                f"source {tuple(src_samples.shape)}, mask {tuple(mask.shape)} "
                f"are not broadcast-compatible. Ensure the mask is preshaped for this latent."
            ) from e

        if clear_mask:
            output.pop("noise_mask", None)

        return io.NodeOutput(output)


class vloGateNone(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("value")
        return io.Schema(
            node_id="vloGateNone",
            search_aliases=["gate", "null gate", "disable pass-through", "none gate"],
            display_name="vlo Gate None",
            category="utils/logic",
            description=(
                "Passes any connected value through unchanged unless disabled is true, "
                "in which case the output is None."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=template,
                    tooltip="Any connected value to pass through or suppress.",
                ),
                io.Boolean.Input(
                    "disabled",
                    default=False,
                    tooltip="When true, suppresses the value and outputs None instead.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="value",
                )
            ],
        )

    @classmethod
    def execute(cls, value, disabled=False) -> io.NodeOutput:
        return io.NodeOutput(None if disabled else value)


class vloLogicNot(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloLogicNot",
            search_aliases=["not", "invert boolean", "logic not", "boolean not"],
            display_name="vlo Logic Not",
            category="utils/logic",
            description="Inverts an incoming boolean value.",
            inputs=[
                io.Boolean.Input(
                    "value",
                    force_input=True,
                    tooltip="The boolean value to invert.",
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="value")
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        return io.NodeOutput(not value)


_TTM_OUTER_SAMPLE_KEY = "vloTTM_OuterSample"


def _ttm_align_mask(
    mask: torch.Tensor,
    latent_shape: torch.Size,
    temporal_ratio: int,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"mask must be [H, W] or [frames, H, W], got {tuple(mask.shape)}."
        )

    latent_frames, latent_height, latent_width = (
        latent_shape[2],
        latent_shape[-2],
        latent_shape[-1],
    )
    mask = mask.to(torch.float32)
    frames = mask.shape[0]

    # The VAE encodes source frame 0 into latent frame 0, then frames
    # (ratio*j - ratio + 1) .. (ratio*j) into latent frame j. Striding by the ratio
    # therefore lands every sampled frame inside the group it masks. Interpolating
    # along time instead would blend neighbouring frames and drift out of alignment,
    # smearing the mask of anything that moves.
    if frames == 1:
        mask = mask.expand(latent_frames, -1, -1)
    elif frames != latent_frames:
        strided = mask[::temporal_ratio]
        if strided.shape[0] == latent_frames:
            mask = strided
        else:
            logger.warning(
                "vloTimeToMove: %d mask frames do not stride onto %d latent frames at "
                "ratio %d; falling back to nearest-frame resampling, which may misalign "
                "the mask. Feed a mask whose frame count matches the reference video.",
                frames,
                latent_frames,
                temporal_ratio,
            )
            index = torch.linspace(0, frames - 1, latent_frames).round().long()
            mask = mask[index]

    # Nearest keeps the mask hard-edged. A linear kernel would leave a ring of partial
    # values around the subject, blending reference into generated content there.
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(1),
        size=(latent_height, latent_width),
        mode="nearest",
    )
    return mask.squeeze(1).view(1, 1, latent_frames, latent_height, latent_width)


class _TTMDenoiseMaskSchedule:
    """Closes the TTM window once the schedule drops past sigma_end.

    KSamplerX0Inpaint holds the masked region for the whole run; TTM only wants it held
    for the opening steps, and this is the hook Comfy provides for varying that per sigma.
    """

    def __init__(self, sigma_end):
        self.sigma_end = sigma_end

    def __call__(self, sigma, denoise_mask, extra_options=None):
        if float(sigma.flatten()[0]) < self.sigma_end:
            return torch.ones_like(denoise_mask)
        return denoise_mask


class _TTMOuterSample:
    def __init__(self, reference_latents, mask, start_step, end_step):
        self.reference_latents = reference_latents
        self.mask = mask
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None, *args, **kwargs):
        guider = executor.class_obj
        model = guider.model_patcher.model

        if torch.count_nonzero(noise) == 0:
            logger.warning(
                "vloTimeToMove: this sampler adds no noise, so there is nothing to seed the "
                "reference into and no noise to hold it with; skipping TTM here. Patch only "
                "the model of the sampler that starts the schedule."
            )
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs)

        latent_shapes = kwargs.get("latent_shapes") or [noise.shape]
        latent_shape = latent_shapes[0]

        reference = self.reference_latents["samples"]
        if tuple(reference.shape) != tuple(latent_shape):
            raise ValueError(
                f"reference_latents {tuple(reference.shape)} must match the sampled latent "
                f"{tuple(latent_shape)}. Encode the reference video at the same resolution "
                f"and frame count the sampler is generating."
            )

        available_steps = sigmas.shape[-1] - 1
        if self.start_step >= available_steps:
            raise ValueError(
                f"start_step ({self.start_step}) must be less than the {available_steps} "
                f"steps this sampler runs."
            )
        if self.start_step == 0:
            logger.warning(
                "vloTimeToMove: start_step is 0, where sigma is 1.0 and the init is therefore "
                "pure noise with no trace of the reference. The motion cue TTM relies on comes "
                "from that init; use 1 or more."
            )
        if denoise_mask is not None:
            logger.warning(
                "vloTimeToMove: replacing the noise_mask already on the sampled latent; TTM "
                "drives the denoise mask itself."
            )

        # Handed over raw. inner_sample runs process_latent_in on it, and KSAMPLER.sample then
        # keeps that normalised tensor as KSamplerX0Inpaint's reference -- so both the init and
        # the per-step hold read the reference in model space, from one source.
        # outer_sample moves it onto the load device below us, along with noise and sigmas.
        latent_image = reference

        # Dropping the leading sigmas is what makes x0 a partially-noised reference rather than
        # pure noise, and skips the steps we jumped over. Equivalent to raising the sampler's
        # start_at_step, but kept here so the two can't fall out of step.
        sigmas = sigmas[self.start_step:]

        # -1 because holding the region *during* the step at sigmas[k] leaves it held at
        # sigmas[k+1]. So to stop holding at end_step, the last pinned call is end_step - 1.
        # This is what makes end_step mean the same thing it means in the TTM reference
        # implementation, where the hold runs while step < end_step.
        last_index = self.end_step - self.start_step - 1
        if last_index < 0:
            # Window closed: seed the init from the reference and let the region run free.
            return executor(noise, latent_image, sampler, sigmas, None, *args, **kwargs)

        # The load device, not noise.device: noise is still on the CPU here, and unlike noise,
        # latent_image and sigmas, outer_sample never moves denoise_mask -- Comfy normally
        # places it in CFGGuider.sample via prepare_mask, which has already run by now.
        ttm_mask = _ttm_align_mask(
            self.mask,
            latent_shape,
            getattr(model.latent_format, "temporal_downscale_ratio", 4),
        ).to(device=guider.model_patcher.load_device, dtype=torch.float32)

        # Comfy's denoise_mask marks where to *denoise*; ours marks where to hold the
        # reference, so it goes in inverted. KSamplerX0Inpaint (samplers.py:633) then noises
        # the reference to each evaluation's own sigma on the way in and pins the x0
        # prediction to it on the way out -- which is what actually holds the region, and
        # holds it for any solver, including ones that evaluate off-schedule.
        denoise_mask = 1.0 - ttm_mask

        sigma_end = float(sigmas[min(last_index, sigmas.shape[-1] - 1)])
        guider.model_options["denoise_mask_function"] = _TTMDenoiseMaskSchedule(sigma_end)

        return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs)


class vloTimeToMove(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloTimeToMove",
            search_aliases=["ttm", "time to move", "motion transfer", "cut and drag"],
            display_name="vlo Time-to-Move (TTM)",
            category="advanced/model",
            description=(
                "Patches a video model to follow the motion in a reference clip, as in "
                "Time-to-Move (https://github.com/time-to-move/TTM). The reference latents "
                "seed the sampler's starting latent, which is what carries the intended "
                "motion, and the masked region is then held to the reference for the opening "
                "steps via Comfy's own inpaint path, so it works with any sampler. Patch only "
                "the model of the sampler that starts the schedule, and leave that sampler's "
                "start_at_step at 0. Drives the denoise mask, so any noise_mask already on "
                "the sampled latent is replaced."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "reference_latents",
                    tooltip=(
                        "Encoded reference video, e.g. the cut-and-drag clip. Must match the "
                        "resolution and frame count being sampled. This replaces whatever "
                        "latent is wired into the sampler."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "White marks the region held to the reference; black is left free for "
                        "the model to generate. For a moving subject that means white "
                        "background and a black hole over the subject. Pixel resolution and "
                        "frame count are matched to the latent grid automatically."
                    ),
                ),
                io.Int.Input(
                    "start_step",
                    default=1,
                    min=0,
                    max=1000,
                    tooltip=(
                        "Step whose noise level the reference is seeded at. The sampler skips "
                        "the steps before it. Higher values leave less noise on the reference, "
                        "binding the result more tightly to it at the cost of a step and of "
                        "the model's freedom to clean up paste artifacts. 0 is a no-op: sigma "
                        "is 1.0 there and the reference washes out entirely."
                    ),
                ),
                io.Int.Input(
                    "end_step",
                    default=2,
                    min=0,
                    max=1000,
                    tooltip=(
                        "The step at which the region stops being held to the reference and "
                        "starts denoising freely. Exclusive, and counted the same way the TTM "
                        "reference implementation counts it. Set at or below start_step to "
                        "seed the init only and never hold the region at all."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, reference_latents, mask, start_step=1, end_step=2) -> io.NodeOutput:
        patched = model.clone()
        patched.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            _TTM_OUTER_SAMPLE_KEY,
            _TTMOuterSample(reference_latents, mask, start_step, end_step),
        )
        return io.NodeOutput(patched)


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
            vloLatentCompositeMasked,
            vloGateNone,
            vloLogicNot,
            vloTimeToMove,
        ]


async def comfy_entrypoint() -> vloExtension:
    return vloExtension()
