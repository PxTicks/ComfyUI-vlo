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
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io
from comfy_execution.utils import get_executing_context

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
        alpha = components.alpha
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


def _resample_video_frames_to_fps(
    video: Input.Video,
    *,
    target_fps: float,
) -> Input.Video:
    components = video.get_components()
    source_images = components.images
    source_frame_count = int(source_images.shape[0])
    if source_frame_count <= 0:
        raise ValueError("Video must contain at least one frame to resample FPS.")

    source_fps = Fraction(components.frame_rate)
    if source_fps <= 0:
        raise ValueError(
            f"Video must have a positive frame rate, got {components.frame_rate!r}"
        )

    target_frame_rate = _coerce_positive_fps(target_fps)
    if target_frame_rate == source_fps:
        return video

    # Match the frontend exporter closely: preserve the clip duration coverage by
    # rounding the frame count up to the next target-fps boundary, then sample the
    # nearest source frame for each target timestamp. This duplicates or drops
    # frames, but never blends them, which keeps binary mask mattes crisp.
    duration_seconds = source_frame_count / float(source_fps)
    target_frame_count = max(1, int(math.ceil(duration_seconds * float(target_frame_rate))))
    target_timestamps = torch.arange(target_frame_count, dtype=torch.float64)
    target_timestamps /= float(target_frame_rate)
    source_indices = torch.round(target_timestamps * float(source_fps)).to(
        dtype=torch.long
    )
    source_indices = source_indices.clamp(0, source_frame_count - 1)

    resampled_images = source_images.index_select(0, source_indices.to(source_images.device))
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
        logger.warning("VLO memory media not found: media_id=%s", media_id)
        return _json_error(404, "Unknown media id")
    logger.debug(
        "VLO memory media served: media_id=%s filename=%s content_type=%s size_bytes=%s",
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


@PromptServer.instance.routes.delete("/api/vlo-memory/item/{media_id}")
async def delete_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.delete(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
    return web.json_response(_media_summary(item))


class VLOMemoryLoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOMemoryLoadImage",
            display_name="VLO Memory Load Image",
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
                        "directory instead of the VLO in-memory registry."
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


class VLOMemoryLoadAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOMemoryLoadAudio",
            display_name="VLO Memory Load Audio",
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
                        "directory instead of the VLO in-memory registry."
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


class VLOMemoryLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOMemoryLoadVideo",
            display_name="VLO Memory Load Video",
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
                        "directory instead of the VLO in-memory registry."
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


class VLOVideoConvertFps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOVideoConvertFps",
            search_aliases=[
                "convert video fps",
                "resample video fps",
                "retime video fps",
                "change video fps",
            ],
            display_name="VLO Video Convert FPS",
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


class VLOSaveImageWebsocketBMP(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOSaveImageWebsocketBMP",
            search_aliases=["bmp websocket", "save image websocket bmp"],
            display_name="VLO Save Image Websocket (BMP)",
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


class VLOSaveVideoWebsocket(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOSaveVideoWebsocket",
            search_aliases=["export video websocket", "save video websocket"],
            display_name="VLO Save Video Websocket",
            category="api/video",
            description=(
                "Stores the input video in VLO memory and emits a websocket result "
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
            "Registered VLO websocket video output: media_id=%s filename=%s subfolder=%s content_type=%s size_bytes=%s client_id=%s node_id=%s prompt_id=%s",
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


class VLOLatentCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLOLatentCompositeMasked",
            search_aliases=["vlo composite latent", "vlo inpaint latent"],
            display_name="VLO Latent Composite Masked",
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


class VLOGateNone(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("value")
        return io.Schema(
            node_id="VLOGateNone",
            search_aliases=["gate", "null gate", "disable pass-through", "none gate"],
            display_name="VLO Gate None",
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


class VLOExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VLOMemoryLoadImage,
            VLOMemoryLoadAudio,
            VLOMemoryLoadVideo,
            VLOVideoConvertFps,
            VLOSaveImageWebsocketBMP,
            VLOSaveVideoWebsocket,
            LTXSetAudioLatentBinaryMasks,
            VLOLatentCompositeMasked,
            VLOGateNone,
        ]


async def comfy_entrypoint() -> VLOExtension:
    return VLOExtension()
