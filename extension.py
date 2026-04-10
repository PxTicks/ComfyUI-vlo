from __future__ import annotations

import hashlib
import io as stdlib_io
from typing import Any

import av
import node_helpers
import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence
from server import PromptServer
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, InputImpl, io

from .media_registry import (
    MediaItem,
    MediaRegistry,
    MediaRegistryCapacityError,
    MediaTooLargeError,
)


REGISTRY = MediaRegistry()
_DEFAULT_CONTENT_TYPES = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/wav",
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
    item = REGISTRY.get(media_id)
    if item is None:
        raise ValueError(f"Unknown media id: {media_id}")
    if expected_kind is not None and item.kind != expected_kind:
        raise ValueError(
            f"Media id '{media_id}' has kind '{item.kind}', expected '{expected_kind}'"
        )
    return item


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


def _load_audio_from_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    with av.open(stdlib_io.BytesIO(data)) as audio_file:
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


@PromptServer.instance.routes.get("/api/vlo-memory/view/{media_id}")
async def view_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.get(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
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
                    options=[],
                    upload=io.UploadType.image,
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=image",
                        refresh_button=True,
                    ),
                )
            ],
            outputs=[io.Image.Output(), io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        item = _get_media_item(image, expected_kind="image")
        output_image, output_mask = _load_image_from_bytes(item.data)
        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(cls, image):
        item = REGISTRY.get(image, mark_accessed=False)
        if item is None:
            return image
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, image):
        if REGISTRY.get(image, mark_accessed=False) is None:
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
                io.Combo.Input(
                    "audio",
                    options=[],
                    upload=io.UploadType.audio,
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=audio",
                        refresh_button=True,
                    ),
                )
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        item = _get_media_item(audio, expected_kind="audio")
        waveform, sample_rate = _load_audio_from_bytes(item.data)
        return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})

    @classmethod
    def fingerprint_inputs(cls, audio):
        item = REGISTRY.get(audio, mark_accessed=False)
        if item is None:
            return audio
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, audio):
        if REGISTRY.get(audio, mark_accessed=False) is None:
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
                    options=[],
                    upload=io.UploadType.video,
                    remote=io.RemoteOptions(
                        route="/api/vlo-memory/options?kind=video",
                        refresh_button=True,
                    ),
                )
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        item = _get_media_item(file, expected_kind="video")
        return io.NodeOutput(InputImpl.VideoFromFile(stdlib_io.BytesIO(item.data)))

    @classmethod
    def fingerprint_inputs(cls, file):
        item = REGISTRY.get(file, mark_accessed=False)
        if item is None:
            return file
        return hashlib.sha256(item.data).hexdigest()

    @classmethod
    def validate_inputs(cls, file):
        if REGISTRY.get(file, mark_accessed=False) is None:
            return f"Invalid video id: {file}"
        return True


class VLOMemoryLoaderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VLOMemoryLoadImage,
            VLOMemoryLoadAudio,
            VLOMemoryLoadVideo,
        ]


async def comfy_entrypoint() -> VLOMemoryLoaderExtension:
    return VLOMemoryLoaderExtension()
