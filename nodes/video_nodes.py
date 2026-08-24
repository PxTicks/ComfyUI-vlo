"""Video fps conversion and the websocket image/video output nodes."""

from __future__ import annotations

import io as stdlib_io
import logging
from typing import Any

import folder_paths

from comfy_api.latest import Input, Types, io
from protocol import BinaryEventTypes

from .media_io import _resample_video_frames_to_fps
from .registry import REGISTRY, _build_memory_output_item
from .ws import (
    _build_saved_video_metadata,
    _encode_payload_with_metadata,
    _get_client_id,
    _get_execution_ids,
    _resolve_video_content_type,
    _send_binary_event,
    _send_progress_update,
    _tensor_to_pil_rgb_image,
)

logger = logging.getLogger(__name__)


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
