"""The vloMemoryLoad* nodes: single and batch image, audio and video loaders."""

from __future__ import annotations

import hashlib
import io as stdlib_io
from typing import Any

import folder_paths
import torch

from comfy_api.latest import Input, InputImpl, io

from ..batch_loader_utils import (
    normalize_memory_batch_flags,
    normalize_memory_batch_values,
)
from .batch_inputs import (
    _fingerprint_memory_batch_values,
    _memory_batch_input,
    _validate_memory_batch_values,
)
from .media_io import (
    _load_audio_from_bytes,
    _load_audio_from_filepath,
    _load_image_from_bytes,
    _load_image_from_filepath,
)
from .registry import (
    REGISTRY,
    _fingerprint_annotated_filepath,
    _get_media_item,
    _list_input_files,
    _normalize_media_id,
    _should_load_from_filepath,
)


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
