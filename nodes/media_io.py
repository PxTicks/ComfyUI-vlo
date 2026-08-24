"""Decoding media into tensors, and resampling frame tensors to a target fps."""

from __future__ import annotations

import io as stdlib_io
import math
import re
from fractions import Fraction

import av
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy.model_management
from comfy_api.latest import Input, InputImpl, Types


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
