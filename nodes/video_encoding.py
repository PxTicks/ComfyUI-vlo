"""Stable, local video-encoding configuration for vlo output nodes.

ComfyUI exposes the container and codec enums publicly, but its encoder
configuration helpers are implementation details. Keep this small layer local
so an internal ComfyUI module move cannot prevent the vlo node pack loading.
"""

from __future__ import annotations

import os
from typing import Protocol

from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
from comfy_api.latest import Types

VIDEO_ENCODERS: dict[Types.VideoCodec, str] = {
    Types.VideoCodec.H264: "h264",
    Types.VideoCodec.AV1: "libsvtav1",
}
VIDEO_CONTAINER_FORMATS: dict[Types.VideoContainer, str] = {
    Types.VideoContainer.MP4: "mp4",
    Types.VideoContainer.MKV: "matroska",
    Types.VideoContainer.WEBM: "webm",
}
BT2020_NCL = 9
BT709_NCL = 1
HDR_COLOR_TRANSFERS = {
    "HDR": ColorTrc.ARIB_STD_B67,
    "HDR PQ": ColorTrc.SMPTE2084,
}
VIDEO_COLOR_TRANSFERS = {
    "sRGB": ColorTrc.IEC61966_2_1,
    **HDR_COLOR_TRANSFERS,
}


class _ColorPropertiesTarget(Protocol):
    color_primaries: ColorPrimaries
    color_trc: ColorTrc
    colorspace: int
    color_range: ColorRange


def video_output_config(
    path: str | os.PathLike[str],
    container: Types.VideoContainer | str,
    codec: Types.VideoCodec | str,
) -> tuple[dict[str, object], Types.VideoContainer, Types.VideoCodec]:
    """Resolve the public video choices into PyAV output configuration."""
    container = Types.VideoContainer(container)
    codec = Types.VideoCodec(codec)

    if container == Types.VideoContainer.AUTO:
        extension = os.path.splitext(os.fspath(path))[1].lower()
        container = {
            ".mkv": Types.VideoContainer.MKV,
            ".webm": Types.VideoContainer.WEBM,
        }.get(extension, Types.VideoContainer.MP4)
    if codec == Types.VideoCodec.AUTO:
        codec = (
            Types.VideoCodec.AV1
            if container == Types.VideoContainer.WEBM
            else Types.VideoCodec.H264
        )
    if container == Types.VideoContainer.WEBM and codec != Types.VideoCodec.AV1:
        raise ValueError("WebM output requires the AV1 codec")

    open_kwargs: dict[str, object] = {
        "mode": "w",
        "format": VIDEO_CONTAINER_FORMATS[container],
    }
    if container == Types.VideoContainer.MP4:
        open_kwargs["options"] = {"movflags": "use_metadata_tags+faststart"}
    return open_kwargs, container, codec


def video_encoder_options(
    codec: Types.VideoCodec,
    crf: float | None,
) -> dict[str, str]:
    """Build the encoder options used by ComfyUI's tensor-video path."""
    if crf is None:
        return {}
    if codec == Types.VideoCodec.AV1 and crf == 0:
        return {"svtav1-params": "lossless=1"}
    return {"crf": str(crf)}


def set_video_color_properties(
    target: _ColorPropertiesTarget,
    color_space: str,
) -> None:
    """Tag a PyAV frame or codec context with its output color properties."""
    is_hdr = color_space in HDR_COLOR_TRANSFERS
    target.color_primaries = ColorPrimaries.BT2020 if is_hdr else ColorPrimaries.BT709
    target.color_trc = VIDEO_COLOR_TRANSFERS[color_space]
    target.colorspace = BT2020_NCL if is_hdr else BT709_NCL
    target.color_range = ColorRange.MPEG
