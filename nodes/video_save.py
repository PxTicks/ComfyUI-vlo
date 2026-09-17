"""Save an IMAGE batch as video, resizing each frame inside the encode loop.

The encode loop mirrors ComfyUI's `VideoFromComponents.save_to`. The RGB -> YUV
`reformat` also scales: FFmpeg's swscale resizes each frame while it converts,
so a resized batch is never built. This is also far faster than resizing
tensors per frame (~3 ms vs ~350 ms per 4K frame on CPU).

Container, codec and color configuration lives in a small local compatibility
layer. It uses ComfyUI's public enums without making the node pack depend on
private `_input_impl` modules.
"""

from __future__ import annotations

import json
import logging
import math
import os
from fractions import Fraction
from typing import Any, Optional

import av
import numpy as np
from av.video.reformatter import Interpolation

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import Input, InputImpl, Types, io, ui

from .video_encoding import (
    BT2020_NCL,
    BT709_NCL,
    HDR_COLOR_TRANSFERS,
    VIDEO_COLOR_TRANSFERS,
    VIDEO_ENCODERS,
    set_video_color_properties,
    video_encoder_options,
    video_output_config,
)
from .ws import _build_saved_video_metadata

logger = logging.getLogger(__name__)

# Same method names and order as ComfyUI's ImageScale, backed by swscale.
SCALE_METHODS: dict[str, Interpolation] = {
    "nearest-exact": Interpolation.POINT,
    "bilinear": Interpolation.BILINEAR,
    "area": Interpolation.AREA,
    "bicubic": Interpolation.BICUBIC,
    "lanczos": Interpolation.LANCZOS,
}
CROP_METHODS = ["disabled", "center"]
MAX_H264_CRF = 51.0


def _round_to_even(value: float) -> int:
    return max(2, 2 * round(value / 2))


def resolve_output_size(
    source_width: int, source_height: int, width: int, height: int
) -> tuple[int, int]:
    """Resolve the output size the way ImageScale does: 0 keeps that side
    proportional, and 0x0 keeps the source size. A derived side is rounded to
    an even number because YUV 4:2:0 needs even dimensions. A side you set
    explicitly must already be even."""
    if width == 0 and height == 0:
        width, height = source_width, source_height
    elif width == 0:
        width = _round_to_even(source_width * height / source_height)
    elif height == 0:
        height = _round_to_even(source_height * width / source_width)

    if width % 2 or height % 2:
        raise ValueError(
            f"Video output size {width}x{height} must have even dimensions "
            "(yuv420 chroma subsampling). Pick even width/height, or set one to 0 "
            "to derive it from the aspect ratio."
        )
    return width, height


def center_crop_box(
    source_width: int, source_height: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """(x, y, crop_width, crop_height) of the centered source region matching
    the output aspect, using `comfy.utils.common_upscale`'s formula."""
    old_aspect = source_width / source_height
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((source_width - source_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((source_height - source_height * (old_aspect / new_aspect)) / 2)
    return x, y, source_width - x * 2, source_height - y * 2


def resolve_container_and_codec(
    format: str, codec: str
) -> tuple[Types.VideoContainer, Types.VideoCodec]:
    """Save Video's rule: auto format is WebM for AV1, otherwise MP4. The
    native helper resolves an auto codec and rejects WebM without AV1."""
    if format == Types.VideoContainer.AUTO:
        format = Types.VideoContainer.WEBM if codec == Types.VideoCodec.AV1 else Types.VideoContainer.MP4
    _, container, video_codec = video_output_config(
        "", Types.VideoContainer(format), Types.VideoCodec(codec)
    )
    return container, video_codec


def encode_resized_video(
    path: str,
    images: Input.Image,
    *,
    frame_rate: Fraction,
    width: int,
    height: int,
    scale_method: str = "bicubic",
    crop: str = "disabled",
    format: Types.VideoContainer = Types.VideoContainer.MP4,
    codec: Types.VideoCodec = Types.VideoCodec.H264,
    crf: float | None = None,
    bit_depth: int = 8,
    color_space: str = "sRGB",
    audio: Optional[Input.Audio] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """`VideoFromComponents.save_to` with scaling done in the per-frame reformat."""
    if color_space not in VIDEO_COLOR_TRANSFERS:
        raise ValueError(f"Unsupported video color space: {color_space}")
    open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
    is_10bit = bit_depth >= 10
    pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
    dst_colorspace = (
        BT2020_NCL
        if color_space in HDR_COLOR_TRANSFERS
        else BT709_NCL
    )

    frame_count, source_height, source_width = (int(d) for d in images.shape[:3])
    if crop == "center":
        crop_x, crop_y, crop_width, crop_height = center_crop_box(source_width, source_height, width, height)
    else:
        crop_x, crop_y, crop_width, crop_height = 0, 0, source_width, source_height
    # The filter also drives chroma subsampling, so leave swscale's default when
    # nothing is scaled: an unscaled save then matches native Save Video exactly.
    scaled = (crop_width, crop_height) != (width, height)
    interpolation = SCALE_METHODS[scale_method] if scaled else None

    progress = comfy.utils.ProgressBar(frame_count)
    with av.open(path, **open_kwargs) as output:
        if metadata is not None:
            for key, value in metadata.items():
                output.metadata[key] = json.dumps(value)

        frame_rate = Fraction(round(frame_rate * 1000), 1000)
        video_stream = output.add_stream(VIDEO_ENCODERS[output_codec], rate=frame_rate)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = pix_fmt
        video_stream.options = video_encoder_options(output_codec, crf)
        set_video_color_properties(video_stream.codec_context, color_space)

        audio_sample_rate = 1
        audio_resampler = None
        audio_stream: Optional[av.AudioStream] = None
        if audio:
            source_audio_sample_rate = int(audio["sample_rate"])
            audio_sample_rate = 48000 if output_format == Types.VideoContainer.WEBM else source_audio_sample_rate
            waveform = audio["waveform"]
            waveform = waveform[0, :, :math.ceil((source_audio_sample_rate / frame_rate) * frame_count)]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
            audio_codec = "libopus" if output_format == Types.VideoContainer.WEBM else "aac"
            audio_stream = output.add_stream(audio_codec, rate=audio_sample_rate, layout=layout)
            if audio_sample_rate != source_audio_sample_rate:
                audio_resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=audio_sample_rate)

        for index in range(frame_count):
            comfy.model_management.throw_exception_if_processing_interrupted()
            # A view: only this frame's crop is converted and copied.
            frame = images[index, crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :3]
            if is_10bit or scaled:
                # Preserve float detail until swscale has resized the frame. The
                # unscaled 8-bit path stays byte-identical to native Save Video.
                img = (frame.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)
                av_frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
            else:
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy()
                av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            av_frame = av_frame.reformat(
                width=width,
                height=height,
                format=pix_fmt,
                dst_colorspace=dst_colorspace,
                interpolation=interpolation,
            )
            set_video_color_properties(av_frame, color_space)
            output.mux(video_stream.encode(av_frame))
            progress.update(1)

        output.mux(video_stream.encode(None))

        if audio_stream is not None:
            audio_frame = av.AudioFrame.from_ndarray(
                waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout
            )
            audio_frame.sample_rate = source_audio_sample_rate
            audio_frame.pts = 0
            frames = [audio_frame] if audio_resampler is None else audio_resampler.resample(audio_frame)
            for resampled in frames:
                output.mux(audio_stream.encode(resampled))
            if audio_resampler is not None:
                for resampled in audio_resampler.resample(None):
                    output.mux(audio_stream.encode(resampled))
            output.mux(audio_stream.encode(None))


class vloSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloSaveVideo",
            search_aliases=[
                "save video",
                "save video resize",
                "upscale video save",
                "images to video resize",
            ],
            display_name="vlo Save Video",
            category="image/video",
            description=(
                "Encodes an image batch to video, resizing each frame during encoding "
                "so no resized batch is held in memory. Uses native Save Video "
                "encoding. With save_output off, the video goes to the temp "
                "folder as a preview."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames to encode."),
                io.Float.Input("fps", default=30.0, min=0.01, max=1000.0, step=0.01),
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="The prefix for the file to save. Formatting tokens follow the same rules as Save Video.",
                ),
                io.Boolean.Input(
                    "save_output",
                    default=True,
                    tooltip=(
                        "On: save to the output folder. Off: encode to the temp folder "
                        "for preview only."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    tooltip="Output width. 0 keeps the aspect ratio (or the source size when height is also 0). Must be even.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    tooltip="Output height. 0 keeps the aspect ratio (or the source size when width is also 0). Must be even.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=list(SCALE_METHODS),
                    default="bicubic",
                    tooltip="Scaling filter, applied by FFmpeg (swscale) during encoding.",
                ),
                io.Combo.Input(
                    "crop",
                    options=CROP_METHODS,
                    default="disabled",
                    tooltip="How to handle an aspect ratio mismatch: 'disabled' stretches, 'center' crops to keep the aspect ratio.",
                ),
                io.Combo.Input(
                    "format",
                    options=Types.VideoContainer.as_input(),
                    default="auto",
                    tooltip="The output container. Auto uses WebM for AV1 and MP4 otherwise.",
                ),
                io.Combo.Input(
                    "codec",
                    options=Types.VideoCodec.as_input(),
                    default="auto",
                    tooltip="The output codec. Auto uses AV1 for WebM and H.264 otherwise.",
                ),
                io.Float.Input(
                    "crf",
                    default=23.0,
                    min=0.0,
                    max=63.0,
                    step=1.0,
                    tooltip="Lower values produce higher quality and larger files. H.264 accepts 0-51, AV1 0-63.",
                ),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
                io.Combo.Input(
                    "bit_depth",
                    options=["auto", 8, 10],
                    default="auto",
                    optional=True,
                    tooltip="Auto uses 8-bit for sRGB and 10-bit for HDR.",
                ),
                io.Combo.Input(
                    "color_space",
                    options=["sRGB", "HDR", "HDR PQ"],
                    default="sRGB",
                    optional=True,
                    tooltip="Color space of the input images. HDR selects BT.2020/HLG and HDR PQ selects BT.2020/PQ.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video", tooltip="The encoded video, read back from the saved file.")],
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        fps: float,
        filename_prefix: str,
        save_output: bool,
        width: int,
        height: int,
        upscale_method: str,
        crop: str,
        format: str,
        codec: str,
        crf: float,
        audio: Optional[Input.Audio] = None,
        bit_depth: int | str = "auto",
        color_space: str = "sRGB",
    ) -> io.NodeOutput:
        if images.shape[0] == 0:
            raise ValueError("vlo Save Video received an empty image batch.")
        source_height, source_width = int(images.shape[1]), int(images.shape[2])
        out_width, out_height = resolve_output_size(source_width, source_height, width, height)
        container, video_codec = resolve_container_and_codec(format, codec)
        if video_codec == Types.VideoCodec.H264 and crf > MAX_H264_CRF:
            raise ValueError(f"H.264 crf must be between 0 and {MAX_H264_CRF:g}, got {crf:g}.")
        if bit_depth == "auto":
            bit_depth = 10 if color_space in HDR_COLOR_TRANSFERS else 8

        folder_type = io.FolderType.output if save_output else io.FolderType.temp
        base_directory = (
            folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, base_directory, out_width, out_height
        )
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(container)}"
        path = os.path.join(full_output_folder, file)

        try:
            encode_resized_video(
                path,
                images,
                frame_rate=Fraction(fps),
                width=out_width,
                height=out_height,
                scale_method=upscale_method,
                crop=crop,
                format=container,
                codec=video_codec,
                crf=crf,
                bit_depth=int(bit_depth),
                color_space=color_space,
                audio=audio,
                metadata=_build_saved_video_metadata(cls),
            )
        except BaseException:
            # Don't leave a truncated file behind after an interrupt or encoder error.
            if os.path.exists(path):
                os.remove(path)
            raise

        logger.info(
            "Saved resized video %s (%dx%d -> %dx%d, %d frames, %s)",
            path,
            source_width,
            source_height,
            out_width,
            out_height,
            int(images.shape[0]),
            folder_type.value,
        )
        return io.NodeOutput(
            InputImpl.VideoFromFile(path),
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, folder_type)]),
        )
