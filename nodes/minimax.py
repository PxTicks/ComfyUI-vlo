"""Batch adapter around ComfyUI's native MiniMax H3 reference-to-video node."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import torch

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .media_io import _resample_frame_tensor_to_fps


_COMFY_MAX_RESOLUTION = 16384


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
