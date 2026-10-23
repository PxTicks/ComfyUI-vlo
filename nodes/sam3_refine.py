"""Mask-and-point corrections for SAM3 / SAM3.1, following Meta's interactive policy.

Core's `forward_segment` serves its own SAM3 Detect node well, but it is a poor fit
for correcting an existing selection: it reruns the vision backbone on every call,
returns only the upsampled high-resolution logits, and forces multimask on for a
single click with no fallback otherwise. Meta's SAM3 and SAM3.1 trackers instead
(sam3/model_builder.py, sam3/model/video_tracking_multiplex.py,
sam3/sam/mask_decoder.py):

- use multimask, keeping the best predicted IoU, for at most one point
  (`multimask_min_pt_num=0`, `multimask_max_pt_num=1`);
- otherwise keep the single-mask token unless its stability score is below 0.98,
  in which case they fall back to the best multimask output
  (`dynamic_multimask_via_stability`, delta 0.05);
- feed a later decoder pass the previous pass's *low-resolution* logits.

Core's port has no stability fallback, so this node reproduces it on top of
core's own prompt encoder and mask decoder. The image features are computed
once and shared by every decoder pass.

A saved selection is binary, so it is mapped onto +-10 logits, the scale Meta's
mask-as-output path uses. That is an approximation of the model's own logits,
not an equivalent: real logits carry less confidence near the edges.

The copied setup is only trusted against the core source it was verified with.
When that source changes, the node falls back to core's `forward_segment`, the
call core's SAM3 Detect makes, so ComfyUI keeps it working for its own node.
With one decoder pass, the fallback differs only in lacking the stability
fallback. The `pipeline` output reports which path ran.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import textwrap

import comfy.model_management
import comfy.utils
import torch
import torch.nn.functional as F
from comfy.ops import cast_to_input
from comfy_api.latest import io

from .mask_utils import _normalize_mask_frames

logger = logging.getLogger(__name__)


# SAM3 runs at a fixed square resolution; core's SAM3 Detect resizes to it the same way.
_SAM3_IMAGE_SIZE = 1008

# Meta's SAM3 / SAM3.1 tracker settings (sam3/model_builder.py).
_MULTIMASK_MAX_POINTS = 1
_STABILITY_DELTA = 0.05
_STABILITY_THRESHOLD = 0.98

# Binary masks become logits on the scale of Meta's mask-as-output path:
# 1 -> +10, 0 -> -10 (sigmoid(-10) ~ 4.5e-5).
_MASK_LOGIT_SCALE = 20.0
_MASK_LOGIT_BIAS = -10.0

_MAX_POINTS_PER_LABEL = 64

_SELECTION_MODES = ("reference", "single", "multimask")


# Verified against ComfyUI 0.41.0, commit 6607f36b53c7b695b160ac5b52d595e9e3752a2d,
# and 0.37.0 (4ef23c34d950), whose SAM3 sources are identical. Regenerate with
# `COMFYUI_PATH=... python tests/regen_fingerprint.py` after reviewing the upstream
# diff to comfy/ldm/sam3 and re-running tests/test_sam3_refine.py.
TESTED_COMFYUI_COMMIT = "6607f36b53c7b695b160ac5b52d595e9e3752a2d"
TESTED_COMFYUI_VERSION = "0.41.0"
TESTED_SOURCE_FINGERPRINT = "bed2c626c32bf94455a4e176a6bc658c0edb130f720428c1dce6c43061f3cbb4"

# (label, module, dotted path). The node copies `forward_segment`'s feature setup and
# relies on the heads' return order, multimask selection and mask-prompt resizing,
# none of which a symbol probe would notice changing.
FINGERPRINT_SOURCES = (
    ("SAM3Model.forward_segment", "detector", "SAM3Model.forward_segment"),
    ("forward_sam_heads", "tracker", "forward_sam_heads"),
    ("SAM3Tracker._forward_sam_heads", "tracker", "SAM3Tracker._forward_sam_heads"),
    ("SAM31Tracker._forward_sam_heads", "tracker", "SAM31Tracker._forward_sam_heads"),
    ("SAMMaskDecoder.forward", "tracker", "SAMMaskDecoder.forward"),
)

OVERRIDE_ENV = "VLO_SAM3_REFINE_ALLOW_UNVERIFIED"

INCOMPATIBLE = (
    "vlo SAM3 Refine Mask reuses ComfyUI's SAM3 decoder internals. The installed "
    "ComfyUI implementation is incompatible: {}"
)


_CORE_CHANGED_MODES = ("native", "stop")

# Pipelines reported on the node's `pipeline` output.
PIPELINE_VERIFIED = "verified"
PIPELINE_UNVERIFIED = "unverified"
PIPELINE_NATIVE = "native"

# The fallback calls core's `forward_segment`, as core's own SAM3 Detect does, so it
# only needs these keyword arguments to keep existing.
_NATIVE_SEGMENT_PARAMS = ("images", "point_inputs", "mask_inputs")

_warned: set = set()


def _warn_once(message: str, *args) -> None:
    key = message % args
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def sam3_modules() -> dict:
    """Core's SAM3 modules, imported lazily so the rest of the pack loads without them."""
    try:
        import comfy.ldm.sam3.detector as detector
        import comfy.ldm.sam3.tracker as tracker
    except ImportError as exc:  # pragma: no cover - depends on the ComfyUI build
        raise RuntimeError("comfy.ldm.sam3 is unavailable") from exc
    return {"detector": detector, "tracker": tracker}


def _resolve(module, dotted):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def core_source_fingerprint(modules=None) -> str:
    """Hash the core SAM3 source this node depends on, ignoring blank lines and trailing space."""
    modules = modules or sam3_modules()
    digest = hashlib.sha256()
    for label, module_name, dotted in FINGERPRINT_SOURCES:
        try:
            source = textwrap.dedent(inspect.getsource(_resolve(modules[module_name], dotted)))
        except (AttributeError, OSError, TypeError) as exc:
            raise RuntimeError("cannot read the source of {}: {}".format(label, exc)) from exc
        body = "\n".join(line.rstrip() for line in source.split("\n") if line.strip())
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(body.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def core_mismatch() -> str | None:
    """None when the installed SAM3 source is the verified one, otherwise why it is not."""
    try:
        found = core_source_fingerprint()
    except RuntimeError as exc:
        return str(exc)
    if found == TESTED_SOURCE_FINGERPRINT:
        return None
    return (
        "comfy/ldm/sam3 has changed since this node was verified. Expected source "
        "fingerprint {} (ComfyUI {}, commit {}), found {}. Review the upstream diff, "
        "re-run tests/test_sam3_refine.py, then update TESTED_SOURCE_FINGERPRINT."
    ).format(TESTED_SOURCE_FINGERPRINT, TESTED_COMFYUI_VERSION, TESTED_COMFYUI_COMMIT[:12], found)


def _require_native_segment(sam, detail: str) -> None:
    try:
        params = inspect.signature(sam.forward_segment).parameters
    except (AttributeError, TypeError, ValueError):
        params = {}
    missing = [name for name in _NATIVE_SEGMENT_PARAMS if name not in params]
    if missing:
        raise RuntimeError(INCOMPATIBLE.format(
            "{} The native fallback cannot run either: forward_segment no longer takes {}."
            .format(detail, ", ".join(missing))))


def select_pipeline(sam, if_core_changed: str) -> str:
    """Pick the verified copy of core's decoder setup, or what to do when core has moved."""
    detail = core_mismatch()
    if detail is None:
        return PIPELINE_VERIFIED
    if os.environ.get(OVERRIDE_ENV):
        _warn_once("SAM3 refine: %s Running the copied decoder setup anyway, at your own "
                   "risk, because %s is set.", detail, OVERRIDE_ENV)
        return PIPELINE_UNVERIFIED
    if if_core_changed == "native":
        _require_native_segment(sam, detail)
        _warn_once("SAM3 refine: %s Falling back to core's forward_segment: no stability "
                   "fallback, and the backbone reruns on every pass.", detail)
        return PIPELINE_NATIVE
    raise RuntimeError(INCOMPATIBLE.format(
        "{} Set if_core_changed to 'native' to use core's forward_segment instead, or set "
        "{}=1 to run the copied setup anyway at your own risk.".format(detail, OVERRIDE_ENV)))


def _parse_points(raw: str, name: str, width: int, height: int) -> list[tuple[float, float]]:
    try:
        points = json.loads(raw) if raw and raw.strip() else []
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON list of {{\"x\", \"y\"}} points.") from exc
    if not isinstance(points, list):
        raise ValueError(f"{name} must be a JSON list of {{\"x\", \"y\"}} points.")
    if len(points) > _MAX_POINTS_PER_LABEL:
        raise ValueError(f"{name} has {len(points)} points; the limit is {_MAX_POINTS_PER_LABEL}.")
    parsed = []
    for point in points:
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a JSON list of {{\"x\", \"y\"}} points.") from exc
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError(f"{name} has a point outside the {width}x{height} image: ({x}, {y}).")
        parsed.append((x, y))
    return parsed


def _stability_scores(logits: torch.Tensor) -> torch.Tensor:
    """Meta's mask stability: area above +delta over area above -delta, per mask."""
    flat = logits.flatten(-2)
    area_i = torch.sum(flat > _STABILITY_DELTA, dim=-1).float()
    area_u = torch.sum(flat > -_STABILITY_DELTA, dim=-1).float()
    return torch.where(area_u > 0, area_i / area_u, torch.ones_like(area_u))


def _interactive_features(sam, frame: torch.Tensor):
    """Backbone features for the interactive decoder, as core's `forward_segment` builds them."""
    backbone = sam.detector.backbone["vision_backbone"]
    if backbone.multiplex:
        _, _, features, _ = backbone(frame, tracker_mode="interactive")
    else:
        _, _, features, _ = backbone(frame, need_tracker=True)
        if sam.detector.scalp > 0:
            features = features[:-sam.detector.scalp]

    high_res = list(features[:-1])
    image_features = features[-1]
    batch, channels, height, width = image_features.shape
    # The init-frame path: no memory has been encoded yet, so add the no-memory embedding.
    no_mem = getattr(sam.tracker, "interactivity_no_mem_embed", None)
    if no_mem is None:
        no_mem = getattr(sam.tracker, "no_mem_embed", None)
    if no_mem is not None:
        flat = image_features.flatten(2).permute(0, 2, 1)
        flat = flat + cast_to_input(no_mem, flat)
        image_features = flat.view(batch, height, width, channels).permute(0, 3, 1, 2)
    return image_features, high_res


def _decode(tracker, image_features, high_res, points, mask_logits, selection: str) -> torch.Tensor:
    """One decoder pass. Returns `[1, 1, h, w]` low-resolution logits."""
    def heads(multimask: bool) -> torch.Tensor:
        low_res, _, _, _ = tracker._forward_sam_heads(
            backbone_features=image_features, point_inputs=points, mask_inputs=mask_logits,
            box_inputs=None, high_res_features=high_res, multimask_output=multimask)
        return low_res

    num_points = points["point_labels"].shape[1]
    if selection == "multimask" or (selection == "reference" and num_points <= _MULTIMASK_MAX_POINTS):
        return heads(True)
    single = heads(False)
    if selection == "reference" and bool(_stability_scores(single).min() < _STABILITY_THRESHOLD):
        # Core gates a no-object prediction to a constant before this point, which scores
        # as stable, so that case keeps the single mask; Meta's fallback would gate it too.
        return heads(True)
    return single


class vloSAM3RefineMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloSAM3RefineMask",
            search_aliases=["vlo sam3 refine", "vlo sam3 points", "vlo correct mask"],
            display_name="vlo SAM3 Refine Mask",
            category="image/detection",
            description=(
                "Segments one image from include/exclude points with SAM3 or SAM3.1, "
                "optionally conditioned on an existing mask so the points correct it. "
                "Follows Meta's interactive decoding: multimask for a single point, a "
                "stability fallback otherwise, and low-resolution logits between passes. "
                "The vision backbone runs once however many passes are made."
            ),
            inputs=[
                io.Model.Input("model", tooltip="A SAM3 or SAM3.1 checkpoint."),
                io.Image.Input("image", tooltip="One image."),
                io.String.Input(
                    "positive_coords",
                    default="[]",
                    tooltip='Include points as JSON [{"x": int, "y": int}, ...] in image pixels.',
                ),
                io.String.Input(
                    "negative_coords",
                    default="[]",
                    tooltip='Exclude points as JSON [{"x": int, "y": int}, ...] in image pixels.',
                ),
                io.Combo.Input(
                    "mask_selection",
                    options=list(_SELECTION_MODES),
                    default="reference",
                    tooltip=(
                        "'reference' follows Meta's SAM3 policy: the best of three masks "
                        "for one point, otherwise the single mask unless it is unstable. "
                        "'single' and 'multimask' force one decoder output for comparisons."
                    ),
                ),
                io.Int.Input(
                    "refine_iterations",
                    default=1,
                    min=1,
                    max=5,
                    tooltip=(
                        "Decoder passes with the same points, each fed the last pass's "
                        "logits. Meta makes one pass per interaction."
                    ),
                ),
                io.Combo.Input(
                    "if_core_changed",
                    options=list(_CORE_CHANGED_MODES),
                    default="native",
                    tooltip=(
                        "What to do when ComfyUI's SAM3 code differs from the version this "
                        "node was verified against. 'native' falls back to core's own "
                        "forward_segment, which has no stability fallback and ignores "
                        "mask_selection. 'stop' raises an error instead."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Existing selection to correct, the same size as the image. "
                        "Without it the points select a new object."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output("mask"),
                io.String.Output(
                    "pipeline",
                    tooltip=(
                        "'verified', 'native' when core changed and the node fell back, or "
                        "'unverified' when the override ran the copied setup anyway."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, image, positive_coords="[]", negative_coords="[]",
                mask_selection="reference", refine_iterations=1, if_core_changed="native",
                mask=None) -> io.NodeOutput:
        if mask_selection not in _SELECTION_MODES:
            raise ValueError(f"mask_selection must be one of {', '.join(_SELECTION_MODES)}.")
        if if_core_changed not in _CORE_CHANGED_MODES:
            raise ValueError(f"if_core_changed must be one of {', '.join(_CORE_CHANGED_MODES)}.")
        if image.shape[0] != 1:
            raise ValueError(f"vlo SAM3 Refine Mask takes one image, got a batch of {image.shape[0]}.")
        _, height, width, _ = image.shape

        prior = None
        if mask is not None:
            prior = _normalize_mask_frames(mask)
            if tuple(prior.shape) != (1, height, width):
                raise ValueError(
                    f"mask must be one {width}x{height} mask matching the image, "
                    f"got {tuple(prior.shape)}.")

        positive = _parse_points(positive_coords, "positive_coords", width, height)
        negative = _parse_points(negative_coords, "negative_coords", width, height)
        if prior is None and not positive:
            raise ValueError("Add at least one include point, or connect a mask to correct.")
        if not positive and not negative:
            raise ValueError("Add at least one include or exclude point.")

        sam = getattr(model.model, "diffusion_model", None)
        if not (hasattr(sam, "detector") and hasattr(sam, "tracker")):
            raise ValueError("vlo SAM3 Refine Mask needs a SAM3 or SAM3.1 checkpoint.")
        pipeline = select_pipeline(sam, if_core_changed)

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()

        frame = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1), _SAM3_IMAGE_SIZE, _SAM3_IMAGE_SIZE, "bilinear",
            crop="disabled").to(device=device, dtype=dtype)
        coords = [[x / width * _SAM3_IMAGE_SIZE, y / height * _SAM3_IMAGE_SIZE]
                  for x, y in positive + negative]
        points = {
            "point_coords": torch.tensor([coords], device=device, dtype=dtype),
            "point_labels": torch.tensor([[1] * len(positive) + [0] * len(negative)],
                                         device=device, dtype=torch.int32),
        }
        # Core's heads resize a full-size mask prompt to the decoder's prompt size.
        logits = None
        if prior is not None:
            logits = (prior.unsqueeze(1) * _MASK_LOGIT_SCALE + _MASK_LOGIT_BIAS).to(
                device=device, dtype=dtype)

        if pipeline == PIPELINE_NATIVE:
            if mask_selection != "reference":
                _warn_once("SAM3 refine: mask_selection '%s' is ignored by the native "
                           "fallback; core's forward_segment chooses the output itself.",
                           mask_selection)
            # Core's SAM3 Detect loop: high-resolution logits, resized again as the next prompt.
            for _ in range(int(refine_iterations)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                logits = sam.forward_segment(frame, point_inputs=points, mask_inputs=logits)
        else:
            image_features, high_res = _interactive_features(sam, frame)
            for _ in range(int(refine_iterations)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                logits = _decode(sam.tracker, image_features, high_res, points, logits, mask_selection)

        out = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
        selection = (out[:, 0] > 0).float().to(comfy.model_management.intermediate_device())
        return io.NodeOutput(selection, pipeline)
