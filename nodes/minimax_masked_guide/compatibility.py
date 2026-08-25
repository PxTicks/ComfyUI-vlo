"""Structural checks against the installed ComfyUI MiniMax H3 implementation.

This package forks a copy of `MiniMaxH3Model._forward` and relies on internals
that only exist in recent ComfyUI: arbitrary-frame image guides (PR #15439) and
per-token modulation-row indices (PR #15375). Neither is version-stamped, so the
compatibility gate probes for the behaviour instead of a version number, and
fails loudly rather than falling back to something subtly wrong.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import textwrap

import torch


# Forked against ComfyUI 0.33.0, commit 5f0c4e18cb7e98f0e7c46c2c7ce928d641351e67
# (comfy/ldm/minimax/model.py). Re-verify `tests/test_forward_equivalence.py`
# after updating ComfyUI.
TESTED_COMFYUI_COMMIT = "5f0c4e18cb7e98f0e7c46c2c7ce928d641351e67"
TESTED_COMFYUI_VERSION = "0.33.0"

# sha256 over the source of every core function the fork copies or depends on the
# exact behaviour of. Symbol probes cannot catch an upstream edit *inside*
# `_forward` -- the fork would keep running its stale copy and quietly diverge --
# so the source itself is the version check. Regenerate with
# `COMFYUI_PATH=... python tests/regen_fingerprint.py` after reviewing
# the upstream diff and re-running the equivalence tests.
TESTED_SOURCE_FINGERPRINT = "fa56c10a6bc313ee1663d9db2d1dd7bceea57620726611b44c581ba46fbc850f"

# (label, dotted path from the H3 module) -- label is hashed too, so reordering
# or renaming is itself a change.
FINGERPRINT_SOURCES = (
    ("MiniMaxH3Model._forward", "MiniMaxH3Model._forward"),
    ("MiniMaxH3Model._cond_video_rows", "MiniMaxH3Model._cond_video_rows"),
    ("PackedLayout.__init__", "PackedLayout.__init__"),
    ("patchify_video", "patchify_video"),
    ("_mod_row", "_mod_row"),
    ("_mod_scale_shift", "_mod_scale_shift"),
    ("_mod_gate", "_mod_gate"),
)

# Escape hatch for someone who has read the upstream diff and wants to try anyway.
OVERRIDE_ENV = "VLO_MASKED_GUIDE_ALLOW_UNVERIFIED"

INCOMPATIBLE = (
    "This experimental masked-guide patch targets ComfyUI MiniMax-H3 with "
    "arbitrary-frame image guides (PR #15439) and per-token timestep support "
    "(PR #15375). The installed ComfyUI implementation is incompatible: {}"
)

_REQUIRED_MODULE_ATTRS = (
    "MiniMaxH3Model", "PackedLayout", "patchify_video", "unpatchify_video",
    "pack_audio", "unpack_audio", "rope_rotation_table", "time_shift_sigma",
    "mask_row_values", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP",
)

# patch_size is set per instance, so it is checked in is_h3_diffusion_model instead
_REQUIRED_MODEL_ATTRS = ("_forward", "_cond_video_rows", "_cond_audio_rows")


def h3_module():
    """The core MiniMax H3 module, imported lazily so the rest of the pack loads without it."""
    try:
        import comfy.ldm.minimax.model as module
    except ImportError as exc:  # pragma: no cover - depends on the ComfyUI build
        raise RuntimeError(INCOMPATIBLE.format("comfy.ldm.minimax.model is unavailable")) from exc
    return module


def _resolve(module, dotted):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def core_source_fingerprint(module=None) -> str:
    """Hash the core H3 source the fork is pinned to, ignoring blank lines and trailing space."""
    module = module or h3_module()
    digest = hashlib.sha256()
    for label, dotted in FINGERPRINT_SOURCES:
        try:
            source = textwrap.dedent(inspect.getsource(_resolve(module, dotted)))
        except (AttributeError, OSError, TypeError) as exc:
            raise RuntimeError(INCOMPATIBLE.format(
                "cannot read the source of {}: {}".format(label, exc))) from exc
        body = "\n".join(line.rstrip() for line in source.split("\n") if line.strip())
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(body.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _check_source_fingerprint(module) -> None:
    found = core_source_fingerprint(module)
    if found == TESTED_SOURCE_FINGERPRINT:
        return
    detail = (
        "comfy/ldm/minimax/model.py has changed since this fork was taken. Expected source "
        "fingerprint {} (ComfyUI {}, commit {}), found {}. The fork would run a stale copy of "
        "_forward against a newer core. Review the upstream diff, re-run "
        "tests/test_masked_guide_forward_equivalence.py, then update "
        "TESTED_SOURCE_FINGERPRINT. Set {}=1 to run anyway at your own risk."
    ).format(TESTED_SOURCE_FINGERPRINT, TESTED_COMFYUI_VERSION, TESTED_COMFYUI_COMMIT[:12],
             found, OVERRIDE_ENV)
    if os.environ.get(OVERRIDE_ENV):
        logging.warning("MiniMax H3 masked guides: %s (continuing because %s is set)",
                        detail, OVERRIDE_ENV)
        return
    raise RuntimeError(INCOMPATIBLE.format(detail))


def _probe_wrapper_chain() -> None:
    """The wrapper rebuilds the executor chain, so those attributes have to exist."""
    import comfy.patcher_extension as ext

    executor = ext.WrapperExecutor.new_class_executor(lambda: None, object(), [])
    missing = [name for name in ("wrappers", "idx", "class_obj") if not hasattr(executor, name)]
    if missing or not hasattr(ext.WrapperExecutor, "new_class_executor"):
        raise RuntimeError(INCOMPATIBLE.format(
            "comfy.patcher_extension.WrapperExecutor is missing " + ", ".join(missing or ["new_class_executor"])))


def _probe_per_token_modulation(module) -> None:
    """PR #15375 made `_mod_row` accept a per-token index tensor. Prove it does."""
    mod_row = getattr(module, "_mod_row", None)
    if mod_row is None:
        raise RuntimeError(INCOMPATIBLE.format("comfy.ldm.minimax.model._mod_row is missing"))
    table = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    rows = torch.tensor([2, 0], dtype=torch.long)
    try:
        out = mod_row(table, rows, torch.float32)
    except Exception as exc:
        raise RuntimeError(INCOMPATIBLE.format("_mod_row rejects per-token indices: {}".format(exc))) from exc
    if tuple(out.shape) != (2, 3) or not torch.equal(out, table[rows]):
        raise RuntimeError(INCOMPATIBLE.format("_mod_row does not gather per-token modulation rows"))


def _probe_keyframe_layout(module) -> None:
    """PR #15439 gives keyframes a `resolved_frame_index` and their own `cond` rows."""
    latent = torch.zeros(1, 24, 1, 4, 6)
    try:
        layout = module.PackedLayout(3, 2, 4, 6, 5,
                                     keyframes=[{"resolved_frame_index": 7, "latent": latent}])
    except Exception as exc:
        raise RuntimeError(INCOMPATIBLE.format("PackedLayout rejects frame-anchored keyframes: {}".format(exc))) from exc
    cond = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond) != 1 or cond[0][1] - cond[0][0] != 2 * 3:
        raise RuntimeError(INCOMPATIBLE.format("PackedLayout does not emit one cond row per 2x2 guide patch"))


def check_core_compatible() -> None:
    """Raise unless the installed H3 implementation matches what the fork assumes."""
    module = h3_module()
    missing = [name for name in _REQUIRED_MODULE_ATTRS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(INCOMPATIBLE.format("comfy.ldm.minimax.model is missing " + ", ".join(missing)))
    missing = [name for name in _REQUIRED_MODEL_ATTRS if not hasattr(module.MiniMaxH3Model, name)]
    if missing:
        raise RuntimeError(INCOMPATIBLE.format("MiniMaxH3Model is missing " + ", ".join(missing)))
    _probe_per_token_modulation(module)
    _probe_keyframe_layout(module)
    _probe_wrapper_chain()
    _check_source_fingerprint(module)


def is_h3_diffusion_model(model) -> bool:
    """True when `model` is the core H3 DiT with the patch geometry the fork assumes."""
    try:
        module = h3_module()
    except RuntimeError:
        return False
    return isinstance(model, module.MiniMaxH3Model) and tuple(model.patch_size) == (1, 2, 2)
