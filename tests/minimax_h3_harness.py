"""Shared setup for the MiniMax H3 masked-guide tests.

The tests exercise a real (tiny, randomly weighted) `MiniMaxH3Model` on CPU so
that the forked forward pass can be compared against core's, byte for byte.
That needs two pieces of scaffolding: ComfyUI on `sys.path`, and stand-ins for
the comfy-kitchen fused ops when the installed comfy-kitchen predates them. The
stand-ins are shared by both code paths, so equivalence results do not depend on
their numerical fidelity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


def _shim_comfy_kitchen():
    try:
        import comfy_kitchen
    except ImportError:
        return

    if not hasattr(comfy_kitchen, "int8_attention_is_available"):
        comfy_kitchen.int8_attention_is_available = lambda: False

    if not hasattr(comfy_kitchen, "rms_rope_split_half_"):
        def rms_rope_split_half_(q, k, rope_freqs, q_weight, k_weight, epsilon=1e-5, rot_dim=None):
            # [1, S, H, D] RMSNorm + split-half rope, in place. rope_freqs is the
            # [1, S, 1, D_rot/2, 2, 2] rotation table comfy builds per forward.
            for x, w in ((q, q_weight), (k, k_weight)):
                f = x.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + epsilon) * w.float()
                rot = f.shape[-1] if rot_dim is None else int(rot_dim)
                half = rot // 2
                lo, hi = f[..., :half].clone(), f[..., half:rot].clone()
                f[..., :half] = rope_freqs[..., 0, 0] * lo + rope_freqs[..., 0, 1] * hi
                f[..., half:rot] = rope_freqs[..., 1, 0] * lo + rope_freqs[..., 1, 1] * hi
                x.copy_(f.to(x.dtype))
            return q, k

        comfy_kitchen.rms_rope_split_half_ = rms_rope_split_half_
        comfy_kitchen.rms_rope_split_half = (
            lambda q, k, *a, **kw: rms_rope_split_half_(q.clone(), k.clone(), *a, **kw))


def _shim_av():
    # comfy_api's video types import colour enums that only exist in newer PyAV.
    # None of the masked-guide code touches video decoding, so stand-ins are enough
    # to let the node module import.
    try:
        import av.video.reformatter as reformatter
    except ImportError:
        return
    class _AnyMember(type):
        # comfy_api reads named members off these enums at import time; the exact
        # values are irrelevant here, only that the attributes exist.
        def __getattr__(cls, name):
            value = cls._members.setdefault(name, object())
            return value

    for name in ("ColorPrimaries", "ColorTrc", "ColorRange"):
        if not hasattr(reformatter, name):
            setattr(reformatter, name, _AnyMember(name, (), {"_members": {}}))


def comfyui_on_path():
    """Put ComfyUI on sys.path with args parsing enabled, or skip the test."""
    raw = os.environ.get("COMFYUI_PATH")
    if not raw:
        pytest.skip("Set COMFYUI_PATH to run ComfyUI node integration tests")
    path = Path(raw).resolve()
    if not (path / "comfy_api").is_dir():
        pytest.fail(f"COMFYUI_PATH is not a ComfyUI checkout: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    if "comfy.options" not in sys.modules:
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
    _shim_comfy_kitchen()
    _shim_av()
    return path


def h3_model_module():
    comfyui_on_path()
    try:
        import comfy.ldm.minimax.model as module
    except Exception as exc:  # pragma: no cover - depends on the ComfyUI build
        pytest.skip(f"MiniMax H3 model is unavailable in this ComfyUI build: {exc}")
    return module


def masked_guide_package():
    """The masked-guide package, mounted standalone as `vlo_masked_guide`.

    Importing it through `nodes/__init__.py` would drag in the whole vlo node
    registry (and its PromptServer routes). A bare namespace parent keeps the
    subpackage's relative imports working without any of that, and lets the
    forward-pass tests run without touching the ComfyUI node API at all.
    """
    comfyui_on_path()
    if "vlo_masked_guide" not in sys.modules:
        import types

        pkg = Path(__file__).resolve().parents[1] / "nodes" / "minimax_masked_guide"
        module = types.ModuleType("vlo_masked_guide")
        module.__path__ = [str(pkg)]
        module.__package__ = "vlo_masked_guide"
        sys.modules["vlo_masked_guide"] = module
    return sys.modules["vlo_masked_guide"]


def masked_guide_module(name):
    """One submodule of the masked-guide package, e.g. "masks" or "nodes"."""
    masked_guide_package()
    import importlib

    return importlib.import_module(f"vlo_masked_guide.{name}")


HIDDEN = 32
HEADS = 2
HEAD_DIM = 128  # rope rotates 96 dims (3 axes x 16 inv-freqs x 2), so heads cannot be narrower


def tiny_h3_model(seed=0):
    """A randomly weighted H3 DiT small enough to run a full forward on CPU."""
    module = h3_model_module()
    import comfy.ops

    torch.manual_seed(seed)
    model = module.MiniMaxH3Model(
        hidden_size=HIDDEN, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, ffn_hidden_size=HIDDEN,
        latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=16,
        timestep_input_dim=16, time_embed_hidden_size=HIDDEN, time_embed_dim=24,
        rope_inv_freq_len=16, dtype=torch.float32, device="cpu",
        operations=comfy.ops.disable_weight_init)
    state = model.state_dict()
    for key, value in state.items():
        if value.is_floating_point():
            state[key] = torch.randn_like(value) * 0.05
    model.load_state_dict(state)
    model.eval()
    return model


def tiny_inputs(latent_t=2, lat_h=4, lat_w=6, audio_t=3, text_len=5, seed=1):
    torch.manual_seed(seed)
    return {
        "x": [torch.randn(1, 24, latent_t, lat_h, lat_w), torch.randn(1, 32, 2, audio_t)],
        "timestep": torch.tensor([500.0]),
        "context": torch.randn(1, text_len, HIDDEN),
    }


def guide_payload(strengths=None, *, latent_t=1, lat_h=4, lat_w=6, frame_idx=3, seed=7,
                  min_aug=0.0, extra_keyframes=()):
    """A payload with one image guide, optionally carrying a masked-guide spec."""
    masked_key = masked_guide_module("masked_h3_forward").MASKED_GUIDE_KEY

    torch.manual_seed(seed)
    latent = torch.randn(1, 24, latent_t, lat_h, lat_w)
    keyframe = {"resolved_frame_index": frame_idx, "latent": latent}
    if strengths is not None:
        keyframe[masked_key] = {"strengths": strengths, "min_aug": min_aug}
    keyframes = [keyframe] + list(extra_keyframes)
    return {
        "keyframes": keyframes,
        "cond_video_latents": [kf["latent"] for kf in keyframes if kf.get("latent") is not None],
        "cond_audio_latents": [],
        "seed": seed,
        "audio_scale": 1.0,
    }
