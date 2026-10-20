"""LoRA loading for pruned MiniMax H3 bases: full-model adaln updates refitted to the curve.

A full H3 adaln projection reads silu(t_emb), 2688 wide, so a LoRA trained on it adds
`up @ down @ silu(t_emb(t))` to each one. A pruned base feeds the same projection an
8-dim point on a curve instead, `adaln_t_table[t]`, so ComfyUI cannot apply that update
("shape '[96768, 8]' is invalid") and skips it, leaving the LoRA running without its
time conditioning. The update is still a smooth function of t alone, though, and the
curve plus a bias spans it to within ~0.1%: the same fit that built the pruned weights
from the full ones. So each adaln pair is replaced by a weight and bias delta fitted
over the whole curve, and the LoRA goes through ComfyUI's ordinary loader from there.

The pruned adaln linears are held in fp32, so the fitted delta merges into them without
rounding away. Everything else in the LoRA is applied exactly as the stock loader would.
"""

from __future__ import annotations

import logging

import torch

import comfy.lora
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .h3_time_grid import VARIANTS, grid_for_base

logger = logging.getLogger(__name__)


_ADALN_WEIGHT_SUFFIX = "adaln_proj.linear.weight"

# (down, up) key suffixes of the LoRA formats an H3 adaln pair arrives in.
_PAIR_SUFFIXES = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora.down.weight", ".lora.up.weight"),
)

_BASE_OPTIONS = ("auto", *VARIANTS)


def curve_projection(table: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """`[d + 1, width]` map from a full-space adaln update to its least-squares curve fit.

    The fit of a target `Y[t] = M @ grid[t]` by `D @ table[t] + b` over every grid row is
    `[D^T; b] = pinv([table, 1]) @ grid @ M^T`. The first two factors depend only on the
    base, so they are formed once and each module costs a pair of skinny products.
    """
    basis = torch.cat([table.double(), table.new_ones(table.shape[0], 1).double()], dim=1)
    return torch.linalg.pinv(basis) @ grid.double()


def fit_curve_delta(
    projection: torch.Tensor, down: torch.Tensor, up: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(weight delta `[out, d]`, bias delta `[out]`) reproducing `scale * up @ down` on the curve."""
    coefficients = (projection @ down.double().T) @ up.double().T * scale
    return coefficients[:-1].T.float().contiguous(), coefficients[-1].float().contiguous()


def _find_pair(lora: dict, prefix: str) -> tuple[str, str] | None:
    for down_suffix, up_suffix in _PAIR_SUFFIXES:
        down_key, up_key = prefix + down_suffix, prefix + up_suffix
        if down_key in lora and up_key in lora:
            return down_key, up_key
    return None


def convert_adaln_for_curve_base(
    lora: dict[str, torch.Tensor],
    key_map: dict[str, str],
    model,
    projection_for_base,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The LoRA with every full-space adaln pair replaced by a fitted `.diff` / `.diff_b`.

    `key_map` is ComfyUI's LoRA-prefix -> model-weight map. `projection_for_base` is only
    called when a pair actually needs converting, so a LoRA without full-space adaln
    updates never fetches a time grid. Returns the new dict and the converted prefixes.
    """
    converted: dict[str, torch.Tensor] = dict(lora)
    prefixes: list[str] = []
    projection = None
    for prefix, weight_key in key_map.items():
        if not weight_key.endswith(_ADALN_WEIGHT_SUFFIX):
            continue
        pair = _find_pair(lora, prefix)
        if pair is None:
            continue
        down_key, up_key = pair
        down, up = lora[down_key], lora[up_key]
        in_features = comfy.utils.get_attr(model, weight_key).shape[1]
        if down.shape[1] == in_features:
            continue
        if projection is None:
            projection = projection_for_base()
        if down.shape[1] != projection.shape[1] or projection.shape[0] != in_features + 1:
            logger.warning(
                "vlo: %s is %d wide but the time grid is %d wide; left for the stock loader",
                down_key,
                down.shape[1],
                projection.shape[1],
            )
            continue
        alpha = lora.get(prefix + ".alpha")
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        diff, diff_bias = fit_curve_delta(projection, down, up, scale)
        trained_bias = lora.get(prefix + ".diff_b")
        if trained_bias is not None:
            # A bias delta is the same at every t, so it carries over beside the fit as is.
            diff_bias = diff_bias + trained_bias.float()
        for key in (down_key, up_key, prefix + ".alpha"):
            converted.pop(key, None)
        converted[prefix + ".diff"] = diff
        converted[prefix + ".diff_b"] = diff_bias
        prefixes.append(prefix)
    return converted, prefixes


def _warn_on_bias_free_curve_patches(lora: dict, key_map: dict[str, str]) -> None:
    """Flag pre-converted curve patches whose bias deltas are all zero.

    A full-space adaln update is not a pure function of the curve coordinates: fitting it
    without the bias leaves a residual about as large as the update itself, so such a
    file applies a wrong time conditioning. The full-space original converts correctly.
    """
    for prefix, weight_key in key_map.items():
        if not weight_key.endswith(_ADALN_WEIGHT_SUFFIX):
            continue
        bias = lora.get(prefix + ".diff_b")
        if prefix + ".diff" in lora and bias is not None and not bias.any():
            logger.warning(
                "vlo: this LoRA's pruned adaln patches have no bias delta, which misfits the "
                "time conditioning; load the full-model version of it here instead"
            )
            return


def load_lora_for_h3(model, lora: dict, strength: float, base: str = "auto", lora_metadata=None):
    """`comfy.sd.load_lora_for_models` for the model alone, converting adaln for a pruned base."""
    diffusion_model = getattr(model.model, "diffusion_model", None)
    if getattr(diffusion_model, "use_adaln_curves", False):
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        _warn_on_bias_free_curve_patches(lora, key_map)
        resolved = {}

        def projection_for_base():
            table = diffusion_model.adaln_t_table
            resolved["variant"], grid = grid_for_base(table, base)
            return curve_projection(table.detach().float().cpu(), grid)

        lora, prefixes = convert_adaln_for_curve_base(lora, key_map, model.model, projection_for_base)
        if prefixes:
            logger.info(
                "vlo: refitted %d adaln updates onto the pruned %s time curve",
                len(prefixes),
                resolved["variant"],
            )
    patched, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0, lora_metadata=lora_metadata)
    return patched


class vloMiniMaxH3LoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloMiniMaxH3LoraLoader",
            search_aliases=["h3 lora", "pruned lora", "minimax lora", "turbo lora", "adaln lora"],
            display_name="vlo MiniMax H3 LoRA Loader",
            category="model/loaders",
            description=(
                "LoRA Loader (model only) that also works on pruned MiniMax H3 checkpoints. "
                "A LoRA trained on the full model carries time-conditioning (adaln) updates "
                "that a pruned base cannot take, so the stock loader logs a shape error and "
                "drops them. Here they are refitted to the pruned base's time curve instead. "
                "The first use per base variant fetches its time embedder (~63 MB) from the "
                "Comfy-Org MiniMax-H3 repository into models/vlo/h3_time_grids. On a full "
                "base, or for a LoRA without adaln updates, this is the stock loader."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Combo.Input(
                    "base",
                    options=list(_BASE_OPTIONS),
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "Which full H3 model the pruned base was cut from, which sets the "
                        "time embedding the LoRA's adaln updates are refitted against. "
                        "'auto' recognises the stock fl2va and ref2va curves and close "
                        "finetunes of them, and refuses anything else."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, base="auto") -> io.NodeOutput:
        if strength_model == 0:
            return io.NodeOutput(model)
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        return io.NodeOutput(load_lora_for_h3(model, lora, strength_model, base, lora_metadata=metadata))
