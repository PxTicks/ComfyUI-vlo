"""Guide-mask geometry: canvas resize, token-grid pooling, strength -> noise aug.

A MiniMax H3 image guide reaches the DiT as a *grid* of condition tokens, one
per 2x2 patch of the guide's VAE latent. To give a guide a spatial strength map
the mask has to land on that same grid, so this module mirrors, step for step,
what `MiniMaxH3AddGuide` does to the guide image:

    image --(crop+resize to canvas)--> VAE --> latent [C,T,Hl,Wl] --2x2 patches--> rows
    mask  --(crop+resize to canvas)------------------------------ area pool ----> rows

The two paths must agree on both geometry and row order, so pooling targets the
latent's own token grid (Hl//2, Wl//2) and flattens in `patchify_video` order
(t, then h, then w).
"""

from __future__ import annotations

import torch

import comfy.utils


# Quantization grid for the per-token strengths. Each distinct strength becomes
# a distinct condition timestep, and every distinct timestep costs a row in the
# model's AdaLN modulation table, so soft masks are snapped to a finite ladder
# rather than carrying thousands of near-identical float levels.
MASK_LEVELS = 256

POOLING_MODES = ("average", "max", "min")


def check_mask_matches_image(mask: torch.Tensor, image: torch.Tensor, *, tolerance: float = 0.01) -> None:
    """Reject masks that would crop differently from the guide image.

    Both are cover-cropped to the canvas aspect, so a mask of a different aspect
    ratio silently slides against the image it is supposed to annotate. That is
    exactly the failure mode that still produces plausible video, so it is an
    error rather than a warning.
    """
    mask_aspect = mask.shape[-1] / mask.shape[-2]
    image_aspect = image.shape[-2] / image.shape[-3]
    if abs(mask_aspect - image_aspect) > tolerance * max(mask_aspect, image_aspect):
        raise ValueError(
            "guide mask aspect ratio {}x{} does not match the guide image's {}x{}; "
            "the mask must cover the same framing as the image it weights".format(
                mask.shape[-1], mask.shape[-2], image.shape[-2], image.shape[-3]))


def resize_mask_to_canvas(mask: torch.Tensor, width: int, height: int, crop: str = "center") -> torch.Tensor:
    """[B,H,W] or [H,W] MASK -> [B, height, width] float32, cropped like the guide image.

    `MiniMaxH3AddGuide` resizes guide frames with `common_upscale(..., "lanczos",
    "center")`; the crop geometry lives in `common_upscale` itself, so calling it
    with the same crop mode is what keeps the two aligned. The kernel differs on
    purpose: lanczos rings past [0, 1] on hard mask edges, bilinear does not.

    The batch dimension is preserved rather than collapsed: silently keeping only
    mask 0 of a batch is the kind of thing that produces a plausible-looking wrong
    answer, so callers that can only use one mask say so via `single_mask`.
    """
    m = mask.to(torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    m = m.reshape(-1, *m.shape[-2:]).unsqueeze(1)  # [B, 1, H, W]
    m = comfy.utils.common_upscale(m, width, height, "bilinear", crop)
    return m[:, 0].clamp(0.0, 1.0)


def single_mask(mask: torch.Tensor, *, label: str = "mask") -> torch.Tensor:
    """[B,H,W] or [H,W] MASK -> [H, W], refusing a batch it would have to pick from."""
    m = mask if mask.ndim > 2 else mask.unsqueeze(0)
    m = m.reshape(-1, *m.shape[-2:])
    if m.shape[0] != 1:
        raise ValueError(
            "{} carries {} masks; this node weights a single guide image and has no way to "
            "choose between them".format(label, m.shape[0]))
    return m[0]


def pool_mask_to_tokens(mask_hw: torch.Tensor, token_h: int, token_w: int,
                        mode: str = "average") -> torch.Tensor:
    """[H, W] -> [token_h, token_w].

    Area averaging by default: a token half covered by the mask is worth 0.5, so
    soft edges survive the trip to the token grid. Core H3 max-pools its *denoise*
    masks instead, because there a partially covered token must still be allowed
    to generate; a guidance-strength map has no such conservative direction.
    """
    x = mask_hw.to(torch.float32)[None, None]
    if mode == "average":
        pooled = torch.nn.functional.interpolate(x, size=(token_h, token_w), mode="area")
    elif mode == "max":
        pooled = torch.nn.functional.adaptive_max_pool2d(x, (token_h, token_w))
    elif mode == "min":
        pooled = -torch.nn.functional.adaptive_max_pool2d(-x, (token_h, token_w))
    else:
        raise ValueError("unknown pooling mode {!r}, expected one of {}".format(mode, POOLING_MODES))
    return pooled[0, 0].clamp(0.0, 1.0)


def quantize_strengths(strengths: torch.Tensor, levels: int = MASK_LEVELS) -> torch.Tensor:
    """Snap to a `levels`-step ladder, keeping 0 and 1 exact."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    steps = float(levels - 1)
    return torch.round(strengths * steps) / steps


def shape_strengths(pooled: torch.Tensor, *, strength: float = 1.0, gamma: float = 1.0,
                    levels: int = MASK_LEVELS) -> torch.Tensor:
    """Pooled mask values -> quantized guide strengths in [0, 1].

    Returns float64: the strengths become condition timesteps, and matching core's
    python-float timestep bookkeeping exactly is what keeps a fully-open mask on
    the stock code path (see `strengths_to_aug`). Shared by the still-image and the
    guide-clip paths so a mask means the same thing on both.
    """
    s = pooled.to(torch.float64).clamp(0.0, 1.0)
    if gamma != 1.0:
        s = s.pow(float(gamma))
    s = (s * float(strength)).clamp(0.0, 1.0)
    return quantize_strengths(s, levels)


def guide_token_strengths(mask: torch.Tensor, *, width: int, height: int,
                          token_t: int, token_h: int, token_w: int,
                          strength: float = 1.0, gamma: float = 1.0,
                          pooling: str = "average", levels: int = MASK_LEVELS) -> torch.Tensor:
    """MASK -> flat per-condition-token strengths in [0, 1], `patchify_video` order."""
    canvas = resize_mask_to_canvas(single_mask(mask, label="guide mask"), width, height)
    pooled = pool_mask_to_tokens(canvas[0], token_h, token_w, pooling)
    s = shape_strengths(pooled, strength=strength, gamma=gamma, levels=levels)
    # patchify_video emits rows as (t, h, w); a still guide shares one map across t
    return s.reshape(1, token_h * token_w).expand(token_t, -1).reshape(-1).contiguous()


def strengths_to_aug(strengths: torch.Tensor, a_max: float, a_min: float = 0.0) -> torch.Tensor:
    """Per-token strength -> per-token condition noise-augmentation coefficient.

    s = 1 keeps the token at the model's stock `visual_cond_noise_aug` (~0.999,
    i.e. essentially clean); s = 0 drops it to `a_min` (0.0 = pure noise). The
    endpoints are pinned rather than interpolated so that a fully-open mask is
    *bit-identical* to stock `MiniMaxH3AddGuide` instead of a ulp away from it.
    """
    s = strengths.to(torch.float64)
    a = float(a_min) + s * (float(a_max) - float(a_min))
    a = torch.where(s >= 1.0, torch.full_like(a, float(a_max)), a)
    return torch.where(s <= 0.0, torch.full_like(a, float(a_min)), a)


# How a guide token's confidence becomes a condition timestep. The four clocks
# differ along two axes -- what coefficient a mask value of 0 maps to, and whether
# the modulation label is allowed to disagree with the latent the token carries.
#
#   stock            corrupt per token, label every guide row `visual_cond_noise_aug`.
#                    The baseline: latent corruption with no timestep story at all.
#   floored          label max(t_v, a), core's `max(t_v, visual_cond_noise_aug)` with
#                    the coefficient substituted in. The floor never fires for core
#                    (a is pinned at 0.999); here it fires constantly, and a token
#                    holding pure noise gets labelled as partly informative.
#   matched          label a. What `aug_to_cond_timestep` was always documented to do.
#   target_relative  a mask value of 0 lands the token level with the *target* rather
#                    than at pure noise, then labels it there. This is core's own
#                    denoise-mask row formula -- `t = 1 - m*sigma` -- read backwards,
#                    with guide confidence playing the role of `1 - m`. A zero-
#                    confidence token then carries no *marginal* information rather
#                    than no information; that is a different promise from the other
#                    three, and deliberately so.
GUIDE_CLOCKS = ("stock", "floored", "matched", "target_relative")
DEFAULT_GUIDE_CLOCK = "matched"


def check_guide_clock(clock: str) -> str:
    if clock not in GUIDE_CLOCKS:
        raise ValueError("unknown guide clock {!r}, expected one of {}".format(clock, GUIDE_CLOCKS))
    return clock


def guide_aug_floor(min_aug: float, t_v: float, a_max: float,
                    clock: str = DEFAULT_GUIDE_CLOCK) -> float:
    """The coefficient a mask value of 0 maps to, under `clock`.

    Only `target_relative` moves with the schedule. Raising the floor to `t_v` is
    what makes a zero-confidence token sit exactly where the target sits; the cap at
    `a_max` matters at the very end of sampling, where `t_v` overtakes the condition
    timestep and every guide row collapses back onto the stock scalar path.
    """
    check_guide_clock(clock)
    floor = max(float(min_aug), 0.0)
    if clock == "target_relative":
        floor = max(floor, float(t_v))
    return min(floor, float(a_max))


def aug_to_cond_timestep(aug_rows: torch.Tensor, t_v: float, a_max: float,
                         floor: bool = True) -> torch.Tensor:
    """Per-token condition timestep for a token corrupted to coefficient a.

    `floor=True` reproduces core's `max(t_v, ...)` guard, under which a token
    holding pure noise is labelled as clean as the target has become.

    `floor=False` labels each token as noisy as it actually is -- *except* at the
    open end of the mask. A row sitting exactly at `a_max` is a token the mask
    left fully open, and such a token has to stay a stock guide token in every
    respect, core's `max(t_v, visual_cond_noise_aug)` tail rule included, or a
    fully-open mask stops being bit-identical to a stock `MiniMaxH3AddGuide`
    once `t_v` overtakes `a_max` in the last step or two.

    That endpoint is pinned rather than left to the honest rule for the same
    reason `strengths_to_aug` pins its own: the invariant outranks the labelling
    story, and every later observation about masked guides depends on it. The
    cost is bounded and confined to the open end -- at `t_v > a_max` a fully
    trusted token is labelled `t_v` rather than `a_max`, exactly as core labels
    it. Every token the mask actually closes down keeps its honest label.
    """
    rows = aug_rows.to(torch.float64)
    if floor:
        return torch.clamp(rows, min=float(t_v))
    stock_cond_t = max(float(t_v), float(a_max))
    return torch.where(rows >= float(a_max), torch.full_like(rows, stock_cond_t), rows)
