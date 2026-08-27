"""Turning one masked video into several masked guide clips.

A per-frame mask over a whole video is not one guide: the subject it marks comes
and goes, and MiniMax H3 only accepts guide clips of 1, 5, 22, 39, ... (17k + 5)
frames. This module is the planning half of that -- pure functions over the mask,
with no conditioning or VAE in sight -- so the segmentation strategy can be
changed and tested on its own.

V1's strategy is deliberately the simple one:

  * a frame whose mask carries no guidance at all is dropped;
  * each contiguous run of surviving frames becomes one guide clip;
  * the run is rounded *down* to the nearest length H3 accepts.

The temporal half of the mask pooling lives here too, because it is the piece
that has no equivalent in the still-image path: a guide clip's latent compresses
pixel frames unevenly (`FRAME_PER_TOKEN` is `(1, 4, 4, 4, 4)`), so each latent
time token has to pool exactly the frames it was encoded from.
"""

from __future__ import annotations

import torch

from .compatibility import h3_module
from .masks import MASK_LEVELS, pool_mask_to_tokens, resize_mask_to_canvas, shape_strengths

# H3's guide-clip grid: 5 frames, then one more every 17. Anything shorter than 5
# is a single-frame still guide, which is core's own rule in `MiniMaxH3AddGuide`.
CLIP_BASE = 5
CLIP_PERIOD = 17

TIME_POOLING_MODES = ("average", "max")
CHUNK_ALIGN_MODES = ("start", "center")


def _frame_per_token():
    # Read from core rather than restating it: the cycle is a model constant and
    # a local copy would drift silently.
    return h3_module().FRAME_PER_TOKEN


def permissible_clip_length(n: int) -> int:
    """Largest guide length <= `n` that H3 accepts: 1, 5, 22, 39, ... (17k + 5).

    Returns 0 for `n < 1`. Note 2-4 frames round down to a *single-frame* guide,
    not to nothing: core reads any sub-5 batch as its first frame, so one frame is
    a real length on this ladder rather than a rounding failure.
    """
    n = int(n)
    if n < CLIP_BASE:
        return 1 if n >= 1 else 0
    return ((n - CLIP_BASE) // CLIP_PERIOD) * CLIP_PERIOD + CLIP_BASE


def frame_groups(latent_t: int) -> list[tuple[int, int]]:
    """Pixel frames each latent time token of a guide clip covers, as [start, end).

    The `FRAME_PER_TOKEN` cycle restarts at the clip's own frame 0 regardless of
    where the clip is anchored, so this depends on the clip's length alone.
    """
    fpt = _frame_per_token()
    groups = []
    start = 0
    for k in range(int(latent_t)):
        end = start + int(fpt[k % len(fpt)])
        groups.append((start, end))
        start = end
    return groups


def frames_in_latent_t(latent_t: int) -> int:
    """Pixel frames a guide clip of `latent_t` latent time tokens was encoded from."""
    groups = frame_groups(latent_t)
    return groups[-1][1] if groups else 0


def normalize_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    """[B,H,W] or [H,W] MASK -> [B, H, W] float32, without collapsing the batch."""
    m = mask.to(torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return m.reshape(-1, *m.shape[-2:])


def masks_to_canvas(masks: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """One mask per guide frame -> [N, height, width], cover-cropped like the frames.

    Everything downstream -- which frames are worth guiding with, and what each
    token's strength is -- must be decided on *this* result rather than on the
    masks as they arrived. The canvas crop can discard a whole subject when the
    guide's aspect ratio differs from the target's, and a frame measured before
    the crop but pooled after it can pass `min_coverage` and still produce an
    all-zero strength grid: not an absent guide, but a full segment of pure-noise
    condition tokens.
    """
    return resize_mask_to_canvas(normalize_mask_batch(masks), width, height)


def frame_coverage(canvas_masks: torch.Tensor) -> torch.Tensor:
    """Mean mask value per frame, i.e. how much guidance that frame carries.

    Expects canvas-space masks from `masks_to_canvas`, so coverage means the
    fraction of the *canvas* the mask covers -- what the node's tooltip promises,
    and what the token pooling will actually see.
    """
    m = normalize_mask_batch(canvas_masks).clamp(0.0, 1.0)
    return m.reshape(m.shape[0], -1).mean(dim=1)


def frame_keep_flags(canvas_masks: torch.Tensor, min_coverage: float = 0.0) -> torch.Tensor:
    """Per-frame keep flags. A completely masked-out frame guides nothing, so it goes.

    Takes canvas-space masks; see `masks_to_canvas` for why the crop has to come
    first.

    The comparison is strict, so the 0.0 default drops exactly the frames whose
    mask is empty and keeps everything that carries any guidance at all.
    """
    return frame_coverage(canvas_masks) > float(min_coverage)


def _runs(flags: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, length)."""
    runs = []
    start = None
    values = flags.reshape(-1).tolist()
    for i, on in enumerate(values):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(values) - start))
    return runs


def frames_inside_target(keep: torch.Tensor, *, frame_idx: int, frame_count: int) -> torch.Tensor:
    """Keep flags narrowed to the frames that actually land inside the target video."""
    flags = keep.reshape(-1).to(torch.bool)
    target = torch.arange(flags.shape[0]) + int(frame_idx)
    return flags & (target >= 0) & (target < int(frame_count))


def plan_video_guides(keep: torch.Tensor, *, frame_idx: int, frame_count: int,
                      align: str = "start") -> list[tuple[int, int, int]]:
    """Keep flags -> `(source_start, target_start, length)` for every guide clip.

    Frames that land outside the target video are dropped *before* the runs are
    measured, so a run is rounded down exactly once and the clip it produces always
    fits. Rounding down has to drop frames from somewhere, and `align` says where
    from: `start` keeps the head of the run (core's own truncation), `center` keeps
    its middle.
    """
    if align not in CHUNK_ALIGN_MODES:
        raise ValueError("unknown chunk alignment {!r}, expected one of {}".format(align, CHUNK_ALIGN_MODES))
    inside = frames_inside_target(keep, frame_idx=frame_idx, frame_count=frame_count)

    chunks = []
    for start, length in _runs(inside):
        clip_length = permissible_clip_length(length)
        if clip_length < 1:
            continue
        offset = 0 if align == "start" else (length - clip_length) // 2
        source_start = start + offset
        chunks.append((source_start, source_start + int(frame_idx), clip_length))
    return chunks


def clip_token_strengths(canvas_masks: torch.Tensor, *,
                         token_t: int, token_h: int, token_w: int,
                         strength: float = 1.0, gamma: float = 1.0,
                         spatial_pooling: str = "average", time_pooling: str = "average",
                         levels: int = MASK_LEVELS) -> torch.Tensor:
    """One canvas-space mask per clip frame -> flat per-token strengths, `patchify_video` order.

    Takes the masks already on the canvas (`masks_to_canvas`) rather than resizing
    here, so the frames this pools are the same ones `frame_keep_flags` judged.

    Time first, then space. `average` treats a token whose frames are half covered
    the way the spatial pooling treats a token half covered, which keeps one policy
    across both axes; `max` takes the union of the frames instead, which is what a
    subject moving across the four frames behind one token actually needs.
    """
    if time_pooling not in TIME_POOLING_MODES:
        raise ValueError("unknown time pooling {!r}, expected one of {}".format(time_pooling, TIME_POOLING_MODES))
    groups = frame_groups(token_t)
    expected = groups[-1][1] if groups else 0
    canvas = normalize_mask_batch(canvas_masks)
    if canvas.shape[0] != expected:
        raise ValueError(
            "a {} token guide clip is encoded from {} pixel frames, but {} masks were "
            "given".format(token_t, expected, canvas.shape[0]))

    rows = []
    for a, b in groups:
        span = canvas[a:b]
        merged = span.amax(dim=0) if time_pooling == "max" else span.mean(dim=0)
        rows.append(pool_mask_to_tokens(merged, token_h, token_w, spatial_pooling))
    # patchify_video emits rows as (t, h, w), which is exactly this stack's order
    s = shape_strengths(torch.stack(rows), strength=strength, gamma=gamma, levels=levels)
    return s.reshape(-1).contiguous()
