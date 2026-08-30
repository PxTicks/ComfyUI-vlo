"""One guide plan, built once and consumed by both conditioning paths.

A MiniMax H3 guide reaches the model through two entirely separate channels:

    aligned pixels --VAE--> condition latent rows   (the DiT / PackedLayout path)
    aligned pixels --Qwen-> timestamped vision block (the semantic path)

Those two have to describe the *same* observation or the model is being told
two different things about one moment. That is not a property you can get by
running the planning twice: the canvas cover-crop, the `min_coverage` filter,
the run segmentation and H3's clip-length rounding each discard pixels, and a
second implementation would have to discard exactly the same ones forever.

So the plan is computed once, here, and both consumers read the result. In
particular `GuideChunk.aligned_frames` is the tensor that was handed to the VAE
-- the same object, not an equal one -- so the semantic path cannot show Qwen a
frame the latent guide does not contain.

The masking half stays where it was: this module only *carries* the per-token
strengths that `masks.py` and `clips.py` compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .clips import (
    clip_token_strengths,
    frame_keep_flags,
    frames_in_latent_t,
    frames_inside_target,
    masks_to_canvas,
    normalize_mask_batch,
    plan_video_guides,
)
from .masks import MASK_LEVELS, check_mask_matches_image, guide_token_strengths


@dataclass(frozen=True)
class GuideChunk:
    """One guide clip (or still): its pixels, its latent, its per-token strengths."""

    aligned_frames: torch.Tensor  # [N, H, W, C] on the target canvas -- the guide's only geometry
    latent: torch.Tensor  # vae.encode(aligned_frames)
    strengths: torch.Tensor  # flat per-condition-token, patchify_video order
    target_start: int  # resolved frame index in the *target* timeline
    source_start: int
    length: int
    token_t: int
    token_h: int
    token_w: int
    strength: float
    min_aug: float
    gamma: float

    @property
    def token_count(self) -> int:
        return self.token_t * self.token_h * self.token_w

    @property
    def semantic_eligible(self) -> bool:
        """Whether every condition token of this chunk is at full confidence.

        The test is exact equality with 1.0, not a threshold, and it is exact
        because `shape_strengths` quantizes onto a `MASK_LEVELS` ladder before
        this ever runs: a white mask that resampling left at 0.99999994 rounds
        back to exactly 1.0, while a genuine 0.99 lands two steps below it. The
        quantizer is what makes the strict comparison safe, so a threshold here
        would only widen the gate, never rescue a rounding artefact.

        Anything short of that -- a partial mask, a feathered edge, `strength`
        below 1, a temporally half-covered token -- means the latent guide is
        deliberately telling the model that part of this observation is absent
        or unreliable. Qwen has no way to represent that, so it gets nothing.
        """
        return bool(torch.all(self.strengths == 1.0))

    def keyframe(self, masked_guide_key: str) -> dict:
        """The `minimax_keyframes` entry: core's own fields plus this pack's."""
        return {
            "resolved_frame_index": self.target_start,
            "latent": self.latent,
            # Core ignores the extra key; only the patched forward reads it.
            masked_guide_key: {
                "strengths": self.strengths,
                "strength": self.strength,
                "min_aug": self.min_aug,
                "gamma": self.gamma,
                "token_t": self.token_t,
                "token_h": self.token_h,
                "token_w": self.token_w,
                "resolved_frame_index": self.target_start,
            },
        }


@dataclass(frozen=True)
class GuideSpec:
    """Guide chunks plus the target they were planned against."""

    width: int
    height: int
    frame_count: int
    chunks: tuple[GuideChunk, ...] = field(default_factory=tuple)

    def check_target(self, width: int, height: int, frame_count: int) -> None:
        """Refuse a spec planned against a different video than it is being applied to.

        The spec is built before the conditioning node makes its own latent, so
        the two geometries are stated twice and nothing but this check stops them
        drifting. A mismatch is not recoverable: every crop, every clip-length
        rounding and every timestamp in the spec was decided on the other canvas.
        """
        if (self.width, self.height, self.frame_count) != (width, height, frame_count):
            raise ValueError(
                "this guide spec was planned for a {}x{} video of {} frames, but it is being "
                "applied to a {}x{} video of {} frames; build the spec from the same latent "
                "the conditioning node is given".format(
                    self.width, self.height, self.frame_count, width, height, frame_count))


def _token_grid(latent: torch.Tensor) -> tuple[int, int, int]:
    if latent.ndim != 5 or latent.shape[3] % 2 or latent.shape[4] % 2:
        raise ValueError(
            "guide latent {} does not tile into H3's 2x2 condition patches".format(tuple(latent.shape)))
    return int(latent.shape[2]), int(latent.shape[3]) // 2, int(latent.shape[4]) // 2


def resolve_frame_idx(frame_idx: int, frame_count: int) -> int:
    return int(frame_idx) if frame_idx >= 0 else int(frame_count) + int(frame_idx)


def build_still_guide(*, vae, resize, image: torch.Tensor, mask: torch.Tensor | None,
                      width: int, height: int, frame_count: int, frame_idx: int,
                      strength: float = 1.0, min_aug: float = 0.0,
                      gamma: float = 1.0) -> GuideSpec:
    """One still image -> a one-chunk spec. `mask=None` means full confidence."""
    if image.shape[0] != 1:
        # Core's AddGuide would read a >= 5 frame batch as a guide clip and a shorter
        # one as its first frame; neither is a thing this node can weight with one
        # mask, so both are refused rather than quietly truncated.
        raise ValueError(
            "masked guides support single-image guides only; received a batch of {} "
            "images. Use the from-video node for guide clips".format(image.shape[0]))
    if mask is None:
        mask = torch.ones(1, image.shape[1], image.shape[2], dtype=torch.float32)
    else:
        check_mask_matches_image(mask, image)

    resolved = resolve_frame_idx(frame_idx, frame_count)
    if resolved < 0 or resolved >= frame_count:
        raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))

    aligned = resize(image, width, height, "center")
    latent = vae.encode(aligned)
    token_t, token_h, token_w = _token_grid(latent)
    strengths = guide_token_strengths(
        mask, width=width, height=height, token_t=token_t, token_h=token_h, token_w=token_w,
        strength=strength, gamma=gamma, pooling="average", levels=MASK_LEVELS)

    chunk = GuideChunk(
        aligned_frames=aligned, latent=latent, strengths=strengths,
        target_start=resolved, source_start=0, length=1,
        token_t=token_t, token_h=token_h, token_w=token_w,
        strength=float(strength), min_aug=float(min_aug), gamma=float(gamma))
    return GuideSpec(width=width, height=height, frame_count=frame_count, chunks=(chunk,))


def build_video_guides(*, vae, resize, video: torch.Tensor, mask: torch.Tensor | None,
                       width: int, height: int, frame_count: int, frame_idx: int,
                       strength: float = 1.0, min_aug: float = 0.0, gamma: float = 1.0,
                       min_coverage: float = 0.0, time_pooling: str = "average",
                       chunk_align: str = "start") -> tuple[GuideSpec, str]:
    """A masked video -> a spec of several guide clips, plus the plan as text."""
    if mask is None:
        masks = torch.ones(video.shape[0], video.shape[1], video.shape[2], dtype=torch.float32)
    else:
        masks = normalize_mask_batch(mask)
        if masks.shape[0] == 1 and video.shape[0] != 1:
            # A single mask over a clip is a still confidence map, not a per-frame one;
            # that is a real (if degenerate) request, so broadcast it explicitly.
            masks = masks.expand(video.shape[0], -1, -1)
        elif masks.shape[0] != video.shape[0]:
            raise ValueError(
                "received {} guide frames and {} masks; pass one mask per frame, or a single "
                "mask to apply to all of them".format(video.shape[0], masks.shape[0]))
        check_mask_matches_image(masks, video)

    # Crop to the canvas once, up front: which frames are worth guiding with and
    # what each token's strength is are then decided on the same pixels. Measured
    # before the crop, a subject sitting in a band the crop discards passes
    # min_coverage and then pools to an all-zero strength grid -- which is not an
    # absent guide but a segment of pure-noise condition tokens.
    canvas_masks = masks_to_canvas(masks, width, height)
    start = resolve_frame_idx(frame_idx, frame_count)
    keep = frame_keep_flags(canvas_masks, min_coverage)
    plan = plan_video_guides(keep, frame_idx=start, frame_count=frame_count, align=chunk_align)
    if not plan:
        # Wiring up a video and a mask and silently adding no guidance at all is
        # exactly the kind of plausible-looking nothing this pack refuses to do.
        raise ValueError(
            "no guide clips survive: of {} guide frames anchored at frame {}, {} carry a "
            "mask above min_coverage {} and none of the runs they form reaches a frame that "
            "fits inside the target video's {} frames".format(
                video.shape[0], start, int(keep.sum()), min_coverage, frame_count))

    chunks = []
    report = []
    total_tokens = 0
    for source_start, target_start, length in plan:
        aligned = resize(video[source_start:source_start + length], width, height, "center")
        latent = vae.encode(aligned)
        token_t, token_h, token_w = _token_grid(latent)
        if frames_in_latent_t(token_t) != length:
            # The mask is pooled onto the latent's time grid by FRAME_PER_TOKEN, so a VAE
            # whose temporal compression differs would slide the mask against the frames.
            raise ValueError(
                "the vae encoded {} guide frames into {} latent time tokens, which cover {} "
                "frames on H3's grid; the mask cannot be aligned to that".format(
                    length, token_t, frames_in_latent_t(token_t)))
        strengths = clip_token_strengths(
            canvas_masks[source_start:source_start + length],
            token_t=token_t, token_h=token_h, token_w=token_w, strength=strength,
            gamma=gamma, spatial_pooling="average", time_pooling=time_pooling,
            levels=MASK_LEVELS)
        chunks.append(GuideChunk(
            aligned_frames=aligned, latent=latent, strengths=strengths,
            target_start=target_start, source_start=source_start, length=length,
            token_t=token_t, token_h=token_h, token_w=token_w,
            strength=float(strength), min_aug=float(min_aug), gamma=float(gamma)))
        tokens = token_t * token_h * token_w
        total_tokens += tokens
        report.append(
            "  source {}-{} -> target {}-{} ({} frames, {}x{}x{} = {} tokens, "
            "strength mean {:.3f})".format(
                source_start, source_start + length - 1, target_start,
                target_start + length - 1, length, token_t, token_h, token_w, tokens,
                float(strengths.mean())))

    # The three ways a frame can fail to become guidance are worth telling apart:
    # an empty mask is the input's business, rounding is this node's.
    inside = frames_inside_target(keep, frame_idx=start, frame_count=frame_count)
    guided = sum(length for _, _, length in plan)
    text = "\n".join(
        ["{} masked guide clip(s) covering {} of {} frames ({} dropped by the mask, "
         "{} outside the target video, {} to clip-length rounding)".format(
             len(plan), guided, int(video.shape[0]),
             int(video.shape[0] - keep.sum()), int(keep.sum() - inside.sum()),
             int(inside.sum()) - guided)]
        + report
        + ["condition tokens riding every sampling step: {}".format(total_tokens)])
    return GuideSpec(width=width, height=height, frame_count=frame_count,
                     chunks=tuple(chunks)), text
