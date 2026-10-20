"""Seam anchor: present kept MiniMax H3 tokens near an inpaint seam at the sampler's noise.

H3 holds kept tokens clean: it labels them with the conditioning timestep and its
scale_latent_inpaint injects them almost noise-free at every step, which is how it
treats keyframes and references too. Inside one frame that puts a clean patch right
beside pure noise, and the seam forms exactly there. This patch keeps those tokens fully
anchored to the original but, near the seam, injects them noised to (a fraction of) the
current sigma, as Time-to-Move and RePaint do, so the generated tokens beside them see an
ordinary noisy neighbourhood. Tokens further away stay clean, so the model keeps strong
context for what it has to match.

Like TTM's end_step, the rings nearest the seam can also be released partway through:
from then on they stop being held and denoise with the generated area, so the kept edge
can blend into it. Released content no longer ends exactly at the original.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

from .ttm import (
    _TTMDenoiseMaskSchedule,
    _TTMMinimaxModelMaskWrapper,
    _ttm_can_sync_minimax_token_clock,
    _ttm_hold_until,
    _ttm_is_minimax_h3,
)

logger = logging.getLogger(__name__)


_SEAM_ANCHOR_OUTER_SAMPLE_KEY = "vloSeamAnchor_OuterSample"
_SEAM_ANCHOR_APPLY_MODEL_KEY = "vloSeamAnchor_ApplyModel"
_SEAM_ANCHOR_MODES = ("ramp", "noised", "clean")

# H3's DiT patchifies 2x2 latent cells into one token and labels tokens, not cells.
_H3_TOKEN_CELLS = 2

# A phase floor every sigma clears: the last phase of a schedule applies to the end.
_TO_THE_END = float("-inf")


def _cells_to_tokens(cells: torch.Tensor) -> torch.Tensor:
    """`[T, H, W]` cell values -> `[T, ceil(H/2), ceil(W/2)]` token peaks, as H3 pools."""
    height, width = cells.shape[-2:]
    pad = (0, -width % _H3_TOKEN_CELLS, 0, -height % _H3_TOKEN_CELLS)
    padded = F.pad(cells.unsqueeze(1), pad, mode="replicate")
    return F.max_pool2d(padded, _H3_TOKEN_CELLS).squeeze(1)


def _tokens_to_cells(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    cells = tokens.repeat_interleave(_H3_TOKEN_CELLS, dim=-2)
    cells = cells.repeat_interleave(_H3_TOKEN_CELLS, dim=-1)
    return cells[..., :height, :width]


def _seam_rings(generate: torch.Tensor, rings: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each cell's ring around the generated area, and which cells belong to kept tokens.

    `generate` is the video denoise mask on the latent grid, `[T, H, W]`, 1 where the
    sampler generates. A kept token is ring k when it is k tokens (Chebyshev, within its
    frame) from the nearest generated token; rings past `rings`, and every cell of a token
    holding any generated cell, are 0. Those tokens are generated in H3's eyes (it pools
    with amax), so they are left to the sampler.
    """
    height, width = generate.shape[-2:]
    generated = _cells_to_tokens((generate > 0).float())
    ring_index = torch.zeros_like(generated)
    reached = generated
    for ring in range(1, rings + 1):
        grown = F.max_pool2d(reached.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        ring_index[(grown > 0) & (reached == 0)] = ring
        reached = grown
    kept = _tokens_to_cells((generated == 0).float(), height, width) > 0
    return _tokens_to_cells(ring_index, height, width), kept


def _levels_from_rings(
    ring_index: torch.Tensor, kept: torch.Tensor, mode: str, ramp_tokens: int
) -> torch.Tensor:
    """The noise level, as a fraction of sigma, to present each kept cell at."""
    if mode == "clean":
        return torch.zeros_like(ring_index)
    if mode == "noised":
        return kept.to(ring_index.dtype)
    if mode == "ramp":
        # Ring k gets (R - k + 1) / R: 1 against the seam, 1/R at the outermost ring,
        # and 0 (clean, as H3 holds it natively) beyond.
        in_ramp = (ring_index > 0) & (ring_index <= ramp_tokens)
        return torch.where(
            in_ramp, (ramp_tokens - ring_index + 1) / ramp_tokens, torch.zeros_like(ring_index)
        )
    raise ValueError(f"Unsupported seam anchor mode: {mode}")


def _seam_noise_levels(generate: torch.Tensor, mode: str, ramp_tokens: int) -> torch.Tensor:
    """The noise level, as a fraction of sigma, for each cell of `[T, H, W]` `generate`."""
    ring_index, kept = _seam_rings(generate, ramp_tokens if mode == "ramp" else 0)
    return _levels_from_rings(ring_index, kept, mode, ramp_tokens)


def _noised_injection(original, levels: torch.Tensor, latent_shapes):
    """Wrap H3's scale_latent_inpaint so kept tokens with a level are injected noised.

    H3's own injection runs first, so audio, its audio scaling and every other token are
    exactly what H3 would hand the sampler. Only the video cells with a level are
    replaced, by the reference noised to level * sigma with the sampler's own fixed noise.
    """

    def scale_latent_inpaint(sigma, noise, latent_image, x=None, denoise_mask=None, **kwargs):
        injected = original(
            sigma=sigma,
            noise=noise,
            latent_image=latent_image,
            x=x,
            denoise_mask=denoise_mask,
            **kwargs,
        )
        streams = comfy.utils.unpack_latents(injected, latent_shapes)
        reference = comfy.utils.unpack_latents(latent_image, latent_shapes)[0]
        noise_video = comfy.utils.unpack_latents(noise, latent_shapes)[0]

        video = streams[0]
        level = levels.to(device=video.device, dtype=video.dtype)
        sigma_v = sigma.to(device=video.device, dtype=video.dtype)
        sigma_v = sigma_v.reshape(-1, *([1] * (video.ndim - 1)))
        noise_level = level * sigma_v
        noised = (1.0 - noise_level) * reference.to(video.dtype) + noise_level * noise_video.to(
            video.dtype
        )
        streams[0] = torch.where(level > 0, noised, video)
        packed, _ = comfy.utils.pack_latents(streams)
        return packed

    return scale_latent_inpaint


class _SeamAnchorOuterSample:
    def __init__(self, mode: str, ramp_tokens: int, release_step: int = 0, release_tokens: int = 1):
        self.mode = mode
        self.ramp_tokens = ramp_tokens
        self.release_step = release_step
        self.release_tokens = release_tokens

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None, *args, **kwargs):
        def run():
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs)

        guider = executor.class_obj
        model = guider.model_patcher.model
        latent_shapes = kwargs.get("latent_shapes") or [noise.shape]

        if denoise_mask is None:
            logger.info("vloH3SeamAnchor: the sampled latent has no noise mask, so no seam.")
            return run()
        if (
            len(latent_shapes) < 2
            or not _ttm_is_minimax_h3(model)
            or not _ttm_can_sync_minimax_token_clock(model)
            or not hasattr(model, "scale_latent_inpaint")
        ):
            logger.warning(
                "vloH3SeamAnchor: this only applies to MiniMax H3 sampling video and audio "
                "together, where kept tokens are held clean; leaving sampling unchanged. "
                "(A video-only H3 latent already injects kept tokens noised.)"
            )
            return run()

        # 1. The fields, from the mask the sampler will actually use: each kept cell's
        #    noise level, and which kept cells are released partway through.
        streams = comfy.utils.unpack_latents(denoise_mask, latent_shapes)
        video_mask = streams[0]
        releasing = self.release_step > 0
        reach = self.ramp_tokens if self.mode == "ramp" else 0
        if releasing:
            reach = max(reach, self.release_tokens)
        ring_index, kept = _seam_rings(video_mask[0, 0], reach)
        levels = _levels_from_rings(ring_index, kept, self.mode, self.ramp_tokens)
        released = (ring_index > 0) & (ring_index <= self.release_tokens) if releasing else None

        noising = bool((levels > 0).any())
        if not noising and (released is None or not bool(released.any())):
            logger.info("vloH3SeamAnchor: no kept tokens border the generated area.")
            return run()

        def as_video(field):
            return field.to(device=video_mask.device).view(1, 1, *field.shape)

        def packed_with_video(video):
            packed, _ = comfy.utils.pack_latents([video] + list(streams[1:]))
            return packed

        levels = as_video(levels).to(video_mask.dtype)

        # 2. The labels: the real mask, with each ramp token at its level, so H3 reads the
        #    noise it is handed (mask value m labels a token at m * sigma). Released
        #    tokens are then labelled like the generated area.
        label_video = torch.where(levels > 0, levels, video_mask)
        label_phases = [(_TO_THE_END, packed_with_video(label_video))]

        # 3. The release, on TTM's clock: held while step < release_step. The sampler
        #    keeps its own state for held tokens at the reference noised to the global
        #    sigma (their prediction is pinned to the reference), so a released token
        #    resumes from a consistent state whatever level it was shown at.
        if released is not None and bool(released.any()):
            released = as_video(released)
            hold_sigma = _ttm_hold_until(sigmas, self.release_step)
            sampler_after = packed_with_video(
                torch.where(released, torch.ones_like(video_mask), video_mask)
            )
            label_after = packed_with_video(
                torch.where(released, torch.ones_like(label_video), label_video)
            )
            label_phases = [(hold_sigma, label_phases[0][1]), (_TO_THE_END, label_after)]
            if "denoise_mask_function" in guider.model_options:
                logger.warning(
                    "vloH3SeamAnchor: replacing a denoise_mask_function another patch set "
                    "(TTM?); the two cannot share one sampler."
                )
            guider.model_options["denoise_mask_function"] = _TTMDenoiseMaskSchedule(
                [(hold_sigma, denoise_mask), (_TO_THE_END, sampler_after)]
            )

        comfy.patcher_extension.add_wrapper_with_key(
            WrappersMP.APPLY_MODEL,
            _SEAM_ANCHOR_APPLY_MODEL_KEY,
            _TTMMinimaxModelMaskWrapper(_TTMDenoiseMaskSchedule(label_phases), latent_shapes),
            guider.model_options,
            is_model_options=True,
        )
        logger.debug(
            "vloH3SeamAnchor: mode=%s ramp_tokens=%d noised_cells=%d release_step=%d "
            "released_cells=%d",
            self.mode,
            self.ramp_tokens,
            int((levels > 0).sum()),
            self.release_step,
            int(released.sum()) if released is not None else 0,
        )

        if not noising:
            return run()

        # 4. The injection. ComfyUI has no hook for scale_latent_inpaint, so it is swapped
        #    on the model instance for this run only and always put back.
        had_override = "scale_latent_inpaint" in vars(model)
        previous = vars(model).get("scale_latent_inpaint")
        model.scale_latent_inpaint = _noised_injection(
            model.scale_latent_inpaint, levels, latent_shapes
        )
        try:
            return run()
        finally:
            if had_override:
                model.scale_latent_inpaint = previous
            else:
                del model.scale_latent_inpaint


class vloH3SeamAnchor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloH3SeamAnchor",
            search_aliases=["seam anchor", "inpaint seam", "noised anchor", "repaint"],
            display_name="vlo H3 Seam Anchor",
            category="advanced/model",
            description=(
                "Patches MiniMax H3 inpainting so kept tokens beside the generated area are "
                "presented at the sampler's noise level instead of clean. H3 normally holds "
                "every kept token clean, which puts a clean patch right next to pure noise "
                "inside one frame; that is where seams form. The kept content is still "
                "replaced by the original at every step and ends exactly as it was: only "
                "the noise it is shown with changes. Optionally, the rings nearest the seam "
                "are released at a step, like TTM's end_step, and denoise freely from then "
                "on so the kept edge can blend in. Applies to H3 sampling video and audio "
                "together, with a noise mask on the latent; anything else passes through."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode",
                    options=list(_SEAM_ANCHOR_MODES),
                    default="ramp",
                    tooltip=(
                        "'ramp' noises kept tokens near the seam, fully against it and less "
                        "further out, leaving the rest clean. 'noised' noises every kept "
                        "token, which gives the model only a faint view of what it has to "
                        "match. 'clean' is H3's own behaviour, for comparison."
                    ),
                ),
                io.Int.Input(
                    "ramp_tokens",
                    default=2,
                    min=1,
                    max=32,
                    tooltip=(
                        "How many tokens out from the seam the ramp reaches. One H3 token is "
                        "2x2 latent cells, 32 px. Ring k of R is noised to (R - k + 1) / R "
                        "of sigma."
                    ),
                ),
                io.Int.Input(
                    "release_step",
                    default=0,
                    min=0,
                    max=1000,
                    tooltip=(
                        "The step at which the rings nearest the seam stop being held and "
                        "start denoising freely, counted like TTM's end_step: held while "
                        "step < release_step. 0 never releases, so kept content ends exactly "
                        "as it was. H3 keeps sigma high until late, so on a 20-step run a "
                        "release at 15 (sigma ~0.8) still lets shapes adjust, while 18 "
                        "(~0.57) blends mostly colour and texture. Released content changes."
                    ),
                ),
                io.Int.Input(
                    "release_tokens",
                    default=1,
                    min=1,
                    max=32,
                    tooltip=(
                        "How many rings out from the seam are released, in tokens of 32 px. "
                        "Only used when release_step is above 0."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls, model, mode="ramp", ramp_tokens=2, release_step=0, release_tokens=1
    ) -> io.NodeOutput:
        if mode == "clean" and release_step <= 0:
            return io.NodeOutput(model)
        patched = model.clone()
        patched.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            _SEAM_ANCHOR_OUTER_SAMPLE_KEY,
            _SeamAnchorOuterSample(mode, int(ramp_tokens), int(release_step), int(release_tokens)),
        )
        return io.NodeOutput(patched)
