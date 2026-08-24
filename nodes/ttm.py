"""Time-to-Move: a sampler wrapper that gates the denoise mask by sigma."""

from __future__ import annotations

import logging

import torch

from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

logger = logging.getLogger(__name__)


_TTM_OUTER_SAMPLE_KEY = "vloTTM_OuterSample"


def _ttm_align_mask(
    mask: torch.Tensor,
    latent_shape: torch.Size,
    temporal_ratio: int,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"mask must be [H, W] or [frames, H, W], got {tuple(mask.shape)}."
        )

    latent_frames, latent_height, latent_width = (
        latent_shape[2],
        latent_shape[-2],
        latent_shape[-1],
    )
    mask = mask.to(torch.float32)
    frames = mask.shape[0]

    # The VAE encodes source frame 0 into latent frame 0, then frames
    # (ratio*j - ratio + 1) .. (ratio*j) into latent frame j. Striding by the ratio
    # therefore lands every sampled frame inside the group it masks. Interpolating
    # along time instead would blend neighbouring frames and drift out of alignment,
    # smearing the mask of anything that moves.
    if frames == 1:
        mask = mask.expand(latent_frames, -1, -1)
    elif frames != latent_frames:
        strided = mask[::temporal_ratio]
        if strided.shape[0] == latent_frames:
            mask = strided
        else:
            logger.warning(
                "vloTimeToMove: %d mask frames do not stride onto %d latent frames at "
                "ratio %d; falling back to nearest-frame resampling, which may misalign "
                "the mask. Feed a mask whose frame count matches the reference video.",
                frames,
                latent_frames,
                temporal_ratio,
            )
            index = torch.linspace(0, frames - 1, latent_frames).round().long()
            mask = mask[index]

    # Nearest keeps the mask hard-edged. A linear kernel would leave a ring of partial
    # values around the subject, blending reference into generated content there.
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(1),
        size=(latent_height, latent_width),
        mode="nearest",
    )
    return mask.squeeze(1).view(1, 1, latent_frames, latent_height, latent_width)


class _TTMDenoiseMaskSchedule:
    """Closes the TTM window once the schedule drops past sigma_end.

    KSamplerX0Inpaint holds the masked region for the whole run; TTM only wants it held
    for the opening steps, and this is the hook Comfy provides for varying that per sigma.
    """

    def __init__(self, sigma_end):
        self.sigma_end = sigma_end

    def __call__(self, sigma, denoise_mask, extra_options=None):
        if float(sigma.flatten()[0]) < self.sigma_end:
            return torch.ones_like(denoise_mask)
        return denoise_mask


class _TTMOuterSample:
    def __init__(self, reference_latents, mask, start_step, end_step):
        self.reference_latents = reference_latents
        self.mask = mask
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None, *args, **kwargs):
        guider = executor.class_obj
        model = guider.model_patcher.model

        if torch.count_nonzero(noise) == 0:
            logger.warning(
                "vloTimeToMove: this sampler adds no noise, so there is nothing to seed the "
                "reference into and no noise to hold it with; skipping TTM here. Patch only "
                "the model of the sampler that starts the schedule."
            )
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs)

        latent_shapes = kwargs.get("latent_shapes") or [noise.shape]
        latent_shape = latent_shapes[0]

        reference = self.reference_latents["samples"]
        if tuple(reference.shape) != tuple(latent_shape):
            raise ValueError(
                f"reference_latents {tuple(reference.shape)} must match the sampled latent "
                f"{tuple(latent_shape)}. Encode the reference video at the same resolution "
                f"and frame count the sampler is generating."
            )

        available_steps = sigmas.shape[-1] - 1
        if self.start_step >= available_steps:
            raise ValueError(
                f"start_step ({self.start_step}) must be less than the {available_steps} "
                f"steps this sampler runs."
            )
        if self.start_step == 0:
            logger.warning(
                "vloTimeToMove: start_step is 0, where sigma is 1.0 and the init is therefore "
                "pure noise with no trace of the reference. The motion cue TTM relies on comes "
                "from that init; use 1 or more."
            )
        if denoise_mask is not None:
            logger.warning(
                "vloTimeToMove: replacing the noise_mask already on the sampled latent; TTM "
                "drives the denoise mask itself."
            )

        # Handed over raw. inner_sample runs process_latent_in on it, and KSAMPLER.sample then
        # keeps that normalised tensor as KSamplerX0Inpaint's reference -- so both the init and
        # the per-step hold read the reference in model space, from one source.
        # outer_sample moves it onto the load device below us, along with noise and sigmas.
        latent_image = reference

        # Dropping the leading sigmas is what makes x0 a partially-noised reference rather than
        # pure noise, and skips the steps we jumped over. Equivalent to raising the sampler's
        # start_at_step, but kept here so the two can't fall out of step.
        sigmas = sigmas[self.start_step:]

        # -1 because holding the region *during* the step at sigmas[k] leaves it held at
        # sigmas[k+1]. So to stop holding at end_step, the last pinned call is end_step - 1.
        # This is what makes end_step mean the same thing it means in the TTM reference
        # implementation, where the hold runs while step < end_step.
        last_index = self.end_step - self.start_step - 1
        if last_index < 0:
            # Window closed: seed the init from the reference and let the region run free.
            return executor(noise, latent_image, sampler, sigmas, None, *args, **kwargs)

        # The load device, not noise.device: noise is still on the CPU here, and unlike noise,
        # latent_image and sigmas, outer_sample never moves denoise_mask -- Comfy normally
        # places it in CFGGuider.sample via prepare_mask, which has already run by now.
        ttm_mask = _ttm_align_mask(
            self.mask,
            latent_shape,
            getattr(model.latent_format, "temporal_downscale_ratio", 4),
        ).to(device=guider.model_patcher.load_device, dtype=torch.float32)

        # Comfy's denoise_mask marks where to *denoise*; ours marks where to hold the
        # reference, so it goes in inverted. KSamplerX0Inpaint (samplers.py:633) then noises
        # the reference to each evaluation's own sigma on the way in and pins the x0
        # prediction to it on the way out -- which is what actually holds the region, and
        # holds it for any solver, including ones that evaluate off-schedule.
        denoise_mask = 1.0 - ttm_mask

        sigma_end = float(sigmas[min(last_index, sigmas.shape[-1] - 1)])
        guider.model_options["denoise_mask_function"] = _TTMDenoiseMaskSchedule(sigma_end)

        return executor(noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs)


class vloTimeToMove(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloTimeToMove",
            search_aliases=["ttm", "time to move", "motion transfer", "cut and drag"],
            display_name="vlo Time-to-Move (TTM)",
            category="advanced/model",
            description=(
                "Patches a video model to follow the motion in a reference clip, as in "
                "Time-to-Move (https://github.com/time-to-move/TTM). The reference latents "
                "seed the sampler's starting latent, which is what carries the intended "
                "motion, and the masked region is then held to the reference for the opening "
                "steps via Comfy's own inpaint path, so it works with any sampler. Patch only "
                "the model of the sampler that starts the schedule, and leave that sampler's "
                "start_at_step at 0. Drives the denoise mask, so any noise_mask already on "
                "the sampled latent is replaced."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "reference_latents",
                    tooltip=(
                        "Encoded reference video, e.g. the cut-and-drag clip. Must match the "
                        "resolution and frame count being sampled. This replaces whatever "
                        "latent is wired into the sampler."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "White marks the region held to the reference; black is left free for "
                        "the model to generate. For a moving subject that means white "
                        "background and a black hole over the subject. Pixel resolution and "
                        "frame count are matched to the latent grid automatically."
                    ),
                ),
                io.Int.Input(
                    "start_step",
                    default=1,
                    min=0,
                    max=1000,
                    tooltip=(
                        "Step whose noise level the reference is seeded at. The sampler skips "
                        "the steps before it. Higher values leave less noise on the reference, "
                        "binding the result more tightly to it at the cost of a step and of "
                        "the model's freedom to clean up paste artifacts. 0 is a no-op: sigma "
                        "is 1.0 there and the reference washes out entirely."
                    ),
                ),
                io.Int.Input(
                    "end_step",
                    default=2,
                    min=0,
                    max=1000,
                    tooltip=(
                        "The step at which the region stops being held to the reference and "
                        "starts denoising freely. Exclusive, and counted the same way the TTM "
                        "reference implementation counts it. Set at or below start_step to "
                        "seed the init only and never hold the region at all."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, reference_latents, mask, start_step=1, end_step=2) -> io.NodeOutput:
        patched = model.clone()
        patched.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            _TTM_OUTER_SAMPLE_KEY,
            _TTMOuterSample(reference_latents, mask, start_step, end_step),
        )
        return io.NodeOutput(patched)
