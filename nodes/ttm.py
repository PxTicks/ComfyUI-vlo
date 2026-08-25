"""Time-to-Move: a sampler wrapper that gates the denoise mask by sigma."""

from __future__ import annotations

import logging
import math

import torch

import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.utils

from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

logger = logging.getLogger(__name__)


_TTM_OUTER_SAMPLE_KEY = "vloTTM_OuterSample"
_TTM_APPLY_MODEL_KEY = "vloTTM_ApplyModel"

# The conds MiniMax H3's extra_conds builds from a denoise mask, and the only ones TTM
# overrides. Audio is left out entirely: TTM never holds it, so it denoises normally.
_TTM_MODEL_MASK_CONDS = ("denoise_mask", "audio_denoise_mask")

# MiniMax H3's video VAE encodes in fixed clips rather than at a flat stride: it cuts the
# source into 17-frame clips, pads each one up so it divides by the temporal ratio of 4,
# encodes it to 5 latent frames, then drops the last 3 latent frames of the whole run.
# That is what makes its frame counts 17k + 5 -> 5k + 2 (73 -> 22, 90 -> 27) instead of the
# (n - 1) / 4 + 1 a striding VAE would give, and it restarts the grouping at every clip
# boundary, which a single ratio cannot express. Mirrors comfy/ldm/minimax/vae.py's
# MiniMaxH3VideoVAE defaults and comfy_extras/nodes_minimax_h3.py's video_latent_t.
_H3_CLIP_FRAMES = 17
_H3_FRAMES_PER_LATENT = 4
_H3_LATENTS_PER_CLIP = math.ceil(_H3_CLIP_FRAMES / _H3_FRAMES_PER_LATENT)  # 5
_H3_CLIP_PRE_PADDING = (-_H3_CLIP_FRAMES) % _H3_FRAMES_PER_LATENT  # 3
_H3_DROPPED_LATENTS = 3


def _h3_latent_frames(source_frames: int) -> int:
    """Latent frames MiniMax H3's video VAE produces from this many source frames."""
    clips = math.ceil(max(1, source_frames) / _H3_CLIP_FRAMES)
    return clips * _H3_LATENTS_PER_CLIP - _H3_DROPPED_LATENTS


def _h3_source_frames(latent_frames: int) -> int | None:
    """The canonical 17k + 5 source length for 5k + 2 latent frames, if there is one."""
    if latent_frames < 2 or (latent_frames - 2) % _H3_LATENTS_PER_CLIP != 0:
        return None
    return (latent_frames - 2) // _H3_LATENTS_PER_CLIP * _H3_CLIP_FRAMES + 5


def _h3_anchor_frames(latent_frames: int, source_frames: int) -> list[int]:
    """The last source frame feeding each latent frame.

    Within a clip the causal encoder maps source frame 0 to latent frame 0, then frames
    (4j - 3) .. (4j) to latent frame j. Taking the last frame of each group is the same
    choice the generic stride path makes, and it lands every sampled frame inside the group
    it masks. Frames past the end of the mask clamp to its last frame, matching the way the
    encoder pads a short final clip by repeating that frame.
    """
    anchors = []
    for latent_frame in range(latent_frames):
        clip_start = latent_frame // _H3_LATENTS_PER_CLIP * _H3_CLIP_FRAMES
        index_in_clip = latent_frame % _H3_LATENTS_PER_CLIP
        group_end = min(
            _H3_CLIP_FRAMES,
            (index_in_clip + 1) * _H3_FRAMES_PER_LATENT - _H3_CLIP_PRE_PADDING,
        )
        anchors.append(min(clip_start + group_end - 1, source_frames - 1))
    return anchors


def _ttm_mask_frames(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"mask must be [H, W] or [frames, H, W], got {tuple(mask.shape)}."
        )
    return mask.to(torch.float32)


def _ttm_mask_to_latent_grid(
    frames: torch.Tensor,
    latent_shape: torch.Size,
) -> torch.Tensor:
    latent_frames, latent_height, latent_width = (
        latent_shape[2],
        latent_shape[-2],
        latent_shape[-1],
    )
    # Nearest keeps the mask hard-edged. A linear kernel would leave a ring of partial
    # values around the subject, blending reference into generated content there.
    frames = torch.nn.functional.interpolate(
        frames.unsqueeze(1),
        size=(latent_height, latent_width),
        mode="nearest",
    )
    return frames.squeeze(1).view(1, 1, latent_frames, latent_height, latent_width)


def _ttm_align_mask(
    mask: torch.Tensor,
    latent_shape: torch.Size,
    temporal_ratio: int,
) -> torch.Tensor:
    mask = _ttm_mask_frames(mask)
    latent_frames = latent_shape[2]
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

    return _ttm_mask_to_latent_grid(mask, latent_shape)


def _ttm_align_mask_minimax(
    mask: torch.Tensor,
    latent_shape: torch.Size,
) -> torch.Tensor:
    """Align a source-frame mask onto MiniMax H3's clipped video latent timeline."""
    mask = _ttm_mask_frames(mask)
    latent_frames = latent_shape[2]
    frames = mask.shape[0]

    if frames == 1:
        return _ttm_mask_to_latent_grid(mask.expand(latent_frames, -1, -1), latent_shape)

    canonical_frames = _h3_source_frames(latent_frames)
    if canonical_frames is None:
        logger.warning(
            "vloTimeToMove: %d video latent frames are not on MiniMax H3's 5k + 2 grid, so "
            "the clip geometry cannot be reconstructed; falling back to stride alignment, "
            "which may misalign the mask.",
            latent_frames,
        )
        return _ttm_align_mask(mask, latent_shape, _H3_FRAMES_PER_LATENT)

    if _h3_latent_frames(frames) != latent_frames:
        logger.warning(
            "vloTimeToMove: a %d-frame mask encodes to %d MiniMax H3 latent frames, but the "
            "sampled video latent has %d (%d source frames). The mask is being stretched or "
            "clipped onto that timeline and will misalign. Trim the mask and the reference "
            "video together to %d frames before encoding.",
            frames,
            _h3_latent_frames(frames),
            latent_frames,
            canonical_frames,
            canonical_frames,
        )
    elif frames > canonical_frames:
        # 77 frames and 73 frames both encode to 22 latents: the trailing 4 land in latent
        # frames the VAE drops, so the reference never carried them either.
        logger.info(
            "vloTimeToMove: MiniMax H3 keeps only the first %d of %d mask frames; the rest "
            "fall in latent frames its VAE drops.",
            canonical_frames,
            frames,
        )
    elif frames < canonical_frames:
        # Every length in a clip's bucket encodes to the same latent count, so this mask
        # cannot be told apart from a canonical one by its shape -- but its tail anchors
        # clamp to its last frame. That is right only if the reference really was this
        # short; if the reference was canonical, the closing latent frames read a mask
        # frame up to a clip early, which drags behind anything moving there.
        logger.warning(
            "vloTimeToMove: a %d-frame mask and a %d-frame one both encode to %d MiniMax H3 "
            "latent frames, so this mask's last %d latent frame(s) repeat its final frame. "
            "If the reference video was %d frames, the mask is short by %d and its ending "
            "will lag. Feed a mask with one frame per reference frame.",
            frames,
            canonical_frames,
            latent_frames,
            sum(1 for anchor in _h3_anchor_frames(latent_frames, frames) if anchor == frames - 1),
            canonical_frames,
            canonical_frames - frames,
        )

    anchors = torch.tensor(_h3_anchor_frames(latent_frames, frames), dtype=torch.long)
    return _ttm_mask_to_latent_grid(mask[anchors], latent_shape)


def _ttm_replace_video_reference(
    latent_image: torch.Tensor,
    reference: torch.Tensor,
    latent_shapes: list[torch.Size],
) -> torch.Tensor:
    """Swap the video stream of a packed multi-stream latent for the TTM reference.

    Models like MiniMax H3 sample video and audio as one flat packed latent. Handing the
    sampler a bare video tensor in its place would leave every downstream unpack reading
    the wrong stream boundaries; the other streams have nothing to do with TTM and are
    carried through untouched.
    """
    streams = comfy.utils.unpack_latents(latent_image, latent_shapes)
    streams[0] = reference.to(device=latent_image.device, dtype=latent_image.dtype)
    packed, _ = comfy.utils.pack_latents(streams)
    return packed


def _ttm_prepare_packed_mask(
    video_mask: torch.Tensor,
    latent_shapes: list[torch.Size],
    device,
) -> torch.Tensor:
    """Pack the video denoise mask with an all-ones mask per remaining stream.

    CFGGuider.sample normally does this before outer_sample runs: it expands each stream's
    mask to that stream's full channel shape, invents all-ones masks for the streams the
    caller left out, and packs them. TTM builds its mask after that has already happened,
    so it has to do the same preparation itself -- a bare [1, 1, T, H, W] video mask packed
    against a video+audio latent is the wrong length and blows up on the first unpack.
    """
    masks = [comfy.sampler_helpers.prepare_mask(video_mask, latent_shapes[0], device)]
    for shape in latent_shapes[1:]:
        # Ones means "denoise normally", so the other streams are untouched by TTM.
        masks.append(torch.ones(shape, device=device, dtype=torch.float32))
    packed, _ = comfy.utils.pack_latents(masks)
    return packed.float()


def _ttm_can_sync_minimax_token_clock(model) -> bool:
    """Whether this model exposes the per-token mask conditioning TTM has to keep in step."""
    if not _ttm_is_minimax_h3(model):
        return False
    if not hasattr(model, "_denoise_mask_values"):
        # Older H3 builds ignore the denoise mask entirely, so there is no clock to sync and
        # the sampler-side hold is the whole of TTM. Newer ones may rename this; say so
        # rather than silently leaving the transformer pinned after the window closes.
        logger.warning(
            "vloTimeToMove: this ComfyUI's MiniMax H3 does not expose the per-token denoise "
            "mask conditioning, so its token clock cannot be moved along with the sampler-"
            "side hold. TTM still runs -- this is how H3 behaved before per-token masks -- "
            "but the held tokens stay on the global clock, so start_step and end_step bind "
            "less tightly than they otherwise would."
        )
        return False
    return True


def _ttm_is_minimax_h3(model) -> bool:
    # A capability check, not a name check: MiniMaxH3AV subclasses the video format, and
    # the clip geometry belongs to the VAE the format describes. Imported lazily so this
    # node pack keeps loading on ComfyUI builds that predate MiniMax H3.
    try:
        from comfy.latent_formats import MiniMaxH3Video
    except (AttributeError, ImportError):
        return False
    return isinstance(getattr(model, "latent_format", None), MiniMaxH3Video)


class _TTMDenoiseMaskSchedule:
    """Which denoise mask applies at a given sigma, and when TTM stops applying one at all.

    KSamplerX0Inpaint holds the masked region for the whole run; TTM wants it held only for
    the opening steps, and this is the hook Comfy provides for varying that per sigma. On a
    packed AV latent there are two windows rather than one -- the whole video stream held
    while the reference is being seeded, then just the TTM region -- so the schedule is a
    list of phases rather than a single cutoff. Phases are ordered from the highest sigma
    down; below the last one TTM is finished and everything denoises freely.

    The one place the window is decided: MiniMax H3's model-side token clock reads
    `mask_at` off this same object, so the sampler and the transformer cannot drift apart.
    """

    def __init__(self, phases):
        self.phases = tuple(phases)

    def mask_at(self, sigma):
        """The denoise mask this evaluation's sigma falls under, or None once TTM is done."""
        value = float(sigma.flatten()[0])
        for sigma_floor, mask in self.phases:
            if value >= sigma_floor:
                return mask
        return None

    def __call__(self, sigma, denoise_mask, extra_options=None):
        mask = self.mask_at(sigma)
        if mask is None:
            return torch.ones_like(denoise_mask)
        return mask


class _TTMMinimaxModelMaskWrapper:
    """Keeps MiniMax H3's per-token clock on the same schedule as the sampler-side hold.

    H3 reads a denoise mask as a per-token diffusion strength rather than as an after-the-
    fact blend: mask value m puts that token's row at sigma = m * sigma_stream, so a held
    row runs at the near-clean conditioning timestep, and scale_latent_inpaint injects its
    latent at that same strength. The catch is the ordering. CFGGuider.inner_sample turns
    the denoise mask into model conditioning once, before the sampler loop, while
    denoise_mask_function runs on every evaluation -- so releasing the sampler-side hold
    would otherwise leave the transformer still believing those tokens are pinned, denoising
    against noise that is no longer being held out of them.
    """

    def __init__(self, schedule, latent_shapes):
        self.schedule = schedule
        self.latent_shapes = latent_shapes
        self._conds = {}

    def _conds_for(self, model, mask):
        # A phase's mask does not vary within that phase, so each one is pooled once for the
        # whole run rather than on every model evaluation. The schedule owns the tensors, so
        # identity is a stable key for as long as this wrapper lives.
        if mask is None:
            return {}
        key = id(mask)
        if key not in self._conds:
            self._conds[key] = model._denoise_mask_values(mask, self.latent_shapes)
        return self._conds[key]

    def __call__(self, executor, x, sigma, *args, **kwargs):
        # apply_model hands the wrapper the raw sigma, the same value KSamplerX0Inpaint
        # passes to denoise_mask_function, so both sides read the window at one clock.
        conds = self._conds_for(executor.class_obj, self.schedule.mask_at(sigma))
        for name in _TTM_MODEL_MASK_CONDS:
            if name in conds:
                kwargs[name] = conds[name]
            else:
                # Dropping the key is how the model is told to run this stream normally;
                # an all-ones mask would take the same path at more cost.
                kwargs.pop(name, None)
        return executor(x, sigma, *args, **kwargs)


class _TTMOuterSample:
    def __init__(self, reference_latents, mask, start_step, end_step):
        self.reference_latents = reference_latents
        self.mask = mask
        self.start_step = start_step
        self.end_step = end_step

    def _video_reference(self, latent_shape: torch.Size) -> torch.Tensor:
        reference = self.reference_latents["samples"]
        if getattr(reference, "is_nested", False):
            streams = reference.unbind()
            if len(streams) > 1:
                logger.info(
                    "vloTimeToMove: using only the video stream of a %d-stream reference "
                    "latent; TTM leaves the other streams to denoise normally.",
                    len(streams),
                )
            reference = streams[0]
        if tuple(reference.shape) != tuple(latent_shape):
            raise ValueError(
                f"reference_latents {tuple(reference.shape)} must match the sampled video "
                f"latent {tuple(latent_shape)}. Encode the reference video at the same "
                f"resolution and frame count the sampler is generating."
            )
        return reference

    def _ttm_mask(self, model, latent_shape, device) -> torch.Tensor:
        """The TTM hold, as a denoise mask on the video latent grid."""
        if _ttm_is_minimax_h3(model):
            ttm_mask = _ttm_align_mask_minimax(self.mask, latent_shape)
        else:
            ttm_mask = _ttm_align_mask(
                self.mask,
                latent_shape,
                getattr(model.latent_format, "temporal_downscale_ratio", 4),
            )
        # Comfy's denoise_mask marks where to *denoise*; ours marks where to hold the
        # reference, so it goes in inverted. KSamplerX0Inpaint (samplers.py:633) then noises
        # the reference to each evaluation's own sigma on the way in and pins the x0
        # prediction to it on the way out -- which is what actually holds the region, and
        # holds it for any solver, including ones that evaluate off-schedule.
        return 1.0 - ttm_mask.to(device=device, dtype=torch.float32)

    def _hold_until(self, sigmas, step):
        """The sigma at which a hold that runs while `step < N` stops holding.

        -1 because holding the region *during* the step at sigmas[k] leaves it held at
        sigmas[k+1]. So to stop holding at step N, the last pinned call is N - 1. This is
        what makes end_step mean the same thing it means in the TTM reference
        implementation, where the hold runs while step < end_step.
        """
        return float(sigmas[min(step - 1, sigmas.shape[-1] - 1)])

    def _single_stream_plan(self, model, latent_shape, sigmas, device):
        """Skip straight to start_step, starting from a partially-noised reference.

        Dropping the leading sigmas is what makes x0 a partially-noised reference rather
        than pure noise, and skips the steps we jumped over. Equivalent to raising the
        sampler's start_at_step, but kept here so the two can't fall out of step.
        """
        sigmas = sigmas[self.start_step:]
        if self.end_step <= self.start_step:
            return sigmas, []
        return sigmas, [(
            self._hold_until(sigmas, self.end_step - self.start_step),
            self._ttm_mask(model, latent_shape, device),
        )]

    def _packed_plan(self, model, latent_shapes, sigmas, device):
        """Run the whole schedule, holding the video rather than skipping the steps.

        A packed latent samples every stream off one shared sigma, so slicing the schedule
        to reach start_step would take the audio stream's opening steps away with it --
        dropping it in already part-denoised, against a clean component of zero, on a clock
        that runs faster than the video's. Instead the full schedule runs and the whole
        video stream is held to the reference until start_step. Pinning the x0 prediction to
        the reference makes the solver integrate to exactly the partially-noised reference
        the sliced path would have started from, so start_step keeps its meaning, at the
        cost of the start_step evaluations that seed it. Audio is never held.
        """
        video_shape = latent_shapes[0]
        phases = []
        if self.start_step > 0:
            seed_mask = torch.zeros(
                (1, 1) + tuple(video_shape[2:]), device=device, dtype=torch.float32
            )
            phases.append((
                self._hold_until(sigmas, self.start_step),
                _ttm_prepare_packed_mask(seed_mask, latent_shapes, device),
            ))
        if self.end_step > self.start_step:
            phases.append((
                self._hold_until(sigmas, self.end_step),
                _ttm_prepare_packed_mask(
                    self._ttm_mask(model, video_shape, device), latent_shapes, device
                ),
            ))
        return sigmas, phases

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

        # More than one shape means the sampler is running on a packed latent -- video plus
        # audio for MiniMax H3. The video stream is always the first.
        latent_shapes = kwargs.get("latent_shapes") or [noise.shape]
        latent_shape = latent_shapes[0]
        packed = len(latent_shapes) > 1

        reference = self._video_reference(latent_shape)

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
        # The load device, not noise.device: noise is still on the CPU here, and unlike noise,
        # latent_image and sigmas, outer_sample never moves denoise_mask -- Comfy normally
        # places it in CFGGuider.sample via prepare_mask, which has already run by now.
        device = guider.model_patcher.load_device
        if packed:
            latent_image = _ttm_replace_video_reference(latent_image, reference, latent_shapes)
            sigmas, phases = self._packed_plan(model, latent_shapes, sigmas, device)
        else:
            latent_image = reference
            sigmas, phases = self._single_stream_plan(model, latent_shape, sigmas, device)

        if not phases:
            # Nothing to hold: the reference only seeds the init.
            return executor(noise, latent_image, sampler, sigmas, None, *args, **kwargs)

        # One schedule object, so the hold and its release cannot drift apart.
        schedule = _TTMDenoiseMaskSchedule(phases)
        guider.model_options["denoise_mask_function"] = schedule
        if packed and _ttm_can_sync_minimax_token_clock(model):
            # Added to the guider's model_options rather than the patcher: the patcher's
            # wrappers were merged into these options before outer_sample ran, and the masks
            # this needs only exist now. CFGGuider.sample restores them afterwards.
            comfy.patcher_extension.add_wrapper_with_key(
                WrappersMP.APPLY_MODEL,
                _TTM_APPLY_MODEL_KEY,
                _TTMMinimaxModelMaskWrapper(schedule, latent_shapes),
                guider.model_options,
                is_model_options=True,
            )
        denoise_mask = phases[0][1]

        logger.debug(
            "vloTimeToMove: latent_shapes=%s reference=%s sampled_latent=%s mask_frames=%d "
            "steps=%d denoise_mask=%s hold_until_sigma=%s",
            [tuple(shape) for shape in latent_shapes],
            tuple(reference.shape),
            tuple(latent_image.shape),
            self.mask.shape[0] if self.mask.ndim == 3 else 1,
            sigmas.shape[-1] - 1,
            tuple(denoise_mask.shape),
            [sigma_floor for sigma_floor, _ in phases],
        )

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
                "the sampled latent is replaced. On MiniMax H3 the reference replaces the "
                "video stream only and the audio stream is never held: it runs the full "
                "schedule, including the steps that seed the reference into the video. The "
                "held video tokens' own diffusion clock is released at end_step along with "
                "the sampler-side hold."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "reference_latents",
                    tooltip=(
                        "Encoded reference video, e.g. the cut-and-drag clip. Must match the "
                        "resolution and frame count being sampled. This replaces whatever "
                        "latent is wired into the sampler. On MiniMax H3 this is the video "
                        "latent; the sampler's own audio latent is kept."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "White marks the region whose motion the reference dictates: it is "
                        "held to the reference through the TTM window. Black is left free for "
                        "the model to generate. For a cut-and-drag clip that means white over "
                        "the dragged subject along its path, and black everywhere else, so the "
                        "model fills in the background it was lifted from. Pixel resolution "
                        "and frame count are matched to the latent grid automatically."
                    ),
                ),
                io.Int.Input(
                    "start_step",
                    default=1,
                    min=0,
                    max=1000,
                    tooltip=(
                        "Step whose noise level the reference is seeded at. Higher values "
                        "leave less noise on the reference, binding the result more tightly "
                        "to it at the cost of the model's freedom to clean up paste "
                        "artifacts. 0 is a no-op: sigma is 1.0 there and the reference washes "
                        "out entirely. On a single-stream model the sampler skips the steps "
                        "before it; on MiniMax H3 it runs them with the video held to the "
                        "reference instead, because the audio stream shares the schedule and "
                        "needs its opening steps."
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
