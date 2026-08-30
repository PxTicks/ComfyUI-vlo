"""Nodes for per-guide-token masked conditioning on MiniMax H3.

`vloMiniMaxH3AddMaskedGuide` is `MiniMaxH3AddGuide` plus a spatial confidence
map: mask 1 means "trust this part of the guide", mask 0 means "make this part
of the guide maximally unreliable". Nothing happens until the model is patched
with `vloMiniMaxH3PatchMaskedGuides`, which installs the forked forward pass
that actually reads the map.

Note the mask convention is *not* ComfyUI's denoise-mask one. Here the mask is
guide confidence, so 1 = strong guide, which is the opposite of "1 = generate".
"""

from __future__ import annotations

import torch

import comfy.patcher_extension
from comfy_api.latest import io

from .clips import CHUNK_ALIGN_MODES, TIME_POOLING_MODES
from .compatibility import check_core_compatible, check_semantic_supported
from .guides import build_still_guide, build_video_guides
from .masked_h3_forward import MASKED_GUIDE_KEY, make_diffusion_model_wrapper
from .masks import (
    DEFAULT_GUIDE_CLOCK,
    GUIDE_CLOCKS,
    MASK_LEVELS,
    check_guide_clock,
    check_mask_matches_image,
    guide_token_strengths,
    resize_mask_to_canvas,
)
from .semantic import (
    DEFAULT_PRESENTATION,
    DEFAULT_SAMPLE_FPS,
    PRESENTATIONS,
    SEMANTIC_MODES,
    clip_with_semantic_items,
    describe_items,
    semantic_items,
)

WRAPPER_KEY = "vlo_minimax_h3_masked_guides"

_MASK_TOOLTIP = ("Guide confidence, not a denoise mask: 1 keeps the guide at full strength, "
                 "0 corrupts it to noise, values in between blend continuously. Must frame "
                 "the same crop as the guide image.")


def _h3_helpers():
    # Keep MiniMax's model stack off this pack's import path, the way nodes/minimax.py does.
    try:
        from comfy_extras.nodes_minimax_h3 import FPS, FRAME_PER_TOKEN, _resize
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "The native MiniMax H3 nodes are unavailable. Update ComfyUI before using "
            "the masked-guide nodes."
        ) from exc
    return FRAME_PER_TOKEN, _resize, FPS


def _av_video_latent(latent):
    """The video stream of an H3 AV latent, with the whole pair's contract checked.

    Both streams are validated, not just the one this returns: a malformed audio
    stream would otherwise fail somewhere deep in sampling, and a node pack whose
    stated policy is to fail loudly should not be the thing that lets it through.
    """
    samples = latent["samples"]
    if not samples.is_nested or len(samples.tensors) != 2:
        raise ValueError("this node expects a MiniMax H3 AV latent (a video/audio nested pair)")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            "this node expects a MiniMax H3 AV latent; its video stream should be "
            "[B, 24, T, H/16, W/16], got {}".format(tuple(video.shape)))
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "this node expects a MiniMax H3 AV latent; its audio stream should be "
            "[B, 32, 2, T], got {}".format(tuple(audio.shape)))
    if video.shape[0] != audio.shape[0]:
        raise ValueError(
            "MiniMax H3 AV latent streams disagree on batch size: video {}, audio {}".format(
                video.shape[0], audio.shape[0]))
    return video


def _canvas(video):
    return video.shape[4] * 16, video.shape[3] * 16  # width, height


def _target_shape(latent):
    """(width, height, frame_count) of an H3 AV latent."""
    video = _av_video_latent(latent)
    frame_per_token, _, _ = _h3_helpers()
    width, height = _canvas(video)
    return width, height, sum(frame_per_token[k % 5] for k in range(video.shape[2]))


def _append_keyframes(positive, spec):
    """Add a spec's chunks to the conditioning's `minimax_keyframes`, keeping any already there."""
    import node_helpers

    keyframes = list(positive[0][1].get("minimax_keyframes", []))
    keyframes.extend(chunk.keyframe(MASKED_GUIDE_KEY) for chunk in spec.chunks)
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


GuideSpecType = io.Custom("VLO_H3_GUIDE_SPEC")


class vloMiniMaxH3AddMaskedGuide(io.ComfyNode):
    """Anchor an image guide at a frame with a continuous spatial strength mask."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3AddMaskedGuide",
            display_name="MiniMax H3 Add Masked Guide (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: anchor an image guide at any frame of a MiniMax H3 video, "
                "weighted by a spatial confidence mask. Needs the model to be patched with "
                "MiniMax H3 Patch Masked Guides; without that patch the mask is ignored and "
                "the guide behaves like a stock Add Guide."),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Image.Input("image", tooltip="Single guide image. Guide clips are not supported yet."),
                io.Mask.Input("mask", tooltip=_MASK_TOOLTIP),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the guide at. Negative values count from the end."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Scales the whole mask. 1.0 leaves a fully open mask identical to a stock guide."),
                io.Float.Input("min_aug", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="Condition noise-augmentation coefficient a mask value of 0 maps to. "
                                       "0.0 replaces those guide tokens with pure noise; raise it to keep a floor of guidance."),
                io.Float.Input("mask_gamma", default=1.0, min=0.1, max=5.0, step=0.05,
                               tooltip="Exponent applied to the mask before it becomes strength. "
                                       ">1 pushes mid-tones toward weak guidance, <1 toward strong."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, vae, image, mask, frame_idx,
                strength=1.0, min_aug=0.0, mask_gamma=1.0) -> io.NodeOutput:
        check_core_compatible()
        frame_per_token, resize, _ = _h3_helpers()

        video = _av_video_latent(latent)
        width, height = _canvas(video)
        frame_count = sum(frame_per_token[k % 5] for k in range(video.shape[2]))

        spec = build_still_guide(
            vae=vae, resize=resize, image=image, mask=mask, width=width, height=height,
            frame_count=frame_count, frame_idx=frame_idx, strength=strength,
            min_aug=min_aug, gamma=mask_gamma)
        return io.NodeOutput(_append_keyframes(positive, spec))


class vloMiniMaxH3AddMaskedGuidesFromVideo(io.ComfyNode):
    """Cut a masked video into guide clips and anchor each one as a masked guide."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3AddMaskedGuidesFromVideo",
            display_name="MiniMax H3 Add Masked Guides from Video (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: weight a whole video by a per-frame mask and anchor the result "
                "as several masked guide clips. Frames whose mask is empty guide nothing and "
                "are dropped, each surviving run becomes one guide, and each run is rounded "
                "down to a length MiniMax H3 accepts (1, 5, 22, 39, ... frames). Needs the "
                "model to be patched with MiniMax H3 Patch Masked Guides; without that patch "
                "the masks are ignored and the clips behave like stock guides."),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Image.Input("video", tooltip="Guide frames, at the target video's frame rate."),
                io.Mask.Input("mask", tooltip="One mask per guide frame (a single mask is applied to all of them). "
                                              + _MASK_TOOLTIP),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Target frame the guide video's first frame lines up with. "
                                     "Negative values count from the end. Guide frames that fall outside "
                                     "the target video are dropped."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Scales every mask. 1.0 leaves a fully open mask identical to a stock guide clip."),
                io.Float.Input("min_aug", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="Condition noise-augmentation coefficient a mask value of 0 maps to. "
                                       "0.0 replaces those guide tokens with pure noise; raise it to keep a floor of guidance."),
                io.Float.Input("mask_gamma", default=1.0, min=0.1, max=5.0, step=0.05,
                               tooltip="Exponent applied to the mask before it becomes strength. "
                                       ">1 pushes mid-tones toward weak guidance, <1 toward strong."),
                io.Float.Input("min_coverage", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="A frame is dropped when its mask covers no more than this fraction of "
                                       "the canvas, measured after the crop that fits the guide to the target's "
                                       "framing. 0.0 drops exactly the frames whose mask is empty; raise it to "
                                       "also drop frames where the subject is barely visible."),
                io.Combo.Input("time_pooling", options=list(TIME_POOLING_MODES), default="average",
                               tooltip="How the masks of the frames behind one latent token are combined. "
                                       "'average' matches the spatial pooling; 'max' takes their union, which "
                                       "is what a subject moving across those frames needs."),
                io.Combo.Input("chunk_align", options=list(CHUNK_ALIGN_MODES), default="start",
                               tooltip="Rounding a run down to a valid clip length drops frames: 'start' keeps "
                                       "the head of the run, 'center' keeps its middle."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"),
                     io.String.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, positive, latent, vae, video, mask, frame_idx,
                strength=1.0, min_aug=0.0, mask_gamma=1.0, min_coverage=0.0,
                time_pooling="average", chunk_align="start") -> io.NodeOutput:
        check_core_compatible()
        frame_per_token, resize, _ = _h3_helpers()

        target = _av_video_latent(latent)
        width, height = _canvas(target)
        frame_count = sum(frame_per_token[k % 5] for k in range(target.shape[2]))

        spec, plan = build_video_guides(
            vae=vae, resize=resize, video=video, mask=mask, width=width, height=height,
            frame_count=frame_count, frame_idx=frame_idx, strength=strength, min_aug=min_aug,
            gamma=mask_gamma, min_coverage=min_coverage, time_pooling=time_pooling,
            chunk_align=chunk_align)
        return io.NodeOutput(_append_keyframes(positive, spec), plan)

_SPEC_TOOLTIP = ("Guide plan from a Build Guide Spec node. It must be built from the same "
                 "latent this graph samples, so the crops, clip lengths and frame indices "
                 "the two conditioning paths use are the same ones.")

_OPTIONAL_MASK_TOOLTIP = (_MASK_TOOLTIP + " Leave unconnected for a full-confidence guide, "
                          "which is what semantic conditioning needs.")


class vloMiniMaxH3BuildGuideSpec(io.ComfyNode):
    """Plan a still guide once, for both the latent and the semantic path."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3BuildGuideSpec",
            display_name="MiniMax H3 Build Guide Spec (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: align, encode and weight a still guide, without touching the "
                "conditioning yet. Feed the result to Add Guides from Spec (the latent guide) "
                "and/or Apply Semantic Guides (the Qwen presentation), so both describe the "
                "same pixels at the same moment. Take the latent from an Empty MiniMax H3 AV "
                "Latent and give the conditioning node the same width/height/length."),
            inputs=[
                io.Latent.Input("latent", tooltip="The AV latent this guide is planned against."),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Image.Input("image", tooltip="Single guide image."),
                io.Mask.Input("mask", optional=True, tooltip=_OPTIONAL_MASK_TOOLTIP),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the guide at. Negative values count from the end."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Scales the whole mask. 1.0 leaves a fully open mask identical to a stock guide."),
                io.Float.Input("min_aug", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="Condition noise-augmentation coefficient a mask value of 0 maps to."),
                io.Float.Input("mask_gamma", default=1.0, min=0.1, max=5.0, step=0.05,
                               tooltip="Exponent applied to the mask before it becomes strength."),
            ],
            outputs=[GuideSpecType.Output(display_name="guide_spec")],
        )

    @classmethod
    def execute(cls, latent, vae, image, frame_idx, mask=None, strength=1.0, min_aug=0.0,
                mask_gamma=1.0) -> io.NodeOutput:
        check_core_compatible()
        _, resize, _ = _h3_helpers()
        width, height, frame_count = _target_shape(latent)
        return io.NodeOutput(build_still_guide(
            vae=vae, resize=resize, image=image, mask=mask, width=width, height=height,
            frame_count=frame_count, frame_idx=frame_idx, strength=strength,
            min_aug=min_aug, gamma=mask_gamma))


class vloMiniMaxH3BuildGuideSpecFromVideo(io.ComfyNode):
    """Plan a masked video's guide clips once, for both conditioning paths."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3BuildGuideSpecFromVideo",
            display_name="MiniMax H3 Build Guide Spec from Video (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: cut a masked video into guide clips and align, encode and weight "
                "each one, without touching the conditioning yet. Frames whose mask is empty are "
                "dropped, each surviving run becomes one guide, and each run is rounded down to a "
                "length MiniMax H3 accepts (1, 5, 22, 39, ... frames). Feed the result to Add "
                "Guides from Spec and/or Apply Semantic Guides."),
            inputs=[
                io.Latent.Input("latent", tooltip="The AV latent these guides are planned against."),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Image.Input("video", tooltip="Guide frames at 24 fps."),
                io.Mask.Input("mask", optional=True, tooltip="One mask per guide frame. " + _OPTIONAL_MASK_TOOLTIP),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Target frame the guide video's first frame lines up with."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Scales the whole mask. 1.0 leaves a fully open mask identical to a stock guide."),
                io.Float.Input("min_aug", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="Condition noise-augmentation coefficient a mask value of 0 maps to."),
                io.Float.Input("mask_gamma", default=1.0, min=0.1, max=5.0, step=0.05,
                               tooltip="Exponent applied to the mask before it becomes strength."),
                io.Float.Input("min_coverage", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Fraction of the canvas a frame's mask must cover to be worth guiding with."),
                io.Combo.Input("time_pooling", options=list(TIME_POOLING_MODES), default="average",
                               tooltip="How the frames behind one latent time token combine."),
                io.Combo.Input("chunk_align", options=list(CHUNK_ALIGN_MODES), default="start",
                               tooltip="Which end of a run clip-length rounding keeps."),
            ],
            outputs=[GuideSpecType.Output(display_name="guide_spec"),
                     io.String.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, latent, vae, video, frame_idx, mask=None, strength=1.0, min_aug=0.0,
                mask_gamma=1.0, min_coverage=0.0, time_pooling="average",
                chunk_align="start") -> io.NodeOutput:
        check_core_compatible()
        _, resize, _ = _h3_helpers()
        width, height, frame_count = _target_shape(latent)
        spec, plan = build_video_guides(
            vae=vae, resize=resize, video=video, mask=mask, width=width, height=height,
            frame_count=frame_count, frame_idx=frame_idx, strength=strength, min_aug=min_aug,
            gamma=mask_gamma, min_coverage=min_coverage, time_pooling=time_pooling,
            chunk_align=chunk_align)
        return io.NodeOutput(spec, plan)


class vloMiniMaxH3AddGuidesFromSpec(io.ComfyNode):
    """Anchor a planned spec's guides on the conditioning (the DiT / VAE path)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3AddGuidesFromSpec",
            display_name="MiniMax H3 Add Guides from Spec (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: add a guide spec's clips to the conditioning as masked guides. "
                "This is the latent half of a guide; Apply Semantic Guides is the Qwen half. "
                "Needs the model to be patched with MiniMax H3 Patch Masked Guides for the "
                "masks to do anything."),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="The AV latent being sampled; checked against the spec."),
                GuideSpecType.Input("guide_spec", tooltip=_SPEC_TOOLTIP),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, guide_spec) -> io.NodeOutput:
        check_core_compatible()
        width, height, frame_count = _target_shape(latent)
        guide_spec.check_target(width, height, frame_count)
        return io.NodeOutput(_append_keyframes(positive, guide_spec))


class vloMiniMaxH3ApplySemanticGuides(io.ComfyNode):
    """Present a spec's full-confidence guides to Qwen, timestamped (the semantic path)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3ApplySemanticGuides",
            display_name="MiniMax H3 Apply Semantic Guides (experimental)",
            category="model/conditioning/minimax",
            description=(
                "Experimental: also show a guide to MiniMax H3's Qwen encoder, as a timestamped "
                "<Video k> reference at the guide's own position in the generated video. Wire "
                "this between the CLIP loader and the conditioning node -- Qwen runs inside that "
                "node, so this has to come first. Only guides whose every condition token is at "
                "full confidence qualify: a masked or weakened guide is telling the model part of "
                "it is unreliable, and Qwen has no way to represent that. Adds no reference "
                "latents and no PackedLayout rows."),
            inputs=[
                io.Clip.Input("clip", tooltip="MiniMax H3 CLIP, on its way to the conditioning node."),
                GuideSpecType.Input("guide_spec", tooltip=_SPEC_TOOLTIP),
                io.Combo.Input("semantic_conditioning", options=list(SEMANTIC_MODES), default="auto",
                               tooltip="'off' passes the CLIP straight through. 'auto' presents every "
                                       "guide whose final token strengths are all exactly 1."),
                io.Combo.Input("presentation", options=list(PRESENTATIONS), default=DEFAULT_PRESENTATION,
                               tooltip="How several guide clips are labelled. 'merged' is one <Video k> "
                                       "whose timestamps jump over the gaps -- one subject seen at "
                                       "intervals. 'separate' is one <Video k> each, which reads as "
                                       "several different videos but lets the prompt address them "
                                       "individually. Neither emits anything for the gaps."),
                io.Float.Input("sample_fps", default=DEFAULT_SAMPLE_FPS, min=0.1, max=24.0, step=0.1,
                               tooltip="How densely a guide clip is sampled for Qwen. 2.0 is core's "
                                       "reference-video rate, but H3 guide clips are 0.2-1.6s long, so "
                                       "at 2 fps a clip yields one or two frames. Each sampled pair "
                                       "costs about a thousand tokens on the text span."),
            ],
            outputs=[io.Clip.Output(display_name="clip"),
                     io.String.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, clip, guide_spec, semantic_conditioning="auto",
                presentation=DEFAULT_PRESENTATION, sample_fps=DEFAULT_SAMPLE_FPS) -> io.NodeOutput:
        if semantic_conditioning not in SEMANTIC_MODES:
            raise ValueError("unknown semantic_conditioning {!r}, expected one of {}".format(
                semantic_conditioning, SEMANTIC_MODES))
        if semantic_conditioning == "off":
            return io.NodeOutput(clip, "semantic conditioning off; the CLIP is unchanged")

        check_semantic_supported()
        _, _, fps = _h3_helpers()
        eligible = [chunk for chunk in guide_spec.chunks if chunk.semantic_eligible]
        if not eligible:
            # Silently conditioning on nothing is the failure this pack refuses to make;
            # the mask that disqualified the guide is the thing worth naming.
            raise ValueError(
                "none of the {} guide(s) in this spec is at full confidence, so none can be "
                "shown to Qwen: their token strengths peak at {}. Semantic conditioning needs "
                "every token of a guide at exactly 1 -- an unconnected or fully open mask with "
                "strength 1.0. Set semantic_conditioning to 'off' to pass the CLIP through "
                "instead.".format(
                    len(guide_spec.chunks),
                    ", ".join("{:.3f}".format(float(c.strengths.max())) for c in guide_spec.chunks)))

        items = semantic_items(eligible, fps=fps, sample_fps=sample_fps, presentation=presentation)
        skipped = len(guide_spec.chunks) - len(eligible)
        report = describe_items(items, height=guide_spec.height, width=guide_spec.width)
        if skipped:
            report += "\n{} guide(s) held back: not at full confidence".format(skipped)
        return io.NodeOutput(clip_with_semantic_items(clip, items), report)


class vloMiniMaxH3PatchMaskedGuides(io.ComfyNode):
    """Install the forked H3 forward pass that reads masked-guide strengths."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3PatchMaskedGuides",
            display_name="MiniMax H3 Patch Masked Guides (experimental)",
            category="model/advanced/minimax",
            description=(
                "Experimental: routes MiniMax H3 sampling through a forked forward pass that "
                "corrupts each guide token by its mask value and labels it with a matching "
                "condition timestep. Samples without a masked guide run the stock path."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("guide_clock", options=list(GUIDE_CLOCKS), default=DEFAULT_GUIDE_CLOCK,
                               tooltip="How a guide token's confidence becomes a condition timestep. "
                                       "'stock': corrupt the latent only, every guide row keeps one global "
                                       "timestep -- the baseline this feature has to beat. "
                                       "'floored': label each token max(t_v, a), core's guard carried over; "
                                       "a token holding pure noise ends up labelled as clean as the target has "
                                       "become. 'matched': label each token as noisy as it actually is. "
                                       "'target_relative': a zero-confidence token sits level with the target "
                                       "instead of at pure noise, so it carries no *marginal* information -- "
                                       "core's own denoise-mask row formula, read backwards."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log one masked-guide report per sampling run."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, guide_clock=DEFAULT_GUIDE_CLOCK, debug=False) -> io.NodeOutput:
        check_core_compatible()
        if isinstance(guide_clock, bool):
            # Tolerate the boolean `sync_timesteps` this input replaced, so an API
            # caller carrying the old argument lands on the arm it used to mean.
            guide_clock = "floored" if guide_clock else "stock"
        check_guide_clock(guide_clock)
        patched = model.clone()
        # re-patching replaces rather than stacks, so chaining the node twice is harmless
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            make_diffusion_model_wrapper(clock=guide_clock, debug=debug),
        )
        return io.NodeOutput(patched)


class vloMiniMaxH3GuideTokenMaskPreview(io.ComfyNode):
    """Render the strength grid the DiT will actually see, one pixel block per guide token."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3GuideTokenMaskPreview",
            display_name="MiniMax H3 Guide Token Mask Preview",
            category="model/conditioning/minimax",
            description=(
                "Pools a guide mask onto H3's condition-token grid exactly as the masked-guide "
                "node does, then blows it back up with nearest-neighbour so mask alignment and "
                "quantization are visible."),
            inputs=[
                io.Latent.Input("latent", tooltip="The target AV latent, for the canvas size."),
                io.Mask.Input("mask", tooltip=_MASK_TOOLTIP),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("mask_gamma", default=1.0, min=0.1, max=5.0, step=0.05),
            ],
            outputs=[io.Image.Output(), io.Mask.Output(display_name="token_mask")],
        )

    @classmethod
    def execute(cls, latent, mask, strength=1.0, mask_gamma=1.0) -> io.NodeOutput:
        video = _av_video_latent(latent)
        width, height = _canvas(video)
        token_h = int(video.shape[3]) // 2
        token_w = int(video.shape[4]) // 2
        grid = guide_token_strengths(
            mask, width=width, height=height, token_t=1, token_h=token_h, token_w=token_w,
            strength=strength, gamma=mask_gamma, pooling="average", levels=MASK_LEVELS,
        ).to(torch.float32).reshape(1, 1, token_h, token_w)
        blown = torch.nn.functional.interpolate(grid, size=(height, width), mode="nearest")
        return io.NodeOutput(blown[0, 0][None, ..., None].repeat(1, 1, 1, 3), blown[0])


class vloMiniMaxH3MaskedGuidePixelFill(io.ComfyNode):
    """Baseline: mask the guide in pixel space before it ever reaches the VAE."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="vloMiniMaxH3MaskedGuidePixelFill",
            display_name="MiniMax H3 Masked Guide: Pixel Fill (baseline)",
            category="model/conditioning/minimax",
            description=(
                "The cheap baseline the token-masked guide has to beat: replace everything "
                "outside the mask with a flat colour or noise, then feed the result to a stock "
                "Add Guide. No model patch involved."),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", tooltip="1 keeps the image, 0 is replaced by the fill."),
                io.Combo.Input("fill", options=["gray", "black", "white", "noise"], default="gray"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Only used by the noise fill."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, mask, fill="gray", seed=0) -> io.NodeOutput:
        check_mask_matches_image(mask, image)
        img = image[..., :3].to(torch.float32)
        m = resize_mask_to_canvas(mask, img.shape[-2], img.shape[-3], crop="disabled")
        if m.shape[0] == 1:
            m = m.expand(img.shape[0], -1, -1)
        elif m.shape[0] != img.shape[0]:
            # Broadcasting mask 0 across the batch would fill every image by the first
            # image's mask, which looks right and is not.
            raise ValueError(
                "received {} images and {} masks; pass one mask per image, or a single mask "
                "to apply to all of them".format(img.shape[0], m.shape[0]))
        m = m.to(img.device).unsqueeze(-1)
        if fill == "noise":
            gen = torch.Generator("cpu").manual_seed(int(seed))
            c = torch.rand(img.shape, generator=gen, dtype=torch.float32).to(img.device)
        else:
            c = torch.full_like(img, {"gray": 0.5, "black": 0.0, "white": 1.0}[fill])
        return io.NodeOutput(m * img + (1.0 - m) * c)
