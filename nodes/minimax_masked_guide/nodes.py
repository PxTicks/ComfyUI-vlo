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

from .clips import (
    CHUNK_ALIGN_MODES,
    TIME_POOLING_MODES,
    clip_token_strengths,
    frame_keep_flags,
    frames_in_latent_t,
    frames_inside_target,
    normalize_mask_batch,
    plan_video_guides,
)
from .compatibility import check_core_compatible
from .masked_h3_forward import MASKED_GUIDE_KEY, make_diffusion_model_wrapper
from .masks import (
    MASK_LEVELS,
    check_mask_matches_image,
    guide_token_strengths,
    resize_mask_to_canvas,
)

WRAPPER_KEY = "vlo_minimax_h3_masked_guides"

_MASK_TOOLTIP = ("Guide confidence, not a denoise mask: 1 keeps the guide at full strength, "
                 "0 corrupts it to noise, values in between blend continuously. Must frame "
                 "the same crop as the guide image.")


def _h3_helpers():
    # Keep MiniMax's model stack off this pack's import path, the way nodes/minimax.py does.
    try:
        from comfy_extras.nodes_minimax_h3 import FRAME_PER_TOKEN, _resize
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "The native MiniMax H3 nodes are unavailable. Update ComfyUI before using "
            "the masked-guide nodes."
        ) from exc
    return FRAME_PER_TOKEN, _resize


def _av_video_latent(latent):
    samples = latent["samples"]
    if (not samples.is_nested or len(samples.tensors) != 2
            or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24):
        raise ValueError("this node expects a MiniMax H3 AV latent")
    return samples.tensors[0]


def _canvas(video):
    return video.shape[4] * 16, video.shape[3] * 16  # width, height


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
        import node_helpers

        check_core_compatible()
        frame_per_token, resize = _h3_helpers()

        video = _av_video_latent(latent)
        width, height = _canvas(video)
        frame_count = sum(frame_per_token[k % 5] for k in range(video.shape[2]))

        if image.shape[0] != 1:
            # Core's AddGuide would read a >= 5 frame batch as a guide clip and a shorter
            # one as its first frame; neither is a thing this node can weight with one
            # mask, so both are refused rather than quietly truncated.
            raise ValueError(
                "masked guides support single-image guides only; received a batch of {} "
                "images. Guide clips (and their time-varying masks) are not supported yet"
                .format(image.shape[0]))
        check_mask_matches_image(mask, image)

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index >= frame_count:
            raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))

        guide_latent = vae.encode(resize(image, width, height, "center"))
        if guide_latent.ndim != 5 or guide_latent.shape[3] % 2 or guide_latent.shape[4] % 2:
            raise ValueError(
                "guide latent {} does not tile into H3's 2x2 condition patches".format(tuple(guide_latent.shape)))
        token_t = int(guide_latent.shape[2])
        token_h = int(guide_latent.shape[3]) // 2
        token_w = int(guide_latent.shape[4]) // 2

        strengths = guide_token_strengths(
            mask, width=width, height=height, token_t=token_t, token_h=token_h, token_w=token_w,
            strength=strength, gamma=mask_gamma, pooling="average", levels=MASK_LEVELS)

        keyframe = {
            "resolved_frame_index": resolved_frame_index,
            "latent": guide_latent,
            # Core ignores the extra key; only the patched forward reads it.
            MASKED_GUIDE_KEY: {
                "strengths": strengths,
                "strength": float(strength),
                "min_aug": float(min_aug),
                "gamma": float(mask_gamma),
                "token_t": token_t,
                "token_h": token_h,
                "token_w": token_w,
                "resolved_frame_index": resolved_frame_index,
            },
        }
        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        return io.NodeOutput(node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}))


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
                                       "the canvas. 0.0 drops exactly the frames whose mask is empty; raise it "
                                       "to also drop frames where the subject is barely visible."),
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
        import node_helpers

        check_core_compatible()
        frame_per_token, resize = _h3_helpers()

        target = _av_video_latent(latent)
        width, height = _canvas(target)
        frame_count = sum(frame_per_token[k % 5] for k in range(target.shape[2]))

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

        start = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        keep = frame_keep_flags(masks, min_coverage)
        chunks = plan_video_guides(keep, frame_idx=start, frame_count=frame_count,
                                   align=chunk_align)
        if not chunks:
            # Wiring up a video and a mask and silently adding no guidance at all is
            # exactly the kind of plausible-looking nothing this pack refuses to do.
            raise ValueError(
                "no guide clips survive: of {} guide frames anchored at frame {}, {} carry a "
                "mask above min_coverage {} and none of the runs they form reaches a frame that "
                "fits inside the target video's {} frames".format(
                    video.shape[0], start, int(keep.sum()), min_coverage, frame_count))

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        report = []
        total_tokens = 0
        for source_start, target_start, length in chunks:
            frames = video[source_start:source_start + length]
            guide_latent = vae.encode(resize(frames, width, height, "center"))
            if guide_latent.ndim != 5 or guide_latent.shape[3] % 2 or guide_latent.shape[4] % 2:
                raise ValueError(
                    "guide latent {} does not tile into H3's 2x2 condition patches".format(
                        tuple(guide_latent.shape)))
            token_t = int(guide_latent.shape[2])
            token_h = int(guide_latent.shape[3]) // 2
            token_w = int(guide_latent.shape[4]) // 2
            if frames_in_latent_t(token_t) != length:
                # The mask is pooled onto the latent's time grid by FRAME_PER_TOKEN, so a VAE
                # whose temporal compression differs would slide the mask against the frames.
                raise ValueError(
                    "the vae encoded {} guide frames into {} latent time tokens, which cover {} "
                    "frames on H3's grid; the mask cannot be aligned to that".format(
                        length, token_t, frames_in_latent_t(token_t)))

            strengths = clip_token_strengths(
                masks[source_start:source_start + length], width=width, height=height,
                token_t=token_t, token_h=token_h, token_w=token_w, strength=strength,
                gamma=mask_gamma, spatial_pooling="average", time_pooling=time_pooling,
                levels=MASK_LEVELS)

            keyframes.append({
                "resolved_frame_index": target_start,
                "latent": guide_latent,
                # Core ignores the extra key; only the patched forward reads it.
                MASKED_GUIDE_KEY: {
                    "strengths": strengths,
                    "strength": float(strength),
                    "min_aug": float(min_aug),
                    "gamma": float(mask_gamma),
                    "token_t": token_t,
                    "token_h": token_h,
                    "token_w": token_w,
                    "resolved_frame_index": target_start,
                },
            })
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
        guided = sum(length for _, _, length in chunks)
        plan = "\n".join(
            ["{} masked guide clip(s) covering {} of {} frames ({} dropped by the mask, "
             "{} outside the target video, {} to clip-length rounding)".format(
                 len(chunks), guided, int(video.shape[0]),
                 int(video.shape[0] - keep.sum()), int(keep.sum() - inside.sum()),
                 int(inside.sum()) - guided)]
            + report
            + ["condition tokens riding every sampling step: {}".format(total_tokens)])
        return io.NodeOutput(
            node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes}),
            plan)

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
                io.Boolean.Input("sync_timesteps", default=True,
                                 tooltip="On: each corrupted guide token also gets a matching condition "
                                         "timestep / AdaLN row. Off: only the latent is corrupted and the "
                                         "guide keeps one global timestep -- the baseline this feature is "
                                         "meant to beat."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log one masked-guide report per sampling run."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, sync_timesteps=True, debug=False) -> io.NodeOutput:
        check_core_compatible()
        patched = model.clone()
        # re-patching replaces rather than stacks, so chaining the node twice is harmless
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            make_diffusion_model_wrapper(sync_timesteps=sync_timesteps, debug=debug),
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
