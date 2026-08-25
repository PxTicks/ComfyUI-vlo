"""A fork of `MiniMaxH3Model._forward` that honours per-token guide strengths.

Forked from ComfyUI `comfy/ldm/minimax/model.py` -- see `compatibility.py` for
the exact commit it was taken from and re-run `tests/test_forward_equivalence.py`
after every ComfyUI update. The fork exists because the values that need
changing (`seg_t`, `unique_t`, `t_row`, `mod_segments`, `cond_video_rows`) are
locals inside core's `_forward`; there is no seam to hook.

Everything below is core's code verbatim except three blocks marked `vlo:`

  A. per-condition-row noise-augmentation coefficients, built from the guide masks
  B. `_cond_video_rows` mixes each row with noise at *its own* coefficient
  C. `cond` segments get per-token modulation-row indices, the way PR #15375
     already does for target `video`/`audio` rows

With every guide mask fully open this file must reproduce stock output bit for
bit: the strength -> coefficient map pins its endpoints, and a condition segment
whose rows all share one timestep collapses back onto core's scalar path.
"""

from __future__ import annotations

import logging

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension

from .compatibility import h3_module, is_h3_diffusion_model
from .masks import aug_to_cond_timestep, strengths_to_aug


# Key under which `vloMiniMaxH3AddMaskedGuide` stores its spec on a keyframe dict.
# Core ignores unknown keyframe keys, so masked guides still travel through
# stock layout construction, VAE handling and payload assembly.
MASKED_GUIDE_KEY = "vlo_masked_guide"


def _latent_cond_rows(latent) -> int:
    """Condition rows one visual condition latent contributes, at H3's 1x2x2 patching."""
    return int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)


def _visual_conditions(payload):
    """(latent, masked_spec, is_keyframe) for every visual condition, in cond-row order.

    Order matters more than anything else here: `model_base` builds
    `cond_video_latents` as keyframe latents followed by reference latents, and
    `PackedLayout` lays their rows down in that same order. Rebuilding the list
    from `keyframes`/`refs` rather than from `cond_video_latents` is what lets a
    strength vector be attributed to the right guide; the length check below
    catches any drift between the two.
    """
    out = []
    for keyframe in payload.get("keyframes") or ():
        latent = keyframe.get("latent")
        if latent is not None:
            out.append((latent, keyframe.get(MASKED_GUIDE_KEY), True))
    for ref in payload.get("refs") or ():
        if "latent" in ref:  # model_base's own test, so the two lists cannot drift
            out.append((ref["latent"], None, False))  # references keep stock behaviour

    expected = payload.get("cond_video_latents") or []
    if len(out) != len(expected):
        raise RuntimeError(
            "masked guide row alignment failed: {} visual conditions reconstructed from "
            "keyframes/refs, but the payload carries {} condition latents".format(len(out), len(expected)))
    return out


def has_masked_guides(payload) -> bool:
    if not payload:
        return False
    return any(isinstance(kf, dict) and kf.get(MASKED_GUIDE_KEY)
               for kf in (payload.get("keyframes") or ()))


class CondRowPlan:
    """Per-condition-row augmentation coefficients and modulation timesteps.

    `aug_rows` spans every visual condition row (keyframes then references);
    `segment_rows_t` has one entry per `cond` segment of the packed layout, i.e.
    per keyframe that carries a latent, and is None where that guide runs stock.
    """

    def __init__(self, aug_rows, segment_rows_t, report):
        self.aug_rows = aug_rows
        self.segment_rows_t = segment_rows_t
        self.report = report


def build_cond_row_plan(payload, t_v, vis_aug, sync_timesteps=True):
    """vlo: change A -- flatten the guide masks into per-condition-row values."""
    conditions = _visual_conditions(payload)

    aug_parts = []
    segment_rows_t = []
    report = []
    for index, (latent, spec, is_keyframe) in enumerate(conditions):
        n_rows = _latent_cond_rows(latent)
        rows_t = None
        if spec is None:
            aug = torch.full((n_rows,), float(vis_aug), dtype=torch.float64)
        else:
            strengths = spec["strengths"].to(torch.float64).reshape(-1)
            if strengths.numel() != n_rows:
                raise RuntimeError(
                    "masked guide row alignment failed: guide {} carries {} token strengths "
                    "but its condition latent {} patchifies to {} rows".format(
                        index, strengths.numel(), tuple(latent.shape), n_rows))
            min_aug = min(max(float(spec.get("min_aug", 0.0)), 0.0), float(vis_aug))
            aug = strengths_to_aug(strengths, a_max=float(vis_aug), a_min=min_aug)
            rows_t = aug_to_cond_timestep(aug, t_v)
            report.append((index, spec, latent, strengths, aug))
        aug_parts.append(aug)
        if is_keyframe:
            segment_rows_t.append(rows_t if sync_timesteps else None)

    return CondRowPlan(torch.cat(aug_parts) if aug_parts else None, segment_rows_t, report)


def masked_cond_video_rows(model, payload, device, aug_rows):
    """vlo: change B -- core's `_cond_video_rows` with a per-row augmentation coefficient.

    Core mixes every condition row with noise at one scalar `a`; here each row
    gets its own. The coefficients are split into (a, 1-a) in float64 before
    being cast down, so a row sitting at the stock coefficient reproduces core's
    `aug * r + (1.0 - aug) * noise` exactly rather than one ulp away from it.

    Noise seeding is unchanged, deliberately: core restarts the same RNG stream
    for every condition, which is already fixed for the whole sample, so a
    guide's corrupted form never flickers between diffusion steps.
    """
    if aug_rows is None:
        return model._cond_video_rows(payload, device)
    module = h3_module()
    rows = []
    seed = int(payload.get("seed", 0))
    offset = 0
    # every condition intentionally restarts the same RNG stream
    for z in payload.get("cond_video_latents", []):
        r = module.patchify_video(z.to(torch.float32), model.patch_size)
        a64 = aug_rows[offset:offset + r.shape[0]].to(torch.float64).view(-1, 1)
        offset += r.shape[0]
        if bool((a64 < 1.0).any()):
            gen = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
            a = a64.to(torch.float32).to(r.device)
            b = (1.0 - a64).to(torch.float32).to(r.device)
            r = a * r + b * noise.to(r.device)
        rows.append(r.to(device))
    if offset != aug_rows.shape[0]:
        raise RuntimeError(
            "masked guide row alignment failed: {} augmentation values for {} condition "
            "rows".format(aug_rows.shape[0], offset))
    return torch.cat(rows, dim=0) if rows else None


def _log_plan(payload, plan, layout, debug):
    """One report per sampling run, not per step."""
    if not debug or payload.get("_vlo_masked_guide_logged"):
        return
    payload["_vlo_masked_guide_logged"] = True
    cond_rows = sum(b - a for a, b, kind in layout.segments if kind == "cond")
    for index, spec, latent, strengths, aug in plan.report:
        logging.info(
            "Masked H3 Guide %d:\n  frame: %s\n  latent: %s\n  token mask: [%d, %d]\n"
            "  token count: %d\n  strength min/mean/max: %.3f / %.3f / %.3f\n"
            "  unique aug levels: %d",
            index, spec.get("resolved_frame_index"), list(latent.shape),
            int(spec.get("token_h", 0)), int(spec.get("token_w", 0)), strengths.numel(),
            float(strengths.min()), float(strengths.mean()), float(strengths.max()),
            int(aug.unique().numel()))
    logging.info("Masked H3 Guides: cond rows expected: %d, cond mask rows produced: %d",
                 cond_rows, int(plan.aug_rows.shape[0]) if plan.aug_rows is not None else 0)


def masked_forward(model, x, timestep, context, transformer_options={}, minimax_payload=None,
                   denoise_mask=None, audio_denoise_mask=None,
                   sync_timesteps=True, debug=False, **kwargs):
    module = h3_module()
    patchify_video = module.patchify_video
    unpatchify_video = module.unpatchify_video
    pack_audio = module.pack_audio
    unpack_audio = module.unpack_audio

    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype  # compute dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    # extra_conds prebuilds the layout once per sampling run
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = module.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                     keyframes=payload.get("keyframes"),
                                     refs=payload.get("refs"))

    # model_base passes model_sampling.timestep(sigma) = sigma * 1000
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - module.time_shift_sigma(sigma_v, shift_v, shift_a))

    # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
    vis_aug = float(payload.get("visual_cond_noise_aug", module.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", module.AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}

    # vlo: change A -- per-condition-row augmentation coefficients and timesteps
    plan = build_cond_row_plan(payload, t_v, vis_aug, sync_timesteps=sync_timesteps)
    _log_plan(payload, plan, layout, debug)
    cond_seg_rows_t = list(plan.segment_rows_t)

    # masked rows run at their own strength: mask value m puts a row at sigma = m * sigma_stream,
    # so its label is 1 - m * sigma, clamped at the cond timestep for fully preserved rows
    t_pin_v = max(t_v, module.VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, module.AUDIO_COND_TIMESTEP)
    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        m = module.mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    # vlo: a guide whose rows all share one timestep goes back on the scalar path,
    # so a fully-open mask reproduces stock modulation exactly
    cond_seg_scalar_t = []
    for i, rows_t in enumerate(cond_seg_rows_t):
        if rows_t is not None and rows_t.unique().numel() == 1:
            cond_seg_scalar_t.append(float(rows_t[0]))
            cond_seg_rows_t[i] = None
        else:
            cond_seg_scalar_t.append(None)

    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set())
                      # vlo: change C -- guide rows contribute their own timesteps
                      | {t for t in cond_seg_scalar_t if t is not None}
                      | {t for rows_t in cond_seg_rows_t if rows_t is not None
                         for t in rows_t.unique().tolist()})
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    def rows_to_mod_index(rows_t, tag):
        # per-row timestep values -> per-row mod-row indices into the t_emb table
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                            dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    cond_index = 0
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
        elif kind == "cond":
            # vlo: change C -- one modulation row per guide token, paired with the
            # corruption that token's latent actually received
            rows_t = cond_seg_rows_t[cond_index]
            scalar_t = cond_seg_scalar_t[cond_index]
            cond_index += 1
            if rows_t is not None:
                if rows_t.shape[0] != b - a:
                    raise RuntimeError(
                        "masked guide row alignment failed: {} guide mask rows for a {} row "
                        "cond segment".format(rows_t.shape[0], b - a))
                mod_segments.append((a, b, rows_to_mod_index(rows_t.to(device), seg_tag[kind])))
            elif scalar_t is not None:
                mod_segments.append((a, b, t_row[scalar_t] * 3 + seg_tag[kind]))
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))
    if cond_index != len(cond_seg_rows_t):
        raise RuntimeError(
            "masked guide row alignment failed: {} cond segments in the packed layout, "
            "{} guide strength vectors".format(cond_index, len(cond_seg_rows_t)))

    # embed
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), model.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    # vlo: change B -- per-row condition noise augmentation
    cond_video_rows = masked_cond_video_rows(model, payload, device, plan.aug_rows)
    cond_audio_rows = model._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = model.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != model.hidden_size:
        text_states = model.token_refiner(model.condition_proj(text_states),
                                          transformer_options=transformer_options)

    # segments are contiguous: assemble by slices, embed rows follow segment order
    h = torch.empty(layout.seq_len, model.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:  # ref_audio / audio
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
    else:
        t_emb = model.time_embedder(t_vals).to(dtype)

    # rotation table computed once per forward, consumed by the kitchen split-half rope
    rope_freqs = module.rope_rotation_table(model.rope_freqs(layout.position_ids, device), dtype)

    # blocks
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(model.blocks), device, transformer_options)
    for i, block in enumerate(model.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    # target streams are single contiguous segments (audio then video, last two)
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
    if video_rows_t is not None:
        video_seg = (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
    else:
        video_seg = (va, vb, t_row[seg_t["video"]])
    if audio_rows_t is not None:
        audio_seg = (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
    else:
        audio_seg = (aa, ab, t_row[seg_t["audio"]])
    v, a = model.final_layer(h, t_emb, video_seg, audio_seg)

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, model.latents_dim, model.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)

    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def make_diffusion_model_wrapper(sync_timesteps=True, debug=False):
    """A `WrappersMP.DIFFUSION_MODEL` wrapper that only diverts masked-guide samples.

    On a masked sample this does *not* short-circuit the chain. It rebuilds the
    executor with the masked forward pass substituted for the innermost function
    and the remaining wrappers left in place, so anything registered after this
    node -- EasyCache, TeaCache, block-swap patches -- still runs. Returning
    `masked_forward(...)` directly would silently drop them and make the result
    depend on which order the model patch nodes happen to be chained in.
    """

    def wrapper(executor, *args, **kwargs):
        model = getattr(executor, "class_obj", None)
        payload = kwargs.get("minimax_payload")
        if not has_masked_guides(payload) or not is_h3_diffusion_model(model):
            return executor(*args, **kwargs)  # stock behaviour, untouched

        def replacement(*inner_args, **inner_kwargs):
            # stands in for the bound MiniMaxH3Model._forward, so it takes no self
            return masked_forward(model, *inner_args, sync_timesteps=sync_timesteps,
                                  debug=debug, **inner_kwargs)

        remaining = list(executor.wrappers)[executor.idx + 1:]
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            replacement, executor.class_obj, remaining).execute(*args, **kwargs)

    return wrapper
