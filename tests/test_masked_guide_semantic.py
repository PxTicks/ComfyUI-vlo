"""The semantic half of a guide: eligibility, pixel identity, timing, presentation.

The rule these tests exist to hold: Qwen is shown the guide's *actual* pixels at
the guide's *actual* moment, or it is shown nothing at all.
"""

from __future__ import annotations

import pytest
import torch

from minimax_h3_harness import comfyui_on_path, h3_model_module, masked_guide_module

FPS = 24.0


@pytest.fixture(scope="module")
def node_module():
    comfyui_on_path()
    return masked_guide_module("nodes")


@pytest.fixture(scope="module")
def guides():
    comfyui_on_path()
    return masked_guide_module("guides")


@pytest.fixture(scope="module")
def semantic():
    comfyui_on_path()
    return masked_guide_module("semantic")


@pytest.fixture(scope="module")
def clips():
    comfyui_on_path()
    return masked_guide_module("clips")


class _Vae:
    """Stands in for the H3 video VAE, recording exactly what it was handed."""

    def __init__(self, clips=None):
        self._clips = clips
        self.encoded = []

    def encode(self, frames):
        self.encoded.append(frames)
        n, h, w, _ = frames.shape
        if self._clips is None:
            latent_t = 1
        else:
            latent_t = next(t for t in range(1, n + 2) if self._clips.frames_in_latent_t(t) == n)
        return torch.zeros(1, 24, latent_t, h // 16, w // 16)


def _av_latent(width=128, height=64, latent_t=37, audio_t=5):
    import comfy.nested_tensor

    video = torch.zeros(1, 24, latent_t, height // 16, width // 16)
    audio = torch.zeros(1, 32, 2, audio_t)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


FRAME_COUNT = 124  # latent_t 37 on H3's (1, 4, 4, 4, 4) grid


def _still(node_module, mask=None, image=None, **kwargs):
    params = dict(latent=_av_latent(), vae=_Vae(), mask=mask, frame_idx=0,
                  image=torch.zeros(1, 64, 128, 3) if image is None else image)
    params.update(kwargs)
    return node_module.vloMiniMaxH3BuildGuideSpec.execute(**params).result[0]


def _video_and_mask(n, kept, value=1.0):
    masks = torch.zeros(n, 64, 128)
    masks[list(kept)] = value
    # each frame carries its own index as a constant, so a sampled frame can be traced
    video = (torch.arange(n, dtype=torch.float32) / 100.0).reshape(n, 1, 1, 1)
    return video.expand(n, 64, 128, 3).contiguous(), masks


def _from_video(node_module, clips, video, mask, **kwargs):
    params = dict(latent=_av_latent(), vae=_Vae(clips), video=video, mask=mask, frame_idx=0)
    params.update(kwargs)
    return node_module.vloMiniMaxH3BuildGuideSpecFromVideo.execute(**params).result[0]


# --- eligibility: only a genuinely complete observation qualifies ----------


def test_a_full_mask_at_full_strength_is_eligible(node_module):
    spec = _still(node_module, torch.ones(1, 64, 128))
    assert spec.chunks[0].semantic_eligible


def test_no_mask_at_all_is_eligible(node_module):
    """An unconnected mask is a full-confidence guide, not a missing one."""
    spec = _still(node_module, None)
    assert spec.chunks[0].semantic_eligible
    assert torch.equal(spec.chunks[0].strengths, torch.ones_like(spec.chunks[0].strengths))


def test_strength_below_one_is_not_eligible(node_module):
    """Qwen must not be able to bypass a deliberately weakened latent guide."""
    spec = _still(node_module, torch.ones(1, 64, 128), strength=0.99)
    assert not spec.chunks[0].semantic_eligible


def test_a_nearly_white_mask_is_not_eligible(node_module):
    mask = torch.ones(1, 64, 128)
    mask[:, :4, :8] = 0.0  # a small but semantically real hole
    spec = _still(node_module, mask)
    assert not spec.chunks[0].semantic_eligible


def test_a_feathered_mask_is_not_eligible(node_module):
    mask = torch.ones(1, 64, 128)
    mask[:, :, :16] = torch.linspace(0.0, 1.0, 16)
    spec = _still(node_module, mask)
    assert not spec.chunks[0].semantic_eligible


def test_gamma_on_a_full_mask_stays_eligible(node_module):
    """mask_gamma and min_aug need no separate check: 1 ** gamma is still 1."""
    spec = _still(node_module, torch.ones(1, 64, 128), mask_gamma=3.0, min_aug=0.4)
    assert spec.chunks[0].semantic_eligible


def test_a_temporally_partial_token_is_not_eligible(node_module, clips):
    """A token whose four frames are half covered is not a complete observation."""
    video, masks = _video_and_mask(22, range(22), value=0.5)
    spec = _from_video(node_module, clips, video, masks)
    assert not spec.chunks[0].semantic_eligible


def test_eligibility_reads_the_final_strengths_not_the_input_mask(node_module):
    """The quantizer is what makes an exact == 1.0 test safe on a resampled mask."""
    spec = _still(node_module, torch.ones(1, 64, 128))
    strengths = spec.chunks[0].strengths
    assert strengths.min().item() == 1.0 and strengths.dtype == torch.float64


# --- pixel identity: one alignment, shared by both paths ------------------


def test_the_semantic_frames_are_the_tensor_the_vae_encoded(node_module, semantic):
    vae = _Vae()
    spec = node_module.vloMiniMaxH3BuildGuideSpec.execute(
        latent=_av_latent(), vae=vae, image=torch.rand(1, 64, 128, 3),
        mask=torch.ones(1, 64, 128), frame_idx=0).result[0]
    chunk = spec.chunks[0]
    assert chunk.aligned_frames is vae.encoded[0]
    frames, _ = semantic.chunk_presentation(chunk, fps=FPS)
    assert torch.equal(frames[0], chunk.aligned_frames[0])


def test_qwen_never_sees_pixels_the_canvas_crop_removed(node_module, semantic):
    """A 1:1 source on a 2:1 canvas loses its top and bottom bands. The latent guide
    does not contain them, so the semantic presentation must not either."""
    image = torch.zeros(1, 64, 64, 3)
    image[:, :10, :, :] = 1.0  # a subject only in the discarded top band
    spec = _still(node_module, torch.ones(1, 64, 64), image=image)
    frames, _ = semantic.chunk_presentation(spec.chunks[0], fps=FPS)
    assert float(frames.max()) < 0.1


# --- timing: both channels read the same resolved frame index -------------


@pytest.mark.parametrize("frame_idx,resolved", [
    (0, 0),
    (60, 60),
    (FRAME_COUNT - 1, FRAME_COUNT - 1),
    (-1, FRAME_COUNT - 1),
    (-24, FRAME_COUNT - 24),
])
def test_the_timestamp_comes_from_the_resolved_frame_index(node_module, semantic,
                                                           frame_idx, resolved):
    spec = _still(node_module, torch.ones(1, 64, 128), frame_idx=frame_idx)
    chunk = spec.chunks[0]
    assert chunk.target_start == resolved
    _, timestamps = semantic.chunk_presentation(chunk, fps=FPS)
    # a still is padded to the tokenizer's two-frame block, both copies at its own time
    assert timestamps == [resolved / FPS, resolved / FPS]


def test_a_still_is_one_frame_the_tokenizer_will_pair_with_itself(node_module, semantic):
    spec = _still(node_module, torch.ones(1, 64, 128), frame_idx=48)
    items = semantic.semantic_items(spec.chunks, fps=FPS)
    assert len(items) == 1
    frames, stamps = items[0]["data"], items[0]["timestamps"]
    assert frames.shape[0] == 2 and stamps == [2.0, 2.0]
    assert torch.equal(frames[0], frames[1])
    # the tokenizer labels a block with the pair's mean, so a still lands exactly on its frame
    assert "%.1f" % ((stamps[0] + stamps[1]) / 2.0) == "2.0"


# --- guide clips: only what survives planning is presented ----------------


def test_frames_dropped_by_clip_length_rounding_are_absent(node_module, clips, semantic):
    """A 30 frame run becomes a 22 frame guide; the other 8 guide nothing."""
    video, masks = _video_and_mask(60, range(30))
    spec = _from_video(node_module, clips, video, masks)
    assert [(c.source_start, c.length) for c in spec.chunks] == [(0, 22)]
    frames, _ = semantic.chunk_presentation(spec.chunks[0], fps=FPS, sample_fps=FPS)
    seen = sorted(round(float(f.mean()) * 100) for f in frames)
    assert seen == list(range(22))


def test_frames_outside_the_target_are_absent(node_module, clips, semantic):
    video, masks = _video_and_mask(60, range(60))
    spec = _from_video(node_module, clips, video, masks, frame_idx=115)
    chunk = spec.chunks[0]
    assert (chunk.source_start, chunk.length, chunk.target_start) == (0, 5, 115)
    frames, timestamps = semantic.chunk_presentation(chunk, fps=FPS, sample_fps=FPS)
    assert sorted(round(float(f.mean()) * 100) for f in frames) == [0, 1, 2, 3, 4, 4]
    assert timestamps[0] == 115 / FPS


def test_timestamps_describe_the_generated_timeline_not_the_source_clip(node_module, clips,
                                                                       semantic):
    video, masks = _video_and_mask(22, range(22))
    spec = _from_video(node_module, clips, video, masks, frame_idx=48)
    _, timestamps = semantic.chunk_presentation(spec.chunks[0], fps=FPS)
    assert timestamps == [48 / FPS, 60 / FPS]  # source frames 0 and 12, anchored at 48


def test_only_full_strength_chunks_are_exposed(node_module, clips, semantic):
    """Eligibility is per chunk: one weakened run does not veto a complete one."""
    masks = torch.zeros(60, 64, 128)
    masks[0:22] = 1.0
    masks[30:52] = 0.5
    video = _video_and_mask(60, [])[0]
    spec = _from_video(node_module, clips, video, masks)
    assert [c.semantic_eligible for c in spec.chunks] == [True, False]


# --- presentation: merged vs separate -------------------------------------


def _two_run_spec(node_module, clips):
    video, masks = _video_and_mask(60, list(range(22)) + list(range(30, 52)))
    return _from_video(node_module, clips, video, masks)


def test_separate_gives_each_chunk_its_own_video_label(node_module, clips, semantic):
    spec = _two_run_spec(node_module, clips)
    items = semantic.semantic_items(spec.chunks, fps=FPS, presentation="separate")
    assert len(items) == 2
    assert [item["timestamps"][0] for item in items] == [0.0, 30 / FPS]


def test_merged_is_one_label_whose_timestamps_jump_the_gap(node_module, clips, semantic):
    spec = _two_run_spec(node_module, clips)
    items = semantic.semantic_items(spec.chunks, fps=FPS, presentation="merged")
    assert len(items) == 1
    stamps = items[0]["timestamps"]
    assert stamps == sorted(stamps)
    assert stamps == [0.0, 12 / FPS, 30 / FPS, 42 / FPS]
    # nothing at all is emitted for frames 22-29
    assert items[0]["data"].shape[0] == 4


def test_each_merged_chunk_is_padded_to_an_even_block_boundary(node_module, clips, semantic):
    """Blocks pair adjacent frames, so an odd chunk would fuse a cut into one patch."""
    masks = torch.zeros(60, 64, 128)
    masks[0:5] = 1.0
    masks[30:35] = 1.0
    video = _video_and_mask(60, [])[0]
    spec = _from_video(node_module, clips, video, masks)
    # 5 frames at 8 fps -> offsets 0, 3 ... an odd count only at a coarser rate
    items = semantic.semantic_items(spec.chunks, fps=FPS, sample_fps=2.0, presentation="merged")
    stamps = items[0]["timestamps"]
    assert len(stamps) % 2 == 0
    assert stamps == [0.0, 0.0, 30 / FPS, 30 / FPS]
    pairs = [(stamps[i] + stamps[i + 1]) / 2.0 for i in range(0, len(stamps), 2)]
    assert pairs == [0.0, 30 / FPS]  # no block straddles the gap


def test_an_unknown_presentation_is_refused(node_module, semantic):
    spec = _still(node_module, torch.ones(1, 64, 128))
    with pytest.raises(ValueError, match="unknown presentation"):
        semantic.semantic_items(spec.chunks, fps=FPS, presentation="both")


def test_sample_fps_controls_how_densely_a_clip_is_seen(node_module, clips, semantic):
    video, masks = _video_and_mask(22, range(22))
    spec = _from_video(node_module, clips, video, masks)
    counts = {fps: semantic.chunk_presentation(spec.chunks[0], fps=FPS, sample_fps=fps)[0].shape[0]
              for fps in (2.0, 8.0, 24.0)}
    assert counts == {2.0: 2, 8.0: 8, 24.0: 22}


# --- the CLIP wrapper ------------------------------------------------------


class _Clip:
    """A stand-in with core's clone semantics: instance attributes are NOT carried."""

    def __init__(self):
        self.calls = []

    def clone(self):
        return _Clip()

    def tokenize(self, text, return_word_ids=False, **kwargs):
        self.calls.append((text, kwargs))
        return kwargs


def _item(tag):
    return {"type": "video", "data": torch.full((2, 8, 8, 3), float(tag)), "timestamps": [0.0, 0.0]}


def test_the_wrapper_appends_its_items_to_the_conditioning_nodes_own(semantic):
    wrapped = semantic.clip_with_semantic_items(_Clip(), [_item(1)])
    native = {"type": "video", "data": torch.zeros(2, 8, 8, 3), "timestamps": [0.0, 0.0]}
    out = wrapped.tokenize("prompt", minimax_ref_items=[native])
    assert out["minimax_ref_items"][0] is native      # native refs keep their positions
    assert out["minimax_ref_items"][1]["data"].max() == 1.0


def test_images_are_folded_into_reference_items_unchanged(semantic):
    """<Picture i> is emitted identically by both tokenizer branches, so this is lossless."""
    image = torch.rand(1, 16, 16, 3)
    wrapped = semantic.clip_with_semantic_items(_Clip(), [_item(1)])
    items = wrapped.tokenize("prompt", images=[image])["minimax_ref_items"]
    assert items[0] == {"type": "image", "data": image}
    assert items[1]["type"] == "video"
    assert "images" not in wrapped.calls[0][1]


def test_chaining_accumulates_without_nesting(semantic):
    once = semantic.clip_with_semantic_items(_Clip(), [_item(1)])
    twice = semantic.clip_with_semantic_items(once, [_item(2)])
    items = twice.tokenize("prompt")["minimax_ref_items"]
    assert [float(i["data"].max()) for i in items] == [1.0, 2.0]
    assert len(twice.calls) == 1  # one wrapper reached core, not two nested ones
    assert len(getattr(twice, semantic.PENDING_ATTR)) == 2


def test_a_node_presenting_both_images_and_ref_items_is_refused(semantic):
    wrapped = semantic.clip_with_semantic_items(_Clip(), [_item(1)])
    with pytest.raises(ValueError, match="cannot tell what order"):
        wrapped.tokenize("prompt", images=[torch.zeros(1, 8, 8, 3)],
                         minimax_ref_items=[_item(2)])


def test_no_items_passes_the_clip_straight_through(semantic):
    clip = _Clip()
    assert semantic.clip_with_semantic_items(clip, []) is clip


# --- the node --------------------------------------------------------------


def test_semantic_conditioning_off_leaves_the_clip_alone(node_module):
    spec = _still(node_module, torch.ones(1, 64, 128))
    clip = _Clip()
    out, report = node_module.vloMiniMaxH3ApplySemanticGuides.execute(
        clip=clip, guide_spec=spec, semantic_conditioning="off").result
    assert out is clip
    assert "off" in report


def test_a_spec_with_nothing_eligible_is_refused_rather_than_doing_nothing(node_module):
    spec = _still(node_module, torch.full((1, 64, 128), 0.5))
    with pytest.raises(ValueError, match="none of the 1 guide"):
        node_module.vloMiniMaxH3ApplySemanticGuides.execute(clip=_Clip(), guide_spec=spec)


def test_held_back_chunks_are_reported(node_module, clips):
    masks = torch.zeros(60, 64, 128)
    masks[0:22] = 1.0
    masks[30:52] = 0.5
    spec = _from_video(node_module, clips, _video_and_mask(60, [])[0], masks)
    _, report = node_module.vloMiniMaxH3ApplySemanticGuides.execute(
        clip=_Clip(), guide_spec=spec).result
    assert "1 guide(s) held back" in report
    assert "Qwen tokens added to the text span" in report


def test_add_guides_from_spec_matches_the_all_in_one_node(node_module):
    """The refactor's contract: same plan, same keyframes, whichever node applies it."""
    forward = masked_guide_module("masked_h3_forward")
    mask = torch.ones(1, 64, 128)
    direct = node_module.vloMiniMaxH3AddMaskedGuide.execute(
        positive=[[torch.zeros(1, 4, 16), {}]], latent=_av_latent(), vae=_Vae(),
        image=torch.zeros(1, 64, 128, 3), mask=mask, frame_idx=7).result[0]
    spec = _still(node_module, mask, frame_idx=7)
    viaspec = node_module.vloMiniMaxH3AddGuidesFromSpec.execute(
        positive=[[torch.zeros(1, 4, 16), {}]], latent=_av_latent(), guide_spec=spec).result[0]
    a = direct[0][1]["minimax_keyframes"][0]
    b = viaspec[0][1]["minimax_keyframes"][0]
    assert a["resolved_frame_index"] == b["resolved_frame_index"]
    assert torch.equal(a[forward.MASKED_GUIDE_KEY]["strengths"],
                       b[forward.MASKED_GUIDE_KEY]["strengths"])


def test_a_spec_planned_for_another_video_is_refused(node_module):
    spec = _still(node_module, torch.ones(1, 64, 128))
    with pytest.raises(ValueError, match="planned for a 128x64 video"):
        node_module.vloMiniMaxH3AddGuidesFromSpec.execute(
            positive=[[torch.zeros(1, 4, 16), {}]],
            latent=_av_latent(width=256, height=128), guide_spec=spec)


def test_semantic_conditioning_adds_no_reference_blocks(node_module):
    """Qwen-only: no minimax_refs, so no reference span and no cursor movement."""
    spec = _still(node_module, torch.ones(1, 64, 128))
    node_module.vloMiniMaxH3ApplySemanticGuides.execute(clip=_Clip(), guide_spec=spec)
    positive = node_module.vloMiniMaxH3AddGuidesFromSpec.execute(
        positive=[[torch.zeros(1, 4, 16), {}]], latent=_av_latent(), guide_spec=spec).result[0]
    assert "minimax_refs" not in positive[0][1]


# --- what the vision block does to PackedLayout ---------------------------


def _guide_position(text_len, frame_idx=7):
    model = h3_model_module()
    layout = model.PackedLayout(text_len, 2, 4, 8, 5,
                                keyframes=[{"resolved_frame_index": frame_idx,
                                            "latent": torch.zeros(1, 24, 1, 4, 8)}])
    start = next(a for a, _, kind in layout.segments if kind == "cond")
    return float(layout.position_ids[start, 0])


def test_a_vision_block_shifts_the_guide_absolutely_but_not_relatively():
    """The invariant worth asserting is the guide's offset from the target origin.

    `PackedLayout` starts the target timeline at `text_len`, so lengthening the text
    span with a vision block moves every row -- guide and target alike -- by the same
    amount. An absolute-coordinate check would fail on a correct implementation.
    """
    model = h3_model_module()
    short, long = _guide_position(8), _guide_position(8 + 1008)
    assert long != short
    assert long - (8 + 1008) == pytest.approx(short - 8)
    assert short - 8 == pytest.approx(model.FRAME_RESCALE * 7)


def test_a_semantic_guide_costs_no_time_axis_span_the_way_a_reference_does():
    """A native reference pushes the whole target timeline back; a semantic one cannot,
    because it never becomes a `minimax_refs` block."""
    model = h3_model_module()
    keyframes = [{"resolved_frame_index": 7, "latent": torch.zeros(1, 24, 1, 4, 8)}]
    plain = model.PackedLayout(8, 2, 4, 8, 5, keyframes=keyframes)
    with_ref = model.PackedLayout(8, 2, 4, 8, 5, keyframes=keyframes,
                                  refs=[{"kind": "image", "latent_h": 4, "latent_w": 8,
                                         "latent": torch.zeros(1, 24, 1, 4, 8)}])
    a = next(s for s, _, kind in plain.segments if kind == "cond")
    b = next(s for s, _, kind in with_ref.segments if kind == "cond")
    assert float(with_ref.position_ids[b, 0]) - float(plain.position_ids[a, 0]) == pytest.approx(1.0)


# --- the core contract this rides on --------------------------------------


def test_core_still_presents_timed_video_references():
    """The probe the semantic node runs: run it here too, so the suite says which
    half broke when a ComfyUI update changes the tokenizer rather than the DiT."""
    compat = masked_guide_module("compatibility")
    compat.check_semantic_supported()


def test_the_new_nodes_declare_schemas_that_wire_together(node_module):
    """The spec output and the two spec inputs must agree on one custom type."""
    produced = {name: node_module.__dict__[name].define_schema().outputs[0].io_type
                for name in ("vloMiniMaxH3BuildGuideSpec", "vloMiniMaxH3BuildGuideSpecFromVideo")}
    consumed = {name: [i.io_type for i in node_module.__dict__[name].define_schema().inputs
                       if i.id == "guide_spec"][0]
                for name in ("vloMiniMaxH3AddGuidesFromSpec", "vloMiniMaxH3ApplySemanticGuides")}
    assert set(produced.values()) == set(consumed.values()) == {"VLO_H3_GUIDE_SPEC"}


def test_the_mask_input_is_optional_on_the_spec_builders(node_module):
    """A full-confidence guide should not have to invent an all-white mask to exist."""
    for name in ("vloMiniMaxH3BuildGuideSpec", "vloMiniMaxH3BuildGuideSpecFromVideo"):
        mask = [i for i in node_module.__dict__[name].define_schema().inputs if i.id == "mask"][0]
        assert mask.optional
