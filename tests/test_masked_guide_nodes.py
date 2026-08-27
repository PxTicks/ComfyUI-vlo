"""Node behaviour: what the masked-guide nodes put on the conditioning, and refuse to."""

from __future__ import annotations


import pytest
import torch

from minimax_h3_harness import comfyui_on_path, masked_guide_module


@pytest.fixture(scope="module")
def node_module():
    comfyui_on_path()
    return masked_guide_module("nodes")


@pytest.fixture(scope="module")
def forward_module():
    return masked_guide_module("masked_h3_forward")


class _Vae:
    """Stands in for the H3 video VAE: 16x spatial, one latent frame per guide image."""

    def encode(self, frames):
        b, h, w, _ = frames.shape
        return torch.zeros(1, 24, 1, h // 16, w // 16)


def _av_latent(width=128, height=64, latent_t=2, audio_t=5):
    import comfy.nested_tensor

    video = torch.zeros(1, 24, latent_t, height // 16, width // 16)
    audio = torch.zeros(1, 32, 2, audio_t)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _positive():
    return [[torch.zeros(1, 4, 16), {}]]


def _add(node_module, mask, **kwargs):
    params = dict(positive=_positive(), latent=_av_latent(), vae=_Vae(),
                  image=torch.zeros(1, 64, 128, 3), mask=mask, frame_idx=0)
    params.update(kwargs)
    return node_module.vloMiniMaxH3AddMaskedGuide.execute(**params)


def test_masked_guide_rides_on_a_stock_keyframe(node_module, forward_module):
    """Core still owns timing, layout and the VAE latent; only the extra key is ours."""
    out = _add(node_module, torch.ones(1, 64, 128), frame_idx=3).result[0]
    keyframes = out[0][1]["minimax_keyframes"]
    assert len(keyframes) == 1
    keyframe = keyframes[0]
    assert keyframe["resolved_frame_index"] == 3
    assert tuple(keyframe["latent"].shape) == (1, 24, 1, 4, 8)
    spec = keyframe[forward_module.MASKED_GUIDE_KEY]
    assert (spec["token_t"], spec["token_h"], spec["token_w"]) == (1, 2, 4)
    assert spec["strengths"].shape == (8,)
    assert torch.equal(spec["strengths"], torch.ones(8, dtype=torch.float64))


def test_chaining_appends_instead_of_replacing(node_module):
    first = _add(node_module, torch.ones(1, 64, 128), frame_idx=0).result[0]
    second = node_module.vloMiniMaxH3AddMaskedGuide.execute(
        positive=first, latent=_av_latent(), vae=_Vae(), image=torch.zeros(1, 64, 128, 3),
        mask=torch.zeros(1, 64, 128), frame_idx=4).result[0]
    keyframes = second[0][1]["minimax_keyframes"]
    assert [kf["resolved_frame_index"] for kf in keyframes] == [0, 4]


def test_negative_frame_idx_counts_from_the_end(node_module):
    latent = _av_latent(latent_t=2)
    out = node_module.vloMiniMaxH3AddMaskedGuide.execute(
        positive=_positive(), latent=latent, vae=_Vae(), image=torch.zeros(1, 64, 128, 3),
        mask=torch.ones(1, 64, 128), frame_idx=-1).result[0]
    # latent_t = 2 -> FRAME_PER_TOKEN[0] + FRAME_PER_TOKEN[1] = 1 + 4 = 5 pixel frames
    assert out[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 4


def test_out_of_range_frame_idx_is_refused(node_module):
    with pytest.raises(ValueError, match="outside the video"):
        _add(node_module, torch.ones(1, 64, 128), frame_idx=99)


@pytest.mark.parametrize("frames", [2, 4, 5, 22])
def test_image_batches_are_refused_rather_than_silently_truncated(node_module, frames):
    """One mask cannot weight several guide frames, whatever core would have done
    with the batch -- a 2-4 frame batch used to be quietly reduced to its first frame."""
    with pytest.raises(ValueError, match="single-image guides only"):
        _add(node_module, torch.ones(1, 64, 128), image=torch.zeros(frames, 64, 128, 3))


def test_mismatched_mask_framing_is_refused(node_module):
    with pytest.raises(ValueError, match="aspect ratio"):
        _add(node_module, torch.ones(1, 128, 64))


def test_strength_and_gamma_reach_the_keyframe(node_module, forward_module):
    out = _add(node_module, torch.full((1, 64, 128), 0.5),
               strength=0.8, min_aug=0.1, mask_gamma=2.0).result[0]
    spec = out[0][1]["minimax_keyframes"][0][forward_module.MASKED_GUIDE_KEY]
    assert spec["min_aug"] == pytest.approx(0.1)
    assert spec["strengths"].unique().numel() == 1
    assert spec["strengths"][0].item() == pytest.approx(0.8 * 0.25, abs=1.0 / 510.0)


def test_patch_node_installs_one_wrapper_even_when_chained(node_module):
    import comfy.patcher_extension

    class _Patcher:
        def __init__(self):
            self.wrappers = {}

        def clone(self):
            copy = _Patcher()
            copy.wrappers = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.wrappers.items()}
            return copy

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.get(wrapper_type, {}).pop(key, None)

    once = node_module.vloMiniMaxH3PatchMaskedGuides.execute(model=_Patcher()).result[0]
    twice = node_module.vloMiniMaxH3PatchMaskedGuides.execute(model=once).result[0]
    installed = twice.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][node_module.WRAPPER_KEY]
    assert len(installed) == 1


def test_token_mask_preview_shows_the_real_grid(node_module):
    mask = torch.zeros(1, 64, 128)
    mask[:, :, :64] = 1.0
    image, token_mask = node_module.vloMiniMaxH3GuideTokenMaskPreview.execute(
        latent=_av_latent(), mask=mask).result
    assert tuple(image.shape) == (1, 64, 128, 3)
    assert tuple(token_mask.shape) == (1, 64, 128)
    assert image[0, :, :60, :].min().item() == pytest.approx(1.0)   # masked half stays strong
    assert image[0, :, 68:, :].max().item() == pytest.approx(0.0)


def test_guide_refuses_a_batch_of_masks_it_would_have_to_choose_between(node_module):
    with pytest.raises(ValueError, match="carries 2 masks"):
        _add(node_module, torch.ones(2, 64, 128))


def test_pixel_fill_uses_each_images_own_mask(node_module):
    """Broadcasting mask 0 across the batch fills every image by the first one's mask."""
    images = torch.ones(2, 32, 64, 3)
    masks = torch.stack([torch.ones(32, 64), torch.zeros(32, 64)])
    out = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(
        image=images, mask=masks, fill="black").result[0]
    assert [float(out[i].mean()) for i in range(2)] == [pytest.approx(1.0), pytest.approx(0.0)]


def test_pixel_fill_broadcasts_a_single_mask_across_the_batch(node_module):
    images = torch.ones(3, 32, 64, 3)
    mask = torch.zeros(1, 32, 64)
    out = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(
        image=images, mask=mask, fill="black").result[0]
    assert out.shape[0] == 3 and float(out.max()) == pytest.approx(0.0)


def test_pixel_fill_refuses_a_mask_count_it_cannot_pair_up(node_module):
    with pytest.raises(ValueError, match="3 images and 2 masks"):
        node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(
            image=torch.ones(3, 32, 64, 3), mask=torch.zeros(2, 32, 64), fill="black")


def test_pixel_fill_baseline_replaces_only_the_masked_out_region(node_module):
    image = torch.ones(1, 64, 128, 3)
    mask = torch.zeros(1, 64, 128)
    mask[:, :, :64] = 1.0
    out = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(
        image=image, mask=mask, fill="gray").result[0]
    assert out[0, :, :64].min().item() == pytest.approx(1.0)
    assert out[0, :, 64:].max().item() == pytest.approx(0.5)


def test_pixel_fill_noise_is_deterministic(node_module):
    image = torch.ones(1, 32, 32, 3)
    mask = torch.zeros(1, 32, 32)
    kwargs = dict(image=image, mask=mask, fill="noise")
    first = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(seed=3, **kwargs).result[0]
    again = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(seed=3, **kwargs).result[0]
    other = node_module.vloMiniMaxH3MaskedGuidePixelFill.execute(seed=4, **kwargs).result[0]
    assert torch.equal(first, again) and not torch.equal(first, other)


def test_nodes_refuse_latents_that_are_not_h3_av_pairs(node_module):
    with pytest.raises(ValueError, match="MiniMax H3 AV latent"):
        _add(node_module, torch.ones(1, 64, 128), latent={"samples": torch.zeros(1, 24, 2, 4, 8)})


def _patcher_stub():
    class _Patcher:
        def __init__(self):
            self.wrappers = {}

        def clone(self):
            copy = _Patcher()
            copy.wrappers = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.wrappers.items()}
            return copy

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.get(wrapper_type, {}).pop(key, None)

    return _Patcher()


@pytest.mark.parametrize("clock", ["stock", "floored", "matched", "target_relative"])
def test_patch_node_accepts_every_guide_clock(node_module, clock):
    out = node_module.vloMiniMaxH3PatchMaskedGuides.execute(model=_patcher_stub(), guide_clock=clock)
    assert out.result[0] is not None


def test_patch_node_refuses_an_unknown_guide_clock(node_module):
    with pytest.raises(ValueError, match="unknown guide clock"):
        node_module.vloMiniMaxH3PatchMaskedGuides.execute(model=_patcher_stub(), guide_clock="honest")


@pytest.mark.parametrize("legacy,expected", [(True, "floored"), (False, "stock")])
def test_patch_node_maps_the_legacy_sync_timesteps_boolean(node_module, legacy, expected):
    """`guide_clock` replaced a boolean; an API caller carrying the old argument
    should land on the arm it used to mean rather than tripping the combo check."""
    import comfy.patcher_extension

    patched = node_module.vloMiniMaxH3PatchMaskedGuides.execute(
        model=_patcher_stub(), guide_clock=legacy).result[0]
    installed = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][node_module.WRAPPER_KEY]
    assert len(installed) == 1
    assert node_module.DEFAULT_GUIDE_CLOCK == "matched"   # the documented arm is the default
