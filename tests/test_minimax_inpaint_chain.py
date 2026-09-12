"""The masked-latent chain of the MiniMax H3 inpaint workflow, at real shapes."""

import math

import pytest
import torch
from test_batch_nodes_integration import nodes_module  # noqa: F401
from test_mask_to_latent_mask import _MiniMaxVae


def test_minimax_inpaint_chain(nodes_module):
    """nodes 66 -> 67 -> 68 -> 79 -> 25 of the workflow, at real MiniMax shapes."""
    nt = nodes_module.comfy.nested_tensor
    W, H, F = 832, 480, 124                      # after the 17k+5 snap
    latent_t = (F - 5) // 17 * 5 + 2             # comfy/sd.py downscale_ratio
    audio_t = math.floor(F / 24 * 40)
    video = torch.randn(1, 24, latent_t, H // 16, W // 16)
    audio = torch.randn(1, 32, 2, audio_t)
    latent = {"samples": nt.NestedTensor((video, audio))}

    masks = torch.zeros(F, H, W)
    masks[30:90, 100:300, 200:500] = 1.0         # a moving-ish region, mid clip

    # 66: vloMaskToLatentMask
    latent_mask = nodes_module.vloMaskToLatentMask.execute(
        latent=latent, vae=_MiniMaxVae(), masks=masks, resize_mode="bilinear"
    ).result[0]
    assert tuple(latent_mask.shape) == (latent_t, H // 16, W // 16)

    # 67: SetLatentNoiseMask (stock)
    masked = latent.copy()
    masked["noise_mask"] = latent_mask.reshape(
        (-1, 1, latent_mask.shape[-2], latent_mask.shape[-1])
    )

    # 68: vloSetAudioLatentBinaryMasks
    masked = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=masked, masks=masks, audio_vae=_MiniMaxAudioVae()
    ).result[0]
    vm, am = masked["noise_mask"].unbind()
    assert tuple(am.shape) == tuple(audio.shape)

    # 79: LatentMultiply(0.0)
    blank = masked.copy()
    blank["samples"] = masked["samples"] * 0.0

    # 25: vloLatentCompositeMasked, force_binary_mask=True
    out = nodes_module.vloLatentCompositeMasked.execute(
        destination=masked, source=blank, force_binary_mask=True
    ).result[0]
    ov, oa = out["samples"].unbind()
    assert tuple(ov.shape) == tuple(video.shape)
    assert tuple(oa.shape) == tuple(audio.shape)

    # masked region blanked, everything else untouched
    vmask = (nodes_module.comfy.utils.reshape_mask(vm, video.shape) >= 0.5)
    assert torch.equal(ov[~vmask], video[~vmask])
    assert torch.count_nonzero(ov[vmask]) == 0
    assert vmask.any() and not vmask.all()
    amask = (am >= 0.5)
    assert torch.equal(oa[~amask], audio[~amask])
    assert torch.count_nonzero(oa[amask]) == 0
    assert amask.any() and not amask.all()
    print(f"latent_t={latent_t} audio_t={audio_t} "
          f"video masked {vmask.float().mean():.3f}, audio masked {amask.float().mean():.3f}")


class MiniMaxH3AudioVAE:  # name is how the node detects the architecture
    sample_rate = 32000
    samples_per_latent = 800


class _MiniMaxAudioVae:
    """comfy/sd.py's MiniMax H3 audio VAE config."""
    latent_dim = 2
    upscale_ratio = downscale_ratio = 800

    def __init__(self):
        self.first_stage_model = MiniMaxH3AudioVAE()


def test_feather_after_the_composite_keeps_audio_under_the_ramp(nodes_module):
    """nodes 66 -> 67 -> 68 -> 79 -> 25 -> feather, at real MiniMax shapes.

    The feather has to run after node 25, not before it. Node 25 composites a
    blank latent wherever the mask is set, so feathering first would leave the
    ramp steps blending toward silence instead of toward the original audio.
    """
    nt = nodes_module.comfy.nested_tensor
    W, H, F = 832, 480, 124
    latent_t = (F - 5) // 17 * 5 + 2
    audio_t = math.floor(F / 24 * 40)
    video = torch.randn(1, 24, latent_t, H // 16, W // 16)
    audio = torch.randn(1, 32, 2, audio_t)
    latent = {"samples": nt.NestedTensor((video, audio))}

    masks = torch.zeros(F, H, W)
    masks[30:90, 100:300, 200:500] = 1.0

    latent_mask = nodes_module.vloMaskToLatentMask.execute(
        latent=latent, vae=_MiniMaxVae(), masks=masks, resize_mode="bilinear"
    ).result[0]
    masked = latent.copy()
    masked["noise_mask"] = latent_mask.reshape(
        (-1, 1, latent_mask.shape[-2], latent_mask.shape[-1])
    )
    masked = nodes_module.vloSetAudioLatentBinaryMasks.execute(
        audio_latent=masked, masks=masks, audio_vae=_MiniMaxAudioVae()
    ).result[0]
    blank = masked.copy()
    blank["samples"] = masked["samples"] * 0.0
    composited = nodes_module.vloLatentCompositeMasked.execute(
        destination=masked, source=blank, force_binary_mask=True
    ).result[0]

    feathered = nodes_module.vloFeatherAudioLatentMask.execute(
        audio_latent=composited,
        audio_vae=_MiniMaxAudioVae(),
        lead_ramp=0.08,
        tail_ramp=0.12,
    ).result[0]

    binary_audio_mask = masked["noise_mask"].unbind()[1]
    feathered_audio_mask = feathered["noise_mask"].unbind()[1]
    audio_samples = feathered["samples"].unbind()[1]

    ramp = (feathered_audio_mask > 0) & (feathered_audio_mask < 1)
    assert ramp.any(), "the feather produced no partial steps"
    # Every ramp step sits outside the binary core, and still holds real audio to
    # blend toward - which is exactly what running after node 25 buys.
    assert torch.all(binary_audio_mask[ramp] == 0)
    assert torch.equal(audio_samples[ramp], audio[ramp])
    # The solid core is unchanged by the feather, and still blanked.
    core = feathered_audio_mask >= 1.0
    assert torch.equal(core, binary_audio_mask >= 0.5)
    assert torch.count_nonzero(audio_samples[core]) == 0
    # The video mask rides through untouched.
    assert torch.equal(
        feathered["noise_mask"].unbind()[0], composited["noise_mask"].unbind()[0]
    )


def test_feathered_mask_reaches_the_model_as_distinct_row_timesteps(nodes_module):
    """The feather survives MiniMax's own pooling and drives per-row timesteps.

    comfy/ldm/minimax/model.py puts a masked audio row at sigma = m * sigma_audio,
    so a decaying mask is a genuine per-step denoise strength rather than a
    post-hoc crossfade. This pins that contract to the real ComfyUI code.
    """
    import types

    import comfy.model_base
    import comfy.sampler_helpers
    import comfy.utils

    class _PooledLike(comfy.model_base.MiniMaxH3):
        def __init__(self):  # the real methods, none of the model machinery
            self.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))

    video_shape, audio_shape = (1, 24, 7, 30, 52), (1, 32, 2, 80)
    audio_mask = torch.zeros(audio_shape)
    audio_mask[..., 30:50] = 1.0
    latent = {
        "samples": nodes_module.comfy.nested_tensor.NestedTensor(
            (torch.zeros(video_shape), torch.zeros(audio_shape))
        ),
        "noise_mask": nodes_module.comfy.nested_tensor.NestedTensor(
            (torch.ones((1, 1) + video_shape[2:]), audio_mask)
        ),
    }

    feathered = nodes_module.vloFeatherAudioLatentMask.execute(
        audio_latent=latent,
        audio_vae=_MiniMaxAudioVae(),
        lead_ramp=0.1,
        tail_ramp=0.1,
        curve="linear",
    ).result[0]

    # comfy/samplers.py reshapes each stream mask to its full latent shape and
    # packs them before the model ever sees them.
    latent_shapes = [video_shape, audio_shape]
    packed, _ = comfy.utils.pack_latents(
        [
            comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
            for mask, shape in zip(feathered["noise_mask"].unbind(), latent_shapes)
        ]
    )
    values = _PooledLike()._denoise_mask_values(packed, latent_shapes)

    assert "audio_denoise_mask" in values, "the model saw a fully-denoised audio mask"
    rows = values["audio_denoise_mask"].reshape(-1)

    sigma_audio = 0.4  # any mid-sampling audio sigma
    row_timesteps = (1.0 - rows * sigma_audio).clamp(max=1.0)
    # The core denoises hardest, the ramp progressively less, the rest not at all.
    assert row_timesteps.min() == pytest.approx(1.0 - sigma_audio)
    assert row_timesteps.max() == pytest.approx(1.0)
    assert len(set(row_timesteps.tolist())) > 3


def _composited_chain(audio_t=60):
    """A nested AV latent past node 25, plus the pre-composite latent."""
    import test_batch_nodes_integration as integration

    m = integration._load_nodes_module()
    nt = m.comfy.nested_tensor
    video = torch.randn(1, 24, 7, 30, 52)
    audio = torch.randn(1, 32, 2, audio_t)
    audio_mask = torch.zeros(1, 32, 2, audio_t)
    audio_mask[..., 20:40] = 1.0
    before = {
        "samples": nt.NestedTensor((video, audio)),
        "noise_mask": nt.NestedTensor((torch.ones(1, 1, 7, 30, 52), audio_mask)),
    }
    blank = before.copy()
    blank["samples"] = before["samples"] * 0.0
    after = m.vloLatentCompositeMasked.execute(
        destination=before, source=blank, force_binary_mask=True
    ).result[0]
    return m, before, after, audio


@pytest.mark.parametrize("mode", ["outer", "centered", "inner"])
def test_every_mode_blends_toward_real_audio_when_the_original_is_supplied(mode):
    """The blank composite clears the binary region; inner and centered ramp into it.

    Without the original latent those ramp steps read a blanked latent image and
    blend toward silence, which is a worse artefact than the seam being fixed.
    """
    m, before, after, audio = _composited_chain()

    out = m.vloFeatherAudioLatentMask.execute(
        audio_latent=after,
        original_audio_latent=before,
        audio_vae=_MiniMaxAudioVae(),
        lead_ramp=0.1,
        tail_ramp=0.1,
        curve="linear",
        mode=mode,
    ).result[0]

    feathered_mask = out["noise_mask"].unbind()[1]
    samples = out["samples"].unbind()[1]
    ramp = (feathered_mask > 0) & (feathered_mask < 1)

    assert ramp.any()
    assert torch.equal(samples[ramp], audio[ramp])
    # Fully generated steps stay blanked: they never read their latent image.
    solid = feathered_mask >= 1.0
    assert torch.count_nonzero(samples[solid]) == 0


@pytest.mark.parametrize("mode", ["centered", "inner"])
def test_inner_ramps_blend_toward_silence_without_the_original(mode):
    """Pins the hazard the original_audio_latent input exists to remove."""
    m, _, after, _ = _composited_chain()

    out = m.vloFeatherAudioLatentMask.execute(
        audio_latent=after,
        audio_vae=_MiniMaxAudioVae(),
        lead_ramp=0.1,
        tail_ramp=0.1,
        curve="linear",
        mode=mode,
    ).result[0]

    feathered_mask = out["noise_mask"].unbind()[1]
    samples = out["samples"].unbind()[1]
    ramp = (feathered_mask > 0) & (feathered_mask < 1)

    assert ramp.any()
    assert torch.count_nonzero(samples[ramp]) < int(ramp.sum())


def test_outer_ramps_need_no_original_latent():
    """Outer ramps fall outside the region the composite cleared."""
    m, _, after, audio = _composited_chain()

    out = m.vloFeatherAudioLatentMask.execute(
        audio_latent=after,
        audio_vae=_MiniMaxAudioVae(),
        lead_ramp=0.1,
        tail_ramp=0.1,
        mode="outer",
    ).result[0]

    feathered_mask = out["noise_mask"].unbind()[1]
    samples = out["samples"].unbind()[1]
    ramp = (feathered_mask > 0) & (feathered_mask < 1)

    assert ramp.any()
    assert torch.equal(samples[ramp], audio[ramp])


def test_a_mismatched_original_latent_is_rejected():
    m, _, after, _ = _composited_chain()
    wrong = {"samples": torch.randn(1, 32, 2, 999)}

    with pytest.raises(ValueError, match="must be the same latent"):
        m.vloFeatherAudioLatentMask.execute(
            audio_latent=after,
            original_audio_latent=wrong,
            audio_vae=_MiniMaxAudioVae(),
            lead_ramp=0.1,
            mode="inner",
        )
