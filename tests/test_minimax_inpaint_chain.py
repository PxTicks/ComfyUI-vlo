"""The masked-latent chain of the MiniMax H3 inpaint workflow, at real shapes."""

import math, torch
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
