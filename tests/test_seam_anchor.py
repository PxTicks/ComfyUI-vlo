"""vloH3SeamAnchor: kept H3 tokens near a seam are injected and labelled at a noise level."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


# A 2-frame video latent of 12 x 16 cells (6 x 8 H3 tokens), plus an audio stream.
T, H, W = 2, 12, 16
GENERATED_TOKEN = (2, 3)  # token row, token column


def _h3_model():
    """A MiniMaxH3 stand-in carrying the real mask and injection methods, minus the weights."""
    model_base = __import__("comfy.model_base", fromlist=["MiniMaxH3"])
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    h3 = model_base.MiniMaxH3

    class _StubH3:
        _pool_masks_to_token_grid = h3._pool_masks_to_token_grid
        _token_grid_masks = h3._token_grid_masks
        _denoise_mask_values = h3._denoise_mask_values
        scale_latent_inpaint = h3.scale_latent_inpaint
        audio_scale = h3.audio_scale

        def __init__(self):
            self.latent_format = latent_formats.MiniMaxH3AV()
            self.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
            self.model_sampling = SimpleNamespace(audio_scale=1.0)
            self.latent_shapes = None

    return _StubH3()


class _FakeGuider:
    def __init__(self, model):
        self.model_patcher = SimpleNamespace(model=model, load_device=torch.device("cpu"))
        self.model_options = {}


class _FakeExecutor:
    """The next OUTER_SAMPLE wrapper; records what sampling would have seen."""

    def __init__(self, guider, on_call=None):
        self.class_obj = guider
        self.on_call = on_call
        self.call = None

    def __call__(self, noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs):
        model = self.class_obj.model_patcher.model
        self.call = {
            "denoise_mask": denoise_mask,
            "overridden": "scale_latent_inpaint" in vars(model),
            "scale_latent_inpaint": getattr(model, "scale_latent_inpaint", None),
        }
        if self.on_call is not None:
            self.on_call()
        return latent_image


class _FakeApplyExecutor:
    def __init__(self, model):
        self.class_obj = model
        self.call = None

    def __call__(self, x, sigma, *args, **kwargs):
        self.call = {"kwargs": kwargs}
        return x


def _generate_cells() -> torch.Tensor:
    """1 over the cells of the one generated token, in every frame."""
    mask = torch.zeros(T, H, W)
    r, c = GENERATED_TOKEN
    mask[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2] = 1.0
    return mask


def _expected_token_levels(ramp: int) -> torch.Tensor:
    levels = torch.zeros(H // 2, W // 2)
    for r in range(H // 2):
        for c in range(W // 2):
            distance = max(abs(r - GENERATED_TOKEN[0]), abs(c - GENERATED_TOKEN[1]))
            if 1 <= distance <= ramp:
                levels[r, c] = (ramp - distance + 1) / ramp
    return levels


def _packed(nodes_module, audio_mask_value=1.0):
    comfy = nodes_module.comfy
    video = torch.randn(1, 24, T, H, W)
    audio = torch.randn(1, 32, 2, 10)
    latent, shapes = comfy.utils.pack_latents([video, audio])
    video_mask = _generate_cells().view(1, 1, T, H, W).expand(1, 24, T, H, W)
    audio_mask = torch.full((1, 32, 2, 10), audio_mask_value)
    mask, _ = comfy.utils.pack_latents([video_mask.contiguous(), audio_mask])
    return latent, shapes, mask


SIGMAS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


def _sample(nodes_module, model, *, mode="ramp", ramp=2, mask=None, shapes=None,
            latent=None, on_call=None, release_step=0, release_tokens=1):
    anchor = nodes_module.seam_anchor
    guider = _FakeGuider(model)
    executor = _FakeExecutor(guider, on_call)
    wrapper = anchor._SeamAnchorOuterSample(mode, ramp, release_step, release_tokens)
    noise = torch.randn_like(latent)
    wrapper(executor, noise, latent, object(), SIGMAS, mask,
            None, False, 0, latent_shapes=shapes)
    return guider, executor


def _ring_cells(ring: int) -> torch.Tensor:
    """`[T, H, W]` cells of the tokens exactly `ring` tokens from the generated one."""
    tokens = torch.zeros(H // 2, W // 2, dtype=torch.bool)
    for r in range(H // 2):
        for c in range(W // 2):
            distance = max(abs(r - GENERATED_TOKEN[0]), abs(c - GENERATED_TOKEN[1]))
            tokens[r, c] = distance == ring
    cells = tokens.repeat_interleave(2, 0).repeat_interleave(2, 1)
    return cells.expand(T, H, W)


def _video_of(nodes_module, packed, shapes):
    return nodes_module.comfy.utils.unpack_latents(packed, shapes)[0][0, 0]


def _labels_at(nodes_module, guider, model, latent, sigma):
    apply_executor = _FakeApplyExecutor(model)
    _label_wrapper(nodes_module, guider)(
        apply_executor, latent, torch.tensor([sigma]), None, None, None, {}
    )
    minimax = __import__("comfy.ldm.minimax.model", fromlist=["mask_row_values"])
    video_label = apply_executor.call["kwargs"]["denoise_mask"]
    return minimax.mask_row_values(video_label[0, 0], T, H, W).view(T, H // 2, W // 2)


def _token_value(rows: torch.Tensor, ring: int) -> torch.Tensor:
    """Every token value in `ring`, from a `[T, H/2, W/2]` token grid."""
    cells = _ring_cells(ring)[:, ::2, ::2]
    return rows[cells]


def _label_wrapper(nodes_module, guider):
    comfy = nodes_module.comfy
    wrappers = guider.model_options["transformer_options"]["wrappers"]
    registered = wrappers[comfy.patcher_extension.WrappersMP.APPLY_MODEL][
        nodes_module.seam_anchor._SEAM_ANCHOR_APPLY_MODEL_KEY
    ]
    assert len(registered) == 1
    return registered[0]


# --- the field ----------------------------------------------------------------


@pytest.mark.parametrize("ramp", [1, 2, 3])
def test_ramp_levels_fall_with_token_distance_from_the_seam(nodes_module, ramp):
    levels = nodes_module.seam_anchor._seam_noise_levels(_generate_cells(), "ramp", ramp)

    expected = _expected_token_levels(ramp)
    expected = expected.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.allclose(levels, expected.expand(T, H, W))


def test_noised_mode_noises_every_kept_cell(nodes_module):
    generate = _generate_cells()
    levels = nodes_module.seam_anchor._seam_noise_levels(generate, "noised", 2)

    assert torch.equal(levels, 1.0 - generate)


def test_a_token_holding_any_generated_cell_is_left_to_the_sampler(nodes_module):
    # One generated cell makes its whole token generated in H3's eyes (amax), so none of
    # that token's kept cells may be noised as if they were kept tokens.
    generate = torch.zeros(T, H, W)
    generate[:, 4, 6] = 1.0

    levels = nodes_module.seam_anchor._seam_noise_levels(generate, "ramp", 1)

    assert float(levels[:, 4:6, 6:8].abs().max()) == 0.0
    assert float(levels[:, 2:4, 4:6].min()) == 1.0  # its neighbour token is ring 1


def test_odd_grids_keep_their_shape(nodes_module):
    generate = torch.zeros(1, 7, 9)
    generate[:, 3, 4] = 1.0

    levels = nodes_module.seam_anchor._seam_noise_levels(generate, "ramp", 2)

    assert tuple(levels.shape) == (1, 7, 9)
    assert float(levels[:, 3, 4]) == 0.0


def test_an_unknown_mode_is_rejected(nodes_module):
    with pytest.raises(ValueError, match="Unsupported seam anchor mode"):
        nodes_module.seam_anchor._seam_noise_levels(_generate_cells(), "spiral", 2)


# --- the injection --------------------------------------------------------------


def test_the_override_is_in_place_only_while_sampling(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)

    _, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert executor.call["overridden"]
    assert "scale_latent_inpaint" not in vars(model)


def test_the_override_is_removed_even_when_sampling_fails(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)

    def fail():
        raise RuntimeError("sampler blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent, on_call=fail)
    assert "scale_latent_inpaint" not in vars(model)


def test_an_existing_instance_override_is_put_back(nodes_module):
    model = _h3_model()

    def earlier_override(**kwargs):
        return kwargs["latent_image"]

    model.scale_latent_inpaint = earlier_override
    latent, shapes, mask = _packed(nodes_module)

    _, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert executor.call["scale_latent_inpaint"] is not earlier_override
    assert vars(model)["scale_latent_inpaint"] is earlier_override


def test_ramp_cells_are_injected_at_their_noise_level_and_the_rest_natively(nodes_module):
    comfy = nodes_module.comfy
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    model.latent_shapes = shapes
    _, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)
    override = executor.call["scale_latent_inpaint"]

    sigma = torch.tensor([0.8])
    noise = torch.randn_like(latent)
    x = torch.randn_like(latent)
    native_fn = type(model).scale_latent_inpaint
    native = native_fn(model, sigma=sigma, noise=noise, latent_image=latent, x=x, denoise_mask=mask)
    ours = override(sigma=sigma, noise=noise, latent_image=latent, x=x, denoise_mask=mask)

    ours_v, ours_a = comfy.utils.unpack_latents(ours, shapes)
    native_v, native_a = comfy.utils.unpack_latents(native, shapes)
    ref_v = comfy.utils.unpack_latents(latent, shapes)[0]
    noise_v = comfy.utils.unpack_latents(noise, shapes)[0]

    levels = nodes_module.seam_anchor._seam_noise_levels(_generate_cells(), "ramp", 2)
    level = levels.view(1, 1, T, H, W)
    expected_ramp = (1 - level * 0.8) * ref_v + level * 0.8 * noise_v
    ramp = (level > 0).expand_as(ours_v)
    assert torch.allclose(ours_v[ramp], expected_ramp.expand_as(ours_v)[ramp], atol=1e-6)
    assert torch.equal(ours_v[~ramp], native_v[~ramp])
    assert torch.equal(ours_a, native_a)


def test_the_sampler_keeps_the_binary_mask(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)

    _, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert executor.call["denoise_mask"] is mask


# --- the labels -----------------------------------------------------------------


def test_ramp_tokens_are_labelled_at_their_level(nodes_module):
    minimax = __import__("comfy.ldm.minimax.model", fromlist=["mask_row_values"])
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    apply_executor = _FakeApplyExecutor(model)
    _label_wrapper(nodes_module, guider)(
        apply_executor, latent, torch.tensor([0.8]), None, None, None, {}
    )
    video_label = apply_executor.call["kwargs"]["denoise_mask"]

    expected = _expected_token_levels(2)
    expected[GENERATED_TOKEN] = 1.0
    rows = minimax.mask_row_values(video_label[0, 0], T, H, W)
    assert torch.allclose(rows, expected.flatten().repeat(T), atol=1 / 256)

    # H3's own timestep formula then puts each ramp token at its level of sigma.
    rows_t = (1.0 - rows * 0.8).clamp(max=max(0.2, minimax.VISUAL_COND_TIMESTEP))
    ring_1 = expected.flatten().repeat(T) == 1.0
    assert torch.allclose(rows_t[ring_1], torch.tensor(0.2))


def test_audio_labels_are_the_real_masks(nodes_module):
    model = _h3_model()
    # A held audio stream (all zeros) must stay held.
    latent, shapes, mask = _packed(nodes_module, audio_mask_value=0.0)
    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    apply_executor = _FakeApplyExecutor(model)
    _label_wrapper(nodes_module, guider)(
        apply_executor, latent, torch.tensor([0.8]), None, None, None, {}
    )

    native = model._denoise_mask_values(mask, shapes)["audio_denoise_mask"]
    assert torch.equal(apply_executor.call["kwargs"]["audio_denoise_mask"], native)


# --- when it steps aside --------------------------------------------------------


def test_no_noise_mask_passes_through(nodes_module):
    model = _h3_model()
    latent, shapes, _ = _packed(nodes_module)

    guider, executor = _sample(nodes_module, model, mask=None, shapes=shapes, latent=latent)

    assert not executor.call["overridden"]
    assert "transformer_options" not in guider.model_options


def test_a_single_stream_latent_passes_through(nodes_module, caplog):
    model = _h3_model()
    video = torch.randn(1, 24, T, H, W)
    mask = _generate_cells().view(1, 1, T, H, W).expand_as(video).contiguous()

    with caplog.at_level("WARNING"):
        guider, executor = _sample(
            nodes_module, model, mask=mask, shapes=[video.shape], latent=video
        )

    assert not executor.call["overridden"]
    assert "transformer_options" not in guider.model_options
    assert "only applies to MiniMax H3" in caplog.text


def test_a_non_h3_model_passes_through(nodes_module):
    model = SimpleNamespace(latent_format=object())
    latent, shapes, mask = _packed(nodes_module)

    guider, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert not executor.call["overridden"]
    assert "transformer_options" not in guider.model_options


def test_an_all_generated_mask_has_no_seam(nodes_module):
    comfy = nodes_module.comfy
    model = _h3_model()
    latent, shapes, _ = _packed(nodes_module)
    mask, _ = comfy.utils.pack_latents(
        [torch.ones(1, 24, T, H, W), torch.ones(1, 32, 2, 10)]
    )

    guider, executor = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert not executor.call["overridden"]
    assert "transformer_options" not in guider.model_options


# --- the node -------------------------------------------------------------------


def test_clean_mode_returns_the_model_unpatched(nodes_module):
    model = object()
    out = nodes_module.vloH3SeamAnchor.execute(model, mode="clean").result[0]
    assert out is model


def test_other_modes_patch_a_clone(nodes_module):
    class _Patcher:
        def __init__(self):
            self.wrappers = []

        def clone(self):
            return _Patcher()

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.append((wrapper_type, key, wrapper))

    original = _Patcher()
    patched = nodes_module.vloH3SeamAnchor.execute(original, mode="ramp", ramp_tokens=3).result[0]

    assert patched is not original and original.wrappers == []
    [(wrapper_type, key, wrapper)] = patched.wrappers
    assert wrapper_type == nodes_module.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    assert key == nodes_module.seam_anchor._SEAM_ANCHOR_OUTER_SAMPLE_KEY
    assert (wrapper.mode, wrapper.ramp_tokens) == ("ramp", 3)


# --- the release ----------------------------------------------------------------


def test_no_release_leaves_the_sampler_mask_alone(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)

    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent)

    assert "denoise_mask_function" not in guider.model_options


def test_released_rings_are_freed_on_ttms_clock(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent,
                        ramp=2, release_step=3, release_tokens=1)
    schedule = guider.model_options["denoise_mask_function"]

    # Held while step < 3: the evaluations at sigmas[0..2] still hold, sigmas[3] does not.
    for sigma in SIGMAS[:3]:
        assert schedule(sigma.view(1), mask) is mask
    after = schedule(SIGMAS[3].view(1), mask)

    video_after = _video_of(nodes_module, after, shapes)
    video_before = _video_of(nodes_module, mask, shapes)
    ring_1 = _ring_cells(1)
    assert float(video_after[ring_1].min()) == 1.0
    assert torch.equal(video_after[~ring_1], video_before[~ring_1])
    audio_after = nodes_module.comfy.utils.unpack_latents(after, shapes)[1]
    audio_before = nodes_module.comfy.utils.unpack_latents(mask, shapes)[1]
    assert torch.equal(audio_after, audio_before)


def test_labels_release_with_the_sampler(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent,
                        ramp=2, release_step=3, release_tokens=2)

    before = _labels_at(nodes_module, guider, model, latent, float(SIGMAS[1]))
    after = _labels_at(nodes_module, guider, model, latent, float(SIGMAS[3]))

    assert torch.allclose(_token_value(before, 2), torch.tensor(0.5), atol=1 / 256)
    assert torch.allclose(_token_value(after, 2), torch.tensor(1.0))
    # Beyond the released rings, kept tokens stay held clean on both sides of it.
    assert float(_token_value(before, 3).max()) == 0.0
    assert float(_token_value(after, 3).max()) == 0.0


def test_clean_mode_can_still_release(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    guider, executor = _sample(nodes_module, model, mode="clean", mask=mask, shapes=shapes,
                               latent=latent, release_step=2, release_tokens=1)

    # Nothing is noised, so H3's own injection is left in place...
    assert not executor.call["overridden"]
    # ...but ring 1 is held clean, then released like the generated area.
    before = _labels_at(nodes_module, guider, model, latent, float(SIGMAS[0]))
    after = _labels_at(nodes_module, guider, model, latent, float(SIGMAS[3]))
    assert float(_token_value(before, 1).max()) == 0.0
    assert torch.allclose(_token_value(after, 1), torch.tensor(1.0))


def test_a_release_past_the_schedule_never_frees_anything(nodes_module):
    model = _h3_model()
    latent, shapes, mask = _packed(nodes_module)
    guider, _ = _sample(nodes_module, model, mask=mask, shapes=shapes, latent=latent,
                        release_step=99, release_tokens=1)
    schedule = guider.model_options["denoise_mask_function"]

    for sigma in SIGMAS[:-1]:
        assert schedule(sigma.view(1), mask) is mask


def test_clean_mode_with_a_release_patches_the_model(nodes_module):
    class _Patcher:
        def __init__(self):
            self.wrappers = []

        def clone(self):
            return _Patcher()

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.append(wrapper)

    patched = nodes_module.vloH3SeamAnchor.execute(
        _Patcher(), mode="clean", release_step=15, release_tokens=2
    ).result[0]

    [wrapper] = patched.wrappers
    assert (wrapper.mode, wrapper.release_step, wrapper.release_tokens) == ("clean", 15, 2)


def test_comfys_sampler_releases_held_tokens_from_a_consistent_state(nodes_module):
    """The contract the release relies on, run through ComfyUI's own inpaint sampler.

    While a token is held its prediction is pinned to the reference, so the sampler's
    own state for it stays exactly the reference noised to the global sigma with the
    fixed noise. A released token therefore resumes from a correctly noised state,
    whatever it was shown at, then evolves freely; held tokens still end at the reference.
    """
    samplers = __import__("comfy.samplers", fromlist=["KSamplerX0Inpaint"])
    k_sampling = __import__("comfy.k_diffusion.sampling", fromlist=["sample_euler"])
    schedule_type = nodes_module.ttm._TTMDenoiseMaskSchedule

    torch.manual_seed(0)
    reference, noise = torch.randn(1, 4, 6, 6), torch.randn(1, 4, 6, 6)
    held = torch.zeros(1, 4, 6, 6)
    held[..., 3:] = 1.0  # right half generated, left half held
    released = torch.zeros_like(held, dtype=torch.bool)
    released[..., 2] = True  # one held column released
    sigmas = torch.linspace(1.0, 0.0, 9)
    release_step = 5

    class _Inner:
        def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
            # ComfyUI's flow injection (BaseModel.scale_latent_inpaint for CONST).
            return (1 - sigma) * latent_image + sigma * noise

    class _Wrap:
        inner_model = _Inner()

        def __call__(self, x, sigma, model_options=None, seed=None):
            return torch.randn_like(x)  # any prediction; held tokens get pinned anyway

    model_k = samplers.KSamplerX0Inpaint(_Wrap(), sigmas)
    model_k.latent_image, model_k.noise = reference, noise
    after = torch.where(released, torch.ones_like(held), held)
    schedule = schedule_type(
        [(nodes_module.ttm._ttm_hold_until(sigmas, release_step), held), (float("-inf"), after)]
    )

    states = []
    final = k_sampling.sample_euler(
        model_k, noise.clone(), sigmas,
        extra_args={"denoise_mask": held, "model_options": {"denoise_mask_function": schedule},
                    "seed": 0},
        callback=lambda info: states.append(info["x"].clone()),
        disable=True,
    )

    kept = held == 0
    for step, x in enumerate(states[:release_step + 1]):
        sigma = sigmas[step]
        expected = (1 - sigma) * reference + sigma * noise
        assert torch.allclose(x[kept], expected[kept], atol=1e-5), step
    still_kept = kept & ~released
    assert torch.allclose(final[still_kept], reference[still_kept], atol=1e-5)
    assert not torch.allclose(final[released], reference[released], atol=1e-3)
