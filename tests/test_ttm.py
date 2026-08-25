"""vloTimeToMove's outer_sample wrapper, on single-stream and packed AV latents."""

from __future__ import annotations

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


AUDIO_FPS = 40
VIDEO_FPS = 24


class _FakeModel:
    def __init__(self, latent_format):
        self.latent_format = latent_format


class _FakePatcher:
    def __init__(self, model):
        self.model = model
        self.load_device = torch.device("cpu")


class _FakeGuider:
    def __init__(self, model):
        self.model_patcher = _FakePatcher(model)
        self.model_options = {}


class _FakeExecutor:
    """Stands in for the next wrapper in the OUTER_SAMPLE chain."""

    def __init__(self, guider):
        self.class_obj = guider
        self.call = None

    def __call__(self, noise, latent_image, sampler, sigmas, denoise_mask, *args, **kwargs):
        self.call = {
            "noise": noise,
            "latent_image": latent_image,
            "sigmas": sigmas,
            "denoise_mask": denoise_mask,
            "kwargs": kwargs,
        }
        return latent_image


def _h3_latent_t(frame_count):
    return (frame_count - 5) // 17 * 5 + 2


def _minimax_streams(nodes_module, frame_count=73, width=832, height=480):
    latent_t = _h3_latent_t(frame_count)
    audio_t = round(frame_count / VIDEO_FPS * AUDIO_FPS)
    video = torch.randn(1, 24, latent_t, height // 16, width // 16)
    audio = torch.randn(1, 32, 2, audio_t)
    return video, audio


def _ramp_mask(frames, height, width):
    """Frame f is filled with f, so an aligned mask reveals exactly which frames it read."""
    return torch.arange(frames, dtype=torch.float32).view(frames, 1, 1).expand(
        frames, height, width
    ).contiguous()


def _run(nodes_module, *, model, latent_image, latent_shapes, mask,
         reference_latents, start_step=1, end_step=3, steps=10):
    ttm = nodes_module.ttm
    guider = _FakeGuider(model)
    executor = _FakeExecutor(guider)
    wrapper = ttm._TTMOuterSample(reference_latents, mask, start_step, end_step)
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    noise = torch.randn_like(latent_image)
    out = wrapper(
        executor, noise, latent_image, object(), sigmas, None,
        None, False, 0, latent_shapes=latent_shapes,
    )
    return guider, executor, out


# --- H3 temporal geometry ---------------------------------------------------


@pytest.mark.parametrize(
    "frame_count,latent_frames", [(5, 2), (73, 22), (90, 27), (124, 37)]
)
def test_h3_frame_counts_round_trip(nodes_module, frame_count, latent_frames):
    """The 17k + 5 -> 5k + 2 mapping, both directions, against ComfyUI's own formula."""
    ttm = nodes_module.ttm
    assert _h3_latent_t(frame_count) == latent_frames
    assert ttm._h3_latent_frames(frame_count) == latent_frames
    assert ttm._h3_source_frames(latent_frames) == frame_count


def test_h3_source_frames_rejects_off_grid_counts(nodes_module):
    assert nodes_module.ttm._h3_source_frames(23) is None
    assert nodes_module.ttm._h3_source_frames(1) is None


def test_h3_anchor_frames_restart_at_every_clip(nodes_module):
    """Each 17-frame clip restarts the grouping: 0, 4, 8, 12, 16 then 17, 21, ..."""
    anchors = nodes_module.ttm._h3_anchor_frames(22, 73)
    assert anchors == [
        0, 4, 8, 12, 16,
        17, 21, 25, 29, 33,
        34, 38, 42, 46, 50,
        51, 55, 59, 63, 67,
        68, 72,
    ]
    # Every anchor lands inside the real footage, and the last one reaches its end.
    assert anchors[-1] == 72
    assert anchors == sorted(anchors)


def test_h3_align_mask_reads_the_clip_anchors(nodes_module):
    """A moving mask must not smear: each latent frame reads one specific source frame."""
    ttm = nodes_module.ttm
    aligned = ttm._ttm_align_mask_minimax(_ramp_mask(73, 30, 52), torch.Size((1, 24, 22, 30, 52)))
    assert tuple(aligned.shape) == (1, 1, 22, 30, 52)
    read = aligned[0, 0, :, 0, 0].tolist()
    assert read == [float(f) for f in ttm._h3_anchor_frames(22, 73)]


def test_h3_align_mask_ignores_frames_the_vae_drops(nodes_module):
    """77 and 73 frames both encode to 22 latents, so both must align identically."""
    ttm = nodes_module.ttm
    shape = torch.Size((1, 24, 22, 30, 52))
    short = ttm._ttm_align_mask_minimax(_ramp_mask(73, 30, 52), shape)
    long = ttm._ttm_align_mask_minimax(_ramp_mask(77, 30, 52), shape)
    assert torch.equal(short, long)


def test_h3_align_mask_expands_a_single_frame(nodes_module):
    aligned = nodes_module.ttm._ttm_align_mask_minimax(
        torch.ones(1, 30, 52), torch.Size((1, 24, 27, 30, 52))
    )
    assert tuple(aligned.shape) == (1, 1, 27, 30, 52)
    assert torch.equal(aligned, torch.ones_like(aligned))


def test_h3_align_mask_warns_on_a_mismatched_frame_count(nodes_module, caplog):
    ttm = nodes_module.ttm
    with caplog.at_level("WARNING"):
        aligned = ttm._ttm_align_mask_minimax(
            _ramp_mask(90, 30, 52), torch.Size((1, 24, 22, 30, 52))
        )
    assert tuple(aligned.shape) == (1, 1, 22, 30, 52)
    assert "27 MiniMax H3 latent frames" in caplog.text


# --- packed AV sampling -----------------------------------------------------


def test_minimax_reference_replaces_video_and_keeps_audio(nodes_module):
    """The previously failing workflow: the AV latent must stay packed and unpackable."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])
    reference = torch.randn_like(video)

    guider, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=_ramp_mask(73, 480, 832).clamp(0.0, 1.0),
        reference_latents={"samples": reference},
    )

    out_video, out_audio = comfy.utils.unpack_latents(
        executor.call["latent_image"], latent_shapes
    )
    assert torch.equal(out_video, reference)
    assert torch.equal(out_audio, audio)


def test_minimax_denoise_mask_packs_video_and_audio(nodes_module):
    """Reproduces the shape '[1, 24, ...]' is invalid failure: the mask must unpack."""
    comfy = nodes_module.comfy
    ttm = nodes_module.ttm
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    mask = torch.zeros(73, 480, 832)
    mask[:, 100:300, 200:500] = 1.0

    guider, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=mask,
        reference_latents={"samples": torch.randn_like(video)},
    )

    # Every phase's mask must unpack cleanly, audio included.
    for sigma_floor, packed_mask in _phases(guider):
        assert packed_mask.shape == latent_image.shape
        video_mask, audio_mask = comfy.utils.unpack_latents(packed_mask, latent_shapes)
        assert tuple(video_mask.shape) == tuple(video.shape)
        assert tuple(audio_mask.shape) == tuple(audio.shape)
        # Audio denoises normally in every phase.
        assert torch.equal(audio_mask, torch.ones_like(audio_mask))

    seed_mask, ttm_mask = (comfy.utils.unpack_latents(m, latent_shapes)[0]
                           for _, m in _phases(guider))
    # Seeding the reference holds the whole video stream; the TTM window holds the region.
    assert torch.equal(seed_mask, torch.zeros_like(seed_mask))
    expected = 1.0 - ttm._ttm_align_mask_minimax(mask, latent_shapes[0])
    assert torch.equal(ttm_mask, expected.expand_as(ttm_mask))
    assert ttm_mask.min() == 0.0 and ttm_mask.max() == 1.0


def test_minimax_closed_window_still_packs_the_reference(nodes_module):
    """end_step <= start_step seeds the init only -- and must not unpack the AV latent."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module, frame_count=90)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])
    reference = torch.randn_like(video)

    guider, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(90, 480, 832),
        reference_latents={"samples": reference},
        start_step=2,
        end_step=2,
    )

    # No TTM window, but the reference still has to be seeded -- and on a packed latent
    # that is a hold over the whole video stream, not a slice of the shared schedule.
    assert len(_phases(guider)) == 1
    seed_video, seed_audio = comfy.utils.unpack_latents(
        _phases(guider)[0][1], latent_shapes
    )
    assert torch.equal(seed_video, torch.zeros_like(seed_video))
    assert torch.equal(seed_audio, torch.ones_like(seed_audio))
    out_video, out_audio = comfy.utils.unpack_latents(
        executor.call["latent_image"], latent_shapes
    )
    assert torch.equal(out_video, reference)
    assert torch.equal(out_audio, audio)


def test_minimax_reference_shape_is_validated_against_the_video_stream(nodes_module):
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    with pytest.raises(ValueError, match="must match the sampled video latent"):
        _run(
            nodes_module,
            model=_FakeModel(latent_formats.MiniMaxH3AV()),
            latent_image=latent_image,
            latent_shapes=latent_shapes,
            mask=torch.ones(73, 480, 832),
            reference_latents={"samples": torch.randn(1, 24, 27, 30, 52)},
        )


def test_minimax_accepts_a_nested_av_reference(nodes_module):
    """An AV-encoded reference contributes its video stream; the sampler keeps its audio."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])
    reference_video = torch.randn_like(video)
    reference = comfy.nested_tensor.NestedTensor(
        (reference_video, torch.randn_like(audio))
    )

    _, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(73, 480, 832),
        reference_latents={"samples": reference},
    )

    out_video, out_audio = comfy.utils.unpack_latents(
        executor.call["latent_image"], latent_shapes
    )
    assert torch.equal(out_video, reference_video)
    assert torch.equal(out_audio, audio)


# --- single-stream behaviour is unchanged -----------------------------------


class _WanLatentFormat:
    temporal_downscale_ratio = 4


def test_single_stream_path_is_unchanged(nodes_module):
    latent = torch.randn(1, 16, 21, 30, 52)
    reference = torch.randn_like(latent)
    mask = torch.zeros(81, 480, 832)
    mask[:, :100] = 1.0

    _, executor, _ = _run(
        nodes_module,
        model=_FakeModel(_WanLatentFormat()),
        latent_image=latent,
        latent_shapes=[latent.shape],
        mask=mask,
        reference_latents={"samples": reference},
    )

    assert torch.equal(executor.call["latent_image"], reference)
    # Still the broadcastable [1, 1, T, H, W] mask, not a channel-expanded packed one.
    assert tuple(executor.call["denoise_mask"].shape) == (1, 1, 21, 30, 52)


def test_single_stream_closed_window_seeds_the_init_only(nodes_module):
    latent = torch.randn(1, 16, 21, 30, 52)
    reference = torch.randn_like(latent)

    _, executor, _ = _run(
        nodes_module,
        model=_FakeModel(_WanLatentFormat()),
        latent_image=latent,
        latent_shapes=[latent.shape],
        mask=torch.ones(81, 480, 832),
        reference_latents={"samples": reference},
        start_step=3,
        end_step=1,
    )

    assert executor.call["denoise_mask"] is None
    assert torch.equal(executor.call["latent_image"], reference)


def test_start_step_beyond_the_schedule_is_rejected(nodes_module):
    latent = torch.randn(1, 16, 21, 30, 52)
    with pytest.raises(ValueError, match="must be less than the 10 steps"):
        _run(
            nodes_module,
            model=_FakeModel(_WanLatentFormat()),
            latent_image=latent,
            latent_shapes=[latent.shape],
            mask=torch.ones(81, 480, 832),
            reference_latents={"samples": torch.randn_like(latent)},
            start_step=10,
        )


def test_noiseless_sampler_is_left_alone(nodes_module, caplog):
    latent = torch.randn(1, 16, 21, 30, 52)
    ttm = nodes_module.ttm
    guider = _FakeGuider(_FakeModel(_WanLatentFormat()))
    executor = _FakeExecutor(guider)
    wrapper = ttm._TTMOuterSample(
        {"samples": torch.randn_like(latent)}, torch.ones(81, 480, 832), 1, 3
    )
    with caplog.at_level("WARNING"):
        wrapper(
            executor, torch.zeros_like(latent), latent, object(),
            torch.linspace(1.0, 0.0, 11), None, latent_shapes=[latent.shape],
        )
    assert torch.equal(executor.call["latent_image"], latent)
    assert "adds no noise" in caplog.text


# --- the release schedule ---------------------------------------------------


def _phases(guider):
    return guider.model_options["denoise_mask_function"].phases


def _hold_sigmas(guider):
    return [sigma_floor for sigma_floor, _ in _phases(guider)]


@pytest.mark.parametrize("start_step,end_step", [(1, 3), (2, 5), (0, 1)])
def test_release_sigma_is_the_step_before_end_step(nodes_module, start_step, end_step):
    """end_step counts the way the TTM reference does: the last pinned call is end_step - 1."""
    latent = torch.randn(1, 16, 21, 30, 52)
    sigmas = torch.linspace(1.0, 0.0, 11)

    guider, _, _ = _run(
        nodes_module,
        model=_FakeModel(_WanLatentFormat()),
        latent_image=latent,
        latent_shapes=[latent.shape],
        mask=torch.ones(81, 480, 832),
        reference_latents={"samples": torch.randn_like(latent)},
        start_step=start_step,
        end_step=end_step,
    )
    assert _hold_sigmas(guider) == pytest.approx([float(sigmas[end_step - 1])])


@pytest.mark.parametrize("start_step,end_step", [(1, 3), (2, 5), (3, 3)])
def test_packed_hold_sigmas_are_absolute_not_relative(nodes_module, start_step, end_step):
    """Nothing is sliced away, so both holds index the schedule the sampler was given."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])
    sigmas = torch.linspace(1.0, 0.0, 11)

    guider, _, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(73, 480, 832),
        reference_latents={"samples": torch.randn_like(video)},
        start_step=start_step,
        end_step=end_step,
    )

    expected = [float(sigmas[start_step - 1])] if start_step > 0 else []
    if end_step > start_step:
        expected.append(float(sigmas[end_step - 1]))
    assert _hold_sigmas(guider) == pytest.approx(expected)


def test_schedule_releases_on_sigma_not_on_a_step_counter(nodes_module):
    """Solvers that evaluate off-schedule must release at the same sigma, not a step index."""
    held = torch.zeros(1, 1, 4, 2, 2)
    schedule = nodes_module.ttm._TTMDenoiseMaskSchedule([(0.75, held)])

    # On-schedule and off-schedule sigmas above the release point keep the hold.
    for sigma in (1.0, 0.9, 0.8, 0.75):
        assert torch.equal(schedule(torch.tensor([sigma]), held), held)
    # Anything below it, including an intermediate evaluation, releases.
    for sigma in (0.749, 0.5, 0.0):
        assert torch.equal(schedule(torch.tensor([sigma]), held), torch.ones_like(held))


def test_schedule_hands_back_each_phase_in_turn(nodes_module):
    """Two windows on one clock: seed the reference, then hold only the TTM region."""
    seed = torch.zeros(1, 1, 4, 2, 2)
    window = torch.tensor([0.0, 1.0]).repeat(1, 1, 4, 2, 1)
    schedule = nodes_module.ttm._TTMDenoiseMaskSchedule([(0.75, seed), (0.5, window)])

    for sigma in (1.0, 0.9, 0.75):
        assert schedule.mask_at(torch.tensor([sigma])) is seed
    for sigma in (0.749, 0.6, 0.5):
        assert schedule.mask_at(torch.tensor([sigma])) is window
    for sigma in (0.499, 0.0):
        assert schedule.mask_at(torch.tensor([sigma])) is None
        assert torch.equal(schedule(torch.tensor([sigma]), seed), torch.ones_like(seed))


def test_schedule_release_covers_every_packed_stream(nodes_module):
    """Release must free the whole packed mask, video and audio alike."""
    comfy = nodes_module.comfy
    packed, shapes = comfy.utils.pack_latents(
        [torch.zeros(1, 24, 22, 30, 52), torch.ones(1, 32, 2, 121)]
    )
    schedule = nodes_module.ttm._TTMDenoiseMaskSchedule([(0.5, packed)])
    released = schedule(torch.tensor([0.1]), packed)
    video_mask, audio_mask = comfy.utils.unpack_latents(released, shapes)
    assert torch.equal(video_mask, torch.ones_like(video_mask))
    assert torch.equal(audio_mask, torch.ones_like(audio_mask))


# --- MiniMax H3's model-side token clock ------------------------------------


def _h3_mask_model(nodes_module):
    """A MiniMaxH3 stand-in carrying the real mask-conditioning methods, minus the weights.

    Those methods only need the DiT's patch size, so binding them onto a stub exercises
    ComfyUI's own pooling and quantisation rather than a copy of it.
    """
    from types import SimpleNamespace

    model_base = __import__("comfy.model_base", fromlist=["MiniMaxH3"])
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    h3 = model_base.MiniMaxH3

    class _StubH3:
        _pool_masks_to_token_grid = h3._pool_masks_to_token_grid
        _token_grid_masks = h3._token_grid_masks
        _denoise_mask_values = h3._denoise_mask_values

        def __init__(self):
            self.latent_format = latent_formats.MiniMaxH3AV()
            self.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))

    return _StubH3()


class _FakeApplyExecutor:
    def __init__(self, model):
        self.class_obj = model
        self.call = None

    def __call__(self, x, sigma, *args, **kwargs):
        self.call = {"x": x, "sigma": sigma, "args": args, "kwargs": kwargs}
        return x


HELD_LATENT_FRAMES = 11  # the first half of a 22-frame clip


def _packed_ttm_mask(nodes_module):
    comfy = nodes_module.comfy
    video_hold = torch.zeros(1, 24, 22, 30, 52)
    video_hold[:, :, :HELD_LATENT_FRAMES] = 1.0
    packed, shapes = comfy.utils.pack_latents(
        [1.0 - video_hold, torch.ones(1, 32, 2, 121)]
    )
    return packed, shapes


def test_h3_token_clock_is_pinned_while_the_window_is_open(nodes_module):
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    schedule = ttm._TTMDenoiseMaskSchedule([(0.5, denoise_mask)])
    wrapper = ttm._TTMMinimaxModelMaskWrapper(schedule, shapes)

    executor = _FakeApplyExecutor(model)
    wrapper(executor, torch.zeros(1, 1, 4), torch.tensor([0.9]), None, None, None, {})

    video_mask = executor.call["kwargs"]["denoise_mask"]
    # H3 reads [0, 0] of a [1, 1, T, H, W] video mask; held tokens must read 0.
    assert tuple(video_mask.shape) == (1, 1, 22, 30, 52)
    assert float(video_mask[0, 0, :HELD_LATENT_FRAMES].max()) == 0.0
    assert float(video_mask[0, 0, HELD_LATENT_FRAMES:].min()) == 1.0
    # Audio is never held, so H3 is left on its normal global clock.
    assert "audio_denoise_mask" not in executor.call["kwargs"]


def test_h3_token_clock_releases_at_the_same_sigma_as_the_sampler(nodes_module):
    """The whole point of the wrapper: one schedule, so the two sides cannot drift."""
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    schedule = ttm._TTMDenoiseMaskSchedule([(0.5, denoise_mask)])
    wrapper = ttm._TTMMinimaxModelMaskWrapper(schedule, shapes)

    for sigma in (1.0, 0.6, 0.5, 0.499, 0.2, 0.0):
        sigma_t = torch.tensor([sigma])
        executor = _FakeApplyExecutor(model)
        wrapper(executor, torch.zeros(1, 1, 4), sigma_t, None, None, None, {})

        sampler_held = not torch.equal(
            schedule(sigma_t, denoise_mask), torch.ones_like(denoise_mask)
        )
        model_held = "denoise_mask" in executor.call["kwargs"]
        assert sampler_held == model_held, sigma


def test_h3_token_clock_release_drops_the_cond_comfy_built(nodes_module):
    """Release must remove the frozen cond, not just leave a stale one in place."""
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    wrapper = ttm._TTMMinimaxModelMaskWrapper(
        ttm._TTMDenoiseMaskSchedule([(0.5, denoise_mask)]), shapes
    )

    executor = _FakeApplyExecutor(model)
    stale = model._denoise_mask_values(denoise_mask, shapes)
    wrapper(
        executor, torch.zeros(1, 1, 4), torch.tensor([0.1]), None, None, None, {},
        denoise_mask=stale["denoise_mask"], audio_denoise_mask=torch.zeros(1, 1, 2, 121),
    )
    assert "denoise_mask" not in executor.call["kwargs"]
    assert "audio_denoise_mask" not in executor.call["kwargs"]


def test_h3_token_clock_conds_match_comfys_own_pooling(nodes_module):
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    wrapper = ttm._TTMMinimaxModelMaskWrapper(
        ttm._TTMDenoiseMaskSchedule([(0.5, denoise_mask)]), shapes
    )

    executor = _FakeApplyExecutor(model)
    wrapper(executor, torch.zeros(1, 1, 4), torch.tensor([0.9]), None, None, None, {})
    expected = model._denoise_mask_values(denoise_mask, shapes)
    assert torch.equal(executor.call["kwargs"]["denoise_mask"], expected["denoise_mask"])


def test_h3_token_clock_pools_the_mask_only_once(nodes_module):
    """It is a per-evaluation hook, so the pooling pass must not run per evaluation."""
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    wrapper = ttm._TTMMinimaxModelMaskWrapper(
        ttm._TTMDenoiseMaskSchedule([(0.5, denoise_mask)]), shapes
    )

    calls = 0
    real = model._denoise_mask_values

    def counted(mask, latent_shapes):
        nonlocal calls
        calls += 1
        return real(mask, latent_shapes)

    model._denoise_mask_values = counted
    for _ in range(5):
        wrapper(
            _FakeApplyExecutor(model), torch.zeros(1, 1, 4), torch.tensor([0.9]),
            None, None, None, {},
        )
    assert calls == 1


def test_minimax_registers_the_token_clock_wrapper(nodes_module):
    """End to end: sampling H3 through TTM must leave an apply_model wrapper in place."""
    comfy = nodes_module.comfy
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    model = _h3_mask_model(nodes_module)
    guider, executor, _ = _run(
        nodes_module,
        model=model,
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(73, 480, 832),
        reference_latents={"samples": torch.randn_like(video)},
    )

    wrappers = guider.model_options["transformer_options"]["wrappers"]
    registered = wrappers[comfy.patcher_extension.WrappersMP.APPLY_MODEL][
        nodes_module.ttm._TTM_APPLY_MODEL_KEY
    ]
    assert len(registered) == 1
    # Same schedule object on both sides, which is what keeps them from drifting.
    assert registered[0].schedule is guider.model_options["denoise_mask_function"]
    # ... and the sampler starts on that schedule's first phase.
    assert executor.call["denoise_mask"] is _phases(guider)[0][1]


def test_single_stream_does_not_register_the_token_clock_wrapper(nodes_module):
    latent = torch.randn(1, 16, 21, 30, 52)
    guider, _, _ = _run(
        nodes_module,
        model=_FakeModel(_WanLatentFormat()),
        latent_image=latent,
        latent_shapes=[latent.shape],
        mask=torch.ones(81, 480, 832),
        reference_latents={"samples": torch.randn_like(latent)},
    )
    assert "wrappers" not in guider.model_options.get("transformer_options", {})


def test_h3_without_per_token_masking_warns_instead_of_pinning_silently(nodes_module, caplog):
    """An H3 build with no mask conditioning still runs, but must say end_step is weakened."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    with caplog.at_level("WARNING"):
        guider, _, _ = _run(
            nodes_module,
            model=_FakeModel(latent_formats.MiniMaxH3AV()),  # no _denoise_mask_values
            latent_image=latent_image,
            latent_shapes=latent_shapes,
            mask=torch.ones(73, 480, 832),
            reference_latents={"samples": torch.randn_like(video)},
        )

    assert "token clock cannot be moved" in caplog.text
    assert "wrappers" not in guider.model_options.get("transformer_options", {})


def test_h3_reads_our_mask_as_a_per_token_clock(nodes_module):
    """The contract we rely on: mask value m puts a token at sigma = m * sigma_stream.

    Run the mask TTM produces through H3's own row-value pooling and timestep formula, so a
    change to either shows up here rather than as a silently wrong hold.
    """
    minimax = __import__("comfy.ldm.minimax.model", fromlist=["mask_row_values"])
    model = _h3_mask_model(nodes_module)
    denoise_mask, shapes = _packed_ttm_mask(nodes_module)
    video_mask = model._denoise_mask_values(denoise_mask, shapes)["denoise_mask"]

    rows = minimax.mask_row_values(video_mask[0, 0], 22, 30, 52)
    assert rows is not None

    sigma_v = torch.tensor(0.9)
    t_v = 1.0 - float(sigma_v)
    t_pin = max(t_v, minimax.VISUAL_COND_TIMESTEP)
    rows_t = (1.0 - rows * sigma_v).clamp(max=t_pin)

    # Two clocks only, and they are the extremes: held tokens pinned at the conditioning
    # timestep, free tokens on the model's normal global one.
    assert sorted(rows_t.unique().tolist()) == pytest.approx([t_v, t_pin])
    held_rows = HELD_LATENT_FRAMES * (30 // 2) * (52 // 2)
    assert float(rows_t[:held_rows].min()) == pytest.approx(t_pin)
    assert float(rows_t[held_rows:].max()) == pytest.approx(t_v)


def test_phase_boundaries_land_exactly_on_their_own_schedule_steps(nodes_module):
    """The last held step must be held: both sides read the boundary off one sigma tensor."""
    ttm = nodes_module.ttm
    sigmas = torch.linspace(1.0, 0.0, 31)
    sigmas = sigmas / (12.0 - 11.0 * sigmas).clamp(min=1e-6) * 12.0  # H3's shift-12 spacing
    held = torch.zeros(1, 1, 4, 2, 2)

    for step in (1, 2, 7, 29):
        schedule = ttm._TTMDenoiseMaskSchedule([(float(sigmas[step - 1]), held)])
        # Held through the step before the cutoff, free from the cutoff step onward.
        assert schedule.mask_at(sigmas[step - 1:step]) is held
        assert schedule.mask_at(sigmas[step:step + 1]) is None


# --- audio keeps the shared schedule ----------------------------------------


@pytest.mark.parametrize("start_step", [1, 3, 8])
def test_packed_sampling_keeps_the_whole_shared_schedule(nodes_module, start_step):
    """H3 clocks audio off the shared sigma, so slicing to start_step would rob audio too."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])
    sigmas = torch.linspace(1.0, 0.0, 31)

    _, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(73, 480, 832),
        reference_latents={"samples": torch.randn_like(video)},
        start_step=start_step,
        end_step=start_step + 2,
        steps=30,
    )

    assert torch.equal(executor.call["sigmas"], sigmas)
    # Sampling still starts at sigma_max, so the audio stream is seeded with real full-
    # strength noise rather than dropped in part-denoised against a zero clean component.
    assert float(executor.call["sigmas"][0]) == 1.0


@pytest.mark.parametrize("start_step", [1, 3])
def test_single_stream_still_skips_to_start_step(nodes_module, start_step):
    """No second stream to strand, so the cheap path stays: skip the steps outright."""
    latent = torch.randn(1, 16, 21, 30, 52)
    sigmas = torch.linspace(1.0, 0.0, 31)

    _, executor, _ = _run(
        nodes_module,
        model=_FakeModel(_WanLatentFormat()),
        latent_image=latent,
        latent_shapes=[latent.shape],
        mask=torch.ones(81, 480, 832),
        reference_latents={"samples": torch.randn_like(latent)},
        start_step=start_step,
        end_step=start_step + 2,
        steps=30,
    )
    assert torch.equal(executor.call["sigmas"], sigmas[start_step:])


def test_packed_seed_phase_holds_video_and_never_audio(nodes_module):
    """The seed is bought with a video hold, so audio has to stay free throughout it."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    mask = torch.zeros(73, 480, 832)
    mask[:, 100:300, 200:500] = 1.0

    guider, _, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=mask,
        reference_latents={"samples": torch.randn_like(video)},
        start_step=2,
        end_step=5,
    )

    schedule = guider.model_options["denoise_mask_function"]
    sigmas = torch.linspace(1.0, 0.0, 11)
    for step in range(10):
        current = schedule(sigmas[step:step + 1], _phases(guider)[0][1])
        video_mask, audio_mask = comfy.utils.unpack_latents(current, latent_shapes)
        assert torch.equal(audio_mask, torch.ones_like(audio_mask)), step
        if step < 2:
            assert float(video_mask.max()) == 0.0, step        # whole stream held
        elif step < 5:
            assert 0.0 == float(video_mask.min()) < float(video_mask.max()) == 1.0, step
        else:
            assert float(video_mask.min()) == 1.0, step        # free


def test_packed_start_step_zero_has_no_seed_phase(nodes_module):
    """start_step 0 seeds nothing, so there is no video-wide hold to pay for."""
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video, audio = _minimax_streams(nodes_module)
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    guider, _, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=torch.ones(73, 480, 832),
        reference_latents={"samples": torch.randn_like(video)},
        start_step=0,
        end_step=2,
    )
    assert len(_phases(guider)) == 1


def test_h3_token_clock_follows_the_seed_phase_too(nodes_module):
    """Each phase gets its own conditioning, so the seed hold is not a stale TTM mask."""
    comfy = nodes_module.comfy
    ttm = nodes_module.ttm
    model = _h3_mask_model(nodes_module)
    video_hold = torch.zeros(1, 24, 22, 30, 52)
    video_hold[:, :, :HELD_LATENT_FRAMES] = 1.0
    seed, shapes = comfy.utils.pack_latents(
        [torch.zeros(1, 24, 22, 30, 52), torch.ones(1, 32, 2, 121)]
    )
    window, _ = comfy.utils.pack_latents([1.0 - video_hold, torch.ones(1, 32, 2, 121)])
    schedule = ttm._TTMDenoiseMaskSchedule([(0.75, seed), (0.5, window)])
    wrapper = ttm._TTMMinimaxModelMaskWrapper(schedule, shapes)

    def conds_at(sigma):
        executor = _FakeApplyExecutor(model)
        wrapper(executor, torch.zeros(1, 1, 4), torch.tensor([sigma]), None, None, None, {})
        return executor.call["kwargs"]

    # Seeding: every video token pinned. Window: only the held region. After: no mask.
    assert float(conds_at(0.9)["denoise_mask"].max()) == 0.0
    in_window = conds_at(0.6)["denoise_mask"]
    assert float(in_window[0, 0, :HELD_LATENT_FRAMES].max()) == 0.0
    assert float(in_window[0, 0, HELD_LATENT_FRAMES:].min()) == 1.0
    assert "denoise_mask" not in conds_at(0.1)


# --- mask polarity and short-mask alignment ---------------------------------


def test_mask_tooltip_matches_the_implemented_polarity(nodes_module):
    """The tooltip told users to invert the mask; the code was right, the words were not."""
    schema = nodes_module.vloTimeToMove.define_schema()
    tooltip = {i.id: i.tooltip for i in schema.inputs}["mask"]
    assert "white background" not in tooltip.lower()
    assert "white over the dragged subject" in tooltip


def test_h3_align_mask_warns_when_a_same_bucket_mask_is_short(nodes_module, caplog):
    """69-72 frames also encode to 22 latents, so only a warning can catch the mismatch."""
    ttm = nodes_module.ttm
    with caplog.at_level("WARNING"):
        aligned = ttm._ttm_align_mask_minimax(
            _ramp_mask(70, 30, 52), torch.Size((1, 24, 22, 30, 52))
        )
    assert tuple(aligned.shape) == (1, 1, 22, 30, 52)
    assert "short by 3" in caplog.text

    # The tail really does repeat, which is what the warning is about.
    read = aligned[0, 0, :, 0, 0].tolist()
    assert read[-2:] == [68.0, 69.0]  # canonical would have been 68, 72


def test_h3_align_mask_stays_quiet_on_the_canonical_length(nodes_module, caplog):
    ttm = nodes_module.ttm
    with caplog.at_level("WARNING"):
        ttm._ttm_align_mask_minimax(_ramp_mask(73, 30, 52), torch.Size((1, 24, 22, 30, 52)))
    assert caplog.text == ""


def test_reported_73_frame_512x288_run_packs_its_masks(nodes_module):
    """The exact shapes from the reported failure.

    A pre-fix build handed the sampler a bare [1, 1, 22, 18, 32] video mask; H3's
    per-token mask conditioning then tried to unpack it against the packed AV
    latent_shapes and raised "shape '[1, 24, 22, 18, 32]' is invalid for input of
    size 12672" -- 12672 being 22*18*32, one channel of video and no audio stream.
    """
    comfy = nodes_module.comfy
    latent_formats = __import__("comfy.latent_formats", fromlist=["MiniMaxH3AV"])
    video = torch.randn(1, 24, 22, 18, 32)          # 73 frames at 512x288
    audio = torch.randn(1, 32, 2, round(73 / VIDEO_FPS * AUDIO_FPS))
    latent_image, latent_shapes = comfy.utils.pack_latents([video, audio])

    mask = torch.zeros(73, 288, 512)
    mask[:, 60:220, 120:400] = 1.0

    guider, executor, _ = _run(
        nodes_module,
        model=_FakeModel(latent_formats.MiniMaxH3AV()),
        latent_image=latent_image,
        latent_shapes=latent_shapes,
        mask=mask,
        reference_latents={"samples": torch.randn_like(video)},
    )

    bare_video_mask_numel = 22 * 18 * 32
    for _, packed_mask in _phases(guider):
        assert packed_mask.numel() != bare_video_mask_numel
        # The failing call, verbatim: this is what H3's extra_conds does.
        video_mask, audio_mask = comfy.utils.unpack_latents(packed_mask, latent_shapes)
        assert tuple(video_mask.shape) == (1, 24, 22, 18, 32)
        assert torch.equal(audio_mask, torch.ones_like(audio_mask))

    assert executor.call["denoise_mask"] is _phases(guider)[0][1]
