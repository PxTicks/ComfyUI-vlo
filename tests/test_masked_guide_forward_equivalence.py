"""The forked H3 forward pass must be stock-identical until the mask says otherwise.

This is the milestone the whole feature rests on: if the fork drifts from core
for reasons unrelated to masking, every later observation about masked guides is
uninterpretable. Re-run this after every ComfyUI update.
"""

from __future__ import annotations

import pytest
import torch

from minimax_h3_harness import guide_payload, masked_guide_module, tiny_h3_model, tiny_inputs


def _run(model, forward, payload, inputs):
    with torch.no_grad():
        return forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                       minimax_payload=payload)


def _core(model, payload, inputs):
    with torch.no_grad():
        return model._forward(inputs["x"], inputs["timestep"], inputs["context"], {},
                              minimax_payload=payload)


@pytest.fixture(scope="module")
def fork():
    return masked_guide_module("masked_h3_forward")


@pytest.fixture(scope="module")
def model():
    return tiny_h3_model()


def _tokens(payload):
    latent = payload["keyframes"][0]["latent"]
    return latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)


def test_fork_matches_core_without_masks(model, fork):
    """Step 6: the copied _forward reproduces core exactly on an unmasked guide."""
    inputs = tiny_inputs()
    expected = _core(model, guide_payload(), inputs)
    actual = _run(model, fork.masked_forward, guide_payload(), inputs)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def test_fully_open_mask_matches_stock_add_guide(model, fork):
    """Step 8, the hard acceptance requirement: mask == 1 everywhere is stock behaviour."""
    inputs = tiny_inputs()
    expected = _core(model, guide_payload(), inputs)
    strengths = torch.ones(_tokens(guide_payload()), dtype=torch.float64)
    actual = _run(model, fork.masked_forward, guide_payload(strengths), inputs)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def test_fully_open_mask_keeps_the_scalar_modulation_path(model, fork):
    """A uniform guide must not split the modulation table into per-token rows."""
    payload = guide_payload(torch.ones(_tokens(guide_payload()), dtype=torch.float64))
    plan = fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999)
    rows_t = plan.segment_rows_t[0]
    assert rows_t is not None and rows_t.unique().numel() == 1
    assert float(rows_t[0]) == max(0.5, 0.999)


def test_zero_mask_changes_the_output(model, fork):
    inputs = tiny_inputs()
    expected = _core(model, guide_payload(), inputs)
    strengths = torch.zeros(_tokens(guide_payload()), dtype=torch.float64)
    actual = _run(model, fork.masked_forward, guide_payload(strengths), inputs)
    assert not torch.equal(actual[0], expected[0])


def test_intermediate_masks_are_distinct_and_bracketed(model, fork):
    """Mask strength has to move the output continuously, not flip between two states."""
    inputs = tiny_inputs()
    n = _tokens(guide_payload())
    outs = {}
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        payload = guide_payload(torch.full((n,), s, dtype=torch.float64))
        outs[s] = _run(model, fork.masked_forward, payload, inputs)[0]
    for a, b in zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.0)):
        assert not torch.equal(outs[a], outs[b])
    # distance from the full-strength result should grow as the guide is trusted less
    distances = [float((outs[s] - outs[1.0]).abs().mean()) for s in (0.75, 0.5, 0.25, 0.0)]
    assert distances == sorted(distances)


def test_partial_mask_only_perturbs_through_the_guide(model, fork):
    """A half-open mask must land strictly between the fully open and fully closed cases."""
    inputs = tiny_inputs()
    n = _tokens(guide_payload())
    spatial = torch.zeros(n, dtype=torch.float64)
    spatial[: n // 2] = 1.0
    out = _run(model, fork.masked_forward, guide_payload(spatial), inputs)[0]
    open_out = _run(model, fork.masked_forward, guide_payload(torch.ones(n, dtype=torch.float64)), inputs)[0]
    closed_out = _run(model, fork.masked_forward, guide_payload(torch.zeros(n, dtype=torch.float64)), inputs)[0]
    assert not torch.equal(out, open_out) and not torch.equal(out, closed_out)


def test_stock_clock_differs_from_matched(model, fork):
    """The central experiment's A/B: latent corruption alone vs corruption + timestep."""
    inputs = tiny_inputs()
    n = _tokens(guide_payload())
    strengths = torch.full((n,), 0.4, dtype=torch.float64)
    with torch.no_grad():
        synced = fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                                     minimax_payload=guide_payload(strengths), clock="matched")
        noise_only = fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                                         minimax_payload=guide_payload(strengths), clock="stock")
    assert not torch.equal(synced[0], noise_only[0])


def test_noise_only_mode_keeps_the_stock_condition_timestep(model, fork):
    payload = guide_payload(torch.full((_tokens(guide_payload()),), 0.4, dtype=torch.float64))
    plan = fork.build_cond_row_plan(payload, t_v=0.5, vis_aug=0.999, clock="stock")
    assert plan.segment_rows_t == [None]
    assert plan.aug_rows is not None and float(plan.aug_rows.max()) < 0.999


def _chain(model, wrappers, payload, inputs, original=None):
    """Run a real DIFFUSION_MODEL wrapper chain, the way MiniMaxH3Model.forward does."""
    import comfy.patcher_extension as ext

    return ext.WrapperExecutor.new_class_executor(
        original if original is not None else model._forward, model, list(wrappers)
    ).execute(inputs["x"], inputs["timestep"], inputs["context"], {}, minimax_payload=payload)


def test_wrapper_bypasses_samples_without_masked_guides(model, fork):
    """Compatibility floor: an unmasked sample must reach the stock forward untouched."""
    inputs = tiny_inputs()
    wrapper = fork.make_diffusion_model_wrapper()
    with torch.no_grad():
        through = _chain(model, [wrapper], guide_payload(), inputs)
        stock = _core(model, guide_payload(), inputs)
    for got, want in zip(through, stock):
        assert torch.equal(got, want)


def test_wrapper_diverts_masked_samples(model, fork):
    inputs = tiny_inputs()
    strengths = torch.zeros(_tokens(guide_payload()), dtype=torch.float64)
    with torch.no_grad():
        through = _chain(model, [fork.make_diffusion_model_wrapper()], guide_payload(strengths), inputs)
        stock = _core(model, guide_payload(strengths), inputs)
    assert not torch.equal(through[0], stock[0])


def test_later_wrappers_still_run_on_a_masked_sample(model, fork):
    """The masked branch must stay in the chain: a wrapper added after this node
    (EasyCache and friends) would otherwise be silently skipped, making the result
    depend on which order the model patch nodes were chained in."""
    seen = []

    def later(executor, *args, **kwargs):
        seen.append(kwargs.get("minimax_payload"))
        out = executor(*args, **kwargs)
        return [out[0] * 0.0, out[1]]          # an unmistakable fingerprint

    inputs = tiny_inputs()
    strengths = torch.zeros(_tokens(guide_payload()), dtype=torch.float64)
    wrappers = [fork.make_diffusion_model_wrapper(), later]
    with torch.no_grad():
        out = _chain(model, wrappers, guide_payload(strengths), inputs)
    assert len(seen) == 1                      # the later wrapper actually ran
    assert torch.count_nonzero(out[0]) == 0    # and its effect survived


def test_earlier_wrappers_still_wrap_a_masked_sample(model, fork):
    """Order independence, from the other side."""
    seen = []

    def earlier(executor, *args, **kwargs):
        seen.append(True)
        return executor(*args, **kwargs)

    inputs = tiny_inputs()
    strengths = torch.zeros(_tokens(guide_payload()), dtype=torch.float64)
    with torch.no_grad():
        first = _chain(model, [earlier, fork.make_diffusion_model_wrapper()],
                       guide_payload(strengths), inputs)
        second = _chain(model, [fork.make_diffusion_model_wrapper(), earlier],
                        guide_payload(strengths), inputs)
    assert len(seen) == 2
    for got, want in zip(first, second):
        assert torch.equal(got, want)


def test_wrapper_bypasses_non_h3_models(fork):
    import comfy.patcher_extension as ext

    strengths = torch.zeros(_tokens(guide_payload()), dtype=torch.float64)
    executor = ext.WrapperExecutor.new_class_executor(
        lambda *a, **k: "stock", torch.nn.Linear(2, 2), [fork.make_diffusion_model_wrapper()])
    assert executor.execute(None, None, None, {}, minimax_payload=guide_payload(strengths)) == "stock"


def test_masked_guide_coexists_with_refs_and_a_denoise_mask(model, fork, caplog):
    """The combined path: masked guide + untouched reference + per-token video mask."""
    inputs = tiny_inputs()
    n = _tokens(guide_payload())
    strengths = torch.zeros(n, dtype=torch.float64)
    strengths[: n // 2] = 1.0

    ref_latent = torch.zeros(1, 24, 1, 2, 2)
    ref = {"kind": "image", "latent": ref_latent, "latent_h": 2, "latent_w": 2}

    def payload():
        p = guide_payload(strengths)
        p["refs"] = [ref]
        p["cond_video_latents"] = p["cond_video_latents"] + [ref_latent]
        return p

    denoise_mask = torch.ones(1, 1, 2, 4, 6)
    denoise_mask[..., :1, :] = 0.25          # part of the target still holds content
    with caplog.at_level("INFO"), torch.no_grad():
        out = fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                                  minimax_payload=payload(), denoise_mask=denoise_mask, debug=True)
        unmasked = model._forward(inputs["x"], inputs["timestep"], inputs["context"], {},
                                  minimax_payload=payload(), denoise_mask=denoise_mask)
    assert out[0].shape == unmasked[0].shape
    assert not torch.equal(out[0], unmasked[0])
    assert "Masked H3 Guide 0" in caplog.text
    assert "cond rows expected: {}".format(n) in caplog.text


def test_debug_report_is_emitted_once_per_sampling_run(model, fork, caplog):
    """A per-step report would drown the log over a 30 step sample."""
    inputs = tiny_inputs()
    payload = guide_payload(torch.zeros(_tokens(guide_payload()), dtype=torch.float64))
    with caplog.at_level("INFO"), torch.no_grad():
        for _ in range(3):
            fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                                minimax_payload=payload, debug=True)
    assert caplog.text.count("Masked H3 Guide 0") == 1


def test_debug_stays_silent_when_it_is_off(model, fork, caplog):
    inputs = tiny_inputs()
    payload = guide_payload(torch.zeros(_tokens(guide_payload()), dtype=torch.float64))
    with caplog.at_level("INFO"), torch.no_grad():
        fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"], {},
                            minimax_payload=payload)
    assert "Masked H3 Guide" not in caplog.text


# --- guide clips ----------------------------------------------------------
#
# `vloMiniMaxH3AddMaskedGuidesFromVideo` anchors multi-frame guides, so a guide
# latent can carry several time tokens. Core's layout already handles that; what
# these check is that the fork's per-row bookkeeping spans the whole clip instead
# of only its first latent frame.


def test_fully_open_clip_mask_matches_a_stock_guide_clip(model, fork):
    inputs = tiny_inputs()
    clip = dict(latent_t=3, frame_idx=2)
    expected = _core(model, guide_payload(**clip), inputs)
    strengths = torch.ones(_tokens(guide_payload(**clip)), dtype=torch.float64)
    actual = _run(model, fork.masked_forward, guide_payload(strengths, **clip), inputs)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def test_a_clip_mask_weights_each_latent_frame_on_its_own(model, fork):
    """A mask that closes down only the clip's last latent frame must not act like
    one that closes down its first: per-token rows span (t, h, w), not just (h, w)."""
    clip = dict(latent_t=3, frame_idx=2)
    inputs = tiny_inputs()
    rows = _tokens(guide_payload(**clip))
    per_frame = rows // 3

    head = torch.ones(rows, dtype=torch.float64)
    head[:per_frame] = 0.0
    tail = torch.ones(rows, dtype=torch.float64)
    tail[-per_frame:] = 0.0

    open_out = _run(model, fork.masked_forward, guide_payload(torch.ones(rows, dtype=torch.float64), **clip), inputs)
    head_out = _run(model, fork.masked_forward, guide_payload(head, **clip), inputs)
    tail_out = _run(model, fork.masked_forward, guide_payload(tail, **clip), inputs)
    assert not torch.equal(head_out[0], open_out[0])
    assert not torch.equal(tail_out[0], open_out[0])
    assert not torch.equal(head_out[0], tail_out[0])


# --- guide clocks ---------------------------------------------------------
#
# Four ways to turn a token's confidence into a condition timestep. They differ
# only in the floor the coefficient interpolates up from and in whether the
# modulation label is allowed to disagree with the latent the token carries.

CLOCKS = ("stock", "floored", "matched", "target_relative")
A_MAX = 0.999


def _plan(fork, strengths, t_v, clock, min_aug=0.0):
    payload = guide_payload(strengths, min_aug=min_aug)
    return fork.build_cond_row_plan(payload, t_v=t_v, vis_aug=A_MAX, clock=clock)


# timestep 0.5 -> sigma 0.0005 -> t_v 0.9995, i.e. past visual_cond_noise_aug.
# Core switches its condition label to t_v there; a clock that labels purely by
# coefficient would stay at 0.999 and silently stop being stock-identical, so the
# invariant has to be asserted in the tail and not only mid-schedule.
TAIL_TIMESTEP = 0.5


@pytest.mark.parametrize("clock", CLOCKS)
@pytest.mark.parametrize("timestep", [500.0, TAIL_TIMESTEP], ids=["mid_schedule", "t_v_past_vis_aug"])
def test_every_clock_keeps_a_fully_open_mask_bit_identical(model, fork, clock, timestep):
    """The invariant that outranks all four arms: mask == 1 everywhere is stock."""
    inputs = tiny_inputs()
    inputs["timestep"] = torch.tensor([timestep])
    expected = _core(model, guide_payload(), inputs)
    strengths = torch.ones(_tokens(guide_payload()), dtype=torch.float64)
    with torch.no_grad():
        actual = fork.masked_forward(model, inputs["x"], inputs["timestep"], inputs["context"],
                                     {}, minimax_payload=guide_payload(strengths), clock=clock)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def test_matched_pins_the_open_end_without_touching_the_closed_end(fork):
    """Pinning the endpoint must not leak into the tokens the mask closes down."""
    n = _tokens(guide_payload())
    t_v = 0.9995                                   # past vis_aug
    half = torch.cat([torch.ones(n // 2, dtype=torch.float64),
                      torch.zeros(n - n // 2, dtype=torch.float64)])
    rows = _plan(fork, half, t_v=t_v, clock="matched").segment_rows_t[0]
    assert float(rows[0]) == pytest.approx(t_v)    # open end: core's max(t_v, vis_aug)
    assert float(rows[-1]) == 0.0                  # closed end: still the noise it is


def test_target_relative_really_does_land_on_the_stock_scalar_in_the_tail(fork):
    """Once t_v passes vis_aug the floor caps at a_max, so *every* row collapses --
    and it has to collapse onto core's own label, t_v, not onto a_max."""
    n = _tokens(guide_payload())
    t_v = 0.9995
    half = torch.cat([torch.ones(n // 2, dtype=torch.float64),
                      torch.zeros(n - n // 2, dtype=torch.float64)])
    plan = _plan(fork, half, t_v=t_v, clock="target_relative")
    assert float(plan.aug_rows.min()) == A_MAX                  # corruption collapsed
    rows = plan.segment_rows_t[0]
    assert rows.unique().numel() == 1                           # one label for the segment
    assert float(rows[0]) == pytest.approx(t_v)                 # and it is core's


def test_matched_labels_a_token_as_noisy_as_it_actually_is(fork):
    """The arm the docstring always described: no floor, so label == coefficient."""
    n = _tokens(guide_payload())
    plan = _plan(fork, torch.zeros(n, dtype=torch.float64), t_v=0.43, clock="matched")
    assert float(plan.aug_rows.max()) == 0.0
    assert float(plan.segment_rows_t[0].max()) == 0.0      # honest: pure noise is t=0


def test_floored_labels_a_pure_noise_token_as_clean_as_the_target(fork):
    """Core's guard carried over. The lie this arm tells is the whole reason for the others."""
    n = _tokens(guide_payload())
    plan = _plan(fork, torch.zeros(n, dtype=torch.float64), t_v=0.43, clock="floored")
    assert float(plan.aug_rows.max()) == 0.0               # content is still pure noise
    assert float(plan.segment_rows_t[0].min()) == pytest.approx(0.43)


def test_target_relative_puts_a_zero_confidence_token_level_with_the_target(fork):
    """core's `t = 1 - m*sigma` read backwards, with guide confidence as `1 - m`."""
    n = _tokens(guide_payload())
    for t_v in (0.0, 0.25, 0.43):
        plan = _plan(fork, torch.zeros(n, dtype=torch.float64), t_v=t_v, clock="target_relative")
        # content and label both sit at t_v -- no marginal information, and no lie
        assert float(plan.aug_rows.max()) == pytest.approx(t_v)
        assert float(plan.segment_rows_t[0].max()) == pytest.approx(t_v)


def test_target_relative_leaves_the_open_end_of_the_mask_pinned(fork):
    """Raising the floor must not drag the trusted end off `visual_cond_noise_aug`."""
    n = _tokens(guide_payload())
    s = torch.ones(n, dtype=torch.float64)
    plan = _plan(fork, s, t_v=0.43, clock="target_relative")
    assert float(plan.aug_rows.min()) == A_MAX


def test_target_relative_collapses_to_stock_once_the_target_overtakes_the_guide(fork):
    """Late in sampling t_v passes the condition timestep; the mask then stops mattering."""
    n = _tokens(guide_payload())
    plan = _plan(fork, torch.zeros(n, dtype=torch.float64), t_v=0.9999, clock="target_relative")
    assert float(plan.aug_rows.min()) == A_MAX             # floor capped at a_max
    assert plan.segment_rows_t[0].unique().numel() == 1    # -> back on the scalar path


def test_min_aug_still_raises_the_floor_under_every_clock(fork):
    n = _tokens(guide_payload())
    for clock in CLOCKS:
        plan = _plan(fork, torch.zeros(n, dtype=torch.float64), t_v=0.1, clock=clock, min_aug=0.3)
        assert float(plan.aug_rows.max()) == pytest.approx(0.3), clock


def test_the_four_clocks_are_actually_different(model, fork):
    inputs = tiny_inputs()
    strengths = torch.full((_tokens(guide_payload()),), 0.4, dtype=torch.float64)
    outs = {}
    for clock in CLOCKS:
        with torch.no_grad():
            outs[clock] = fork.masked_forward(
                model, inputs["x"], inputs["timestep"], inputs["context"], {},
                minimax_payload=guide_payload(strengths), clock=clock)[0]
    for a in CLOCKS:
        for b in CLOCKS:
            if a < b:
                assert not torch.equal(outs[a], outs[b]), "{} == {}".format(a, b)


def test_an_unknown_clock_is_refused(fork):
    with pytest.raises(ValueError, match="unknown guide clock"):
        _plan(fork, torch.zeros(_tokens(guide_payload()), dtype=torch.float64),
              t_v=0.1, clock="honest")
