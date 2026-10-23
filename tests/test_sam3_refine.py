from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


@pytest.fixture(scope="module")
def refine(nodes_module):
    return importlib.import_module(nodes_module.__name__ + ".sam3_refine")


LOW_RES = 8


class FakeBackbone:
    def __init__(self, multiplex=True):
        self.multiplex = multiplex
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append((tuple(frame.shape), kwargs))
        features = [torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 1, 1)]
        return None, None, features, None


class FakeTracker:
    """Records every heads call and answers with a fixed mask per output kind."""

    def __init__(self, single, multi):
        self.single, self.multi = single, multi
        self.interactivity_no_mem_embed = torch.full((1, 1, 3), 2.0)
        self.no_mem_embed = torch.full((1, 1, 3), 4.0)
        self.calls = []

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None,
                           box_inputs=None, high_res_features=None, multimask_output=False):
        self.calls.append({
            "backbone_features": backbone_features, "high_res_features": high_res_features,
            "point_inputs": point_inputs, "mask_inputs": mask_inputs,
            "multimask_output": multimask_output,
        })
        low = (self.multi if multimask_output else self.single).clone()
        return low, None, None, None


def _logits(value):
    return torch.full((1, 1, LOW_RES, LOW_RES), float(value))


def _left_half():
    logits = _logits(-8.0)
    logits[..., : LOW_RES // 2] = 8.0
    return logits


def _unstable():
    # Mostly in (-delta, +delta]: a large union over a small intersection.
    logits = _logits(0.01)
    logits[..., 0, 0] = 8.0
    return logits


@pytest.fixture
def fake(refine, monkeypatch):
    import comfy.model_management as management

    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(management, "throw_exception_if_processing_interrupted", lambda: None)
    monkeypatch.setattr(refine, "_warned", set())

    def build(single=None, multi=None, multiplex=True, scalp=0):
        backbone = FakeBackbone(multiplex)
        tracker = FakeTracker(_left_half() if single is None else single,
                              _logits(8.0) if multi is None else multi)
        segment_calls = []

        def forward_segment(images, point_inputs=None, box_inputs=None, mask_inputs=None):
            segment_calls.append({"images": images, "point_inputs": point_inputs,
                                  "mask_inputs": mask_inputs})
            return _left_half()

        sam = SimpleNamespace(
            detector=SimpleNamespace(backbone={"vision_backbone": backbone}, scalp=scalp),
            tracker=tracker, forward_segment=forward_segment)
        model = SimpleNamespace(model=SimpleNamespace(
            diffusion_model=sam, get_dtype=lambda: torch.float32))
        return SimpleNamespace(model=model, sam=sam, backbone=backbone, tracker=tracker,
                               segment_calls=segment_calls)

    return build


@pytest.fixture
def core_changed(refine, monkeypatch):
    """Make the installed core look different from the verified one."""
    monkeypatch.delenv(refine.OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(refine, "TESTED_SOURCE_FINGERPRINT", "0" * 64)


IMAGE = torch.zeros(1, 4, 8, 3)  # 4 x 8 (H x W)
ONE_POINT = '[{"x": 2, "y": 1}]'
TWO_POINTS = '[{"x": 2, "y": 1}, {"x": 6, "y": 3}]'


def _run(refine, model, **kwargs):
    return refine.vloSAM3RefineMask.execute(model, IMAGE, **kwargs).result[0]


def test_one_point_uses_the_best_of_three_like_meta(refine, fake):
    sam = fake()
    _run(refine, sam.model, positive_coords=ONE_POINT)
    assert [call["multimask_output"] for call in sam.tracker.calls] == [True]


def test_one_exclude_point_on_a_mask_uses_multimask_too(refine, fake):
    # Meta's policy counts points, not whether a mask prompt is present.
    sam = fake()
    _run(refine, sam.model, negative_coords=ONE_POINT, mask=torch.ones(1, 4, 8))
    assert [call["multimask_output"] for call in sam.tracker.calls] == [True]


def test_stable_single_mask_is_kept_for_several_points(refine, fake):
    sam = fake()
    out = _run(refine, sam.model, positive_coords=TWO_POINTS)
    assert [call["multimask_output"] for call in sam.tracker.calls] == [False]
    assert out[0, :, :4].eq(1).all() and out[0, :, 4:].eq(0).all()


def test_unstable_single_mask_falls_back_to_multimask(refine, fake):
    sam = fake(single=_unstable(), multi=_left_half())
    out = _run(refine, sam.model, positive_coords=TWO_POINTS)
    assert [call["multimask_output"] for call in sam.tracker.calls] == [False, True]
    assert out[0, :, :4].eq(1).all() and out[0, :, 4:].eq(0).all()


def test_forced_modes_skip_the_policy(refine, fake):
    sam = fake(single=_unstable())
    _run(refine, sam.model, positive_coords=ONE_POINT, mask_selection="single")
    _run(refine, sam.model, positive_coords=TWO_POINTS, mask_selection="multimask")
    assert [call["multimask_output"] for call in sam.tracker.calls] == [False, True]


def test_passes_share_features_and_feed_back_low_res_logits(refine, fake):
    sam = fake()
    _run(refine, sam.model, positive_coords=TWO_POINTS, negative_coords=ONE_POINT,
         refine_iterations=3)
    first, second, third = sam.tracker.calls
    assert len(sam.backbone.calls) == 1
    assert sam.backbone.calls[0] == ((1, 3, 1008, 1008), {"tracker_mode": "interactive"})
    assert first["backbone_features"] is second["backbone_features"] is third["backbone_features"]
    # The low-resolution output, not an upsampled copy, is the next pass's prompt.
    assert tuple(second["mask_inputs"].shape) == (1, 1, LOW_RES, LOW_RES)
    assert torch.equal(second["mask_inputs"], _left_half())
    assert first["point_inputs"] is second["point_inputs"]


def test_points_are_scaled_to_the_model_resolution(refine, fake):
    sam = fake()
    _run(refine, sam.model, positive_coords=ONE_POINT, negative_coords='[{"x": 0, "y": 3}]')
    points = sam.tracker.calls[0]["point_inputs"]
    assert points["point_coords"].tolist() == [[[252.0, 252.0], [0.0, 756.0]]]
    assert points["point_labels"].tolist() == [[1, 0]]
    assert points["point_labels"].dtype == torch.int32


def test_a_prior_mask_becomes_signed_logits(refine, fake):
    sam = fake()
    prior = torch.zeros(1, 4, 8)
    prior[:, :, :3] = 1.0
    _run(refine, sam.model, negative_coords=ONE_POINT, mask=prior)
    assert torch.equal(sam.tracker.calls[0]["mask_inputs"], prior.unsqueeze(1) * 20 - 10)


def test_without_a_prior_the_first_pass_has_no_mask_prompt(refine, fake):
    sam = fake()
    _run(refine, sam.model, positive_coords=TWO_POINTS, refine_iterations=2)
    assert sam.tracker.calls[0]["mask_inputs"] is None
    assert sam.tracker.calls[1]["mask_inputs"] is not None


def test_memory_embedding_matches_core_for_both_backbones(refine, fake):
    sam = fake()
    _run(refine, sam.model, positive_coords=TWO_POINTS)
    assert torch.equal(sam.tracker.calls[0]["backbone_features"], torch.full((1, 3, 1, 1), 3.0))

    # Core prefers the interactivity embedding whenever the tracker has one ...
    sam = fake(multiplex=False, scalp=1)
    _run(refine, sam.model, positive_coords=TWO_POINTS)
    assert sam.backbone.calls[0][1] == {"need_tracker": True}
    call = sam.tracker.calls[0]
    assert torch.equal(call["backbone_features"], torch.full((1, 3, 2, 2), 2.0))
    assert len(call["high_res_features"]) == 1

    # ... and falls back to the plain SAM3 one.
    sam = fake(multiplex=False)
    sam.tracker.interactivity_no_mem_embed = None
    _run(refine, sam.model, positive_coords=TWO_POINTS)
    assert torch.equal(sam.tracker.calls[0]["backbone_features"], torch.full((1, 3, 1, 1), 5.0))


def test_stability_score_matches_meta(refine):
    logits = torch.tensor([[[[1.0, 0.0], [-0.04, -1.0]]]])
    # area above +0.05 is 1; area above -0.05 is 3
    assert refine._stability_scores(logits).item() == pytest.approx(1 / 3)
    assert refine._stability_scores(torch.full((1, 1, 2, 2), -1024.0)).item() == 1.0


@pytest.mark.parametrize("kwargs, message", [
    ({}, "include point"),
    ({"negative_coords": ONE_POINT}, "include point"),
    ({"mask": torch.ones(1, 4, 8)}, "include or exclude"),
    ({"positive_coords": '[{"x": 8, "y": 0}]'}, "outside the 8x4 image"),
    ({"positive_coords": '[{"x": NaN, "y": 0}]'}, "outside"),
    ({"positive_coords": '{"x": 1, "y": 1}'}, "JSON list"),
    ({"positive_coords": "[1, 2]"}, "JSON list"),
    ({"positive_coords": "not json"}, "JSON list"),
    ({"positive_coords": "[" + ",".join(['{"x": 1, "y": 1}'] * 65) + "]"}, "limit is 64"),
    ({"positive_coords": ONE_POINT, "mask": torch.ones(1, 8, 4)}, "matching the image"),
    ({"positive_coords": ONE_POINT, "mask_selection": "best"}, "mask_selection"),
    ({"positive_coords": ONE_POINT, "if_core_changed": "ignore"}, "if_core_changed"),
])
def test_rejects_bad_requests_before_running(refine, fake, kwargs, message):
    sam = fake()
    with pytest.raises(ValueError, match=message):
        _run(refine, sam.model, **kwargs)
    assert sam.tracker.calls == [] and sam.backbone.calls == []


def test_rejects_batches_and_other_models(refine, fake):
    sam = fake()
    with pytest.raises(ValueError, match="one image"):
        refine.vloSAM3RefineMask.execute(sam.model, torch.zeros(2, 4, 8, 3), positive_coords=ONE_POINT)
    other = SimpleNamespace(model=SimpleNamespace(diffusion_model=object()))
    with pytest.raises(ValueError, match="SAM3 or SAM3.1 checkpoint"):
        _run(refine, other, positive_coords=ONE_POINT)


def _execute(refine, model, **kwargs):
    return refine.vloSAM3RefineMask.execute(model, IMAGE, **kwargs).result


def test_installed_core_matches_the_pinned_fingerprint(refine):
    """If this fails, ComfyUI moved: review the diff, re-run these tests, re-pin."""
    assert refine.core_source_fingerprint() == refine.TESTED_SOURCE_FINGERPRINT
    assert refine.core_mismatch() is None


def test_a_verified_core_reports_the_verified_pipeline(refine, fake):
    sam = fake()
    _, pipeline = _execute(refine, sam.model, positive_coords=TWO_POINTS)
    assert pipeline == "verified"
    assert sam.segment_calls == []


def test_fingerprint_catches_an_edit_inside_the_heads(refine, monkeypatch):
    tracker = refine.sam3_modules()["tracker"]
    monkeypatch.setattr(tracker, "forward_sam_heads", lambda *args, **kwargs: None)
    detail = refine.core_mismatch()
    assert "has changed since this node was verified" in detail
    assert refine.TESTED_COMFYUI_COMMIT[:12] in detail


def test_a_missing_core_function_counts_as_a_change(refine, monkeypatch):
    tracker = refine.sam3_modules()["tracker"]
    monkeypatch.delattr(tracker.SAM31Tracker, "_forward_sam_heads")
    assert "cannot read the source of SAM31Tracker._forward_sam_heads" in refine.core_mismatch()


def test_a_changed_core_falls_back_to_native_forward_segment(refine, fake, core_changed, caplog):
    sam = fake()
    prior = torch.zeros(1, 4, 8)
    prior[:, :, :3] = 1.0
    with caplog.at_level("WARNING"):
        out, pipeline = _execute(refine, sam.model, positive_coords=ONE_POINT,
                                 negative_coords=ONE_POINT, mask=prior, refine_iterations=2)
    assert pipeline == "native"
    # Nothing from the copied setup runs: core's own call does all the work.
    assert sam.backbone.calls == [] and sam.tracker.calls == []
    first, second = sam.segment_calls
    assert tuple(first["images"].shape) == (1, 3, 1008, 1008)
    assert torch.equal(first["mask_inputs"], prior.unsqueeze(1) * 20 - 10)
    assert first["point_inputs"]["point_labels"].tolist() == [[1, 0]]
    assert torch.equal(second["mask_inputs"], _left_half())
    assert out[0, :, :4].eq(1).all() and out[0, :, 4:].eq(0).all()
    assert "Falling back to core's forward_segment" in caplog.text


def test_the_fallback_warns_once(refine, fake, core_changed, caplog):
    sam = fake()
    with caplog.at_level("WARNING"):
        _execute(refine, sam.model, positive_coords=ONE_POINT)
        _execute(refine, sam.model, positive_coords=ONE_POINT)
    assert caplog.text.count("Falling back") == 1


def test_the_fallback_says_it_ignores_forced_selection(refine, fake, core_changed, caplog):
    sam = fake()
    with caplog.at_level("WARNING"):
        _execute(refine, sam.model, positive_coords=ONE_POINT, mask_selection="single")
    assert "mask_selection 'single' is ignored" in caplog.text


def test_stop_raises_instead_of_falling_back(refine, fake, core_changed):
    sam = fake()
    with pytest.raises(RuntimeError) as excinfo:
        _execute(refine, sam.model, positive_coords=ONE_POINT, if_core_changed="stop")
    message = str(excinfo.value)
    assert "has changed since this node was verified" in message
    assert "if_core_changed" in message and refine.OVERRIDE_ENV in message
    assert sam.segment_calls == [] and sam.tracker.calls == []


def test_the_fallback_refuses_a_changed_forward_segment(refine, fake, core_changed):
    sam = fake()
    sam.sam.forward_segment = lambda frames, points=None: None
    with pytest.raises(RuntimeError, match="no longer takes images, point_inputs, mask_inputs"):
        _execute(refine, sam.model, positive_coords=ONE_POINT)
    del sam.sam.forward_segment
    with pytest.raises(RuntimeError, match="native fallback cannot run either"):
        _execute(refine, sam.model, positive_coords=ONE_POINT)


def test_the_override_runs_the_copied_setup_and_says_so(refine, fake, core_changed,
                                                       monkeypatch, caplog):
    monkeypatch.setenv(refine.OVERRIDE_ENV, "1")
    sam = fake()
    with caplog.at_level("WARNING"):
        _, pipeline = _execute(refine, sam.model, positive_coords=ONE_POINT, if_core_changed="stop")
    assert pipeline == "unverified"
    assert len(sam.tracker.calls) == 1 and sam.segment_calls == []
    assert "at your own risk" in caplog.text


@pytest.fixture(scope="module")
def checkpoint(refine):
    raw = os.environ.get("VLO_SAM3_CHECKPOINT")
    if not raw:
        pytest.skip("Set VLO_SAM3_CHECKPOINT to a SAM3 or SAM3.1 checkpoint to run real-weight tests")
    import comfy.sd

    model, *_ = comfy.sd.load_checkpoint_guess_config(str(Path(raw).expanduser()))
    return model


def _real_inputs():
    torch.manual_seed(0)
    image = torch.rand(1, 96, 128, 3)
    image[:, 24:72, 32:80] = torch.tensor([0.9, 0.1, 0.1])
    return image


@pytest.mark.parametrize("num_points", [1, 2])
def test_features_and_heads_reproduce_core_forward_segment(refine, checkpoint, num_points):
    """The copied feature setup must stay exactly what core's forward_segment does."""
    import comfy.model_management
    import comfy.utils

    image = _real_inputs()
    comfy.model_management.load_model_gpu(checkpoint)
    device = comfy.model_management.get_torch_device()
    dtype = checkpoint.model.get_dtype()
    sam = checkpoint.model.diffusion_model
    frame = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, "bilinear",
                                       crop="disabled").to(device=device, dtype=dtype)
    coords = [[56 / 128 * 1008, 48 / 96 * 1008], [100 / 128 * 1008, 10 / 96 * 1008]][:num_points]
    point_inputs = {
        "point_coords": torch.tensor([coords], device=device, dtype=dtype),
        "point_labels": torch.tensor([[1] * len(coords)], device=device, dtype=torch.int32),
    }
    with torch.inference_mode():
        expected = sam.forward_segment(frame, point_inputs=point_inputs)
        features, high_res = refine._interactive_features(sam, frame)
        _, actual, _, _ = sam.tracker._forward_sam_heads(
            backbone_features=features, point_inputs=point_inputs, mask_inputs=None,
            box_inputs=None, high_res_features=high_res, multimask_output=num_points == 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("core_moved", [False, True])
def test_real_weights_select_the_marked_square(refine, checkpoint, monkeypatch, core_moved):
    if core_moved:
        monkeypatch.delenv(refine.OVERRIDE_ENV, raising=False)
        monkeypatch.setattr(refine, "TESTED_SOURCE_FINGERPRINT", "0" * 64)
    image = _real_inputs()
    with torch.inference_mode():
        out, pipeline = refine.vloSAM3RefineMask.execute(
            checkpoint, image, positive_coords='[{"x": 56, "y": 48}]').result
    assert pipeline == ("native" if core_moved else "verified")
    inside = out[0, 24:72, 32:80].mean().item()
    outside = (out[0].sum().item() - out[0, 24:72, 32:80].sum().item()) / (96 * 128 - 48 * 48)
    assert inside > 0.9 and outside < 0.05
