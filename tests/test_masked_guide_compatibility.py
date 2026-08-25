"""The compatibility gate has to notice an upstream `_forward` edit, not just missing names."""

from __future__ import annotations

import pytest

from minimax_h3_harness import h3_model_module, masked_guide_module


@pytest.fixture(scope="module")
def compat():
    return masked_guide_module("compatibility")


def test_installed_core_matches_the_pinned_fingerprint(compat):
    """If this fails, ComfyUI moved: review the diff, re-run the equivalence tests, re-pin."""
    assert compat.core_source_fingerprint() == compat.TESTED_SOURCE_FINGERPRINT
    compat.check_core_compatible()


def test_fingerprint_covers_the_forked_forward(compat, monkeypatch):
    """A change inside _forward that keeps every symbol must still be caught."""
    module = h3_model_module()
    before = compat.core_source_fingerprint()

    def edited_forward(self, *args, **kwargs):
        return None  # same name, same signature, different body

    monkeypatch.setattr(module.MiniMaxH3Model, "_forward", edited_forward)
    assert compat.core_source_fingerprint() != before
    with pytest.raises(RuntimeError, match="has changed since this fork was taken"):
        compat.check_core_compatible()


def test_fingerprint_covers_the_modulation_helpers(compat, monkeypatch):
    module = h3_model_module()
    before = compat.core_source_fingerprint()
    monkeypatch.setattr(module, "_mod_row", lambda vecs, row, dtype: vecs[row].to(dtype))
    assert compat.core_source_fingerprint() != before


def test_the_override_is_a_deliberate_opt_in(compat, monkeypatch, caplog):
    module = h3_model_module()
    monkeypatch.setattr(module.MiniMaxH3Model, "_forward", lambda self, *a, **k: None)
    monkeypatch.setenv(compat.OVERRIDE_ENV, "1")
    with caplog.at_level("WARNING"):
        compat.check_core_compatible()
    assert "at your own risk" in caplog.text


def test_error_names_the_version_it_was_pinned_to(compat, monkeypatch):
    module = h3_model_module()
    monkeypatch.setattr(module.MiniMaxH3Model, "_forward", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError) as excinfo:
        compat.check_core_compatible()
    message = str(excinfo.value)
    assert compat.TESTED_COMFYUI_VERSION in message
    assert compat.TESTED_COMFYUI_COMMIT[:12] in message
    assert compat.OVERRIDE_ENV in message


def test_symbol_probes_still_run_first(compat, monkeypatch):
    module = h3_model_module()
    monkeypatch.delattr(module, "mask_row_values")
    with pytest.raises(RuntimeError, match="missing mask_row_values"):
        compat.check_core_compatible()


def test_wrapper_chain_probe_matches_the_executor_this_pack_rebuilds(compat):
    import comfy.patcher_extension as ext

    executor = ext.WrapperExecutor.new_class_executor(lambda: None, object(), [])
    assert hasattr(executor, "wrappers") and hasattr(executor, "idx")
    compat._probe_wrapper_chain()
