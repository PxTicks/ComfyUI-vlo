"""vloMiniMaxH3LoraLoader: full-model adaln LoRA updates refitted to a pruned base's curve."""

from __future__ import annotations

import io
import logging
import math
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save as save_safetensors

from test_batch_nodes_integration import nodes_module  # noqa: F401


ROWS, CURVE, WIDTH = 33, 4, 12  # curve grid rows, curve dims, full silu(t_emb) width
OUT, RANK = 20, 3
ADALN = "blocks.0.adaln_proj.linear"
QKV = "blocks.0.attn.qkv_proj"


def _curve(rows=ROWS, dims=CURVE, seed=0):
    t = torch.linspace(0.0, 1.0, rows, dtype=torch.float64)[:, None]
    phases = torch.rand(1, dims, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.cos(math.pi * (t + phases) * torch.arange(1, dims + 1))


def _spanned_grid(table, seed=1):
    """A grid the curve plus a bias spans exactly, as the real H3 curves very nearly do."""
    basis = torch.cat([table, torch.ones(table.shape[0], 1, dtype=table.dtype)], dim=1)
    mix = torch.randn(basis.shape[1], WIDTH, generator=torch.Generator().manual_seed(seed), dtype=table.dtype)
    return basis @ mix


def _pair(seed=2, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(RANK, width, generator=generator), torch.randn(OUT, RANK, generator=generator)


def _model(in_features=CURVE):
    block = torch.nn.Module()
    block.adaln_proj = torch.nn.Module()
    block.adaln_proj.linear = torch.nn.Linear(in_features, OUT)
    block.attn = torch.nn.Module()
    block.attn.qkv_proj = torch.nn.Linear(6, 6, bias=False)
    diffusion_model = torch.nn.Module()
    diffusion_model.blocks = torch.nn.ModuleList([block])
    model = torch.nn.Module()
    model.diffusion_model = diffusion_model
    return model


KEY_MAP = {
    ADALN: f"diffusion_model.{ADALN}.weight",
    QKV: f"diffusion_model.{QKV}.weight",
}


def _never():
    raise AssertionError("the time grid was requested")


# --- the fit ----------------------------------------------------------------------------


def test_fit_reproduces_the_update_on_a_spanned_curve(nodes_module) -> None:
    h3_lora = nodes_module.h3_lora
    table = _curve()
    grid = _spanned_grid(table)
    down, up = _pair()

    diff, diff_bias = h3_lora.fit_curve_delta(h3_lora.curve_projection(table, grid), down, up, 0.5)

    fitted = table @ diff.double().T + diff_bias.double()
    target = 0.5 * grid @ down.double().T @ up.double().T
    assert diff.shape == (OUT, CURVE) and diff_bias.shape == (OUT,)
    torch.testing.assert_close(fitted, target, rtol=1e-5, atol=1e-5)


def test_fit_is_the_least_squares_fit_when_the_curve_does_not_span(nodes_module) -> None:
    h3_lora = nodes_module.h3_lora
    table = _curve()
    grid = torch.randn(ROWS, WIDTH, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    down, up = _pair()

    diff, diff_bias = h3_lora.fit_curve_delta(h3_lora.curve_projection(table, grid), down, up, 1.0)

    # The residual of a least-squares fit is orthogonal to every basis column.
    basis = torch.cat([table, torch.ones(ROWS, 1, dtype=table.dtype)], dim=1)
    residual = basis @ torch.cat([diff.double().T, diff_bias.double()[None]]) - grid @ down.double().T @ up.double().T
    torch.testing.assert_close(basis.T @ residual, torch.zeros(CURVE + 1, OUT, dtype=torch.float64), atol=1e-3, rtol=0)


# --- the LoRA rewrite, through ComfyUI's own patch path ----------------------------------


def test_converted_adaln_patches_apply_the_full_update_through_comfy(nodes_module) -> None:
    h3_lora, comfy = nodes_module.h3_lora, nodes_module.comfy
    table = _curve()
    grid = _spanned_grid(table)
    down, up = _pair()
    qkv_down, qkv_up = torch.randn(2, 6), torch.randn(6, 2)
    lora = {
        f"{ADALN}.lora_A.weight": down,
        f"{ADALN}.lora_B.weight": up,
        f"{ADALN}.alpha": torch.tensor(1.5),
        f"{QKV}.lora_A.weight": qkv_down,
        f"{QKV}.lora_B.weight": qkv_up,
    }
    model = _model()

    converted, prefixes = h3_lora.convert_adaln_for_curve_base(
        lora, KEY_MAP, model, lambda: h3_lora.curve_projection(table, grid)
    )

    assert prefixes == [ADALN]
    assert not any(key.startswith(ADALN + ".lora") or key == ADALN + ".alpha" for key in converted)
    assert converted[f"{QKV}.lora_A.weight"] is qkv_down and converted[f"{QKV}.lora_B.weight"] is qkv_up

    patches = comfy.lora.load_lora(converted, KEY_MAP)
    linear = model.diffusion_model.blocks[0].adaln_proj.linear
    weight_key, bias_key = KEY_MAP[ADALN], KEY_MAP[ADALN][: -len(".weight")] + ".bias"
    strength = 0.75
    weight = comfy.lora.calculate_weight(
        [(strength, patches[weight_key], 1.0, None, None)], linear.weight.detach().clone(), weight_key
    )
    bias = comfy.lora.calculate_weight(
        [(strength, patches[bias_key], 1.0, None, None)], linear.bias.detach().clone(), bias_key
    )

    def on_curve(w, b):
        return table @ w.double().T + b.double()

    added = on_curve(weight, bias) - on_curve(linear.weight.detach(), linear.bias.detach())
    expected = strength * (1.5 / RANK) * grid @ down.double().T @ up.double().T
    torch.testing.assert_close(added, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "down_suffix, up_suffix",
    [(".lora_down.weight", ".lora_up.weight"), (".lora.down.weight", ".lora.up.weight")],
)
def test_other_pair_namings_are_converted(nodes_module, down_suffix, up_suffix) -> None:
    h3_lora = nodes_module.h3_lora
    table = _curve()
    down, up = _pair()
    lora = {ADALN + down_suffix: down, ADALN + up_suffix: up}

    converted, prefixes = h3_lora.convert_adaln_for_curve_base(
        lora, KEY_MAP, _model(), lambda: h3_lora.curve_projection(table, _spanned_grid(table))
    )

    assert prefixes == [ADALN]
    assert set(converted) == {ADALN + ".diff", ADALN + ".diff_b"}


def test_adaln_pairs_that_already_fit_the_base_are_left_alone(nodes_module) -> None:
    h3_lora = nodes_module.h3_lora
    down, up = _pair(width=CURVE)
    lora = {f"{ADALN}.lora_A.weight": down, f"{ADALN}.lora_B.weight": up}

    converted, prefixes = h3_lora.convert_adaln_for_curve_base(lora, KEY_MAP, _model(), _never)

    assert prefixes == [] and converted == lora


def test_a_lora_without_adaln_updates_never_fetches_a_grid(nodes_module) -> None:
    h3_lora = nodes_module.h3_lora
    lora = {f"{QKV}.lora_A.weight": torch.randn(2, 6), f"{QKV}.lora_B.weight": torch.randn(6, 2)}

    converted, prefixes = h3_lora.convert_adaln_for_curve_base(lora, KEY_MAP, _model(), _never)

    assert prefixes == [] and converted == lora


def test_a_trained_bias_delta_is_kept_beside_the_fit(nodes_module) -> None:
    h3_lora = nodes_module.h3_lora
    table = _curve()
    projection = h3_lora.curve_projection(table, _spanned_grid(table))
    down, up = _pair()
    pair = {f"{ADALN}.lora_A.weight": down, f"{ADALN}.lora_B.weight": up}

    plain, _ = h3_lora.convert_adaln_for_curve_base(pair, KEY_MAP, _model(), lambda: projection)
    with_bias, _ = h3_lora.convert_adaln_for_curve_base(
        {**pair, f"{ADALN}.diff_b": torch.full((OUT,), 7.0)}, KEY_MAP, _model(), lambda: projection
    )

    torch.testing.assert_close(with_bias[f"{ADALN}.diff_b"] - plain[f"{ADALN}.diff_b"], torch.full((OUT,), 7.0))
    torch.testing.assert_close(with_bias[f"{ADALN}.diff"], plain[f"{ADALN}.diff"])


def test_bias_free_curve_patches_are_flagged(nodes_module, caplog) -> None:
    lora = {f"{ADALN}.diff": torch.randn(OUT, CURVE), f"{ADALN}.diff_b": torch.zeros(OUT)}

    with caplog.at_level(logging.WARNING):
        nodes_module.h3_lora._warn_on_bias_free_curve_patches(lora, KEY_MAP)

    assert "no bias delta" in caplog.text


def test_loader_converts_only_on_a_curve_base(nodes_module, monkeypatch) -> None:
    h3_lora = nodes_module.h3_lora
    table = _curve()
    grid = _spanned_grid(table)
    down, up = _pair()
    lora = {f"{ADALN}.lora_A.weight": down, f"{ADALN}.lora_B.weight": up}
    seen = {}

    def fake_load(model, clip, lora_sd, strength_model, strength_clip, lora_metadata=None):
        seen["lora"], seen["strength"] = lora_sd, strength_model
        return "patched", None

    def fake_grid_for_base(base_table, variant):
        seen["variant"] = variant
        torch.testing.assert_close(base_table.double(), table, rtol=0, atol=1e-6)
        return "fl2va", grid

    monkeypatch.setattr(h3_lora.comfy.sd, "load_lora_for_models", fake_load)
    monkeypatch.setattr(h3_lora.comfy.lora, "model_lora_keys_unet", lambda model, key_map: dict(KEY_MAP))
    monkeypatch.setattr(h3_lora, "grid_for_base", fake_grid_for_base)

    curve_model = _model()
    curve_model.diffusion_model.use_adaln_curves = True
    curve_model.diffusion_model.register_buffer("adaln_t_table", table.float())
    assert h3_lora.load_lora_for_h3(SimpleNamespace(model=curve_model), lora, 0.8, "ref2va") == "patched"
    assert seen["variant"] == "ref2va" and seen["strength"] == 0.8
    assert set(seen["lora"]) == {ADALN + ".diff", ADALN + ".diff_b"}

    full_model = _model(in_features=WIDTH)
    h3_lora.load_lora_for_h3(SimpleNamespace(model=full_model), lora, 1.0)
    assert seen["lora"] is lora


# --- the time grid ----------------------------------------------------------------------


def _time_embedder_sd(freq=8, hidden=6, out=5, seed=4):
    generator = torch.Generator().manual_seed(seed)
    return {
        "time_embedder.proj_in.weight": torch.randn(hidden, freq, generator=generator),
        "time_embedder.proj_in.bias": torch.randn(hidden, generator=generator),
        "time_embedder.proj_out.weight": torch.randn(out, hidden, generator=generator),
        "time_embedder.proj_out.bias": torch.randn(out, generator=generator),
    }


def test_grid_is_silu_of_the_time_embedding_at_evenly_spaced_t(nodes_module) -> None:
    sd = _time_embedder_sd()

    grid = nodes_module.h3_time_grid.evaluate_grid(sd, 5)

    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)[:, None]
    half = sd["time_embedder.proj_in.weight"].shape[1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    emb = torch.cat([torch.cos(t * freqs), torch.sin(t * freqs)], dim=-1)
    w = {key: value.double() for key, value in sd.items()}
    silu = torch.nn.functional.silu
    hidden = silu(emb @ w["time_embedder.proj_in.weight"].T + w["time_embedder.proj_in.bias"])
    expected = silu(hidden @ w["time_embedder.proj_out.weight"].T + w["time_embedder.proj_out.bias"])
    torch.testing.assert_close(grid.double(), expected, rtol=1e-5, atol=1e-5)


class _FakeHub:
    """Serves byte ranges of in-memory safetensors files the way the Hub resolve URL does."""

    def __init__(self, files, status=206):
        self.files, self.status, self.requests = files, status, []

    def urlopen(self, request, timeout=None):
        self.requests.append(request)
        path = re.sub(r"^.*/resolve/[0-9a-f]+/", "", request.full_url)
        data = self.files[path]
        start, end = map(int, re.match(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        body = data if self.status == 200 else data[start : end + 1]
        response = io.BytesIO(body)
        response.status = self.status
        return response


def _hub_files(grid_module, rows=9):
    files = {}
    for index, (full_path, pruned_path) in enumerate(grid_module.VARIANTS.values()):
        files[full_path] = save_safetensors(
            {**_time_embedder_sd(seed=10 + index), "blocks.0.big": torch.zeros(64)}
        )
        files[pruned_path] = save_safetensors({"adaln_t_table": _curve(rows, seed=20 + index).float()})
    return files


def test_remote_tensors_are_range_read_at_the_pinned_revision(nodes_module, monkeypatch) -> None:
    grid_module = nodes_module.h3_time_grid
    hub = _FakeHub(_hub_files(grid_module))
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", hub.urlopen)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    monkeypatch.setenv("HF_TOKEN", "secret")
    full_path = grid_module.VARIANTS["fl2va"][0]

    tensors = grid_module._read_remote_tensors(full_path, grid_module._TIME_EMBEDDER_KEYS)

    for key, value in _time_embedder_sd(seed=10).items():
        torch.testing.assert_close(tensors[key], value)
    assert all(
        request.full_url == f"https://mirror.example/Comfy-Org/MiniMax-H3/resolve/{grid_module._REVISION}/{full_path}"
        for request in hub.requests
    )
    assert all(request.get_header("Authorization") == "Bearer secret" for request in hub.requests)


def test_a_server_that_ignores_the_range_is_refused(nodes_module, monkeypatch) -> None:
    grid_module = nodes_module.h3_time_grid
    hub = _FakeHub(_hub_files(grid_module), status=200)
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", hub.urlopen)

    with pytest.raises(RuntimeError, match="byte range"):
        grid_module._read_remote_tensors(grid_module.VARIANTS["fl2va"][0], grid_module._TIME_EMBEDDER_KEYS)


@pytest.fixture
def grid_cache(nodes_module, monkeypatch, tmp_path):
    grid_module = nodes_module.h3_time_grid
    monkeypatch.setitem(grid_module.folder_paths.folder_names_and_paths, grid_module.GRID_FOLDER, ([str(tmp_path)], set()))
    monkeypatch.setattr(grid_module, "_loaded", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return tmp_path


def test_a_variant_is_fetched_once_then_read_from_the_cache(nodes_module, monkeypatch, grid_cache) -> None:
    grid_module = nodes_module.h3_time_grid
    hub = _FakeHub(_hub_files(grid_module))
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", hub.urlopen)

    fetched = grid_module.load_variant("ref2va")

    cached = grid_cache / grid_module._cache_filename("ref2va")
    assert cached.is_file() and not list(grid_cache.glob("*.partial"))
    torch.testing.assert_close(fetched[grid_module.TABLE_KEY], _curve(9, seed=21).float())
    torch.testing.assert_close(
        fetched[grid_module.GRID_KEY], grid_module.evaluate_grid(_time_embedder_sd(seed=11), 9)
    )

    monkeypatch.setattr(grid_module, "_loaded", {})
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", lambda *a, **k: _never())
    torch.testing.assert_close(grid_module.load_variant("ref2va")[grid_module.GRID_KEY], fetched[grid_module.GRID_KEY])


def test_offline_without_a_cache_names_the_file_to_place(nodes_module, monkeypatch, grid_cache) -> None:
    grid_module = nodes_module.h3_time_grid
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", lambda *a, **k: _never())

    with pytest.raises(RuntimeError, match=re.escape(grid_module._cache_filename("fl2va"))):
        grid_module.load_variant("fl2va")


def _preloaded(grid_module, monkeypatch, tables):
    loaded = {
        name: {grid_module.GRID_KEY: torch.full((ROWS, WIDTH), float(index)), grid_module.TABLE_KEY: table.float()}
        for index, (name, table) in enumerate(tables.items())
    }
    monkeypatch.setattr(grid_module, "_loaded", loaded)
    return loaded


def test_auto_picks_the_variant_with_the_nearest_curve(nodes_module, monkeypatch) -> None:
    grid_module = nodes_module.h3_time_grid
    loaded = _preloaded(grid_module, monkeypatch, {"fl2va": _curve(seed=0), "ref2va": _curve(seed=5)})
    finetune = _curve(seed=5) * 1.003

    variant, grid = grid_module.grid_for_base(finetune)

    assert variant == "ref2va" and grid is loaded["ref2va"][grid_module.GRID_KEY]


def test_auto_refuses_a_curve_near_no_variant(nodes_module, monkeypatch) -> None:
    grid_module = nodes_module.h3_time_grid
    _preloaded(grid_module, monkeypatch, {"fl2va": _curve(seed=0), "ref2va": _curve(seed=5)})

    with pytest.raises(ValueError, match="Set 'base'"):
        grid_module.grid_for_base(_curve(seed=9))


def test_an_explicit_variant_skips_matching_but_checks_the_grid_length(nodes_module, monkeypatch) -> None:
    grid_module = nodes_module.h3_time_grid
    loaded = _preloaded(grid_module, monkeypatch, {"fl2va": _curve(seed=0), "ref2va": _curve(seed=5)})

    variant, grid = grid_module.grid_for_base(_curve(seed=9), "fl2va")
    assert variant == "fl2va" and grid is loaded["fl2va"][grid_module.GRID_KEY]

    with pytest.raises(ValueError, match="rows"):
        grid_module.grid_for_base(_curve(rows=ROWS + 1, seed=9), "fl2va")


def _write_cache(grid_module, folder, variant, table):
    from safetensors.torch import save_file

    save_file(
        {grid_module.GRID_KEY: torch.full((table.shape[0], WIDTH), 1.0), grid_module.TABLE_KEY: table.float()},
        str(folder / grid_module._cache_filename(variant)),
    )


def test_auto_matches_a_cached_variant_offline_without_the_others(nodes_module, monkeypatch, grid_cache) -> None:
    grid_module = nodes_module.h3_time_grid
    _write_cache(grid_module, grid_cache, "fl2va", _curve(seed=0))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", lambda *a, **k: _never())

    variant, _ = grid_module.grid_for_base(_curve(seed=0))

    assert variant == "fl2va"


def test_auto_fetches_only_the_variants_it_needs(nodes_module, monkeypatch, grid_cache) -> None:
    grid_module = nodes_module.h3_time_grid
    files = _hub_files(grid_module)
    hub = _FakeHub(files)
    monkeypatch.setattr(grid_module.urllib.request, "urlopen", hub.urlopen)
    fl2va_table, ref2va_table = (_curve(9, seed=seed) for seed in (20, 21))

    assert grid_module.grid_for_base(fl2va_table)[0] == "fl2va"
    assert {re.sub(r"^.*/resolve/[0-9a-f]+/", "", r.full_url) for r in hub.requests} == set(grid_module.VARIANTS["fl2va"])

    hub.requests.clear()
    assert grid_module.grid_for_base(ref2va_table)[0] == "ref2va"
    assert {re.sub(r"^.*/resolve/[0-9a-f]+/", "", r.full_url) for r in hub.requests} == set(grid_module.VARIANTS["ref2va"])
