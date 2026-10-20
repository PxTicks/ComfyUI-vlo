"""MiniMax H3 time-embedding grids: silu(t_emb) on the pruned checkpoints' curve grid.

Pruned ("curve") H3 checkpoints drop the time embedder. Each adaln projection reads an
8-dim point on a curve, `adaln_t_table`, instead of the full model's 2688-wide
silu(t_emb). A LoRA trained on the full model still speaks the 2688-wide space, so
converting it for a pruned base needs silu(t_emb) itself, which no pruned checkpoint
carries. The full checkpoints do: their four time_embedder tensors (~63 MB) are
range-read from the official Comfy-Org repository at a pinned revision, the grid is
evaluated once with ComfyUI's own TimeEmbedder, and it is cached under
models/vlo/h3_time_grids beside that variant's pruned `adaln_t_table`, which is how a
loaded base is matched to its variant.

The cache files can also be placed by hand, in which case nothing is fetched.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import urllib.request

import torch
from safetensors.torch import save_file

import comfy.ops
import comfy.utils
import folder_paths
from comfy.ldm.minimax.model import TimeEmbedder

logger = logging.getLogger(__name__)


GRID_FOLDER = "vlo_h3_time_grids"
folder_paths.add_model_folder_path(
    GRID_FOLDER, os.path.join(folder_paths.models_dir, "vlo", "h3_time_grids")
)

# Pinned so a cached grid always describes the same weights, whatever happens upstream.
_REPO = "Comfy-Org/MiniMax-H3"
_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"

# variant -> (full checkpoint with the time embedder, pruned checkpoint with its curve)
VARIANTS = {
    "fl2va": (
        "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
        "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
    ),
    "ref2va": (
        "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
    ),
}

GRID_KEY = "silu_t_emb_grid"
TABLE_KEY = "adaln_t_table"
_TIME_EMBEDDER_KEYS = tuple(
    f"time_embedder.proj_{layer}.{param}" for layer in ("in", "out") for param in ("weight", "bias")
)

# The stock fl2va and ref2va curves are 1.8% apart and a finetune of fl2va (Eros) sits
# 0.16% from it, so a base further than this from every variant is not guessed at. Being
# under half the gap between variants, a base can be within it of only one of them.
_MATCH_TOLERANCE = 0.008

# A safetensors header is a few tens of KB; anything far larger is not the file we expect.
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_TIMEOUT_SECONDS = 120
_SAFETENSORS_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}

_lock = threading.Lock()
_loaded: dict[str, dict[str, torch.Tensor]] = {}


def _cache_filename(variant: str) -> str:
    return f"minimax_h3_{variant}_silu_t_emb_grid.safetensors"


def _source_url(path: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{_REPO}/resolve/{_REVISION}/{path}"


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _read_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end] of a remote file, refusing a server that ignores the range.

    A 200 would stream the whole checkpoint (tens of GB) instead of a slice of it.
    """
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "ComfyUI-vlo"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        if response.status != 206:
            raise RuntimeError(f"{url} did not honour a byte range (HTTP {response.status})")
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"{url} returned {len(data)} bytes for a {end - start + 1}-byte range")
    return data


def _read_remote_tensors(path: str, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Named tensors from a remote safetensors file, without downloading the rest of it."""
    url = _source_url(path)
    (header_bytes,) = struct.unpack("<Q", _read_range(url, 0, 7))
    if header_bytes > _MAX_HEADER_BYTES:
        raise RuntimeError(f"{url} has an implausible {header_bytes}-byte safetensors header")
    header = json.loads(_read_range(url, 8, 8 + header_bytes - 1))
    data_start = 8 + header_bytes
    tensors = {}
    for name in names:
        if name not in header:
            raise RuntimeError(f"{url} has no tensor {name!r}")
        entry = header[name]
        begin, end = entry["data_offsets"]
        raw = _read_range(url, data_start + begin, data_start + end - 1)
        dtype = _SAFETENSORS_DTYPES[entry["dtype"]]
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])
    return tensors


def evaluate_grid(time_embedder_sd: dict[str, torch.Tensor], rows: int) -> torch.Tensor:
    """silu(t_emb) at the `rows` evenly spaced t in [0, 1] that a curve table samples.

    Evaluated with ComfyUI's own TimeEmbedder, in fp32 as the model runs it, so the
    grid is what the full model would feed its adaln projections at each t.
    """
    proj_in = time_embedder_sd["time_embedder.proj_in.weight"]
    proj_out = time_embedder_sd["time_embedder.proj_out.weight"]
    embedder = TimeEmbedder(
        proj_in.shape[1],
        proj_in.shape[0],
        proj_out.shape[0],
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    embedder.load_state_dict(
        {key.removeprefix("time_embedder."): value.float() for key, value in time_embedder_sd.items()}
    )
    t = torch.linspace(0.0, 1.0, rows, dtype=torch.float32)
    with torch.no_grad():
        return torch.nn.functional.silu(embedder(t))


def _fetch(variant: str, destination: str) -> None:
    if _offline():
        raise RuntimeError(
            f"HF_HUB_OFFLINE is set and the MiniMax H3 {variant} time grid is not cached. "
            f"Place {_cache_filename(variant)} in {os.path.dirname(destination)}."
        )
    full_path, pruned_path = VARIANTS[variant]
    logger.info(
        "vlo: fetching the MiniMax H3 %s time embedder (~63 MB) from %s@%s",
        variant,
        _REPO,
        _REVISION[:8],
    )
    table = _read_remote_tensors(pruned_path, (TABLE_KEY,))[TABLE_KEY].float()
    grid = evaluate_grid(_read_remote_tensors(full_path, _TIME_EMBEDDER_KEYS), table.shape[0])
    metadata = {
        "repo": _REPO,
        "revision": _REVISION,
        "full_checkpoint": full_path,
        "pruned_checkpoint": pruned_path,
    }
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".partial"
    save_file({GRID_KEY: grid.contiguous(), TABLE_KEY: table.contiguous()}, partial, metadata=metadata)
    os.replace(partial, destination)


def _read_cache(variant: str) -> dict[str, torch.Tensor] | None:
    """The variant's tensors if loaded or on disk, without fetching. Hold `_lock`."""
    if variant in _loaded:
        return _loaded[variant]
    path = folder_paths.get_full_path(GRID_FOLDER, _cache_filename(variant))
    if path is None:
        return None
    tensors = comfy.utils.load_torch_file(path, safe_load=True)
    grid, table = tensors.get(GRID_KEY), tensors.get(TABLE_KEY)
    if grid is None or table is None or grid.ndim != 2 or table.ndim != 2 or len(grid) != len(table):
        raise RuntimeError(f"{path} is not a MiniMax H3 time grid; delete it to fetch a fresh one")
    _loaded[variant] = {GRID_KEY: grid.float(), TABLE_KEY: table.float()}
    return _loaded[variant]


def cached_variant(variant: str) -> dict[str, torch.Tensor] | None:
    """The variant's {GRID_KEY, TABLE_KEY} tensors if already cached, else None."""
    with _lock:
        return _read_cache(variant)


def load_variant(variant: str) -> dict[str, torch.Tensor]:
    """The variant's {GRID_KEY, TABLE_KEY} tensors, fetched and cached on first use."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown MiniMax H3 variant {variant!r}; expected one of {sorted(VARIANTS)}")
    with _lock:
        tensors = _read_cache(variant)
        if tensors is None:
            filename = _cache_filename(variant)
            _fetch(variant, os.path.join(folder_paths.get_folder_paths(GRID_FOLDER)[0], filename))
            tensors = _read_cache(variant)
        return tensors


def _curve_distance(table: torch.Tensor, reference: torch.Tensor) -> float:
    if table.shape != reference.shape:
        return float("inf")
    return float((table.double() - reference.double()).norm() / reference.double().norm())


def grid_for_base(table: torch.Tensor, variant: str = "auto") -> tuple[str, torch.Tensor]:
    """(variant, silu(t_emb) grid) for a pruned base, identified by its adaln_t_table.

    "auto" picks the variant whose stock curve the base is within tolerance of, and
    refuses a base near none of them rather than convert a LoRA against the wrong time
    embedding. The tolerance is under half the distance between the stock curves, so a
    match is never ambiguous: cached variants are tried first and the search stops at
    the first match, fetching (or needing) the others only when nothing cached fits.
    """
    table = table.detach().float().cpu()
    if variant != "auto":
        tensors = load_variant(variant)
        if tensors[GRID_KEY].shape[0] != table.shape[0]:
            raise ValueError(
                f"The {variant} time grid has {tensors[GRID_KEY].shape[0]} rows but this base's "
                f"adaln_t_table has {table.shape[0]}."
            )
        return variant, tensors[GRID_KEY]
    distances: dict[str, float] = {}
    for load in (cached_variant, load_variant):
        for name in VARIANTS:
            if name in distances:
                continue
            tensors = load(name)
            if tensors is None:
                continue
            distances[name] = _curve_distance(table, tensors[TABLE_KEY])
            if distances[name] <= _MATCH_TOLERANCE:
                return name, tensors[GRID_KEY]
    found = ", ".join(f"{name} {distance:.2%}" for name, distance in distances.items())
    raise ValueError(
        "This pruned MiniMax H3 base does not match a known time curve "
        f"(distance {found}). Set 'base' to the variant it was made from."
    )
