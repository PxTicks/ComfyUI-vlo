"""The memory-batch widget plus its validation and IS_CHANGED fingerprinting."""

from __future__ import annotations

import hashlib
from typing import Any

import folder_paths

from comfy_api.latest import io

from ..batch_loader_utils import normalize_memory_batch_values
from .registry import (
    REGISTRY,
    _fingerprint_annotated_filepath,
    _should_load_from_filepath,
)


def _validate_memory_batch_values(
    raw_values: Any,
    *,
    label: str,
    expected_kind: str,
    disable_in_memory: bool,
) -> bool | str:
    try:
        values = normalize_memory_batch_values(raw_values, label=label)
    except ValueError as exc:
        return str(exc)

    for value in values:
        if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
            if not folder_paths.exists_annotated_filepath(value):
                return f"Invalid {expected_kind} file: {value}"
            continue

        item = REGISTRY.get(value, mark_accessed=False)
        if item is None:
            return f"Invalid {expected_kind} id: {value}"
        if item.kind != expected_kind:
            return (
                f"Media id '{value}' has kind '{item.kind}', "
                f"expected '{expected_kind}'"
            )
    return True


def _fingerprint_memory_batch_values(
    raw_values: Any,
    *,
    label: str,
    expected_kind: str,
    disable_in_memory: bool,
    use_mtime: bool,
) -> tuple[bool, tuple[str | float, ...]]:
    try:
        values = normalize_memory_batch_values(raw_values, label=label)
    except ValueError:
        return disable_in_memory, ("__unset__",)

    fingerprints: list[str | float] = []
    for value in values:
        if _should_load_from_filepath(value, disable_in_memory=disable_in_memory):
            fingerprints.append(
                _fingerprint_annotated_filepath(value, use_mtime=use_mtime)
            )
            continue

        item = REGISTRY.get(value, mark_accessed=False)
        if item is None or item.kind != expected_kind:
            fingerprints.append(value)
        else:
            fingerprints.append(hashlib.sha256(item.data).hexdigest())
    return disable_in_memory, tuple(fingerprints)


def _memory_batch_input(
    input_id: str,
    *,
    display_name: str,
    placeholder: str,
) -> io.MultiCombo.Input:
    # ComfyUI does not support remote options on MultiCombo. The bundled web
    # extension replaces this inert stock widget with an ordered selector that
    # reads the live registry/input-folder routes. Keep an empty option set here
    # so object_info never advertises a stale or semantically wrong source.
    return io.MultiCombo.Input(
        input_id,
        options=[],
        display_name=display_name,
        default=[],
        placeholder=placeholder,
        chip=True,
    )
