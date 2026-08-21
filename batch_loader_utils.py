from __future__ import annotations

from typing import Any


MAX_MEMORY_BATCH_ITEMS = 100


def normalize_memory_batch_values(
    raw_values: Any,
    *,
    label: str,
    max_items: int = MAX_MEMORY_BATCH_ITEMS,
) -> list[str]:
    if isinstance(raw_values, str):
        values: list[Any] = [raw_values]
    elif isinstance(raw_values, (list, tuple)):
        values = list(raw_values)
    else:
        raise ValueError(f"{label} must be an ordered list of selections")

    normalized: list[str] = []
    for index, raw_value in enumerate(values):
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{label} item {index + 1} is not a valid selection")
        normalized.append(raw_value.strip())

    if not normalized:
        raise ValueError(f"Select at least one {label.lower()}")
    if len(normalized) > max_items:
        raise ValueError(f"{label} supports at most {max_items} items")
    return normalized


_TRUE_FLAG_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAG_TOKENS = frozenset({"0", "false", "no", "off", ""})


def normalize_memory_batch_flags(
    raw_flags: Any,
    *,
    count: int,
    label: str,
) -> list[bool]:
    """Normalize a per-item flag list to exactly one boolean per batch item.

    Accepts the comma-separated string the widget carries ("1,0,1"), or an
    already-split sequence. Items the caller never set are False, so a partly
    filled list stays meaningful; more flags than items is a mismatch worth
    reporting rather than silently trimming.
    """
    if raw_flags is None:
        tokens: list[Any] = []
    elif isinstance(raw_flags, str):
        tokens = [token for token in raw_flags.split(",")] if raw_flags.strip() else []
    elif isinstance(raw_flags, (list, tuple)):
        tokens = list(raw_flags)
    elif isinstance(raw_flags, bool):
        tokens = [raw_flags]
    else:
        raise ValueError(f"{label} must be a comma-separated flag list")

    flags: list[bool] = []
    for index, token in enumerate(tokens):
        if isinstance(token, bool):
            flags.append(token)
            continue
        if isinstance(token, (int, float)):
            flags.append(token != 0)
            continue
        if not isinstance(token, str):
            raise ValueError(f"{label} item {index + 1} is not a valid flag")
        normalized = token.strip().lower()
        if normalized in _TRUE_FLAG_TOKENS:
            flags.append(True)
        elif normalized in _FALSE_FLAG_TOKENS:
            flags.append(False)
        else:
            raise ValueError(f"{label} item {index + 1} is not a valid flag")

    if len(flags) > count:
        raise ValueError(
            f"{label} has {len(flags)} flags for {count} items"
        )
    return flags + [False] * (count - len(flags))


__all__ = [
    "MAX_MEMORY_BATCH_ITEMS",
    "normalize_memory_batch_flags",
    "normalize_memory_batch_values",
]
