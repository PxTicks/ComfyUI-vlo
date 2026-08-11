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


__all__ = ["MAX_MEMORY_BATCH_ITEMS", "normalize_memory_batch_values"]
