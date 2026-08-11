import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_loader_utils import normalize_memory_batch_values  # noqa: E402


def test_normalize_memory_batch_values_preserves_order() -> None:
    assert normalize_memory_batch_values(
        [" second ", "first", "third"], label="image"
    ) == [
        "second",
        "first",
        "third",
    ]


def test_normalize_memory_batch_values_accepts_one_legacy_scalar() -> None:
    assert normalize_memory_batch_values("one", label="audio clip") == ["one"]


def test_normalize_memory_batch_values_rejects_empty_and_oversized_lists() -> None:
    with pytest.raises(ValueError, match="Select at least one video"):
        normalize_memory_batch_values([], label="video", max_items=2)

    with pytest.raises(ValueError, match="video supports at most 2 items"):
        normalize_memory_batch_values(
            ["one", "two", "three"], label="video", max_items=2
        )


def test_normalize_memory_batch_values_rejects_invalid_item_types() -> None:
    with pytest.raises(ValueError, match="item 2 is not a valid selection"):
        normalize_memory_batch_values(["one", None], label="image")
