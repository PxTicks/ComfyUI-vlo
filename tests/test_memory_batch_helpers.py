import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_loader_utils import (  # noqa: E402
    normalize_memory_batch_flags,
    normalize_memory_batch_values,
)


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


def test_normalize_memory_batch_flags_pads_unset_items() -> None:
    assert normalize_memory_batch_flags(
        "1,0", count=3, label="Video audio inclusion"
    ) == [True, False, False]
    assert normalize_memory_batch_flags(
        "", count=2, label="Video audio inclusion"
    ) == [False, False]
    assert normalize_memory_batch_flags(
        None, count=1, label="Video audio inclusion"
    ) == [False]


def test_normalize_memory_batch_flags_accepts_split_sequences() -> None:
    assert normalize_memory_batch_flags(
        [True, "0", 1], count=3, label="Video audio inclusion"
    ) == [True, False, True]


def test_normalize_memory_batch_flags_rejects_drift_and_bad_tokens() -> None:
    with pytest.raises(ValueError, match="has 3 flags for 2 items"):
        normalize_memory_batch_flags(
            "1,1,1", count=2, label="Video audio inclusion"
        )

    with pytest.raises(ValueError, match="item 2 is not a valid flag"):
        normalize_memory_batch_flags(
            "1,maybe", count=2, label="Video audio inclusion"
        )
