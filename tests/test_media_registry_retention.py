"""Retention rules for the in-memory media registry.

The TTLs measure read recency. That is a fine proxy for "nobody wants this any
more" when one prompt owns an item, and a bad one for a submitted-ahead batch:
its copies share a single item and read it minutes or hours apart. These tests
pin the rule that actually matters — in-flight work keeps its media alive.
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_registry import (  # noqa: E402
    MediaRegistry,
    MediaRegistryCapacityError,
    collect_media_ids_from_queue,
)


def make_registry(**kwargs) -> MediaRegistry:
    return MediaRegistry(
        unread_ttl_seconds=100,
        accessed_ttl_seconds=10,
        **kwargs,
    )


def register(registry: MediaRegistry, data: bytes = b"payload") -> str:
    return registry.register(
        kind="video",
        filename="prepared.mp4",
        content_type="video/mp4",
        data=data,
    ).media_id


def advance(registry: MediaRegistry, seconds: float) -> None:
    base = registry._now()
    registry._now = lambda: base + seconds  # type: ignore[method-assign]


def test_a_queued_prompt_keeps_its_media_past_the_accessed_ttl():
    """The batch case: copy 1 reads the media, copy 2 runs an hour later."""

    referenced: set[str] = set()
    registry = make_registry(referenced_ids_provider=lambda: referenced)
    media_id = register(registry)
    referenced.add(media_id)

    # Copy 1 loads it, starting the accessed countdown.
    assert registry.get(media_id) is not None

    advance(registry, 10_000)

    # Copy 2 is still queued, so the item survives its own cleanup pass.
    assert registry.get(media_id) is not None


def test_media_nothing_references_still_expires():
    """Retention must not become "keep forever" — that is a leak."""

    registry = make_registry(referenced_ids_provider=lambda: set())
    media_id = register(registry)
    assert registry.get(media_id) is not None

    advance(registry, 11)
    assert registry.get(media_id) is None


def test_media_expires_once_the_queue_lets_go_of_it():
    referenced: set[str] = set()
    registry = make_registry(referenced_ids_provider=lambda: referenced)
    media_id = register(registry)
    referenced.add(media_id)
    registry.get(media_id)

    advance(registry, 10_000)
    # Probed without marking, so the surviving item's own countdown is not
    # refreshed and the assertion below is about the reference set alone.
    assert registry.get(media_id, mark_accessed=False) is not None

    # The batch finished; nothing names the id any more.
    referenced.discard(media_id)
    assert registry.get(media_id) is None


def test_an_unreadable_queue_keeps_media_rather_than_guessing():
    """Evicting media a queued prompt needs fails that generation outright;
    keeping it merely costs memory until the next sweep."""

    def unreadable() -> set[str]:
        raise RuntimeError("prompt queue unavailable")

    registry = make_registry(referenced_ids_provider=unreadable)
    media_id = register(registry)
    registry.get(media_id)

    advance(registry, 10_000)
    assert registry.get(media_id) is not None


def test_capacity_eviction_never_takes_media_the_queue_needs():
    referenced: set[str] = set()
    registry = MediaRegistry(
        max_total_size_bytes=200,
        unread_ttl_seconds=100,
        accessed_ttl_seconds=10,
        referenced_ids_provider=lambda: referenced,
    )
    queued_id = register(registry, data=b"q" * 100)
    referenced.add(queued_id)

    # No room for this one without evicting the queued item, which is exactly
    # what must not happen.
    with pytest.raises(MediaRegistryCapacityError):
        register(registry, data=b"n" * 150)

    assert registry.get(queued_id) is not None


def test_capacity_eviction_refuses_to_guess_when_the_queue_is_unreadable():
    """Making room means proving an item is unwanted. With the queue
    unreadable that proof is missing, so the new registration must fail rather
    than evict something a queued prompt may depend on."""

    def unreadable() -> set[str]:
        raise RuntimeError("prompt queue unavailable")

    registry = MediaRegistry(
        max_total_size_bytes=200,
        unread_ttl_seconds=100,
        accessed_ttl_seconds=10,
        referenced_ids_provider=unreadable,
    )
    existing_id = register(registry, data=b"e" * 100)

    with pytest.raises(MediaRegistryCapacityError):
        register(registry, data=b"n" * 150)

    assert registry.get(existing_id, mark_accessed=False) is not None


def test_capacity_eviction_still_reclaims_unreferenced_media():
    referenced: set[str] = set()
    registry = MediaRegistry(
        max_total_size_bytes=200,
        unread_ttl_seconds=100,
        accessed_ttl_seconds=10,
        referenced_ids_provider=lambda: referenced,
    )
    abandoned_id = register(registry, data=b"a" * 100)

    replacement_id = register(registry, data=b"r" * 150)

    assert registry.get(abandoned_id) is None
    assert registry.get(replacement_id) is not None


def test_a_registry_without_a_provider_keeps_the_plain_ttl_behaviour():
    registry = make_registry()
    media_id = register(registry)
    registry.get(media_id)

    advance(registry, 11)
    assert registry.get(media_id) is None


MEDIA_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
OTHER_MEDIA_ID = "9c858901-8a57-4791-81fe-4c455b099bc9"


def queue_entry(prompt: dict) -> tuple:
    # QueueTuple: (number, prompt_id, prompt, extra_data, outputs).
    return (1, "prompt-1", prompt, {}, [])


def test_the_walker_finds_a_plain_loader_value():
    entries = [
        queue_entry(
            {
                "20": {
                    "class_type": "VLOMemoryLoadVideo",
                    "inputs": {"video": MEDIA_ID},
                }
            }
        )
    ]
    assert collect_media_ids_from_queue(entries) == {MEDIA_ID}


def test_the_walker_finds_every_id_in_a_batch_loader_list():
    """A batch loader's value is a list, so a scalar-only walk would miss most
    of the media the prompt depends on."""

    entries = [
        queue_entry(
            {
                "20": {
                    "class_type": "VLOMemoryLoadImageBatch",
                    "inputs": {"images": [MEDIA_ID, OTHER_MEDIA_ID]},
                }
            }
        )
    ]
    assert collect_media_ids_from_queue(entries) == {MEDIA_ID, OTHER_MEDIA_ID}


def test_the_walker_ignores_prompt_text_and_node_links():
    entries = [
        queue_entry(
            {
                "10": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "a very long positive prompt about a cat",
                        "clip": ["4", 1],
                    },
                },
                "20": {
                    "class_type": "VLOMemoryLoadVideo",
                    "inputs": {"video": MEDIA_ID},
                },
            }
        )
    ]
    assert collect_media_ids_from_queue(entries) == {MEDIA_ID}


def test_the_walker_survives_an_unexpected_queue_shape():
    """A retention sweep must never be the thing that breaks on a queue entry
    ComfyUI shaped differently than expected."""

    entries = [None, (), (1, "prompt-1"), (1, "prompt-1", "not-a-dict", {}, [])]
    assert collect_media_ids_from_queue(entries) == set()
