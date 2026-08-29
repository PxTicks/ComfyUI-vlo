from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Literal


# A media id is a uuid4 string. Matching on that shape keeps the walk below from
# dragging every prompt string (positive prompts included) into the set.
_MEDIA_ID_LENGTH = 36


def _looks_like_media_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _MEDIA_ID_LENGTH
        and all(char.isalnum() or char == "-" for char in value)
    )


def _collect_candidate_ids(value: Any, into: set[str]) -> None:
    """Gather id-shaped strings from a node input value.

    Recurses because a batch loader's value is a list of ids rather than one.
    """

    if _looks_like_media_id(value):
        into.add(value)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            _collect_candidate_ids(entry, into)
    elif isinstance(value, dict):
        for entry in value.values():
            _collect_candidate_ids(entry, into)


def collect_media_ids_from_queue(entries: Iterable[Any]) -> set[str]:
    """Media ids named by ComfyUI queue entries.

    ``entries`` are QueueTuples — ``(number, prompt_id, prompt, extra_data,
    outputs)`` — from the running and pending lists. Anything that does not look
    like one is skipped rather than raising: a retention sweep must never be the
    thing that breaks on an unexpected queue shape.
    """

    ids: set[str] = set()
    for entry in entries:
        try:
            prompt = entry[2]
        except (TypeError, IndexError, KeyError):
            continue
        if not isinstance(prompt, dict):
            continue
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                for value in inputs.values():
                    _collect_candidate_ids(value, ids)
    return ids


MediaKind = Literal["image", "video", "audio"]
_VALID_MEDIA_KINDS: tuple[MediaKind, ...] = ("image", "video", "audio")


class MediaRegistryError(Exception):
    """Base error for memory-backed media registry failures."""


class MediaTooLargeError(MediaRegistryError):
    """Raised when a single media item exceeds the per-item cap."""


class MediaRegistryCapacityError(MediaRegistryError):
    """Raised when the registry cannot make room for a new media item."""


@dataclass(slots=True)
class MediaItem:
    media_id: str
    kind: MediaKind
    filename: str
    content_type: str
    data: bytes
    created_at: float
    last_accessed_at: float
    accessed_once: bool = False
    client_id: str | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class MediaRegistry:
    def __init__(
        self,
        *,
        max_item_size_bytes: int = 512 * 1024 * 1024,
        max_total_size_bytes: int = 2 * 1024 * 1024 * 1024,
        unread_ttl_seconds: int = 2 * 60 * 60,
        accessed_ttl_seconds: int = 10 * 60,
        referenced_ids_provider: Callable[[], AbstractSet[str] | None] | None = None,
    ) -> None:
        self._max_item_size_bytes = max_item_size_bytes
        self._max_total_size_bytes = max_total_size_bytes
        self._unread_ttl_seconds = unread_ttl_seconds
        self._accessed_ttl_seconds = accessed_ttl_seconds
        # Media ids that queued or running work still needs. The TTLs measure
        # read recency, which is a fine proxy for "nobody wants this any more"
        # for a single prompt and a bad one for a batch: several prompts share
        # one item and read it minutes or hours apart, so the first read would
        # otherwise start a 10-minute countdown on media the rest still need.
        #
        # Returning None means "cannot tell", which is treated as "keep":
        # evicting media a queued prompt depends on fails that generation
        # outright, whereas keeping it costs memory until the next sweep.
        self._referenced_ids_provider = referenced_ids_provider
        self._items: dict[str, MediaItem] = {}
        self._total_size_bytes = 0

    def _now(self) -> float:
        return time.time()

    def _remove(self, media_id: str) -> MediaItem | None:
        item = self._items.pop(media_id, None)
        if item is not None:
            self._total_size_bytes -= item.size_bytes
        return item

    def _referenced_ids(self) -> AbstractSet[str] | None:
        """Ids in-flight work still needs, or None when that cannot be read."""

        if self._referenced_ids_provider is None:
            return frozenset()
        try:
            return self._referenced_ids_provider()
        except Exception:
            return None

    def cleanup(self) -> None:
        referenced = self._referenced_ids()
        if referenced is None:
            # Unknown: skip this pass rather than risk evicting media a queued
            # prompt is about to load. The next call retries.
            return

        now = self._now()
        expired_ids = [
            media_id
            for media_id, item in self._items.items()
            if media_id not in referenced
            and now - item.last_accessed_at
            >= (
                self._accessed_ttl_seconds
                if item.accessed_once
                else self._unread_ttl_seconds
            )
        ]
        for media_id in expired_ids:
            self._remove(media_id)

    def register(
        self,
        *,
        kind: str,
        filename: str,
        content_type: str,
        data: bytes,
        client_id: str | None = None,
    ) -> MediaItem:
        normalized_kind = kind.strip().lower()
        if normalized_kind not in _VALID_MEDIA_KINDS:
            raise ValueError(f"Unsupported media kind: {kind}")

        size_bytes = len(data)
        if size_bytes > self._max_item_size_bytes:
            raise MediaTooLargeError(
                f"Media item exceeds the {self._max_item_size_bytes} byte cap"
            )

        self.cleanup()

        if self._total_size_bytes + size_bytes > self._max_total_size_bytes:
            referenced = self._referenced_ids()
            if referenced is None:
                # Making room means proving an item is unwanted, and that proof
                # is exactly what is missing. Failing this registration reports
                # a clear error on work that has not started; evicting blind
                # would break a prompt already accepted into the queue, and it
                # would fail later and far less legibly, at execution time.
                raise MediaRegistryCapacityError(
                    "Registry is full and the prompt queue could not be "
                    "inspected, so no item can be shown to be free"
                )
            unread_items = sorted(
                (
                    item
                    for item in self._items.values()
                    if not item.accessed_once
                    and item.media_id not in referenced
                ),
                key=lambda item: (item.last_accessed_at, item.created_at),
            )
            for item in unread_items:
                if self._total_size_bytes + size_bytes <= self._max_total_size_bytes:
                    break
                self._remove(item.media_id)

        if self._total_size_bytes + size_bytes > self._max_total_size_bytes:
            raise MediaRegistryCapacityError(
                "Registry is full and could not evict enough unread items"
            )

        now = self._now()
        media_item = MediaItem(
            media_id=str(uuid.uuid4()),
            kind=normalized_kind,  # type: ignore[arg-type]
            filename=filename,
            content_type=content_type,
            data=data,
            created_at=now,
            last_accessed_at=now,
            accessed_once=False,
            client_id=client_id,
        )
        self._items[media_item.media_id] = media_item
        self._total_size_bytes += media_item.size_bytes
        return media_item

    def get(self, media_id: str, *, mark_accessed: bool = True) -> MediaItem | None:
        self.cleanup()
        item = self._items.get(media_id)
        if item is None:
            return None
        if mark_accessed:
            now = self._now()
            item.last_accessed_at = now
            item.accessed_once = True
        return item

    def delete(self, media_id: str) -> MediaItem | None:
        self.cleanup()
        return self._remove(media_id)

    def list_media(self, *, kind: str | None = None) -> list[MediaItem]:
        self.cleanup()
        normalized_kind = kind.strip().lower() if isinstance(kind, str) else None
        return sorted(
            [
                item
                for item in self._items.values()
                if normalized_kind is None or item.kind == normalized_kind
            ],
            key=lambda item: (item.created_at, item.media_id),
            reverse=True,
        )

    @property
    def total_size_bytes(self) -> int:
        self.cleanup()
        return self._total_size_bytes
