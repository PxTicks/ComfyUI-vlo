from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Literal


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
    ) -> None:
        self._max_item_size_bytes = max_item_size_bytes
        self._max_total_size_bytes = max_total_size_bytes
        self._unread_ttl_seconds = unread_ttl_seconds
        self._accessed_ttl_seconds = accessed_ttl_seconds
        self._items: dict[str, MediaItem] = {}
        self._total_size_bytes = 0

    def _now(self) -> float:
        return time.time()

    def _remove(self, media_id: str) -> MediaItem | None:
        item = self._items.pop(media_id, None)
        if item is not None:
            self._total_size_bytes -= item.size_bytes
        return item

    def cleanup(self) -> None:
        now = self._now()
        expired_ids = [
            media_id
            for media_id, item in self._items.items()
            if now - item.last_accessed_at
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
            unread_items = sorted(
                (
                    item
                    for item in self._items.values()
                    if not item.accessed_once
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
