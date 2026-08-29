"""In-memory media registry singleton and the helpers that resolve media ids.

Anything that needs to turn a widget value into bytes goes through here: the
registry lookup, the ComfyUI input-folder fallback, and the fingerprints that
back IS_CHANGED.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import folder_paths
from aiohttp import web

from ..media_registry import (
    MediaItem,
    MediaRegistry,
    collect_media_ids_from_queue,
)


def _media_ids_in_flight() -> set[str] | None:
    """Media ids named by prompts ComfyUI is running or has queued.

    This is what makes retention correct for a submitted-ahead batch: its
    copies all reference one registry item and read it minutes apart, so read
    recency alone would expire media the queue still depends on.

    Returns None when the queue cannot be read, which the registry treats as
    "keep everything" rather than guessing.
    """

    try:
        from server import PromptServer  # Imported late: ComfyUI owns it.

        queue = PromptServer.instance.prompt_queue
        # The volatile variant shallow-copies under the mutex; the plain one
        # deep-copies every queued prompt, which is far too much work for
        # something on the path of every media read. It is also the newer of
        # the two, and this package runs against whatever ComfyUI the user has.
        read_queue = getattr(queue, "get_current_queue_volatile", None) or (
            queue.get_current_queue
        )
        running, pending = read_queue()
    except Exception:
        return None
    return collect_media_ids_from_queue((*running, *pending))


REGISTRY = MediaRegistry(referenced_ids_provider=_media_ids_in_flight)


_DEFAULT_CONTENT_TYPES = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/wav",
}
_UI_PLACEHOLDER_MEDIA_IDS = frozenset({"loading..."})


_INPUT_KIND_CONTENT_TYPES: dict[str, list[str]] = {
    "image": ["image"],
    "audio": ["audio", "video"],
    "video": ["video"],
}


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _normalize_kind(raw_kind: Any) -> str | None:
    if not isinstance(raw_kind, str):
        return None
    kind = raw_kind.strip().lower()
    if kind in _DEFAULT_CONTENT_TYPES:
        return kind
    return None


def _media_summary(item: MediaItem) -> dict[str, Any]:
    return {
        "media_id": item.media_id,
        "kind": item.kind,
        "filename": item.filename,
        "content_type": item.content_type,
        "size_bytes": item.size_bytes,
        "created_at": item.created_at,
        "last_accessed_at": item.last_accessed_at,
        "accessed_once": item.accessed_once,
    }


def _get_media_item(media_id: str, *, expected_kind: str | None = None) -> MediaItem:
    normalized_media_id = _normalize_media_id(media_id)
    if normalized_media_id is None:
        expected_label = expected_kind or "media"
        raise ValueError(f"No {expected_label} selected")

    item = REGISTRY.get(normalized_media_id)
    if item is None:
        raise ValueError(f"Unknown media id: {normalized_media_id}")
    if expected_kind is not None and item.kind != expected_kind:
        raise ValueError(
            f"Media id '{normalized_media_id}' has kind '{item.kind}', expected '{expected_kind}'"
        )
    return item


def _normalize_media_id(raw_media_id: Any) -> str | None:
    if not isinstance(raw_media_id, str):
        return None
    media_id = raw_media_id.strip()
    if not media_id:
        return None
    if media_id.lower() in _UI_PLACEHOLDER_MEDIA_IDS:
        return None
    return media_id


def _build_memory_output_item(
    item: MediaItem,
    *,
    subfolder: str = "",
) -> dict[str, Any]:
    return {
        "filename": item.filename,
        "subfolder": subfolder,
        "type": "output",
        "content_type": item.content_type,
        "view_url": f"/api/vlo-memory/view/{item.media_id}",
    }


def _list_input_files(content_types: list[str]) -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = [
        filename
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename))
    ]
    return sorted(folder_paths.filter_files_content_types(files, content_types))


def _annotated_filepath_exists(raw_value: Any) -> bool:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return False
    try:
        return folder_paths.exists_annotated_filepath(raw_value)
    except Exception:
        return False


def _should_load_from_filepath(raw_value: Any, *, disable_in_memory: bool) -> bool:
    if disable_in_memory:
        return True

    normalized_media_id = _normalize_media_id(raw_value)
    if normalized_media_id is not None and REGISTRY.get(normalized_media_id, mark_accessed=False) is not None:
        return False

    return _annotated_filepath_exists(raw_value)


def _fingerprint_annotated_filepath(raw_value: str, *, use_mtime: bool) -> str | float:
    media_path = folder_paths.get_annotated_filepath(raw_value)
    if use_mtime:
        return os.path.getmtime(media_path)

    digest = hashlib.sha256()
    with open(media_path, "rb") as media_file:
        digest.update(media_file.read())
    return digest.hexdigest()
