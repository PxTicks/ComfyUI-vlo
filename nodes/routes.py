"""The /api/vlo-memory HTTP routes.

These register as an import side effect via the PromptServer decorators, so this
module must be imported for the frontend extension to have anything to talk to.
"""

from __future__ import annotations

import logging

from aiohttp import web
from server import PromptServer

from ..media_registry import MediaRegistryCapacityError, MediaTooLargeError
from .registry import (
    _DEFAULT_CONTENT_TYPES,
    _INPUT_KIND_CONTENT_TYPES,
    REGISTRY,
    _json_error,
    _list_input_files,
    _media_summary,
    _normalize_kind,
)

logger = logging.getLogger(__name__)


@PromptServer.instance.routes.post("/api/vlo-memory/register")
async def register_memory_media(request: web.Request) -> web.Response:
    post_data = await request.post()
    file_field = post_data.get("media")
    if not isinstance(file_field, web.FileField):
        return _json_error(400, "Missing uploaded file field 'media'")

    kind = _normalize_kind(post_data.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    filename = post_data.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        filename = file_field.filename or f"upload.{kind}"

    content_type = post_data.get("content_type")
    if not isinstance(content_type, str) or not content_type.strip():
        content_type = file_field.content_type or _DEFAULT_CONTENT_TYPES[kind]

    client_id = post_data.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        client_id = None

    media_bytes = file_field.file.read()

    try:
        item = REGISTRY.register(
            kind=kind,
            filename=filename,
            content_type=content_type,
            data=media_bytes,
            client_id=client_id,
        )
    except MediaTooLargeError as exc:
        return _json_error(413, str(exc))
    except MediaRegistryCapacityError as exc:
        return _json_error(507, str(exc))
    except ValueError as exc:
        return _json_error(400, str(exc))

    return web.json_response(_media_summary(item))


@PromptServer.instance.routes.get("/api/vlo-memory/options")
async def list_memory_media_options(request: web.Request) -> web.Response:
    kind = _normalize_kind(request.query.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    options = [item.media_id for item in REGISTRY.list_media(kind=kind)]
    return web.json_response(options)


@PromptServer.instance.routes.get("/api/vlo-memory/input-files")
async def list_memory_input_files(request: web.Request) -> web.Response:
    kind = _normalize_kind(request.query.get("kind"))
    if kind is None:
        return _json_error(400, "Invalid or missing media kind")

    content_types = _INPUT_KIND_CONTENT_TYPES.get(kind)
    if content_types is None:
        return _json_error(400, "Unsupported media kind for input folder listing")

    return web.json_response(_list_input_files(content_types))


@PromptServer.instance.routes.get("/api/vlo-memory/view/{media_id}")
async def view_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.get(media_id)
    if item is None:
        logger.warning("vlo memory media not found: media_id=%s", media_id)
        return _json_error(404, "Unknown media id")
    logger.debug(
        "vlo memory media served: media_id=%s filename=%s content_type=%s size_bytes=%s",
        media_id,
        item.filename,
        item.content_type,
        item.size_bytes,
    )
    return web.Response(
        body=item.data,
        content_type=item.content_type,
        headers={"Content-Disposition": f'inline; filename="{item.filename}"'},
    )


@PromptServer.instance.routes.get("/api/vlo-memory/item/{media_id}")
async def get_memory_media_item(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.get(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
    return web.json_response(_media_summary(item))


@PromptServer.instance.routes.delete("/api/vlo-memory/item/{media_id}")
async def delete_memory_media(request: web.Request) -> web.Response:
    media_id = request.match_info.get("media_id", "")
    item = REGISTRY.delete(media_id)
    if item is None:
        return _json_error(404, "Unknown media id")
    return web.json_response(_media_summary(item))
