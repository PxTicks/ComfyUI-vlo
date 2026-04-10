async def comfy_entrypoint():
    from .extension import comfy_entrypoint as _comfy_entrypoint

    return await _comfy_entrypoint()


__all__ = ["comfy_entrypoint"]
