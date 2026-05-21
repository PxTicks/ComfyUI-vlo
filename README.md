# ComfyUI-vlo

Utility nodes for [vlo](https://github.com/PxTicks/vlo). The workflow provides quality-of-life nodes for interaction with vlo, such as the memory loaders described below. It is not required for vlo to work with Comfy, but it IS used in the default workflows.

## Memory loaders

The **vlo Memory Load** family of nodes
(`vlo Memory Load Image`, `vlo Memory Load Audio`, `vlo Memory Load Video`).

Ordinarily, feeding media into ComfyUI means uploading it into the
`ComfyUI/input` folder. When an external app like vlo is generating many short-lived
inputs per run, that folder fills up quickly with throwaway files. The memory
loaders avoid this: media is held in an in-memory registry and referenced by id,
so nothing is written to `ComfyUI/input` just to be passed into a graph.

Each loader also keeps a `disable_in_memory` toggle, which falls back to loading
the selected file from the normal input directory — handy for testing workflows
by hand without going through vlo.

## Installation

Clone (or symlink) this repository into your ComfyUI `custom_nodes` directory and
restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PxTicks/ComfyUI-vlo.git
```

The web extension under `web/` is registered automatically.
