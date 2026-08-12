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

### Batch loaders

The corresponding **Batch** nodes load an ordered multi-selection:

- `vlo Memory Load Image Batch` outputs ordered `IMAGE` and `MASK` lists.
- `vlo Memory Load Audio Batch` outputs an ordered `AUDIO` list.
- `vlo Memory Load Video Batch` outputs an ordered `VIDEO` list.

These are ComfyUI list outputs rather than concatenated tensors. Images may
therefore have different dimensions, audio clips may have different durations,
and each video remains an independent video. Downstream nodes that need the
whole collection in one execution must opt into ComfyUI list inputs; ordinary
nodes will execute once per list item.

The batch nodes replace ComfyUI's static multi-select with an ordered selector
owned by this extension. In the default mode it refreshes directly from vlo's
in-memory registry. The `disable_in_memory` toggle reuses the same selector for
files already present in `ComfyUI/input`. Arrow controls determine the exact
list order sent downstream.

Selections are capped at 100 items as a general safety bound. Model-specific
nodes should enforce their own lower limits when consuming a collection.

The vlo application does not yet inject collection-valued media inputs. That
requires a separate application-side cardinality and upload contract; these
nodes currently provide the ComfyUI execution and authoring primitives for that
work.

### MiniMax H3 batch adapter

`vlo MiniMax H3 Reference to Video (Batch)` wraps ComfyUI's native MiniMax H3
reference-conditioning node so the three batch loaders can feed it directly.
It consumes each connected list in one execution and preserves its order. The
wrapper reads reference limits and socket prefixes from the installed native
node schema, and stops with a compatibility error if that contract changes.

The adapter converts each `VIDEO` to the native node's expected 24 fps image
frames. The `ref_audios` socket is for standalone audio references.

#### Reference video audio

MiniMax treats a reference video's own soundtrack as a separate `<Audio N>`
reference that has to be enabled: an ordinary reference video does not become an
audio reference merely because its file contains sound. Enabling one also
consumes an `<Audio N>` ordinal, which shifts the numbering of every later audio
tag, because `<Video N>` and `<Audio N>` are numbered independently and the
indices do not encode the pairing. The association is carried structurally, not
by the tag numbers.

`use_embedded_video_audio` therefore defaults to off. It accepts either form:

- a single value, which applies to every reference video;
- a `BOOLEAN` list with one entry per reference video, bound positionally.

Both work because the node uses Comfy list inputs, so a widget arrives as a
one-item list and a connected list arrives with one entry per video. Per-video
gating needs no schema change when it is driven from a real per-video source.

An `AUDIO` list connected to `ref_video_audios` overrides soundtracks
positionally and always wins, whether or not embedded audio is enabled for that
video. Videos with neither an override nor enabled embedded audio are passed as
video-only references.

The wrapper expands to a real native node in the execution graph rather than
calling its Python method directly. ComfyUI therefore applies the native node's
normal V3 lifecycle, validation, caching, and resource handling.

## Installation

Clone (or symlink) this repository into your ComfyUI `custom_nodes` directory and
restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PxTicks/ComfyUI-vlo.git
```

The web extension under `web/` is registered automatically.
