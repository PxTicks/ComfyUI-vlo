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
- `vlo Memory Load Video Batch` outputs an ordered `VIDEO` list plus a matching
  `BOOLEAN` "use audio" list, one flag per video.

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

#### Per-video audio flags

The video loader also carries a per-item switch. `include_audio` is a
comma-separated flag list in selection order (`1,0,1`); the selector renders it
as a checkbox on each row, and vlo writes it from the speaker toggles on its
batch slot. Unset items are false, and the flags travel with their video through
adds, removals, and reordering. The loader emits them as its second output, so a
consumer that takes a `BOOLEAN` list — such as the MiniMax H3 adapter's
`use_embedded_video_audio` — receives one flag per delivered video.

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
one-item list and a connected list arrives with one entry per video. The shipped
vlo workflow uses the second form: the video batch loader's "use audio" output
is linked to this input, so inclusion is decided per video rather than once for
the whole batch.

An `AUDIO` list connected to `ref_video_audios` overrides soundtracks
positionally and always wins, whether or not embedded audio is enabled for that
video. Videos with neither an override nor enabled embedded audio are passed as
video-only references.

The wrapper expands to a real native node in the execution graph rather than
calling its Python method directly. ComfyUI therefore applies the native node's
normal V3 lifecycle, validation, caching, and resource handling.

### MiniMax H3 masked guides (experimental)

`MiniMax H3 Add Masked Guide` gives an H3 image guide a continuous spatial
confidence mask: 1 keeps the guide at full strength, 0 corrupts that part of it
to noise. It works by giving each guide *token* its own condition noise level
and a matching condition timestep, generalizing the per-token modulation ComfyUI
already uses for masked target rows.

The mask does nothing until the model passes through `MiniMax H3 Patch Masked
Guides`, which installs a forked H3 forward pass. Samples without a masked guide
take the stock path untouched, and a fully open mask is bit-identical to a stock
`MiniMaxH3AddGuide`.

This is research code: it carries a copy of ComfyUI's `MiniMaxH3Model._forward`
and is tied to the ComfyUI version it was forked from. See
[nodes/minimax_masked_guide/README.md](nodes/minimax_masked_guide/README.md) for
the semantics, the compatibility rules and the experiment protocol.

## Installation

Clone (or symlink) this repository into your ComfyUI `custom_nodes` directory and
restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PxTicks/ComfyUI-vlo.git
```

The web extension under `web/` is registered automatically.
