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

### MiniMax H3 LoRAs on pruned checkpoints

Pruned H3 checkpoints (`*_pruned_*`) replace the time embedder with an 8-dim
curve, `adaln_t_table`. A LoRA trained on the full model carries time-conditioning
updates for the 2688-wide adaln input, which the stock loader cannot apply there:
it logs `shape '[96768, 8]' is invalid` for every block and drops them. Turbo
LoRAs lean on those updates.

`vlo MiniMax H3 LoRA Loader` is the model-only LoRA loader with those updates
refitted to the base's curve. Each becomes a weight and bias delta fitted over
every point of the curve, to within about 0.01% on the turbo LoRAs tested, and
merges into the fp32 adaln weights. Nothing runs per step. On a full base, or
for a LoRA without adaln updates, it is the stock loader.

The fit needs the full model's time embedding, which pruned checkpoints don't
carry. The first time a pruned base is used, the node range-reads the four
`time_embedder` tensors (~63 MB per variant, not the whole checkpoint) from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) at a pinned
revision. It caches the result in `models/vlo/h3_time_grids/` (folder name
`vlo_h3_time_grids` in `extra_model_paths.yaml`). The node honours
`HF_ENDPOINT`, `HF_TOKEN` and `HF_HUB_OFFLINE`, and cache files placed by hand
are used without fetching. `base: auto` matches the checkpoint's curve to the
stock fl2va or ref2va one, accepting close finetunes, and refuses anything else.

Pre-converted pruned LoRAs whose adaln bias deltas are all zero misfit the time
conditioning and are flagged in the log. Load the full-model version of them
here instead.

## Audio latent masks

`vlo Set Audio Latent Binary Masks` turns a mask video into a temporal noise mask
on an audio latent, so an inpaint regenerates audio over exactly the masked
frames. The mask it produces is binary by design: it flips between preserved and
generated in a single audio latent step (25 ms for MiniMax H3, 40 ms for LTX).

That hard edge is audible. Generated audio meets the surrounding audio with no
shared phase and no shared level, which reads as a click or a skip at the seam.

`vlo Feather Audio Latent Mask` softens those time edges. A fractional mask value
is not a crossfade after the fact — comfy/ldm/minimax/model.py puts a masked
audio row at `sigma = mask * sigma_audio` and conditions it at that timestep, so
a decaying mask is a genuine per-step denoise strength and the model generates
the transition itself.

The default `outer` mode keeps the masked region fully solid and decays outward
into the preserved audio, so nothing you asked to regenerate loses authority.
`centered` straddles the original edge and `inner` keeps the whole ramp inside
the region. Ramp and hold lengths are given in seconds and converted using the
VAE's audio latent rate. The holds are worth reaching for when the audible event
outruns the frames that were masked — an onset leads visible mouth motion, and
reverb outlasts it.

**Place it after any blank latent composite.** `vlo Latent Composite Masked`
clears the destination wherever the mask is set. That is harmless while the mask
is binary, because a fully generated step never reads its latent image, but a
ramp step does read it, weighted by `1 - mask`. Feather before the composite and
the ramp is cleared along with the core; feather after it and outer ramps land
outside the cleared region, on the original audio, which is what they need.

**Inner and centered ramps need `original_audio_latent`.** Those modes put the
ramp *inside* the binary region, which is exactly the part the composite cleared,
so ordering alone cannot save them and they would blend toward silence. Connect
the pre-composite latent to `original_audio_latent` and the node restores the
original audio underneath the ramp. Outer ramps never need it.

## Video latent mask feathering

`vlo Feather Video Latent Mask` does the same for the video stream. A mask that
switches from preserved to generated in one latent frame makes inpainted footage
cut in: motion, lighting and texture jump at the seam. MiniMax H3 and LTX both run
a masked video token at `sigma = mask * sigma`, so a ramp of fractional values
has the model generate the transition rather than snapping to it.

Ramps and holds are counted in latent frames, about 4 video frames for MiniMax H3
and 8 for LTX. The mask must already be at latent resolution, as
`vlo Mask to Latent Mask` produces it.

Each pixel is feathered on its own along time:

- **outer** ramp frames sit outside the mask and borrow the nearest masked
  frame's value, so the fade keeps the shape that is fading. One region can stop
  while another carries on, and only the one that stops gets a ramp.
- **inner** ramp frames sit inside the mask and are weighted by their distance
  from its nearest edge. On a binary timeline this matches the audio node's
  inner ramp. Where the audio node refuses a region shorter than its ramp, this
  one lets the region peak lower. That matters per pixel, because a moving mask's
  edge pixels are covered for only a frame or two.
- **centered** splits the ramp across the edge, as the audio node does.

The ordering rules match the audio node. Run it after any blank latent composite.
Inner and centered ramps need `original_latent` connected, which restores the
original video wherever the mask ends up partial.

In vlo, this complements the editor's own temporal mask feather. For a stitched
generation (context | gap | context), this node softens the seam inside the
generated clip. The editor's feather then crossfades the generated context
frames into the footage underneath. With both set to `outer`, keep the editor's
feather at least as long as this ramp in video frames. Otherwise the editor cuts
back to the footage while the generated frames are still partly regenerated.
This node's default of 2 latent frames is about 8 frames on H3, twice the
editor's 4-frame default for generation masks.

## Mask growing

`vlo Grow Mask` grows a mask sequence by `expand` pixels with a square kernel.
It matches stock `Grow Mask` with `tapered_corners` off, but runs as two 1-D max
pools on the torch device, a chunk of frames at a time. The stock node dilates
3x3 once per pixel per frame on the CPU, which takes minutes on a long 720p clip.

The kernel is square because latent cells are square. To regenerate every
latent cell that a masked pixel touches, the mask has to reach a fixed distance
along both axes, and a round kernel would need about 1.4 times the radius to
cover the diagonal. One latent cell is a good default: 8 px for Wan and 16 px for
MiniMax H3, applied at the resolution the video is encoded at.

## Resized video saving

`vlo Save Video` encodes an `IMAGE` batch straight to MP4, MKV or
WebM, resizing each frame as it is encoded. It takes the same inputs as
ComfyUI's Create Video + Save Video (fps, audio, bit depth, color space,
container, codec, crf), plus `width`/`height`, `upscale_method` (default
`bicubic`) and `crop`. These follow the Upscale Image node: 0 keeps that side
proportional, and 0x0 keeps the source size.

The encode loop matches native `VideoFromComponents.save_to`. FFmpeg's swscale
resizes each frame while it converts RGB to YUV, so no resized batch is ever
held in memory. Resized frames retain 16-bit RGB precision until that conversion;
with no resize, the output is identical to native Save Video.

`save_output` works like VideoHelperSuite's Video Combine. On, the file goes to
the output folder. Off, it goes to ComfyUI's temp folder and still previews in
the node. Output dimensions must be even; a side derived from the aspect ratio
is rounded to even for you.

## Installation

Install from the [Comfy Registry](https://registry.comfy.org/nodes/comfyui-vlo)
with ComfyUI-Manager (search for "vlo"), or with comfy-cli:

```bash
comfy node install comfyui-vlo
```

To install manually, clone (or symlink) this repository into your ComfyUI `custom_nodes` directory and
restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PxTicks/ComfyUI-vlo.git
```

The web extension under `web/` is registered automatically.
