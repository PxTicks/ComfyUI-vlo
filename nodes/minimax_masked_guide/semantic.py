"""The semantic half of a guide: the same pixels, presented to Qwen at a time.

MiniMax H3 conditions on a guide through two channels, and stock ComfyUI already
uses both for first/last-frame guides -- `MiniMaxH3ImageToVideo` hands the guide
image to `clip.tokenize(..., images=...)` *and* VAE-encodes it as a keyframe.
What has no stock equivalent is doing that for a guide anchored at an arbitrary
frame, because the `<Picture N>` presentation carries no time.

H3's other presentation does. A `<Video k>` reference is emitted as

    "<Video k>: " ("<T.T seconds>" <2-frame vision block>)*

and core's tokenizer takes the timestamp list as a parameter, defaulting it to
2 fps only because that is how `MiniMaxH3ReferenceToVideo` samples. Handing it
target-timeline timestamps instead is the whole mechanism here: no tokenizer
fork, no new vocabulary, just the arbitrary-time presentation H3 already speaks.

Two properties are load-bearing and easy to lose:

  * The frames come from `GuideChunk.aligned_frames`, the very tensor the VAE
    encoded. Qwen never sees a pixel the latent guide does not contain -- in
    particular nothing the canvas cover-crop removed.
  * A semantic item produces no `minimax_refs` entry, so it adds no reference
    block, occupies no time-axis span in `PackedLayout`, and does not move the
    target cursor. It is Qwen information and nothing else.

Note that the guide's *absolute* rope coordinate does still shift, because the
vision block lengthens the text span and `PackedLayout` starts its target
timeline at `text_len`. Every row shifts by the same amount; what stays fixed is
the guide's offset from the target origin, which is the coordinate that means
anything.
"""

from __future__ import annotations

import torch

from .guides import GuideChunk

# Core's reference-video sampling rate. It is tuned for 2-15 second reference
# videos; H3 guide clips are 5/22/39 frames (0.2/0.9/1.6 s), so at 2 fps a guide
# clip yields one or two samples and the clip presentation collapses into the
# still one. That is why the rate is a parameter here rather than a constant.
DEFAULT_SAMPLE_FPS = 2.0

# How several guide chunks from one source video are presented.
#
#   separate  one "<Video k>" per chunk. Says "here are k different videos",
#             which is what core emits for k reference videos -- but core's are
#             each 0-based, and conveying a target-timeline position needs
#             non-zero starts either way, so neither arm reproduces training.
#   merged    one "<Video k>" whose samples jump: "<1.1 seconds>" ... "<5.7
#             seconds>". Nothing is emitted for the gap in either arm; the only
#             representation of the hole is the jump in the digits, because each
#             vision block is encoded independently with grid_t=1 and takes a
#             single M-RoPE time position from the running text counter. So a
#             gap costs nothing structurally, and one subject observed at
#             intervals is described as one subject.
PRESENTATIONS = ("merged", "separate")
DEFAULT_PRESENTATION = "merged"

SEMANTIC_MODES = ("off", "auto")

# Mirrors process_video_block's patch geometry: 16px patches merged 2x2.
_PATCH = 16
_MERGE = 2
_MAX_PIXELS = 12845056
_MIN_PIXELS = 3136

PENDING_ATTR = "_vlo_minimax_semantic_items"


def sample_offsets(length: int, fps: float, sample_fps: float) -> list[int]:
    """Frame offsets within a chunk, sampled at `sample_fps`.

    Rounded to whole source frames the way core does it (`range(0, n, FPS // 2)`),
    so the samples are real frames of the guide rather than interpolations of it.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive, got {}".format(sample_fps))
    step = max(1, int(round(float(fps) / float(sample_fps))))
    return list(range(0, int(length), step))


def chunk_presentation(chunk: GuideChunk, *, fps: float,
                       sample_fps: float = DEFAULT_SAMPLE_FPS) -> tuple[torch.Tensor, list[float]]:
    """A chunk -> (sampled frames, timestamps on the *generated* timeline).

    The timestamps describe where the guide sits in the video being generated, not
    where it sat in the source clip, so a chunk anchored away from frame zero says
    so. They are derived from `target_start`, the same resolved index the latent
    guide's `resolved_frame_index` uses, so the two channels cannot disagree about
    when this observation happens.

    The result is padded to an even frame count. The tokenizer pairs adjacent
    frames into one temporal patch and labels the pair with the *mean* of their
    timestamps, so an odd chunk left unpadded would let the next chunk's first
    frame fall into this chunk's last block -- fusing a cut inside one patch and
    labelling it with a time that belongs to neither. Core repeat-pads the same
    way; doing it per chunk is what makes the merged presentation safe.
    """
    offsets = sample_offsets(chunk.length, fps, sample_fps)
    frames = chunk.aligned_frames[offsets]
    timestamps = [(chunk.target_start + offset) / float(fps) for offset in offsets]
    if frames.shape[0] % 2 == 1:
        frames = torch.cat([frames, frames[-1:]], dim=0)
        timestamps = timestamps + [timestamps[-1]]
    return frames, timestamps


def semantic_items(chunks, *, fps: float, sample_fps: float = DEFAULT_SAMPLE_FPS,
                   presentation: str = DEFAULT_PRESENTATION) -> list[dict]:
    """Guide chunks -> `minimax_ref_items` entries for core's tokenizer.

    Only the tokenizer ever sees these; they deliberately have no `minimax_refs`
    counterpart. Chunks arrive in ascending target order (`plan_video_guides`
    scans runs left to right), so a merged item's timestamps are monotonic by
    construction.
    """
    if presentation not in PRESENTATIONS:
        raise ValueError(
            "unknown presentation {!r}, expected one of {}".format(presentation, PRESENTATIONS))
    chunks = list(chunks)
    if not chunks:
        return []

    parts = [chunk_presentation(c, fps=fps, sample_fps=sample_fps) for c in chunks]
    if presentation == "separate":
        return [{"type": "video", "data": frames, "timestamps": list(timestamps)}
                for frames, timestamps in parts]

    frames = torch.cat([f for f, _ in parts], dim=0)
    timestamps = [t for _, ts in parts for t in ts]
    return [{"type": "video", "data": frames, "timestamps": timestamps}]


def vision_block_tokens(height: int, width: int) -> int:
    """Merged Qwen tokens one 2-frame vision block costs, per `process_video_block`.

    Worth reporting rather than discovering: at a 1344x768 canvas a single block
    is ~1000 tokens against a prompt of a few dozen, it lengthens the text span
    that rides every sampling step, and every extra sample adds another one.
    """
    factor = _PATCH * _MERGE
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > _MAX_PIXELS:
        beta = (height * width / _MAX_PIXELS) ** 0.5
        h_bar = max(factor, int(height / beta / factor) * factor)
        w_bar = max(factor, int(width / beta / factor) * factor)
    elif h_bar * w_bar < _MIN_PIXELS:
        beta = (_MIN_PIXELS / (height * width)) ** 0.5
        h_bar = -(-int(height * beta) // factor) * factor
        w_bar = -(-int(width * beta) // factor) * factor
    return (h_bar // factor) * (w_bar // factor)


def describe_items(items, *, height: int, width: int) -> str:
    blocks = sum(item["data"].shape[0] // 2 for item in items)
    per_block = vision_block_tokens(height, width)
    lines = []
    for i, item in enumerate(items):
        stamps = item["timestamps"]
        pairs = ["{:.1f}".format((stamps[j] + stamps[j + 1]) / 2.0)
                 for j in range(0, len(stamps), 2)]
        lines.append("  <Video {}>: {} frame(s) at {} seconds".format(
            i + 1, item["data"].shape[0], ", ".join(pairs)))
    return "\n".join(
        ["{} semantic reference(s), {} vision block(s)".format(len(items), blocks)]
        + lines
        + ["Qwen tokens added to the text span: ~{} ({} per block)".format(
            blocks * per_block, per_block)])


def clip_with_semantic_items(clip, items):
    """A CLIP whose `tokenize` appends `items` to whatever the next node presents.

    Qwen runs inside the conditioning node, so semantic information has to be in
    place *before* that node executes -- there is no adding it to an encoded
    CONDITIONING afterwards. Wrapping the CLIP is what puts it there without
    replacing core's conditioning nodes.

    Three details this depends on:

      * `CLIP.clone()` copies a fixed set of fields and shares `self.tokenizer`
        with the original, so patching the tokenizer would patch every CLIP in
        the process. The override goes on the clone *instance* instead.
      * The core tokenizer treats `images=` and `minimax_ref_items=` as exclusive
        branches, but both emit "<Picture i>: " plus a plain vision block for an
        image, so folding a downstream node's `images` into ref items leaves its
        presentation byte for byte unchanged.
      * The items are appended *last*. Reference labels and packed reference rows
        correspond by position, so a semantic item -- which has no packed rows --
        inserted ahead of a native reference would desynchronise every reference
        after it. Appending also continues core's per-type "<Video k>" numbering
        rather than colliding with it.

    Chaining works because `clone()` does not carry the instance override: each
    application reads the accumulated items off its input and installs one fresh
    wrapper over core's own `tokenize`, so wrappers never nest.
    """
    if not items:
        return clip
    wrapped = clip.clone()
    pending = list(getattr(clip, PENDING_ATTR, ())) + list(items)
    setattr(wrapped, PENDING_ATTR, pending)
    base_tokenize = type(wrapped).tokenize

    def tokenize(text, return_word_ids=False, **kwargs):
        ref_items = list(kwargs.pop("minimax_ref_items", None) or ())
        images = list(kwargs.pop("images", None) or ())
        if ref_items and images:
            raise ValueError(
                "the conditioning node presented both images and reference items; the "
                "semantic-guide wrapper cannot tell what order MiniMax H3 should see them in")
        if images:
            ref_items = [{"type": "image", "data": image} for image in images]
        ref_items.extend(pending)
        return base_tokenize(wrapped, text, return_word_ids,
                             minimax_ref_items=ref_items, **kwargs)

    wrapped.tokenize = tokenize
    return wrapped
