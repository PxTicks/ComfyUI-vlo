# MiniMax H3 masked guides (experimental)

A MiniMax-H3 image guide reaches the DiT as a *grid* of condition tokens, one per
2×2 patch of the guide's VAE latent, planted at the requested frame's time
coordinate. That spatial tokenization is what makes per-region guide strength
possible at all. These nodes give a guide a continuous spatial confidence map:

    M = 1  ->  trust this part of the guide
    M = 0  ->  make this part of the guide maximally unreliable

with everything in between blending continuously.

## How it works

Two things change per guide token, together:

1. **Latent corruption.** Core mixes every condition row with noise at one
   coefficient `a` (`visual_cond_noise_aug`, ~0.999 — essentially clean). Here
   each row gets its own `aᵢ = min_aug + sᵢ·(a_max − min_aug)`, so a token the
   mask closes down is replaced by noise.
2. **A matching timestep.** Core labels the whole guide with
   `max(t_v, visual_cond_noise_aug)`. Here each row is labelled
   `max(t_v, aᵢ)` — so the latent a token carries and the AdaLN modulation row it
   selects tell the model the same story about how noisy it is.

Step 2 is the point of the experiment, and it is only possible because ComfyUI PR
#15375 taught `_mod_row`/`_mod_scale_shift`/`_mod_gate` to accept a per-token
index tensor. That PR wired it up for target `video`/`audio` rows; this package
extends the same machinery to guide `cond` rows.

On a masked sample the wrapper substitutes the forked forward pass for the
innermost function and lets the rest of the wrapper chain run, so patches added
after this node (EasyCache, block swap, …) still apply and the result does not
depend on the order the model patch nodes were chained in.

`guide_clock` on the patch node selects how step 2 is done, or turns it off. That
is the A/B the whole feature has to win: **does synchronizing guide-token
corruption with guide-token timestep materially beat simply corrupting the
latent?**

| `guide_clock` | coefficient a mask value of 0 maps to | modulation label |
| --- | --- | --- |
| `stock` | `min_aug` | one global `max(t_v, 0.999)` for the whole guide |
| `floored` | `min_aug` | `max(t_v, a)` |
| `matched` *(default)* | `min_aug` | `a` |
| `target_relative` | `max(min_aug, t_v)` | `a` |

`stock` is step 1 alone — the baseline. `floored` is core's
`max(t_v, visual_cond_noise_aug)` with the coefficient substituted in; the guard
never fires for core, because `a` is pinned at `0.999`, but here it fires on
almost every step and a token holding pure noise ends up labelled as clean as the
target has become. `matched` drops that floor, which is what this package always
documented itself as doing.

`target_relative` is the interesting one. Core already has trained per-row
timestep semantics for exactly this axis — a partially denoised *video* row is
labelled `t = 1 - m*sigma`, where `m` is its denoise mask. Guide confidence runs
the same axis backwards (`s = 1 - m`), so this clock reuses core's own formula:
a fully trusted token sits at `0.999`, and a zero-confidence token sits exactly
level with the target. Note that this **changes what `s = 0` means** — the token
no longer carries *no* information, it carries no *marginal* information, which
is a different promise from the other three arms. If you want `s = 0` to mean the
guide is genuinely absent, no timestep formula delivers that; only omitting the
token does, and that is still deferred (see below).

## Nodes

| Node | Role |
| --- | --- |
| `MiniMax H3 Add Masked Guide` | `MiniMaxH3AddGuide` plus a strength mask. Stores the pooled token strengths on the keyframe; core still owns timing, layout and VAE encoding. |
| `MiniMax H3 Add Masked Guides from Video` | Cuts a masked video into guide clips and anchors each one. The helper the segmentation case actually wants. |
| `MiniMax H3 Patch Masked Guides` | Clones the model and installs the `DIFFUSION_MODEL` wrapper that reads those strengths. Nothing happens without it. Carries `guide_clock`. |
| `MiniMax H3 Guide Token Mask Preview` | Renders the strength grid the DiT actually sees, one block per token. Use it whenever mask alignment is in doubt. |
| `MiniMax H3 Masked Guide: Pixel Fill` | Baseline: mask the guide in pixel space, then feed a stock Add Guide. No patch involved. |

### Mask polarity

This mask is **guide confidence**, not a denoise mask. `1` means strong guide.
That is the opposite of ComfyUI's "1 means generate" convention, and it is
deliberate — the guide node is a trust map, and inheriting the denoise polarity
here would read backwards.

### Parameters

- `strength` scales the whole mask (`sᵢ = strength · mᵢ^gamma`).
- `min_aug` is the coefficient a mask value of 0 maps to. `0.0` replaces those
  tokens with pure noise; raise it to keep a floor of guidance everywhere.
- `mask_gamma` shapes the mid-tones: `>1` pushes them toward weak guidance.

Masks are pooled onto the token grid by **area averaging**, so a token half
covered by the mask is worth 0.5 and soft edges survive. Core max-pools its
*denoise* masks because a partially covered token must still be allowed to
generate; a guidance-strength map has no such conservative direction.

Strengths are quantized to 256 levels. Each distinct strength becomes a distinct
condition timestep, and every distinct timestep costs a row in the AdaLN
modulation table that is rebuilt per block.

## What `mask = 0` does *not* mean

It does not mean "no token", and it does not quite mean "no guidance" either.
It means: *replace this token's guide latent with deterministic pure noise and
present it as a visual condition token at the corresponding timestep*. The token
is still there, still attends, and still occupies the guide's position — it is
maximally unreliable rather than absent. An all-zero mask on a still guide is
therefore **not** equivalent to omitting the guide. Measuring whether
zero-strength tokens have observable residual effects is one of the experiments,
not an assumption of the design. If they do leak, the follow-up is a hybrid mode
that omits `s = 0` tokens entirely while keeping continuous weighting for
`0 < s ≤ 1` — that needs `PackedLayout` and positional-row changes, so it is
deliberately deferred.

## Structure

    masks.py              mask geometry, token pooling, strength -> coefficient map
    clips.py              video -> guide clips: frame selection, chunking, temporal pooling
    compatibility.py      structural probes against the installed ComfyUI
    masked_h3_forward.py  the fork of MiniMaxH3Model._forward
    nodes.py              the five nodes

`masked_h3_forward.py` is a copy of core's `_forward` with three marked changes.
Copying it is the point: the values that need changing (`seg_t`, `unique_t`,
`t_row`, `mod_segments`, `cond_video_rows`) are locals inside core's method and
there is no seam to hook. Keeping the copy in one file means it can be deleted
wholesale if the feature ever lands upstream.

**The fork is version-sensitive, and enforces it.** `compatibility.py` hashes the
source of every core function the fork copies or depends on (`_forward`,
`_cond_video_rows`, `PackedLayout.__init__`, `patchify_video`, the three `_mod_*`
helpers, and `FinalLayer.forward`) and refuses to run if it does not match the
pinned fingerprint. Symbol probes alone cannot catch an upstream edit *inside*
`_forward` — the fork would keep running its stale copy and quietly diverge — so
the source itself is the version check.

`FinalLayer.forward` is in that list because the fork hands it a per-token row
*tensor* whenever a denoise mask splits the target rows, which is behaviour PR
#15375 added. Probing `_mod_row` does not cover it: `FinalLayer` could stop
forwarding the selector while `_mod_row` still accepts one.

The behavioural probes match that standard. `_probe_wrapper_indexing` builds a
real three-wrapper chain and asserts that rebuilding from `wrappers[idx + 1:]`
runs exactly the wrappers registered after ours, in order — because the fork
depends on what those attributes *mean*, not merely that they exist. If `idx`
ever stopped being this wrapper's own position, the rebuilt chain would silently
drop or repeat wrappers rather than fail.

When ComfyUI moves:

```bash
COMFYUI_PATH=~/ComfyUI python tests/regen_fingerprint.py
```

Review the upstream diff, re-run `tests/test_masked_guide_forward_equivalence.py`,
port anything that changed, *then* re-pin `TESTED_SOURCE_FINGERPRINT`. Setting
`VLO_MASKED_GUIDE_ALLOW_UNVERIFIED=1` downgrades the refusal to a warning if you
have read the diff and want to try anyway.

## The invariant that matters

A fully open mask (`M = 1` everywhere, `strength = 1`, `mask_gamma = 1`) must be
**bit-identical** to a stock `MiniMaxH3AddGuide`, under every clock and at every
point on the schedule. Three details make that hold, and all are load-bearing:

- the strength → coefficient map pins its endpoints instead of interpolating to
  them, so `s = 1` lands exactly on `visual_cond_noise_aug`;
- the whole strength → timestep chain stays in float64, because float32 turns
  `0.999` into `0.9990000128746033` and silently splits the modulation table;
- the *timestep* endpoint is pinned too. A row sitting exactly at
  `visual_cond_noise_aug` is labelled `max(t_v, visual_cond_noise_aug)` — core's
  own condition timestep — rather than by the clock's rule. Without that, `matched`
  and `target_relative` label it `0.999` while core labels it `t_v`, and a
  fully-open mask stops being stock-identical for the last step or two, where
  `t_v` overtakes `0.999`.

That third pin is a real, deliberate exception to `matched`'s "label each token as
noisy as it actually is". The trade is not avoidable: once `t_v > 0.999` you
cannot both label a token at its actual corruption level and reproduce core's
label. The invariant wins, because every later observation about masked guides
depends on it; the cost is bounded, confined to the open end of the mask, and
smaller than `5e-4`. Every token the mask actually closes down keeps its honest
label. `target_relative` reaches the same place from the other side: its floor
caps at `visual_cond_noise_aug`, so in that tail the whole guide collapses to one
coefficient and takes core's scalar label with it.

A condition segment whose rows all share one timestep then collapses back onto
core's scalar path. `test_masked_guide_forward_equivalence.py` asserts this with
`torch.equal`; if it ever fails, no later observation about masked guides is
interpretable.

## Scope

Supported: still-image guides, guide clips with a time-varying mask, arbitrary
`frame_idx`, multiple masked guides mixed freely with unmasked guides and
references, batch size 1.

Not yet: guide audio, masks on reference images, sparse token omission.

`Add Masked Guide` stays single-image on purpose, and enforces it rather than
assuming it: it refuses any `IMAGE` batch other than one frame, and any `MASK`
batch other than one mask, rather than quietly using the first of each. (Core's
`AddGuide` reads a ≥ 5 frame batch as a guide clip and a shorter one as its first
frame; neither is something one mask can weight.) Clips are `Add Masked Guides
from Video`'s job, because a clip needs a mask per frame *and* a policy for the
frames that carry no mask at all. The `Pixel Fill` baseline takes batches too —
it pairs each image with its own mask, or broadcasts a single mask across all of
them.

## Masked guides from a video

The motivating case is a segmented subject: one source video, one SAM2 mask track,
and the wish to say "anchor this person, wherever they are, and leave the rest of
the frame free". That is not one guide. The subject leaves and re-enters, and H3
only accepts guide clips of **1, 5, 22, 39, ... (17k + 5)** frames, so the mask
track has to be cut into pieces that fit.

`Add Masked Guides from Video` does the cutting. V1's strategy is deliberately the
plain one:

1. **Drop the frames that guide nothing.** A frame whose mask is empty contributes
   a grid of tokens that are all corrupted to noise — pure cost, no guidance. The
   `min_coverage` threshold decides what counts as empty (`0.0`, the default,
   catches exactly the all-zero masks). Coverage is measured **after** the crop
   that fits the guide to the target's framing, not before: judged on the raw
   mask, a subject lying entirely in a band the cover-crop discards would pass the
   threshold and then pool to an all-zero grid — which, per the section above, is
   not an absent guide but a whole segment of pure-noise condition tokens. The
   node crops once, up front, and both decisions read that same result.
2. **One guide per surviving run.** Contiguous kept frames become one clip.
3. **Round the run down** to the nearest length on the ladder. Rounding has to
   throw frames away, and `chunk_align` says which end they come off.

Frames landing outside the target video are dropped *before* the runs are measured,
so a clip is rounded once and always fits. The node outputs a `plan` string saying
what it built, including the condition-token count — those rows ride through every
sampling step, so a generous mask over a long video is not free.

### Time-varying masks on the token grid

A clip's latent compresses pixel frames unevenly: `FRAME_PER_TOKEN` is
`(1, 4, 4, 4, 4)`, so latent frame 0 covers one pixel frame and each of the next
four covers four. The mask is pooled onto exactly those groups before the spatial
pooling runs, which is what keeps it from sliding against the frames it annotates.

`time_pooling` is the fork in the road that has no spatial equivalent:

- `average` treats a token whose four frames are half covered the way the spatial
  pooling treats a token half covered — one policy across both axes;
- `max` takes the union across the four frames, which is what a subject *moving*
  through them needs, since averaging smears a moving mask into a weak halo.

### Known crudeness

The strategy is intentionally naive, and it shows in two places worth knowing about
before reading results:

- **A one-frame mask dropout splits a run.** A 40 frame run with a single empty
  frame in the middle becomes two 5 frame guides instead of one 39 frame guide.
  Real SAM2 tracks flicker, so a gap tolerance is the first thing to add.
- **No repacking.** A 30 frame run becomes one 22 frame guide and 8 wasted frames,
  where 22 + 5 would have covered 27 of them.

Both live in `plan_video_guides`, which is a pure function over the keep flags, so
a better strategy is a change to one function and its tests.

## Running the experiments

Work synthetic before real. Suggested order:

1. **All-one mask** — must match stock Add Guide. (Covered by the unit tests, but
   worth confirming end to end on a real model.)
2. **All-zero mask** — compare against stock Add Guide, against no guide at all,
   and against the masked guide. Does `M = 0` behave approximately like no guide?
   This is the most important conceptual test.
3. **Binary spatial mask** — red square left, blue square right, mask only the
   red. Does the red region keep its keyframe influence while the blue region
   goes free?
4. **Gradient mask** — `M(x) = x / W`. Tests whether the weighting is genuinely
   continuous rather than an accidental threshold.
5. **Discrete strengths** — `s ∈ {0, 0.25, 0.5, 0.75, 1}`, measuring similarity at
   the guide frame. Expect broadly monotonic, not linear.
6. **`guide_clock` across all four arms** at the same mask — the central
   experiment. `stock` vs `matched` is the original question; `floored` vs
   `matched` isolates whether core's carried-over floor was doing harm; and
   `target_relative` asks whether a zero-confidence token is better off level
   with the target than at pure noise.
7. **Pixel-fill baseline** vs both of the above.
8. **Segmented human subject** — the motivating case. Look for identity
   preservation, pose anchoring, background contamination from the guide,
   boundary artifacts.

For the quantitative version, decode nothing: compare the output video latent
against the guide latent per token at the guide frame, and plot token strength
`mᵢ` against `‖z_out − z_guide‖`. Strength up should mean distance down.

Turn on `debug` for a one-line-per-guide report at the start of each sampling
run (token grid, token count, strength min/mean/max, unique coefficient levels,
and the cond-row count reconciliation). It logs once per run, not per step.
