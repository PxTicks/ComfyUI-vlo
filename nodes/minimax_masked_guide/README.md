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

`sync_timesteps` on the patch node turns step 2 off, leaving step 1 in place.
That is the A/B the whole feature has to win: **does synchronizing guide-token
corruption with guide-token timestep materially beat simply corrupting the
latent?**

## Nodes

| Node | Role |
| --- | --- |
| `MiniMax H3 Add Masked Guide` | `MiniMaxH3AddGuide` plus a strength mask. Stores the pooled token strengths on the keyframe; core still owns timing, layout and VAE encoding. |
| `MiniMax H3 Patch Masked Guides` | Clones the model and installs the `DIFFUSION_MODEL` wrapper that reads those strengths. Nothing happens without it. |
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

It does not mean "no token". The token is still there and still attends; it just
contains approximately noise, and its modulation says so. Measuring whether
zero-strength tokens have observable residual effects is one of the experiments,
not an assumption of the design. If they do leak, the follow-up is a hybrid mode
that omits `s = 0` tokens entirely while keeping continuous weighting for
`0 < s ≤ 1` — that needs `PackedLayout` and positional-row changes, so it is
deliberately deferred.

## Structure

    masks.py              mask geometry, token pooling, strength -> coefficient map
    compatibility.py      structural probes against the installed ComfyUI
    masked_h3_forward.py  the fork of MiniMaxH3Model._forward
    nodes.py              the four nodes

`masked_h3_forward.py` is a copy of core's `_forward` with three marked changes.
Copying it is the point: the values that need changing (`seg_t`, `unique_t`,
`t_row`, `mod_segments`, `cond_video_rows`) are locals inside core's method and
there is no seam to hook. Keeping the copy in one file means it can be deleted
wholesale if the feature ever lands upstream.

**The fork is version-sensitive, and enforces it.** `compatibility.py` hashes the
source of every core function the fork copies or depends on (`_forward`,
`_cond_video_rows`, `PackedLayout.__init__`, `patchify_video`, the three `_mod_*`
helpers) and refuses to run if it does not match the pinned fingerprint. Symbol
probes alone cannot catch an upstream edit *inside* `_forward` — the fork would
keep running its stale copy and quietly diverge — so the source itself is the
version check.

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
**bit-identical** to a stock `MiniMaxH3AddGuide`. Two details make that hold, and
both are load-bearing:

- the strength → coefficient map pins its endpoints instead of interpolating to
  them, so `s = 1` lands exactly on `visual_cond_noise_aug`;
- the whole strength → timestep chain stays in float64, because float32 turns
  `0.999` into `0.9990000128746033` and silently splits the modulation table.

A condition segment whose rows all share one timestep then collapses back onto
core's scalar path. `test_masked_guide_forward_equivalence.py` asserts this with
`torch.equal`; if it ever fails, no later observation about masked guides is
interpretable.

## Scope

Supported: still-image guides, one mask per guide, arbitrary `frame_idx`,
multiple masked guides mixed freely with unmasked guides and references, batch
size 1.

Not yet: guide clips, guide audio, time-varying masks, masks on reference images,
sparse token omission.

The single-image contract is enforced, not assumed: `Add Masked Guide` refuses
any `IMAGE` batch other than one frame, and any `MASK` batch other than one mask,
rather than quietly using the first of each. (Core's `AddGuide` reads a ≥ 5 frame
batch as a guide clip and a shorter one as its first frame; neither is something
one mask can weight.) The `Pixel Fill` baseline does take batches — it pairs each
image with its own mask, or broadcasts a single mask across all of them.

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
6. **`sync_timesteps` on vs off** at the same mask — the central experiment.
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
