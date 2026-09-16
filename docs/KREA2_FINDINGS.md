# Krea-2 RAW LoRA Training: Findings

This document is the research trail behind this repository's default
settings. It records what was tried, what worked, what didn't, and
why -- so future changes don't repeat solved problems.

## Summary

| Configuration | Steps | Rank | Face-mask weighting | Peak ArcFace similarity | Status |
|---|---|---|---|---|---|
| v1 | 1600 | 16/16 | No | 0.4242 (still rising) | Undertrained |
| v2 | 2800 | 32/32 | Yes (mask_min_value 0.10) | **0.5106** | Converged |

For reference, parallel Flux.2 Klein LoRA research on the same subject
and holdout set converged at:
- Klein 4B: 0.5498 (rank 64, face-mask weighted, 2000 steps of a 3000-step run)
- Klein 9B: 0.5296 (rank 64, face-mask weighted, 2200 steps of a 3000-step run)

Krea-2 RAW's 0.5106 is in the same convergence range, achieved at
rank 32 (not 64) -- a smaller adapter than Flux needed, but the
regional face-mask weighting appears to be the more important lever
than raw rank once a reasonable rank floor is met.

## What ArcFace similarity means here

All evaluation in this project uses the same protocol: a fixed set of
"true holdout" photos of the subject that are *never* included in the
training dataset, embedded with ArcFace (`w600k_r50`), averaged into a
single centroid identity vector. Every generated sample during
training is then compared against that centroid with cosine
similarity. In facial recognition literature, cosine similarity above
roughly 0.50 against an un-augmented, multi-pose holdout centroid
indicates a confident positive identity match -- this is the target
threshold used throughout this project's research, on both Flux and
Krea-2.

## v1: undertrained, not converged (rank 16, no masking, 1600 steps)

| Step | Mean ArcFace Sim |
|---|---|
| 400 | 0.0363 |
| 800 | 0.1272 |
| 1200 | 0.3328 |
| 1600 | 0.4242 |

The similarity curve was still climbing steeply and monotonically at
step 1600 with no sign of flattening -- this run was stopped too early
relative to the capacity of the adapter, not because training had
plateaued.

**Root causes identified:**

1. **Rank 16/16 was too low relative to model size.** Krea-2 RAW is a
   12.9B-parameter single-stream MMDiT -- roughly 3x the parameter
   count of Flux Klein 4B, and larger than Klein 9B. The Flux runs
   that converged used rank 64. Community guidance (multiple
   independent Reddit/HuggingFace sources) converges on **rank 32 as
   the practical minimum** for Krea-2 character LoRAs; several report
   rank 32-64 for strong face likeness at 1750-3000 steps.
2. **No face-mask regional loss weighting was used.** The single
   largest lever found in this project's Flux.2 research (the v5 to
   v6 breakthrough) was introducing `mask_path` + `mask_min_value`
   regional loss weighting, concentrating gradient on the face region
   instead of weighting the whole frame equally. That lever was
   simply absent from the v1 Krea-2 run.

## v2: converged (rank 32, face-mask weighting, 2800 steps)

| Step | Mean ArcFace Sim |
|---|---|
| 400 | 0.2879 |
| 800 | 0.3755 |
| 1200 | 0.4772 |
| 1600 | 0.4846 |
| 2000 | 0.5082 |
| 2400 | 0.4643 (noise dip, not attractor collapse) |
| **2800** | **0.5106 (peak)** |

Applying both fixes simultaneously (rank 32/32 + face-mask weighting)
closed the gap to Flux-level convergence. The curve visibly flattens
after step ~2000 rather than climbing steeply, indicating this run
reached (or is very near) the adapter's practical capacity ceiling for
this dataset size, unlike v1.

**These are the settings baked into `pipeline/generate_training_config.py`
as the repository's defaults.**

## What did NOT need to change

- **Dataset size/quality**: the same 20-image dataset was used for
  both v1 and v2; the bottleneck was never the data.
- **Trigger word strategy**: a short trigger token combined with a
  fuller descriptive caption sentence (not a bare trigger+class tag)
  is used, because Krea-2's Qwen3-VL-4B text encoder responds more
  strongly to descriptive natural language than to short trigger tags
  (multiple independent community reports converge on this; see
  "Captioning for Krea-2" below).
- **Quantization (`qfloat8` for both transformer and text encoder)**:
  this is the community-standard AI-Toolkit setting for Krea-2 and was
  never implicated in any convergence or quality problem.
- **Frozen text encoder + cached text embeddings**: required to fit a
  12.9B model + 4B text encoder in 24GB VRAM; unrelated to likeness
  quality.

## Captioning for Krea-2

Krea-2 uses Qwen3-VL-4B-Instruct as its text conditioning encoder
(not CLIP+T5 like Flux). Multiple independent community sources
(Reddit threads on Krea-2 LoRA captioning, HuggingFace model
discussions) report that this text encoder responds much more weakly
to short trigger-word-style captions than Flux/CLIP-style models do,
and responds much better to fuller, descriptive natural-language
sentences. This repository's caption generator
(`pipeline/ingest_dataset.py::build_caption`) therefore produces
captions like:

```
A photo of kmb, a woman, facing the camera directly in natural lighting.
```

rather than a bare `kmb woman, frontal, natural lighting` tag list.

## Inference-side finding: training convergence and inference realism are separate problems

A LoRA that has converged identity-wise (high ArcFace similarity in
training samples) can still produce images with a "plastic/AI" look
during inference if the base-model sampling settings are wrong. In
this project's research, the same converged LoRA rendered with
Krea-2's official recommended inference settings (**50 sample steps,
guidance_scale 3.5-4.5** -- not 32 steps) at close visual inspection
scored dramatically higher on photographic realism. Krea-2 RAW is
explicitly documented by its own authors as not intended for
out-of-the-box high-quality generation at low step counts -- it is a
training substrate, and needs its full recommended step budget at
inference time regardless of how well the LoRA itself converged.

A related, more serious finding: **directly merging a second LoRA's
weight deltas into a quantized (qfloat8) transformer's raw parameter
tensors does not work** -- the forward pass reads the quantized
storage via specialized kernels, not the float32 proxy tensors that
`named_parameters()` exposes, so such a merge is silently a no-op (or
worse, can corrupt state) with no error raised. Stack additional
inference-time LoRAs (e.g. a general realism/skin-texture LoRA)
through your inference tool's native runtime LoRA-loading mechanism
(e.g. ComfyUI's `LoraLoaderModelOnly`, chained in sequence), never by
manually merging state dicts into an already-quantized model in a
custom script.
