# Krea-2 RAW Character LoRA Training Pipeline

A fully automated, end-to-end LoRA training pipeline for **Krea-2 RAW**
that solves the same core problem this project's parallel FLUX.2 Klein
work ran into: undertrained/undersized adapters that never actually
converge on the subject's identity, mistaken for "done" because the
loss curve looks fine.

The fix here is **rank 32/32 (not 16) + face-region regional loss
weighting** during training (via the
[ai-toolkit-perceptual](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual)
fork of [Ostris ai-toolkit](https://github.com/ostris/ai-toolkit)),
combined with descriptive natural-language captioning (Krea-2's
Qwen3-VL-4B text encoder responds far more weakly to short trigger-tag
captions than Flux/CLIP-style models do) and forensic holdout
evaluation that measures identity fidelity mathematically instead of
eyeballing generated samples.

Everything here was built and validated across real training runs —
see [`docs/KREA2_FINDINGS.md`](docs/KREA2_FINDINGS.md) for the full
forensic history: an undertrained rank-16 run plateauing at ArcFace
similarity 0.42, and the rank-32 + face-mask run that converged to
0.51, matching this project's parallel Flux.2 Klein 4B/9B research on
the same subject.

**Designed for:** a fresh Windows machine with an NVIDIA GPU, nothing
installed yet — no Python packages, no conda env, no models, no
ai-toolkit. Three setup scripts get you from nothing to a working
double-click training pipeline.

---

## What this actually does differently

A LoRA trained at too-low rank, or with plain full-frame loss
weighting, can *look* like it's converging (the loss number drops)
while the ArcFace-measured identity similarity against true holdout
photos is still climbing steeply with no plateau in sight — meaning it
was stopped before it actually learned the person, not because it was
finished.

This pipeline adds:

1. **Rank 32/32 as the validated floor**, not rank 16. Krea-2 RAW is a
   12.9B-parameter single-stream MMDiT — roughly 3x the size of Flux
   Klein 4B — and a rank-16 adapter simply doesn't have the capacity to
   bind a specific face at that scale in a normal step budget.
2. **Face-region regional loss weighting** (`mask_path` +
   `mask_min_value: 0.10`) — the single highest-impact fix found across
   this project's Flux.2 research, ported to Krea-2's architecture via
   the `ai-toolkit-perceptual` fork. Concentrates gradient on the face
   instead of weighting the whole frame (background, clothing, hair)
   equally.
3. **Descriptive natural-language captioning** — Krea-2's Qwen3-VL-4B
   text encoder responds weakly to bare trigger-word tag captions
   (a real, documented difference from Flux/CLIP-style models).
   Captions here are full sentences (`"A photo of kmb, a woman, facing
   the camera directly in natural lighting."`) with the trigger kept
   as a light identity anchor, not the whole caption.
4. **True holdout evaluation** — ~15% of your source photos are never
   shown to the trainer, reserved purely to mathematically score final
   identity fidelity (ArcFace cosine similarity against a holdout
   centroid) — not "does this look right to me," which is subject to
   confirmation bias once you've stared at generated samples for hours.

---

## Prerequisites

- **Windows**, NVIDIA GPU with **24GB VRAM recommended** (Krea-2 RAW is
  a 12.9B-parameter model; this pipeline was built and validated on an
  RTX 3090).
- **Git for Windows**: https://git-scm.com/download/win
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html
- ~40GB free disk space (Krea-2 RAW ~13GB, Qwen3-VL-4B-Instruct ~9GB,
  Qwen-Image VAE ~0.25GB, plus training checkpoints).
- A [HuggingFace account](https://huggingface.co/join) with access
  granted to the gated `krea/Krea-2-Raw` model repo (see Setup step 2
  below). Free, takes a couple of minutes.

You do **not** need to have ai-toolkit, PyTorch, CUDA toolkit, or any
Python packages pre-installed — the setup scripts below handle
everything.

---

## Setup (run once)

### 1. Clone this repo

```bat
git clone https://github.com/cccgi/AI-toolkit-Krea-2-Lora-training.git
cd AI-toolkit-Krea-2-Lora-training
```

### 2. Request HuggingFace access to Krea-2 RAW

1. Create a free account at https://huggingface.co/join if you don't
   have one.
2. Visit https://huggingface.co/krea/Krea-2-Raw and request/accept
   access (this is a gated repo). Usually granted automatically or
   within a short time.
3. You'll log in interactively via `hf auth login` during step 4 below
   — no token pasted into any file.

### 3. Install ai-toolkit-perceptual and its conda environment

```bat
setup\00_install_ai_toolkit_perceptual.bat
```

This clones the `ai-toolkit-perceptual` fork (see
[`docs/WHY_THIS_FORK.md`](docs/WHY_THIS_FORK.md) for why this fork
specifically) into a sibling folder next to this repo, creates a
dedicated `ai-toolkit-perceptual` conda environment (Python 3.11), and
installs PyTorch (CUDA 12.1 build) plus all of ai-toolkit's training
dependencies. Takes 10-30 minutes depending on your connection.

### 4. Download the Krea-2 model files

```bat
setup\01_download_krea2_model.bat
```

Logs you into HuggingFace (interactive browser login via `hf auth
login`) and downloads:
- Krea-2 RAW transformer (~13GB)
- Qwen3-VL-4B-Instruct text encoder (~9GB)
- Qwen-Image VAE (~0.25GB)

If this fails with a "gated repo"/"access restricted" error, you
haven't completed step 2 yet — go request access, wait for it to be
granted, then re-run (already-downloaded files are skipped, safe to
re-run).

### 5. Download face-detection and upscaling models

```bat
setup\02_download_face_models.bat
```

Downloads ~240MB of small models (YOLOv8-face for detection, ArcFace
w600k_r50 for identity embeddings/masks, 4x-UltraSharp for upscaling
small photos) directly from their original public sources — FaceFusion's
own GitHub release assets and InsightFace's own official release. No
dependency on FaceFusion, Automatic1111/Forge, or ComfyUI being
installed anywhere. See
[`docs/FACE_MODEL_MIRRORS.md`](docs/FACE_MODEL_MIRRORS.md) for exact
sources and manual download instructions if this ever fails.

---

## Training a LoRA (every time after setup)

### Double-click

Open `RUN_ME_train_lora.bat` in a text editor, set these 4 values near
the top, save, then double-click:

```bat
set SUBJECT_NAME=kimberly
set RAW_PHOTOS_DIR=C:\path\to\your\raw\photos\folder
set TRIGGER=kmb
set CLASS_NOUN=woman
```

### Command line (equivalent)

```bat
conda activate ai-toolkit-perceptual
python pipeline\run_full_pipeline.py ^
  --ai_toolkit_dir "ai-toolkit-perceptual" ^
  --raw_photos_dir "C:\path\to\your\raw\photos\folder" ^
  --subject_name kimberly ^
  --trigger kmb ^
  --class_noun woman
```

### What happens

```
[Raw Photos]
     |
     v Stage 1: Ingestion & Quality Gating (pipeline/ingest_dataset.py)
[Framed Photos + Face-Region Masks + Reserved True Holdouts + Captions]
     |
     v Stage 2: Config Generation (pipeline/generate_training_config.py)
[Smoke-Test Config + Full Training Config, from validated rank32/mask/2800-step defaults]
     |
     v Stage 3: 1-Step Smoke Test
[Catches config/environment problems before committing GPU time]
     |
     v Stage 4: Full Training Run (~6-7 hours on an RTX 3090)
[Checkpoints Every 400 Steps + Sample Images]
     |
     v Stage 5: Forensic Evaluation (pipeline/evaluate_run.py)
[ArcFace Cosine Identity Scorecard + Visual Holdout Audit Grid + Peak Checkpoint]
```

Everything above runs automatically as a single chain when you
double-click `RUN_ME_train_lora.bat` — there is no separate manual
evaluation step to remember to run afterward.

Your trained LoRA `.safetensors` file and the evaluation report land
in:

```
workspace\<SUBJECT_NAME>\output\<SUBJECT_NAME>_krea2_raw_r32_2800\
  evaluation_report.md
  holdout_evaluation_grid.jpg
  <SUBJECT_NAME>_krea2_raw_r32_2800.safetensors        (final checkpoint)
  <SUBJECT_NAME>_krea2_raw_r32_2800_000002000.safetensors  (intermediate checkpoints)
  samples/
```

The report tells you the peak checkpoint (highest ArcFace similarity
against your true holdout photos, ~0.50+ indicates confident
convergence) and flags drift if the model started degrading late in
training.

---

## Validated training defaults

| Setting | Value | Why |
|---|---|---|
| Rank / alpha | 32 / 32 | Rank 16 plateaued at ArcFace 0.42 (still climbing, undertrained). Rank 32 reached 0.51 (converged). See `docs/KREA2_FINDINGS.md`. |
| Face-region mask weighting | `mask_min_value: 0.10` | The single highest-impact fix found across this project's Flux.2 and Krea-2 research. |
| Steps | 2800 | Validated baseline for a ~20-image dataset; similarity curve visibly flattens after ~step 2000. |
| Resolution | 1024 | Krea-2's native/recommended training resolution. |
| Quantization | `qfloat8` (transformer + text encoder) | Community-standard AI-Toolkit setting for Krea-2; required to fit a 12.9B model + 4B text encoder in 24GB VRAM. |
| Text encoder | Frozen, cached embeddings | Required for VRAM budget; unrelated to likeness quality. |

Override `--rank`, `--steps`, or `--resolution` on the
`run_full_pipeline.py` command line to deviate from these defaults.
See [`docs/KREA2_FINDINGS.md`](docs/KREA2_FINDINGS.md) for what
happens if rank/masking are skipped.

---

## Repository layout

```
pipeline/
  ingest_dataset.py            Stage 1: quality gate, smart-crop, upscale, mask, holdout split, captioning
  generate_training_config.py  Stage 2: generates ai-toolkit YAML configs from validated defaults
  evaluate_run.py               Stage 5: ArcFace holdout scoring, visual grid, markdown report
  face_engine.py                 Shared: YOLOv8-face detection + ArcFace embeddings + mask generation
  upscale_engine.py               Shared: ESRGAN upscaling for undersized source photos
  run_full_pipeline.py             Orchestrates Stages 1-5 end to end
setup/
  00_install_ai_toolkit_perceptual.bat   Clones ai-toolkit-perceptual, creates conda env, installs torch/deps
  01_download_krea2_model.bat             Downloads Krea-2 RAW + Qwen3-VL-4B + Qwen-Image VAE from HuggingFace
  02_download_face_models.bat             Downloads face/upscale models
  02_download_face_models.py              (called by the .bat above)
  download_krea2_model.py                 (called by 01's .bat above)
docs/
  KREA2_FINDINGS.md              Forensic history: rank-16 vs rank-32, what failed, why, and the fix
  WHY_THIS_FORK.md                Why ai-toolkit-perceptual instead of vanilla ai-toolkit
  FACE_MODEL_MIRRORS.md            Exact sources for the face/upscale models + manual download instructions
  QUICKSTART.md                     Full step-by-step walkthrough with troubleshooting
RUN_ME_train_lora.bat            Windows double-click entry point
```

---

## Troubleshooting

- **"GPU already has more than 2GB in use"** — another process is
  using the GPU. Check `nvidia-smi`, close it, and re-run; this
  launcher never kills other processes for you.
- **401/403 or "Access to model krea/Krea-2-Raw is restricted"** —
  your HuggingFace account hasn't been granted access to the gated
  repo yet; see Setup step 2.
- **Smoke test fails with a `mask_path` or `mask_min_value` error** —
  you may be running vanilla `ostris/ai-toolkit` instead of the
  `ai-toolkit-perceptual` fork. See
  [`docs/WHY_THIS_FORK.md`](docs/WHY_THIS_FORK.md).
- **Face detection rejects most/all of my photos** — check
  `ingest_summary.json` in your dataset output folder for per-image
  rejection reasons (blurry, no face detected, severe exposure
  issues). Very low-resolution, heavily cropped, or extreme-angle
  photos are the most common causes.
- **Peak ArcFace similarity is well below ~0.50** — the curve may not
  have plateaued yet. Try more steps and/or confirm rank 32 and the
  face-mask weighting are actually active in your generated config
  (`workspace\<subject>\configs\*.yaml`). See
  [`docs/KREA2_FINDINGS.md`](docs/KREA2_FINDINGS.md) for the full
  rank-16-vs-32 comparison.

---

## License

This repository's own code (`pipeline/`, `setup/`) is MIT licensed —
see [LICENSE](LICENSE). `ai-toolkit-perceptual` and `ai-toolkit` are
separate MIT-licensed projects installed alongside this one. Krea-2,
Qwen3-VL, and all downloaded model weights retain their own original
licenses — Krea-2 in particular has a non-commercial-by-default use
policy; read it at https://www.krea.ai/krea-2-use-policy before use.
