# AI Toolkit Krea-2 LoRA Training

A fully automated, one-click character LoRA training pipeline for
**Krea-2 RAW**, built on top of [ai-toolkit](https://github.com/ostris/ai-toolkit)
(via the [ai-toolkit-perceptual](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual)
fork). Give it a folder of raw, unprocessed photos of a person and it
will crop, upscale, mask, caption, train, and forensically evaluate a
converged identity LoRA -- no manual dataset prep required.

This repository packages the settings and pipeline that were actually
validated to work, documented in detail in [`docs/KREA2_FINDINGS.md`](docs/KREA2_FINDINGS.md).
The short version: **rank 32/32 + face-region regional loss weighting
+ ~2800 steps** took a from-scratch Krea-2 RAW LoRA from an ArcFace
identity-similarity score of 0.42 (undertrained, rank 16, no masking)
to **0.51** (converged) against true holdout photos never seen during
training -- matching the convergence level of parallel Flux.2 Klein
4B/9B LoRA research on the same subject.

## What this does, end to end

```
raw photo folder
      |
      v
[1] Quality gate + face detect + smart crop + AI upscale (pipeline/ingest_dataset.py)
      |
      v
[2] Face-region mask generation + descriptive captioning
      |
      v
[3] True holdout partitioning (~15% of images, never trained on)
      |
      v
[4] 1-step smoke test (catches config/env problems before wasting GPU time)
      |
      v
[5] Full Krea-2 RAW LoRA training (rank 32, masked regional loss, 2800 steps)
      |
      v
[6] ArcFace forensic evaluation against the true holdouts (pipeline/evaluate_run.py)
      |
      v
trained LoRA + evaluation_report.md + visual audit grid
```

## Quick start (fresh Windows machine, nothing installed)

1. Install prerequisites: [Git](https://git-scm.com/), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), an NVIDIA GPU with recent drivers (24GB VRAM recommended; see [Hardware requirements](#hardware-requirements)).
2. Clone this repo:
   ```bat
   git clone https://github.com/cccgi/AI-toolkit-Krea-2-Lora-training.git
   cd AI-toolkit-Krea-2-Lora-training
   ```
3. Run the three setup scripts in order, once:
   ```bat
   setup\00_install_ai_toolkit_perceptual.bat
   setup\01_download_krea2_model.bat
   setup\02_download_face_models.bat
   ```
4. Edit `RUN_ME_train_lora.bat`: set `SUBJECT_NAME`, `RAW_PHOTOS_DIR`, `TRIGGER`, `CLASS_NOUN`.
5. Double-click `RUN_ME_train_lora.bat`.

Full walkthrough with screenshots-in-words and troubleshooting: [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## Why ai-toolkit-perceptual, not vanilla ai-toolkit

This project trains on the `ai-toolkit-perceptual` fork rather than
upstream `ostris/ai-toolkit`. See [`docs/WHY_THIS_FORK.md`](docs/WHY_THIS_FORK.md)
for the specific fixes and features this fork carries that were needed
to get Krea-2 RAW training working reliably (face-mask regional loss
weighting support, dataset caching fixes, and Krea-2 architecture
compatibility patches).

## Why no dependency on FaceFusion / A1111 / ComfyUI

The face-detection (YOLOv8-face), face-embedding (ArcFace w600k_r50),
and upscaling (4x-UltraSharp ESRGAN) models used here are small
(~240MB combined) and downloaded directly from their original public
sources -- FaceFusion's own GitHub release assets and InsightFace's
own official release. You do not need FaceFusion, Automatic1111,
Forge, or ComfyUI installed anywhere on the machine. See
[`docs/FACE_MODEL_MIRRORS.md`](docs/FACE_MODEL_MIRRORS.md) for exact
sources and manual download instructions if the automated script's
mirrors ever go down.

## Hardware requirements

- NVIDIA GPU with 24GB VRAM (Krea-2 RAW is a 12.9B-parameter model;
  this was trained and validated on an RTX 3090).
- ~40GB free disk space for model downloads (Krea-2 RAW ~13GB,
  Qwen3-VL-4B-Instruct ~9GB, Qwen-Image VAE ~0.25GB, plus training
  checkpoints).
- A full 2800-step training run takes roughly 6-7 hours on an RTX 3090.

## Repository layout

```
setup/                          One-time setup scripts (run in order 00, 01, 02)
pipeline/
  face_engine.py                 Face detection + ArcFace embedding + mask generation
  upscale_engine.py               ESRGAN upscaling
  ingest_dataset.py                Dataset ingestion: crop, upscale, mask, caption, holdout split
  generate_training_config.py       Generates ai-toolkit YAML configs from validated defaults
  evaluate_run.py                    ArcFace forensic evaluation against true holdouts
  run_full_pipeline.py                Orchestrates all of the above end to end
RUN_ME_train_lora.bat            Double-click entry point (edit 4 values, then run)
docs/
  QUICKSTART.md                   Detailed walkthrough + troubleshooting
  KREA2_FINDINGS.md                Full research trail: what worked, what didn't, and why
  WHY_THIS_FORK.md                  Why ai-toolkit-perceptual instead of vanilla ai-toolkit
  FACE_MODEL_MIRRORS.md              Exact sources for the face/upscale models
workspace/                       Created at runtime: per-subject datasets, configs, outputs (gitignored)
model_cache/                     Created at runtime: Hugging Face model cache (gitignored)
models/                          Created at runtime: face/upscale models (gitignored)
```

## License

This repository's own scripts are MIT-licensed (see [`LICENSE`](LICENSE)).
Krea-2, Qwen3-VL, and all downloaded model weights retain their own
original licenses -- review each model's license/usage policy before
use (Krea-2 in particular has a non-commercial-by-default use policy;
see https://www.krea.ai/krea-2-use-policy).
