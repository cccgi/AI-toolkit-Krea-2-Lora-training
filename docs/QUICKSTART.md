# Quickstart: Training Your First Krea-2 Character LoRA

This walks through the complete process on a fresh Windows machine
with nothing installed yet.

## Prerequisites

1. **NVIDIA GPU, 24GB VRAM recommended.** Krea-2 RAW is a 12.9B
   parameter model; this pipeline was built and validated on an
   RTX 3090. Less VRAM may work with more aggressive quantization but
   is untested by this project.
2. **Git for Windows**: https://git-scm.com/download/win
3. **Miniconda**: https://docs.conda.io/en/latest/miniconda.html
4. **~40GB free disk space** for model downloads and training
   checkpoints.
5. **A Hugging Face account** with access granted to the gated
   `krea/Krea-2-Raw` model repository (see Step 2 below).

## Step 1: Clone this repository

```bat
git clone https://github.com/cccgi/AI-toolkit-Krea-2-Lora-training.git
cd AI-toolkit-Krea-2-Lora-training
```

## Step 2: Request Hugging Face access to Krea-2 RAW

Krea-2 RAW is a gated model. Before downloading it:

1. Create a free account at https://huggingface.co/join if you don't
   have one.
2. Visit https://huggingface.co/krea/Krea-2-Raw and click to
   request/accept access. This is usually granted automatically or
   within a short time.
3. You will log in via the `hf auth login` command during Step 4
   below (interactive browser login, no token pasted into any file).

## Step 3: Install ai-toolkit-perceptual and its conda environment

```bat
setup\00_install_ai_toolkit_perceptual.bat
```

This clones the `ai-toolkit-perceptual` fork (see
[`WHY_THIS_FORK.md`](WHY_THIS_FORK.md) for why this fork specifically),
creates a `ai-toolkit-perceptual` conda environment with Python 3.11,
and installs PyTorch (CUDA 12.1 build) plus all of ai-toolkit's
requirements. This step downloads several GB of Python packages and
can take 10-30 minutes depending on your connection.

## Step 4: Download the Krea-2 model files

```bat
setup\01_download_krea2_model.bat
```

This logs you into Hugging Face (interactive, via `hf auth login` --
opens a browser, no token typed into any window) and downloads:
- Krea-2 RAW transformer (~13GB)
- Qwen3-VL-4B-Instruct text encoder (~9GB)
- Qwen-Image VAE (~0.25GB)

If this fails with a "gated repo" / "access restricted" error, you
have not yet completed Step 2 above -- go request access, wait for it
to be granted, then re-run this script (it's safe to re-run; already
downloaded files are skipped).

## Step 5: Download face-detection and upscaling models

```bat
setup\02_download_face_models.bat
```

Downloads ~240MB of small models (face detector, face embedder,
upscaler) directly from their original public sources. See
[`FACE_MODEL_MIRRORS.md`](FACE_MODEL_MIRRORS.md) for exact sources and
manual download instructions if this ever fails.

## Step 6: Gather your subject's photos

Put 15-30 clear photos of the person you want to train a LoRA of into
a folder. Guidelines:

- Vary pose, lighting, expression, and background across the photos.
- Include some close-up face shots and some wider shots with visible
  shoulders/torso.
- Avoid heavily filtered, sunglasses-obscured, or extreme-angle-only
  photos.
- The pipeline automatically upscales images below 1024px on their
  short side, so lower-resolution photos are usable, but sharper is
  always better.
- You do not need to crop, resize, or caption anything yourself -- the
  pipeline does all of that.

## Step 7: Edit and run the training launcher

Open `RUN_ME_train_lora.bat` in a text editor and set these 4 values
near the top:

```bat
set SUBJECT_NAME=kimberly
set RAW_PHOTOS_DIR=C:\path\to\your\raw\photos\folder
set TRIGGER=kmb
set CLASS_NOUN=woman
```

- `SUBJECT_NAME`: a short lowercase identifier, used for file/folder
  naming (no spaces).
- `RAW_PHOTOS_DIR`: the folder from Step 6.
- `TRIGGER`: a short, distinctive token used in captions and at
  inference time to invoke this specific LoRA (e.g. a made-up short
  name or nickname). Avoid common English words.
- `CLASS_NOUN`: `woman`, `man`, or `person`.

Save the file, then double-click it. The pipeline will:

1. Ingest and prepare your dataset (crop, upscale, mask, caption,
   partition holdouts) -- a few minutes.
2. Run a 1-step smoke test to catch any environment/config problems
   before committing GPU time -- under 2 minutes.
3. Run full training (2800 steps, ~6-7 hours on an RTX 3090).
4. Run ArcFace forensic evaluation against the held-out photos and
   generate a report + visual grid.

## Step 8: Review your results

After completion, check:

```
workspace\<SUBJECT_NAME>\output\<SUBJECT_NAME>_krea2_raw_r32_2800\evaluation_report.md
workspace\<SUBJECT_NAME>\output\<SUBJECT_NAME>_krea2_raw_r32_2800\holdout_evaluation_grid.jpg
```

The report tells you the peak checkpoint (highest ArcFace similarity
against your true holdout photos) and whether the run converged
cleanly or showed signs of drift. See
[`KREA2_FINDINGS.md`](KREA2_FINDINGS.md) for what a converged score
(~0.50+) looks like and what to try if your run scores lower.

Your trained LoRA `.safetensors` file for the peak checkpoint is in
that same output folder, ready to use in ComfyUI or any other
Krea-2-compatible inference tool.

## Troubleshooting

**"GPU already has more than 2GB in use"** -- another process is using
the GPU. Close it (check `nvidia-smi`) before re-running; this
launcher never kills other processes for you.

**Smoke test fails with a `mask_path` or `mask_min_value` error** --
you may be running vanilla `ostris/ai-toolkit` instead of the
`ai-toolkit-perceptual` fork. See
[`WHY_THIS_FORK.md`](WHY_THIS_FORK.md).

**"Access to model krea/Krea-2-Raw is restricted"** -- see Step 2;
your Hugging Face account has not yet been granted access to the
gated repo.

**Face detection rejects most/all of my photos** -- check the
`ingest_summary.json` in your dataset output folder for per-image
rejection reasons (blurry, no face detected, severe exposure issues).
Very low-resolution, heavily cropped, or extreme-angle photos are the
most common causes.
