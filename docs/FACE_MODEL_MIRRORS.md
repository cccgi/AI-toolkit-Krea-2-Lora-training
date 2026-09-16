# Face/Upscale Model Sources & Manual Download Mirrors

`setup/02_download_face_models.py` downloads three small models
(~240MB combined) directly from their original public sources -- no
FaceFusion, Automatic1111/Forge, or ComfyUI installation is required
anywhere on the machine.

## 1. YOLOv8-face (face detection)

- **File**: `yoloface_8n.onnx` (~12MB)
- **Source**: FaceFusion's own official GitHub release assets
- **URL**: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx
- **Destination**: `models/face/yoloface_8n.onnx`

If this link ever breaks, check
https://github.com/facefusion/facefusion-assets/releases for a newer
release tag that still includes `yoloface_8n.onnx` as an asset (some
newer releases replaced it with different detector models -- the
tag `models-3.0.0` specifically was verified to include it).

## 2. ArcFace w600k_r50 (face embedding / identity)

- **File**: `arcface_w600k_r50.onnx` (~166MB, extracted from a zip)
- **Source**: InsightFace's own official GitHub release (`buffalo_l`
  model pack)
- **URL**: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  (the setup script downloads this zip, extracts only
  `w600k_r50.onnx`, then deletes the zip)
- **Destination**: `models/face/arcface_w600k_r50.onnx`

If this link ever breaks, check
https://github.com/deepinsight/insightface/releases for the current
release tag. Do not use unofficial re-uploads unless you have
verified the file hash against a known-good copy -- this is a face
recognition model and unofficial copies should be treated with the
same caution as any other unverified binary.

## 3. 4x-UltraSharp (ESRGAN upscaling)

- **File**: `4x-UltraSharp.pth` (~64MB)
- **Source**: Public Hugging Face mirror
- **URL**: https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth
- **Destination**: `models/upscale/4x-UltraSharp.pth`

## Manual download instructions

If `setup/02_download_face_models.py` fails (e.g. due to a network
block or a broken link), download the three files above manually and
place them at the destination paths listed, relative to the repo
root. No renaming or reformatting is needed -- the pipeline reads
these exact filenames from `pipeline/face_engine.py` and
`pipeline/upscale_engine.py`.
