"""
Setup step 2: download the small (~100-300MB total) face-detection,
face-embedding, and upscaling models used by the dataset ingestion
pipeline. No dependency on FaceFusion, Automatic1111/Forge, or ComfyUI --
every model here is fetched directly from its original public source
(FaceFusion's own asset releases, InsightFace's own official release,
and the public 4x-UltraSharp mirror on Hugging Face).

Run via setup\\02_download_face_models.bat, or directly:
    python setup\\02_download_face_models.py
"""
import os
import sys
import zipfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(REPO_ROOT, "models", "face")
UPSCALE_DIR = os.path.join(REPO_ROOT, "models", "upscale")

# Direct downloads: (url, destination filename, subfolder, approx size)
DOWNLOADS = [
    (
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx",
        "yoloface_8n.onnx",
        FACE_DIR,
        "~12MB",
    ),
    (
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        "4x-UltraSharp.pth",
        UPSCALE_DIR,
        "~64MB",
    ),
]

# ArcFace ships inside InsightFace's official buffalo_l.zip release bundle;
# we download the zip once and extract only the one .onnx file we need.
ARCFACE_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_ZIP_MEMBER = "w600k_r50.onnx"
ARCFACE_DEST_NAME = "arcface_w600k_r50.onnx"


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        if pct % 5 == 0:
            sys.stdout.write(f"\r  {pct:3d}%")
            sys.stdout.flush()


def download_file(url, dest_path, label):
    if os.path.exists(dest_path):
        print(f"[skip] {os.path.basename(dest_path)} already exists.")
        return
    print(f"[download] {os.path.basename(dest_path)} ({label}) ...")
    tmp_path = dest_path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path, _progress_hook)
        os.replace(tmp_path, dest_path)
        print(f"\n  -> saved to {dest_path}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"\n  [ERROR] Failed to download {url}: {e}")
        print(
            "  If this mirror is unreachable, see docs/FACE_MODEL_MIRRORS.md "
            "for alternate download links you can fetch manually."
        )
        raise


def download_and_extract_arcface():
    """
    Downloads InsightFace's official buffalo_l.zip release bundle and
    extracts only w600k_r50.onnx (the ArcFace recognition model) into
    models/face/, then deletes the temporary zip.
    """
    dest_path = os.path.join(FACE_DIR, ARCFACE_DEST_NAME)
    if os.path.exists(dest_path):
        print(f"[skip] {ARCFACE_DEST_NAME} already exists.")
        return

    zip_path = os.path.join(FACE_DIR, "_buffalo_l_tmp.zip")
    print(f"[download] buffalo_l.zip (~280MB, contains {ARCFACE_ZIP_MEMBER}) ...")
    try:
        urllib.request.urlretrieve(ARCFACE_ZIP_URL, zip_path, _progress_hook)
        print()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            member = next((n for n in names if n.endswith(ARCFACE_ZIP_MEMBER)), None)
            if member is None:
                raise FileNotFoundError(
                    f"{ARCFACE_ZIP_MEMBER} not found inside buffalo_l.zip "
                    f"(contents: {names})"
                )
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
        print(f"  -> extracted {ARCFACE_ZIP_MEMBER} to {dest_path}")
    except Exception as e:
        print(f"\n  [ERROR] Failed to fetch ArcFace model: {e}")
        print(
            "  If this mirror is unreachable, see docs/FACE_MODEL_MIRRORS.md "
            "for alternate download links you can fetch manually."
        )
        raise
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def main():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(UPSCALE_DIR, exist_ok=True)

    print("=== Downloading face-detection / embedding / upscale models ===")
    for url, filename, folder, label in DOWNLOADS:
        dest_path = os.path.join(folder, filename)
        download_file(url, dest_path, label)

    download_and_extract_arcface()

    print("\n=== Done. Verifying files ===")
    ok = True
    all_expected = [(f, d) for _, f, d, _ in DOWNLOADS] + [(ARCFACE_DEST_NAME, FACE_DIR)]
    for filename, folder in all_expected:
        dest_path = os.path.join(folder, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
            print(f"  OK: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
        else:
            print(f"  MISSING or too small: {dest_path}")
            ok = False

    if not ok:
        print("\nSome downloads failed or produced unexpectedly small files.")
        print("See docs/FACE_MODEL_MIRRORS.md for manual download instructions.")
        sys.exit(1)

    print("\nAll face/upscale models are ready.")


if __name__ == "__main__":
    main()
