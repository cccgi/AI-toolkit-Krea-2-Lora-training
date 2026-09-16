"""
Downloads the Krea-2 RAW transformer, Qwen3-VL-4B-Instruct text encoder,
and Qwen-Image VAE from Hugging Face into a local cache directory that
the training configs point to.

krea/Krea-2-Raw is a gated repository. Before running this script:
  1. Log in at https://huggingface.co/join if you don't have an account.
  2. Visit https://huggingface.co/krea/Krea-2-Raw and request/accept access.
  3. Run `hf auth login` (or `huggingface-cli login`) once per machine.

This script is idempotent -- re-running it will skip files already
present in the cache.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(REPO_ROOT, "model_cache", "huggingface")

MODELS = [
    ("krea/Krea-2-Raw", "Krea-2 RAW transformer (~13GB)"),
    ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct text encoder (~9GB)"),
    ("Qwen/Qwen-Image", "Qwen-Image VAE (~0.25GB, only the VAE subfolder is used)"),
]


def main():
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from huggingface_hub import get_token, snapshot_download
    from huggingface_hub.utils import GatedRepoError, LocalTokenNotFoundError

    if not get_token():
        print("[ERROR] No Hugging Face token found. Run `hf auth login` first.")
        sys.exit(1)

    for repo_id, label in MODELS:
        print(f"\n=== Downloading {repo_id} ({label}) ===")
        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"  -> cached at {path}")
        except GatedRepoError:
            print(
                f"[ERROR] Access to {repo_id} is gated and not yet granted to your "
                f"account. Visit https://huggingface.co/{repo_id} and request access, "
                f"then re-run this script."
            )
            sys.exit(1)
        except LocalTokenNotFoundError:
            print("[ERROR] No Hugging Face token found. Run `hf auth login` first.")
            sys.exit(1)

    print("\nAll model files downloaded successfully.")
    print(f"Cache location: {os.environ['HF_HUB_CACHE']}")


if __name__ == "__main__":
    main()
