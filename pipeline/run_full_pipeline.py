"""
Master automation script: given a raw photo folder and a subject name,
runs the entire chain end to end:

  1. Ingest dataset (crop, upscale, mask, caption, holdout partition)
  2. Smoke-test train (1 step) to catch config/environment problems
     before committing GPU time to a full run
  3. Full Krea-2 RAW LoRA training run
  4. ArcFace forensic evaluation against the true holdouts

Designed to be invoked from RUN_ME_train_lora.bat with all parameters
already filled in for a specific subject folder -- see that file's
header comment for the parameters you edit per person.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(label, cmd, cwd=None, env=None):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"\n[ABORT] Step failed: {label} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Full Krea-2 LoRA training pipeline")
    parser.add_argument("--ai_toolkit_dir", required=True, help="Path to ai-toolkit-perceptual install")
    parser.add_argument("--raw_photos_dir", required=True, help="Folder of raw, unprocessed subject photos")
    parser.add_argument("--subject_name", required=True, help="Short identifier, e.g. 'kimberly'")
    parser.add_argument("--trigger", required=True, help="LoRA trigger token, e.g. 'kmb' or a made-up name")
    parser.add_argument("--class_noun", default="person", help="Class noun: woman, man, person, etc.")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (linear/linear_alpha). 32 is the validated baseline.")
    parser.add_argument("--steps", type=int, default=2800, help="Total training steps. 2800 is the validated baseline for ~20 images.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--skip_smoke_test", action="store_true", help="Skip the 1-step smoke test gate (not recommended)")
    args = parser.parse_args()

    dataset_dir = os.path.join(REPO_ROOT, "workspace", args.subject_name, "dataset")
    output_dir = os.path.join(REPO_ROOT, "workspace", args.subject_name, "output")
    configs_dir = os.path.join(REPO_ROOT, "workspace", args.subject_name, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    python_exe = sys.executable

    # 1. Ingest dataset
    run_step(
        "STEP 1/4: Ingesting and preparing dataset",
        [
            python_exe, "-m", "pipeline.ingest_dataset",
            "--input_dir", args.raw_photos_dir,
            "--output_dir", dataset_dir,
            "--trigger", args.trigger,
            "--class_noun", args.class_noun,
            "--target_res", str(args.resolution),
        ],
    )

    # 2. Generate training configs (smoke + full) from the template
    from pipeline.generate_training_config import generate_configs
    smoke_config_path, full_config_path = generate_configs(
        subject_name=args.subject_name,
        trigger=args.trigger,
        class_noun=args.class_noun,
        dataset_train_dir=os.path.join(dataset_dir, "training"),
        dataset_mask_dir=os.path.join(dataset_dir, "masks"),
        output_dir=output_dir,
        configs_dir=configs_dir,
        rank=args.rank,
        steps=args.steps,
        resolution=args.resolution,
    )

    # The calling .bat activates the ai-toolkit-perceptual conda env before
    # invoking this script, so plain "python" already resolves to the
    # correct interpreter without needing an explicit path.
    run_python = "python"

    env = os.environ.copy()
    hf_cache = os.path.join(REPO_ROOT, "model_cache", "huggingface")
    env["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            env["HF_TOKEN"] = f.read().strip()

    # 3. Smoke test
    if not args.skip_smoke_test:
        run_step(
            "STEP 2/4: Running 1-step smoke test",
            [run_python, "run.py", smoke_config_path],
            cwd=args.ai_toolkit_dir,
            env=env,
        )
    else:
        print("\n[SKIPPED] Smoke test (--skip_smoke_test set)")

    # 4. Full training run
    run_step(
        f"STEP 3/4: Running full training ({args.steps} steps)",
        [run_python, "run.py", full_config_path],
        cwd=args.ai_toolkit_dir,
        env=env,
    )

    # 5. Evaluate
    run_step(
        "STEP 4/4: Running ArcFace forensic evaluation",
        [
            run_python, "-m", "pipeline.evaluate_run",
            "--run_dir", os.path.join(output_dir, f"{args.subject_name}_krea2_raw_r{args.rank}_{args.steps}"),
            "--holdout_dir", os.path.join(dataset_dir, "holdouts"),
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    print("\n\n=== ALL STEPS COMPLETE ===")
    print(f"LoRA output: {output_dir}")
    print(f"Evaluation report + visual grid are in the run's output folder above.")


if __name__ == "__main__":
    main()
