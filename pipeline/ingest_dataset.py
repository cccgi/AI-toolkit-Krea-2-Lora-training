"""
Smart LoRA dataset ingestor for Krea-2 (and Flux-family) character training.

Pipeline stages, all automated from a single raw photo folder:
  1. Quality gating: rejects blurry, severely under/over-exposed, or
     faceless images.
  2. Face detection + pose classification (YOLOv8-face).
  3. Smart portrait cropping around the detected face with natural
     head/shoulder framing.
  4. AI upscaling (ESRGAN) only for images below the target resolution.
  5. Face-region mask generation for regional loss weighting
     (mask_min_value in the ai-toolkit dataset config) -- this was the
     single highest-impact fix found in this project's Flux.2 research.
  6. True holdout partitioning: reserves ~15% of accepted images,
     evenly spread across detected poses, that are NEVER used for
     training -- only for post-training ArcFace convergence evaluation.
  7. Caption generation. Krea-2's Qwen3-VL-4B text encoder responds far
     more strongly to descriptive natural-language captions than short
     trigger-tag captions (see docs/KREA2_FINDINGS.md) -- captions here
     are deliberately fuller than a Flux/CLIP-style trigger+class tag.
"""
import os
import glob
import json
import re
import argparse
import cv2
import numpy as np
from PIL import Image

from pipeline.face_engine import FaceEngine
from pipeline.upscale_engine import UpscaleEngine


POSE_PHRASES = {
    "frontal": "facing the camera directly",
    "three_quarter_left": "turned slightly to her left",
    "three_quarter_right": "turned slightly to her right",
    "profile_left": "in left profile",
    "profile_right": "in right profile",
}


def compute_blur_score(cv_img):
    """Laplacian variance; higher = sharper. Below ~75 is typically blurry."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure_stats(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    underexposed = float(np.mean(gray < 10))
    overexposed = float(np.mean(gray > 245))
    return mean_val, underexposed, overexposed


def calculate_smart_portrait_crop(w_orig, h_orig, box):
    """Expands a detected face box into a natural head-and-torso portrait crop."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    margin_top = int(bh * 0.65)
    margin_bottom = int(bh * 1.85)
    margin_x = int(bw * 1.10)

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_top)
    crop_x2 = min(w_orig, x2 + margin_x)
    crop_y2 = min(h_orig, y2 + margin_bottom)

    return (crop_x1, crop_y1, crop_x2, crop_y2)


def build_caption(trigger, class_noun, pose, style_hint=""):
    """
    Builds a descriptive natural-language caption suited to Krea-2's
    Qwen3-VL text encoder. `trigger` is kept as a light identity anchor
    at the start of the sentence rather than the whole caption.
    """
    pose_phrase = POSE_PHRASES.get(pose, "looking toward the camera")
    base = f"A photo of {trigger}, a {class_noun}, {pose_phrase} in natural lighting"
    if style_hint:
        base += f", {style_hint}"
    return base + "."


def ingest_dataset(
    input_dir,
    output_dir,
    trigger="sks",
    class_noun="person",
    target_res=1024,
    holdout_ratio=0.15,
    min_blur=75.0,
    enable_upscale=True,
):
    print("=== Starting Smart Dataset Ingestion ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Trigger: '{trigger}', Class noun: '{class_noun}'")

    train_dir = os.path.join(output_dir, "training")
    mask_dir = os.path.join(output_dir, "masks")
    holdout_dir = os.path.join(output_dir, "holdouts")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    face_engine = FaceEngine()
    upscale_engine = UpscaleEngine() if enable_upscale else None

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    raw_files = []
    for ext in exts:
        raw_files.extend(glob.glob(os.path.join(input_dir, ext)))
        raw_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    raw_files = sorted(list(set(raw_files)))
    print(f"Found {len(raw_files)} raw input candidate images.")

    accepted = []
    rejected = []

    for fpath in raw_files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            rejected.append((fname, "Unreadable image file"))
            continue

        h, w = img.shape[:2]

        blur = compute_blur_score(img)
        if blur < min_blur:
            rejected.append((fname, f"Blurry (Laplacian var {blur:.1f} < {min_blur})"))
            continue

        mean_lum, underexposed, overexposed = compute_exposure_stats(img)
        if mean_lum < 30.0 or underexposed > 0.40:
            rejected.append((fname, f"Severe underexposure (mean lum {mean_lum:.1f})"))
            continue
        if mean_lum > 225.0 or overexposed > 0.40:
            rejected.append((fname, f"Severe overexposure (mean lum {mean_lum:.1f})"))
            continue

        detect = face_engine.detect_face(img)
        if detect is None or detect["confidence"] < 0.50:
            rejected.append((fname, "No clear primary face detected"))
            continue

        box = detect["box"]
        landmarks = detect["landmarks"]
        pose = detect["pose"]
        emb = face_engine.extract_embedding(img, box)

        crop_box = calculate_smart_portrait_crop(w, h, box)
        cx1, cy1, cx2, cy2 = crop_box
        cropped_bgr = img[cy1:cy2, cx1:cx2]

        rel_box = (box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1)
        rel_landmarks = landmarks - np.array([cx1, cy1], dtype=np.float32)

        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(cropped_rgb)

        upscaled = False
        cw, ch = pil_crop.size
        if enable_upscale and min(cw, ch) < target_res:
            try:
                pil_crop = upscale_engine.upscale(pil_crop, target_min_dim=target_res)
                nw, nh = pil_crop.size
                actual_scale = nw / float(cw)
                rel_box = tuple(int(v * actual_scale) for v in rel_box)
                rel_landmarks = rel_landmarks * actual_scale
                upscaled = True
            except Exception as e:
                print(f"Warning: Upscaling failed for {fname}: {e}")

        accepted.append({
            "fname": fname,
            "orig_path": fpath,
            "pil_crop": pil_crop,
            "rel_box": rel_box,
            "rel_landmarks": rel_landmarks,
            "pose": pose,
            "embedding": emb,
            "upscaled": upscaled,
            "blur_score": blur,
        })

    print(f"Quality gating complete: {len(accepted)} accepted, {len(rejected)} rejected.")

    if len(accepted) == 0:
        print("ERROR: No images passed the quality gate. Please review the input folder.")
        return

    num_holdouts = max(2, int(round(len(accepted) * holdout_ratio)))
    accepted.sort(key=lambda x: (x["pose"], -x["blur_score"]))

    step = max(1, len(accepted) // num_holdouts)
    holdout_indices = set(range(0, len(accepted), step)[:num_holdouts])

    training_items = [item for idx, item in enumerate(accepted) if idx not in holdout_indices]
    holdout_items = [item for idx, item in enumerate(accepted) if idx in holdout_indices]

    print(f"Partitioned: {len(training_items)} training images, {len(holdout_items)} true holdouts reserved.")

    holdout_manifest = []
    for idx, h_item in enumerate(holdout_items):
        stem = re.sub(r'^(train|holdout)_\d+_', '', os.path.splitext(h_item['fname'])[0])
        base_name = f"holdout_{idx + 1:02d}_{stem}"
        h_out_path = os.path.join(holdout_dir, f"{base_name}.png")
        h_item["pil_crop"].save(h_out_path, quality=95)
        holdout_manifest.append({
            "name": base_name,
            "path": h_out_path,
            "orig_file": h_item["fname"],
            "pose": h_item["pose"],
            "embedding": h_item["embedding"].tolist() if h_item["embedding"] is not None else None,
        })
    with open(os.path.join(holdout_dir, "holdout_manifest.json"), "w") as f:
        json.dump(holdout_manifest, f, indent=2)

    training_manifest = []
    for idx, t_item in enumerate(training_items):
        stem = os.path.splitext(t_item["fname"])[0]
        stem = re.sub(r'^(train|holdout)_\d+_', '', stem)
        base_name = f"train_{idx + 1:03d}_{stem}"
        t_out_path = os.path.join(train_dir, f"{base_name}.png")
        t_item["pil_crop"].save(t_out_path, quality=95)

        mask = face_engine.generate_face_mask(
            t_item["pil_crop"],
            t_item["rel_box"],
            t_item["rel_landmarks"],
            feather_radius=int(min(t_item["pil_crop"].size) * 0.025),
        )
        mask_out_path = os.path.join(mask_dir, f"{base_name}.png")
        mask.save(mask_out_path)

        caption = build_caption(trigger, class_noun, t_item["pose"])
        cap_path = os.path.join(train_dir, f"{base_name}.txt")
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(caption)

        training_manifest.append({
            "name": base_name,
            "image": t_out_path,
            "mask": mask_out_path,
            "caption": caption,
            "pose": t_item["pose"],
        })

    summary = {
        "total_inputs": len(raw_files),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "training_count": len(training_items),
        "holdout_count": len(holdout_items),
        "rejected_details": rejected,
        "training_manifest": training_manifest,
    }
    with open(os.path.join(output_dir, "ingest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Dataset Ingestion Succeeded! ===")
    print(f"Training images & captions: {train_dir}")
    print(f"Face regional masks: {mask_dir}")
    print(f"Holdout images (for evaluation only): {holdout_dir}")
    print(f"Summary saved to: {os.path.join(output_dir, 'ingest_summary.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart LoRA Dataset Ingestor & Formatter")
    parser.add_argument("--input_dir", required=True, help="Path to raw photos directory")
    parser.add_argument("--output_dir", required=True, help="Path to destination dataset directory")
    parser.add_argument("--trigger", default="sks", help="LoRA trigger token (e.g. sks, or a made-up name)")
    parser.add_argument("--class_noun", default="person", help="Class noun (e.g. woman, man, person)")
    parser.add_argument("--target_res", type=int, default=1024, help="Target minimum resolution")
    parser.add_argument("--holdout_ratio", type=float, default=0.15, help="Ratio of images reserved as true holdouts")
    parser.add_argument("--min_blur", type=float, default=75.0, help="Minimum Laplacian variance threshold")
    parser.add_argument("--no_upscale", action="store_true", help="Disable AI upscaling")

    args = parser.parse_args()
    ingest_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trigger=args.trigger,
        class_noun=args.class_noun,
        target_res=args.target_res,
        holdout_ratio=args.holdout_ratio,
        min_blur=args.min_blur,
        enable_upscale=not args.no_upscale,
    )
