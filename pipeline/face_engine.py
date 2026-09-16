"""
Face detection, embedding, and mask generation engine.

Self-contained: uses only onnxruntime + opencv, no dependency on
FaceFusion, ComfyUI, or any other third-party face-swap/detection
application. All models are small (~100-300MB total) and downloaded
by setup/02_download_face_models.py into models/face/.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_YOLO_PATH = os.path.join(REPO_ROOT, "models", "face", "yoloface_8n.onnx")
DEFAULT_ARCFACE_PATH = os.path.join(REPO_ROOT, "models", "face", "arcface_w600k_r50.onnx")


class FaceEngine:
    def __init__(self, yolo_path=DEFAULT_YOLO_PATH, arcface_path=DEFAULT_ARCFACE_PATH, device="cpu"):
        for p, name in [(yolo_path, "YOLOv8-face"), (arcface_path, "ArcFace w600k_r50")]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{name} model not found at {p}. Run setup\\02_download_face_models.bat first."
                )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.yolo_sess = ort.InferenceSession(yolo_path, providers=["CPUExecutionProvider"])
        self.arc_sess = ort.InferenceSession(arcface_path, providers=["CPUExecutionProvider"])

    def detect_face(self, img_bgr_or_rgb, is_rgb=False, min_conf=0.45):
        """
        Detects the primary (highest-confidence) face in an image.
        Returns dict with: box (x1, y1, x2, y2), confidence, landmarks (5, 2), pose, width, height.
        Returns None if no face is found above min_conf.
        """
        if is_rgb:
            bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = img_bgr_or_rgb

        h_orig, w_orig = bgr.shape[:2]
        scale = min(640.0 / w_orig, 640.0 / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(bgr, (nw, nh))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        dx = (640 - nw) // 2
        dy = (640 - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        inp = (canvas[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        out = self.yolo_sess.run(None, {'input': inp})[0]
        preds = out[0].T  # [8400, 20]
        boxes = preds[:, :4]
        confs = preds[:, 4]
        kpts = preds[:, 5:]  # 5 keypoints (x, y, conf)

        valid = confs > min_conf
        if not np.any(valid):
            return None

        valid_indices = np.where(valid)[0]
        best_idx = valid_indices[np.argmax(confs[valid_indices])]
        conf = float(confs[best_idx])
        bx, by, bw, bh = boxes[best_idx]

        bx = (bx - dx) / scale
        by = (by - dy) / scale
        bw /= scale
        bh /= scale

        x1 = max(0, int(bx - bw / 2.0))
        y1 = max(0, int(by - bh / 2.0))
        x2 = min(w_orig, int(bx + bw / 2.0))
        y2 = min(h_orig, int(by + bh / 2.0))

        raw_kpts = kpts[best_idx].reshape(-1, 3)
        rescaled_kpts = []
        for p in raw_kpts:
            px = (p[0] - dx) / scale
            py = (p[1] - dy) / scale
            rescaled_kpts.append([px, py])
        landmarks = np.array(rescaled_kpts, dtype=np.float32)

        # Classify pose/angle from eye-to-nose horizontal offsets
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        d_left = abs(nose[0] - left_eye[0])
        d_right = abs(right_eye[0] - nose[0])
        total_eye_w = abs(right_eye[0] - left_eye[0]) + 1e-6
        ratio = (d_left - d_right) / total_eye_w

        if ratio > 0.35:
            pose = "profile_right"
        elif ratio > 0.15:
            pose = "three_quarter_right"
        elif ratio < -0.35:
            pose = "profile_left"
        elif ratio < -0.15:
            pose = "three_quarter_left"
        else:
            pose = "frontal"

        return {
            "box": (x1, y1, x2, y2),
            "confidence": conf,
            "landmarks": landmarks,
            "pose": pose,
            "width": w_orig,
            "height": h_orig,
        }

    def extract_embedding(self, img_bgr_or_rgb, box=None, is_rgb=False):
        """Extracts a 512-dim, unit-normalized ArcFace identity embedding."""
        if is_rgb:
            bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = img_bgr_or_rgb

        h, w = bgr.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_img = bgr[y1:y2, x1:x2]
        else:
            face_img = bgr

        if face_img.size == 0:
            return None

        face_resized = cv2.resize(face_img, (112, 112))
        inp = ((face_resized[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis, ...]
        emb = self.arc_sess.run(None, {'input': inp})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def compute_similarity(emb1, emb2):
        """Cosine similarity between two unit-normalized ArcFace embeddings (-1..1)."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def generate_face_mask(self, img_pil, box, landmarks=None, feather_radius=25):
        """
        Generates a feathered 8-bit grayscale mask for regional loss weighting.
        White (255) = face (100% loss weight). Black (0) = background, scaled
        by `mask_min_value` in the ai-toolkit dataset config (e.g. 0.10).
        """
        w, h = img_pil.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if landmarks is not None and len(landmarks) >= 5:
            center_x = float(landmarks[2][0])  # nose tip
            center_y = float((landmarks[0][1] + landmarks[3][1]) / 2.0)  # eyes/mouth midpoint
            rad_x = bw * 0.52
            rad_y = bh * 0.62
        else:
            center_x = x1 + bw / 2.0
            center_y = y1 + bh / 2.0
            rad_x = bw * 0.50
            rad_y = bh * 0.58

        bbox = [
            int(center_x - rad_x),
            int(center_y - rad_y),
            int(center_x + rad_x),
            int(center_y + rad_y),
        ]
        draw.ellipse(bbox, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        return mask
