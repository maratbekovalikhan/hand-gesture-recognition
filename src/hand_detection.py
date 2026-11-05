"""Hand localisation + image views shared by data preparation, training, inference and the app.

Backends
--------
* ``mediapipe`` : MediaPipe Hands landmarks -> tight hand bounding box (preferred).
* ``skin``      : classical YCrCb skin-colour segmentation fallback (OpenCV only).
                  Works on plain backgrounds; less robust than MediaPipe.
The backend actually used is recorded in dataset/processed/meta.json and in every checkpoint.
"""
import cv2
import numpy as np

from .config import IMG_SIZE, ROI_MARGIN


def letterbox_square(rgb, size=IMG_SIZE):
    """Pad (edge replicate) to a square without distortion, then resize."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(rgb, top, s - h - top, left, s - w - left, cv2.BORDER_REPLICATE)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def crop_square(rgb, box, margin=ROI_MARGIN, size=IMG_SIZE):
    """Square crop centred on ``box`` (x0,y0,x1,y1) with a margin; edge-replicated if it leaves the frame."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(int(max(x1 - x0, y1 - y0) * (1 + 2 * margin)), 32)
    p = cv2.copyMakeBorder(rgb, side, side, side, side, cv2.BORDER_REPLICATE)
    xs, ys = int(round(cx - side / 2)) + side, int(round(cy - side / 2)) + side
    return cv2.resize(p[ys:ys + side, xs:xs + side], (size, size), interpolation=cv2.INTER_AREA)


def enhance_clahe(rgb, clip=2.0, grid=8):
    """CLAHE on the L channel (classical low-exposure enhancement; NOT ArtAug)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class HandROI:
    def __init__(self, backend="auto", streaming=False):
        self._mp = None
        self.backend = "skin"
        if backend in ("auto", "mediapipe"):
            try:
                import mediapipe as mp
                self._mp = mp.solutions.hands.Hands(
                    static_image_mode=not streaming, max_num_hands=1, model_complexity=0,
                    min_detection_confidence=0.4, min_tracking_confidence=0.4)
                self.backend = "mediapipe"
            except Exception as e:  # missing package / no mp.solutions in newer builds
                if backend == "mediapipe":
                    raise RuntimeError(f"MediaPipe Hands unavailable: {e!r}")

    def detect(self, rgb):
        """Return (x0, y0, x1, y1) in pixel coords of ``rgb`` or None."""
        return self._detect_mp(rgb) if self._mp is not None else self._detect_skin(rgb)

    def _detect_mp(self, rgb):
        res = self._mp.process(np.ascontiguousarray(rgb))
        if not res.multi_hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        pts = np.array([[l.x * w, l.y * h] for l in res.multi_hand_landmarks[0].landmark])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        return (int(x0), int(y0), int(x1), int(y1))

    @staticmethod
    def _detect_skin(rgb):
        h, w = rgb.shape[:2]
        sc = 320 / max(h, w)
        small = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        ycc = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
        mask = cv2.inRange(ycc, (0, 135, 85), (255, 175, 130))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.02 * mask.size:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        return (int(x / sc), int(y / sc), int((x + bw) / sc), int((y + bh) / sc))
