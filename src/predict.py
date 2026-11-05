"""Inference wrapper used by the web app, the webcam processor and evaluation (end-to-end latency)."""
import time
import numpy as np
import torch

from .config import *
from .gradcam import GradCAM, overlay
from .hand_detection import HandROI, crop_square, enhance_clahe, letterbox_square
from .model import get_device, load_checkpoint


def available_models():
    return [n for n in EXPERIMENTS if (MODELS / f"{n}.pth").exists()]


class GesturePredictor:
    def __init__(self, name, device=None, streaming=False):
        self.device = device or get_device()
        self.name = name
        self.model, self.ck = load_checkpoint(MODELS / f"{name}.pth", self.device)
        self.arch, self.input, self.enhance = self.ck["arch"], self.ck["input"], self.ck["enhance"]
        self.detector = None
        if self.input == "roi":
            # must be the same detector backend that produced the training crops
            self.detector = HandROI(self.ck.get("roi_backend", "mediapipe"), streaming=streaming)
            if self.detector.backend != self.ck.get("roi_backend", self.detector.backend):
                raise RuntimeError("ROI backend mismatch between training and inference")
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def view(self, rgb):
        """Return (224x224 model input image, box or None, roi_ms)."""
        box, roi_ms = None, 0.0
        if self.input == "roi":
            t = time.perf_counter()
            box = self.detector.detect(rgb)
            roi_ms = (time.perf_counter() - t) * 1000
            img = crop_square(rgb, box) if box else letterbox_square(rgb)
        else:
            img = letterbox_square(rgb)
        if self.enhance:
            img = enhance_clahe(img)
        return img, box, roi_ms

    def tensor(self, img):
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _forward(self, x):
        return torch.softmax(self.model(x), 1)[0].cpu().numpy()

    def predict(self, rgb, gradcam=False):
        """rgb: HxWx3 uint8. Latency covers detection + preprocessing + forward (Grad-CAM excluded)."""
        t0 = time.perf_counter()
        img, box, roi_ms = self.view(rgb)
        x = self.tensor(img)
        self._sync(); t1 = time.perf_counter()
        probs = self._forward(x)
        self._sync(); t2 = time.perf_counter()
        idx = int(probs.argmax())
        out = dict(probs=probs, idx=idx, label=CLASSES[idx], display=DISPLAY[CLASSES[idx]], conf=float(probs[idx]),
                   latency_ms=(t2 - t0) * 1000, model_ms=(t2 - t1) * 1000, roi_ms=roi_ms,
                   box=box, roi_found=(box is not None) if self.input == "roi" else None, input_image=img)
        if gradcam:
            cam_engine = GradCAM(self.model, self.arch)
            try:
                cam, _, _ = cam_engine(self.tensor(img).requires_grad_(False), idx)
            finally:
                cam_engine.remove()
            out["cam"] = cam
            out["heatmap"], out["overlay"] = overlay(img, cam)
        return out
