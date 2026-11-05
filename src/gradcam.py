"""Grad-CAM (Selvaraju et al., 2017) implemented with forward hooks + tensor gradient hooks."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import get_target_layer


class GradCAM:
    def __init__(self, model, arch):
        self.model, self.act, self.grad = model, None, None
        self.h = get_target_layer(model, arch).register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, class_idx=None):
        self.model.eval(); self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()
        w = self.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        return cam, probs, idx

    def remove(self):
        self.h.remove()


def overlay(rgb224, cam, alpha=0.45):
    """Return (heatmap_rgb, overlay_rgb) uint8."""
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return heat, np.uint8((1 - alpha) * rgb224 + alpha * heat)
