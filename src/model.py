"""Model definitions: custom CNN baseline and MobileNetV2 transfer-learning model."""
import torch
import torch.nn as nn
from torchvision import models

from .config import CLASSES


def _block(i, o):
    # ReLU is NOT in-place so Grad-CAM hooks stay valid
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2))


class SimpleCNN(nn.Module):
    """4 conv blocks (32-64-128-128) + global average pooling + dropout + linear."""
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch, num_classes=len(CLASSES), pretrained=True):
    if arch == "cnn":
        return SimpleCNN(num_classes)
    if arch == "mobilenet":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)  # downloads ImageNet weights on first use (needs internet once)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
        return m
    raise ValueError(arch)


def get_target_layer(model, arch):
    """Last convolutional feature map used by Grad-CAM."""
    if arch == "cnn":
        return model.features[-1][-2]      # ReLU after the last conv, before max-pool (14x14)
    return model.features[-1]              # MobileNetV2 final 1x1 conv block (7x7x1280)


def param_groups(model, arch, lr):
    if arch == "mobilenet":
        return [dict(params=model.features.parameters(), lr=lr * 0.1),
                dict(params=model.classifier.parameters(), lr=lr)]
    return [dict(params=model.parameters(), lr=lr)]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
