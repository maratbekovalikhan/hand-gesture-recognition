"""PyTorch dataset over the processed frames. Only 'train' frames are ever augmented."""
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import *
from .hand_detection import enhance_clahe


def load_manifest():
    f = PROCESSED / "manifest.csv"
    if not f.exists():
        raise SystemExit("dataset/processed/manifest.csv not found - run: python -m src.prepare_data")
    return pd.read_csv(f)


def _enh(im):
    return Image.fromarray(enhance_clahe(np.array(im)))


def build_transform(train, aug, enhance):
    ops = [T.Lambda(_enh)] if enhance else []
    if train and aug:
        ops += [T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(),                      # left/right hand
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.05),  # exposure / colour shift
                T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3)]
    else:
        ops += [T.Resize((IMG_SIZE, IMG_SIZE))]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class GestureDataset(Dataset):
    def __init__(self, df, kind, transform):
        self.df, self.kind, self.tf = df.reset_index(drop=True), kind, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(PROCESSED / r.split / self.kind / r.stem).convert("RGB")
        return self.tf(img), int(r.label)


def get_split(df, subset, kind, train=False, aug=False, enhance=False):
    return GestureDataset(df[df.subset == subset], kind, build_transform(train, aug, enhance))
