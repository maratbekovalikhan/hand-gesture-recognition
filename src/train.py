"""Train one experiment defined in config.EXPERIMENTS.

    python -m src.train --name cnn_baseline
Model selection uses ONLY the validation split (last 20% of the two train videos). Test videos are never touched.
"""
import argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import *
from .dataset import get_split, load_manifest
from .model import build_model, count_params, get_device, param_groups


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, device, crit, opt=None, max_batches=None):
    train = opt is not None
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, correct / n


def train(name, epochs=None, batch_size=32, pretrained=True, max_batches=None, patience=5):
    cfg = EXPERIMENTS[name]
    epochs = epochs or cfg["epochs"]
    seed_all()
    device = get_device()
    df = load_manifest()
    meta = json.loads((PROCESSED / "meta.json").read_text())
    tr = get_split(df, "train", cfg["input"], train=True, aug=cfg["aug"], enhance=cfg["enhance"])
    va = get_split(df, "val", cfg["input"], enhance=cfg["enhance"])
    g = torch.Generator().manual_seed(SEED)
    tl = DataLoader(tr, batch_size, shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(va, batch_size, shuffle=False, num_workers=0)
    model = build_model(cfg["arch"], pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["arch"], cfg["lr"]), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    hist = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best, best_ep, bad = 1e9, -1, 0
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    print(f"[{name}] device={device} train={len(tr)} val={len(va)} params={count_params(model):,} cfg={cfg}")
    t0 = time.time()
    for ep in range(epochs):
        trl, tra = run_epoch(model, tl, device, crit, opt, max_batches)
        val, vaa = run_epoch(model, vl, device, crit, None, max_batches)
        sched.step()
        for k, v in zip(hist, (trl, tra, val, vaa)):
            hist[k].append(round(v, 5))
        print(f"  ep {ep+1:02d}/{epochs} train {trl:.4f}/{tra:.3f}  val {val:.4f}/{vaa:.3f}")
        if val < best:
            best, best_ep, bad = val, ep + 1, 0
            torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, arch=cfg["arch"],
                            input=cfg["input"], enhance=cfg["enhance"], classes=CLASSES, img_size=IMG_SIZE,
                            epoch=ep + 1, roi_backend=meta["roi_backend"], name=name), MODELS / f"{name}.pth")
        else:
            bad += 1
            if bad >= patience:
                print("  early stopping"); break
    hist.update(best_epoch=best_ep, train_seconds=round(time.time() - t0, 1), device=str(device), config=cfg,
                n_train=len(tr), n_val=len(va), params=count_params(model))
    (RESULTS / f"history_{name}.json").write_text(json.dumps(hist, indent=2))
    print(f"[{name}] best epoch {best_ep}, weights -> models/{name}.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true", help="offline smoke test only")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches: checks the code runs")
    a = ap.parse_args()
    train(a.name, 1 if a.smoke else a.epochs, a.batch_size, not a.no_pretrained, 2 if a.smoke else None)
