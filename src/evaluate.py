"""Evaluate trained models on the held-out TEST videos and write results/*.

    python -m src.evaluate            # all models that have weights
    python -m src.evaluate --names improved_model mobilenetv2
Everything in results/metrics.json is computed here from real predictions; nothing is hand-written.
"""
import argparse, json, platform, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .config import *
from .dataset import get_split, load_manifest
from .model import count_params, get_device
from .predict import GesturePredictor

LABELS = [DISPLAY[c] for c in CLASSES]


@torch.no_grad()
def predict_test(pred, df):
    ds = get_split(df, "test", pred.input, enhance=pred.enhance)
    ys, ps = [], []
    for x, y in DataLoader(ds, 64, shuffle=False, num_workers=0):
        ps += pred.model(x.to(pred.device)).argmax(1).cpu().tolist(); ys += y.tolist()
    return np.array(ys), np.array(ps)


def measure_latency(pred, n=60, warmup=5):
    """End-to-end latency (detector + preprocessing + forward) on RAW test-video frames, batch size 1."""
    vids = sorted((DATASET / "test").glob("*.mp4"))
    tot, mdl, roi = [], [], []
    for vp in vids:
        cap = cv2.VideoCapture(str(vp)); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(30, N - 30, n // max(len(vids), 1) + warmup).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, bgr = cap.read()
            if not ok:
                continue
            r = pred.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            tot.append(r["latency_ms"]); mdl.append(r["model_ms"]); roi.append(r["roi_ms"])
    k = warmup  # discard first frames (warm-up)
    return dict(latency_ms_total_mean=float(np.mean(tot[k:])), latency_ms_total_std=float(np.std(tot[k:])),
                latency_ms_model_mean=float(np.mean(mdl[k:])), latency_ms_roi_mean=float(np.mean(roi[k:])),
                latency_n=len(tot) - k, latency_input="raw 720x1280 video frames")


def plot_cm(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=20); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")


def plot_curves(names):
    for n in names:
        f = RESULTS / f"history_{n}.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text()); e = range(1, len(h["train_loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(e, h["train_loss"], label="train"); ax[0].plot(e, h["val_loss"], label="val"); ax[0].set_title("Loss")
        ax[1].plot(e, h["train_acc"], label="train"); ax[1].plot(e, h["val_acc"], label="val"); ax[1].set_title("Accuracy")
        for a in ax:
            a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
        fig.suptitle(PRETTY.get(n, n)); fig.tight_layout(); fig.savefig(RESULTS / f"curves_{n}.png", dpi=130); plt.close(fig)
    main = [n for n in MAIN_MODELS if (RESULTS / f"history_{n}.json").exists()]
    if main:
        fig, ax = plt.subplots(2, len(main), figsize=(4.2 * len(main), 6), squeeze=False)
        for c, n in enumerate(main):
            h = json.loads((RESULTS / f"history_{n}.json").read_text()); e = range(1, len(h["train_loss"]) + 1)
            for r, k in enumerate(["loss", "acc"]):
                ax[r, c].plot(e, h[f"train_{k}"], label="train"); ax[r, c].plot(e, h[f"val_{k}"], label="val")
                ax[r, c].set_title(f"{PRETTY[n]}\n{k}", fontsize=8); ax[r, c].grid(alpha=.3); ax[r, c].legend()
        fig.tight_layout(); fig.savefig(RESULTS / "training_curves.png", dpi=130); plt.close(fig)


def main(names):
    RESULTS.mkdir(exist_ok=True)
    df = load_manifest()
    device = get_device()
    metrics = {}
    old = RESULTS / "metrics.json"
    if old.exists():
        metrics = json.loads(old.read_text())
    for n in names:
        pred = GesturePredictor(n, device)
        y, p = predict_test(pred, df)
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        m = dict(name=n, pretty=PRETTY[n], config=EXPERIMENTS[n], n_test=int(len(y)), accuracy=float(accuracy_score(y, p)),
                 macro_precision=float(pr.mean()), macro_recall=float(rc.mean()), macro_f1=float(f1.mean()),
                 per_class={c: dict(precision=float(pr[i]), recall=float(rc[i]), f1=float(f1[i]), support=int(sup[i]))
                            for i, c in enumerate(CLASSES)},
                 confusion_matrix=cm.tolist(), params=count_params(pred.model), best_epoch=pred.ck["epoch"],
                 device=str(device), roi_backend=pred.ck.get("roi_backend"))
        t = df[df.subset == "test"]
        if pred.input == "roi":
            m["test_hand_not_found"] = int((t.roi_found == 0).sum())
        m.update(measure_latency(pred))
        metrics[n] = m
        fig, ax = plt.subplots(figsize=(3.6, 3.4)); plot_cm(ax, cm, PRETTY[n]); fig.tight_layout()
        fig.savefig(RESULTS / f"cm_{n}.png", dpi=130); plt.close(fig)
        print(f"{n:26s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} latency={m['latency_ms_total_mean']:.1f} ms")
    metrics["_meta"] = dict(python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
                            machine=platform.machine(), device=str(device), dataset=json.loads((PROCESSED / "meta.json").read_text()))
    old.write_text(json.dumps(metrics, indent=2))
    # combined figures
    done = [n for n in EXPERIMENTS if n in metrics]
    main_done = [n for n in MAIN_MODELS if n in metrics]
    if main_done:
        fig, ax = plt.subplots(1, len(main_done), figsize=(3.6 * len(main_done), 3.5), squeeze=False)
        for a, n in zip(ax[0], main_done):
            plot_cm(a, np.array(metrics[n]["confusion_matrix"]), PRETTY[n])
        fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrix.png", dpi=130); plt.close(fig)
    if done:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(done)), 4)); w = 0.2
        for k, (key, lab) in enumerate([("accuracy", "Accuracy"), ("macro_precision", "Macro P"),
                                        ("macro_recall", "Macro R"), ("macro_f1", "Macro F1")]):
            ax.bar(np.arange(len(done)) + k * w, [metrics[n][key] for n in done], w, label=lab)
        ax.set_xticks(np.arange(len(done)) + 1.5 * w, [n.replace("ablation_", "abl_") for n in done], rotation=25, ha="right")
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
        fig.savefig(RESULTS / "comparison.png", dpi=130); plt.close(fig)
    plot_curves(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    a = ap.parse_args()
    from .predict import available_models
    main(a.names or available_models())
