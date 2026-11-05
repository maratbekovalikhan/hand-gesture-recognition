"""Train + evaluate every experiment (main models and ablations).   python -m src.run_experiments"""
import argparse
from .config import EXPERIMENTS
from .train import train
from . import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), help="subset of experiments")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()
    names = a.only or [n for n in EXPERIMENTS if not (a.skip_ablations and n.startswith("ablation_"))]
    for n in names:
        train(n)
    evaluate.main(names)
