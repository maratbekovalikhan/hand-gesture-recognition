"""Central configuration: paths, classes, preprocessing and experiment definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
PROCESSED = DATASET / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

CLASSES = ["peace", "thumb", "stop"]  # label index = position in this list
DISPLAY = {"peace": "Peace", "thumb": "Thumbs Up", "stop": "Stop"}

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEED = 42

# --- frame extraction -------------------------------------------------------
FRAME_STEP = 6        # keep every 6th frame (30 fps -> 5 fps candidates)
DEDUP_THRESH = 2.0    # skip frame if mean abs diff of 32x32 gray < threshold
VAL_FRACTION = 0.20   # last 20% (in time) of every TRAIN video -> validation
VAL_GAP = 10          # kept frames dropped between train and val block (leakage guard)
ROI_MARGIN = 0.20     # margin added on each side of the hand box

# --- experiments ------------------------------------------------------------
# input: "full" = whole frame (letterboxed to square), "roi" = hand crop
EXPERIMENTS = {
    # Model 1: baseline
    "cnn_baseline":  dict(arch="cnn",       input="full", aug=True,  enhance=False, epochs=25, lr=1e-3),
    # Model 2: transfer learning
    "mobilenetv2":   dict(arch="mobilenet", input="full", aug=True,  enhance=False, epochs=12, lr=1e-3),
    # Model 3: improved pipeline = hand detection/ROI crop + MobileNetV2
    "improved_model": dict(arch="mobilenet", input="roi", aug=True,  enhance=False, epochs=12, lr=1e-3),
    # Ablations (all compared against improved_model)
    "ablation_improved_noaug": dict(arch="mobilenet", input="roi", aug=False, enhance=False, epochs=12, lr=1e-3),
    "ablation_improved_clahe": dict(arch="mobilenet", input="roi", aug=True,  enhance=True,  epochs=12, lr=1e-3),
    "ablation_cnn_roi":        dict(arch="cnn",       input="roi", aug=True,  enhance=False, epochs=25, lr=1e-3),
}
MAIN_MODELS = ["cnn_baseline", "mobilenetv2", "improved_model"]
PRETTY = {
    "cnn_baseline": "Custom CNN (baseline)",
    "mobilenetv2": "MobileNetV2 (full frame)",
    "improved_model": "Improved: Hand-ROI + MobileNetV2",
    "ablation_improved_noaug": "Ablation: Improved, no augmentation",
    "ablation_improved_clahe": "Ablation: Improved + CLAHE",
    "ablation_cnn_roi": "Ablation: CNN on Hand-ROI",
}
