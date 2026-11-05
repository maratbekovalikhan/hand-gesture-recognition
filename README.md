# Hand Gesture Recognition (Peace · Thumbs Up · Stop)
Astana IT University – Computer Vision – Assignment 2. Python + PyTorch + Streamlit.

> **Status of this repository:** the code was written and the data-preparation stage was run, but the models have **not yet been trained
> or evaluated**, and the web app has **not yet been run in a browser** (the authoring sandbox had no PyTorch/Streamlit/internet).
> Everything below is what *you* run on the MacBook. No metrics are shipped – `results/` is empty until you run the experiments.

## 0. ⚠️ Dataset check (important)
In the files that were provided, `stop_03.mp4` is **byte-identical to `peace_03.mp4`** (same MD5, shows the Peace gesture).
`src/prepare_data.py` refuses to run until this is fixed. **Replace `stop_03.mp4` with the real Stop test video.**
Also note: all 6 training videos are on a light wall, all 3 test videos on a wooden panel → the test set measures robustness to a
background/lighting change (see report limitations).

## 1. Setup (macOS, Python 3.11 recommended)
```bash
cd hand-gesture-recognition
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Internet is needed once (pip, and torchvision downloads the ImageNet MobileNetV2 weights on the first training run).

## 2. Data
Put the videos here (exact names):
```
dataset/train/{peace,thumb,stop}_01.mp4  dataset/train/{peace,thumb,stop}_02.mp4
dataset/test/{peace,thumb,stop}_03.mp4
```
```bash
python -m src.prepare_data          # uses MediaPipe Hands (default); prints frames per class/subset
```
Output: `dataset/processed/{train,test}/{full,roi}/<class>/*.jpg`, `manifest.csv`, `meta.json`, `results/dataset_stats.json`.
Read the printed *median hand-box area*: it should be well below 0.5. If the script warns that boxes are not tight, MediaPipe was not used
and the "improved" model is not a valid hand-localisation experiment.
Validation = last 20 % (in time) of each training video, separated by a 10-frame gap. Test videos are used only in `evaluate.py`.

## 3. Train
Quick code check (1 epoch, 2 batches): `python -m src.train --name improved_model --smoke`
```bash
python -m src.train --name cnn_baseline
python -m src.train --name mobilenetv2
python -m src.train --name improved_model            # hand-ROI + MobileNetV2
# ablations
python -m src.train --name ablation_improved_noaug
python -m src.train --name ablation_improved_clahe
python -m src.train --name ablation_cnn_roi
```
Weights: `models/<name>.pth` (best epoch by validation loss). Curves data: `results/history_<name>.json`.
Uses Apple GPU (MPS) automatically if available.

## 4. Evaluate
```bash
python -m src.evaluate        # test-video metrics, confusion matrices, curves, latency -> results/
# or everything in one go (train + evaluate, ~ all experiments):
python -m src.run_experiments
```
Produces `results/metrics.json`, `confusion_matrix.png`, `training_curves.png`, `comparison.png`, `cm_*.png`, `curves_*.png`.

## 5. Run the website
```bash
streamlit run app.py        # http://localhost:8501
```
Pages: Home · Image Recognition · Live Webcam · Model Performance · Explainability · About.

### Webcam
* Press **Enable Camera** → the browser shows its own permission dialog → Allow. **Stop Camera** ends the stream.
* Works on `http://localhost` or HTTPS only. A deployed app must be served over HTTPS.
* macOS: System Settings → Privacy & Security → Camera → allow your browser. Close other apps using the camera (Zoom, etc.).
* If "Connecting…" never finishes: check internet (default Google STUN), disable VPN, try Chrome.
* If the camera is denied/unavailable use the **Image Recognition** page.
* Frames are processed in memory; nothing is recorded or saved.

## 6. Layout
`src/config.py` (all settings/experiments) · `prepare_data.py` · `dataset.py` · `model.py` · `train.py` · `evaluate.py` ·
`predict.py` · `gradcam.py` · `hand_detection.py` · `webcam.py` · `app.py`

## 7. What is and is not implemented
* Implemented: custom CNN, MobileNetV2 transfer learning, Hand-ROI (MediaPipe) + MobileNetV2, Grad-CAM, ablations (augmentation, CLAHE, ROI with CNN).
* **Not implemented:** Step-Video-T2V and ArtAug (suggested in the assignment) – too heavy for this environment; no synthetic media are used.
  CLAHE is a classical contrast method, not ArtAug. MediaPipe is used as an ROI stage and is not claimed to be SOTA by itself.
* No Assignment 1 results are included; any baseline numbers are new experiments from this code.
