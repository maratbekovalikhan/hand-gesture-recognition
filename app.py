"""Hand Gesture Recognition - Streamlit web app.   Run:  streamlit run app.py"""
import json, platform, sys, time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.config import *
from src.predict import GesturePredictor, available_models

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
:root{--ink:#1e1b4b;--accent:#4f46e5;--teal:#0d9488;--soft:#eef2ff;}
html,body,[class*="css"]{font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;}
.hero{background:linear-gradient(120deg,#4f46e5,#0d9488);color:#fff;padding:2.2rem 2rem;border-radius:18px;margin-bottom:1.2rem}
.hero h1{margin:0 0 .4rem 0;font-size:2.3rem;color:#fff}.hero p{margin:0;font-size:1.05rem;opacity:.95}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:1.1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card .emoji{font-size:3rem}.card,.card h3{color:#1e1b4b !important}.card h3{margin:.3rem 0}
.step{background:var(--soft);border-radius:14px;padding:.9rem 1rem;height:100%}
.step,.step *{color:#1f2937 !important}
.step b{color:#4f46e5 !important}
.pred{background:var(--soft);border-left:6px solid var(--accent);padding:1rem 1.2rem;border-radius:12px;margin:.5rem 0}
.pred,.pred h2{color:#1e1b4b !important}.pred h2{margin:0}.pred .small{color:#6b7280 !important}
.small{color:#6b7280;font-size:.85rem}
</style>""", unsafe_allow_html=True)

EMOJI = {"peace": "✌️", "thumb": "👍", "stop": "✋"}
PAGES = ["Home", "Image Recognition", "Live Webcam", "Model Performance", "Explainability", "About Project"]
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("lat", [])
st.session_state.setdefault("cam_on", False)


def go(p):
    st.session_state.page = p


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(name, streaming=False):
    return GesturePredictor(name, streaming=streaming)


def load_metrics():
    f = RESULTS / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def env_info():
    ua = "n/a"
    try:
        ua = st.context.headers.get("User-Agent", "n/a")
    except Exception:
        pass
    import torch
    from src.model import get_device
    return {"Server OS": platform.platform(), "Python": sys.version.split()[0], "PyTorch": torch.__version__,
            "Inference device": str(get_device()), "Browser (User-Agent)": ua}


def sample_images(model_input="full", n=1, subset="test"):
    """Return {class: [Path, ...]} of existing processed frames.
    NOTE: the 'val' subset is stored under dataset/processed/train/ (it is cut from the train videos)."""
    out = {}
    folder = "test" if subset == "test" else "train"
    try:
        df = pd.read_csv(PROCESSED / "manifest.csv")
        for c in CLASSES:
            d = df[(df.subset == subset) & (df.cls == c)]
            paths = [PROCESSED / folder / model_input / s for s in d.sample(min(n, len(d)), random_state=3)["stem"]]
            out[c] = [p for p in paths if p.exists()]
    except Exception:
        pass
    return out


def read_img(path):
    """Read an image into an RGB array (avoids Streamlit file-path serving issues)."""
    return np.array(Image.open(path).convert("RGB"))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ✋ Gesture Recognition")
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    models = available_models()
    default = models.index("improved_model") if "improved_model" in models else 0
    model_name = st.selectbox("Model", models, index=default if models else None,
                              format_func=lambda n: PRETTY[n]) if models else None
    if not models:
        st.warning("No trained models in `models/`. Train first (see README).")
    st.caption("Astana IT University · Computer Vision · Assignment 2")

page = st.session_state.page

# ----------------------------------------------------------------------------- Home
if page == "Home":
    st.markdown('<div class="hero"><h1>Hand Gesture Recognition</h1>'
                '<p>Recognise <b>Peace</b>, <b>Thumbs Up</b> and <b>Stop</b> from an uploaded image or your live webcam, '
                'with Grad-CAM explanations and measured latency.</p></div>', unsafe_allow_html=True)
    imgs = sample_images("full", 1, "train")
    cols = st.columns(3)
    for col, c in zip(cols, CLASSES):
        with col:
            st.markdown(f'<div class="card"><div class="emoji">{EMOJI[c]}</div><h3>{DISPLAY[c]}</h3></div>', unsafe_allow_html=True)
            if imgs.get(c):
                st.image(read_img(imgs[c][0]), use_container_width=True, caption="example training frame")
    st.markdown("### Pipeline")
    steps = [("1 · Input", "Uploaded image or browser webcam frame (processed in memory only)."),
             ("2 · Hand localisation", "MediaPipe Hands landmarks → hand box → square crop (improved model)."),
             ("3 · Classifier", "Custom CNN, or MobileNetV2 fine-tuned from ImageNet weights."),
             ("4 · Explanation", "Grad-CAM heatmap over the last convolutional feature map.")]
    for col, (t, d) in zip(st.columns(4), steps):
        col.markdown(f'<div class="step"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)
    st.markdown("### Quick start")
    a, b = st.columns(2)
    a.button("📷 Upload an image", on_click=go, args=("Image Recognition",), use_container_width=True, type="primary")
    b.button("🎥 Live webcam", on_click=go, args=("Live Webcam",), use_container_width=True)

# ----------------------------------------------------------------------------- Image
elif page == "Image Recognition":
    st.title("Image Recognition")
    if not model_name:
        st.stop()
    up = st.file_uploader("Upload a JPG / JPEG / PNG image of a hand gesture", type=["jpg", "jpeg", "png"])
    st.caption("🔒 The image is processed in memory and is not saved.")
    if up:
        pil = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        rgb = np.array(pil)
        pred = get_predictor(model_name)
        with st.spinner("Running model…"):
            r = pred.predict(rgb, gradcam=True)
        st.session_state.lat.append(r["latency_ms"])
        left, right = st.columns([1, 1])
        left.image(pil, caption="Uploaded image", use_container_width=True)
        with right:
            st.markdown(f'<div class="pred"><span class="small">Predicted gesture</span>'
                        f'<h2>{EMOJI[r["label"]]} {r["display"]}</h2></div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f'{r["conf"]*100:.1f}%')
            m2.metric("Inference time", f'{r["latency_ms"]:.0f} ms')
            m3.metric("Avg this session", f'{np.mean(st.session_state.lat):.0f} ms')
            for c, p in zip(CLASSES, r["probs"]):
                st.write(f"**{DISPLAY[c]}**: {p*100:.1f}%")
                st.progress(float(p))
            if r["roi_found"] is False:
                st.warning("No hand detected – the whole image was used instead of a hand crop.")
            st.caption(f'Detection {r["roi_ms"]:.0f} ms · network {r["model_ms"]:.0f} ms (Grad-CAM time not included)')
        st.subheader("Grad-CAM explanation")
        c1, c2, c3 = st.columns(3)
        c1.image(r["input_image"], caption="Model input (224×224)", use_container_width=True)
        c2.image(r["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
        c3.image(r["overlay"], caption="Overlay", use_container_width=True)
        with st.expander("Environment"):
            st.json(env_info())

# ----------------------------------------------------------------------------- Live
elif page == "Live Webcam":
    st.title("Live Webcam Recognition")
    st.info("This page needs access to your camera. Nothing happens until you press **Enable Camera**; your browser will then "
            "ask for permission. Frames are analysed in memory and are never recorded or stored.")
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from src.webcam import GestureProcessor
        ok = True
    except Exception as e:
        ok = False
        st.error(f"streamlit-webrtc is not available: {e}")
    if not model_name:
        st.stop()
    if ok:
        b1, b2, _ = st.columns([1, 1, 3])
        b1.button("📷 Enable Camera", type="primary", on_click=lambda: st.session_state.update(cam_on=True),
                  use_container_width=True)
        b2.button("⏹ Stop Camera", on_click=lambda: st.session_state.update(cam_on=False), use_container_width=True)
        if st.session_state.cam_on:
            pred = get_predictor(model_name, True)
            vcol, scol = st.columns([3, 2])
            with vcol:
                ctx = webrtc_streamer(
                    key=f"gesture-{model_name}", mode=WebRtcMode.SENDRECV, desired_playing_state=True,
                    video_processor_factory=lambda: GestureProcessor(pred),
                    media_stream_constraints={"video": {"width": {"ideal": 640}}, "audio": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    async_processing=True)
            with scol:
                box = st.empty()
                if ctx.state.playing:
                    while ctx.state.playing:
                        proc = ctx.video_processor
                        if proc:
                            s = proc.get_stats()
                            with box.container():
                                st.markdown(f'<div class="pred"><span class="small">Current gesture</span>'
                                            f'<h2>{s["label"]}</h2></div>', unsafe_allow_html=True)
                                a, b, c = st.columns(3)
                                a.metric("Confidence", f'{s["conf"]*100:.0f}%')
                                b.metric("Latency", f'{s["latency_ms"]:.0f} ms')
                                c.metric("Video FPS", f'{s["fps"]:.0f}')
                                st.caption(f'Inference rate: {s["inf_fps"]:.1f}/s')
                                for cl, p in zip(CLASSES, s["probs"]):
                                    st.write(f"{DISPLAY[cl]}: {p*100:.0f}%"); st.progress(min(max(p, 0.0), 1.0))
                        time.sleep(0.25)
                else:
                    st.caption("Waiting for the camera… If the browser blocked it or no camera is available, "
                               "allow camera access in the address bar or use **Image Recognition** instead.")
            st.button("Use image upload instead", on_click=go, args=("Image Recognition",))
        with st.expander("Browser / HTTPS requirements"):
            st.markdown("Browsers only expose the camera on **HTTPS** or on **localhost**. `streamlit run app.py` on your own "
                        "machine (http://localhost:8501) is fine; a deployed server must be served over HTTPS. "
                        "The default STUN server needs internet access.")

# ----------------------------------------------------------------------------- Performance
elif page == "Model Performance":
    st.title("Model Performance")
    M = load_metrics()
    if not M:
        st.warning("No results yet. Run `python -m src.run_experiments` (see README).")
        st.stop()
    names = [n for n in EXPERIMENTS if n in M]
    st.caption(f'Test set: {M[names[0]]["n_test"]} frames from the 3 held-out test videos. All numbers come from `src/evaluate.py`.')
    rows = [{"Model": PRETTY[n], "Accuracy": M[n]["accuracy"], "Precision (macro)": M[n]["macro_precision"],
             "Recall (macro)": M[n]["macro_recall"], "F1 (macro)": M[n]["macro_f1"],
             "Latency (ms, end-to-end)": round(M[n]["latency_ms_total_mean"], 1), "Params": M[n]["params"]} for n in names]
    df = pd.DataFrame(rows)
    main_df = df[~df.Model.str.startswith("Ablation")]
    st.subheader("Comparison")
    st.dataframe(main_df.style.format({"Accuracy": "{:.3f}", "Precision (macro)": "{:.3f}", "Recall (macro)": "{:.3f}",
                                       "F1 (macro)": "{:.3f}", "Params": "{:,}"}), hide_index=True, use_container_width=True)
    if len(main_df):
        b = main_df.loc[main_df["F1 (macro)"].idxmax()]
        st.success(f'Highest macro-F1 on the test videos: **{b.Model}** ({b["F1 (macro)"]:.3f}), '
                   f'accuracy {b.Accuracy:.3f}, latency {b["Latency (ms, end-to-end)"]} ms.')
    st.caption("⚠️ Test set = 3 videos of one scene/person; differences between models may reflect this narrow data rather than general ability.")
    abl = df[df.Model.str.startswith("Ablation")]
    if len(abl):
        st.subheader("Ablation")
        st.dataframe(pd.concat([df[df.Model.str.startswith("Improved")], abl]).style.format(
            {"Accuracy": "{:.3f}", "Precision (macro)": "{:.3f}", "Recall (macro)": "{:.3f}", "F1 (macro)": "{:.3f}", "Params": "{:,}"}),
            hide_index=True, use_container_width=True)
    if (RESULTS / "comparison.png").exists():
        st.image(str(RESULTS / "comparison.png"), use_container_width=True)
    tabs = st.tabs([PRETTY[n] for n in names])
    for t, n in zip(tabs, names):
        with t:
            a, b = st.columns(2)
            if (RESULTS / f"cm_{n}.png").exists():
                a.image(str(RESULTS / f"cm_{n}.png"), caption="Confusion matrix (test)")
            if (RESULTS / f"curves_{n}.png").exists():
                b.image(str(RESULTS / f"curves_{n}.png"), caption="Training / validation curves")
            pc = pd.DataFrame(M[n]["per_class"]).T.rename_axis("class").reset_index()
            st.dataframe(pc, hide_index=True, use_container_width=True)
            st.caption(f'Best epoch (by validation loss): {M[n]["best_epoch"]} · device: {M[n]["device"]} · '
                       f'latency: {M[n]["latency_ms_total_mean"]:.1f} ± {M[n]["latency_ms_total_std"]:.1f} ms '
                       f'(n={M[n]["latency_n"]}, {M[n]["latency_input"]})')
    st.subheader("Environment of the measurements")
    st.json(M.get("_meta", {}) | {"app_environment": env_info()})

# ----------------------------------------------------------------------------- Explainability
elif page == "Explainability":
    st.title("Explainability – Grad-CAM")
    st.markdown("Grad-CAM weights the last convolutional feature maps by the gradient of the predicted class score and shows "
                "**where the network looked**. Red = more influence on the score.")
    if model_name:
        split = st.radio("Examples from", ["test", "val"], horizontal=True)
        pred = get_predictor(model_name)
        imgs = sample_images("full", 2, split)
        if not imgs:
            st.warning("Processed dataset not found – run `python -m src.prepare_data`.")
        for c in CLASSES:
            for pth in imgs.get(c, []):
                rgb = read_img(pth)
                # processed images are already 224x224 'full' frames; the predictor re-applies its own view
                r = pred.predict(rgb, gradcam=True)
                k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
                k1.image(rgb, caption=f"True: {DISPLAY[c]}", use_container_width=True)
                k2.image(r["heatmap"], caption="Heatmap", use_container_width=True)
                k3.image(r["overlay"], caption="Overlay", use_container_width=True)
                k4.markdown(f'**Predicted:** {r["display"]}  \n**Confidence:** {r["conf"]*100:.1f}%  \n'
                            f'{"✅ correct" if r["label"] == c else "❌ wrong"}')
    st.subheader("How to read it – and its limits")
    st.markdown("- The heatmap is a *coarse* (7×7 for MobileNetV2, 14×14 for the CNN, upsampled) indication of influence, not a segmentation.\n"
                "- It explains the model's evidence, not whether the model is *right*; a model can look at the background and still be correct on this dataset.\n"
                "- Heatmaps are class-specific and can change with the chosen layer; they are qualitative, not a proof of reasoning.")

# ----------------------------------------------------------------------------- About
else:
    st.title("About the Project")
    meta = {}
    try:
        meta = json.loads((PROCESSED / "meta.json").read_text())
    except Exception:
        pass
    st.markdown("""
**Objective.** Continue Assignment 1 (hand gesture recognition: Peace, Thumbs Up, Stop) by adding a modern CV component,
comparing against the CNN/MobileNetV2 baselines and shipping an interactive demo with explanations.

**Dataset.** 9 videos (~1 min each). Per class: videos `_01` and `_02` for training/validation, video `_03` for testing only.
Frames are sampled every 6th frame and near-duplicates are dropped. In each training video the last 20 % (in time) is used as validation,
with a gap to the training part. Test frames are never used for training, validation, tuning or augmentation.
Consecutive frames are highly correlated – the dataset effectively has **6 training recordings**, not thousands of independent samples.""")
    if meta:
        st.json(meta.get("counts", {}))
    st.markdown("""
**Models.** (1) Custom 4-block CNN (baseline). (2) MobileNetV2 with ImageNet weights, new 3-class head, full fine-tuning.
(3) *Improved pipeline*: MediaPipe Hands localises the hand → square crop → MobileNetV2.

**Honest note on the "SOTA" component.** The assignment suggests Step-Video-T2V and ArtAug. Neither was implemented (they need large
generative models/GPU). The improvement used here is hand localisation with MediaPipe Hands as a *pre-processing/ROI stage*; it is a
well-established, efficient component, not claimed to be state of the art by itself. CLAHE (classical contrast enhancement) is used in one
ablation and is **not** ArtAug.

**Technologies.** Python, PyTorch, torchvision, OpenCV, NumPy, scikit-learn, Matplotlib, MediaPipe, Streamlit, streamlit-webrtc.

**Privacy.** Uploaded images and webcam frames are processed in memory only; nothing is saved. The camera starts only after you press
*Enable Camera* and your browser grants permission; it stops when you press *Stop Camera*.

**Limitations & ethics.** One person, two backgrounds, one camera orientation; no skin-tone/age/hand-size diversity – the model may perform worse
on people unlike the data subject. Only three gestures; unknown gestures are still forced into one of the three classes.
Do not use for security or surveillance decisions. No synthetic media were used.""")
