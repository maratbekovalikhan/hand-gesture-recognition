"""Extract frames from the 9 videos, build full-frame and hand-ROI views, write manifest.

Usage:  python -m src.prepare_data
Deterministic (no randomness) -> same output on every run for the same videos/backend.
Splits
  train videos (*_01, *_02): first 80% of kept frames -> 'train'; gap; last 20% -> 'val'
  test  videos (*_03)       : 'test'  (never used for training / validation / tuning / augmentation)
"""
import argparse, hashlib, json
import cv2
import pandas as pd

from .config import *
from .hand_detection import HandROI, letterbox_square, crop_square


def extract_video(path, detector, step, thresh):
    cap = cv2.VideoCapture(str(path))
    frames, last, i = [], None, 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32)).astype("float32")
                if last is None or abs(g - last).mean() >= thresh:   # drop near-duplicates
                    last = g
                    box = detector.detect(rgb)
                    full = letterbox_square(rgb)
                    roi = crop_square(rgb, box) if box else full
                    bf = ((box[2]-box[0])*(box[3]-box[1]))/(rgb.shape[0]*rgb.shape[1]) if box else 0.0
                    frames.append((i, full, roi, box is not None, bf))
        i += 1
    cap.release()
    return frames, i


def check_videos():
    """Fail loudly on duplicated videos (e.g. the same file saved under two class names)."""
    seen = {}
    for split in ["train", "test"]:
        for vp in sorted((DATASET / split).glob("*.mp4")):
            h = hashlib.md5(vp.resolve().read_bytes()).hexdigest()
            if h in seen:
                raise SystemExit(f"ERROR: {vp.name} is byte-identical to {seen[h]}. "
                                 "Two different classes/splits cannot share one video - replace the wrong file.")
            seen[h] = vp.name
    expected = {f"{c}_{n:02d}" for c in CLASSES for n in (1, 2, 3)}
    found = {p.stem for s_ in ("train", "test") for p in (DATASET / s_).glob("*.mp4")}
    if expected - found:
        raise SystemExit(f"ERROR: missing videos: {sorted(expected - found)}")


def save(img_rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main(backend="auto"):
    check_videos()
    detector = HandROI(backend)
    print(f"Hand ROI backend: {detector.backend}")
    rows, video_info = [], {}
    for split in ["train", "test"]:
        for vp in sorted((DATASET / split).glob("*.mp4")):
            cls = vp.stem.split("_")[0]
            assert cls in CLASSES, f"Cannot derive class from filename {vp.name}"
            frames, total = extract_video(vp, detector, FRAME_STEP, DEDUP_THRESH)
            n = len(frames)
            if split == "train":
                cut = n - int(round(n * VAL_FRACTION))
                subset_of = lambda k: "train" if k < cut - VAL_GAP else ("gap" if k < cut else "val")
            else:
                subset_of = lambda k: "test"
            kept = 0
            for k, (fi, full, roi, found, bf) in enumerate(frames):
                sub = subset_of(k)
                if sub == "gap":
                    continue
                stem = f"{cls}/{vp.stem}_{fi:05d}.jpg"
                save(full, PROCESSED / split / "full" / stem)
                save(roi, PROCESSED / split / "roi" / stem)
                rows.append(dict(split=split, subset=sub, cls=cls, label=CLASSES.index(cls),
                                 video=vp.stem, frame=fi, stem=stem, roi_found=int(found), box_frac=round(bf, 4)))
                kept += 1
            video_info[vp.stem] = dict(total_frames=total, candidates=n, saved=kept)
            print(f"{vp.name:14s} total={total:5d} kept(after dedup)={n:4d} saved={kept:4d}")
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "manifest.csv", index=False)
    counts = df.groupby(["subset", "cls"]).size().unstack(fill_value=0)
    print("\nFrames per subset/class:\n", counts)
    roi_rate = df.groupby("subset").roi_found.mean().round(4).to_dict()
    print("Hand detection success rate:", roi_rate)
    med = float(df[df.roi_found == 1].box_frac.median())
    print(f"Median hand-box area / frame area: {med:.2f}")
    if detector.backend != "mediapipe" or med > 0.5:
        print("WARNING: hand boxes are not tight (backend=%s, median box fraction=%.2f). The 'roi' view is then "
              "almost the full frame and the 'improved' pipeline is NOT a valid test of hand localisation. "
              "Install mediapipe==0.10.14 and re-run." % (detector.backend, med))
    meta = dict(roi_backend=detector.backend, median_box_frac=round(med, 4), frame_step=FRAME_STEP, dedup_thresh=DEDUP_THRESH,
                val_fraction=VAL_FRACTION, val_gap=VAL_GAP, img_size=IMG_SIZE, roi_margin=ROI_MARGIN,
                counts={s: {c: int(counts.loc[s, c]) for c in counts.columns} for s in counts.index},
                roi_found_rate=roi_rate, videos=video_info)
    (PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mediapipe", choices=["auto", "mediapipe", "skin"])
    main(ap.parse_args().backend)
