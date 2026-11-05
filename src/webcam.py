"""Live webcam recognition via streamlit-webrtc. Frames are processed in memory and never stored."""
import threading
import time

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from .config import CLASSES, DISPLAY


class GestureProcessor(VideoProcessorBase):
    def __init__(self, predictor, interval=0.10, smooth=0.6):
        self.pred, self.interval, self.smooth = predictor, interval, smooth
        self.lock = threading.Lock()
        self._last_inf, self._ema = 0.0, None
        self._frames, self._t0, self._inf_n = 0, time.time(), 0
        self.stats = dict(label="-", conf=0.0, probs=[0, 0, 0], latency_ms=0.0, fps=0.0, inf_fps=0.0, box=None, found=None)
        self._lat = []

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        now = time.time()
        self._frames += 1
        if now - self._last_inf >= self.interval:                      # throttle inference
            self._last_inf = now
            r = self.pred.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            p = r["probs"] if self._ema is None else self.smooth * self._ema + (1 - self.smooth) * r["probs"]
            self._ema = p
            self._inf_n += 1
            self._lat = (self._lat + [r["latency_ms"]])[-30:]
            i = int(np.argmax(p))
            with self.lock:
                self.stats.update(label=DISPLAY[CLASSES[i]], conf=float(p[i]), probs=[float(v) for v in p],
                                  latency_ms=float(np.mean(self._lat)), box=r["box"], found=r["roi_found"])
        dt = now - self._t0
        if dt >= 1.0:
            with self.lock:
                self.stats.update(fps=self._frames / dt, inf_fps=self._inf_n / dt)
            self._frames, self._inf_n, self._t0 = 0, 0, now
        with self.lock:
            s = dict(self.stats)
        if s["box"]:
            x0, y0, x1, y1 = s["box"]
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (80, 200, 120), 2)
        txt = f'{s["label"]}  {s["conf"]*100:.0f}%'
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 44), (30, 30, 30), -1)
        cv2.putText(bgr, txt, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    def get_stats(self):
        with self.lock:
            return dict(self.stats)
