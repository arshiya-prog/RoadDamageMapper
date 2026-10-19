"""Is a ride bright enough for the detector to be trusted?

The model was trained on daytime RDD2022 images. Measured on the road area ahead of the vehicle
(0–255 grey level), those images have a median of ~95 and 95% are above ~57; overcast monsoon
dashcam footage from Pune sits around 72–96. Frames much darker than the training data produce
missed defects and false alarms (headlight glare, shadows), so we warn rather than guess.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DARK_LUMA = 45.0          # darker than almost every training image
DIM_LUMA = 60.0           # at the dark edge of the training data
TOO_DARK_FRACTION = 0.5   # half the ride is dark  -> "too_dark"
LOW_LIGHT_FRACTION = 0.3  # a third is dim or worse -> "low_light"
SAMPLES = 24


def road_luma(frame: np.ndarray) -> float:
    """Mean grey level of the road ahead: middle band of the lower half, skipping sky and bonnet."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray[int(h * 0.50):int(h * 0.85), int(w * 0.15):int(w * 0.85)].mean())


def verdict(lumas: list[float]) -> str:
    if not lumas:
        return "unknown"
    a = np.asarray(lumas)
    if (a < DARK_LUMA).mean() >= TOO_DARK_FRACTION:
        return "too_dark"
    if (a < DIM_LUMA).mean() >= LOW_LIGHT_FRACTION:
        return "low_light"
    return "ok"


def assess(video_path: str | Path, samples: int = SAMPLES) -> dict:
    """Sample frames evenly through the video. Cheap (a second or two) so it can run before detection."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    lumas = []
    for i in range(samples if total else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
        ok, frame = cap.read()
        if ok:
            lumas.append(road_luma(frame))
    cap.release()
    a = np.asarray(lumas) if lumas else np.asarray([np.nan])
    return {
        "lighting": verdict(lumas),
        "median_luma": round(float(np.median(a)), 1) if lumas else None,
        "dark_fraction": round(float((a < DARK_LUMA).mean()), 2) if lumas else None,
    }
