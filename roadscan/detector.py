"""Object detector wrapper. Anything that yields per-frame tracked boxes works; YOLO is the default."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .severity import canonical_class


@dataclass
class FrameDetection:
    track_id: int
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class Detector(Protocol):
    def __call__(self, frame: np.ndarray) -> list[FrameDetection]: ...
    def reset(self) -> None: ...


class YoloDetector:
    """Runs an Ultralytics YOLO model with ByteTrack so the same pothole keeps one ID across frames."""

    def __init__(self, weights: str, conf: float = 0.3, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.names = {i: canonical_class(n) for i, n in self.model.names.items()}
        unknown = [self.model.names[i] for i, c in self.names.items() if c is None]
        if len(unknown) == len(self.names):
            raise ValueError(
                f"None of the model's classes look like road damage: {list(self.model.names.values())}. "
                "Train one with training/train.py or pass weights from a road-damage dataset."
            )
        self._fresh = True

    def reset(self) -> None:
        self._fresh = True

    def __call__(self, frame: np.ndarray) -> list[FrameDetection]:
        res = self.model.track(
            frame, persist=not self._fresh, conf=self.conf, imgsz=self.imgsz,
            tracker="bytetrack.yaml", device=self.device, verbose=False,
        )[0]
        self._fresh = False
        out: list[FrameDetection] = []
        if res.boxes is None or res.boxes.id is None:
            return out
        for box, cls_i, conf, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                         res.boxes.conf.tolist(), res.boxes.id.tolist()):
            cls = self.names.get(int(cls_i))
            if cls:
                out.append(FrameDetection(int(tid), cls, float(conf), tuple(box)))
        return out
