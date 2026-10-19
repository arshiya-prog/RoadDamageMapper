import os
import sys
import tempfile
from pathlib import Path

# Isolate test data before roadscan.db computes its paths.
os.environ["ROADSCAN_DATA"] = tempfile.mkdtemp(prefix="roadscan-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from roadscan.detector import FrameDetection  # noqa: E402

FPS, W, H = 15, 640, 360
# Two potholes approach the camera: one during 1-4 s, one during 6-9 s.
EPISODES = [(1.0, 4.0, 1), (6.0, 9.0, 2)]


def pothole_box(t: float):
    for start, end, tid in EPISODES:
        if start <= t <= end:
            k = (t - start) / (end - start)          # 0 = far away, 1 = right in front
            cy = 200 + k * 140
            half_w, half_h = 12 + k * 70, 4 + k * 22
            return tid, (W / 2 - half_w, cy - half_h, W / 2 + half_w, cy + half_h)
    return None


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "ride.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * 10):
        frame = np.full((H, W, 3), 110, np.uint8)
        frame[: H // 2 - 20] = (200, 170, 140)  # sky
        hit = pothole_box(i / FPS)
        if hit:
            x1, y1, x2, y2 = (int(v) for v in hit[1])
            cv2.ellipse(frame, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (25, 25, 25), -1)
        out.write(frame)
    out.release()
    return path


class BlobDetector:
    """Stand-in for YOLO: finds dark blobs on the road with plain thresholding."""

    def __init__(self):
        self.next_id, self.last_seen = 0, False

    def reset(self):
        self.next_id, self.last_seen = 0, False

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = (gray < 60).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 30:
                dets.append((x, y, x + w, y + h))
        if dets and not self.last_seen:
            self.next_id += 1
        self.last_seen = bool(dets)
        return [FrameDetection(self.next_id, "pothole", 0.8, tuple(map(float, b))) for b in dets[:1]]


@pytest.fixture
def blob_detector():
    return BlobDetector()


@pytest.fixture
def gps_csv(tmp_path) -> Path:
    """Driving due north at ~10 m/s from Shivajinagar."""
    p = tmp_path / "track.csv"
    rows = ["t,lat,lon"] + [f"{t},{18.5300 + t * 10 / 111_320:.7f},73.8475" for t in range(0, 12)]
    p.write_text("\n".join(rows))
    return p
