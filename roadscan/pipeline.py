"""Video + GPS → a list of geolocated, de-duplicated, severity-rated road defects.

Each physical pothole shows up in dozens of frames as the vehicle approaches it.
The tracker gives it a stable ID; when the ID disappears we collapse all its
observations into one RideDetection, using only frames where it was at a
distance we can measure reliably.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import Detector, FrameDetection
from .geo import GpsTrack, destination
from .severity import Camera, SizeEstimate, estimate_size, level_for, severity_score

RELIABLE_RANGE_M = (2.5, 18.0)
TRACK_TIMEOUT_S = 1.5


@dataclass
class Observation:
    t: float
    det: FrameDetection
    size: SizeEstimate
    lat: float
    lon: float


@dataclass
class RideDetection:
    t: float
    lat: float
    lon: float
    cls: str
    conf: float
    severity: float
    level: str
    area_m2: float
    length_m: float
    n_frames: int
    crop_jpg: bytes | None = field(default=None, repr=False)


def locate(track: GpsTrack, t: float, box, frame_w: int, frame_h: int, size: SizeEstimate, cam: Camera):
    """Project the defect from the vehicle's GPS position to where it actually sits on the road."""
    lat, lon = track.position(t)
    heading = track.heading(t)
    ahead = size.distance_m + size.depth_m / 2
    lat, lon = destination(lat, lon, heading, ahead)
    cx = (box[0] + box[2]) / 2
    lateral = (cx - frame_w / 2) * size.distance_m / cam.focal_px(frame_h)
    if abs(lateral) > 0.3:
        lat, lon = destination(lat, lon, (heading + 90) % 360, lateral)
    return lat, lon


def summarise(obs: list[Observation]) -> RideDetection | None:
    if len(obs) < 2 and max(o.det.conf for o in obs) < 0.6:
        return None  # single weak frame: most likely a shadow or a patch
    usable = [o for o in obs if RELIABLE_RANGE_M[0] <= o.size.distance_m <= RELIABLE_RANGE_M[1]] or obs
    votes = Counter()
    for o in obs:
        votes[o.det.cls] += o.det.conf
    cls = votes.most_common(1)[0][0]
    usable = [o for o in usable if o.det.cls == cls] or usable
    confs = sorted((o.det.conf for o in usable), reverse=True)
    conf = statistics.mean(confs[:5])
    med = SizeEstimate(
        distance_m=statistics.median(o.size.distance_m for o in usable),
        width_m=statistics.median(o.size.width_m for o in usable),
        depth_m=statistics.median(o.size.depth_m for o in usable),
        area_m2=statistics.median(o.size.area_m2 for o in usable),
        length_m=statistics.median(o.size.length_m for o in usable),
    )
    sev = severity_score(cls, med, conf)
    w = [o.det.conf for o in usable]
    lat = sum(o.lat * k for o, k in zip(usable, w)) / sum(w)
    lon = sum(o.lon * k for o, k in zip(usable, w)) / sum(w)
    return RideDetection(
        t=round(usable[0].t, 2), lat=lat, lon=lon, cls=cls, conf=round(conf, 3), severity=sev,
        level=level_for(sev), area_m2=round(med.area_m2, 3), length_m=round(med.length_m, 2), n_frames=len(obs),
    )


def _crop(frame: np.ndarray, box) -> bytes:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_x, pad_y = max(40, (x2 - x1) // 2), max(40, (y2 - y1) // 2)
    cx1, cy1, cx2, cy2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    img = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(img, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 80, 255), 2)
    scale = 320 / max(img.shape[1], 1)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def process_video(
    video_path: str | Path,
    track: GpsTrack,
    detector: Detector,
    cam: Camera | None = None,
    sample_fps: float = 8.0,
    on_progress: Callable[[float], None] | None = None,
    on_detection: Callable[[RideDetection], None] | None = None,
) -> list[RideDetection]:
    cam = cam or Camera()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, round(fps / sample_fps))
    detector.reset()

    open_tracks: dict[int, list[Observation]] = defaultdict(list)
    last_seen: dict[int, float] = {}
    best_crop: dict[int, tuple[float, bytes]] = {}
    results: list[RideDetection] = []

    def close(tid: int):
        obs = open_tracks.pop(tid)
        last_seen.pop(tid, None)
        rd = summarise(obs)
        if rd:
            rd.crop_jpg = best_crop.pop(tid, (0, None))[1]
            results.append(rd)
            if on_detection:
                on_detection(rd)
        best_crop.pop(tid, None)

    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            fh, fw = frame.shape[:2]
            for d in detector(frame):
                size = estimate_size(d.box, fw, fh, cam)
                lat, lon = locate(track, t, d.box, fw, fh, size, cam)
                open_tracks[d.track_id].append(Observation(t, d, size, lat, lon))
                last_seen[d.track_id] = t
                # Best thumbnail: largest box that is still fully in view and within reliable range.
                score = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                if d.box[3] < fh * 0.97 and score > best_crop.get(d.track_id, (0, None))[0]:
                    best_crop[d.track_id] = (score, _crop(frame, d.box))
            for tid in [k for k, ts in last_seen.items() if t - ts > TRACK_TIMEOUT_S]:
                close(tid)
            if on_progress and idx % (step * 10) == 0:
                on_progress(min(idx / total, 0.999))
        idx += 1
    for tid in list(open_tracks):
        close(tid)
    cap.release()
    if on_progress:
        on_progress(1.0)
    return sorted(results, key=lambda r: r.t)
