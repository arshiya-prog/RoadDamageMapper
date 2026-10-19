"""Runs the full pipeline for one stored ride: detect → geolocate → register → refresh passes."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from . import db, lighting, registry
from .geo import GpsTrack
from .pipeline import RideDetection, process_video
from .severity import Camera

DEFAULT_WEIGHTS = os.environ.get("ROADSCAN_WEIGHTS", str(db.ROOT / "weights" / "road_damage.pt"))

_detector_cache: dict = {}


def get_detector(weights: str = DEFAULT_WEIGHTS):
    from .detector import YoloDetector

    if weights not in _detector_cache:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No model weights at {weights}. Train with `python training/train.py` "
                "or set ROADSCAN_WEIGHTS to a road-damage YOLO .pt file."
            )
        _detector_cache[weights] = YoloDetector(weights)
    return _detector_cache[weights]


def process_ride(ride_id: int, weights: str = DEFAULT_WEIGHTS, cam: Camera | None = None, sample_fps: float = 8.0):
    conn = db.connect()
    ride = conn.execute("SELECT * FROM rides WHERE id=?", (ride_id,)).fetchone()
    try:
        db.set_ride_status(conn, ride_id, "processing", 0)
        # Checked first so the dashboard can warn while detection is still running.
        db.set_lighting(conn, ride_id, lighting.assess(db.MEDIA_DIR / ride["video_path"]))
        track = GpsTrack.from_json(json.loads(ride["track_json"]))
        detector = get_detector(weights)

        # Commit each defect as soon as its track closes, so the live map can show it immediately.
        def on_detection(d: RideDetection):
            registry.add_detections(conn, ride_id, [d])

        process_video(
            db.MEDIA_DIR / ride["video_path"], track, detector, cam=cam, sample_fps=sample_fps,
            on_progress=lambda p: db.set_ride_status(conn, ride_id, "processing", round(p, 3)),
            on_detection=on_detection,
        )
        db.set_ride_status(conn, ride_id, "done", 1.0)
        registry.refresh_passes(conn)
    except Exception as e:  # surface the failure in the UI rather than dying silently in a thread
        traceback.print_exc()
        db.set_ride_status(conn, ride_id, "failed", error=str(e))
    finally:
        conn.close()


def backfill_lighting() -> int:
    """Check rides that have a video but were processed before the lighting check existed."""
    conn = db.connect()
    rows = conn.execute("SELECT id, video_path FROM rides WHERE video_path IS NOT NULL AND lighting IS NULL").fetchall()
    for r in rows:
        path = db.MEDIA_DIR / r["video_path"]
        if path.exists():
            db.set_lighting(conn, r["id"], lighting.assess(path))
    conn.close()
    return len(rows)
