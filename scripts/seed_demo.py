"""Seed a realistic demo: four survey rides over the same 5 km Pune route across the 2026 monsoon.

Ground-truth potholes grow month to month, a few new ones appear, one stretch gets
patched, and one ride is driven in the opposite direction. The detections go through
the real registry (matching, trends, missed passes) so the dashboard shows exactly
what the system would produce from real rides.

Usage:  python scripts/seed_demo.py [--reset]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadscan import db, registry  # noqa: E402
from roadscan.geo import Fix, GpsTrack, bearing_deg, destination, haversine_m  # noqa: E402
from roadscan.pipeline import RideDetection  # noqa: E402
from roadscan.severity import (CRACK_ALLIGATOR, CRACK_LONG, CRACK_TRANS, POTHOLE, SizeEstimate,  # noqa: E402
                               level_for, severity_score)

RIDES = [
    ("Pre-monsoon baseline", "2026-06-18T08:10:00+05:30", False),
    ("Mid-monsoon survey", "2026-07-24T07:45:00+05:30", False),
    ("Return trip (FC Rd → University)", "2026-08-27T18:20:00+05:30", True),
    ("Post-monsoon survey", "2026-09-29T08:05:00+05:30", False),
]
RIDE_DATES = [datetime.fromisoformat(r[1]) for r in RIDES]


class Route:
    def __init__(self, coords: list[list[float]]):
        self.pts = [(lat, lon) for lon, lat in coords]
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + haversine_m(*a, *b))
        self.length = self.cum[-1]

    def at(self, s: float) -> tuple[float, float, float]:
        s = max(0.0, min(s, self.length - 0.01))
        i = max(1, next(k for k, c in enumerate(self.cum) if c >= s))
        a, b = self.pts[i - 1], self.pts[i]
        k = (s - self.cum[i - 1]) / max(self.cum[i] - self.cum[i - 1], 1e-6)
        return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, bearing_deg(*a, *b)


def simulate_drive(route: Route, reverse: bool, rng: random.Random) -> tuple[GpsTrack, list[float]]:
    """1 Hz GPS with Pune-traffic speeds (stop at signals, 15-35 km/h otherwise) and ~3 m noise.
    Returns the noisy track and the true distance-along-route for each second."""
    s, t, fixes, along = 0.0, 0, [], []
    stops = sorted(rng.uniform(300, route.length - 300) for _ in range(4))
    stop_left = 0
    while s < route.length:
        pos = route.length - s if reverse else s
        lat, lon, _ = route.at(pos)
        nlat, nlon = destination(lat, lon, rng.uniform(0, 360), abs(rng.gauss(0, 2.5)))
        fixes.append(Fix(float(t), nlat, nlon))
        along.append(pos)
        if stop_left > 0:
            stop_left -= 1
        elif stops and s >= stops[0]:
            stops.pop(0)
            stop_left = rng.randint(15, 45)
        else:
            speed_kmh = 24 + 9 * math.sin(t / 37) + rng.uniform(-4, 4)
            s += speed_kmh / 3.6
        t += 1
    return GpsTrack(fixes), along


def make_truth(route: Route, rng: random.Random) -> list[dict]:
    kinds = [POTHOLE] * 15 + [CRACK_ALLIGATOR] * 5 + [CRACK_LONG] * 5 + [CRACK_TRANS] * 4
    rng.shuffle(kinds)
    truth = []
    # A bad stretch near Shivajinagar plus scattered damage elsewhere.
    positions = [rng.uniform(2400, 2750) for _ in range(7)] + [rng.uniform(150, route.length - 150) for _ in range(22)]
    for i, (s, cls) in enumerate(zip(positions, kinds)):
        lat, lon, brg = route.at(s)
        lat, lon = destination(lat, lon, (brg + 90) % 360, rng.uniform(-3.5, 3.5))
        appears = 0 if rng.random() > 0.3 else rng.choice([1, 2])     # new monsoon damage
        truth.append({
            "s": s, "lat": lat, "lon": lon, "cls": cls, "appears": appears,
            "size0": {POTHOLE: rng.uniform(0.04, 0.35), CRACK_ALLIGATOR: rng.uniform(0.4, 1.6)}.get(cls, rng.uniform(0.6, 2.5)),
            "growth": rng.choice([0.05, 0.25, 0.45, 0.7]) if cls == POTHOLE else rng.choice([0.02, 0.1, 0.25]),
            "repaired_before": None,
        })
    # PMC patched two potholes in August.
    for d in [t for t in truth if t["cls"] == POTHOLE and t["appears"] == 0][:2]:
        d["repaired_before"] = 2
    return truth


def size_at(d: dict, ride_idx: int, rng: random.Random) -> SizeEstimate:
    months = (RIDE_DATES[ride_idx] - RIDE_DATES[d["appears"]]).days / 30
    g = (1 + d["growth"]) ** months
    if d["cls"] in (POTHOLE, CRACK_ALLIGATOR):
        area = d["size0"] * g * rng.uniform(0.9, 1.1)
        w = math.sqrt(area * 4 / math.pi * 1.3)
        return SizeEstimate(8.0, w, area * 4 / math.pi / w, area, max(w, area * 4 / math.pi / w))
    length = d["size0"] * g * rng.uniform(0.9, 1.1)
    return SizeEstimate(8.0, length, 0.05, length * 0.05, length)


def render_crop(cls: str, size: SizeEstimate, sev: float, rng: random.Random) -> bytes:
    """Synthetic road thumbnail so the demo has imagery; real rides store true video crops."""
    h, w = 200, 300
    img = (np.random.default_rng(rng.randint(0, 1 << 30)).normal(95, 14, (h, w, 1))).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(np.repeat(img, 3, axis=2), (3, 3), 0)
    cx, cy = w // 2, h // 2
    px_per_m = 110
    if cls == POTHOLE:
        ax, ay = int(min(size.width_m * px_per_m / 2, 120)), int(min(size.depth_m * px_per_m / 2 * 0.55, 75))
        cv2.ellipse(img, (cx, cy), (ax + 4, ay + 4), 0, 0, 360, (70, 72, 74), -1)
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (38, 40, 44), -1)
        if sev > 60:
            cv2.ellipse(img, (cx + ax // 5, cy + ay // 5), (ax // 2, ay // 2), 0, 0, 360, (90, 70, 55), -1)  # water/mud
        box = (cx - ax - 6, cy - ay - 6, cx + ax + 6, cy + ay + 6)
    else:
        n = {CRACK_ALLIGATOR: 22, CRACK_LONG: 1, CRACK_TRANS: 1}[cls]
        span = int(min(size.length_m * px_per_m / 2, 135))
        for _ in range(n):
            if cls == CRACK_ALLIGATOR:
                x0, y0 = cx + rng.randint(-span, span), cy + rng.randint(-span // 2, span // 2)
                pts = [(x0, y0)]
                for _ in range(3):
                    pts.append((pts[-1][0] + rng.randint(-25, 25), pts[-1][1] + rng.randint(-25, 25)))
            elif cls == CRACK_LONG:
                pts = [(cx + rng.randint(-8, 8), cy - span // 2 + k * span // 6) for k in range(7)]
            else:
                pts = [(cx - span + k * span // 4, cy + rng.randint(-6, 6)) for k in range(9)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (30, 30, 32), 2)
        box = (cx - span - 8, cy - span // 2 - 10, cx + span + 8, cy + span // 2 + 10) if cls != CRACK_LONG \
            else (cx - 20, cy - span // 2 - 10, cx + 20, cy + span // 2 + 10)
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (max(x1, 2), max(y1, 2)), (min(x2, w - 3), min(y2, h - 3)), (0, 80, 255), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="wipe the database and media first")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.reset:
        db.DB_PATH.unlink(missing_ok=True)
        for p in (db.DB_PATH.with_suffix(".db-wal"), db.DB_PATH.with_suffix(".db-shm")):
            p.unlink(missing_ok=True)
        shutil.rmtree(db.MEDIA_DIR, ignore_errors=True)

    rng = random.Random(args.seed)
    route_data = json.loads((db.ROOT / "demo" / "route.json").read_text())
    route = Route(route_data["coordinates"])
    truth = make_truth(route, rng)
    conn = db.connect()

    for idx, (name, when, reverse) in enumerate(RIDES):
        track, along = simulate_drive(route, reverse, rng)
        ride_id = db.create_ride(conn, name, when, track.to_json(), None, track.duration, track.distance_m(),
                                 source="synthetic", gps_source="simulated")
        dets = []
        for d in truth:
            if d["appears"] > idx or (d["repaired_before"] is not None and idx >= d["repaired_before"]):
                continue
            if rng.random() > 0.93:  # detector miss
                continue
            # Video time when the defect is first confidently in view (~12 m ahead).
            target = d["s"] + 12 if reverse else d["s"] - 12
            t = min(range(len(along)), key=lambda k: abs(along[k] - target))
            size = size_at(d, idx, rng)
            conf = rng.uniform(0.45, 0.92)
            sev = severity_score(d["cls"], size, conf)
            lat, lon = destination(d["lat"], d["lon"], rng.uniform(0, 360), abs(rng.gauss(0, 2.5)))
            rd = RideDetection(t=float(t), lat=lat, lon=lon, cls=d["cls"], conf=round(conf, 3), severity=sev,
                               level=level_for(sev), area_m2=round(size.area_m2, 3), length_m=round(size.length_m, 2),
                               n_frames=rng.randint(6, 30))
            rd.crop_jpg = render_crop(d["cls"], size, sev, rng)
            dets.append(rd)
        # A couple of false positives (shadows, manhole covers) that never repeat.
        for _ in range(2):
            s = rng.uniform(100, route.length - 100)
            lat, lon, _ = route.at(s)
            t = min(range(len(along)), key=lambda k: abs(along[k] - s))
            size = SizeEstimate(8, 0.3, 0.2, 0.05, 0.3)
            sev = severity_score(POTHOLE, size, 0.35)
            dets.append(RideDetection(float(t), lat, lon, POTHOLE, 0.35, sev, level_for(sev), 0.05, 0.3, 3,
                                      render_crop(POTHOLE, size, sev, rng)))
        dets.sort(key=lambda r: r.t)
        registry.add_detections(conn, ride_id, dets)
        db.set_ride_status(conn, ride_id, "done", 1.0)
        print(f"ride {ride_id}: {name:<36} {track.distance_m() / 1000:.1f} km, {track.duration / 60:.0f} min, "
              f"{len(dets)} detections")

    registry.refresh_passes(conn)
    n = conn.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
    print(f"{n} unique defects registered (ground truth: {len(truth)} + one-off false positives)")


if __name__ == "__main__":
    main()
