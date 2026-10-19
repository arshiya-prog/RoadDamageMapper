"""Cross-ride defect registry: links detections of the same pothole across rides and tracks how it changes."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .db import MEDIA_DIR, write_lock
from .geo import GpsTrack, haversine_m
from .pipeline import RideDetection
from .severity import class_family, level_for

MATCH_RADIUS_M = 10.0      # phone GPS is typically ±3-8 m
PASS_RADIUS_M = 12.0       # a ride "passed" a defect if its track came this close
REPAIRED_AFTER_MISSES = 2  # one miss can be glare or a parked car; two in a row is a pattern


def _days(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp() / 86400


def _ride_source(conn, ride_id: int) -> str:
    row = conn.execute("SELECT source FROM rides WHERE id=?", (ride_id,)).fetchone()
    return row["source"] if row else "model"


def find_match(conn, d: RideDetection, ride_id: int) -> int | None:
    # Synthetic demo defects and real model detections are never merged into one defect.
    # Cheap bounding-box prefilter (~0.0002° ≈ 22 m) before exact distance.
    cands = conn.execute(
        "SELECT id, cls, lat, lon FROM defects WHERE source=? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (_ride_source(conn, ride_id), d.lat - 0.0002, d.lat + 0.0002, d.lon - 0.0002, d.lon + 0.0002),
    ).fetchall()
    best, best_dist = None, MATCH_RADIUS_M
    for c in cands:
        if class_family(c["cls"]) != class_family(d.cls):
            continue
        dist = haversine_m(d.lat, d.lon, c["lat"], c["lon"])
        if dist <= best_dist:
            best, best_dist = c["id"], dist
    return best


def add_detections(conn, ride_id: int, detections: list[RideDetection]) -> list[int]:
    """Store a ride's detections, attach each to an existing defect or a new one. Returns detection ids."""
    crop_dir = MEDIA_DIR / "crops" / str(ride_id)
    ids = []
    with write_lock():
        for d in detections:
            defect_id = find_match(conn, d, ride_id)
            if defect_id is None:
                defect_id = conn.execute(
                    "INSERT INTO defects (cls, lat, lon, source) VALUES (?,?,?,?)",
                    (d.cls, d.lat, d.lon, _ride_source(conn, ride_id)),
                ).lastrowid
            cur = conn.execute(
                """INSERT INTO detections (ride_id, defect_id, t, lat, lon, cls, conf, severity, level,
                   area_m2, length_m, n_frames) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ride_id, defect_id, d.t, d.lat, d.lon, d.cls, d.conf, d.severity, d.level,
                 d.area_m2, d.length_m, d.n_frames),
            )
            det_id = cur.lastrowid
            if d.crop_jpg:
                crop_dir.mkdir(parents=True, exist_ok=True)
                path = crop_dir / f"{det_id}.jpg"
                path.write_bytes(d.crop_jpg)
                conn.execute("UPDATE detections SET crop_path=? WHERE id=?",
                             (str(path.relative_to(MEDIA_DIR)), det_id))
            refresh_defect(conn, defect_id)
            ids.append(det_id)
        conn.commit()
    return ids


def refresh_defect(conn, defect_id: int) -> None:
    """Recompute a defect's location, current severity and growth trend from all its detections."""
    dets = conn.execute(
        """SELECT d.*, r.recorded_at FROM detections d JOIN rides r ON r.id = d.ride_id
           WHERE d.defect_id = ? ORDER BY r.recorded_at, d.t""", (defect_id,)
    ).fetchall()
    if not dets:
        conn.execute("DELETE FROM defects WHERE id=?", (defect_id,))
        return
    # One value per ride (a defect may be caught twice on a loop route; keep the worst view).
    per_ride: dict[int, dict] = {}
    for d in dets:
        if d["ride_id"] not in per_ride or d["severity"] > per_ride[d["ride_id"]]["severity"]:
            per_ride[d["ride_id"]] = dict(d)
    history = sorted(per_ride.values(), key=lambda d: d["recorded_at"])
    latest = history[-1]
    w = [d["conf"] for d in dets]
    lat = sum(d["lat"] * k for d, k in zip(dets, w)) / sum(w)
    lon = sum(d["lon"] * k for d, k in zip(dets, w)) / sum(w)
    cls = Counter(d["cls"] for d in dets).most_common(1)[0][0]
    trend = None
    if len(history) >= 2:
        xs = [_days(h["recorded_at"]) for h in history]
        ys = [h["severity"] for h in history]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var > 0:
            trend = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var * 30, 2)
    span_days = round(_days(latest["recorded_at"]) - _days(history[0]["recorded_at"]), 2)
    conn.execute(
        """UPDATE defects SET cls=?, lat=?, lon=?, first_seen=?, last_seen=?, n_obs=?, severity=?, level=?,
           area_m2=?, length_m=?, trend_per_month=?, first_severity=?, span_days=? WHERE id=?""",
        (cls, lat, lon, history[0]["recorded_at"], latest["recorded_at"], len(history), latest["severity"],
         level_for(latest["severity"]), latest["area_m2"], latest["length_m"], trend,
         history[0]["severity"], span_days, defect_id),
    )


def refresh_passes(conn) -> None:
    """Mark defects that later rides drove past without detecting — candidates for 'already repaired'."""
    rides = conn.execute("SELECT id, recorded_at, track_json, source FROM rides WHERE status='done'").fetchall()
    tracks = {r["id"]: (r["recorded_at"], GpsTrack.from_json(json.loads(r["track_json"])), r["source"]) for r in rides}
    seen = defaultdict(set)
    for r in conn.execute("SELECT defect_id, ride_id FROM detections"):
        seen[r["defect_id"]].add(r["ride_id"])
    with write_lock():
        for d in conn.execute("SELECT id, lat, lon, last_seen, source FROM defects").fetchall():
            missed = sum(
                1 for rid, (when, trk, src) in tracks.items()
                if src == d["source"] and when > d["last_seen"] and rid not in seen[d["id"]]
                and trk.min_distance_to(d["lat"], d["lon"]) <= PASS_RADIUS_M
            )
            conn.execute("UPDATE defects SET missed_passes=?, status=? WHERE id=?",
                         (missed, "possibly_repaired" if missed >= REPAIRED_AFTER_MISSES else "active", d["id"]))
        conn.commit()


def delete_ride(conn, ride_id: int) -> None:
    with write_lock():
        affected = [r[0] for r in conn.execute("SELECT DISTINCT defect_id FROM detections WHERE ride_id=?", (ride_id,))]
        conn.execute("DELETE FROM detections WHERE ride_id=?", (ride_id,))
        conn.execute("DELETE FROM rides WHERE id=?", (ride_id,))
        for d in affected:
            refresh_defect(conn, d)
        conn.commit()
    refresh_passes(conn)
