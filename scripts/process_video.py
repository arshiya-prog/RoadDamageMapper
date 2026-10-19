"""Process a recorded ride from the command line (no server needed).

  python scripts/process_video.py ride.mp4 track.gpx --name "Karve Rd" --recorded-at 2026-10-12T08:00:00+05:30

The ride and its defects land in the same database the dashboard reads.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadscan import db, jobs  # noqa: E402
from roadscan.geo import GpsTrack  # noqa: E402
from roadscan.severity import Camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("gps", type=Path, help=".json (recorder), .gpx or .csv with t,lat,lon")
    ap.add_argument("--name", default=None)
    ap.add_argument("--recorded-at", default=None, help="ISO timestamp; defaults to now")
    ap.add_argument("--gps-offset", type=float, default=0.0, help="seconds GPS logging started before the video")
    ap.add_argument("--gps-source", choices=db.GPS_SOURCES, default="device",
                    help="'reconstructed' if the track was rebuilt from a map rather than logged")
    ap.add_argument("--weights", default=jobs.DEFAULT_WEIGHTS)
    ap.add_argument("--fps", type=float, default=8.0, help="frames per second to analyse")
    ap.add_argument("--cam-height", type=float, default=1.3, help="metres; ~1.0 for a bike handlebar mount")
    ap.add_argument("--cam-pitch", type=float, default=8.0, help="degrees the phone tilts down")
    args = ap.parse_args()

    track = GpsTrack.load(args.gps, offset_s=args.gps_offset)
    recorded_at = args.recorded_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = db.connect()
    ride_id = db.create_ride(conn, args.name or args.video.stem, recorded_at, track.to_json(), None,
                             track.duration, track.distance_m(), source="model", gps_source=args.gps_source)
    dest = db.MEDIA_DIR / "rides" / str(ride_id)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.video, dest / f"video{args.video.suffix.lower()}")
    conn.execute("UPDATE rides SET video_path=? WHERE id=?", (f"rides/{ride_id}/video{args.video.suffix.lower()}", ride_id))
    conn.commit()

    print(f"Processing ride {ride_id} ({track.distance_m() / 1000:.1f} km)…")
    jobs.process_ride(ride_id, weights=args.weights, sample_fps=args.fps,
                      cam=Camera(height_m=args.cam_height, pitch_deg=args.cam_pitch))
    r = conn.execute("SELECT status, error, lighting, dark_fraction FROM rides WHERE id=?", (ride_id,)).fetchone()
    if r["lighting"] in ("too_dark", "low_light"):
        print(f"WARNING: video is {r['lighting'].replace('_', ' ')} ({r['dark_fraction']:.0%} of sampled frames dark) "
              "— detections are unreliable; re-record in daylight.")
    n = conn.execute("SELECT COUNT(*) FROM detections WHERE ride_id=?", (ride_id,)).fetchone()[0]
    print(f"Ride {ride_id}: {r['status']}, {n} defects" + (f" — {r['error']}" if r["error"] else ""))


if __name__ == "__main__":
    main()
