import json
import math

import pytest

from roadscan import db, registry
from roadscan.geo import Fix, GpsTrack, destination, haversine_m
from roadscan.pipeline import RideDetection, process_video
from roadscan.priority import score_defects
from roadscan.severity import POTHOLE, Camera, SizeEstimate, estimate_size, level_for, severity_score


# ---- geo -------------------------------------------------------------------
def test_destination_roundtrip():
    lat, lon = destination(18.53, 73.85, 45, 100)
    assert haversine_m(18.53, 73.85, lat, lon) == pytest.approx(100, abs=0.01)


def test_track_interpolates_and_heads_north(gps_csv):
    trk = GpsTrack.load(gps_csv)
    lat, _ = trk.position(2.5)
    assert lat == pytest.approx(18.5300 + 25 / 111_320, abs=1e-7)
    assert trk.heading(5) == pytest.approx(0, abs=1)
    assert trk.distance_m() == pytest.approx(110, rel=0.01)


def test_gps_offset_shifts_time(gps_csv):
    trk = GpsTrack.load(gps_csv, offset_s=2)
    assert trk.position(0) == GpsTrack.load(gps_csv).position(2)


# ---- severity ----------------------------------------------------------------
def test_lower_rows_are_closer():
    cam = Camera()
    assert cam.ground_distance(700, 720) < cam.ground_distance(500, 720) < cam.ground_distance(400, 720)


def test_same_box_further_away_is_bigger():
    cam = Camera()
    near = estimate_size((600, 600, 700, 650), 1280, 720, cam)
    far = estimate_size((600, 420, 700, 470), 1280, 720, cam)
    assert far.area_m2 > near.area_m2


def test_box_reaching_horizon_stays_plausible():
    # Bottom edge 6 m ahead, top edge above the horizon: must not become hundreds of m².
    size = estimate_size((400, 300, 900, 560), 1280, 720, Camera())
    assert size.depth_m <= 4.0 + 1e-6
    assert size.area_m2 < 25


def test_severity_grows_with_area_and_levels():
    small = severity_score(POTHOLE, SizeEstimate(8, 0.2, 0.2, 0.03, 0.2), 0.9)
    big = severity_score(POTHOLE, SizeEstimate(8, 1.2, 0.9, 0.85, 1.2), 0.9)
    assert small < big
    assert level_for(small) == "low" and level_for(big) == "critical"


# ---- pipeline ----------------------------------------------------------------
def test_video_pipeline_dedupes_and_geolocates(synthetic_video, gps_csv, blob_detector):
    trk = GpsTrack.load(gps_csv)
    dets = process_video(synthetic_video, trk, blob_detector, sample_fps=8)
    assert len(dets) == 2, "each pothole spans many frames but should be reported once"
    assert all(d.cls == POTHOLE and d.crop_jpg for d in dets)
    assert 1 <= dets[0].t < 4 and 6 <= dets[1].t < 9
    # Projected ahead of the vehicle, i.e. north of where the car was at detection time.
    for d in dets:
        car_lat, _ = trk.position(d.t)
        assert d.lat > car_lat
        assert d.n_frames >= 10


# ---- registry & priority -----------------------------------------------------
def _ride(conn, when, track_rows):
    return db.create_ride(conn, "r", when, track_rows, None, 10, 100)


def test_cross_ride_matching_trend_and_repair(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    base = (18.5300, 73.8475)
    track = [[t, base[0] + (t - 5) * 2e-5, base[1]] for t in range(11)]  # passes right over the pothole

    def det(sev, dlat=0.0, cls=POTHOLE):
        return RideDetection(1.0, base[0] + dlat, base[1], cls, 0.8, sev, level_for(sev), 0.3, 0.6, 12)

    r1 = _ride(conn, "2026-06-01T08:00:00+05:30", track)
    registry.add_detections(conn, r1, [det(40)])
    r2 = _ride(conn, "2026-07-01T08:00:00+05:30", track)
    registry.add_detections(conn, r2, [det(70, dlat=4e-5)])        # ~4.5 m GPS drift: same pothole
    registry.add_detections(conn, r2, [det(30, dlat=3e-4)])        # ~33 m away: a different one
    for r in (r1, r2):
        db.set_ride_status(conn, r, "done", 1)

    defects = db.rows(conn.execute("SELECT * FROM defects ORDER BY id"))
    assert len(defects) == 2
    main = defects[0]
    assert main["n_obs"] == 2 and main["severity"] == 70
    assert main["trend_per_month"] == pytest.approx(30, rel=0.05)

    # Two later rides drive past without seeing it → flagged as possibly repaired.
    for when in ("2026-08-01T08:00:00+05:30", "2026-09-01T08:00:00+05:30"):
        db.set_ride_status(conn, _ride(conn, when, track), "done", 1)
    registry.refresh_passes(conn)
    status = conn.execute("SELECT status, missed_passes FROM defects WHERE id=?", (main["id"],)).fetchone()
    assert status["status"] == "possibly_repaired" and status["missed_passes"] == 2

    ranked = score_defects(db.rows(conn.execute("SELECT * FROM defects")))
    assert ranked[0]["status"] == "active", "repaired defects drop to the bottom of the worklist"
    assert any("verify repair" in r for r in ranked[-1]["reasons"])
