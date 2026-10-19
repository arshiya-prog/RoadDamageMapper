"""Rules that keep the dashboard honest: data sources stay separate, and deterioration
is only claimed when the observations support it."""
import pytest

from roadscan import db, registry
from roadscan.pipeline import RideDetection
from roadscan.priority import score_defects, trend_status
from roadscan.severity import POTHOLE, level_for

BASE = (18.5300, 73.8475)
TRACK = [[t, BASE[0] + (t - 5) * 2e-5, BASE[1]] for t in range(11)]


def det(sev, dlat=0.0):
    return RideDetection(1.0, BASE[0] + dlat, BASE[1], POTHOLE, 0.8, sev, level_for(sev), 0.3, 0.6, 12)


def ride(conn, when, source="model", gps_source="device"):
    rid = db.create_ride(conn, "r", when, TRACK, None, 10, 100, source=source, gps_source=gps_source)
    db.set_ride_status(conn, rid, "done", 1)
    return rid


def defect_row(**kw):
    base = dict(id=1, cls=POTHOLE, lat=BASE[0], lon=BASE[1], n_obs=1, severity=60, first_severity=60, span_days=0,
                level="high", status="active", missed_passes=0, trend_per_month=None, source="model",
                first_seen="2026-07-01T00:00:00+05:30", last_seen="2026-07-01T00:00:00+05:30", area_m2=0.3)
    return {**base, **kw}


def test_synthetic_and_model_defects_never_merge(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    registry.add_detections(conn, ride(conn, "2026-06-01T08:00:00+05:30", source="synthetic", gps_source="simulated"), [det(50)])
    registry.add_detections(conn, ride(conn, "2026-07-01T08:00:00+05:30", source="model"), [det(70)])  # same spot
    rows = db.rows(conn.execute("SELECT source, n_obs FROM defects ORDER BY id"))
    assert rows == [{"source": "synthetic", "n_obs": 1}, {"source": "model", "n_obs": 1}]


def test_repair_check_ignores_rides_from_other_source(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    registry.add_detections(conn, ride(conn, "2026-06-01T08:00:00+05:30"), [det(60)])
    for when in ("2026-07-01T08:00:00+05:30", "2026-08-01T08:00:00+05:30"):
        ride(conn, when, source="synthetic", gps_source="simulated")  # synthetic rides pass by without seeing it
    registry.refresh_passes(conn)
    assert conn.execute("SELECT status, missed_passes FROM defects").fetchone()[:] == ("active", 0)


def test_evidence_fields_recorded(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    registry.add_detections(conn, ride(conn, "2026-06-01T08:00:00+05:30"), [det(40)])
    registry.add_detections(conn, ride(conn, "2026-06-21T08:00:00+05:30"), [det(65)])
    d = dict(conn.execute("SELECT first_severity, severity, span_days, n_obs FROM defects").fetchone())
    assert d == {"first_severity": 40, "severity": 65, "span_days": 20, "n_obs": 2}


@pytest.mark.parametrize("row, expected", [
    (dict(n_obs=1), "single"),
    (dict(n_obs=2, span_days=1, first_severity=40, severity=90, trend_per_month=1500), "too_close"),
    (dict(n_obs=2, span_days=30, first_severity=60, severity=66, trend_per_month=6), "stable"),
    (dict(n_obs=3, span_days=30, first_severity=50, severity=72, trend_per_month=22), "worsening"),
    (dict(n_obs=2, span_days=30, first_severity=70, severity=50, trend_per_month=-20), "decreased"),
])
def test_trend_status_rules(row, expected):
    assert trend_status(defect_row(**row))[0] == expected


def test_growth_points_only_when_worsening_is_supported():
    noisy = defect_row(n_obs=2, span_days=1, first_severity=40, severity=90, trend_per_month=1500)
    backed = defect_row(id=2, lat=BASE[0] + 0.01, n_obs=2, span_days=30, first_severity=50, severity=90, trend_per_month=40)
    out = {d["id"]: d for d in score_defects([noisy, backed])}
    assert not any("Growth" in b["label"] for b in out[1]["breakdown"])
    assert any(b["label"].startswith("Growth") and b["points"] == 20 for b in out[2]["breakdown"])
    assert not any("rose" in r for r in out[1]["reasons"])


def test_breakdown_adds_up_to_priority():
    rows = [defect_row(id=i, lat=BASE[0] + i * 1e-4, n_obs=i, severity=30 + 10 * i, level=level_for(30 + 10 * i))
            for i in range(1, 5)]
    rows.append(defect_row(id=9, lat=BASE[0] + 0.02, status="possibly_repaired", missed_passes=2))
    for d in score_defects(rows):
        assert sum(b["points"] for b in d["breakdown"]) == pytest.approx(d["priority"], abs=0.3)


def test_reasons_do_not_state_physical_size():
    out = score_defects([defect_row(area_m2=1.61)])[0]
    assert not any("m²" in r for r in out["reasons"])


def test_neighbour_bonus_counts_same_source_only():
    a = defect_row(id=1)
    b = defect_row(id=2, lat=BASE[0] + 1e-4, source="synthetic")
    c = defect_row(id=3, lat=BASE[0] + 2e-4, source="synthetic")
    out = {d["id"]: d for d in score_defects([a, b, c])}
    assert not any(x["label"].startswith("Nearby") for x in out[1]["breakdown"])
    assert any(x["label"].startswith("Nearby") for x in out[2]["breakdown"])
