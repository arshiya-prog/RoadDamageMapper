import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roadscan import db, jobs, lighting


def make_video(path, road_level, sky_level=200, seconds=3, fps=10, w=320, h=180):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(seconds * fps):
        frame = np.full((h, w, 3), road_level, np.uint8)
        frame[: h // 2] = sky_level  # a bright sky must not hide a dark road
        out.write(frame)
    out.release()
    return path


@pytest.mark.parametrize("road, expected", [(110, "ok"), (55, "low_light"), (20, "too_dark")])
def test_verdict_from_road_brightness(tmp_path, road, expected):
    result = lighting.assess(make_video(tmp_path / "v.mp4", road))
    assert result["lighting"] == expected
    assert result["median_luma"] == pytest.approx(road, abs=3)


def test_verdict_thresholds():
    assert lighting.verdict([]) == "unknown"
    assert lighting.verdict([30] * 5 + [100] * 5) == "too_dark"      # half dark
    assert lighting.verdict([30] * 4 + [100] * 6) == "low_light"     # 40% dim or worse
    assert lighting.verdict([50] * 2 + [100] * 8) == "ok"            # 20% dim: a few shadows are fine


def test_dark_upload_is_flagged_and_streamed(monkeypatch, tmp_path, gps_csv, blob_detector):
    monkeypatch.setattr(jobs, "get_detector", lambda weights=None: blob_detector)
    from server.app import app

    client = TestClient(app)
    video = make_video(tmp_path / "night.mp4", road_level=15, sky_level=10, seconds=10)
    with video.open("rb") as v, gps_csv.open("rb") as g:
        rid = client.post("/api/rides", files={"video": ("night.mp4", v), "gps": ("track.csv", g)}).json()["id"]
    for _ in range(100):
        ride = client.get(f"/api/rides/{rid}").json()
        if ride["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert ride["status"] == "done"
    assert ride["lighting"] == "too_dark" and ride["dark_fraction"] == 1.0
    assert '"lighting": "too_dark"' in client.get(f"/api/rides/{rid}/live").text
    client.delete(f"/api/rides/{rid}")


def test_backfill_checks_older_rides(tmp_path):
    conn = db.connect()
    rel = "rides/backfill-test/video.mp4"
    (db.MEDIA_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    make_video(db.MEDIA_DIR / rel, road_level=110)
    rid = db.create_ride(conn, "old", "2026-07-01T08:00:00+05:30", [[0, 18.5, 73.8], [1, 18.5001, 73.8]], rel, 1, 10)
    assert conn.execute("SELECT lighting FROM rides WHERE id=?", (rid,)).fetchone()[0] is None
    assert jobs.backfill_lighting() >= 1
    assert conn.execute("SELECT lighting FROM rides WHERE id=?", (rid,)).fetchone()[0] == "ok"
    conn.execute("DELETE FROM rides WHERE id=?", (rid,))
    conn.commit()
