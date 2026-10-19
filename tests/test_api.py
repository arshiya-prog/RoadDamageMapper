import time

from fastapi.testclient import TestClient

from roadscan import jobs


def test_upload_processes_ride_and_streams_detections(monkeypatch, synthetic_video, gps_csv, blob_detector):
    monkeypatch.setattr(jobs, "get_detector", lambda weights=None: blob_detector)
    from server.app import app

    client = TestClient(app)
    with synthetic_video.open("rb") as v, gps_csv.open("rb") as g:
        r = client.post("/api/rides", files={"video": ("ride.mp4", v, "video/mp4"), "gps": ("track.csv", g, "text/csv")},
                        data={"name": "Test ride", "recorded_at": "2026-10-01T08:00:00+05:30"})
    assert r.status_code == 200, r.text
    ride_id = r.json()["id"]

    for _ in range(100):
        ride = client.get(f"/api/rides/{ride_id}").json()
        if ride["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert ride["status"] == "done", ride.get("error")
    assert len(ride["detections"]) == 2
    assert ride["detections"][0]["crop_url"].startswith("/media/crops/")
    assert client.get(ride["detections"][0]["crop_url"]).status_code == 200

    events = client.get(f"/api/rides/{ride_id}/live").text
    assert events.count("event: detection") == 2 and '"status": "done"' in events

    worklist = client.get("/api/defects").json()
    assert [d["rank"] for d in worklist] == [1, 2]
    csv = client.get("/api/priority.csv").text
    assert csv.splitlines()[0].startswith("rank,priority,type")

    assert client.delete(f"/api/rides/{ride_id}").json() == {"ok": True}
    assert client.get("/api/defects").json() == []


def test_rejects_bad_gps_file(synthetic_video, tmp_path):
    from server.app import app

    bad = tmp_path / "track.txt"
    bad.write_text("nope")
    with synthetic_video.open("rb") as v, bad.open("rb") as g:
        r = TestClient(app).post("/api/rides", files={"video": ("ride.mp4", v), "gps": ("track.txt", g)})
    assert r.status_code == 400


def test_source_filters_summary_and_csv_columns(monkeypatch, synthetic_video, gps_csv, blob_detector):
    from roadscan import db, registry
    from roadscan.pipeline import RideDetection
    from server.app import app, conn

    monkeypatch.setattr(jobs, "get_detector", lambda weights=None: blob_detector)
    client = TestClient(app)
    with synthetic_video.open("rb") as v, gps_csv.open("rb") as g:
        rid = client.post("/api/rides", files={"video": ("ride.mp4", v), "gps": ("track.csv", g)},
                          data={"gps_source": "reconstructed", "recorded_at": "2026-10-02T08:00:00+05:30"}).json()["id"]
    for _ in range(100):
        if client.get(f"/api/rides/{rid}").json()["status"] == "done":
            break
        time.sleep(0.1)
    ride = client.get(f"/api/rides/{rid}").json()
    assert (ride["source"], ride["gps_source"]) == ("model", "reconstructed")

    syn = db.create_ride(conn, "demo", "2026-06-01T08:00:00+05:30", [[0, 18.6, 73.9], [1, 18.6001, 73.9]], None, 1, 10,
                         source="synthetic", gps_source="simulated")
    registry.add_detections(conn, syn, [RideDetection(0.5, 18.6, 73.9, "pothole", 0.9, 80, "critical", 0.5, 0.8, 9)])
    db.set_ride_status(conn, syn, "done", 1)

    model = client.get("/api/defects?source=model").json()
    synthetic = client.get("/api/defects?source=synthetic").json()
    assert model and all(d["source"] == "model" for d in model)
    assert len(synthetic) == 1 and synthetic[0]["source"] == "synthetic" and synthetic[0]["rank"] == 1
    assert {"code", "breakdown", "trend_status", "trend_label", "last_ride", "mean_conf", "gps_sources"} <= set(model[0])
    assert model[0]["gps_sources"] == ["reconstructed"]
    assert client.get("/api/defects?source=bogus").status_code == 400

    s = client.get("/api/stats").json()
    assert s["by_source"]["synthetic"]["rides"] == 1 and s["by_source"]["model"]["rides"] >= 1
    assert {"date_from", "date_to", "processing", "failed", "multi_ride"} <= set(s)
    assert client.get("/api/stats?source=synthetic").json()["rides"] == 1

    detail = client.get(f"/api/defects/{synthetic[0]['id']}").json()
    assert detail["history"][0]["ride_source"] == "synthetic"

    header = client.get("/api/priority.csv?source=model").text.splitlines()[0].split(",")
    assert header[:16] == ["rank", "priority", "type", "severity", "level", "area_m2", "trend_pts_per_month",
                           "rides_seen", "first_seen", "last_seen", "status", "action", "reasons", "lat", "lon", "map"]
    assert header[16:] == ["defect_id", "data_source", "trend_status", "mean_confidence"]

    for r in (rid, syn):
        client.delete(f"/api/rides/{r}")


def test_rejects_bad_gps_source(synthetic_video, gps_csv):
    from server.app import app

    with synthetic_video.open("rb") as v, gps_csv.open("rb") as g:
        r = TestClient(app).post("/api/rides", files={"video": ("ride.mp4", v), "gps": ("track.csv", g)},
                                 data={"gps_source": "guessed"})
    assert r.status_code == 400


def test_chunked_upload_reassembles_video(monkeypatch, synthetic_video, gps_csv, blob_detector):
    monkeypatch.setattr(jobs, "get_detector", lambda weights=None: blob_detector)
    from server.app import app

    client = TestClient(app)
    data = synthetic_video.read_bytes()
    upload_id = client.post("/api/uploads").json()["upload_id"]
    size = len(data) // 3 + 1
    pieces = [data[i:i + size] for i in range(0, len(data), size)]
    for i, piece in enumerate(pieces):
        assert client.put(f"/api/uploads/{upload_id}/{i}", content=piece).json() == {"index": i, "bytes": len(piece)}

    # A missing chunk is reported rather than producing a corrupt video.
    with gps_csv.open("rb") as g:
        r = client.post(f"/api/uploads/{upload_id}/complete", data={"parts": len(pieces) + 1},
                        files={"gps": ("track.csv", g)})
    assert r.status_code == 400 and "missing chunks" in r.text

    with gps_csv.open("rb") as g:
        ride_id = client.post(f"/api/uploads/{upload_id}/complete", files={"gps": ("track.csv", g)},
                              data={"parts": len(pieces), "filename": "ride.mp4", "name": "chunked"}).json()["id"]
    for _ in range(100):
        ride = client.get(f"/api/rides/{ride_id}").json()
        if ride["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert ride["status"] == "done" and len(ride["detections"]) == 2
    assert client.get(ride["video_url"]).content == data
    assert client.put(f"/api/uploads/{'0' * 32}/0", content=b"x").status_code == 404
    # Upload ids must be 32 hex chars before they are used in a filesystem path.
    assert client.put("/api/uploads/not-an-id/0", content=b"x").status_code == 400
    client.delete(f"/api/rides/{ride_id}")


def test_access_key_guards_changes_but_not_viewing(monkeypatch):
    import server.app as srv

    monkeypatch.setattr(srv, "ACCESS_KEY", "s3cret")
    client = TestClient(srv.app)
    assert client.get("/api/stats").status_code == 200
    assert client.post("/api/uploads").status_code == 401
    assert client.post("/api/uploads", headers={"X-RoadScan-Key": "wrong"}).status_code == 401
    assert client.delete("/api/rides/999999").status_code == 401
    assert client.post("/api/uploads", headers={"X-RoadScan-Key": "s3cret"}).status_code == 200
    assert client.post("/api/uploads?key=s3cret").status_code == 200


def test_stale_uploads_are_cleared():
    import os
    import server.app as srv

    old = srv.UPLOAD_DIR / ("a" * 32)
    (old / "parts").mkdir(parents=True)
    os.utime(old, (0, 0))
    fresh_id = TestClient(srv.app).post("/api/uploads").json()["upload_id"]
    assert not old.exists() and (srv.UPLOAD_DIR / fresh_id).exists()


def test_delete_ride_rules(monkeypatch, synthetic_video, gps_csv, blob_detector):
    from roadscan import db
    from server.app import app, conn

    monkeypatch.setattr(jobs, "get_detector", lambda weights=None: blob_detector)
    client = TestClient(app)
    assert client.delete("/api/rides/999999").status_code == 404

    # A ride that is still being processed can't be deleted (the detector is writing to it).
    busy = db.create_ride(conn, "busy", "2026-10-01T08:00:00+05:30", [[0, 18.5, 73.8], [1, 18.5001, 73.8]], None, 1, 10)
    db.set_ride_status(conn, busy, "processing", 0.4)
    r = client.delete(f"/api/rides/{busy}")
    assert r.status_code == 409 and "still being processed" in r.json()["detail"]
    db.set_ride_status(conn, busy, "done", 1.0)
    assert client.delete(f"/api/rides/{busy}").json() == {"ok": True}

    # A processed ride: the dialog's impact count, then everything is removed.
    with synthetic_video.open("rb") as v, gps_csv.open("rb") as g:
        rid = client.post("/api/rides", files={"video": ("ride.mp4", v), "gps": ("track.csv", g)}).json()["id"]
    for _ in range(100):
        ride = client.get(f"/api/rides/{rid}").json()
        if ride["status"] == "done":
            break
        time.sleep(0.1)
    assert ride["defects_only_here"] == 2
    crop = ride["detections"][0]["crop_url"]
    assert client.get(crop).status_code == 200
    assert client.delete(f"/api/rides/{rid}").json() == {"ok": True}
    assert client.get(f"/api/rides/{rid}").status_code == 404
    assert client.get(ride["video_url"]).status_code == 404 and client.get(crop).status_code == 404
    assert not [d for d in client.get("/api/defects").json() if d["last_ride"] == ride["name"]]
