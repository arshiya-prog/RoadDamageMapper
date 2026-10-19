"""HTTP API + static dashboard.

Run:  uvicorn server.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from roadscan import db, jobs, registry
from roadscan.geo import GpsTrack
from roadscan.priority import score_defects
from roadscan.severity import CLASS_LABELS

app = FastAPI(title="RoadScan Pune")


class _ThreadConnection:
    """One SQLite connection per thread. FastAPI runs requests on a thread pool, and sharing a single
    connection between concurrent requests returns mixed-up rows (seen as intermittent 500s)."""

    def __init__(self):
        self._local = threading.local()

    def _get(self):
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = db.connect()
        return c

    def __getattr__(self, name):
        return getattr(self._get(), name)


conn = _ThreadConnection()
executor = ThreadPoolExecutor(max_workers=1)  # one video at a time; YOLO saturates the GPU anyway
WEB_DIR = db.ROOT / "web"
UPLOAD_DIR = db.DATA_DIR / "uploads"  # outside /media so half-finished uploads are never served
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# When set (e.g. while exposed through a public tunnel), requests that change data must carry this key
# as an X-RoadScan-Key header or ?key= query parameter. Viewing stays open.
ACCESS_KEY = os.environ.get("ROADSCAN_KEY", "")


@app.middleware("http")
async def require_key_for_changes(request, call_next):
    if ACCESS_KEY and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        given = request.headers.get("x-roadscan-key") or request.query_params.get("key", "")
        if not hmac.compare_digest(given.encode(), ACCESS_KEY.encode()):
            return JSONResponse({"detail": "missing or wrong access key"}, status_code=401)
    return await call_next(request)


def _ride_summary(r) -> dict:
    d = dict(r)
    d.pop("track_json", None)
    d["n_detections"] = conn.execute("SELECT COUNT(*) FROM detections WHERE ride_id=?", (r["id"],)).fetchone()[0]
    d["video_url"] = f"/media/{r['video_path']}" if r["video_path"] else None
    return d


def _detection_out(d) -> dict:
    d = dict(d)
    d["label"] = CLASS_LABELS.get(d["cls"], d["cls"])
    d["crop_url"] = f"/media/{d['crop_path']}" if d.get("crop_path") else None
    return d


# ---- rides ---------------------------------------------------------------
@app.get("/api/rides")
def list_rides():
    return [_ride_summary(r) for r in conn.execute("SELECT * FROM rides ORDER BY recorded_at")]


@app.get("/api/rides/{ride_id}")
def get_ride(ride_id: int):
    r = conn.execute("SELECT * FROM rides WHERE id=?", (ride_id,)).fetchone()
    if not r:
        raise HTTPException(404, "ride not found")
    out = _ride_summary(r)
    out["track"] = json.loads(r["track_json"])
    out["detections"] = [_detection_out(d) for d in conn.execute(
        "SELECT * FROM detections WHERE ride_id=? ORDER BY t", (ride_id,))]
    # Defects seen on no other ride: these disappear if this ride is deleted.
    out["defects_only_here"] = conn.execute(
        """SELECT COUNT(DISTINCT defect_id) FROM detections WHERE ride_id=? AND defect_id NOT IN
           (SELECT defect_id FROM detections WHERE ride_id<>? AND defect_id IS NOT NULL)""", (ride_id, ride_id)).fetchone()[0]
    return out


def _finish_upload(staging: Path, video_name: str, gps_bytes: bytes, gps_filename: str, name: str,
                   recorded_at: str, gps_offset_s: float, gps_source: str) -> dict:
    """Shared by the single-request and chunked uploads: read the GPS track, register the ride, start processing.
    `staging` already holds the video as `video_name`."""
    if gps_source not in db.GPS_SOURCES:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(400, f"gps_source must be one of {db.GPS_SOURCES}")
    gps_suffix = Path(gps_filename or "track.json").suffix.lower()
    if gps_suffix not in {".json", ".gpx", ".csv"}:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(400, "GPS file must be .json, .gpx or .csv")
    gps_path = staging / f"gps{gps_suffix}"
    gps_path.write_bytes(gps_bytes)
    try:
        track = GpsTrack.load(gps_path, offset_s=gps_offset_s)
    except Exception as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(400, f"Could not read GPS track: {e}")

    if not recorded_at and gps_suffix == ".json":
        recorded_at = json.loads(gps_path.read_text()).get("started_at", "")
    recorded_at = recorded_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    name = name or f"Ride {recorded_at[:16].replace('T', ' ')}"

    with db.write_lock():
        # Uploaded rides are always run through the detector, so their source is "model".
        ride_id = db.create_ride(conn, name, recorded_at, track.to_json(), None, track.duration, track.distance_m(),
                                 source="model", gps_source=gps_source)
    final = db.MEDIA_DIR / "rides" / str(ride_id)
    shutil.rmtree(final, ignore_errors=True)
    final.parent.mkdir(parents=True, exist_ok=True)  # first upload on a fresh install
    shutil.move(str(staging), str(final))
    with db.write_lock():
        conn.execute("UPDATE rides SET video_path=? WHERE id=?", (f"rides/{ride_id}/{video_name}", ride_id))
        conn.commit()
    executor.submit(jobs.process_ride, ride_id)
    return {"id": ride_id, "status": "pending"}


def _video_name(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return "video" + (suffix if suffix in {".mp4", ".webm", ".mov", ".mkv", ".avi"} else ".mp4")


@app.post("/api/rides")
async def upload_ride(
    video: UploadFile = File(...),
    gps: UploadFile = File(...),
    name: str = Form(""),
    recorded_at: str = Form(""),
    gps_offset_s: float = Form(0.0),
    gps_source: str = Form("device"),
):
    staging = UPLOAD_DIR / uuid.uuid4().hex
    staging.mkdir(parents=True)
    video_name = _video_name(video.filename)
    with (staging / video_name).open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return _finish_upload(staging, video_name, await gps.read(), gps.filename, name, recorded_at, gps_offset_s, gps_source)


# ---- chunked upload ------------------------------------------------------
# Tunnels such as Cloudflare's cap one request at ~100 MB, so the phone recorder sends the video in parts:
#   POST /api/uploads -> {upload_id};  PUT /api/uploads/{id}/{index} (raw bytes, <= 64 MB each);
#   POST /api/uploads/{id}/complete (form: parts, filename, gps file + the same fields as /api/rides).
MAX_CHUNK_BYTES = 64 * 1024 * 1024


def _upload_dir(upload_id: str) -> Path:
    if not (len(upload_id) == 32 and all(c in "0123456789abcdef" for c in upload_id)):
        raise HTTPException(400, "bad upload id")
    d = UPLOAD_DIR / upload_id
    if not d.is_dir():
        raise HTTPException(404, "upload not found or expired")
    return d


STALE_UPLOAD_S = 24 * 3600


def _clear_stale_uploads() -> None:
    """Abandoned uploads (closed tab, lost signal) would otherwise sit on disk forever."""
    cutoff = datetime.now().timestamp() - STALE_UPLOAD_S
    for d in UPLOAD_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


@app.post("/api/uploads")
def start_upload():
    _clear_stale_uploads()
    upload_id = uuid.uuid4().hex
    (UPLOAD_DIR / upload_id / "parts").mkdir(parents=True)
    return {"upload_id": upload_id, "max_chunk_bytes": MAX_CHUNK_BYTES}


@app.put("/api/uploads/{upload_id}/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request):
    d = _upload_dir(upload_id)
    if not 0 <= index < 10_000:
        raise HTTPException(400, "bad chunk index")
    part = d / "parts" / f"{index:05d}"
    size = 0
    with part.open("wb") as f:
        async for block in request.stream():
            size += len(block)
            if size > MAX_CHUNK_BYTES:
                f.close()
                part.unlink(missing_ok=True)
                raise HTTPException(413, f"chunk larger than {MAX_CHUNK_BYTES} bytes")
            f.write(block)
    return {"index": index, "bytes": size}


@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(
    upload_id: str,
    parts: int = Form(...),
    filename: str = Form("ride.mp4"),
    gps: UploadFile = File(...),
    name: str = Form(""),
    recorded_at: str = Form(""),
    gps_offset_s: float = Form(0.0),
    gps_source: str = Form("device"),
):
    d = _upload_dir(upload_id)
    files = [d / "parts" / f"{i:05d}" for i in range(parts)]
    missing = [i for i, f in enumerate(files) if not f.exists()]
    if parts < 1 or missing:
        raise HTTPException(400, f"missing chunks: {missing[:10]}")
    video_name = _video_name(filename)
    with (d / video_name).open("wb") as out:
        for f in files:
            with f.open("rb") as src:
                shutil.copyfileobj(src, out)
    shutil.rmtree(d / "parts")
    return _finish_upload(d, video_name, await gps.read(), gps.filename, name, recorded_at, gps_offset_s, gps_source)


@app.delete("/api/rides/{ride_id}")
def remove_ride(ride_id: int):
    r = conn.execute("SELECT status FROM rides WHERE id=?", (ride_id,)).fetchone()
    if not r:
        raise HTTPException(404, "ride not found")
    if r["status"] in ("pending", "processing"):
        # The detector thread is still writing detections for this ride.
        raise HTTPException(409, "This ride is still being processed. Wait for it to finish, then delete it.")
    registry.delete_ride(conn, ride_id)
    shutil.rmtree(db.MEDIA_DIR / "rides" / str(ride_id), ignore_errors=True)
    shutil.rmtree(db.MEDIA_DIR / "crops" / str(ride_id), ignore_errors=True)
    return {"ok": True}


@app.get("/api/rides/{ride_id}/live")
async def live(ride_id: int, after: int = 0):
    """Server-sent events: progress updates and each defect as soon as it is detected."""

    async def stream():
        last = after
        while True:
            r = conn.execute("SELECT status, progress, error FROM rides WHERE id=?", (ride_id,)).fetchone()
            if not r:
                yield "event: error\ndata: {\"error\": \"ride not found\"}\n\n"
                return
            for d in conn.execute("SELECT * FROM detections WHERE ride_id=? AND id>? ORDER BY id", (ride_id, last)):
                last = d["id"]
                yield f"event: detection\ndata: {json.dumps(_detection_out(d))}\n\n"
            yield f"event: progress\ndata: {json.dumps(dict(r))}\n\n"
            if r["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---- defects & priority ----------------------------------------------------
def _check_source(source: str | None) -> None:
    if source not in (None, *db.SOURCES):
        raise HTTPException(400, f"source must be one of {db.SOURCES}")


def _scored(source: str | None = None) -> list[dict]:
    """Ranked defects. With a source, ranks are within that data source only."""
    _check_source(source)
    sql, args = "SELECT * FROM defects WHERE n_obs > 0", ()
    if source:
        sql, args = sql + " AND source=?", (source,)
    defects = db.rows(conn.execute(sql, args))
    # Context for each defect: latest ride (SQLite returns the row holding MAX), GPS quality, confidence.
    ctx = {r["defect_id"]: dict(r) for r in conn.execute(
        """SELECT det.defect_id, r.name AS last_ride, MAX(r.recorded_at) AS _latest, AVG(det.conf) AS mean_conf,
                  GROUP_CONCAT(DISTINCT r.gps_source) AS gps_sources
           FROM detections det JOIN rides r ON r.id = det.ride_id GROUP BY det.defect_id""")}
    for d in defects:
        c = ctx.get(d["id"], {})
        d["last_ride"] = c.get("last_ride")
        d["mean_conf"] = round(c.get("mean_conf") or 0, 3)
        d["gps_sources"] = sorted((c.get("gps_sources") or "").split(",")) if c.get("gps_sources") else []
    return score_defects(defects)


@app.get("/api/defects")
def list_defects(source: str | None = None):
    return _scored(source)


@app.get("/api/defects/{defect_id}")
def get_defect(defect_id: int):
    row = conn.execute("SELECT source FROM defects WHERE id=?", (defect_id,)).fetchone()
    d = row and next((d for d in _scored(row["source"]) if d["id"] == defect_id), None)
    if not d:
        raise HTTPException(404, "defect not found")
    d["history"] = [_detection_out(h) for h in conn.execute(
        """SELECT det.*, r.name AS ride_name, r.recorded_at, r.source AS ride_source, r.gps_source,
                  r.video_path IS NOT NULL AS has_video
           FROM detections det JOIN rides r ON r.id = det.ride_id
           WHERE det.defect_id=? ORDER BY r.recorded_at, det.t""", (defect_id,))]
    return d


@app.get("/api/priority.csv")
def priority_csv(source: str | None = None):
    buf = io.StringIO()
    w = csv.writer(buf)
    # Columns after "map" were added later; existing columns keep their order and meaning.
    # area_m2 is an uncalibrated estimate from the flat-road camera model (not field-validated).
    w.writerow(["rank", "priority", "type", "severity", "level", "area_m2", "trend_pts_per_month",
                "rides_seen", "first_seen", "last_seen", "status", "action", "reasons", "lat", "lon", "map",
                "defect_id", "data_source", "trend_status", "mean_confidence"])
    confs = {r[0]: r[1] for r in conn.execute("SELECT defect_id, AVG(conf) FROM detections GROUP BY defect_id")}
    for d in _scored(source):
        w.writerow([d["rank"], d["priority"], d["label"], d["severity"], d["level"], d["area_m2"],
                    d["trend_per_month"], d["n_obs"], d["first_seen"][:10], d["last_seen"][:10], d["status"],
                    d["action"], "; ".join(d["reasons"]), round(d["lat"], 6), round(d["lon"], 6), d["maps_url"],
                    d["code"], d["source"], d["trend_status"], round(confs.get(d["id"], 0), 3)])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=road_repair_priority.csv"})


@app.get("/api/stats")
def stats(source: str | None = None):
    ds = _scored(source)
    active = [d for d in ds if d["status"] == "active"]
    where, args = ("WHERE source=?", (source,)) if source else ("", ())
    r = conn.execute(f"""SELECT COUNT(*), SUM(distance_m), MIN(recorded_at), MAX(recorded_at),
                                SUM(status IN ('pending','processing')), SUM(status='failed')
                         FROM rides {where}""", args).fetchone()
    by_source = {s: {"rides": 0, "defects": 0} for s in db.SOURCES}
    for row in conn.execute("SELECT source, COUNT(*) FROM rides GROUP BY source"):
        by_source[row[0]]["rides"] = row[1]
    for row in conn.execute("SELECT source, COUNT(*) FROM defects WHERE n_obs > 0 GROUP BY source"):
        by_source[row[0]]["defects"] = row[1]
    return {
        "rides": r[0],
        "km_surveyed": round((r[1] or 0) / 1000, 1),
        "active_defects": len(active),
        "critical": sum(d["level"] == "critical" for d in active),
        # Only defects whose deterioration is supported by observations (see priority.trend_status).
        "worsening": sum(d["trend_status"] == "worsening" for d in active),
        "possibly_repaired": len(ds) - len(active),
        "multi_ride": sum(d["n_obs"] >= 2 for d in active),
        "date_from": r[2],
        "date_to": r[3],
        "processing": r[4] or 0,
        "failed": r[5] or 0,
        "by_source": by_source,
    }


# ---- static ------------------------------------------------------------------
db.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=db.MEDIA_DIR), name="media")


@app.middleware("http")
async def revalidate_dashboard(request, call_next):
    """Make browsers revalidate the dashboard's HTML/JS/CSS so they never run a stale copy after an update.
    Ride media keeps normal caching."""
    response = await call_next(request)
    if not request.url.path.startswith(("/media/", "/api/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/record")
def recorder():
    return FileResponse(WEB_DIR / "record.html")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
