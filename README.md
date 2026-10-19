# RoadScan Pune — pothole & road-damage mapping from dashcam video

Mount a phone on a car or bike, record a ride, and RoadScan finds potholes and cracks with a YOLO
detector, estimates their real size and severity, and pins each one on a map from the phone's GPS.
Across repeated rides it recognises the *same* pothole, tracks whether it is getting worse, notices
when one has been patched, and produces a ranked repair worklist a ward office can act on.

```
 phone (/record) ──video + GPS──▶ FastAPI ──▶ YOLO + ByteTrack ──▶ size & severity ──▶ geolocate
                                     │                                                  │
   dashboard ◀── SSE live markers ───┤◀── defect registry (cross-ride matching, trends) ◀┘
   (replay · defects · worklist)     └──▶ priority.csv
```

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/seed_demo.py --reset      # 4 demo rides over a 5 km Pune route
.venv/bin/uvicorn server.app:app --port 8000
```

Open http://localhost:8000. The demo data covers Ganeshkhind Rd → Shivajinagar → FC Road / Jangli
Maharaj Rd, surveyed four times across the 2026 monsoon. One ride is driven the other way, and PMC
patches two potholes in August.

**Demo script:** *Ride replay* → pick "Post-monsoon survey" → 30× → Play. Damage markers drop onto the
map as the vehicle reaches them. Then open *Defects* → #1 to see that pothole's severity climb across
four rides with a photo from each, and *Repair worklist* → Export CSV.

## Recording a real ride

Phones only allow the camera and GPS on HTTPS pages, so the recorder is served through a free
Cloudflare quick tunnel (`brew install cloudflared` once; no account needed).

1. On the Mac, run `scripts/start_tunnel.sh`. It starts the server, opens the tunnel, keeps the Mac
   awake, and prints a phone link with a QR code. Leave it running. Ctrl+C stops everything.
2. Scan the QR code with the phone. If it doesn't open straight away, wait about 30 s for the tunnel's
   DNS to spread. Allow camera and location access when asked.
3. Mount the phone landscape, low on the windshield and level, with the bonnet just out of frame.
   Tap **Start recording** and drive. Video and GPS are kept on the phone during the ride.
4. Tap **Stop & upload**. The video is sent in 20 MB pieces, and each piece is retried if the mobile
   connection drops. If the upload fails, **Retry upload** or **Save to phone instead** keeps the
   recording. Processing then starts, and the dashboard shows each defect as it's found.

Details worth knowing:

- **Data use:** the recorder uses 2.5 Mbps, about 19 MB per minute, uploaded once at the end of the ride.
- **Access key:** the script sets `ROADSCAN_KEY`, and the link carries it once as `?key=…`.
  Uploads and deletes need the key, but anyone with the link can view the dashboard. The tunnel URL
  changes on every start and dies with Ctrl+C.
- **Mac must stay on:** it has to stay on, with the lid open on battery, while you record and
  upload.

Recorded with another app instead? Upload any video plus a `.gpx` (e.g. from *GPS Logger*) or a
`.csv` with `t,lat,lon` from the dashboard's **＋ Ride** button, or from the command line:

```bash
.venv/bin/python scripts/process_video.py ride.mp4 track.gpx --recorded-at 2026-10-12T08:00:00+05:30 \
    --gps-offset 3.5 --cam-height 1.0    # bike mount; GPS started 3.5 s before the camera
```

## The detector

Processing needs road-damage weights at `weights/road_damage.pt` (or set `ROADSCAN_WEIGHTS`).

**RDD2022** is the recommended dataset. It has about 47k images from six countries, including about
10k from Indian roads, labelled with potholes (D40) plus longitudinal (D00), transverse (D10) and
alligator (D20) cracks.

```bash
# download RDD2022 (13 GB) from https://figshare.com/articles/dataset/21431547 and unzip, then:
.venv/bin/python training/prepare_rdd2022.py --src ~/Downloads/RDD2022 --countries India Japan Czech
.venv/bin/python training/train.py --data rdd2022 --model yolo11s.pt --epochs 80
```

**Current model** (`weights/road_damage.pt`, not in git): YOLO11s, 60 epochs on the RDD2022 India
subset (3,489 train / 396 val images), about 3.7 hours on an M-series Mac (MPS). Validation mAP50:

| Class | mAP50 | Recall | Notes |
|---|---|---|---|
| Alligator cracking | 0.64 | 0.62 | |
| Pothole | 0.51 | 0.45 | Per-image score. Video tracking sees each pothole in 10+ frames, so per-ride recall is higher. |
| Longitudinal crack | 0.42 | 0.41 | |
| Transverse crack | ≈0 | 0 | Only 39 training examples in the India subset. Add Japan/Czech data to fix. |

Any YOLO-format dataset also works, such as a Roboflow Universe or Kaggle pothole export:
`training/train.py --data path/to/data.yaml`. Class names are mapped through
`roadscan/severity.py:LABEL_ALIASES`.

## How it works

| Stage | File | What it does |
|---|---|---|
| Sync | `roadscan/geo.py` | Aligns GPS fixes with video time, interpolates position and heading at any frame. |
| Detect + track | `roadscan/detector.py` | YOLO with ByteTrack at ~8 fps, so one pothole keeps one ID over the 1–3 s it is in view. |
| Measure | `roadscan/severity.py` | Treats the road as flat: with the phone's height, tilt and FOV, each image row maps to a ground distance. Box pixels then become metres, giving pothole area or crack length. |
| Summarise | `roadscan/pipeline.py` | Collapses a track to one detection using frames 2.5–18 m away, where size estimates are reliable. Drops single-frame blips and saves the best crop. |
| Geolocate | `roadscan/pipeline.py` | Projects from the car's GPS fix forward along the heading by the measured distance, and sideways by the horizontal offset, so the pin is on the pothole and not on the car. |
| Match across rides | `roadscan/registry.py` | Treats detections of the same family (pothole vs crack) within 10 m as the same defect. Recomputes location as a confidence-weighted mean, latest severity, and a growth trend from a regression of severity over time. |
| Repairs | `roadscan/registry.py` | If two later rides pass within 12 m without seeing a defect, it becomes "possibly repaired", which also clears one-off false positives. |
| Prioritise | `roadscan/priority.py` | Score = severity + growth rate + type (potholes, then alligator cracking) + confirmations + nearby damage. Each item gets an action (48 h / 7 days / monthly cycle / monitor) and plain-language reasons. |

### What the dashboard's numbers mean

The dashboard's **How scores work** dialog has the full definitions. In short:

- **Data source.** *Model prediction* means detected by the trained model. *Synthetic demo* means
  generated by `seed_demo.py`, so nothing in it was detected. The two are never merged into one defect.
  - Synthetic items are drawn as rings and tagged "Synthetic".
  - The header switch shows model predictions only, synthetic only, or both.
- **GPS source.** *Device* means logged by the phone. *Reconstructed* means rebuilt afterwards from a
  map route and video landmarks (±10–20 m). *Simulated* is used for synthetic rides only.
- **Severity index (0–100).** A relative index from the apparent box size, using an assumed camera
  mount. It is **not field-validated**, so the UI does not display physical sizes. `area_m2` remains in
  the CSV for compatibility, but treat it as uncalibrated.
- **Priority score.** A hand-set heuristic, itemised for every defect in the API (`breakdown`) and in
  the UI:
  - 0.55 × severity
  - +10 for a pothole, +6 for alligator cracking
  - Growth, up to +20, only when worsening is evidence-backed
  - +2.5 per confirming ride, up to +10
  - +1 per nearby active defect, up to +5 (same data source only)
  - × 0.2 if possibly repaired
  - Capped at 100
- **Worsening.** A defect is only called worsening when it was seen on 2 or more rides at least
  7 days apart and its severity rose by 10 or more points. Rides closer together than that are
  labelled "too close to judge".

## API

`GET /api/rides`, `GET /api/rides/{id}` (track + detections), `POST /api/rides` (multipart: `video`,
`gps`, `name`, `recorded_at`, `gps_offset_s`), `GET /api/rides/{id}/live` (server-sent events),
`DELETE /api/rides/{id}`, `GET /api/defects`, `GET /api/defects/{id}` (with per-ride history),
`GET /api/priority.csv`, `GET /api/stats`. Chunked upload: `POST /api/uploads`, `PUT /api/uploads/{id}/{n}`,
`POST /api/uploads/{id}/complete`.
`/api/defects`, `/api/stats` and `/api/priority.csv` accept `?source=model|synthetic`. Uploads accept
`gps_source=device|reconstructed`.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests use a synthetic dashcam video and a thresholding stand-in for YOLO. They cover GPS sync,
size estimation, frame de-duplication, geolocation, cross-ride matching, trend, repair detection and
the full upload → process → live stream → worklist flow through the API.

## Known limits and next steps

- Size comes from a flat-road pinhole model, so calibrate `--cam-height` and `--cam-pitch` for your
  mount. Pothole **depth** isn't measured. Adding a segmentation model, or the phone's LiDAR or
  stereo depth, would help.
- Phone GPS is ±3–8 m, and worse under flyovers such as the Shivajinagar stretch. Matching
  tolerates 10 m. Snapping to OSM road geometry would tighten it.
- Next for a municipal pilot: tag each defect with its PMC ward and road name (reverse geocode or
  ward polygons), and route the worklist to ward engineers.
