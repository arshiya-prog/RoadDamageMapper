"""GPS helpers: distances, bearings, and a time-indexed track you can query at any video timestamp."""
from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EARTH_R = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    """Point reached by travelling dist_m from (lat, lon) along bearing (degrees)."""
    d, b = dist_m / EARTH_R, math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


@dataclass
class Fix:
    t: float  # seconds since video start
    lat: float
    lon: float


class GpsTrack:
    """A GPS track aligned to video time. Positions between fixes are linearly interpolated."""

    def __init__(self, fixes: list[Fix]):
        if len(fixes) < 2:
            raise ValueError("GPS track needs at least two fixes")
        self.fixes = sorted(fixes, key=lambda f: f.t)
        self._ts = [f.t for f in self.fixes]

    # ---- loaders -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, offset_s: float = 0.0) -> "GpsTrack":
        """Load .json (from the phone recorder), .gpx or .csv (t,lat,lon).

        offset_s shifts GPS time relative to video time; use it when the GPS logger
        was started before/after the camera.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            fixes = cls._from_json(json.loads(path.read_text()))
        elif suffix == ".gpx":
            fixes = cls._from_gpx(path.read_text())
        elif suffix == ".csv":
            fixes = cls._from_csv(path)
        else:
            raise ValueError(f"Unsupported GPS file type: {suffix}")
        return cls([Fix(f.t - offset_s, f.lat, f.lon) for f in fixes])

    @staticmethod
    def _from_json(data: dict | list) -> list[Fix]:
        points = data["points"] if isinstance(data, dict) else data
        return [Fix(float(p["t"]), float(p["lat"]), float(p["lon"])) for p in points]

    @staticmethod
    def _from_gpx(text: str) -> list[Fix]:
        import gpxpy

        gpx = gpxpy.parse(text)
        raw = [p for trk in gpx.tracks for seg in trk.segments for p in seg.points if p.time]
        if not raw:
            raise ValueError("GPX has no timestamped points")
        t0 = raw[0].time
        return [Fix((p.time - t0).total_seconds(), p.latitude, p.longitude) for p in raw]

    @staticmethod
    def _from_csv(path: Path) -> list[Fix]:
        with path.open() as f:
            return [Fix(float(r["t"]), float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)]

    # ---- queries -------------------------------------------------------
    @property
    def duration(self) -> float:
        return self._ts[-1] - self._ts[0]

    def position(self, t: float) -> tuple[float, float]:
        i = bisect.bisect_right(self._ts, t)
        if i <= 0:
            return self.fixes[0].lat, self.fixes[0].lon
        if i >= len(self.fixes):
            return self.fixes[-1].lat, self.fixes[-1].lon
        a, b = self.fixes[i - 1], self.fixes[i]
        k = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
        return a.lat + (b.lat - a.lat) * k, a.lon + (b.lon - a.lon) * k

    def heading(self, t: float, window_s: float = 2.0) -> float:
        """Direction of travel around time t, smoothed over a short window."""
        lat1, lon1 = self.position(t - window_s / 2)
        lat2, lon2 = self.position(t + window_s / 2)
        if haversine_m(lat1, lon1, lat2, lon2) < 0.5:  # stationary: widen the window
            lat1, lon1 = self.position(t - 5)
            lat2, lon2 = self.position(t + 5)
        return bearing_deg(lat1, lon1, lat2, lon2)

    def distance_m(self) -> float:
        return sum(haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(self.fixes, self.fixes[1:]))

    def min_distance_to(self, lat: float, lon: float) -> float:
        return min(haversine_m(lat, lon, f.lat, f.lon) for f in self.fixes)

    def to_json(self) -> list[list[float]]:
        return [[round(f.t, 2), round(f.lat, 7), round(f.lon, 7)] for f in self.fixes]

    @classmethod
    def from_json(cls, rows: list[list[float]]) -> "GpsTrack":
        return cls([Fix(*r) for r in rows])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
