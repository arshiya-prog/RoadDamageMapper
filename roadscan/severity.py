"""Turn a 2D bounding box into a physical size estimate and a 0-100 severity score.

A phone on a windshield looks at a flat road, so each image row maps to a ground
distance (pinhole camera + known mount height/pitch). That lets us convert box
pixels to approximate metres, which is what a road engineer actually cares about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Canonical damage classes. Detector-specific labels get mapped onto these.
POTHOLE = "pothole"
CRACK_LONG = "longitudinal_crack"
CRACK_TRANS = "transverse_crack"
CRACK_ALLIGATOR = "alligator_crack"
CLASSES = [POTHOLE, CRACK_LONG, CRACK_TRANS, CRACK_ALLIGATOR]

CLASS_LABELS = {
    POTHOLE: "Pothole",
    CRACK_LONG: "Longitudinal crack",
    CRACK_TRANS: "Transverse crack",
    CRACK_ALLIGATOR: "Alligator cracking",
}

# Map common dataset label names (RDD2022, Roboflow pothole sets) to canonical classes.
LABEL_ALIASES = {
    "pothole": POTHOLE, "potholes": POTHOLE, "d40": POTHOLE,
    "d00": CRACK_LONG, "longitudinal crack": CRACK_LONG, "longitudinal_crack": CRACK_LONG,
    "d10": CRACK_TRANS, "transverse crack": CRACK_TRANS, "transverse_crack": CRACK_TRANS,
    "d20": CRACK_ALLIGATOR, "alligator crack": CRACK_ALLIGATOR, "alligator_crack": CRACK_ALLIGATOR,
}


def canonical_class(label: str) -> str | None:
    return LABEL_ALIASES.get(label.strip().lower())


def class_family(cls: str) -> str:
    return "pothole" if cls == POTHOLE else "crack"


LEVELS = [(30, "low"), (55, "medium"), (75, "high"), (101, "critical")]


def level_for(score: float) -> str:
    return next(name for cutoff, name in LEVELS if score < cutoff)


@dataclass
class Camera:
    """Phone mount geometry. Defaults suit a phone on a car windshield."""

    height_m: float = 1.3       # lens height above road (≈1.0 on a bike handlebar)
    pitch_deg: float = 8.0      # tilt below horizontal
    vfov_deg: float = 50.0      # vertical field of view in landscape
    max_range_m: float = 30.0   # beyond this, estimates are too noisy to trust

    def focal_px(self, frame_h: int) -> float:
        return (frame_h / 2) / math.tan(math.radians(self.vfov_deg / 2))

    def ground_distance(self, y_px: float, frame_h: int) -> float:
        """Forward distance (m) to the road point imaged at row y_px."""
        f = self.focal_px(frame_h)
        angle = math.radians(self.pitch_deg) + math.atan((y_px - frame_h / 2) / f)
        if angle <= math.radians(1.0):
            return self.max_range_m
        return min(self.height_m / math.tan(angle), self.max_range_m)


MAX_DEPTH_M = 4.0  # along-road extent of a single pothole or crack patch


@dataclass
class SizeEstimate:
    distance_m: float   # forward distance from camera to the near edge of the defect
    width_m: float
    depth_m: float      # extent along the road
    area_m2: float
    length_m: float     # longest dimension (relevant for cracks)


def estimate_size(box: tuple[float, float, float, float], frame_w: int, frame_h: int, cam: Camera) -> SizeEstimate:
    x1, y1, x2, y2 = box
    d_near = cam.ground_distance(y2, frame_h)
    # Near the horizon a few pixels span many metres, so a box whose top edge creeps up there
    # would claim an absurd extent. Cap how far along the road one defect can reach.
    d_far = min(cam.ground_distance(y1, frame_h), d_near + MAX_DEPTH_M)
    f = cam.focal_px(frame_h)
    d_mid = (d_near + d_far) / 2
    slant = math.hypot(d_mid, cam.height_m)
    width = (x2 - x1) * slant / f
    depth = max(d_far - d_near, 0.05)
    area = math.pi / 4 * width * depth  # potholes are roughly elliptical
    return SizeEstimate(d_near, width, depth, area, max(width, depth))


def _ramp(x: float, x0: float, x1: float) -> float:
    """0 at x0, 1 at x1, log-scaled because defect sizes span orders of magnitude."""
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    return math.log(x / x0) / math.log(x1 / x0)


def severity_score(cls: str, size: SizeEstimate, conf: float) -> float:
    """0-100. Potholes scale with area; cracks with length; alligator cracking is weighted up
    because it signals base failure and usually precedes potholes."""
    if cls == POTHOLE:
        base = 15 + 85 * _ramp(size.area_m2, 0.02, 1.2)
    elif cls == CRACK_ALLIGATOR:
        base = 20 + 65 * _ramp(size.area_m2, 0.1, 4.0)
    else:
        base = 5 + 55 * _ramp(size.length_m, 0.3, 6.0)
    # Low-confidence detections are pulled toward the middle rather than trusted at face value.
    return round(max(0.0, min(100.0, base * (0.75 + 0.25 * conf))), 1)
