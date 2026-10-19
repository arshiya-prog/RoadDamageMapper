"""Ranks defects into a repair worklist a ward engineer can act on, with plain-language reasons.

The priority score is a transparent, hand-set heuristic (not a learned or validated model).
Every point is itemised in `breakdown` so the dashboard can show exactly how a score was built:

  0.55 × severity index                         (0–55)
  + 10 if pothole, + 6 if alligator cracking     (type)
  + up to 20 for growth, ONLY when worsening is supported (see trend_status)
  + up to 10 for independent ride confirmations  (2.5 per ride, max 4 rides)
  + 1 per other active defect within 60 m        (max 5, same data source only)
  × 0.2 if possibly repaired; capped at 100
"""
from __future__ import annotations

from datetime import datetime

from .geo import haversine_m
from .severity import CLASS_LABELS, CRACK_ALLIGATOR, POTHOLE

ACTIONS = {
    "critical": "Patch within 48 hours",
    "high": "Patch within 7 days",
    "medium": "Add to this month's resurfacing cycle",
    "low": "Monitor on next survey",
}
CLUSTER_RADIUS_M = 60.0

# Deterioration is only claimed with enough evidence: at least two rides, far enough apart
# for change to be plausible, and a rise larger than ride-to-ride measurement noise.
WORSENING_MIN_SPAN_DAYS = 7
WORSENING_MIN_RISE = 10.0


def _fmt_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d %b")


def trend_status(d: dict) -> tuple[str, str]:
    """(status, human label). Status is one of: single, too_close, worsening, decreased, stable."""
    if d["n_obs"] < 2:
        return "single", "Seen on 1 ride — no history yet"
    span = d.get("span_days") or 0
    if span < WORSENING_MIN_SPAN_DAYS:
        return "too_close", f"Seen on {d['n_obs']} rides within {span:.0f} day(s) — too close together to judge change"
    rise = (d["severity"] or 0) - (d.get("first_severity") or 0)
    if rise >= WORSENING_MIN_RISE:
        return "worsening", f"Severity rose {rise:+.0f} pts over {span:.0f} days ({d['n_obs']} rides)"
    if rise <= -WORSENING_MIN_RISE:
        return "decreased", f"Severity fell {rise:+.0f} pts over {span:.0f} days — check for partial repair"
    return "stable", f"No significant change over {span:.0f} days ({d['n_obs']} rides)"


def score_defects(defects: list[dict]) -> list[dict]:
    active = [d for d in defects if d["status"] == "active"]
    out = []
    for d in defects:
        reasons, parts = [], []
        sev = d["severity"] or 0
        parts.append(("Severity index × 0.55", 0.55 * sev))
        reasons.append(f"{d['level'].capitalize()} severity index ({sev:.0f}/100)")

        if d["cls"] == POTHOLE:
            parts.append(("Type: pothole", 10.0))
        elif d["cls"] == CRACK_ALLIGATOR:
            parts.append(("Type: alligator cracking", 6.0))
            reasons.append("Alligator cracking — often precedes potholes")

        status, trend_label = trend_status(d)
        trend = d["trend_per_month"] or 0
        if status == "worsening" and trend > 0:
            parts.append(("Growth (evidence-backed)", min(trend, 30) / 30 * 20))
            reasons.append(trend_label)

        parts.append((f"Confirmations ({min(d['n_obs'], 4)} ride(s))", min(d["n_obs"], 4) / 4 * 10))
        if d["n_obs"] >= 2:
            reasons.append(f"Detected on {d['n_obs']} separate rides")

        neighbours = sum(
            1 for o in active
            if o["id"] != d["id"] and o.get("source") == d.get("source")
            and haversine_m(d["lat"], d["lon"], o["lat"], o["lon"]) <= CLUSTER_RADIUS_M
        )
        if neighbours:
            parts.append((f"Nearby defects ({neighbours} within {CLUSTER_RADIUS_M:.0f} m)", float(min(neighbours, 5))))
            if neighbours >= 2:
                reasons.append(f"Part of a damaged stretch ({neighbours + 1} defects within {CLUSTER_RADIUS_M:.0f} m)")

        score = sum(p for _, p in parts)
        if d["status"] == "possibly_repaired":
            parts.append(("Possibly repaired × 0.2", score * 0.2 - score))
            score *= 0.2
            reasons.insert(0, f"Not detected on {d['missed_passes']} later pass(es) — verify repair")
            action = "Verify on site; close if repaired"
        else:
            action = ACTIONS[d["level"]]
            if d["missed_passes"]:
                reasons.append(f"Not detected on {d['missed_passes']} later pass — re-check")

        out.append({
            **d,
            "code": f"D-{d['id']:04d}",
            "label": CLASS_LABELS.get(d["cls"], d["cls"]),
            "priority": round(min(score, 100), 1),
            "breakdown": [{"label": k, "points": round(v, 1)} for k, v in parts],
            "trend_status": status,
            "trend_label": trend_label,
            "action": action,
            "reasons": reasons,
            "maps_url": f"https://www.google.com/maps?q={d['lat']:.6f},{d['lon']:.6f}",
        })
    out.sort(key=lambda d: d["priority"], reverse=True)
    for i, d in enumerate(out, 1):
        d["rank"] = i
    return out
