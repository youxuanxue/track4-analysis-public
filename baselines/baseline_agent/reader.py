"""Rule-based reader/reasoner for the minimal Track 4 baseline.

Replaces the open-weights LLM of the full baseline spec with a transparent rule:
for the EPS-beat family it compares a corpus-extracted reported EPS against the
entity's ``consensus_eps`` using the declared threshold; for other families it
falls back to the most common neutral label. The point forecast and a calibrated
interval are derived from the extracted value.
"""
from __future__ import annotations

import re

_EPS_RE = re.compile(r"earnings per share of \$?([0-9]+\.[0-9]+)", re.IGNORECASE)


def extract_eps(text: str) -> float | None:
    m = _EPS_RE.search(text)
    return float(m.group(1)) if m else None


def classify_eps(reported: float, consensus: float, threshold_pct: float) -> str:
    if reported > consensus * (1 + threshold_pct):
        return "beat"
    if reported < consensus * (1 - threshold_pct):
        return "miss"
    return "inline"


def predict_entity(entity: dict, span_text: str) -> dict:
    """Return {label, point_forecast, lo, hi} for one entity from its top span."""
    consensus = entity.get("consensus_eps")
    threshold = entity.get("threshold_pct", 0.05)
    reported = extract_eps(span_text)
    if consensus is not None and reported is not None:
        label = classify_eps(reported, float(consensus), float(threshold))
        point = reported
    else:
        # No usable evidence — predict the neutral class with a wide interval.
        label = "inline"
        point = float(consensus) if consensus is not None else 0.0
    # Simple symmetric interval: +/- one threshold band around the point forecast.
    band = max(abs(point) * float(threshold) * 2.0, 0.05)
    return {"label": label, "point_forecast": point, "lo": point - band, "hi": point + band}
