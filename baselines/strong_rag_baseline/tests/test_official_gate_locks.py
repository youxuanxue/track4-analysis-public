"""Pin the extract-then-predict answers official DeBERTa scored 11/11.

The snapshot lives in ``locks/official_gate_d4d0584.json``. A future reasoner
patch that changes a label, interval, point, cited ``doc_id``, or span offsets
must fail here — do not retune a PASS to chase predictive quality on this PR.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.cli import run

REPO = Path(__file__).resolve().parents[3]
LOCK_PATH = Path(__file__).resolve().parent / "locks" / "official_gate_d4d0584.json"
UNITS = sorted(p for p in (REPO / "units").iterdir() if (p / "task.json").is_file())


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_lock_covers_every_public_unit() -> None:
    locked = set(_lock()["units"])
    shipped = {p.name for p in UNITS}
    assert locked == shipped, (
        f"lock/units mismatch: extra={sorted(locked - shipped)} "
        f"missing={sorted(shipped - locked)}"
    )


@pytest.mark.parametrize("unit", UNITS, ids=[p.name for p in UNITS])
def test_official_gate_snapshot_is_unchanged(unit: Path, tmp_path: Path) -> None:
    expected_rows = _lock()["units"][unit.name]
    answer = run(
        task_path=unit / "task.json",
        corpus_dir=unit / "corpus",
        out_path=tmp_path / f"{unit.name}.json",
        client=None,
        top_k=10,
        grounded=True,
    )
    got = answer["entity_predictions"]
    assert [p["entity_id"] for p in got] == [r["entity_id"] for r in expected_rows]
    for pred, exp in zip(got, expected_rows, strict=True):
        eid = exp["entity_id"]
        if "label" in exp:
            assert pred.get("label") == exp["label"], f"{unit.name}/{eid} label"
        assert pred["point_forecast"] == pytest.approx(exp["point_forecast"]), (
            f"{unit.name}/{eid} point_forecast"
        )
        assert pred["interval"]["lo"] == pytest.approx(exp["interval"]["lo"]), (
            f"{unit.name}/{eid} interval.lo"
        )
        assert pred["interval"]["hi"] == pytest.approx(exp["interval"]["hi"]), (
            f"{unit.name}/{eid} interval.hi"
        )
        got_claims = [
            {
                "doc_id": c["doc_id"],
                "span_start": c["span_start"],
                "span_end": c["span_end"],
            }
            for c in pred["claims"]
        ]
        assert got_claims == exp["claims"], f"{unit.name}/{eid} claims"
