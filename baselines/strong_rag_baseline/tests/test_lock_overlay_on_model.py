"""Public-unit locks override a local-model reply that would move a locked row."""
from __future__ import annotations

import json
from pathlib import Path

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.client import MockModelClient
from baselines.strong_rag_baseline.locks import load_official_lock

EXAMPLE_UNIT = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"


def test_model_reply_cannot_move_a_locked_public_row(tmp_path: Path) -> None:
    lock = load_official_lock()["units"]["t4-EXAMPLE-eps-beat"][0]
    # A confident-but-wrong model: different label, point, interval, and span.
    reply = json.dumps(
        {
            "label": "miss",
            "point_forecast": 0.01,
            "interval": {"level": 0.90, "lo": 0.0, "hi": 0.02},
            "evidence": [
                {
                    "doc_id": "EDGAR_0000320193_8K_20240201",
                    "quote": "Apple",
                    "claim": "A quote that would otherwise be cited.",
                }
            ],
        }
    )
    answer = run(
        task_path=EXAMPLE_UNIT / "task.json",
        corpus_dir=EXAMPLE_UNIT / "corpus",
        out_path=tmp_path / "answer.json",
        client=MockModelClient(reply=reply),
        top_k=5,
        grounded=False,
    )
    pred = answer["entity_predictions"][0]
    assert pred["label"] == lock["label"]
    assert pred["point_forecast"] == lock["point_forecast"]
    assert pred["interval"]["lo"] == lock["interval"]["lo"]
    assert pred["interval"]["hi"] == lock["interval"]["hi"]
    got = [
        {
            "doc_id": c["doc_id"],
            "span_start": c["span_start"],
            "span_end": c["span_end"],
        }
        for c in pred["claims"]
    ]
    assert got == lock["claims"]
    assert pred["claims"][0]["claim"]
