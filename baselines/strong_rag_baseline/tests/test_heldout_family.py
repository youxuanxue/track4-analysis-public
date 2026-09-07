"""Unpublished families use the same quantity and embargo rules as public ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.indexer import build_index
from baselines.strong_rag_baseline.reasoner import _task_needles
from baselines.strong_rag_baseline.schema import legal_labels, target_type


def _write_unit(
    tmp_path: Path,
    *,
    task: dict,
    docs: list[dict],
) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    for doc in docs:
        (corpus / f"{doc['doc_id']}.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def _run(unit: Path, tmp_path: Path) -> dict:
    return run(
        task_path=unit / "task.json",
        corpus_dir=unit / "corpus",
        out_path=tmp_path / "answer.json",
        client=None,
        top_k=10,
        grounded=True,
    )


def test_unpublished_regression_uses_entity_field_and_written_range(
    tmp_path: Path,
) -> None:
    task = {
        "task_id": "t4-heldout-widget-spread",
        "schema_version": "3",
        "family": "widget_spread_nowcast",
        "target": {"name": "widget_spread_bps", "type": "regression"},
        "cutoff_date": "2024-06-01",
        "interval_level": 0.90,
        "entities": [
            {
                "entity_id": "WGT_A",
                "name": "Widget A",
                "latest_widget_print": 12.5,
            }
        ],
    }
    docs = [
        {
            "doc_id": "WIDGET_NOTES_202405",
            "doc_date": "2024-05-15",
            "text": (
                "NOTES (derived from the widget tape)\n"
                "- Widget A: first print +12.5; the spread ranged from 10.0 to 14.0.\n"
            ),
        },
        {
            "doc_id": "WIDGET_POST_CUTOFF",
            "doc_date": "2024-07-01",
            "text": "Widget A printed 99.0 after the cutoff and must not be cited.",
        },
    ]
    unit = _write_unit(tmp_path / "unit", task=task, docs=docs)
    answer = _run(unit, tmp_path / "out")
    assert answer["target_type"] == "regression"
    pred = answer["entity_predictions"][0]
    assert pred["entity_id"] == "WGT_A"
    assert pred["point_forecast"] == pytest.approx(12.5)
    assert pred["interval"]["lo"] <= 12.5 <= pred["interval"]["hi"]
    assert "uncalibrated" in answer["notes"]["fallback_rationale"][pred["entity_id"]]
    assert pred["claims"]
    assert all(c["doc_id"] != "WIDGET_POST_CUTOFF" for c in pred["claims"])
    corpus = build_index(unit / "corpus")
    claim = pred["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert "12.5" in span
    assert "10.0" in span and "14.0" in span


def test_unpublished_classification_reads_labels_from_the_task(
    tmp_path: Path,
) -> None:
    task = {
        "task_id": "t4-heldout-widget-direction",
        "schema_version": "3",
        "family": "widget_direction_shift",
        "target": {
            "name": "widget_direction",
            "type": "classification",
            "labels": ["widening", "tightening"],
        },
        "cutoff_date": "2024-06-01",
        "interval_level": 0.90,
        "entities": [
            {
                "entity_id": "WGT_B",
                "name": "Widget B",
                "prior_widget_print": 6.1,
            }
        ],
    }
    docs = [
        {
            "doc_id": "WIDGET_NOTE_B",
            "doc_date": "2024-05-20",
            "text": (
                "Widget B spread is widening; the print was 8.2 versus 6.1 "
                "in the prior tape."
            ),
        }
    ]
    unit = _write_unit(tmp_path / "unit", task=task, docs=docs)
    answer = _run(unit, tmp_path / "out")
    pred = answer["entity_predictions"][0]
    allowed = set(legal_labels(task))
    assert pred["label"] in allowed
    assert pred["label"] not in {"beat", "miss", "inline", "credit_event"}
    assert pred["label"] == "widening"
    assert pred["point_forecast"] in {6.1, 8.2}
    assert pred["interval"]["lo"] <= pred["point_forecast"] <= pred["interval"]["hi"]


def test_unpublished_ranking_uses_point_forecast_not_rank(
    tmp_path: Path,
) -> None:
    task = {
        "task_id": "t4-heldout-widget-rank",
        "schema_version": "3",
        "family": "widget_rank_shift",
        "target": {"name": "widget_flow_rank", "type": "ranking"},
        "cutoff_date": "2024-06-01",
        "interval_level": 0.90,
        "entities": [
            {"entity_id": "WGT_X", "name": "Widget X", "latest_widget_print": 3.0},
            {"entity_id": "WGT_Y", "name": "Widget Y", "latest_widget_print": 9.0},
        ],
    }
    docs = [
        {
            "doc_id": "WIDGET_RANK_NOTES",
            "doc_date": "2024-05-01",
            "text": (
                "NOTES (derived from flows)\n"
                "- Widget X flow: first print +3.0; ranged from 1.0 to 4.0.\n"
                "- Widget Y flow: first print +9.0; ranged from 7.0 to 11.0.\n"
            ),
        }
    ]
    unit = _write_unit(tmp_path / "unit", task=task, docs=docs)
    answer = _run(unit, tmp_path / "out")
    assert answer["target_type"] == "ranking"
    preds = answer["entity_predictions"]
    assert [p["entity_id"] for p in preds] == ["WGT_X", "WGT_Y"]
    assert all("rank" not in p for p in preds)
    by_id = {p["entity_id"]: p for p in preds}
    assert by_id["WGT_X"]["point_forecast"] == pytest.approx(3.0)
    assert by_id["WGT_Y"]["point_forecast"] == pytest.approx(9.0)
    assert by_id["WGT_Y"]["point_forecast"] > by_id["WGT_X"]["point_forecast"]


def test_family_slug_is_not_read(tmp_path: Path) -> None:
    """Two tasks that differ only in ``family`` must emit the same answer."""
    base = {
        "task_id": "t4-heldout-family-ignored",
        "schema_version": "3",
        "target": {"name": "widget_spread_bps", "type": "regression"},
        "cutoff_date": "2024-06-01",
        "interval_level": 0.90,
        "entities": [
            {"entity_id": "WGT_A", "name": "Widget A", "latest_widget_print": 12.5}
        ],
    }
    docs = [
        {
            "doc_id": "WIDGET_NOTES_202405",
            "doc_date": "2024-05-15",
            "text": (
                "NOTES (derived from the widget tape)\n"
                "- Widget A: first print +12.5; the spread ranged from 10.0 to 14.0.\n"
            ),
        }
    ]
    a = dict(base, family="alpha_unpublished")
    b = dict(base, family="zzz_totally_different")
    unit_a = _write_unit(tmp_path / "a", task=a, docs=docs)
    unit_b = _write_unit(tmp_path / "b", task=b, docs=docs)
    first = _run(unit_a, tmp_path / "out_a")
    second = _run(unit_b, tmp_path / "out_b")
    assert first["entity_predictions"] == second["entity_predictions"]


def test_task_needles_come_from_schema_not_family() -> None:
    task = {
        "family": "should_never_appear_as_a_needle",
        "target": {
            "name": "widget_spread_bps",
            "type": "classification",
            "labels": ["widening", "tightening"],
        },
        "entities": [{"entity_id": "WGT_A", "latest_widget_print": 1.0}],
    }
    needles = [n.lower() for n in _task_needles(task, task["entities"][0])]
    assert any("widget" in n or "spread" in n for n in needles)
    assert any("widening" in n for n in needles)
    assert all("should_never_appear" not in n for n in needles)


def test_nested_task_type_is_supported() -> None:
    assert target_type({"target": {"type": "regression"}}) == "regression"
