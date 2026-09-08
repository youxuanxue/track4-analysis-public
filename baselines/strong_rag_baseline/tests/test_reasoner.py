"""Quantity, identity, and cutoff contracts across the public input shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.agent import run_entity_grounded
from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.indexer import build_index
from baselines.strong_rag_baseline.quantities import TargetSpec
from baselines.strong_rag_baseline.reasoner import (
    _label,
    extract_numbers,
    number_in_text,
)
from baselines.strong_rag_baseline.retriever import BM25Index

REPO = Path(__file__).resolve().parents[3]
UNITS = sorted(p for p in (REPO / "units").iterdir() if (p / "task.json").is_file())


@pytest.mark.parametrize("baseline", [0, -2.5, 4.25])
def test_equal_numeric_comparison_uses_flat_when_it_is_legal(baseline):
    task = {
        "target": {
            "name": "revision",
            "type": "classification",
            "labels": ["up", "down", "flat"],
        }
    }
    entity = {"latest_precutoff_estimate": baseline}
    spec = TargetSpec.from_task(task, entity)
    assert _label(spec, entity, baseline, "") == "flat"
    assert _label(spec, entity, baseline + 1, "") == "up"
    assert _label(spec, entity, baseline - 1, "") == "down"
    task["target"]["labels"] = ["up", "down"]
    assert _label(TargetSpec.from_task(task, entity), entity, baseline, "") == "down"


@pytest.mark.parametrize("unit", UNITS, ids=[p.name for p in UNITS])
def test_public_predictions_obey_quantity_and_evidence_contracts(
    unit: Path, tmp_path: Path
) -> None:
    task = json.loads((unit / "task.json").read_text())
    answer = run(
        unit / "task.json",
        unit / "corpus",
        tmp_path / "answer.json",
        None,
        10,
        grounded=True,
    )
    assert [p["entity_id"] for p in answer["entity_predictions"]] == [
        e["entity_id"] for e in task["entities"]
    ]
    corpus = build_index(unit / "corpus")
    for entity, prediction in zip(
        task["entities"], answer["entity_predictions"], strict=True
    ):
        TargetSpec.from_task(task, entity).validate_prediction(prediction)
        assert "rank" not in prediction
        assert (
            "uncalibrated" in answer["notes"]["fallback_rationale"][entity["entity_id"]]
        )
        assert prediction["claims"]
        for claim in prediction["claims"]:
            doc_id = claim["doc_id"]
            assert corpus.doc_dates[doc_id] <= task["cutoff_date"]
            assert (
                0
                <= claim["span_start"]
                < claim["span_end"]
                <= len(corpus.doc_texts[doc_id])
            )
            assert (
                claim["claim"]
                == corpus.doc_texts[doc_id][
                    claim["span_start"] : claim["span_end"]
                ].strip()
            )
            if entity.get("cik") and "EDGAR" in doc_id:
                assert str(entity["cik"]).zfill(10) in doc_id


def test_extract_numbers_preserves_numeric_sign_and_scale() -> None:
    assert extract_numbers("first print +0.18%; range -0.06% to +0.44%") == [
        0.18,
        -0.06,
        0.44,
    ]
    assert number_in_text("net position was +296,204 contracts", 296204.0)
    assert not number_in_text("net position was 1296204 contracts", 296204.0)


def test_public_task_id_does_not_override_changed_cutoff() -> None:
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text())
    task["cutoff_date"] = "2024-02-01"
    corpus = build_index(unit / "corpus")
    result = run_entity_grounded(
        task,
        task["entities"][0],
        BM25Index(corpus.chunks, task["cutoff_date"]),
        corpus,
        10,
    )
    for claim in result.prediction["claims"]:
        assert corpus.doc_dates[claim["doc_id"]] <= task["cutoff_date"]
    assert all(
        c["doc_id"] != "EDGAR_0000320193_10Q_20240202"
        for c in result.prediction["claims"]
    )


def test_task_id_cannot_change_predictions() -> None:
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text())
    corpus = build_index(unit / "corpus")
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    first = run_entity_grounded(task, task["entities"][0], index, corpus, 10)
    task["task_id"] = "unseen-renamed-input"
    second = run_entity_grounded(task, task["entities"][0], index, corpus, 10)
    assert first.prediction == second.prediction


def test_credit_probabilities_do_not_copy_accounting_numbers(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-credit-event-2023"
    answer = run(
        unit / "task.json",
        unit / "corpus",
        tmp_path / "answer.json",
        None,
        10,
        grounded=True,
    )
    for pred in answer["entity_predictions"]:
        assert (
            0
            <= pred["interval"]["lo"]
            <= pred["point_forecast"]
            <= pred["interval"]["hi"]
            <= 1
        )


def test_unavailable_future_yields_use_documented_zero_change_baseline(
    tmp_path: Path,
) -> None:
    unit = REPO / "units" / "t4-fomc-curve-20240918"
    answer = run(
        unit / "task.json",
        unit / "corpus",
        tmp_path / "answer.json",
        None,
        10,
        grounded=True,
    )
    for pred in answer["entity_predictions"]:
        assert pred["point_forecast"] == 0.0
        assert "fallback" in answer["notes"]["fallback_rationale"][pred["entity_id"]]


def test_ranking_scores_use_changes_in_percent_of_open_interest(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-cotpos-202411-us10"
    answer = run(
        unit / "task.json",
        unit / "corpus",
        tmp_path / "answer.json",
        None,
        10,
        grounded=True,
    )
    task = json.loads((unit / "task.json").read_text())
    for entity, pred in zip(
        task["entities"], answer["entity_predictions"], strict=True
    ):
        assert pred["point_forecast"] == entity["trailing_4wk_net_change_pct_oi"]
        assert pred["point_forecast"] != entity["net_noncommercial_20241022"]
