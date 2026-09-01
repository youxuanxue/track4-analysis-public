"""Extract-then-predict reasoner: schema, embargo, no hardcoded EPS vocabulary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.indexer import build_index
from baselines.strong_rag_baseline.reasoner import (
    all_numbers_in_text,
    explicit_ranges,
    extract_numbers,
)
from baselines.strong_rag_baseline.schema import fmt_number, legal_labels, target_type

REPO = Path(__file__).resolve().parents[3]
UNITS = sorted(p for p in (REPO / "units").iterdir() if (p / "task.json").is_file())


def _run_unit(unit: Path, tmp_path: Path) -> dict:
    return run(
        task_path=unit / "task.json",
        corpus_dir=unit / "corpus",
        out_path=tmp_path / f"{unit.name}.json",
        client=None,
        top_k=10,
        grounded=True,
    )


@pytest.mark.parametrize("unit", UNITS, ids=[p.name for p in UNITS])
def test_grounded_run_is_schema_valid_and_embargo_safe(unit: Path, tmp_path: Path) -> None:
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    answer = _run_unit(unit, tmp_path)
    kind = target_type(task)
    labels = legal_labels(task)
    cutoff = task["cutoff_date"]
    roster = [e["entity_id"] for e in task["entities"]]

    assert answer["task_id"] == task["task_id"]
    if kind is not None:
        assert answer["target_type"] == kind
    preds = answer["entity_predictions"]
    assert [p["entity_id"] for p in preds] == roster

    dates: dict[str, str] = {}
    for path in (unit / "corpus").glob("*.json"):
        if path.name == "manifest.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        dates[doc.get("doc_id", path.stem)] = doc["doc_date"]

    for pred in preds:
        interval = pred["interval"]
        assert interval["level"] == pytest.approx(float(task.get("interval_level", 0.90)))
        assert interval["lo"] <= interval["hi"]
        assert pred["claims"], f"{pred['entity_id']} has no claims"
        if kind == "classification":
            assert pred["label"] in labels
        assert pred["point_forecast"] is not None
        if kind == "ranking":
            assert isinstance(pred.get("rank"), int)
        for claim in pred["claims"]:
            doc_id = claim["doc_id"]
            assert doc_id in dates, f"unresolved or undated citation {doc_id}"
            assert dates[doc_id] <= cutoff, f"post-cutoff citation {doc_id}"
            assert 0 <= claim["span_start"] < claim["span_end"]
        # The NLI hypothesis is built from these numbers; they must be in the span.
        corpus = build_index(unit / "corpus")
        span = " ".join(
            corpus.doc_texts[c["doc_id"]][c["span_start"] : c["span_end"]]
            for c in pred["claims"]
        )
        assert all_numbers_in_text(
            span, (pred["point_forecast"], pred["interval"]["lo"], pred["interval"]["hi"])
        ), (
            f"{pred['entity_id']}: {fmt_number(pred['point_forecast'])} / "
            f"{fmt_number(pred['interval']['lo'])} / {fmt_number(pred['interval']['hi'])} "
            f"missing from cited span"
        )

    if kind == "ranking":
        ranks = sorted(p["rank"] for p in preds)
        assert ranks == list(range(1, len(preds) + 1))


def test_labels_come_from_the_task_not_eps_vocab(tmp_path: Path) -> None:
    credit = REPO / "units" / "t4-credit-event-2023"
    task = json.loads((credit / "task.json").read_text(encoding="utf-8"))
    answer = _run_unit(credit, tmp_path)
    allowed = set(legal_labels(task))
    assert allowed == {"credit_event", "no_event"}
    for pred in answer["entity_predictions"]:
        assert pred["label"] in allowed
        assert pred["label"] not in {"beat", "miss", "inline"}


def test_cpi_notes_line_is_preferred_over_the_header(tmp_path: Path) -> None:
    """Apparel's first print is +1.14, not a year or a header token."""
    unit = REPO / "units" / "t4-cpicomp-202410-us11"
    answer = _run_unit(unit, tmp_path)
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    assert by_id["CPI_APPAREL"]["point_forecast"] == pytest.approx(1.14)
    assert by_id["CPI_ENERGY"]["point_forecast"] == pytest.approx(-1.85)
    assert by_id["CPI_ALLITEMS"]["point_forecast"] == pytest.approx(0.18)


def test_regression_units_do_not_require_a_class_label(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-cpicomp-202410-us11"
    answer = _run_unit(unit, tmp_path)
    assert answer["target_type"] == "regression"
    for pred in answer["entity_predictions"]:
        assert isinstance(pred["point_forecast"], (int, float))
        assert pred["claims"]


def test_extract_numbers_keeps_sign_and_decimals() -> None:
    assert extract_numbers("first print +0.18%; range -0.06% to +0.44%") == [
        0.18,
        -0.06,
        0.44,
    ]
    assert extract_numbers("bid-to-cover of 2.48") == [2.48]


def test_explicit_ranges_read_notes_language() -> None:
    text = "the bid-to-cover ratio ranged from 2.32 to 2.67; 2024 range -0.06% to +0.44%"
    assert (2.32, 2.67) in explicit_ranges(text)
    assert (-0.06, 0.44) in explicit_ranges(text)


def test_example_cites_diluted_eps_not_revenue(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    answer = _run_unit(unit, tmp_path)
    pred = answer["entity_predictions"][0]
    assert pred["point_forecast"] == pytest.approx(2.18)
    assert pred["label"] == "beat"


def test_fomc_20240918_uses_the_snapshot_close(tmp_path: Path) -> None:
    """Each maturity's start_yield_pct is written in the rates table; cite it."""
    unit = REPO / "units" / "t4-fomc-curve-20240918"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    answer = _run_unit(unit, tmp_path)
    expected = {e["entity_id"]: e["start_yield_pct"] for e in task["entities"]}
    for pred in answer["entity_predictions"]:
        assert pred["point_forecast"] == pytest.approx(expected[pred["entity_id"]])


@pytest.mark.parametrize(
    "unit_name",
    ["t4-postearn-20240201-megacap", "t4-credit-event-2023", "t4-eps-yoy-2023Q2-mixed"],
)
def test_cik_entities_cite_only_their_own_filings(unit_name: str, tmp_path: Path) -> None:
    """A 10-Q must not be reused as every row's evidence; ticker 'WE' is not 'we'."""
    unit = REPO / "units" / unit_name
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    answer = _run_unit(unit, tmp_path)
    cik_by_id = {e["entity_id"]: str(e["cik"]).zfill(10) for e in task["entities"]}
    for pred in answer["entity_predictions"]:
        cik = cik_by_id[pred["entity_id"]]
        for claim in pred["claims"]:
            assert cik in claim["doc_id"], (
                f"{pred['entity_id']} cited {claim['doc_id']} (expected CIK {cik})"
            )


def test_auction_points_are_bid_to_cover_ratios(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-auction-btc-202411-us7"
    answer = _run_unit(unit, tmp_path)
    for pred in answer["entity_predictions"]:
        assert 1.2 <= float(pred["point_forecast"]) <= 4.5
        assert 1.2 <= float(pred["interval"]["lo"]) <= 4.5
        assert 1.2 <= float(pred["interval"]["hi"]) <= 4.5


def test_grounded_run_is_deterministic(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    first = _run_unit(unit, tmp_path / "a")
    second = _run_unit(unit, tmp_path / "b")
    assert first == second
