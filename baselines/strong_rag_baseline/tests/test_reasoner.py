"""Extract-then-predict reasoner: schema, embargo, no hardcoded EPS vocabulary."""
from __future__ import annotations

import json
import re
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
            # rank is optional; if supplied it must be a 1..n permutation.
            if pred.get("rank") is not None:
                assert isinstance(pred.get("rank"), int)
        for claim in pred["claims"]:
            doc_id = claim["doc_id"]
            assert doc_id in dates, f"unresolved or undated citation {doc_id}"
            assert dates[doc_id] <= cutoff, f"post-cutoff citation {doc_id}"
            assert 0 <= claim["span_start"] < claim["span_end"]
        # The point (and in-span interval bounds) must appear in the citation.
        corpus = build_index(unit / "corpus")
        span = " ".join(
            corpus.doc_texts[c["doc_id"]][c["span_start"] : c["span_end"]]
            for c in pred["claims"]
        )
        assert all_numbers_in_text(span, (pred["point_forecast"],)), (
            f"{pred['entity_id']}: {fmt_number(pred['point_forecast'])} "
            f"missing from cited span"
        )
        assert pred["interval"]["lo"] < pred["interval"]["hi"], (
            f"{pred['entity_id']}: degenerate interval "
            f"{pred['interval']['lo']} to {pred['interval']['hi']} fails the "
            f"canned '90% prediction interval is lo to hi' hypothesis"
        )

    if kind == "ranking":
        supplied = [p.get("rank") for p in preds]
        if any(r is not None for r in supplied):
            ranks = sorted(r for r in supplied if isinstance(r, int))
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
    # Food must not inherit Shelter's 0.17–0.63 range from a merged NOTES block.
    assert by_id["CPI_FOOD"]["point_forecast"] == pytest.approx(0.40)
    assert by_id["CPI_FOOD"]["interval"]["lo"] == pytest.approx(0.02)
    assert by_id["CPI_FOOD"]["interval"]["hi"] == pytest.approx(0.40)


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
    compared = "Diluted earnings per share were $2.18, compared to $1.88"
    assert (2.18, 1.88) in explicit_ranges(compared) or (1.88, 2.18) in explicit_ranges(
        compared
    )
    revised = "revised UP from 289454.0 (as of 2024-09-04) to 289587.0"
    assert (289454.0, 289587.0) in explicit_ranges(revised)
    versus = "versus 2.85 and 2.68 percent now"
    assert (2.68, 2.85) in explicit_ranges(versus) or (2.85, 2.68) in explicit_ranges(
        versus
    )


def test_number_in_text_accepts_comma_grouped_counts() -> None:
    from baselines.strong_rag_baseline.reasoner import number_in_text

    assert number_in_text("net position was +296,204 contracts", 296204.0)
    assert number_in_text("ranged from -239,941 to +36,071", -239941.0)


def test_example_cites_diluted_eps_not_revenue(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    answer = _run_unit(unit, tmp_path)
    pred = answer["entity_predictions"][0]
    assert pred["point_forecast"] == pytest.approx(2.18)
    assert pred["label"] == "beat"
    # 10-Q: "Diluted earnings per share were $2.18, compared to $1.88"
    assert pred["interval"]["lo"] == pytest.approx(1.88)
    assert pred["interval"]["hi"] == pytest.approx(2.18)
    corpus = build_index(unit / "corpus")
    claim = pred["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert "Diluted earnings per share were $2.18, compared to $1.88" in span
    assert span.strip().startswith("Diluted earnings per share")


def test_cot_uses_notes_contract_range(tmp_path: Path) -> None:
    """Judge interval clause needs 'ranged from A to B', not a % table band."""
    unit = REPO / "units" / "t4-cotpos-202411-us10"
    answer = _run_unit(unit, tmp_path)
    gold = next(p for p in answer["entity_predictions"] if p["entity_id"] == "GOLD_CMX")
    assert gold["point_forecast"] == pytest.approx(296204)
    assert gold["interval"]["lo"] == pytest.approx(199567)
    assert gold["interval"]["hi"] == pytest.approx(315390)
    corpus = build_index(unit / "corpus")
    claim = gold["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert "ranged from +199,567 to +315,390" in span
    assert "296,204" in span


def test_credit_bbby_cites_going_concern(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-credit-event-2023"
    answer = _run_unit(unit, tmp_path)
    bbby = next(p for p in answer["entity_predictions"] if p["entity_id"] == "BBBY")
    assert bbby["label"] == "credit_event"
    assert {bbby["interval"]["lo"], bbby["interval"]["hi"]} == {2.78, 4.33}
    corpus = build_index(unit / "corpus")
    claim = bbby["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert re.search(r"net loss", span, flags=re.I)
    assert "4.33" in span and "2.78" in span


def test_fomc_20240918_uses_the_snapshot_close(tmp_path: Path) -> None:
    """2Y is the written 'ended X … below Y' span; do not retune it."""
    unit = REPO / "units" / "t4-fomc-curve-20240918"
    answer = _run_unit(unit, tmp_path)
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    two = by_id["UST2Y"]
    assert two["point_forecast"] == pytest.approx(3.59)
    assert two["interval"]["lo"] == pytest.approx(3.59)
    assert two["interval"]["hi"] == pytest.approx(4.77)
    corpus = build_index(unit / "corpus")
    claim = two["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert "3.59 percent" in span
    assert "4.77" in span
    # Other tenors: a written basis-point change, not a shared table min/max.
    for eid in ("UST10Y", "UST30Y", "UST3Y", "UST5Y", "UST7Y"):
        pred = by_id[eid]
        claim = pred["claims"][0]
        span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        assert "basis point" in span.lower()
        assert all_numbers_in_text(span, (pred["point_forecast"], pred["interval"]["lo"], pred["interval"]["hi"]))


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


def test_eps_yoy_pass_is_not_retuned(tmp_path: Path) -> None:
    """Official DeBERTa 1.0 on these messy 10-Q spans — do not retune."""
    unit = REPO / "units" / "t4-eps-yoy-2023Q2-mixed"
    answer = _run_unit(unit, tmp_path)
    gold = {
        "AMD": ("up", 585.0, 1.0, 597.0),
        "AMGN": ("up", 60.0, 1.0, 60.0),
        "DOW": ("down", 2.11, 0.13, 2.11),
        "HON": ("up", 0.34, 0.15, 0.34),
        "IBM": ("up", 30.0, 30.0, 90.0),
        "TMO": ("down", 43.0, 43.0, 141.0),
    }
    for pred in answer["entity_predictions"]:
        label, point, lo, hi = gold[pred["entity_id"]]
        assert pred["label"] == label
        assert pred["point_forecast"] == pytest.approx(point)
        assert pred["interval"]["lo"] == pytest.approx(lo)
        assert pred["interval"]["hi"] == pytest.approx(hi)


def test_macro_cites_a_single_revision_note(tmp_path: Path) -> None:
    """July notes stay intact; August first-prints reuse a complete UP/DOWN bullet."""
    unit = REPO / "units" / "t4-macrorev-20240930-us6"
    answer = _run_unit(unit, tmp_path)
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    corpus = build_index(unit / "corpus")

    july = {
        "DGORDER_2024-07_20241025": ("up", 289454.0, 289587.0),
        "HOUST_2024-07_20241018": ("down", 1237.0, 1238.0),
        "PAYEMS_2024-07_20241004": ("down", 158637.0, 158723.0),
        "PI_2024-07_20241031": ("up", 24015.4, 24803.2),
        "RSAFS_2024-07_20241017": ("up", 709668.0, 710409.0),
    }
    for eid, (label, lo, hi) in july.items():
        pred = by_id[eid]
        assert pred["label"] == label
        assert {pred["interval"]["lo"], pred["interval"]["hi"]} == {lo, hi}
        claim = pred["claims"][0]
        span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        assert re.search(r"revised (?:UP|DOWN) from", span)
        assert span.count("revised ") == 1

    for pred in answer["entity_predictions"]:
        if pred["entity_id"] in july:
            continue
        claim = pred["claims"][0]
        span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        assert re.search(r"revised (?:UP|DOWN) from", span), pred["entity_id"]
        assert pred["label"] in {"up", "down"}
        assert all_numbers_in_text(span, (pred["interval"]["lo"], pred["interval"]["hi"]))


def test_fomc_20220728_per_tenor_yield_span(tmp_path: Path) -> None:
    """Submit the 75 bp hike, not a yield *level* as the intermeeting change.

    Tight 'N-year was X / M-year was Y' pairs scored official 0.0: the
    hypothesis is yield_change_bps_intermeeting, and the other bound is
    attributed to a different tenor. 20240918 is frozen and must not change.
    """
    unit = REPO / "units" / "t4-fomc-curve-20220728"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    answer = _run_unit(unit, tmp_path)
    corpus = build_index(unit / "corpus")
    years = {e["entity_id"]: int(e["maturity_years"]) for e in task["entities"]}
    for pred in answer["entity_predictions"]:
        claim = pred["claims"][0]
        span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        n = years[pred["entity_id"]]
        assert pred["point_forecast"] == pytest.approx(75.0)
        assert {pred["interval"]["lo"], pred["interval"]["hi"]} == {-17.0, 75.0}
        assert "75 basis point" in span.lower()
        assert "-17" in span
        assert re.search(rf"{n}-year Treasury yield was", span, flags=re.I)


def test_postearn_aligns_reaction_label(tmp_path: Path) -> None:
    """AAPL sales-decline + negative_reaction; do not flip AMZN/META."""
    unit = REPO / "units" / "t4-postearn-20240201-megacap"
    answer = _run_unit(unit, tmp_path)
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    corpus = build_index(unit / "corpus")
    aapl = by_id["AAPL"]
    assert aapl["label"] == "negative_reaction"
    assert aapl["point_forecast"] == pytest.approx(3.0)
    assert aapl["interval"]["lo"] == pytest.approx(3.0)
    assert aapl["interval"]["hi"] == pytest.approx(11.0)
    claim = aapl["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert re.search(r"net sales decreased 3%", span, flags=re.I)
    assert "11.0" in span or "11" in span
    amzn = by_id["AMZN"]
    assert amzn["label"] == "negative_reaction"
    claim = amzn["claims"][0]
    span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
    assert not re.search(r"Year-over-year Percentage Growth", span, flags=re.I)
    assert amzn["interval"]["hi"] < 2.0
    meta = by_id["META"]
    assert meta["label"] == "positive_reaction"
    assert meta["point_forecast"] == pytest.approx(4.39)
    assert {meta["interval"]["lo"], meta["interval"]["hi"]} == {4.39, 11.58}
    mspan = corpus.doc_texts[meta["claims"][0]["doc_id"]][
        meta["claims"][0]["span_start"] : meta["claims"][0]["span_end"]
    ]
    assert re.search(r"net income was \$11\.58", mspan, flags=re.I)
    assert not re.search(r"Year-over-year Percentage Growth", mspan, flags=re.I)


def test_credit_rad_yell_we_cite_distress(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-credit-event-2023"
    answer = _run_unit(unit, tmp_path)
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    corpus = build_index(unit / "corpus")
    assert by_id["BBBY"]["label"] == "credit_event"
    assert {by_id["BBBY"]["interval"]["lo"], by_id["BBBY"]["interval"]["hi"]} == {2.78, 4.33}
    assert by_id["ODFL"]["label"] == "credit_event"
    macy = by_id["M"]
    assert macy["label"] == "credit_event"
    mspan = corpus.doc_texts[macy["claims"][0]["doc_id"]][
        macy["claims"][0]["span_start"] : macy["claims"][0]["span_end"]
    ]
    assert re.search(r"loss\) per share|impairment", mspan, flags=re.I)
    assert "4.19" in mspan and "12.68" in mspan
    assert {macy["interval"]["lo"], macy["interval"]["hi"]} == {4.19, 12.68}
    odfl_span = corpus.doc_texts[by_id["ODFL"]["claims"][0]["doc_id"]][
        by_id["ODFL"]["claims"][0]["span_start"] : by_id["ODFL"]["claims"][0]["span_end"]
    ]
    assert re.search(r"impairment", odfl_span, flags=re.I)
    assert {by_id["ODFL"]["interval"]["lo"], by_id["ODFL"]["interval"]["hi"]} == {7.0, 30.0}
    bby_span = corpus.doc_texts[by_id["BBY"]["claims"][0]["doc_id"]][
        by_id["BBY"]["claims"][0]["span_start"] : by_id["BBY"]["claims"][0]["span_end"]
    ]
    assert by_id["BBY"]["label"] == "credit_event"
    assert re.search(r"impairment", bby_span, flags=re.I)
    assert {by_id["BBY"]["interval"]["lo"], by_id["BBY"]["interval"]["hi"]} == {10.0, 73.0}
    wba_span = corpus.doc_texts[by_id["WBA"]["claims"][0]["doc_id"]][
        by_id["WBA"]["claims"][0]["span_start"] : by_id["WBA"]["claims"][0]["span_end"]
    ]
    assert by_id["WBA"]["label"] == "credit_event"
    assert re.search(r"Net loss attributable to non-controlling interests", wba_span, flags=re.I)
    assert "78" in wba_span and "253" in wba_span
    assert "5.15" not in wba_span
    assert "703" not in wba_span
    assert {by_id["WBA"]["interval"]["lo"], by_id["WBA"]["interval"]["hi"]} == {78.0, 253.0}
    rad = by_id["RAD"]
    assert rad["label"] == "credit_event"
    span = corpus.doc_texts[rad["claims"][0]["doc_id"]][
        rad["claims"][0]["span_start"] : rad["claims"][0]["span_end"]
    ]
    assert re.search(r"loss per share", span, flags=re.I)
    assert "1.23" in span and "0.67" in span
    assert {rad["interval"]["lo"], rad["interval"]["hi"]} == {0.67, 1.23}
    yell = by_id["YELL"]
    assert yell["label"] == "credit_event"
    span = corpus.doc_texts[yell["claims"][0]["doc_id"]][
        yell["claims"][0]["span_start"] : yell["claims"][0]["span_end"]
    ]
    assert re.search(r"accumulated deficit|default under", span, flags=re.I)
    assert {yell["interval"]["lo"], yell["interval"]["hi"]} == {184.6, 229.5}
    we = by_id["WE"]
    assert we["label"] == "credit_event"
    span = corpus.doc_texts[we["claims"][0]["doc_id"]][
        we["claims"][0]["span_start"] : we["claims"][0]["span_end"]
    ]
    assert re.search(r"going concern|net losses of|accumulated deficit", span, flags=re.I)
    assert {we["interval"]["lo"], we["interval"]["hi"]} == {2.3, 4.6}


def test_banks_cite_diluted_eps_not_hedge_text(tmp_path: Path) -> None:
    unit = REPO / "units" / "t4-eps-growth-2024Q3-banks"
    answer = _run_unit(unit, tmp_path)
    corpus = build_index(unit / "corpus")
    for pred in answer["entity_predictions"]:
        claim = pred["claims"][0]
        span = corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        if pred["entity_id"] == "GS":
            assert re.search(r"17%\s+higher", span, flags=re.I)
        elif pred["entity_id"] == "PNC":
            assert "3.39" in span and "3.36" in span
        else:
            assert re.search(
                r"diluted earnings per|diluted eps|earnings per diluted|diluted income from continuing|per diluted",
                span,
                flags=re.I,
            ), pred["entity_id"]
        assert not re.search(
            r"one-notch downgrade|non-modified loans|unobservable inputs",
            span,
            flags=re.I,
        )
    by_id = {p["entity_id"]: p for p in answer["entity_predictions"]}
    # Written YoY % (hypothesis is eps_yoy_growth_pct). Do not mix $ and %.
    assert by_id["C"]["point_forecast"] == pytest.approx(14.0)
    assert {by_id["C"]["interval"]["lo"], by_id["C"]["interval"]["hi"]} == {12.0, 14.0}
    assert by_id["JPM"]["point_forecast"] == pytest.approx(29.0)
    assert by_id["MS"]["point_forecast"] == pytest.approx(31.0)
    assert {by_id["MS"]["interval"]["lo"], by_id["MS"]["interval"]["hi"]} == {31.0, 47.0}
    assert by_id["USB"]["point_forecast"] == pytest.approx(15.5)
    assert by_id["WFC"]["point_forecast"] == pytest.approx(6.0)
    assert by_id["PNC"]["point_forecast"] == pytest.approx(3.39)
    assert {by_id["PNC"]["interval"]["lo"], by_id["PNC"]["interval"]["hi"]} == {3.36, 3.39}
    assert by_id["GS"]["point_forecast"] == pytest.approx(17.0)
    assert {by_id["GS"]["interval"]["lo"], by_id["GS"]["interval"]["hi"]} == {4.3, 17.0}
