"""Synthetic semantic regressions, independent of public prediction snapshots."""

from __future__ import annotations

import json

import pytest

from baselines.strong_rag_baseline.agent import run_entity, run_entity_grounded
from baselines.strong_rag_baseline.client import MockModelClient
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.quantities import TargetSpec, finite_number


def predict(
    name: str,
    text: str,
    *,
    prompt: str = "",
    entity: dict | None = None,
    labels: list[str] | None = None,
    target: dict | None = None,
    resolution: str = "",
) -> dict:
    task = {
        "task_id": "synthetic-semantic-audit",
        "cutoff_date": "2024-06-01",
        "interval_level": 0.9,
        "prompt": prompt,
        "resolution_date": resolution,
        "target": dict(
            name=name,
            type="classification" if labels else "regression",
            **({"labels": labels} if labels else {}),
            **(target or {}),
        ),
    }
    entity = {"entity_id": "WGT", "name": "Widget A", **(entity or {})}
    chunk = Chunk("WIDGET_REPORT", "2024-05-01", 0, len(text), text)
    corpus = IndexedCorpus(
        [chunk], {chunk.doc_id: text}, {chunk.doc_id: chunk.doc_date}
    )
    result = run_entity_grounded(
        task, entity, BM25Index(corpus.chunks, task["cutoff_date"]), corpus, 10
    )
    return {**result.prediction, "rationale": result.rationale}


def test_derived_revenue_growth_is_not_dollar_amount() -> None:
    result = predict(
        "revenue_growth_pct",
        "Widget A revenue was 120 million compared to 100 million in the prior year.",
    )
    assert result["point_forecast"] == pytest.approx(20.0)


def test_probability_never_uses_lifetimes_or_losses() -> None:
    result = predict(
        "credit_event_12m",
        "Widget A net loss was 393 million. Estimated building lives are 7 to 30 years.",
        prompt="Predict the probability of a credit event (0 to 1).",
        labels=["credit_event", "no_event"],
    )
    assert 0 <= result["point_forecast"] <= 1
    assert 0 <= result["interval"]["lo"] <= result["interval"]["hi"] <= 1


def test_future_yield_change_does_not_copy_current_yield() -> None:
    result = predict(
        "yield_change_bps_intermeeting",
        "Widget A yield ended at 3.59 percent; the policy rate was cut by 50 basis points.",
        prompt="Predict the future yield change from the cutoff to the next meeting.",
        entity={"start_yield_pct": 3.59},
    )
    assert result["point_forecast"] == 0.0
    assert "fallback" in result["rationale"].lower()


def test_declared_yield_projection_converts_percent_to_bps() -> None:
    result = predict(
        "yield_change_bps_intermeeting",
        "Widget A projected yield is 3.84 percent.",
        entity={"start_yield_pct": 3.59},
    )
    assert result["point_forecast"] == pytest.approx(25.0)


def test_yield_projection_for_another_horizon_is_not_reused() -> None:
    result = predict(
        "yield_change_bps_intermeeting",
        "Widget A projected yield is 3.84 percent on 2025-12-31.",
        entity={"start_yield_pct": 3.59},
        resolution="2024-07-01",
    )
    assert result["point_forecast"] == 0.0
    assert "fallback" in result["rationale"]


def test_stock_return_does_not_copy_revenue_change() -> None:
    result = predict(
        "earnings_reaction",
        "Widget A net sales decreased 3% or 11 billion dollars.",
        prompt="Predict the abnormal stock return (%).",
        labels=["positive_reaction", "negative_reaction", "flat"],
    )
    assert result["point_forecast"] == 0.0
    assert result["label"] == "flat"


def test_unknown_target_does_not_use_unrelated_numeric_feature() -> None:
    result = predict(
        "widget_speed",
        "Widget A has 500 employees and revenue 120 million.",
        entity={"employees": 500},
    )
    assert result["point_forecast"] == 0.0
    assert "fallback" in result["rationale"].lower()


def test_unknown_target_does_not_copy_an_unrelated_print() -> None:
    result = predict("widget_speed", "Widget A revenue first print 500 million.")
    assert result["point_forecast"] == 0.0


def test_negative_growth_keeps_the_direction() -> None:
    result = predict(
        "revenue_growth_pct",
        "Widget A revenue was 80 million compared to 100 million in the prior year.",
    )
    assert result["point_forecast"] == pytest.approx(-20.0)


def test_growth_normalizes_million_and_billion_scales() -> None:
    result = predict(
        "revenue_growth_pct",
        "Widget A revenue was 1.2 billion compared to 1000 million in the prior year.",
    )
    assert result["point_forecast"] == pytest.approx(20.0)


def test_negative_diluted_eps_keeps_accounting_parentheses() -> None:
    result = predict("diluted_eps", "Widget A diluted earnings per share were $(1.23).")
    assert result["point_forecast"] == pytest.approx(-1.23)


def test_zero_prior_does_not_invent_an_infinite_growth_rate() -> None:
    result = predict(
        "revenue_growth_pct",
        "Widget A revenue was 120 million compared to 0 million in the prior year.",
    )
    assert result["point_forecast"] == 0.0
    assert "fallback" in result["rationale"]


@pytest.mark.parametrize(
    "eps,label", [(-1.0, "inline"), (-0.8, "beat"), (-1.2, "miss")]
)
def test_earnings_labels_handle_negative_consensus(eps, label) -> None:
    result = predict(
        "diluted_eps",
        f"Widget A diluted EPS was {eps}.",
        entity={"consensus_eps": -1.0},
        labels=["beat", "miss", "inline"],
    )
    assert result["label"] == label


def test_probability_percent_is_converted_to_fraction() -> None:
    result = predict(
        "credit_event_12m",
        "Widget A probability of a credit event is 25%.",
        labels=["credit_event", "no_event"],
    )
    assert result["point_forecast"] == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [393, float("nan"), float("inf"), True])
def test_shared_validator_rejects_invalid_probability(bad: float) -> None:
    spec = TargetSpec.from_task(
        {
            "target": {
                "name": "credit_event_12m",
                "type": "classification",
                "labels": ["credit_event", "no_event"],
            }
        }
    )
    with pytest.raises(ValueError):
        spec.validate_prediction(
            {
                "label": "credit_event",
                "point_forecast": bad,
                "interval": {"level": 0.9, "lo": 0, "hi": 1},
            }
        )


def test_task_contract_preserves_units_domain_and_horizon() -> None:
    spec = TargetSpec.from_task(
        {
            "cutoff_date": "2024-06-01",
            "resolution_date": "2024-07-01",
            "target": {
                "name": "widget_speed",
                "type": "regression",
                "unit": "m/s",
                "minimum": 0,
                "maximum": 10,
            },
        }
    )
    assert (spec.unit, spec.lower, spec.upper, spec.cutoff, spec.resolution) == (
        "m/s",
        0,
        10,
        "2024-06-01",
        "2024-07-01",
    )


def test_interval_probability_language_does_not_change_declared_quantity() -> None:
    spec = TargetSpec.from_task(
        {
            "prompt": "Predict revenue with 90% probability coverage.",
            "target": {"name": "revenue", "type": "regression", "unit": "USD"},
        }
    )
    assert spec.mode != "probability"
    assert spec.unit == "usd"


@pytest.mark.parametrize("point", [None, "omitted"])
def test_pure_label_prediction_allows_absent_numeric_point(point) -> None:
    spec = TargetSpec.from_task(
        {
            "target": {
                "name": "sentiment",
                "type": "classification",
                "labels": ["positive", "negative"],
            }
        }
    )
    prediction = {"label": "positive", "interval": {"level": 0.9, "lo": 2, "hi": 3}}
    if point is None:
        prediction["point_forecast"] = None
    spec.validate_prediction(prediction)


@pytest.mark.parametrize(
    "target,prompt",
    [
        ({"name": "credit_event_12m"}, ""),
        ({"name": "eps_outcome"}, ""),
        ({"name": "price_direction", "unit": "USD"}, ""),
        ({"name": "widget_direction"}, "Give a point forecast of widget speed."),
    ],
)
def test_numeric_classification_still_requires_a_point(target, prompt) -> None:
    spec = TargetSpec.from_task(
        {
            "prompt": prompt,
            "target": {**target, "type": "classification", "labels": ["up", "down"]},
        }
    )
    for prediction in (
        {"label": "up", "interval": {"level": 0.9, "lo": 0, "hi": 1}},
        {
            "label": "up",
            "point_forecast": None,
            "interval": {"level": 0.9, "lo": 0, "hi": 1},
        },
    ):
        with pytest.raises(ValueError):
            spec.validate_prediction(prediction)


def test_finite_number_rejects_integer_too_large_for_float() -> None:
    assert finite_number(10**10000) is False
    spec = TargetSpec.from_task({"target": {"name": "revenue", "type": "regression"}})
    with pytest.raises(ValueError):
        spec.validate_prediction(
            {"point_forecast": 10**10000, "interval": {"level": 0.9, "lo": 0, "hi": 1}}
        )


@pytest.mark.parametrize(
    "prompt",
    [
        "Predict revenue with 90% probability coverage.",
        "Predict the category. The probability of error is unknown.",
        "Predict probability-adjusted revenue.",
    ],
)
def test_incidental_probability_language_does_not_retype_the_target(prompt) -> None:
    assert (
        TargetSpec.from_task(
            {"prompt": prompt, "target": {"name": "revenue", "type": "regression"}}
        ).mode
        != "probability"
    )


@pytest.mark.parametrize(
    "interval",
    [None, {"level": 0.9, "lo": 3, "hi": 2}, {"level": 0.8, "lo": 0, "hi": 1}],
)
def test_pure_label_still_requires_valid_official_interval(interval) -> None:
    spec = TargetSpec.from_task(
        {
            "target": {
                "name": "category",
                "type": "classification",
                "labels": ["yes", "no"],
            }
        }
    )
    with pytest.raises(ValueError):
        spec.validate_prediction({"label": "yes", "interval": interval})


@pytest.mark.parametrize("include_null", [True, False])
def test_pure_label_model_reply_serializes_without_null_point(include_null) -> None:
    entity = {"entity_id": "WGT", "name": "Widget A"}
    task = {
        "target": {
            "name": "sentiment",
            "type": "classification",
            "labels": ["positive", "negative"],
        },
        "cutoff_date": "2024-06-01",
        "entities": [entity],
    }
    text = "Widget A sentiment is positive."
    chunk = Chunk("WIDGET_REPORT", "2024-05-01", 0, len(text), text)
    corpus = IndexedCorpus(
        [chunk], {chunk.doc_id: text}, {chunk.doc_id: chunk.doc_date}
    )
    reply = {
        "label": "positive",
        "interval": {"level": 0.9, "lo": 2, "hi": 3},
        "evidence": [{"doc_id": chunk.doc_id, "quote": text, "claim": text}],
    }
    if include_null:
        reply["point_forecast"] = None
    raw = json.dumps(reply)
    result = run_entity(
        task,
        entity,
        BM25Index(corpus.chunks, task["cutoff_date"]),
        corpus,
        MockModelClient(raw),
        10,
    )
    assert result.model_raw == raw
    assert "point_forecast" not in result.prediction
