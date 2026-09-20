"""Final-output checks independent of retrieval and prediction implementations."""

from __future__ import annotations

from copy import deepcopy

import pytest

from baselines.strong_rag_baseline.agent import EntityResult
from baselines.strong_rag_baseline.formatter import build_answer
from baselines.strong_rag_baseline.indexer import IndexedCorpus, build_index


def inputs():
    task = {
        "task_id": "output-validation",
        "cutoff_date": "2024-02-01",
        "target": {"type": "classification", "name": "event", "labels": ["yes", "no"]},
        "entities": [{"entity_id": "X"}],
    }
    text = "X expects an event probability of 0.4."
    corpus = IndexedCorpus([], {"evidence": text}, {"evidence": "2024-01-31"})
    prediction = {
        "entity_id": "X",
        "label": "no",
        "point_forecast": 0.4,
        "interval": {"level": 0.9, "lo": 0.1, "hi": 0.8},
        "claims": [
            {
                "doc_id": "evidence",
                "span_start": 0,
                "span_end": len(text),
                "claim": text,
            }
        ],
    }
    return task, corpus, prediction


def assemble(task, corpus, predictions):
    return build_answer(task, [EntityResult(p, 0, "") for p in predictions], corpus)


def test_valid_output_keeps_prediction_and_citation():
    task, corpus, prediction = inputs()
    assert assemble(task, corpus, [prediction])["entity_predictions"] == [prediction]


@pytest.mark.parametrize("date", ["2024-02-02", None, "2024-02-30", "20240131"])
def test_final_check_refuses_unusable_citation_date(date):
    task, corpus, prediction = inputs()
    corpus.doc_dates["evidence"] = date
    with pytest.raises(ValueError, match="date|cutoff"):
        assemble(task, corpus, [prediction])


@pytest.mark.parametrize(
    "change",
    [
        {"point_forecast": float("nan")},
        {"point_forecast": float("inf")},
        {"point_forecast": True},
        {"claims": []},
        {"label": "unknown"},
        {"interval": {"level": 0.9, "lo": 0.8, "hi": 0.1}},
        {"interval": {"level": 0.95, "lo": 0.1, "hi": 0.8}},
    ],
)
def test_final_check_refuses_invalid_prediction(change):
    task, corpus, prediction = inputs()
    prediction.update(change)
    with pytest.raises(ValueError):
        assemble(task, corpus, [prediction])


def test_final_check_rejects_probability_outside_declared_domain():
    task, corpus, prediction = inputs()
    task["prompt"] = "Predict the probability of an event (0 to 1)."
    prediction["point_forecast"] = 159.0
    prediction["interval"] = {"level": 0.9, "lo": 78.0, "hi": 253.0}
    with pytest.raises(ValueError, match="domain"):
        assemble(task, corpus, [prediction])


@pytest.mark.parametrize("ids", [[], ["X", "X"], ["Y"], ["X", "Y"]])
def test_final_check_requires_exact_unique_roster(ids):
    task, corpus, prediction = inputs()
    predictions = [dict(deepcopy(prediction), entity_id=eid) for eid in ids]
    with pytest.raises(ValueError, match="roster|entity"):
        assemble(task, corpus, predictions)


def test_index_uses_citable_filename_instead_of_untrusted_embedded_id(tmp_path):
    import json

    (tmp_path / "filing.json").write_text(
        json.dumps(
            {"doc_id": "WRONG_ID", "doc_date": "2024-01-31", "text": "Citable text."}
        )
    )
    corpus = build_index(tmp_path)
    assert set(corpus.doc_texts) == {"filing"}
    assert corpus.chunks[0].doc_id == "filing"


def test_long_flat_documents_are_bounded_without_losing_source_offsets(tmp_path):
    import json

    text = "Quarterly revenue and costs.\n" * 500
    (tmp_path / "filing.json").write_text(
        json.dumps(
            {
                "doc_date": "2024-01-31",
                "text": text,
            }
        )
    )
    corpus = build_index(tmp_path)
    assert len(corpus.chunks) > 1
    assert max(len(chunk.text) for chunk in corpus.chunks) <= 2400
    covered_until = 0
    for chunk in corpus.chunks:
        assert chunk.span_start <= covered_until
        assert text[chunk.span_start : chunk.span_end] == chunk.text
        covered_until = chunk.span_end
    assert covered_until == len(text)


@pytest.mark.parametrize("source", ["model", "grounded"])
@pytest.mark.parametrize("points", [(1, 3, 2), (0.1, 0.3, 0.2), (2, 2, 1)])
def test_ranking_preserves_target_metric_values(source, points):
    task, corpus, template = inputs()
    task["target"] = {"type": "ranking", "name": "revenue_growth_pct"}
    task["entities"] = [{"entity_id": eid} for eid in ("A", "B", "C")]
    predictions = [
        dict(
            deepcopy(template),
            entity_id=entity["entity_id"],
            label=None,
            point_forecast=point,
            interval={"level": 0.9, "lo": point - 0.1, "hi": point + 0.1},
        )
        for entity, point in zip(task["entities"], points, strict=True)
    ]
    before = deepcopy(predictions)
    results = [EntityResult(p, 0, "", source=source) for p in predictions]
    answer = build_answer(task, results, corpus)
    assert answer["entity_predictions"] == before
    assert predictions == before


@pytest.mark.parametrize(
    "points", [(1.0, 2.0, 30.0), (-20.0, -15.0, 10.0), (0.00001, 0.00002, 0.00003)]
)
def test_regression_preserves_validated_metric_scale_and_domain(points):
    task, corpus, template = inputs()
    task["target"] = {
        "type": "regression",
        "name": "measurement",
        "minimum": min(points),
        "maximum": max(points),
    }
    task["entities"] = [{"entity_id": eid} for eid in ("A", "B", "C")]
    predictions = [
        dict(
            deepcopy(template),
            entity_id=e["entity_id"],
            label=None,
            point_forecast=point,
            interval={"level": 0.9, "lo": point, "hi": point},
        )
        for e, point in zip(task["entities"], points, strict=True)
    ]
    before = deepcopy(predictions)
    result = build_answer(
        task, [EntityResult(p, 0, "", source="model") for p in predictions], corpus
    )
    assert result["entity_predictions"] == before
    assert predictions == before
