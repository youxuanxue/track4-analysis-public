"""Pipeline integration on synthetic inputs without public prediction locks."""

from __future__ import annotations

import json
import pytest

from baselines.strong_rag_baseline.cli import main, run
from baselines.strong_rag_baseline.client import MockModelClient


@pytest.fixture
def unit(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Acme expects revenue growth of 5 percent next quarter, in a range of 4 to 6 percent."
    (corpus / "release.json").write_text(
        json.dumps(
            {
                "doc_id": "release",
                "doc_date": "2024-01-10",
                "text": text,
            }
        )
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "synthetic-pipeline",
                "cutoff_date": "2024-01-31",
                "prompt": "Predict Acme revenue growth, in percent, for the next quarter.",
                "target": {
                    "type": "regression",
                    "name": "revenue_growth_pct",
                    "unit": "percent",
                },
                "entities": [{"entity_id": "ACME", "name": "Acme"}],
                "interval_level": 0.9,
            }
        )
    )
    reply = {
        "point_forecast": 5.0,
        "interval": {"level": 0.9, "lo": 4.0, "hi": 6.0},
        "evidence": [
            {
                "doc_id": "release",
                "quote": text,
                "claim": "Acme expects 5 percent growth.",
            }
        ],
    }
    return task, corpus, reply


def test_pipeline_preserves_valid_model_prediction_and_source_offsets(unit, tmp_path):
    task, corpus, reply = unit
    answer = run(
        task,
        corpus,
        tmp_path / "output" / "answer.json",
        MockModelClient(json.dumps(reply)),
        5,
    )
    [prediction] = answer["entity_predictions"]
    assert prediction["entity_id"] == "ACME"
    assert prediction["point_forecast"] == 5.0
    assert prediction["interval"] == reply["interval"]
    [claim] = prediction["claims"]
    assert claim["doc_id"] == "release"
    assert claim["span_start"] == 0
    assert claim["span_end"] == len(reply["evidence"][0]["quote"])
    assert json.loads((tmp_path / "output" / "answer.json").read_text()) == answer


def test_pipeline_is_deterministic(unit, tmp_path):
    task, corpus, reply = unit
    client = MockModelClient(json.dumps(reply))
    assert run(task, corpus, tmp_path / "a.json", client, 5) == run(
        task, corpus, tmp_path / "b.json", client, 5
    )


def test_missing_interval_uses_the_complete_grounded_prediction(unit, tmp_path):
    task, corpus, reply = unit
    del reply["interval"]
    reply["point_forecast"] = 999.0
    model = run(
        task, corpus, tmp_path / "model.json", MockModelClient(json.dumps(reply)), 5
    )
    fallback = run(task, corpus, tmp_path / "fallback.json", None, 5, grounded=True)
    assert model["entity_predictions"] == fallback["entity_predictions"]
    assert model["entity_predictions"][0]["point_forecast"] != 999.0


def test_mock_cli_never_calls_an_injected_endpoint(unit, tmp_path, monkeypatch):
    task, corpus, _ = unit
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")

    def no_network(*args, **kwargs):
        pytest.fail("--mock must not use a model endpoint")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    out = tmp_path / "answer.json"
    assert (
        main(
            [
                "analyze",
                "--task",
                str(task),
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--mock",
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["entity_predictions"][0]["claims"]
