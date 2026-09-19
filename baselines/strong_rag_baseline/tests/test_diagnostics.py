"""Optional diagnostics explain fallbacks without changing submitted answers."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from baselines.strong_rag_baseline import agent, cli
from baselines.strong_rag_baseline.agent import EntityResult
from baselines.strong_rag_baseline.client import (
    HTTPModelClient,
    MockModelClient,
    ModelBudgetExceeded,
)
from baselines.strong_rag_baseline.config import Config
from baselines.strong_rag_baseline.indexer import build_index
from baselines.strong_rag_baseline.retriever import BM25Index

SECRET = "synthetic-private-token-do-not-log"


@pytest.fixture
def http_model_client(monkeypatch):
    state = {"reply": {}, "delay": 0.0, "calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            if state["delay"]:
                import time

                time.sleep(state["delay"])
            body = {
                "choices": [
                    {
                        "message": {"content": json.dumps(state["reply"])},
                        "finish_reason": "stop",
                    }
                ]
            }
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = Config.from_env()
    client = HTTPModelClient(
        config,
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    client._stub_state = state
    yield client
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
def unit(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Acme expects revenue growth of 5 percent next quarter, in a range of 4 to 6 percent."
    (corpus / "release.json").write_text(
        json.dumps({"doc_date": "2024-01-10", "text": text}), encoding="utf-8"
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "synthetic-diagnostics",
                "cutoff_date": "2024-01-31",
                "target": {
                    "type": "regression",
                    "name": "revenue_growth_pct",
                    "unit": "percent",
                },
                "entities": [{"entity_id": "ACME", "name": "Acme"}],
                "interval_level": 0.9,
            }
        ),
        encoding="utf-8",
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


def test_entity_result_positional_construction_is_compatible():
    result = EntityResult({}, 0, "", "unchanged rationale")
    assert result.rationale == "unchanged rationale"
    assert result.source == "grounded"
    assert result.fallback_reason is None


@pytest.mark.parametrize(
    "stage", ["model_request", "model_json", "model_evidence", "model_prediction"]
)
def test_fallback_stage_is_exact_and_never_contains_raw_errors(unit, tmp_path, stage):
    task, corpus, valid_reply = unit
    reply = deepcopy(valid_reply)
    reply["private_debug"] = SECRET

    def request_failure(*args):
        raise RuntimeError(SECRET)

    if stage == "model_request":
        client = MockModelClient(request_failure)
    elif stage == "model_json":
        client = MockModelClient(SECRET)
    else:
        if stage == "model_evidence":
            reply["evidence"][0]["quote"] = SECRET
        else:
            reply["point_forecast"] = SECRET
        client = MockModelClient(json.dumps(reply))
    diagnostics_path = tmp_path / "diagnostics.json"
    actual = cli.run(
        task,
        corpus,
        tmp_path / "answer.json",
        client,
        5,
        diagnostics_path=diagnostics_path,
    )
    expected = cli.run(task, corpus, tmp_path / "grounded.json", None, 5)
    assert actual == expected
    serialized = diagnostics_path.read_text(encoding="utf-8")
    assert SECRET not in serialized
    diagnostics = json.loads(serialized)
    assert diagnostics["mode"] == "model"
    [entity] = diagnostics["entities"]
    assert entity["source"] == "grounded"
    assert entity["fallback_reason"] == stage
    assert entity["elapsed_s"] >= 0


def test_model_success_records_only_bounded_diagnostics(unit, tmp_path, monkeypatch):
    task, corpus, reply = unit
    reply["private_debug"] = SECRET
    clock = iter([10.0, 12.5, 30.0, 35.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    client = MockModelClient(json.dumps(reply))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    diagnostics_path = tmp_path / "reports" / "diagnostics.json"
    without = cli.run(task, corpus, first, client, 5)
    with_diagnostics = cli.run(
        task, corpus, second, client, 5, diagnostics_path=diagnostics_path
    )
    assert without == with_diagnostics
    assert first.read_bytes() == second.read_bytes()
    assert "elapsed_s" not in second.read_text(encoding="utf-8")
    assert json.loads(diagnostics_path.read_text(encoding="utf-8")) == {
        "request_ledger": None,
        "schema_version": "1",
        "task_id": "synthetic-diagnostics",
        "mode": "model",
        "entities": [
            {
                "entity_id": "ACME",
                "source": "model",
                "fallback_reason": None,
                "elapsed_s": 5.0,
                "dropped_claims": 0,
                "model_accepted": True,
                "stages": {
                    "request": "accepted",
                    "json": "accepted",
                    "evidence": "accepted",
                    "prediction": "accepted",
                },
            }
        ],
    }
    assert SECRET not in diagnostics_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("forced", [False, True])
def test_cli_distinguishes_forced_grounded_and_no_endpoint(
    unit, tmp_path, monkeypatch, forced
):
    task, corpus, _ = unit
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("T4_LOCAL_LLAMA", raising=False)
    diagnostics_path = tmp_path / "diagnostics.json"
    argv = [
        "--task",
        str(task),
        "--corpus",
        str(corpus),
        "--out",
        str(tmp_path / "answer.json"),
        "--diagnostics",
        str(diagnostics_path),
    ]
    if forced:
        monkeypatch.setenv("MODEL_ENDPOINT", "https://unused.example/v1")
        argv.append("--mock")
    assert cli.main(argv) == 0
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["mode"] == "grounded"
    [entity] = diagnostics["entities"]
    assert entity["source"] == "grounded"
    assert entity["fallback_reason"] == ("forced_grounded" if forced else "no_endpoint")


@pytest.mark.parametrize(
    "destination",
    ["answer", "task", "corpus_file", "new_corpus_file", "symlink", "hardlink"],
)
def test_diagnostics_refuses_input_and_answer_aliases(unit, tmp_path, destination):
    task, corpus, _ = unit
    out = tmp_path / "answer.json"
    out.write_text("existing answer", encoding="utf-8")
    if destination == "answer":
        diagnostics_path = out
    elif destination == "task":
        diagnostics_path = task
    elif destination == "corpus_file":
        diagnostics_path = corpus / "release.json"
    elif destination == "new_corpus_file":
        diagnostics_path = corpus / "new.json"
    else:
        diagnostics_path = tmp_path / "alias.json"
        if destination == "symlink":
            diagnostics_path.symlink_to(task)
        else:
            os.link(corpus / "release.json", diagnostics_path)
    before = {path: path.read_bytes() for path in (task, corpus / "release.json", out)}
    with pytest.raises(ValueError, match="diagnostics path"):
        cli.run(task, corpus, out, None, 5, diagnostics_path=diagnostics_path)
    assert {path: path.read_bytes() for path in before} == before


def test_cli_rejects_unsafe_diagnostics_before_running(unit, tmp_path, monkeypatch):
    task, corpus, _ = unit

    def must_not_run(*args, **kwargs):
        pytest.fail("unsafe diagnostics must be rejected before inference")

    monkeypatch.setattr(cli, "run", must_not_run)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--task",
                str(task),
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "answer.json"),
                "--diagnostics",
                str(task),
            ]
        )
    assert error.value.code == 2


def test_failed_answer_does_not_write_diagnostics(unit, tmp_path, monkeypatch):
    task, corpus, _ = unit
    diagnostics_path = tmp_path / "diagnostics.json"

    def invalid_answer(*args):
        raise ValueError("synthetic invalid answer")

    monkeypatch.setattr(cli, "build_answer", invalid_answer)
    with pytest.raises(ValueError, match="invalid answer"):
        cli.run(
            task,
            corpus,
            tmp_path / "answer.json",
            None,
            5,
            diagnostics_path=diagnostics_path,
        )
    assert not diagnostics_path.exists()


def test_offline_diagnostics_record_zero_requests_without_inventing_model_usage(
    unit, tmp_path
):
    task, corpus, _ = unit
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", None, 5, diagnostics_path=path)
    ledger = json.loads(path.read_text())["request_ledger"]
    assert ledger == {
        "version": 1,
        "source": "offline",
        "attempts_used": 0,
        "attempts": [],
    }


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_diagnostics_propagate_model_acceptance_for_each_target_type(
    unit, tmp_path, target_type
):
    task, corpus, reply = unit
    task_data = json.loads(task.read_text())
    task_data["target"] = {"type": target_type, "name": "revenue_growth_pct", "unit": "percent"}
    if target_type == "classification":
        reply["label"] = "up"
        task_data["target"]["labels"] = ["up", "down"]
    elif target_type == "ranking":
        reply["rank"] = 1
        task_data["target"]["name"] = "revenue_rank"
    task.write_text(json.dumps(task_data))
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", MockModelClient(json.dumps(reply)), 5, diagnostics_path=path)
    entity = json.loads(path.read_text())["entities"][0]
    assert entity["model_accepted"] is True
    assert entity["stages"] == {
        "request": "accepted",
        "json": "accepted",
        "evidence": "accepted",
        "prediction": "accepted",
    }


def test_http_client_to_agent_records_accepted_response(unit, tmp_path, http_model_client):
    task, corpus, reply = unit
    http_model_client._stub_state["reply"] = reply
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", http_model_client, 5, diagnostics_path=path)
    entity = json.loads(path.read_text())["entities"][0]
    assert entity["model_accepted"] is True
    assert entity["stages"]["prediction"] == "accepted"
    assert http_model_client.request_ledger()["attempts_used"] == 1


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_http_model_rejection_diagnostics_for_each_target_type(
    unit, tmp_path, http_model_client, target_type
):
    task, corpus, reply = unit
    task_data = json.loads(task.read_text())
    task_data["target"] = {"type": target_type, "name": "revenue_growth_pct", "unit": "percent"}
    if target_type == "classification":
        task_data["target"]["labels"] = ["up", "down"]
        reply["label"] = "not-a-label"
    elif target_type == "ranking":
        task_data["target"]["name"] = "revenue_rank"
        task_data["entities"] = [
            {"entity_id": "ACME", "name": "Acme"},
            {"entity_id": "BETA", "name": "Beta"},
        ]
        (corpus / "beta.json").write_text(
            json.dumps({"doc_date": "2024-01-10", "text": "Beta expects revenue growth of 3 percent next quarter."})
        )
        reply["point_forecast"] = 1.0
        reply["evidence"][0]["claim"] = "Acme expects 5 percent growth."
    else:
        reply["point_forecast"] = 999.0
    task.write_text(json.dumps(task_data))
    http_model_client._stub_state["reply"] = reply
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", http_model_client, 5, diagnostics_path=path)
    entities = json.loads(path.read_text())["entities"]
    assert entities[0]["model_accepted"] is False
    assert entities[0]["source"] == "grounded"
    assert entities[0]["stages"]["request"] == "accepted"
    assert entities[0]["stages"]["json"] == "accepted"
    assert entities[0]["stages"]["prediction"] == "not_attempted"


def test_http_model_budget_diagnostics(unit, tmp_path, http_model_client):
    task, corpus, reply = unit
    http_model_client._stub_state["reply"] = reply
    http_model_client.config = Config.from_env()
    http_model_client.config = http_model_client.config.__class__(
        **{**http_model_client.config.__dict__, "max_requests": 0}
    )
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", http_model_client, 5, diagnostics_path=path)
    entity = json.loads(path.read_text())["entities"][0]
    assert entity["fallback_reason"] == "model_budget"
    assert entity["stages"]["request"] == "budget"


def test_http_model_timeout_diagnostics(unit, tmp_path, http_model_client):
    task, corpus, reply = unit
    http_model_client._stub_state["reply"] = reply
    http_model_client.config = http_model_client.config.__class__(
        **{**http_model_client.config.__dict__, "timeout_s": 0.1, "max_retries": 1}
    )
    http_model_client._stub_state["delay"] = 0.3
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", http_model_client, 5, diagnostics_path=path)
    entity = json.loads(path.read_text())["entities"][0]
    assert entity["fallback_reason"] == "model_request"
    assert entity["stages"]["request"] == "timeout"


def test_no_retrieved_evidence_does_not_claim_request_accepted(unit):
    task_path, corpus_dir, _ = unit
    task = json.loads(task_path.read_text())
    entity = {"entity_id": "MISSING", "name": "Missing"}
    indexed = build_index(corpus_dir)
    result = agent.run_entity(
        task,
        entity,
        BM25Index(indexed.chunks, task["cutoff_date"]),
        indexed,
        MockModelClient("unused"),
        5,
    )
    assert result.diagnostics["request"] == "not_attempted"


def test_diagnostics_explain_timeout_and_budget_fallback(unit, tmp_path):
    task, corpus, _ = unit

    def timeout(*args):
        raise TimeoutError("hidden timeout")

    timeout_path = tmp_path / "timeout.json"
    cli.run(task, corpus, tmp_path / "timeout-answer.json", MockModelClient(timeout), 5, diagnostics_path=timeout_path)
    timeout_entity = json.loads(timeout_path.read_text())["entities"][0]
    assert timeout_entity["fallback_reason"] == "model_request"
    assert timeout_entity["model_accepted"] is False
    assert timeout_entity["stages"]["request"] == "timeout"

    class BudgetClient(MockModelClient):
        def complete(self, *args):
            raise ModelBudgetExceeded("budget")

    budget_path = tmp_path / "budget.json"
    cli.run(task, corpus, tmp_path / "budget-answer.json", BudgetClient("unused"), 5, diagnostics_path=budget_path)
    budget_entity = json.loads(budget_path.read_text())["entities"][0]
    assert budget_entity["fallback_reason"] == "model_budget"
    assert budget_entity["stages"]["request"] == "budget"
