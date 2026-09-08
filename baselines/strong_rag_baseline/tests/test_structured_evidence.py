"""Model IDs and schemas preserve exact evidence and task numeric contracts."""

import io
import json
import urllib.error
from dataclasses import replace

import pytest

from baselines.strong_rag_baseline.agent import _ground_claims, run_entity
from baselines.strong_rag_baseline.client import HTTPModelClient, MockModelClient
from baselines.strong_rag_baseline.config import Config
from baselines.strong_rag_baseline.evidence import evidence_references, prepare_evidence
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.prompts import (
    build_response_schema,
)
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.reasoner import _other_entity_hit


def fixture():
    text = 'Acme expects revenue growth of 5 percent, with a "range" of 4 to 6 percent.'
    chunk = Chunk("release", "2024-01-01", 0, len(text), text)
    corpus = IndexedCorpus([chunk], {"release": text}, {"release": chunk.doc_date})
    entity = {"entity_id": "ACME", "name": "Acme"}
    task = {
        "task_id": "synthetic",
        "cutoff_date": "2024-01-31",
        "entities": [entity],
        "target": {"type": "regression", "name": "revenue_growth", "unit": "percent"},
    }
    reply = {
        "label": None,
        "point_forecast": 5,
        "interval": {"level": 0.9, "lo": 4, "hi": 6},
        "evidence": [
            {
                "evidence_id": next(iter(evidence_references([chunk]))),
                "claim": "Acme expects growth.",
            }
        ],
    }
    return task, entity, corpus, reply


def test_nested_entity_names_do_not_hide_independent_mentions():
    own = ["all items less food and energy"]
    others = ["all items", "food", "energy"]
    assert not _other_entity_hit(
        "All items less food and energy rose 0.2%.", own, others
    )
    assert _other_entity_hit(
        "All items less food and energy rose; food fell.", own, others
    )
    assert _other_entity_hit(
        "Acme Holdings reported growth.", ["acme"], ["acme holdings"]
    )


def test_series_column_table_retains_header_dates_and_exact_offsets():
    entity = {"entity_id": "BOND10", "series_fred": "RATE10"}
    task = {
        "cutoff_date": "2024-01-31",
        "entities": [entity],
        "target": {"type": "regression", "name": "yield_change"},
    }
    text = (
        "date | RATE2 | RATE10\n2024-01-29 | 3.1 | 4.2\n"
        "2024-01-30 | 3.2 | 4.3\n2024-02-01 | 99 | 99\n"
    )
    chunk = Chunk("rates", "2024-01-31", 0, len(text), text)
    corpus = IndexedCorpus([chunk], {"rates": text}, {"rates": chunk.doc_date})
    index = BM25Index([chunk], task["cutoff_date"])
    packet = prepare_evidence(task, entity, index, corpus)
    [record] = packet.ledger["records"]
    assert record["binding"]["source"] == "series_column"
    assert record["binding"]["column"] == "RATE10"
    assert record["text"].startswith("date | RATE2 | RATE10")
    assert "4.3" in record["text"] and "99" not in record["text"]
    assert text[record["span_start"] : record["span_end"]] == record["text"]
    assert not prepare_evidence(
        task, dict(entity, series_fred="MISSING"), index, corpus
    ).chunks
    assert not prepare_evidence(task, entity, index, corpus, max_span_chars=20).chunks


def test_identifiers_restore_quotes_without_model_copying_or_escaping():
    task, entity, corpus, reply = fixture()

    class StructuredClient:
        def complete_json(self, system, user, schema):
            assert reply["evidence"][0]["evidence_id"] in user
            assert (
                "evidence_id" in schema["properties"]["evidence"]["items"]["properties"]
            )
            return json.dumps(reply)

    result = run_entity(
        task,
        entity,
        BM25Index(corpus.chunks, task["cutoff_date"]),
        corpus,
        StructuredClient(),
        4,
    )
    assert result.source == "model"
    [claim] = result.prediction["claims"]
    assert (
        corpus.doc_texts[claim["doc_id"]][claim["span_start"] : claim["span_end"]]
        == corpus.chunks[0].text
    )


@pytest.mark.parametrize("bad_id", ["Eunknown", None, ["Eunknown"]])
def test_unknown_or_invalid_ids_reject_the_entire_model_prediction(bad_id):
    task, entity, corpus, reply = fixture()
    reply["point_forecast"] = 999
    reply["evidence"].append(
        {"evidence_id": bad_id, "claim": "Invalid extra citation."}
    )
    result = run_entity(
        task,
        entity,
        BM25Index(corpus.chunks, task["cutoff_date"]),
        corpus,
        MockModelClient(json.dumps(reply)),
        4,
    )
    assert result.source == "grounded"
    assert result.fallback_reason == "model_evidence"
    assert result.prediction["point_forecast"] != 999


def test_id_cannot_be_overridden_with_a_different_doc_or_quote():
    _, _, corpus, reply = fixture()
    evidence = dict(reply["evidence"][0], doc_id="elsewhere", quote="replacement")
    assert _ground_claims([evidence], corpus, corpus.chunks) == ([], 1)


def test_id_preserves_selected_offset_when_source_text_repeats():
    text = "same text. same text."
    chunks = [
        Chunk("doc", "2024-01-01", 0, 10, text[:10]),
        Chunk("doc", "2024-01-01", 11, 21, text[11:21]),
    ]
    corpus = IndexedCorpus(chunks, {"doc": text}, {"doc": "2024-01-01"})
    evidence_id = next(iter(evidence_references([chunks[1]])))
    claims, dropped = _ground_claims(
        [{"evidence_id": evidence_id, "claim": "Second occurrence."}], corpus, chunks
    )
    assert dropped == 0
    assert claims[0]["span_start"] == 11
    assert claims[0]["span_end"] == 21
    assert _ground_claims(
        [{"evidence_id": evidence_id, "claim": "Hidden occurrence."}],
        corpus,
        chunks[:1],
    ) == ([], 1)


def test_http_sends_json_schema_and_does_not_drop_it(monkeypatch):
    task, entity, corpus, reply = fixture()
    schema = build_response_schema(task, entity, corpus.chunks)
    seen = []

    def respond(req, timeout):
        seen.append(json.loads(req.data))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": json.dumps(reply)},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = HTTPModelClient(
        replace(
            Config.from_env(),
            model_endpoint="http://127.0.0.1:8080/v1",
            model_token=None,
        )
    )
    assert json.loads(client.complete_json("system", "user", schema)) == reply
    assert seen[0]["response_format"]["json_schema"] == {
        "name": "entity_prediction",
        "strict": True,
        "schema": schema,
    }


def test_schema_rejection_never_retries_without_constraints(monkeypatch):
    seen = []

    def reject(req, timeout):
        seen.append(json.loads(req.data))
        raise urllib.error.HTTPError(req.full_url, 400, "unsupported schema", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", reject)
    monkeypatch.setattr("time.sleep", lambda _: None)
    task, entity, corpus, _ = fixture()
    client = HTTPModelClient(
        replace(
            Config.from_env(), model_endpoint="http://127.0.0.1:8080/v1", max_retries=2
        )
    )
    result = run_entity(
        task, entity, BM25Index(corpus.chunks, task["cutoff_date"]), corpus, client, 4
    )
    assert result.source == "grounded"
    assert result.fallback_reason == "model_request"
    assert len(seen) == 2
    assert all(row["response_format"]["type"] == "json_schema" for row in seen)


def test_directory_only_material_is_excluded_but_substantive_exhibit_text_survives():
    entity = {"entity_id": "ACME", "name": "Acme"}
    task = {
        "cutoff_date": "2024-01-31",
        "entities": [entity],
        "target": {"name": "credit_event", "type": "regression"},
    }
    text = (
        "The exhibits listed in the accompanying Exhibit Index are filed as a part of this report.\n"
        "Acme credit agreements in Exhibit 10 restrict borrowing after a default."
    )
    chunk = Chunk("Acme_filing", "2024-01-01", 0, len(text), text)
    corpus = IndexedCorpus(
        [chunk], {chunk.doc_id: text}, {chunk.doc_id: chunk.doc_date}
    )
    packet = prepare_evidence(
        task, entity, BM25Index([chunk], task["cutoff_date"]), corpus
    )
    assert packet.chunks
    assert all("accompanying Exhibit Index" not in c.text for c in packet.chunks)
    assert any("restrict borrowing" in c.text for c in packet.chunks)
