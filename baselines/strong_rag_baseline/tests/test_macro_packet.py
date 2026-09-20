"""Shared policy context reaches the model with intact provenance and budgets."""

from baselines.strong_rag_baseline.evidence import prepare_evidence
from baselines.strong_rag_baseline.prompts import build_user_prompt
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.tests.test_macro_context import _inputs


def test_policy_and_entity_evidence_reach_prompt_with_distinct_bindings():
    policy = (
        "The central bank raised its policy rate target and expects further increases."
    )
    observation = "Benchmark Note A yield was 3.2 percent at the cutoff."
    task, entity, corpus = _inputs(
        [
            ("policy", "2024-05-31", policy),
            ("quote", "2024-05-31", observation),
            ("future_policy", "2024-06-02", policy),
        ]
    )
    packet = prepare_evidence(
        task, entity, BM25Index(corpus.chunks, task["cutoff_date"]), corpus, top_k=2
    )
    assert [(r["text"], r["binding"]["source"]) for r in packet.ledger["records"]] == [
        (policy, "shared_context"),
        (observation, "text_alias"),
    ]
    prompt = build_user_prompt(task, entity, packet.chunks)
    assert policy in prompt and observation in prompt
    assert "future_policy" not in prompt
    assert "Policy-rate decisions and projections are shared macro context" in prompt
    assert "not observed or projected Treasury" in prompt


def test_projection_header_and_row_survive_packet_selection_together():
    header = "Variable\nMedian\n2024\n2025\n"
    row = "Federal funds rate\n4.0\n3.0\n"
    observation = "Benchmark Note A yield was 3.2 percent at the cutoff."
    task, entity, corpus = _inputs(
        [
            ("policy_table", "2024-05-31", header + "Inflation\n2.0\n2.0\n" + row),
            ("quote", "2024-05-31", observation),
        ]
    )
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    packet = prepare_evidence(task, entity, index, corpus, top_k=4, max_chars=600)
    shared = [
        r
        for r in packet.ledger["records"]
        if r["binding"]["source"] == "shared_context"
    ]
    assert [r["text"] for r in shared] == [header, row]
    assert packet.ledger["characters"] == sum(len(c.text) for c in packet.chunks) <= 600
    for c in packet.chunks:
        assert corpus.doc_texts[c.doc_id][c.span_start : c.span_end] == c.text
    small = prepare_evidence(task, entity, index, corpus, top_k=2, max_chars=600)
    assert [c.text for c in small.chunks] == [observation]
