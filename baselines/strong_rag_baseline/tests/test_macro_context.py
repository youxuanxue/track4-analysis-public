"""Shared policy evidence survives identity filtering without future information."""

import pytest

from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.macro_context import shared_macro_context


def _inputs(documents):
    entity = {"entity_id": "NOTE_A", "name": "Benchmark Note A", "unit": "bps_change"}
    task = {
        "target": {"name": "yield_change_bps", "type": "regression"},
        "cutoff_date": "2024-06-01",
        "entities": [entity, {"entity_id": "NOTE_B", "name": "Benchmark Note B"}],
    }
    texts = {key: text for key, _, text in documents}
    dates = {key: date for key, date, _ in documents}
    corpus = IndexedCorpus(
        [Chunk(key, date, 0, len(text), text) for key, date, text in documents],
        texts,
        dates,
    )
    return task, entity, corpus


def test_shared_policy_and_maturity_mechanism_without_entity_alias():
    policy = (
        "The central bank raised its policy rate target and expects further increases."
    )
    curve = "Front-end yields are most sensitive to the expected policy path."
    task, entity, corpus = _inputs([("bulletin", "2024-05-31", policy + "\n" + curve)])
    chunks = shared_macro_context(task, entity, corpus)
    assert [c.text for c in chunks] == [policy, curve]
    assert all(
        corpus.doc_texts[c.doc_id][c.span_start : c.span_end] == c.text for c in chunks
    )


def test_cutoff_issuer_and_other_entity_exclusions():
    allowed = (
        "The central bank raised its policy rate target and expects further increases."
    )
    task, entity, corpus = _inputs(
        [
            ("current", "2024-06-01", allowed),
            ("future", "2024-06-02", allowed),
            ("undated", None, allowed),
            (
                "issuer",
                "2024-05-31",
                "Our company anticipates that the central bank will raise its policy rate target.",
            ),
            ("EDGAR_filing", "2024-05-31", allowed),
            (
                "another_note",
                "2024-05-31",
                "Benchmark Note B is expected to rise when the central bank raises its target rate.",
            ),
        ]
    )
    assert [c.doc_id for c in shared_macro_context(task, entity, corpus)] == ["current"]
    task["target"]["name"] = "diluted_eps"
    entity["unit"] = "currency per share"
    assert shared_macro_context(task, entity, corpus) == []
    task["target"]["name"] = "yield_change_bps"
    entity["cik"] = "123"
    assert shared_macro_context(task, entity, corpus) == []


def test_vertical_projection_row_requires_header_and_joint_budget():
    header = "Table 1. Policy projections\nVariable\nMedian\n2024\n2025\n"
    other = "Inflation\n2.0\n2.0\n"
    policy = "Federal funds rate\n4.0\n3.0\n"
    text = header + other + policy + "Previous projection\n4.5\n3.5\n"
    task, entity, corpus = _inputs([("policy_table", "2024-05-31", text)])
    chunks = shared_macro_context(task, entity, corpus)
    # The selected contiguous header starts at Variable; its adjacent table
    # caption is not needed to preserve the Median/year columns.
    assert [c.text for c in chunks] == [header[header.index("Variable") :], policy]
    assert all(text[c.span_start : c.span_end] == c.text for c in chunks)
    assert shared_macro_context(task, entity, corpus, max_chunks=1) == []
    assert shared_macro_context(task, entity, corpus, max_chars=20) == []
    assert shared_macro_context(task, entity, corpus, max_span_chars=20) == []


def test_budget_keeps_complete_spans_and_does_not_mutate_inputs():
    first = "The central bank raised its target rate and expects further increases."
    second = "Long-end yields are anchored by long-run inflation expectations."
    task, entity, corpus = _inputs([("policy", "2024-05-31", first + "\n" + second)])
    assert [
        c.text for c in shared_macro_context(task, entity, corpus, max_chars=len(first))
    ] == [first]
    assert [
        c.text for c in shared_macro_context(task, entity, corpus, max_chunks=1)
    ] == [first]
    assert corpus.doc_texts["policy"] == first + "\n" + second
    with pytest.raises(ValueError):
        shared_macro_context(task, entity, corpus, max_chunks=0)


def test_action_outweighs_dissent_and_projection_metadata():
    metadata = "Economic projections describe the appropriate federal funds rate for the next calendar year."
    dissent = "A member preferred to lower the federal funds rate by a smaller amount."
    action = "The committee decided to lower the target range for the federal funds rate by one half percentage point."
    task, entity, corpus = _inputs(
        [
            ("a_projection_title", "2024-05-31", metadata),
            ("b_dissent", "2024-05-31", dissent),
            ("z_policy", "2024-05-31", action),
        ]
    )
    assert [
        c.text for c in shared_macro_context(task, entity, corpus, max_chunks=1)
    ] == [action]


def test_market_positioning_survives_policy_projection_context():
    action = (
        "The committee lowered the target range by 50 basis points and will assess incoming data."
    )
    positioning = (
        "Market-implied paths were more aggressive than the Committee's projected path, "
        "and the next scheduled FOMC meeting is inside the inter-meeting window."
    )
    sep = "Variable\nMedian\n2024\n2025\nFederal funds rate\n4.4\n3.4\n"
    task, entity, corpus = _inputs(
        [
            ("action", "2024-05-30", action),
            ("positioning", "2024-05-31", positioning),
            ("sep", "2024-05-30", sep),
        ]
    )
    selected = shared_macro_context(task, entity, corpus, max_chunks=4)
    text = "\n".join(c.text for c in selected)
    assert positioning in text
