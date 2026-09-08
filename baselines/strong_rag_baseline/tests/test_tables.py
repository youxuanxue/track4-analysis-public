"""Synthetic controls for dated column arithmetic and source attribution."""

import json

import pytest

from baselines.strong_rag_baseline.evidence import prepare_evidence
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.prompts import build_user_prompt
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.tables import dated_tables, table_summaries


def chunk(text):
    return Chunk("source", "2024-01-31", 7, 7 + len(text), text)


def test_exact_column_decimal_arithmetic_and_dates():
    source = chunk(
        "RATE observations in percent.\n"
        "date | OTHER | RATE\n2024-01-02 | 900 | 4.10\n2024-01-30 | 800 | 4.02"
    )
    [record] = table_summaries(source, "2024-01-31", ["RATE"])
    assert record["first"] == 4.1
    assert record["last"] == 4.02
    assert record["last_minus_first"] == -0.08
    assert record["last_minus_previous_bps"] == -8
    assert record["first_date"] == "2024-01-02"
    assert record["last_date"] == "2024-01-30"
    start, end = record["last_row_span"]
    assert ("prefix " + source.text)[start:end] == "2024-01-30 | 800 | 4.02"
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "date_header", ["date", "auction_date", "report_date", "observation_date"]
)
def test_date_header_variants_and_formatted_signed_numbers(date_header):
    source = chunk(
        f"{date_header} | amount\n2024-01-02 | -1,234.50\n2024-01-30 | +2,345.75"
    )
    [record] = table_summaries(source, "2024-01-31", ["amount"])
    assert record["last_minus_previous"] == 3580.25
    assert record["unit"] is None
    assert "last_minus_first_bps" not in record


@pytest.mark.parametrize(
    "value", ["", "--", "NaN", "inf", "1e3", "12,34", "1.2garbage", "9" * 400]
)
def test_missing_or_unsupported_cells_are_not_zero_or_partial_numbers(value):
    source = chunk(f"date | value\n2024-01-02 | {value}\n2024-01-30 | 5")
    assert table_summaries(source, "2024-01-31", ["value"]) == []


@pytest.mark.parametrize(
    "dates", [("2024-01-02", "2024-01-02"), ("2024-01-30", "2024-01-02")]
)
def test_duplicate_and_reversed_dates_reject_whole_table(dates):
    assert (
        dated_tables(f"date | A\n{dates[0]} | 1\n{dates[1]} | 2", "2024-01-31", 1000)
        == []
    )


def test_blank_lines_and_future_rows_stop_a_table():
    source = chunk("date | A\n2024-01-02 | 1\n\n2024-01-30 | 900")
    [record] = table_summaries(source, "2024-01-31", ["A"])
    assert record["last"] == 1
    future = chunk("date | A\n2024-01-02 | 1\n2024-02-01 | 900")
    assert table_summaries(future, "2024-01-31", ["A"])[0]["last"] == 1


def test_duplicate_column_names_are_ambiguous():
    assert dated_tables("date | A | a\n2024-01-02 | 1 | 2", "2024-01-31", 1000) == []


def test_markdown_outer_pipes_and_empty_last_cell_are_distinct():
    outer = chunk("| date | A |\n| 2024-01-02 | 1 |")
    assert table_summaries(outer, "2024-01-31", ["A"])[0]["last"] == 1
    empty = chunk("date | A | B\n2024-01-02 | 1 | ")
    assert table_summaries(empty, "2024-01-31", ["A"])[0]["last"] == 1
    assert table_summaries(empty, "2024-01-31", ["B"]) == []


def packet(text, *, series=False):
    entity = {"entity_id": "ALPHA", "name": "Alpha"}
    if series:
        entity["series_fred"] = "RATE"
    task = {
        "cutoff_date": "2024-01-31",
        "target": {"name": "bid_to_cover_ratio", "type": "regression"},
        "entities": [entity, {"entity_id": "BETA", "name": "Beta"}],
    }
    source = Chunk("source", "2024-01-31", 0, len(text), text)
    corpus = IndexedCorpus([source], {"source": text}, {"source": source.doc_date})
    result = prepare_evidence(
        task, entity, BM25Index([source], task["cutoff_date"]), corpus
    )
    return result, task, entity


def test_document_bound_table_keeps_column_headers_and_arithmetic_in_prompt():
    text = "Alpha auction history.\n\nauction_date | offering | bid_to_cover\n2024-01-02 | 42 | 2.50\n2024-01-30 | 43 | 2.65"
    result, task, entity = packet(text)
    [record] = [r for r in result.ledger["records"] if r["table_summaries"]]
    assert record["binding"]["source"] == "document_table"
    assert record["table_summaries"][0]["last"] == 2.65
    assert record["table_summaries"][0]["last_minus_first"] == 0.15
    assert text[record["span_start"] : record["span_end"]] == record["text"]
    prompt = build_user_prompt(task, entity, result.chunks)
    assert '"last": 2.65' in prompt
    assert "NOT a prediction interval" in prompt


def test_multi_entity_title_does_not_grant_table_ownership():
    result, _, _ = packet(
        "Alpha and Beta auction history.\nauction_date | bid_to_cover\n2024-01-02 | 99"
    )
    assert not any(record["table_summaries"] for record in result.ledger["records"])


def test_document_title_cannot_override_other_entities_in_table_rows():
    result, _, _ = packet(
        "Alpha auction history.\nauction_date | company | bid_to_cover\n2024-01-02 | Beta | 99"
    )
    assert not any(record["table_summaries"] for record in result.ledger["records"])


def test_series_unit_declaration_is_preserved_and_not_borrowed_from_target():
    text = "Alpha: RATE observations in percent.\ndate | RATE\n2024-01-02 | 4.10\n2024-01-30 | 4.02"
    result, _, _ = packet(text, series=True)
    [record] = result.ledger["records"]
    assert record["text"] == text
    assert record["table_summaries"][0]["last_minus_first_bps"] == -8
    unknown = chunk(
        text.replace("RATE observations in percent", "OTHER observations in percent")
    )
    assert table_summaries(unknown, "2024-01-31", ["RATE"])[0]["unit"] is None


@pytest.mark.parametrize(
    "prefix",
    [
        "These are not RATE observations in percent.",
        "RATE observations in percent.\nA different table follows.",
    ],
)
def test_negated_or_nonadjacent_units_do_not_enable_conversion(prefix):
    source = chunk(prefix + "\ndate | RATE\n2024-01-02 | 4.10\n2024-01-30 | 4.02")
    [record] = table_summaries(source, "2024-01-31", ["RATE"])
    assert record["unit"] is None
    assert "last_minus_first_bps" not in record
