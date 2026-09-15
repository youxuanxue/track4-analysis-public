"""Synthetic controls for source-bound per-share numbers and column dates."""

import json

import pytest

from baselines.strong_rag_baseline.financials import eps_table_context
from baselines.strong_rag_baseline.indexer import Chunk
from baselines.strong_rag_baseline.prompts import build_user_prompt


def source(text):
    return Chunk("synthetic", "2024-05-01", 7, 7 + len(text), text)


def test_accounting_loss_and_distinct_fiscal_column_dates():
    chunk = source(
        "Three Months Ended April 1, 2023 March 26, 2022 "
        "Earnings (loss) per share: Basic $ ( 0.14 ) $ 0.72 "
        "Diluted $ ( 0.14 ) $ 0.69"
    )
    [row] = eps_table_context(chunk)
    assert [c["value"] for c in row["source_values_left_to_right"]] == [-0.14, 0.69]
    assert [c["period_end"] for c in row["source_values_left_to_right"]] == [
        "2023-04-01",
        "2022-03-26",
    ]
    assert row["first_minus_second"] == -0.83
    assert not row["first_greater_than_second"]
    for cell in row["source_values_left_to_right"]:
        assert ("prefix " + chunk.text)[cell["span_start"] : cell["span_end"]] == cell[
            "text"
        ]


@pytest.mark.parametrize(
    "header, dates",
    [
        (
            "Three months ended April 1, April 2, (In millions, except per-share amounts) 2023 2022 ",
            ["2023-04-01", "2022-04-02"],
        ),
        ("Quarter ended March 31, 2024 2023 ", ["2024-03-31", "2023-03-31"]),
        ("Quarter ended February 30, 2024 2023 ", [None, None]),
        ("Quarter ended March 31, 2024 2024 ", [None, None]),
        ("Quarter ended March 31, 2024 2023 2022 ", [None, None]),
        ("Quarter ended Marsh 31, 2024 2023 ", [None, None]),
        (
            "Quarter ended June 30 Six months ended June 30 2024 2023 2024 2023 ",
            [None, None],
        ),
        ("", [None, None]),
    ],
)
def test_dates_only_come_from_unambiguous_complete_headers(header, dates):
    [row] = eps_table_context(
        source(header + "Diluted earnings per share $ 1.08 $ .79")
    )
    assert [c["period_end"] for c in row["source_values_left_to_right"]] == dates
    assert row["first_minus_second"] == 0.29
    assert row["first_greater_than_second"]


@pytest.mark.parametrize(
    "text",
    [
        "diluted EPS $ 2,111 $ 1,999",
        "Net income allocated for diluted earnings per share $ 2111.00 $ 1999.00",
        "Diluted earnings per share (2) Income from continuing operations 1.03 0.91",
        "Diluted earnings per share 12 11",
        "Diluted earnings per share 1e3 2.1",
        "Diluted earnings per share 1.2bad 2.1",
        "Diluted earnings per share 1.2 2.1 3.1",
        "Diluted earnings per share ( -0.14 ) 0.69",
        "Diluted earnings per share 0.14 ( +0.69 )",
        "Diluted earnings per share 1.2 2.1 3.1bad",
        "Diluted earnings per share 1.2 2.1 — 4.3",
    ],
)
def test_ambiguous_or_incomplete_rows_are_not_given_as_computed_facts(text):
    assert eps_table_context(source(text)) == []


def test_four_columns_keep_order_without_inventing_periods():
    [row] = eps_table_context(
        source("Diluted earnings per common share (a) $ .87 $ .76 $ 1.71 $ 1.50")
    )
    assert [c["value"] for c in row["source_values_left_to_right"]] == [
        0.87,
        0.76,
        1.71,
        1.50,
    ]
    assert all(c["period_end"] is None for c in row["source_values_left_to_right"])
    json.dumps(row, allow_nan=False)


def test_prompt_context_preserves_unknown_period_and_does_not_change_target():
    chunk = source("Diluted earnings per share $ ( 0.14 ) $ 0.69")
    task = {
        "target": {"type": "regression", "name": "eps_yoy_growth_pct"},
        "cutoff_date": "2024-05-01",
    }
    entity = {
        "entity_id": "SYN",
        "quarter_reported": "2024-06-30",
        "prior_year_q_eps": 1.1,
    }
    prompt = build_user_prompt(task, entity, [chunk])
    assert "COMPUTED PER-SHARE ROW CONTEXT" in prompt
    assert '"value": -0.14' in prompt
    assert '"period_end": null' in prompt
    assert "not future forecasts" in prompt
    task["target"]["name"] = "steps_growth"
    assert "COMPUTED PER-SHARE ROW CONTEXT" not in build_user_prompt(
        task, entity, [chunk]
    )
