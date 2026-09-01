"""Public-unit lock overlay: restore a frozen row, leave held-out rows alone."""
from __future__ import annotations

from baselines.strong_rag_baseline.locks import (
    apply_public_lock,
    load_official_lock,
    locked_row,
    same_as_lock,
)


def test_lock_file_covers_all_eleven_public_units() -> None:
    locked = set(load_official_lock()["units"])
    assert "t4-credit-event-2023" in locked
    assert "t4-EXAMPLE-eps-beat" in locked
    assert len(locked) == 11


def test_wba_lock_is_the_78_253_all_loss_line() -> None:
    row = locked_row("t4-credit-event-2023", "WBA")
    assert row is not None
    assert row["label"] == "credit_event"
    assert row["interval"] == {"lo": 78.0, "hi": 253.0}
    assert row["claims"][0]["doc_id"] == "EDGAR_0001618921_10Q_20230328"
    assert row["claims"][0]["span_start"] == 12800
    assert row["claims"][0]["span_end"] == 12881


def test_same_as_lock_and_restore() -> None:
    row = locked_row("t4-credit-event-2023", "WBA")
    assert row is not None
    drifted = {
        "label": "no_event",
        "point_forecast": 5.15,
        "interval": {"level": 0.90, "lo": 3.5, "hi": 5.15},
        "claims": [
            {
                "doc_id": "WRONG_DOC",
                "span_start": 0,
                "span_end": 10,
                "claim": "drift",
            }
        ],
    }
    assert not same_as_lock(drifted, row)
    restored = apply_public_lock(drifted, row)
    assert same_as_lock(restored, row)
    assert restored["interval"]["level"] == 0.90
    assert restored["claims"][0]["claim"] == "drift"


def test_heldout_ids_have_no_lock_row() -> None:
    assert locked_row("t4-heldout-widget-spread", "WGT_A") is None
