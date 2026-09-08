"""Historical builders reject wrong vintages and keep truth out of inputs."""

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest

from baselines.evaluation import historical


def test_exact_single_series_vintage_required():
    data = b"observation_date,DGS2_20250115\n2025-01-13,4.25\n2025-01-14,\n"
    assert historical.parse_snapshot(data, "DGS2", date(2025, 1, 15)) == {
        date(2025, 1, 13): Decimal("4.25")
    }
    with pytest.raises(ValueError, match="wrong series/vintage"):
        historical.parse_snapshot(
            data.replace(b"20250115", b"20260908"), "DGS2", date(2025, 1, 15)
        )
    with pytest.raises(ValueError, match="wrong series/vintage"):
        historical.parse_snapshot(
            b"observation_date,DGS1_20250115,DGS2_20260908\n", "DGS1", date(2025, 1, 15)
        )


@pytest.mark.parametrize(
    "rows",
    [
        "2025-01-16,4.0\n",
        "2025-01-13,4\n2025-01-13,4\n",
        "2025-01-13,NaN\n",
        "2025-01-13,Infinity\n",
        "2025-01-13,4,unexpected\n",
    ],
)
def test_invalid_observations_refused(rows):
    with pytest.raises(ValueError):
        historical.parse_snapshot(
            ("observation_date,DGS2_20250115\n" + rows).encode(),
            "DGS2",
            date(2025, 1, 15),
        )


def test_tampered_cache_is_not_refetched(tmp_path, monkeypatch):
    path = tmp_path / "DGS2-2025-01-15.csv"
    path.write_text("changed")
    path.with_suffix(".json").write_text(
        json.dumps({"url": "wrong", "sha256": "wrong"})
    )
    monkeypatch.setattr(
        historical,
        "_download",
        lambda *a, **k: pytest.fail("must not redownload tampered cache"),
    )
    with pytest.raises(ValueError, match="checksum"):
        historical.snapshot(tmp_path, "DGS2", date(2025, 1, 15), offline=False)


def test_overlap_refused_before_fetch(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {
                        "id": "a",
                        "cutoff": "2025-01-01",
                        "resolution": "2025-02-01",
                        "split": "train",
                    },
                    {
                        "id": "b",
                        "cutoff": "2025-02-01",
                        "resolution": "2025-03-01",
                        "split": "test",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(
        historical, "snapshot", lambda *a, **k: pytest.fail("must reject before fetch")
    )
    with pytest.raises(ValueError, match="precede"):
        historical.build(spec, tmp_path / "cache", tmp_path / "out")


def test_builder_produces_three_correlated_views_without_future_inputs(
    tmp_path, monkeypatch
):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation.dataset import load_cases, stage_inputs
    from baselines.evaluation.assess import assess_unit

    cutoff, resolution = date(2025, 1, 15), date(2025, 2, 4)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {
                        "id": "invented-event",
                        "cutoff": str(cutoff),
                        "resolution": str(resolution),
                        "split": "test",
                    }
                ],
            }
        )
    )

    def invented(cache, series, vintage, **kwargs):
        base = Decimal(historical.SERIES[series]) / 10
        rows = {
            vintage - timedelta(days=i): base + Decimal("0.01") * i
            for i in range(1, 30)
        }
        return rows, {
            "url": "https://example.invalid/synthetic",
            "sha256": "synthetic",
            "vintage": str(vintage),
        }

    monkeypatch.setattr(historical, "snapshot", invented)
    out = tmp_path / "built"
    roster = historical.build(spec, tmp_path / "cache", out)
    cases = load_cases(units=None, manifest=roster)
    assert len(cases) == 3 and len({c.group for c in cases}) == 1
    for case in cases:
        staged = tmp_path / case.case_id
        stage_inputs(case.unit_dir, staged)
        assert set(p.name for p in staged.iterdir()) == {"task.json", "corpus"}
        assert not case.truth_path.is_relative_to(case.unit_dir)
        text = " ".join(p.read_text() for p in (staged / "corpus").glob("*.json"))
        assert str(resolution) not in text
        truth = json.loads(case.truth_path.read_text())
        assert len(truth["outcomes"]) == 8
        output = tmp_path / (case.case_id + "-output")
        output.mkdir()
        # Invalid participant output still retains the official worst-case development score.
        result = assess_unit(case.unit_dir, output, realized=truth)
        assert result["development_score"] == pytest.approx(-0.27)
    with pytest.raises(ValueError, match="must be new"):
        historical.build(spec, tmp_path / "cache", out, offline=True)
