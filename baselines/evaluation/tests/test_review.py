"""Exact source positions cannot manufacture semantic review approval."""

import json

import pytest

from baselines.evaluation.review import digest, summarize_reviews


def packet():
    return {
        "records": [
            {
                "record_id": "a",
                "citations": [{"span_valid": True, "embargo_clean": True}],
            }
        ]
    }


def test_valid_offsets_remain_semantically_unreviewed():
    bundle = packet()
    summary = summarize_reviews(
        bundle,
        {
            "packet_digest": digest(bundle),
            "reviews": [{"record_id": "a", "verdict": "unreviewed"}],
        },
    )
    assert summary["invalid_spans"] == 0
    assert summary["reviewed_records"] == 0
    assert summary["nli_faithfulness"] is None


@pytest.mark.parametrize(
    "defect", ["digest", "duplicate", "missing", "anonymous", "verdict"]
)
def test_unbound_or_unattributed_reviews_refused(defect):
    bundle = packet()
    annotations = {
        "packet_digest": digest(bundle),
        "reviews": [
            {
                "record_id": "a",
                "verdict": "direct",
                "reviewer": "test-reviewer",
                "reason": "Invented explicit support.",
            }
        ],
    }
    if defect == "digest":
        annotations["packet_digest"] = "stale"
    elif defect == "duplicate":
        annotations["reviews"] *= 2
    elif defect == "missing":
        annotations["reviews"] = []
    elif defect == "anonymous":
        annotations["reviews"][0]["reviewer"] = ""
    else:
        annotations["reviews"][0]["verdict"] = "passed"
    with pytest.raises(ValueError):
        summarize_reviews(bundle, annotations)


@pytest.mark.parametrize("change", ["claim_text", "source_text", "duplicate_run"])
def test_packet_rejects_changed_artifacts_after_real_evaluation(tmp_path, change):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation import __main__ as runner, review
    from baselines.evaluation.tests.test_runner_integration import build_unit

    units = tmp_path / "inputs"
    unit = build_unit(units)
    output = tmp_path / "evaluation"
    assert runner.main(["--units", str(units), "--out", str(output)]) == 0
    args = ["--units", str(units), "--report", str(output / "report.json")]
    clean = tmp_path / "clean-review"
    assert review.main([*args, "--out", str(clean)]) == 0
    bundle = json.loads((clean / "packet.json").read_text())
    assert len(bundle["records"]) == 3
    for record in bundle["records"]:
        assert record["original_answer_hash_verified"] is True
        for citation in record["citations"]:
            source = json.loads(
                (unit / "corpus" / (citation["doc_id"] + ".json")).read_text()
            )
            assert (
                citation["text"]
                == source["text"][citation["span_start"] : citation["span_end"]]
            )
    report = json.loads((output / "report.json").read_text())
    row = report["runs"][0]
    if change == "claim_text":
        path = output / row["case_id"] / f"seed-{row['seed']}" / "answer.json"
        answer = json.loads(path.read_text())
        # The canonical hypothesis does not change when participant prose changes.
        answer["entity_predictions"][0]["claims"][0]["claim"] = (
            "Changed after evaluation."
        )
        path.write_text(json.dumps(answer))
    elif change == "source_text":
        path = unit / "corpus/SYN-A.json"
        source = json.loads(path.read_text())
        source["text"] = "Changed after evaluation."
        path.write_text(json.dumps(source))
    else:
        report["runs"].append(row)
        (output / "report.json").write_text(json.dumps(report))
    assert review.main([*args, "--out", str(tmp_path / "stale-review")]) == 2
