"""Public task IDs cannot override evidence or cutoff changes at inference time."""

from __future__ import annotations

import json
from pathlib import Path

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.indexer import build_index

UNIT = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"


def predict(tmp_path, task):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    return run(path, UNIT / "corpus", tmp_path / "answer.json", None, 5, grounded=True)


def test_same_input_has_same_predictions_after_task_id_changes(tmp_path):
    task = json.loads((UNIT / "task.json").read_text())
    original = predict(tmp_path, task)
    task["task_id"] = "new-unseen-task-id"
    renamed = predict(tmp_path, task)
    assert original["entity_predictions"] == renamed["entity_predictions"]
    assert renamed["task_id"] == task["task_id"]


def test_earlier_cutoff_is_respected_for_previously_locked_task(tmp_path):
    task = json.loads((UNIT / "task.json").read_text())
    task["cutoff_date"] = "2024-02-01"
    answer = predict(tmp_path, task)
    corpus = build_index(UNIT / "corpus")
    for prediction in answer["entity_predictions"]:
        for claim in prediction["claims"]:
            assert corpus.doc_dates[claim["doc_id"]] <= task["cutoff_date"]
