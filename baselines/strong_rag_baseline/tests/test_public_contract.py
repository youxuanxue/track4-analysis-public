"""Public practice checks input contracts, never an answer snapshot or score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.strong_rag_baseline.cli import run
from baselines.strong_rag_baseline.indexer import build_index
from baselines.strong_rag_baseline.quantities import TargetSpec
from baselines.strong_rag_baseline.validation import validate_answer

REPO = Path(__file__).resolve().parents[3]
UNITS = sorted(path.parent for path in (REPO / "units").glob("*/task.json"))


@pytest.mark.parametrize("unit", UNITS, ids=lambda path: path.name)
def test_public_unit_obeys_task_and_citation_contract(unit, tmp_path):
    task = json.loads((unit / "task.json").read_text())
    output = tmp_path / "answer.json"
    answer = run(unit / "task.json", unit / "corpus", output, None, 10, grounded=True)
    assert json.loads(output.read_text()) == answer
    validate_answer(task, answer, build_index(unit / "corpus"))
    entities = {row["entity_id"]: row for row in task["entities"]}
    for prediction in answer["entity_predictions"]:
        TargetSpec.from_task(
            task, entities[prediction["entity_id"]]
        ).validate_prediction(prediction)
