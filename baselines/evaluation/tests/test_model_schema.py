"""Validate reader schemas in the toolkit-backed CI job with jsonschema installed."""

import pytest

pytest.importorskip("qfbench2_common")

from jsonschema import Draft202012Validator, ValidationError  # noqa: E402

from baselines.strong_rag_baseline.prompts import (  # noqa: E402
    build_response_schema,
    build_user_prompt,
)
from baselines.strong_rag_baseline.tests.test_structured_evidence import fixture  # noqa: E402


def test_schema_requires_numeric_probability_and_preserves_label_only_tasks():
    task, entity, corpus, reply = fixture()
    task["target"] = {
        "type": "classification",
        "name": "credit_event_probability",
        "labels": ["event", "no_event"],
    }
    schema = build_response_schema(task, entity, corpus.chunks)
    Draft202012Validator.check_schema(schema)
    reply.update(
        label="no_event",
        point_forecast=0.1,
        interval={"level": 0.9, "lo": 0, "hi": 0.2},
    )
    validator = Draft202012Validator(schema)
    validator.validate(reply)
    for value in (None, True, -0.1, 1.1):
        with pytest.raises(ValidationError):
            validator.validate(dict(reply, point_forecast=value))
    with pytest.raises(ValidationError):
        validator.validate(
            dict(
                reply, evidence=[{"evidence_id": "unselected", "claim": "Unsupported."}]
            )
        )
    assert "never null" in build_user_prompt(task, entity, corpus.chunks)
    task["target"] = {
        "type": "classification",
        "name": "category",
        "labels": ["event", "no_event"],
    }
    Draft202012Validator(build_response_schema(task, entity, corpus.chunks)).validate(
        dict(reply, point_forecast=None)
    )
