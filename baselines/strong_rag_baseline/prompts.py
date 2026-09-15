"""Prompt construction for the strong RAG baseline.

One prompt per entity. The model selects identifiers from the current evidence
packet. The program resolves their exact source offsets and still rejects invalid
predictions atomically; identifiers do not prove semantic support.
"""

from __future__ import annotations

import json

from .indexer import Chunk
from .evidence import evidence_references
from .quantities import TargetSpec
from .tables import summary_columns, table_summaries

SYSTEM_PROMPT = """\
You are a careful financial analyst. You predict a target for one entity using ONLY the
evidence excerpts provided — no outside knowledge about events after the stated cutoff date.
You must respond with a single JSON object and nothing else. Cite only evidence IDs supplied
in this request. Do not reproduce quotes or invent IDs. Select the strongest relevant evidence,
not filing directories or signatures. A historical observation is not a realized future outcome.
Evidence must concern this entity, the correct period, metric and units."""

_TARGET_INSTRUCTIONS = {
    "classification": ('Set "label" to exactly one of the allowed labels. '),
    "regression": (
        'Set "point_forecast" to your numeric prediction of the target. '
        '"label" may be null.'
    ),
    "ranking": (
        'Set "point_forecast" to the predicted target metric on its stated scale. '
        'Do not emit "rank"; ordering is derived across all entity predictions.'
    ),
}


def build_response_schema(task: dict, entity: dict, retrieved: list[Chunk]) -> dict:
    spec = TargetSpec.from_task(task, entity)
    number: dict = {"type": "number"}
    if spec.lower is not None:
        number["minimum"] = spec.lower
    if spec.upper is not None:
        number["maximum"] = spec.upper
    point = dict(number)
    if not spec.requires_point:
        point["type"] = ["number", "null"]
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(spec.labels)}
            if spec.kind == "classification"
            else {"type": "null"},
            "point_forecast": point,
            "interval": {
                "type": "object",
                "properties": {
                    "level": {"type": "number", "const": spec.level},
                    "lo": dict(number),
                    "hi": dict(number),
                },
                "required": ["level", "lo", "hi"],
                "additionalProperties": False,
            },
            "evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence_id": {
                            "type": "string",
                            "enum": list(evidence_references(retrieved)),
                        },
                        "claim": {"type": "string", "minLength": 1, "maxLength": 4000},
                    },
                    "required": ["evidence_id", "claim"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["label", "point_forecast", "interval", "evidence"],
        "additionalProperties": False,
    }


def build_user_prompt(task: dict, entity: dict, retrieved: list[Chunk]) -> str:
    spec = TargetSpec.from_task(task, entity)
    kind, labels = spec.kind, spec.labels
    lines: list[str] = []

    lines.append(f"TASK: {task.get('prompt', '')}")
    lines.append(f"CUTOFF DATE: {task.get('cutoff_date', '')}")
    lines.append(f"TARGET: {spec.name} ({kind})")
    lines.append(
        "TARGET CONTRACT: " + json.dumps(task.get("target", {}), ensure_ascii=False)
    )
    lines.append(f"TARGET UNIT: {spec.unit or 'as declared in the task'}")
    if spec.resolution:
        lines.append(f"RESOLUTION DATE: {spec.resolution}")
    if spec.lower is not None or spec.upper is not None:
        lines.append(f"TARGET DOMAIN: minimum={spec.lower}, maximum={spec.upper}")
    if labels:
        lines.append(f"ALLOWED LABELS: {', '.join(labels)}")
    lines.append(
        "POINT FORECAST: a finite number is required; never null."
        if spec.requires_point
        else "POINT FORECAST: may be null for this label-only task."
    )
    if kind == "ranking":
        lines.append(
            "RANKING TASK INSTRUCTION: Output your continuous numeric estimate of the target metric in `point_forecast`. "
            "The scorer ranks entities by this predicted value, where LARGER VALUES correspond to HIGHER RANKS (rank 1 = highest value). "
            "Do NOT output rank integers (1, 2, 3...) in `point_forecast`!"
        )
    if spec.mode == "probability":
        lines.append(
            "Predict the probability of the target event on the 0 to 1 scale, not confidence in your selected label."
        )
    level = spec.level

    lines.append("\nENTITY:")
    for key, value in entity.items():
        if key in ("corpus_ref",):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("\nEVIDENCE EXCERPTS (cite only these):")
    for evidence_id, chunk in evidence_references(retrieved).items():
        lines.append(
            f"[{evidence_id}] doc_id={chunk.doc_id} (doc_date={chunk.doc_date})"
        )
        lines.append(f'"""{chunk.text}"""')
        summaries = table_summaries(
            chunk, spec.cutoff, summary_columns(task, entity, chunk)
        )
        if summaries:
            lines.append(
                "COMPUTED HISTORICAL TABLE CONTEXT (same evidence ID; source column scale; "
                "null unit means unknown; differences are last minus earlier, not growth rates; "
                "historical min/max are NOT a prediction interval; dates cover only this excerpt): "
                + json.dumps(summaries, ensure_ascii=False, allow_nan=False)
            )

    schema = {
        "label": "string or null",
        "point_forecast": "number or null",
        "interval": {
            "level": level,
            "lo": "number",
            "hi": "number",
        },
        "evidence": [
            {
                "evidence_id": "one evidence ID from this request",
                "claim": "one factual sentence directly supported by this excerpt",
            }
        ],
    }
    lines.append(
        "\nRespond with ONE JSON object of this shape (no markdown fences, no prose):"
    )
    lines.append(json.dumps(schema, indent=2))
    lines.append(
        f"\n{_TARGET_INSTRUCTIONS.get(kind, _TARGET_INSTRUCTIONS['classification'])}"
    )
    lines.append(
        f'The "interval" must be your {int(level * 100)}% prediction interval for the '
        "numeric target: wide enough that you expect the realized value to fall inside it "
        f"{int(level * 100)}% of the time. Its bounds must be finite, ordered, and on the target scale. "
        "Use one or two strongest, non-redundant evidence entries. Each must support this entity's predicted label, "
        "target value and interval, including the correct period, units and comparisons. "
        "A number merely appearing in an excerpt is not sufficient. Never invent evidence. "
        "Return evidence_id and claim only; the program supplies the exact source citation."
    )
    return "\n".join(lines)
