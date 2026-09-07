"""Prompt construction for the strong RAG baseline.

One prompt per entity. The model sees the entity's tabular features and the
top-K retrieved excerpts, each tagged with its ``doc_id``, and must return a
single JSON object. Evidence quotes are required to be verbatim substrings of
the provided excerpts. The agent resolves exact character offsets and rejects
the entire model prediction if any quote cannot be located.
"""

from __future__ import annotations

import json

from .indexer import Chunk
from .quantities import TargetSpec

SYSTEM_PROMPT = """\
You are a careful financial analyst. You predict a target for one entity using ONLY the
evidence excerpts provided — no outside knowledge about events after the stated cutoff date.
You must respond with a single JSON object and nothing else. Every evidence quote you return
must be copied verbatim, character for character, from one of the provided excerpts."""

_TARGET_INSTRUCTIONS = {
    "classification": (
        'Set "label" to exactly one of the allowed labels. '
        'Set "point_forecast" to your best numeric estimate of the underlying quantity '
        "if one is defined for this task, else null."
    ),
    "regression": (
        'Set "point_forecast" to your numeric prediction of the target. '
        '"label" may be null.'
    ),
    "ranking": (
        'Set "point_forecast" to the predicted target metric on its stated scale. '
        'Do not emit "rank"; ordering is derived across all entity predictions.'
    ),
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
    level = spec.level

    lines.append("\nENTITY:")
    for key, value in entity.items():
        if key in ("corpus_ref",):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("\nEVIDENCE EXCERPTS (cite only these):")
    for i, chunk in enumerate(retrieved, 1):
        lines.append(f"[{i}] doc_id={chunk.doc_id} (doc_date={chunk.doc_date})")
        lines.append(f'"""{chunk.text}"""')

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
                "doc_id": "doc_id of the excerpt the quote comes from",
                "quote": "verbatim substring copied from that excerpt",
                "claim": "one factual sentence the quote directly supports",
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
        "Give 1 to 4 evidence entries. Every quote must support this entity's predicted label, "
        "target value and interval, including the correct period, units and comparisons. "
        "A number merely appearing in a quote is not sufficient. Never invent evidence."
    )
    return "\n".join(lines)
