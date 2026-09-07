"""Read the published task / card shapes. Never invent a target type or label set.

``SUBMISSION_CLI.md`` publishes ``target_type`` as a top-level field of ``task.json``;
every unit in this repo nests it at ``target.type``. Reading only one shape silently
mislabels the other and is ``t4.target_type_mismatch`` → ``W = -0.27``.

Legal classification labels come from ``target.labels``. Hardcoding ``beat`` / ``miss``
/ ``inline`` fails every unit whose vocabulary is different.
"""
from __future__ import annotations

from typing import Any


def target_type(task: dict[str, Any]) -> str | None:
    """Documented top-level field wins; otherwise the nested ``target.type``."""
    nested = task.get("target")
    declared = task.get("target_type")
    if isinstance(declared, str) and declared:
        return declared
    if isinstance(nested, dict):
        nested_type = nested.get("type")
        if isinstance(nested_type, str) and nested_type:
            return nested_type
    return None


def legal_labels(task: dict[str, Any]) -> list[str]:
    target = task.get("target")
    if not isinstance(target, dict):
        return []
    raw = target.get("labels")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, str) and x]


def target_name(task: dict[str, Any]) -> str:
    target = task.get("target")
    if isinstance(target, dict) and isinstance(target.get("name"), str) and target["name"]:
        return target["name"].replace("_", " ")
    return "outcome"


def interval_level(task: dict[str, Any]) -> float:
    raw = task.get("interval_level")
    try:
        level = float(raw)
    except (TypeError, ValueError):
        return 0.90
    if 0.0 < level < 1.0:
        # Cards pin 0.90; some task.json files write 0.9. Emit the card spelling
        # so a future exact-decimal check cannot treat them as different.
        if abs(level - 0.90) < 1e-9:
            return 0.90
        return level
    return 0.90


def entity_display_name(entity: dict[str, Any]) -> str:
    name = entity.get("name")
    eid = entity.get("entity_id", "")
    if isinstance(name, str) and name:
        return f"{name} ({eid})" if eid else name
    return str(eid)


def fmt_number(value: float) -> str:
    """Match ``qfbench2_track_analysis.hypothesis._fmt`` so we can search the corpus
    for the exact tokens the judge will put in the hypothesis."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"
