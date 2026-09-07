"""Summarize complete evaluation rosters without treating missing scores as zero."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation roster")
    scores = [row.get("assessment", {}).get("development_score") for row in rows]
    all_scored = all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores)
    elapsed = sorted(row["elapsed_s"] for row in rows)
    entities = [
        entity
        for row in rows
        for entity in row.get("diagnostics", {}).get("entities", [])
    ]
    sources = Counter(entity.get("source", "unknown") for entity in entities)
    reasons = Counter(
        entity["fallback_reason"]
        for entity in entities
        if entity.get("fallback_reason")
    )
    complete = all(row["status"] == "completed" for row in rows)
    return {
        "runs": len(rows),
        "complete": complete,
        "rankable": False,
        "official_score": None,
        "execution_successes": sum(row["execution"]["returncode"] == 0 for row in rows),
        "admissible_runs": sum(
            row.get("assessment", {}).get("admissible") is True for row in rows
        ),
        "admission_profile": rows[0]["profile"],
        "development_mean": mean(scores) if all_scored and complete else None,
        "development_scored_runs": sum(v is not None for v in scores),
        "nli_measured_runs": sum(
            row.get("assessment", {}).get("nli_faithfulness") is not None
            for row in rows
        ),
        "elapsed_p95_s": elapsed[max(0, math.ceil(len(elapsed) * 0.95) - 1)],
        "entity_sources": dict(sources),
        "fallback_reasons": dict(reasons),
    }


def markdown(report: dict) -> str:
    summary = report["summary"]

    def show(value):
        return "unmeasured" if value is None else str(value)

    lines = [
        "## Executive summary (read this first)",
        "",
        "This is a local development report, not a leaderboard result.",
        "Public practice tasks do not provide resolved outcomes.",
        "Smoke admission does not apply the production NLI gate.",
        "Missing measurements are reported explicitly and never counted as perfect scores.",
        "",
        f"- Execution mode: {report['mode']}; profile: {report['profile']}",
        f"- Completed processes: {summary['execution_successes']}/{summary['runs']}",
        f"- {report['profile']} admission: {summary['admissible_runs']}/{summary['runs']}",
        f"- Development composite mean: {show(summary['development_mean'])}",
        f"- Production NLI measured runs: {summary['nli_measured_runs']}",
        f"- P95 elapsed seconds: {summary['elapsed_p95_s']:.3f}",
        f"- Entity sources: {summary['entity_sources']}",
        f"- Fallback reasons: {summary['fallback_reasons']}",
        "",
        "| Case | Seed | Execution | Admission | Development score | Seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["runs"]:
        a = row.get("assessment", {})
        lines.append(
            f"| {row['case_id']} | {row['seed']} | {row['execution']['returncode']} | "
            f"{show(a.get('admissible'))} | {show(a.get('development_score'))} | {row['elapsed_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Full provenance, entity diagnostics and evidence candidates are in the adjacent JSON files.",
            "",
        ]
    )
    return "\n".join(lines)
