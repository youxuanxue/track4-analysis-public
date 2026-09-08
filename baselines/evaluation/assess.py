"""Local development assessment through the installed Track 4 scorer.

Toolkit imports stay inside the assessment functions so baseline orchestration
remains importable in the stdlib-only CI job and submission environment.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any


def _validate_realized(ctx: dict[str, Any], realized: dict[str, Any]) -> None:
    from qfbench2_track_analysis.codes import T4OrganizerFault
    from qfbench2_track_analysis.scoring import _true_vectors

    if not isinstance(realized, dict):
        raise T4OrganizerFault("development outcomes must be a JSON object")
    rows = realized.get("outcomes")
    if not isinstance(rows, list) or not rows:
        raise T4OrganizerFault("development outcomes require a nonempty outcomes[]")
    ids = [row.get("entity_id") if isinstance(row, dict) else None for row in rows]
    roster = ctx["_roster"]
    if (
        any(not isinstance(entity_id, str) for entity_id in ids)
        or len(ids) != len(roster.entity_ids)
        or set(ids) != set(roster.entity_ids)
    ):
        raise T4OrganizerFault(
            "development outcomes must cover the exact entity roster"
        )
    labels, values = _true_vectors(realized, roster)
    params = ctx["_params"]
    if params.target_type != "classification" and values is None:
        raise T4OrganizerFault("numeric target types require numeric realized outcomes")
    if params.target_type == "classification" and any(
        not label or (roster.labels is not None and label not in roster.labels)
        for label in labels
    ):
        raise T4OrganizerFault("classification outcomes require valid true labels")
    expected_metadata = {
        "unit_id": ctx["_task"].get("task_id"),
        "cutoff_date": ctx["_task"].get("cutoff_date"),
        "target_type": params.target_type,
    }
    if any(
        key in realized and realized[key] != expected
        for key, expected in expected_metadata.items()
    ):
        raise T4OrganizerFault("development outcome metadata disagrees with the task")


def _hypothesis_records(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    from qfbench2_common.scoring.faithfulness import _doc_text

    from qfbench2_track_analysis.codes import T4ParticipantFailure
    from qfbench2_track_analysis.hypothesis import prediction_claims

    aligned = ctx.get("_aligned")
    if aligned is None:
        return []
    records: list[dict[str, Any]] = []
    corpus = ctx["_corpus"]
    claims = prediction_claims(aligned, ctx["_hypothesis_spec"])
    for entity_id, claim in zip(aligned.entity_ids, claims):
        citations: list[dict[str, Any]] = []
        for citation in claim["citations"]:
            start, end = citation.get("span_start"), citation.get("span_end")
            span = citation.get("span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                start, end = span
            record = {
                "doc_id": citation.get("doc_id"),
                "span_start": start,
                "span_end": end,
                "doc_date": None,
                "span_valid": False,
                "embargo_clean": False,
                "text": None,
            }
            try:
                doc = corpus.resolve(citation.get("doc_id"))
            except T4ParticipantFailure:
                citations.append(record)
                continue
            text = _doc_text(doc.document)
            record["doc_date"] = doc.doc_date.isoformat()
            record["embargo_clean"] = corpus.embargo_report(
                [citation], ctx["_cutoff"]
            ).clean
            record["span_valid"] = (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(text)
                and bool(text[start:end].strip())
            )
            if record["span_valid"]:
                record["text"] = text[start:end]
            citations.append(record)
        records.append(
            {
                "entity_id": entity_id,
                "hypothesis": claim["text"],
                "citations": citations,
            }
        )
    return records


def _numeric_errors(
    ctx: dict[str, Any], realized: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Keep raw errors visible when official normalized quality clips to zero."""
    from math import isfinite
    from qfbench2_track_analysis.scoring import _true_vectors

    aligned = ctx.get("_aligned")
    if realized is None or aligned is None:
        return None
    _, truth = _true_vectors(realized, ctx["_roster"])
    if truth is None or not all(isfinite(value) for value in aligned.pred_values):
        return None
    errors = [
        float(prediction - actual)
        for prediction, actual in zip(aligned.pred_values, truth)
    ]
    if not all(isfinite(value) for value in errors):
        return None
    return {
        "mae": mean(abs(value) for value in errors),
        "mean_signed_error": mean(errors),
        "entity_errors": dict(zip(aligned.entity_ids, errors)),
        "note": "Raw target-scale diagnostics only; the official scorer owns normalized quality.",
    }


def assess_unit(
    unit_dir: Path,
    output_dir: Path,
    *,
    realized: dict[str, Any] | None = None,
    profile: str = "smoke",
) -> dict[str, Any]:
    """Assess produced output, keeping local diagnostics separate from official scores.

    External reference defects raise organizer faults even if the answer is bad.
    A production request requires the configured, pinned judge and never falls
    back to smoke. Neither profile makes a local experiment an official score.
    """
    from qfbench2_common import manifest, taskcard

    from qfbench2_track_analysis.codes import T4OrganizerFault
    from qfbench2_track_analysis.scoring import (
        build_smoke_verifier,
        build_verifier,
        hydrate,
        score_unit,
        scorer_identity,
    )

    if profile not in ("smoke", "production"):
        raise ValueError("profile must be 'smoke' or 'production'")
    if profile == "production" and realized is None:
        raise T4OrganizerFault(
            "production development assessment requires explicit realized outcomes"
        )
    unit_dir, output_dir = Path(unit_dir), Path(output_dir)
    try:
        card, card_errors = taskcard.load_and_validate(unit_dir)
        manifest_errors = manifest.verify_manifest(unit_dir)
    except (OSError, ValueError) as exc:
        raise T4OrganizerFault("development unit metadata is unreadable") from exc
    if card_errors or manifest_errors:
        raise T4OrganizerFault(
            "development unit validation failed "
            f"({len(card_errors)} card, {len(manifest_errors)} manifest errors)"
        )
    if card["task"].get("track") != "analysis":
        raise T4OrganizerFault("development unit must belong to the analysis track")
    ctx: dict[str, Any] = {
        "unit_dir": unit_dir,
        "output_dir": output_dir,
        "card": card,
        "split": card["task"]["split"],
        # Explicit None prevents hydrate() from discovering reference/outcome.json.
        "realized": realized,
    }
    hydrate(ctx)
    if realized is not None:
        _validate_realized(ctx, realized)
    factory = build_smoke_verifier if profile == "smoke" else build_verifier
    verifier = factory(ctx)
    verdict = verifier.run(ctx)
    development_score = verdict.score if realized is not None else None
    if realized is not None and not verdict.admissible:
        # The shared Verdict intentionally has no score on failure. The Track 4
        # entrypoint supplies its canonical worst-case value for our aggregate.
        outcome = score_unit(
            ctx,
            judge=ctx["judge"],
            judge_provenance=ctx["judge_provenance"],
            require_outcome=True,
        )
        development_score = outcome.score
    faithfulness = ctx.get("_faithfulness")
    return {
        "profile": profile,
        "admissible": verdict.admissible,
        "rankable": False,
        "official_score": None,
        "development_score": development_score,
        "predictive_quality": verdict.detail.get("predictive_quality"),
        "interval_coverage": verdict.detail.get("interval_coverage"),
        "numeric_errors": _numeric_errors(ctx, realized),
        "nli_faithfulness": faithfulness if profile == "production" else None,
        "lexical_faithfulness": faithfulness if profile == "smoke" else None,
        "faithfulness_gate_applied": ctx.get("_faithfulness_gate_applied", False),
        "gates": {
            name: {
                "passed": result.passed,
                "label": result.label.value if result.label else None,
                "detail": result.detail,
            }
            for name, result in verdict.gate_results.items()
        },
        "labels": [label.value for label in verdict.labels],
        "hypothesis_records": _hypothesis_records(ctx),
        "scorer": scorer_identity(),
        "judge": ctx["judge_provenance"].to_mapping(),
    }
