"""Export source-bound evidence review packets from completed development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .dataset import load_cases, public_roots, require_external, stage_inputs
from .historical import write_json

VERDICTS = (
    "direct",
    "derivable",
    "historical_only",
    "contradicted",
    "insufficient",
    "unreviewed",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def packet(report_path: Path, cases: list) -> dict:
    from .assess import assess_unit

    report = json.loads(report_path.read_text())
    runs = report["runs"]
    lookup = {case.case_id: case for case in cases}
    seeds = {run["seed"] for run in runs}
    keys = [(run["case_id"], run["seed"]) for run in runs]
    if (
        not seeds
        or len(keys) != len(set(keys))
        or set(keys) != {(cid, seed) for cid in lookup for seed in seeds}
    ):
        raise ValueError(
            "report must cover the exact case/seed roster, without duplicates"
        )
    if any(run.get("status") != "completed" for run in runs):
        raise ValueError("cannot audit an incomplete evaluation")
    records = []
    for case in cases:
        with tempfile.TemporaryDirectory() as temp:
            current_digest = stage_inputs(case.unit_dir, Path(temp) / "input")
        task = json.loads((case.unit_dir / "task.json").read_text())
        for run in (r for r in runs if r["case_id"] == case.case_id):
            if run["input_digest"] != current_digest:
                raise ValueError("audit inputs differ from the evaluated inputs")
            output = report_path.parent / case.case_id / f"seed-{run['seed']}"
            if run["execution"]["returncode"] != 0:
                raise ValueError(
                    "audit requires successful outputs; execution failures stay in the evaluation report"
                )
            answer = json.loads((output / "answer.json").read_text())
            answer_hash = hashlib.sha256(
                (output / "answer.json").read_bytes()
            ).hexdigest()
            if run.get("answer_sha256") and run["answer_sha256"] != answer_hash:
                raise ValueError("answer bytes changed since evaluation")
            assessment = assess_unit(case.unit_dir, output)
            hypotheses = {
                row["entity_id"]: row for row in assessment["hypothesis_records"]
            }
            predictions = {
                row["entity_id"]: row for row in answer["entity_predictions"]
            }
            if set(hypotheses) != {entity["entity_id"] for entity in task["entities"]}:
                raise ValueError(
                    "audit requires a schema-valid, aligned prediction roster"
                )
            old = {
                row["entity_id"]: row for row in run["assessment"]["hypothesis_records"]
            }
            for entity in task["entities"]:
                entity_id = entity["entity_id"]
                hyp = hypotheses[entity_id]
                if old.get(entity_id, {}).get("hypothesis") != hyp["hypothesis"]:
                    raise ValueError(
                        "prediction or canonical hypothesis changed since evaluation"
                    )
                old_citations = old[entity_id]["citations"]
                if [
                    {k: c[k] for k in ("doc_id", "span_start", "span_end")}
                    for c in old_citations
                ] != [
                    {k: c[k] for k in ("doc_id", "span_start", "span_end")}
                    for c in hyp["citations"]
                ]:
                    raise ValueError("citation coordinates changed since evaluation")
                record = {
                    "case_id": case.case_id,
                    "seed": run["seed"],
                    "split": case.split,
                    "group": case.group,
                    "entity": entity,
                    "target": task["target"],
                    "prompt": task["prompt"],
                    "cutoff": task["cutoff_date"],
                    "resolution": task["resolution_date"],
                    "prediction": predictions[entity_id],
                    "hypothesis": hyp["hypothesis"],
                    "citations": hyp["citations"],
                    "input_digest": current_digest,
                    "answer_sha256": answer_hash,
                    "original_answer_hash_verified": bool(run.get("answer_sha256")),
                }
                records.append({"record_id": digest(record), **record})
    return {
        "version": 1,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "official_score": None,
        "rankable": False,
        "records": records,
    }


def summarize_reviews(bundle: dict, annotations: dict) -> dict:
    expected = {row["record_id"] for row in bundle["records"]}
    rows = annotations.get("reviews", [])
    ids = [row["record_id"] for row in rows]
    if (
        annotations.get("packet_digest") != digest(bundle)
        or len(ids) != len(set(ids))
        or set(ids) != expected
    ):
        raise ValueError(
            "reviews must bind to the exact packet and complete record roster"
        )
    counts: Counter = Counter()
    for row in rows:
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError("unknown semantic review verdict")
        if verdict != "unreviewed" and not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("reviewer", "reason")
        ):
            raise ValueError("reviewed records require reviewer identity and reason")
        counts[verdict] += 1
    citations = [citation for row in bundle["records"] for citation in row["citations"]]
    return {
        "records": len(expected),
        "verdict_counts": dict(counts),
        "reviewed_records": len(expected) - counts["unreviewed"],
        "citation_count": len(citations),
        "invalid_spans": sum(not c["span_valid"] for c in citations),
        "embargo_violations": sum(not c["embargo_clean"] for c in citations),
        "nli_faithfulness": None,
        "official_score": None,
        "rankable": False,
        "note": "Semantic reviews are attributed judgments, not production NLI or forecasting accuracy.",
    }


def render(bundle: dict) -> str:
    lines = [
        "## Executive summary (read this first)",
        "",
        "This packet pairs each canonical prediction with its resolved source passages.",
        "Valid offsets and dates do not establish semantic support.",
        "Reviews are initially unreviewed and are never inferred from lexical overlap.",
        "The adjacent JSON packet binds annotations to the exact evaluated inputs and predictions.",
        "",
    ]
    for record in bundle["records"]:
        lines += [
            f"## {record['case_id']} / {record['entity']['entity_id']} / seed {record['seed']}",
            "",
            f"Record: `{record['record_id']}`",
            "",
            f"Cutoff: {record['cutoff']}; resolution: {record['resolution']}",
            "",
            record["hypothesis"],
            "",
        ]
        for citation in record["citations"]:
            lines += [
                f"Source: {citation['doc_id']} [{citation['span_start']}:{citation['span_end']}], {citation['doc_date']}",
                "",
                "```text",
                str(citation["text"]).replace("```", "` ` `"),
                "```",
                "",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--units", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args(argv)
    try:
        for path in (args.report, args.out, args.annotations):
            if path is not None:
                require_external(path, public_roots())
        bundle = packet(
            args.report, load_cases(units=args.units, manifest=args.manifest)
        )
        annotations = (
            json.loads(args.annotations.read_text())
            if args.annotations
            else {
                "version": 1,
                "packet_digest": digest(bundle),
                "reviews": [
                    {
                        "record_id": row["record_id"],
                        "verdict": "unreviewed",
                        "reviewer": "",
                        "reason": "",
                    }
                    for row in bundle["records"]
                ],
            }
        )
        summary = summarize_reviews(bundle, annotations)
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "packet.json", bundle)
        write_json(args.out / "reviews.json", annotations)
        write_json(args.out / "summary.json", summary)
        (args.out / "packet.md").write_text(render(bundle), encoding="utf-8")
        print(json.dumps(summary))
        return 0
    except Exception as exc:
        print(f"Evidence review aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
