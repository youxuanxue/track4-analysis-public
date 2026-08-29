"""Demo: the citation rail catching a planted stale citation and a malformed span.

Runs entirely offline against the public example unit. It simulates an agent
that (wrongly) added a live-fetched, post-cutoff press snippet to its retrieval
pool, cited it, and also emitted one claim with broken span offsets. The rail
flags both before submission; the one clean claim passes.

Usage::

    python -m baselines.guardrails_example.demo \
        [--unit units/t4-EXAMPLE-eps-beat]

Exit code 0 iff the rail caught exactly the planted problems.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .citation_rail import CorpusDoc, check_answer, filter_retrieved, load_corpus

#: A SYNTHETIC post-cutoff document (this text is invented for the demo). It
#: stands in for something the agent fetched live instead of using the frozen
#: corpus — the classic stale-evidence mistake the Family 5 traps elicit.
_STALE_DOC = CorpusDoc(
    doc_id="LIVE_press_20240502",
    text=(
        "SYNTHETIC DEMO TEXT — Apple Inc. reported second quarter results on "
        "May 2, 2024, with diluted EPS of $1.53 versus consensus of $1.50."
    ),
    doc_date="2024-05-02",
)


def build_draft_answer(task: dict, corpus: dict[str, CorpusDoc]) -> dict:
    """A draft answer with one clean claim and two planted problems."""
    # Clean claim: exact span located in a real, pre-cutoff corpus document.
    clean_doc = corpus["EDGAR_0000320193_8K_20240201"]
    needle = "Apple Inc. today an"
    start = clean_doc.text.find(needle)
    assert start >= 0, "demo needle not found in corpus text"
    end = min(start + 200, len(clean_doc.text))

    return {
        "task_id": task["task_id"],
        "schema_version": task.get("schema_version", "3"),
        "entity_predictions": [
            {
                "entity_id": "AAPL",
                "label": "beat",
                "point_forecast": 1.55,
                "interval": {"level": 0.90, "lo": 1.42, "hi": 1.68},
                "claims": [
                    {  # clean — should pass the rail
                        "doc_id": clean_doc.doc_id,
                        "span_start": start,
                        "span_end": end,
                        "claim": "Apple announced Q1 FY2024 results in its February 1, 2024 press release.",
                    },
                    {  # planted stale citation — cites the live-fetched doc
                        "doc_id": _STALE_DOC.doc_id,
                        "span_start": 0,
                        "span_end": 120,
                        "claim": "Apple's reported Q2 FY2024 diluted EPS beat consensus.",
                    },
                    {  # planted malformed span — end before start
                        "doc_id": clean_doc.doc_id,
                        "span_start": 500,
                        "span_end": 120,
                        "claim": "Services revenue set a record in the quarter.",
                    },
                ],
            }
        ],
        "notes": {"demo": "draft answer with planted rail violations"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path("units/t4-EXAMPLE-eps-beat"),
        help="Path to a public unit directory (default: the shipped example).",
    )
    args = parser.parse_args(argv)

    task = json.loads((args.unit / "task.json").read_text(encoding="utf-8"))
    cutoff = task["cutoff_date"]
    corpus = load_corpus(args.unit / "corpus")

    print(f"Unit: {task['task_id']}  cutoff_date: {cutoff}")
    print(f"Frozen corpus: {len(corpus)} document(s)\n")

    # Rail 1 at retrieval time: the stale live-fetched doc never reaches reasoning.
    pool = list(corpus.values()) + [_STALE_DOC]
    usable, stale = filter_retrieved(pool, cutoff)
    print("Retrieval-time date rail:")
    for doc in stale:
        print(f"  dropped {doc.doc_id!r} (doc_date={doc.doc_date}) — after cutoff")
    print(f"  kept {len(usable)} of {len(pool)} docs\n")

    # Rail 2 at submission time: check the draft answer. The demo agent cites the
    # stale doc anyway (it skipped rail 1), so the answer-time rail must catch it.
    # The answer-time corpus view includes the stale doc exactly as the agent saw
    # it, which exercises the stale_doc (not unknown_doc) path.
    answer = build_draft_answer(task, corpus)
    corpus_with_stale = dict(corpus)
    corpus_with_stale[_STALE_DOC.doc_id] = _STALE_DOC
    findings = check_answer(answer, corpus_with_stale, cutoff)

    print("Submission-time rail findings:")
    for f in findings:
        print(f"  {f}")

    codes = sorted(f.code for f in findings)
    expected = ["bad_span", "stale_doc"]
    ok = codes == expected
    n_claims = sum(len(e["claims"]) for e in answer["entity_predictions"])
    print(
        f"\n{n_claims} claims checked: {n_claims - len(findings)} clean, "
        f"{len(findings)} flagged."
    )
    if ok:
        print("DEMO PASS — the rail caught exactly the planted problems.")
        print(
            "Reminder: this rail is advisory and local. The organizer-side "
            "embargo and faithfulness gates are the authority on every submission."
        )
        return 0
    print(f"DEMO FAIL — expected findings {expected}, got {codes}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
