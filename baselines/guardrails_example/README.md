# Guardrails example — a local citation rail for Track 4 agents

## Executive summary (read this first)

A small, optional, participant-side rail you can run on your OWN draft answer before
submitting. It performs two checks: every cited document must exist in the frozen corpus with
`doc_date <= cutoff_date`, and every claim must carry a well-formed `(doc_id, span_start,
span_end)` that resolves inside the cited document's text. These are exactly the mistakes the
stale-filing adversarial variants are built to elicit, so catching them locally reduces YOUR
gate failures. **The rail is advisory.** The competition's embargo and faithfulness gates are
deterministic, organizer-side code and are the authority on every submission — passing this
rail guarantees nothing about scoring; it only stops you from submitting a claim that would
certainly fail.

## What's here

| File | What |
|---|---|
| `citation_rail.py` | The checks, pure standard library. `load_corpus`, `filter_retrieved` (retrieval-time date rail), `check_answer` (submission-time rail → findings list). |
| `demo.py` | Offline demo on `units/t4-EXAMPLE-eps-beat`: an agent "accidentally" cites a live-fetched post-cutoff snippet and emits one malformed span; the rail flags both, the clean claim passes. |
| `rails/` | Illustrative [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) wiring of the same checks as an output rail (`config.yml`, `flows.co`, `actions.py`). Requires `pip install nemoguardrails`; the demo does not. |
| `tests/` | Unit tests for the rail checks. |

## Run the demo

```bash
# from the repo root
python -m baselines.guardrails_example.demo
```

Expected output ends with `DEMO PASS — the rail caught exactly the planted problems.` The
planted post-cutoff document is synthetic text invented for the demo, marked as such in the
source.

## Using the rail in your own agent

```python
from baselines.guardrails_example.citation_rail import (
    load_corpus, filter_retrieved, check_answer,
)

corpus = load_corpus(unit_dir / "corpus")
usable, stale = filter_retrieved(list(corpus.values()), task["cutoff_date"])
# ... retrieve/reason over `usable` only ...
findings = check_answer(draft_answer, corpus, task["cutoff_date"])
if findings:
    ...  # drop or repair the flagged claims, then redraft
```

Two placements, use both: filter the retrieval pool up front so stale material never reaches
the reasoning step, and re-check the assembled answer just before writing it out.

## What this rail does NOT do

- It does not run NLI — a claim can pass both checks and still fail the faithfulness gate if
  the cited span doesn't actually support it.
- It does not validate the full answer schema (use `templates/answer.example.json` and
  `qfbench2_common/schemas/analysis.schema.json` for that).
- It is not part of scoring or admissibility, and never will be — organizer-side gates are
  computed independently of anything you run locally.
