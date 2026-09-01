# Strong RAG baseline — BM25 retrieval + house model over `$MODEL_ENDPOINT`

## Executive summary (read this first)

The track's reference retrieval-augmented agent (Baseline 3 in `../README.md`): for each entity
it retrieves the top-K span-level chunks from the frozen corpus with BM25 (embargo enforced at
retrieval time). When `$MODEL_ENDPOINT` is unset — the local `--network=none` smoke, and
`--mock` — it does **not** emit empty evidence. It runs an extract-then-predict reasoner that
reads `target.type` and the legal label list from the task (both published `task.json` shapes),
pulls numbers and polarity cues from an embargo-safe window, and cites that exact window so the
NLI hypothesis (built from the submitted label / `point_forecast` / interval, never from claim
prose) has a chance of being entailed. When the harness injects `$MODEL_ENDPOINT`, the house
model still runs; a reply that yields zero grounded claims is filled by the same reasoner.
One agent for all units, no `family` dispatch; deterministic given the corpus and seed.

**Status: extract-then-predict is the offline path.** Schema-valid, embargo-safe answers on
every public-dev unit without a model. The reasoner now keeps the submitted
`label` / `point_forecast` / interval tokens inside the cited span (degenerate
`lo = hi = point` unless the span writes an explicit range such as
`ranged from 2.32 to 2.67`), and prefers NOTES / diluted-EPS / going-concern
windows over 10-Q headers. Predictive quality can still improve when a house
model is present; the faithfulness-first milestone is the reasoner.

## Run

```bash
# standard interface contract (extract-then-predict when MODEL_ENDPOINT is unset)
python -m baselines.strong_rag_baseline.cli \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# same reasoner, explicit (no house model)
python -m baselines.strong_rag_baseline.cli \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json \
  --mock

# harness shape — the verb is optional when you invoke the module by hand
python -m baselines.strong_rag_baseline.cli analyze \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json
```

The container command the official harness issues is still `analyze --task --corpus --out`.
Point the image entrypoint at this module (or install a console script that delegates to
`baselines.strong_rag_baseline.cli:main`) so the verb arrives as the first argument.

Environment:

| Var | Meaning | Default |
|---|---|---|
| `MODEL_ENDPOINT` | OpenAI-compatible base URL (harness-injected at scoring time) | — (required unless `--mock`) |
| `MODEL_NAME` | model id sent in the request — this is what the harness injects (see `SUBMISSION_CLI.md`, container environment contract) | empty |
| `MODEL_ID` | local-dev fallback for `MODEL_NAME`; read only when `MODEL_NAME` is unset | empty |
| `MODEL_TOKEN` | bearer token, if the endpoint needs one (local-dev convenience — not part of the published container contract) | none |
| `T4_SEED` | seed forwarded to the model | `20260731` |
| `T4_TOP_K` | retrieved chunks per entity | `10` |
| `T4_MODEL_TIMEOUT_S` / `T4_MODEL_RETRIES` | per-call timeout / retry count | `60` / `3` |

Local model example: `ollama serve` + `MODEL_ENDPOINT=http://localhost:11434/v1 MODEL_ID=qwen2.5:7b`.

## Design

| Module | Role |
|---|---|
| `indexer.py` | One chunk per corpus span; global offsets follow the scorer's join-with-space convention, so every chunk is citation-ready as-is |
| `retriever.py` | Pure-Python Okapi BM25; docs with missing or post-cutoff `doc_date` dropped before scoring; ties break by `(doc_id, span_start)` |
| `client.py` | stdlib HTTP client for `/chat/completions` (temp 0, seed, retries) + `MockModelClient` for tests |
| `prompts.py` | Per-target-type prompt; demands one JSON object with verbatim quotes |
| `span_finder.py` | Locates quotes as exact substrings (length-preserving curly-quote normalization); never trusts model offsets |
| `schema.py` | Reads `target.type` / top-level `target_type`, the legal label list, and `interval_level` from the task — both published shapes |
| `reasoner.py` | Extract-then-predict: embargo-safe entity windows → numbers / polarity → prediction whose tokens are in the cited span |
| `agent.py` | Orchestration; ungroundable quotes fall back to the source chunk's known-good offsets or are dropped; empty model evidence is replaced by the reasoner; off-vocabulary labels and missing intervals get deterministic fallbacks |
| `formatter.py` | Final answer assembly (`target_type` from the task) + hard self-check (spans resolve, intervals complete, `notes` is an object) |

**BM25 only, no dense retrieval** (deviation from the Baseline-3 sketch in `../README.md`): the
eval sandbox's restricted network cannot fetch embedding weights at run time, so a lexical index
keeps the agent reproducible everywhere. The binding constraint is build-time vendoring: a dense
index is permitted if its weights are bundled in the image (`byo-small`/`byo-large` in
`SUBMISSION_CLI.md`), because nothing can be downloaded at run time. The chunking already
targets the corpus's natural citable units (rendered-table NOTES lines, per-span passages),
which recovers much of what dense retrieval would add on these corpora.

## Acceptance

- [x] Schema-valid `answer.json` on every unit under `units/` (extract-then-predict, no model)
- [x] Embargo: every cited `doc_date` is `<= cutoff_date`; undated / unknown ids are not cited
- [x] Labels come from `target.labels`, not a hardcoded EPS vocabulary
- [ ] ≥0.80 citation faithfulness under the pinned judge on all 11 public-dev units (run
      `python faithfulness/judge.py --answer … --unit …`; the NLI hypothesis is the
      submitted prediction, including the interval clause). This machine is Python 3.12
      and cannot install `qfbench2-common` (requires ≥ 3.13), so the official gate was
      not executed here. What the reasoner now guarantees locally, and what still
      needs a DeBERTa pass:

      | Check | Public-dev status |
      |---|---|
      | Schema-valid `answer.json`, full roster, `target_type` from the task | all 11 |
      | Embargo: cited `doc_date <= cutoff` | all 11 |
      | Labels from `target.labels` only | all classification units |
      | `fmt_number(point/lo/hi)` is a substring of the cited span | all 11 (tested) |
      | CIK rows cite only that issuer's filings (ticker `WE` ≠ English "we") | credit / postearn / yoy |

      Remaining NLI risk, with the next patch for each:

      1. Classification labels whose surface form never appears (`credit_event`,
         `positive_reaction`) must be entailed from cues such as "going concern" /
         "record quarter". Next: a local DeBERTa reranker over (span, label) once
         the pinned weights are on the machine.
      2. The interval clause still says "prediction interval" even when the span
         says "ranged from 2.32 to 2.67" or is a degenerate `[point, point]`.
         Next: prefer explicit range windows, already done; rerank survivors
         with the same judge the gate uses.
      3. Bank / YoY 10-Qs rarely write a standalone diluted-EPS figure; the
         reasoner then cites a nearby number that is in-span but weakly related
         (`$11.4 billion of other debt`). Next: neighbourhoods around
         "diluted earnings per share" plus a magnitude prior from
         `prior_year_q_eps` / `consensus_eps` when those fields exist.

      `qfbench2-smoke --profile smoke` (non-rankable lexical proxy) should be run
      on Python ≥ 3.13 with the toolkit pin from this repo's CI workflow. Do not
      read a smoke "admissible" as a production gate pass.
- [ ] Predictive quality strictly above `baseline_agent/`
- [ ] Runs as-is on `sample-tasks/track4-analysis/` and passes `evaluation/check_submission.py`
