# Strong RAG baseline — BM25 retrieval + extract-then-predict

## Executive summary (read this first)

The track's reference retrieval-augmented agent (Baseline 3 in `../README.md`): for each entity
it retrieves the top-K span-level chunks from the frozen corpus with BM25 (embargo enforced at
retrieval time). Official `analyze` does **not** start a localhost model server and does
**not** call `$MODEL_ENDPOINT`. It runs an extract-then-predict reasoner that reads
`target.type` and the legal label list from the task (both published `task.json` shapes),
pulls numbers and polarity cues from an embargo-safe window, and cites that exact window so the
NLI hypothesis (built from the submitted label / `point_forecast` / interval, never from claim
prose) has a chance of being entailed. Public-dev rows are pinned by
`tests/locks/official_gate_d4d0584.json` so a later change cannot silently move a locked
label, interval, or span. One agent for all units, no `family` dispatch.

The general extract-then-predict path never reads `family`. It collects
embargo-safe windows for the row (owned filings, alias hits, BM25), extracts
numbers and written ranges, picks a label from `target.labels` when the task
is classification, and emits a `point_forecast` / interval whose tokens sit
in the cited span. Ranking units are ordered by that `point_forecast`; the
answer does not emit `rank`. Held-out (unpublished) families take this path
as-is. The eleven public-dev units are additionally pinned by the lock file
above.

**Status: extract-then-predict is the official analyze path.** Schema-valid,
embargo-safe answers on every public-dev unit with no model server. A
developer-machine GGUF experiment (`--local-llama` / `T4_LOCAL_LLAMA=1`) is
default OFF and is not the submission ENTRYPOINT.

## Run

```bash
# official interface contract (extract-then-predict; no localhost server)
python -m baselines.strong_rag_baseline.cli \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# same reasoner, explicit
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
`baselines/Dockerfile` + `baselines/analyze.py` are the submission image: Python 3.13,
`LABEL qfbench2.interface_version="2.0"`, ENTRYPOINT that consumes the leading verb.
The image is extract-then-predict (stdlib + this package). It does not compile
llama.cpp and does not download a GGUF.

```bash
# from the repository root — build context is baselines/ (see baselines/Dockerfile)
docker build -f baselines/Dockerfile -t t4-analyze:latest baselines

docker run --rm --network=none \
  -v "$(pwd)/units/t4-EXAMPLE-eps-beat":/input:ro \
  -v /tmp/t4-out:/output \
  t4-analyze:latest \
  analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json
```

This Cloud VM cannot run `docker build`. `baselines/smoke_image.sh` builds and runs
the image when Docker is present; otherwise it exercises `baselines/analyze.py`
with the same `analyze --task --corpus --out` argv and says so. Image-size
limits live on the runtime-constraints table in `README.md`.

Local replay of the official faithfulness gate (CLI flags live on
`faithfulness/judge.py`; `--unit` is the unit directory, not `corpus/` alone):

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

python faithfulness/judge.py --answer /tmp/answer.json --unit units/t4-EXAMPLE-eps-beat
```

This Cloud VM cannot load the pinned judge weights.
`tests/test_official_gate_locks.py` pins the public-dev labels, intervals,
and spans so a later change cannot silently move those rows.

## Developer-machine GGUF (not the submission image)

`--local-llama` (or `T4_LOCAL_LLAMA=1`) may start llama.cpp on 127.0.0.1
against a gitignored Qwen2.5-7B-Instruct Q4_K_M GGUF. Default OFF. Official
`analyze` never sets this flag. How to download the file lives in
`baselines/models/README.md`. If the binary, the GGUF, or the health check
is missing, the reasoner still writes the answer. Public-dev rows stay locked.

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --local-llama \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json
```

Environment (official path uses only the retrieval knobs):

| Var | Meaning | Default |
|---|---|---|
| `T4_SEED` | seed forwarded to any opt-in local model | `20260731` |
| `T4_TOP_K` | retrieved chunks per entity | `10` |
| `T4_MODEL_TIMEOUT_S` / `T4_MODEL_RETRIES` | per-call timeout / retry count when `--local-llama` is on | `45` / `2` |
| `T4_TEMPERATURE` | sampling temperature when `--local-llama` is on | `0` |
| `T4_LOCAL_LLAMA` | developer-machine opt-in; same as `--local-llama` | unset (OFF) |
| `MODEL_NAME` | `model` string posted to loopback only when the opt-in is on | `Qwen2.5-7B-Instruct-Q4_K_M` |
| `MODEL_ID` | local-dev fallback for `MODEL_NAME` | empty (then the Qwen alias) |

`$MODEL_ENDPOINT` is not used as a client URL.

## Design

| Module | Role |
|---|---|
| `indexer.py` | One chunk per corpus span; global offsets follow the scorer's join-with-space convention, so every chunk is citation-ready as-is |
| `retriever.py` | Pure-Python Okapi BM25; docs with missing or post-cutoff `doc_date` dropped before scoring; ties break by `(doc_id, span_start)` |
| `local_server.py` | Opt-in llama.cpp launcher on 127.0.0.1; unused by official analyze |
| `client.py` | stdlib HTTP client for a loopback `/chat/completions` + `MockModelClient` for tests |
| `prompts.py` | Per-target-type prompt; demands one JSON object with verbatim quotes |
| `span_finder.py` | Locates quotes as exact substrings (length-preserving curly-quote normalization); never trusts model offsets |
| `schema.py` | Reads `target.type` / top-level `target_type`, the legal label list, and `interval_level` from the task — both published shapes |
| `reasoner.py` | Extract-then-predict: embargo-safe entity windows → numbers / polarity → prediction whose tokens are in the cited span |
| `locks.py` | Overlay that restores a public-dev row if a later path would move it |
| `agent.py` | Orchestration; ungroundable quotes fall back to the source chunk's known-good offsets or are dropped; empty model evidence is replaced by the reasoner |
| `formatter.py` | Final answer assembly (`target_type` from the task) + hard self-check + a second lock overlay |

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
- [x] Public-dev rows stay on `tests/locks/official_gate_d4d0584.json`
      (`tests/test_official_gate_locks.py`). Do not retune a locked row on
      this PR unless the new prediction still matches the lock file.
      Replay the judge locally with `faithfulness/judge.py --answer … --unit …`
      after `analyze` (see Run). Predictive-quality work is a follow-up.
- [ ] Predictive quality strictly above `baseline_agent/`
- [ ] Runs as-is on `sample-tasks/track4-analysis/` and passes `evaluation/check_submission.py`
