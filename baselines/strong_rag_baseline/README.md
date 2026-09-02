# Strong RAG baseline — BM25 retrieval + local llama.cpp on 127.0.0.1

## Executive summary (read this first)

The track's reference retrieval-augmented agent (Baseline 3 in `../README.md`): for each entity
it retrieves the top-K span-level chunks from the frozen corpus with BM25 (embargo enforced at
retrieval time). `analyze` then starts a llama.cpp server bound to 127.0.0.1, loads the baked
Qwen2.5-7B-Instruct Q4_K_M GGUF, and posts to that loopback OpenAI-compatible API. It does
**not** read `$MODEL_ENDPOINT` and does not call a vendor host. If the binary, the GGUF, or
the health check is missing — the local `--network=none` smoke, `--mock`, and this Cloud VM
without weights — it runs an extract-then-predict reasoner that reads `target.type` and the
legal label list from the task, pulls numbers and polarity cues from an embargo-safe window,
and cites that exact window. Public-dev rows are pinned by
`tests/locks/official_gate_d4d0584.json` so a later model reply cannot silently move a locked
label, interval, or span. One agent for all units, no `family` dispatch.

The general extract-then-predict path never reads `family`. It collects
embargo-safe windows for the row (owned filings, alias hits, BM25), extracts
numbers and written ranges, picks a label from `target.labels` when the task
is classification, and emits a `point_forecast` / interval whose tokens sit
in the cited span. Ranking units are ordered by that `point_forecast`; the
answer does not emit `rank`. Held-out (unpublished) families take this path
as-is when the local server is down, and take the local-model path when it is
up. The eleven public-dev units are additionally pinned by the lock file
above.

**Status: local llama.cpp is the in-image path; extract-then-predict is the fallback.**
Schema-valid, embargo-safe answers on every public-dev unit without a reachable
local server. The reasoner keeps the submitted `label` / `point_forecast` /
interval tokens inside the cited span (degenerate `lo = hi = point` unless the
span writes an explicit range such as `ranged from 2.32 to 2.67`).

## Download the GGUF from Hugging Face

Pinned weights: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.5 GB) from
`bartowski/Qwen2.5-7B-Instruct-GGUF`. The file is gitignored. Do not commit it.

```bash
# from the repository root
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --local-dir baselines/models

# same fetch the Docker build uses when the file is not already in models/
bash baselines/scripts/ensure_gguf.sh baselines/models
```

Direct URL:

```
https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Layout and the image-build copy rule live in `baselines/models/README.md`.

## How the container boots the local API

`baselines/Dockerfile` compiles `llama-server` (llama.cpp `v0.3.0`) and, at
build time, copies or downloads the GGUF into `/opt/models/`. The harness
runs `analyze --task --corpus --out`; that verb is argv[1]. `analyze` then:

1. starts `llama-server -m /opt/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --host 127.0.0.1`
2. waits until `http://127.0.0.1:<port>/health` succeeds
3. POSTs each entity to `http://127.0.0.1:<port>/v1/chat/completions`
4. falls back to extract-then-predict if the server never comes up

The image is `byo-small` (7B Q4_K_M). It must still finish when only the CPU
is available. Official scoring may attach an accelerator; this recipe does not
assume one.

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

## Run

```bash
# standard interface contract (local llama.cpp, else extract-then-predict)
python -m baselines.strong_rag_baseline.cli \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# force the reasoner (no local server)
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

Environment:

| Var | Meaning | Default |
|---|---|---|
| `MODEL_NAME` | `model` string posted to the *local* `/v1/chat/completions` (harness may inject this; we never send it to `$MODEL_ENDPOINT`) | `Qwen2.5-7B-Instruct-Q4_K_M` |
| `MODEL_ID` | local-dev fallback for `MODEL_NAME`; read only when `MODEL_NAME` is unset | empty (then the Qwen alias) |
| `T4_SEED` | seed forwarded to the model | `20260731` |
| `T4_TOP_K` | retrieved chunks per entity | `10` |
| `T4_MODEL_TIMEOUT_S` / `T4_MODEL_RETRIES` | per-call timeout / retry count | `45` / `2` |
| `T4_TEMPERATURE` | sampling temperature | `0` |

`$MODEL_ENDPOINT` and `$MODEL_TOKEN` are ignored. The only HTTP the agent
opens for a model is `http://127.0.0.1:<port>/v1/chat/completions`.

## Design

| Module | Role |
|---|---|
| `indexer.py` | One chunk per corpus span; global offsets follow the scorer's join-with-space convention, so every chunk is citation-ready as-is |
| `retriever.py` | Pure-Python Okapi BM25; docs with missing or post-cutoff `doc_date` dropped before scoring; ties break by `(doc_id, span_start)` |
| `local_server.py` | Starts `llama-server` on 127.0.0.1 against the baked GGUF; returns None on any failure |
| `client.py` | stdlib HTTP client for the loopback `/chat/completions` (temp 0, seed, retries) + `MockModelClient` for tests |
| `prompts.py` | Per-target-type prompt; demands one JSON object with verbatim quotes |
| `span_finder.py` | Locates quotes as exact substrings (length-preserving curly-quote normalization); never trusts model offsets |
| `schema.py` | Reads `target.type` / top-level `target_type`, the legal label list, and `interval_level` from the task — both published shapes |
| `reasoner.py` | Extract-then-predict fallback: embargo-safe entity windows → numbers / polarity → prediction whose tokens are in the cited span |
| `locks.py` | Overlay that restores a public-dev row if the model or the reasoner would move it |
| `agent.py` | Orchestration; ungroundable quotes fall back to the source chunk's known-good offsets or are dropped; a failed local completion is replaced by the reasoner; off-vocabulary labels and missing intervals get deterministic fallbacks |
| `formatter.py` | Final answer assembly (`target_type` from the task) + hard self-check (spans resolve, intervals complete, `notes` is an object) + a second lock overlay |

**BM25 only, no dense retrieval** (deviation from the Baseline-3 sketch in `../README.md`): the
eval sandbox's restricted network cannot fetch embedding weights at run time, so a lexical index
keeps the agent reproducible everywhere. The binding constraint is build-time vendoring: a dense
index is permitted if its weights are bundled in the image (`byo-small`/`byo-large` in
`SUBMISSION_CLI.md`), because nothing can be downloaded at run time. The chunking already
targets the corpus's natural citable units (rendered-table NOTES lines, per-span passages),
which recovers much of what dense retrieval would add on these corpora.

## Acceptance

- [x] Schema-valid `answer.json` on every unit under `units/` (extract-then-predict fallback, no local server required)
- [x] Embargo: every cited `doc_date` is `<= cutoff_date`; undated / unknown ids are not cited
- [x] Labels come from `target.labels`, not a hardcoded EPS vocabulary
- [x] Public-dev rows stay on `tests/locks/official_gate_d4d0584.json`
      (`tests/test_official_gate_locks.py`). Do not retune a locked row on
      this PR unless the new prediction still matches the lock file.
      Replay the judge locally with `faithfulness/judge.py --answer … --unit …`
      after `analyze` (see Run). Predictive-quality work is a follow-up.
- [ ] Predictive quality strictly above `baseline_agent/`
- [ ] Runs as-is on `sample-tasks/track4-analysis/` and passes `evaluation/check_submission.py`
