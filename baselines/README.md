# Track 4 Baselines — Explainability

## Executive summary (read this first)

Track 4 asks agents to predict a target per row in a table of entities — for example, whether
each company in a cross-section will beat its EPS estimate — grounded in a frozen evidence corpus.
This directory describes three reference baselines that span the performance spectrum from
text-blind to fully retrieval-grounded. **Beat the text-blind baselines** (TabPFN and gradient
boosting) to show that reading the evidence corpus adds value. **Beat the RAG baseline** to show
that better reasoning, retrieval, or calibration is possible. All baselines run inside a Docker
container on the restricted eval network: no open internet — the only egress is model-API
traffic through the organizer's audited proxy to the
organizer-hosted `$MODEL_ENDPOINT`. The shipped minimal baseline needs no network at all, so it
also runs under a local `--network=none` smoke run.

### Release status

| Baseline | Status |
|----------|--------|
| Minimal runnable RAG baseline (`baseline_agent/`) | **Shipped & runnable** — pure standard library, no model weights; produces a schema-valid `answer.json`. This is the agent the quick-start commands invoke. |
| TabPFN / gradient-boosting text-blind baselines | **Specification only — not yet released.** No code shipped; see the descriptions below for the intended design. |
| Strong RAG baseline (`strong_rag_baseline/`) | **Runnable.** BM25 retrieval and task-aware prediction through the official `$MODEL_ENDPOINT` using `$MODEL_NAME`. No endpoint or `--mock` selects the offline fallback; `--local-llama` is an opt-in development experiment. All tasks use the same evidence and target handling. Dense retrieval and a learned calibration head are not shipped. See its README for verification limits. |
| Citation rail example (`guardrails_example/`) | **Shipped & runnable** — optional participant-side pre-submission checks (cited `doc_date <= cutoff`, well-formed spans) with an offline demo and illustrative NeMo Guardrails wiring. Advisory only; the organizer-side gates are the authority. Not a baseline agent — a rail you can bolt onto your own. |

The shipped minimal baseline trades predictive strength for zero dependencies: lexical retrieval
(no BM25/dense index), a rule-based EPS classifier (no LLM), and a fixed-band interval (no learned
calibration head). It exists to make the quick-start commands work end-to-end and to provide a
schema-valid reference output, not to be a competitive submission.

---

## Three baselines

### Baseline 1 — TabPFN (text-blind tabular, classification only)

**What it does.** TabPFN (Tabular Prior-Fitted Networks) reads only the numeric and categorical
feature columns from the entity table in `task.json`. It ignores the evidence corpus entirely.
It is a transformer pre-trained on thousands of synthetic tabular datasets and can make
predictions on small tables (a few hundred rows) without any task-specific training.

**Why it is here.** TabPFN is the state-of-the-art text-blind baseline for small cross-sections.
If your agent cannot outperform TabPFN on predictive quality, the evidence corpus and agentic
reasoning have added no value on tabular features alone.

**Limitation.** Text-blind agents cannot produce grounded citations. A text-blind submission
cannot pass the faithfulness gate; an inadmissible unit scores `W = -0.27`. TabPFN is listed to establish
the text-blind predictive floor, not as a submittable strategy.

Expected performance: classification accuracy ~0.45–0.55, faithfulness gate pass rate: 0%,
composite score: ineligible.

### Baseline 2 — Gradient Boosting (text-blind tabular, all target types)

**What it does.** XGBoost or LightGBM trained on the numeric and categorical feature columns.
This is the dominant classical method for tabular data in industry and competitions. Unlike
TabPFN, it supports classification, regression, and ranking targets natively.

**Why it is here.** Gradient boosting is the most competitive text-blind baseline for larger
cross-sections. It also provides a skill-score denominator for regression tasks (the
`baseline_MAE` used in the scorer's skill computation is derived from a gradient-boosting
prediction, not a naive mean, for fairer comparison).

Expected performance: classification accuracy ~0.48–0.58, faithfulness gate pass rate: 0%,
composite score: ineligible.

### Baseline 3 — Retrieval-Augmented LLM-over-Rows with Calibration Head (submittable)

**What it does.** For each entity row in the table, the agent:

1. Uses the `corpus_ref` pointer from the entity row to retrieve relevant passages from the
   frozen corpus using hybrid BM25 + dense retrieval (BAAI/bge-m3 or equivalent).
2. Passes the top-K retrieved passages plus the entity's tabular features to an LLM with a
   structured prompt. Official inference uses the organizer-hosted `$MODEL_ENDPOINT`
   (OpenAI-compatible), with `$MODEL_NAME` supplied by the harness. BYO adapters are also served
   by the organizer; see the submission-category note below.
3. Generates a prediction (label or numeric estimate), a claim sentence, and a citation for each
   material statement.
4. A **calibration head** (a small quantile regression model) converts the LLM's raw confidence
   and retrieval strength into a calibrated 90% prediction interval.

This is the only baseline that can produce grounded citations and therefore the only one that can
be eligible for the composite score.

This architecture is a design sketch, not a measured score for the shipped agent. Public practice
smoke runs cannot establish prediction quality because their resolved outcomes are not available.

---

## Interface contract

All three baselines implement the same `analyze` CLI:

```bash
analyze \
  --task   /input/task.json \
  --corpus /input/corpus \
  --out    /output/answer.json
```

| Argument | Type | Description |
|----------|------|-------------|
| `--task` | file path | Path to the unit `task.json` (read-only) |
| `--corpus` | directory | Path to the frozen corpus directory containing `*.json` docs |
| `--out` | file path | Destination for the output `answer.json` |

Constraints:
- Restricted network is enforced during scoring: no open internet; egress only via the audited
  proxy to the organizer-hosted `$MODEL_ENDPOINT` only; vendor model APIs are refused. Local smoke
  runs without the eval network fall back to `--network=none`, so agents must degrade gracefully
  when model APIs are unreachable.
- The corpus directory is read-only at scoring time.
- Wall-clock budget: 10 minutes per unit. Exceeding this budget causes a timeout failure.
- Output must conform to `analysis.schema.json`.

---

## RAG baseline architecture

### 1. Corpus Indexer

Reads corpus documents, excluding manifest indexes, using their `text` field or the scorer's
`spans[].text` join convention. The runnable indexer builds BM25 chunks with exact character
offsets; the hybrid dense component below is an unimplemented extension.

### 2. Retriever

For each entity row, retrieves the top-K (default K = 10) passages by hybrid score:

```
hybrid_score(span) = alpha * bm25_score(span) + (1 - alpha) * dense_score(span)
```

Default `alpha = 0.4`. Only spans from documents with `doc_date <= cutoff_date` are considered,
enforcing the embargo at retrieval time.

### 3. Reader / Reasoner

The top-K retrieved spans plus the entity's tabular features are assembled into a prompt:

```
[SYSTEM]
You are a financial analyst. Predict the target for each entity using only the evidence
provided. For each material claim, cite the exact document and character span.

[ENTITY]
entity_id: AAPL
sector: Information Technology
consensus_eps: 1.50
threshold_pct: 0.05

[EVIDENCE]
[1] EDGAR_0000320193_10Q_20240202 spans 295-627:
    "Services net sales were $23.1 billion..."
[2] ...

[TASK]
Predict: will AAPL beat, miss, or land inline with consensus EPS (threshold 5%)?

[OUTPUT FORMAT]
{"label": "beat|miss|inline", "point_forecast": <float>, "claims": [...]}
```

The LLM uses the model settings in `strong_rag_baseline/config.py`; the model identity comes from
the harness. Official requests go to `$MODEL_ENDPOINT` through the audited proxy, with
vendor-side tools (web search, code execution, retrieval) disabled. Local GGUF inference is an
explicit development option, not the official serving contract.

### 4. Calibration Head

A quantile regression model that converts the LLM's confidence and retrieval strength into a
calibrated 90% prediction interval:

**Inputs:**
- `llm_confidence`: softmax probability assigned to the predicted label.
- `retrieval_strength`: mean cosine similarity of the top-K retrieved spans.
- `tabular_uncertainty`: spread of predictions from the gradient-boosting ensemble.

**Output:** `lo`, `hi` — lower and upper bounds of the 90% prediction interval.

**Training:**

```python
from sklearn.linear_model import QuantileRegressor

qr_lo = QuantileRegressor(quantile=0.05, alpha=0.01).fit(X_train, y_train)
qr_hi = QuantileRegressor(quantile=0.95, alpha=0.01).fit(X_train, y_train)
```

Pre-trained weights would live at `baselines/weights/calibration_head.joblib` (specification only
— not shipped; the runnable minimal baseline uses a fixed-band interval instead of a learned head).

### 5. Output Formatter

Assembles `answer.json` with `entity_predictions` (one object per entity row), each containing
`label`, `point_forecast`, `interval`, and `claims`. Applies a second embargo filter: silently
drops any citation whose `doc_date > cutoff_date` before writing output. The filter runs before
the >=1-claim fallback, so a dropped citation is replaced with the newest embargo-eligible
document rather than leaving the entity schema-invalid.

The filter only catches citations it can date. A citation whose `doc_id` the agent cannot resolve,
or whose document carries no `doc_date`, passes through it — and since 2026-08-22 the scorer counts
both of those as embargo **violations**, which is ineligibility, not a score penalty. Surviving this
filter is not the same as being eligible.

---

## Faithfulness requirements

| Requirement | Details |
|-------------|---------|
| Non-empty citations | Every claim must include at least one citation with a `doc_id` that appears in `manifest.json`. |
| Valid span offsets | `span_start` and `span_end` must be non-negative integers; `text[span_start:span_end]` must resolve to a non-empty string. |
| Embargo compliance | Every cited `doc_date` must be `<= cutoff_date` from `task.json`. |
| NLI entailment | The judge compares a cited span with the canonical hypothesis built from the submitted prediction, not the claim prose. The supported fraction uses the entity roster as denominator; thresholds come from the unit card. A local judge preview needs version and environment information to be comparable with production. |

---

## Models and deployment

| Role | Model | Licence | Notes |
|------|-------|---------|-------|
| Retrieval encoder | `BAAI/bge-m3` | MIT | 1.5 B params; supports dense, sparse, and multi-vector retrieval |
| Reader / Reasoner | Harness-provided `MODEL_NAME` | Organizer model disclosure | Access through the house endpoint; BYO uses an organizer-hosted adapter |
| NLI judge (local) | `cross-encoder/nli-deberta-v3-large` | MIT | Use before submission to estimate faithfulness score offline |
| Tabular (text-blind) | `TabPFN` | MIT | Best for small cross-sections (< 1000 rows); classification only |

The optional retrieval and tabular models above are development references, not bundled
components. Check current organizer rules and licenses before adding them. No HuggingFace Hub
downloads are possible at scoring time. Disclose every model used, its revision and training
cutoff in submission metadata; reader deployment follows the category note below.

---

## Repository layout

Shipped (runnable today):

```
baselines/
  README.md                      # this file
  baseline_agent.py              # thin shim: `python baselines/baseline_agent.py ...`
  baseline_agent/
    __init__.py
    indexer.py                   # corpus indexer (concatenates spans like the scorer)
    retriever.py                 # embargo-aware lexical retrieval (per entity row)
    reader.py                    # rule-based EPS classifier + point/interval
    formatter.py                 # entity_predictions schema marshalling
    cli.py                       # `analyze` CLI entry point
  Dockerfile                     # house-endpoint agent with offline fallback; verb `analyze`
  analyze.py                     # image ENTRYPOINT; accepts leading `analyze`
  scripts/ensure_gguf.sh         # local-dev GGUF fetch only (not copied into the image)
  models/                        # optional local GGUF home; weights are not committed
  smoke_image.sh                 # actual Docker build+run; fails if Docker is unavailable
  requirements.txt               # dependency notes (baseline is std-lib only)
  tests/
    test_cli_exemplar.py         # end-to-end test against t4-EXAMPLE-eps-beat
```

Specification only — **not yet released** (described above, no code shipped):

```
  baseline_agent/calibration.py  # learned calibration head wrapper
  baseline_agent/tabular_baselines.py  # TabPFN + gradient-boosting text-blind baselines
  weights/calibration_head.joblib      # pre-trained quantile regression weights
  tests/test_indexer.py, test_retriever.py, test_formatter.py
```

---

## Submission checklist

Before submitting your Docker image, verify every item:

- [ ] **1. Docker image built and tested locally.**
  ```bash
  docker build --platform linux/amd64 -f baselines/Dockerfile -t t4-analyze:latest baselines
  ```

- [ ] **2. `analyze` CLI works with the exemplar unit.**
  ```bash
  docker run --rm --platform linux/amd64 --network=none \
    -v $(pwd)/units/t4-EXAMPLE-eps-beat:/input:ro \
    -v /tmp/t4-out:/output \
    t4-analyze:latest \
    analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json
  cat /tmp/t4-out/answer.json
  ```

- [ ] **3. Faithfulness pre-check passes.**
  ```bash
  python faithfulness/judge.py --answer /tmp/t4-out/answer.json \
    --unit units/t4-EXAMPLE-eps-beat
  ```

- [ ] **4. All citations have `doc_date <= cutoff_date`.**
  ```bash
  python -c "
  import json
  t = json.load(open('units/t4-EXAMPLE-eps-beat/task.json'))
  a = json.load(open('/tmp/t4-out/answer.json'))
  cutoff = t['cutoff_date']
  corpus_dir = 'units/t4-EXAMPLE-eps-beat/corpus'
  for ep in a.get('entity_predictions', []):
      for claim in ep.get('claims', []):
          doc = json.load(open(f'{corpus_dir}/{claim[\"doc_id\"]}.json'))
          assert doc['doc_date'] <= cutoff, f'Embargo violation: {claim[\"doc_id\"]}'
  print('Embargo check: OK')
  "
  ```

- [ ] **5. Evidence trace is non-empty and human-readable.**

- [ ] **6. Network behaviour verified.**
  Official scoring runs on the restricted eval network (audited-proxy egress to the house endpoint
  only; every connection logged). Verify your agent has no open-internet dependency and
  degrades gracefully when no network is present (the local smoke fallback):
  ```bash
  bash baselines/smoke_image.sh /tmp/t4-out
  ```
  This command tests the offline fallback in the actual image and requires Docker. Run
  `python baselines/analyze.py analyze --task ... --corpus ... --out ... --mock` separately
  for a local CLI check. Validate endpoint access separately with the official harness.
  Neither check proves production NLI admission or prediction quality.

Submission categories follow the organizer's
[BYO starter pack](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/AGENTS.md#bringing-your-own-model-adapter-only-rank--64)
and [descriptor guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/SUBMISSION-DESCRIPTOR.md).
The default house-endpoint path is `api`. Official BYO supplies one LoRA adapter, rank at most 64,
for the organizer-hosted Nemotron base; the organizer extracts it and starts the server.
Participants use the supplied endpoint and model name, and do not start vLLM themselves.
The older full-weights wording in `SUBMISSION_CLI.md` conflicts with that newer guidance.
A submission with no model calls can still declare a deterministic engine using the documented
legacy `byo-small`, `access: local`, and `none-deterministic-engine` entry.
`--local-llama` remains a development experiment only.

---

## Evaluation metrics (recap)

| Metric | Weight | Description |
|--------|--------|-------------|
| Predictive quality (classification) | 70% | Fraction of entity rows with correct `label`. |
| Predictive quality (regression) | 70% | Skill score = 1 - MAE / baseline_MAE, clamped to [0, 1]. |
| Predictive quality (ranking) | 70% | Spearman rank correlation rescaled to [0, 1]. |
| Calibration penalty | 30% | **Subtracted**, not added: `\|interval_coverage - interval_level\|`, where `interval_coverage` is the fraction of entity rows whose true value falls in [lo, hi]. Coverage is a target to hit, not a quantity to maximise. |
| Faithfulness (gate) | — | Fraction of the **entity roster** whose prediction is entailed by a span cited for it; must be >= 0.80 to be eligible. |

Composite = 0.7 × predictive_quality − 0.3 × |interval_coverage − interval_level| (interval_level
= 0.90). The calibration term is a coverage **penalty**, not a reward — adding raw coverage would
incentivise trivially wide intervals. Higher composite wins. Leaderboard sorted `desc`. Ineligible
units (failed faithfulness or embargo) receive `W = -0.27` and remain in the denominator.
The organizer confirmed this in the [scoring correction](https://github.com/Agenthon-2026/track4-analysis-public/issues/1#issuecomment-5534948217).
Units whose resolved outcome
has no numeric target (pure-label tasks) have no calibration leg: the coverage term is dropped and
composite = 0.7 × predictive_quality.
