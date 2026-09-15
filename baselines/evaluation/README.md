## Executive summary (read this first)

This development tool runs the baseline and records what was actually measured.
Public practice inputs have no resolved outcomes, so their predictive scores remain empty.
Smoke admission does not measure production NLI support.
Private development outcomes are read after prediction and are never staged as agent inputs.
Evidence annotations are inspectable source mentions, not inferred financial relationships.

## Run

Use the same Python environment as the track scorer and shared toolkit.
The [CLI](./__main__.py) lists supported options:

```bash
python -m baselines.evaluation --help
python -m baselines.evaluation \
  --units units \
  --out "$HOME/t4-evaluation/public-baseline" \
  --mode grounded
```

The output directory must be new and outside all worktrees of this public repository.
It contains a Markdown report, a machine-readable report, and per-run answers,
diagnostics, canonical prediction hypotheses, and evidence candidate ledgers.
In model mode, the evidence ledger records the same retrieval strategy and configured budget
used for predictions. In grounded mode it is a separate retrieval diagnostic. Its
`used_for_prediction` field distinguishes these cases; it does not assert semantic support.

Use `--image` to run an already-built baseline image through Docker with no network.
The tool resolves its immutable image ID and compares its runtime source files with the
evaluator checkout before running; rebuild images whose runtime differs. Only declared task and corpus files
are copied into the container. No host data directories or Docker socket are mounted.
Returned artifacts are collected separately; only regular answer and diagnostics files
are imported. Container runtime faults abort evaluation instead of becoming scored failures.
This tests Docker execution, not the organizer's gVisor environment or model service.
The offline container uses conservative local CPU/memory caps. The runner records the
actual Docker configuration, out-of-memory state and process wall time, then checks them
against each unit's frozen card contract. These observations do not establish hardware
equivalence with the official environment.

Before prediction, the runner persists a complete input/seed/resource roster in
`run-plan.json`. Report loading verifies its digest and complete coverage, as well as
persisted result, answer and diagnostics bindings. Input changes during execution abort
the report. Cards without an explicit agent timeout retain unmeasured resource evidence;
a caller's timeout is not a substitute for the official contract. These local integrity
checks detect drift, not deliberate rewriting of all artifacts by their owner.

Without `--image`, the predictor runs in a separate local process with staged inputs.
This prevents accidental truth-path plumbing, but is **not a filesystem security sandbox**.
Use container mode for strict input isolation. Local `--mode model` requires a configured
house endpoint; the [submission contract](../../SUBMISSION_CLI.md) owns its configuration.
Neither endpoint URLs nor credentials nor raw model responses are included in reports.

## Private development data

Keep the roster, original outcomes and run artifacts in a separate private directory or
private repository. Pass its roster with `--manifest` instead of `--units`:

```json
{
  "version": 1,
  "cases": [
    {
      "id": "independent-case-a",
      "unit_dir": "inputs/case-a",
      "split": "test",
      "group": "independent-event-a",
      "truth_path": "truth/case-a.json"
    }
  ]
}
```

Paths resolve relative to the roster. Input units must satisfy the official card and
manifest checks. Outcomes use the existing track scorer's format; the adapter validates
the exact roster and refuses inconsistent reference data. Omitting `truth_path` leaves
prediction metrics unmeasured. A reference defect aborts the evaluation; it is never
silently removed from an average.

[Dataset validation](./dataset.py) rejects duplicated cases, event groups shared across
splits, and overlapping train/calibration/test periods. An earlier split's outcomes must
precede every cutoff in the later split. The tool does not certify data licensing,
historical data vintages or absence of leakage in trained model weights; retain their
provenance independently. The supplied repository tests use invented data only.

## Interpret results

`development_score` uses the installed official scorer. In smoke mode it deliberately
omits the production faithfulness gate and is useful only for local predictive comparisons.
Failed cases with development truth retain the canonical failure score in the denominator.
A missing score on any planned run suppresses the overall mean. Inspect `by_split` to
keep training diagnostics separate from held-out performance.

`official_score` remains null and `rankable` remains false for all local reports.
`nli_faithfulness` remains null under smoke; `lexical_faithfulness` is a diagnostic proxy.
When numeric truth and aligned predictions exist, `numeric_errors` records raw target-scale
absolute and signed errors. These diagnostics keep differences visible when the official
normalized quality clips to zero; they never replace the official scoring function.
The diagnostics also expose the MAE of the realized cross-section mean, the reference used
by regression skill. That mean is known only after resolution and is not an available
forecasting policy. A small raw MAE can still yield zero skill against this reference.
Production mode requires the organizer's configured judge and external outcomes and never
downgrades to smoke. The runner stops before inference when outcomes are undeclared or the
judge cannot be constructed.

Provenance includes the workspace source digest, Git revision and dirty state, Python,
installed scorer identity, and the actual shared-toolkit source digest. A package version
without a verified build stamp is identified as such. The Docker image ID identifies
participant runtime bytes separately from evaluator workspace bytes.

## Historical snapshots and evidence review

The development-only [historical builder](./historical.py) reads an external event specification
and retrieves Treasury, CPI, foreign-exchange and energy observations from ALFRED.
The supported families and series are declared in [`historical.py`](./historical.py).
It requires `curl` for acquisition; cached
snapshots can subsequently be rebuilt offline. Every series is requested separately, and its
returned column must carry the exact requested vintage date. This matters because a multi-series
request can silently return the latest vintage for some columns. Downloaded bytes, source URLs,
vintages and retrieval times are retained outside the public repository.

The specification has `version: 1` and an `events` array. Each event declares `id`, `cutoff`,
`resolution`, and `split` (train, calibration or test). The builder refuses overlapping event
windows and split leakage before downloading. It creates classification, regression and ranking
views of each event, with the same event group: those views are correlated, not additional
independent samples. The default family covers rates only; additional families do not establish
coverage of all hidden tasks or independence between contemporaneous market shocks.
The corpus is a deterministic extract of historical observations, not a financial forecast.
Only prediction inputs go under each unit; future snapshots and outcomes stay outside those units.

For a separate inflation benchmark, set the external specification's `family` to `cpi_mom`
and add `target_month` (an ISO date on the first of the month) to each event. The supported
seasonally adjusted series are declared in [`historical.py`](./historical.py). Each target
is the month-over-month percent change calculated from two index levels in the resolution
vintage; it is **not a certified first-release value**. Both levels use that same vintage,
so a revised denominator is handled consistently. The target month must be absent from
the cutoff snapshot, immediately follow its latest observation, and be the latest month
in the resolution snapshot. Invalid or incomplete events abort the build.

CPI inputs include prior monthly levels and the last available monthly change. Input tables
use observation dates; their document dates identify the snapshot's availability. CPI
components overlap, and the target views share a release event, so neither entity rows nor
views count as independent samples. Keep this domain's development and held-out rosters
separate and retain the same chronological split checks. The default Treasury build and
its existing cache URLs remain unchanged.

The market-return families use the latest common observation available in each requested
vintage. Returns divide the resolution level by the original cutoff reference, so later
revisions cannot change the starting level. Exchange-rate quotation directions remain as
published; yen per dollar is not silently inverted. The builder refuses stale snapshots,
insufficient recent history and resolutions with no observation after cutoff. These are
specified-vintage targets with publication lag, not certified first-release targets.
Private rosters carry the domain alongside the event group. Series metadata, attribution
and reuse links are recorded with fetched snapshots; retain the source-specific license
assessment separately before using a dataset for acceptance. Different domains sharing a
time window may remain dependent and must not inflate independent-event counts.

```bash
python -m baselines.evaluation.historical \
  --spec /private/evaluation/events.json \
  --cache /private/evaluation/alfred-cache \
  --out /private/evaluation/new-benchmark

python -m baselines.evaluation.review \
  --report /private/evaluation/run/report.json \
  --manifest /private/evaluation/new-benchmark/manifest.json \
  --out /private/evaluation/new-review

python -m baselines.evaluation.compare \
  --before /private/evaluation/baseline/report.json \
  --after /private/evaluation/candidate/report.json \
  --out /private/evaluation/comparison.json
```

Use `--units units` instead of `--manifest` to audit public runs. The review packet contains the
canonical hypothesis and resolved citation text for every entity. Its adjacent annotations start
as `unreviewed`; passing date/offset checks never creates semantic approval. Edited annotations
can be passed back with `--annotations` into a new review directory. A completed judgment requires
an attributed reviewer and reason, and annotations bind to the exact packet digest. Automated
or model-assisted judgments must be attributed as such. Historical reports without an original
answer hash explicitly record that limitation; their hypotheses and citation positions are checked.

Comparisons require identical complete case/seed rosters, input and truth digests, split/group
assignments, judge identity and toolkit bytes. Failed units stay in the mean. Bootstrap intervals
resample event groups, averaging correlated views and seeds inside each group. Inspect the test
split separately; the overall result includes every supplied split. Neither a smoke score nor
a small single-domain bootstrap interval establishes production faithfulness or generalization.
One event group may span several domains when they share a shock. The acceptance audit counts
that group once overall and once within each affected domain; adding domains does not multiply
the total independent-event count. Each case must retain its group, domain and target type
across repeated seeds.

Toolkit tag `v2.4.0` still reports package version `2.3.1`; use the recorded source digest and
Git installation identity to distinguish it. The updated tag allows model-free `models: []`
descriptors but does not provide a production NLI judge.

## Frozen-Prediction Interval Calibration

[`calibration.py`](calibration.py) fits interval radii on earlier training-event residuals
and writes adjusted answers for later calibration events. This is an external development
workflow for frozen local predictions; it does not alter the submission defaults
or ship fitted outcome-derived parameters in the public image. Unverified model reports, test
splits, overlapping windows, incomplete runs and mismatched prediction settings are refused.

First run the unchanged grounded policy on separate train and calibration manifests with
the same code, toolkit and retrieval settings. Reports predating recorded prediction settings
must be regenerated. Then run:

```bash
python -m baselines.evaluation.calibration \
  --fit-manifest /private/evaluation/train-manifest.json \
  --fit-report /private/evaluation/train-run/report.json \
  --apply-manifest /private/evaluation/calibration-manifest.json \
  --apply-report /private/evaluation/calibration-run/report.json \
  --out /private/evaluation/new-calibrated-run
```

The finite-sample quantile uses the maximum absolute error within each event. Correlated
entity rows, target views and repeated seeds cannot inflate the event count. Bins match the
declared family, target, units, horizon and roster size; insufficient events or missing bins
produce an error instead of an invented finite interval. The statistical coverage statement
assumes exchangeable event residuals, meaning that past and future error distributions can be
treated alike. Chronological financial data does not establish that assumption.

The tool verifies original answer and input hashes and recomputes fitting residuals through
the shared scorer. It writes all adjusted answers before opening application outcomes, then
rescoring uses the same official formulas. Points, labels, ranks and citations stay unchanged;
interval-width diagnostics accompany the new report. Fitted radii, original answers and reports
remain outside every public worktree. Changing intervals also changes the submitted hypothesis:
historical citations alone do not establish future bounds, and production NLI must be measured
separately. Local coverage on the calibration split is not a held-out or leaderboard claim.

For model predictions, also supply `--fit-runtime` and `--apply-runtime` with external local
launcher records. [`runtime.py`](runtime.py) defines their validation: each completed record
binds the exact report and manifest hashes to a clean code revision, a local launch command,
the runtime executable, weight shards, launcher identity, host and generation settings. The
executable and weight files are rehashed. Only the loopback port may vary between launches;
changing the context, backend or any other recorded setting requires new matching runs.
The model reports must include complete per-entity diagnostics with no grounded fallbacks.
Old launcher records lacking binary/report hashes cannot be used and must be regenerated.

These records are attestations from a trusted local launcher that owns the server, not
cryptographic proof of what an arbitrary remote service executed. The launcher must record
the actual command and effective settings, seal report hashes after evaluation, and mark
completion only after the owned processes have exited successfully. Calibration never
reuses grounded residuals for model predictions. Its artifact retains the runtime record
hashes and common model identity; it remains a development experiment outside the image.

## Internal acceptance and sealed batches

The [acceptance policy](acceptance-policy.json) defines internal sample, gain and
resource-margin targets. [The audit](acceptance.py) uses the existing comparator,
checks persisted run/answer/diagnostics artifacts, and reports PASS, FAIL or UNMEASURED for
measured subchecks. Missing complete-roster, resource/fault or production evidence
keeps the overall goals unmeasured; the audit does not promote a candidate.
The report's declared total or mean is never used to replace per-run measurements.

[Recovery exercises](faults.py) run the image's real analyze process against a synthetic
loopback service with no external network. The frozen protocol covers normal startup,
connection refusal, malformed HTTP response bodies, read timeouts and unit request-budget
exhaustion for each target type. Fixtures are invented forecasts, never modified competition
units or resolved outcomes. Server-observed request hashes/counts are compared with the
client ledger; output schema, complete entity coverage, citations and fallback reasons
are rechecked from original artifacts. This measures recovery, not predictive quality.

```bash
python -m baselines.evaluation.faults --image YOUR_LOCAL_IMAGE --out "$HOME/t4-evaluation/recovery"
python -m baselines.evaluation.acceptance --help
```

Supply the recovery report using the audit's `--before-faults` and/or `--after-faults`
options. Each report must bind the same immutable image and runtime source as its
corresponding evaluation report. The audit checks baseline G1 independently. Missing
recovery evidence remains unmeasured; changed artifacts or an incomplete protocol fail.
An aborted engineering exercise may be fixed and rerun into a new output directory;
this does not consume, replace or qualify as a fresh predictive acceptance batch.

Use an optional `domain` on private manifest cases to enable domain-level
comparisons. It is evaluator metadata, never an agent input. Missing domains on
older reports do not invent cross-domain generalization. Acceptance uses only test
reports, balanced event weights and independent event counts, not entity counts.

[Batch registration](batch.py) freezes a test-only manifest, policy, before/after
source and toolkit identities, input and truth digests, and seeds before invoking
prediction. A versions file maps `before` and `after` to objects containing `repo`,
`python` and `image` (null for a local process). The runner currently supports only
grounded/smoke runs with no model-request budget. Consult its CLI for arguments:

Registration can also freeze recovery reports for both roles. It verifies each image's
runtime source against its declared checkout, binds the recovery report hashes, and rechecks
the complete underlying fault artifacts before execution and decision. Replacing recovery
evidence after seeing acceptance results is refused. A batch without registered recovery
evidence keeps that G1 requirement unmeasured; batch decisions still do not execute promotion.

[The development lifecycle](lifecycle.py) applies a batch decision to the retained incumbent.
Initialize it in the same registry with a clean frozen version, then attach a newly registered
batch before either role starts, after opening a budgeted round with `start-round`.
The attached baseline must be the current incumbent, and
the candidate must be a different frozen version. After both runs and `batch decide`, use
`lifecycle resolve`: it recomputes the saved decision from the original receipts, reports and
recovery artifacts, verifies that both versions still exist unchanged, and updates the development
incumbent only when G1 and G2 pass. Failed or unmeasured acceptance closes the batch and retains
the incumbent. The initial version is explicitly a bootstrap baseline, not an accepted candidate.

```bash
python -m baselines.evaluation.lifecycle --help
python -m baselines.evaluation.lifecycle --registry /private/evaluation/registry initialize --help
python -m baselines.evaluation.lifecycle --registry /private/evaluation/registry start-round --help
python -m baselines.evaluation.lifecycle --registry /private/evaluation/registry attach --help
```

The lifecycle uses one append-only hash-linked journal, published atomically under the registry
lock. `status` derives the current version and rollback history from that journal. `rollback`
requires a reason and verifies the previous retained version before restoring it; if the only
previous version was the bootstrap baseline, it remains unqualified. `abandon` closes an interrupted
batch with a reason while preserving its event reservations and consumed-run markers. A pending
batch must be resolved or abandoned before another selection or rollback. These commands only
change local development state: they do not assign production qualification, deploy or submit.

Each round freezes one hypothesis, the incumbent, the acceptance policy and an external budget
file. [`start_round`](lifecycle.py) defines its candidate, run and reserved-time limits; the
candidate limit cannot exceed the [acceptance policy](acceptance-policy.json). Selection requires
room for both roles and counts the candidate even if the batch is later abandoned. The batch runner
reserves all planned runs and its maximum subprocess duration before publishing a start marker.
Reservations survive errors and crashes, are never refunded, and are checked again with the run
receipts. The subprocess uses the same duration calculated by `execution_budget` in
[`batch.py`](batch.py), including its report/shutdown allowance.

A candidate cannot be retried in the same round. `close-round` records the reason and retains the
spent budget history; it cannot discard a pending batch. Starting another round requires that the
previous one be closed. The supported runner remains offline grounded/smoke with no paid-call
budget. The budget covers registered acceptance execution, not arbitrary shell commands or earlier
development searches. Standalone diagnostic registries without a lifecycle remain available, but
their already-started runs cannot be attached later as budgeted acceptance evidence.

```bash
python -m baselines.evaluation.acceptance --help
python -m baselines.evaluation.batch --help
python -m baselines.evaluation.batch register --help
python -m baselines.evaluation.batch run --help
python -m baselines.evaluation.batch decide --help
```

Store the registry, manifests, version declarations and outputs outside all public
worktrees. Each registry reserves event IDs and input digests, even across renamed
cases or different seeds. Starting a role burns that run; a failed or interrupted
run must not be rerun on the same batch. Successful or participant-failed complete
runs receive a digest-bound receipt. `decide` verifies both receipts and writes an
exclusive decision file; later calls must inspect that file instead of recomputing
selection. A changed source, policy, roster or report is refused.

The registry prevents accidental replay and post-hoc substitution; it is not an
anti-tamper service against someone who controls its files. It does not establish
that the data were previously unseen by people, that event IDs are independent,
or that source licensing and cutoff provenance are valid. These still require
separate provenance evidence. Production eligibility, equivalence and independent confirmations
remain unmeasured until their evidence paths are implemented. Development-search budgets still
need integration with candidate selection; registered acceptance execution now has round limits.
The lifecycle does not declare the overall goal complete.
Saved decisions produced by a different evaluator revision may fail recomputation; keep their
original evidence instead of rewriting them to pass a new transition.
