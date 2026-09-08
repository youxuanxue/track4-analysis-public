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
The tool resolves its immutable image ID before running; build the image from the same
workspace to get the optional diagnostics sidecar. Only declared task and corpus files
are copied into the container. No host data directories or Docker socket are mounted.
Returned artifacts are collected separately; only regular answer and diagnostics files
are imported. Container runtime faults abort evaluation instead of becoming scored failures.
This tests Docker execution, not the organizer's gVisor environment or model service.
The offline container uses conservative local CPU/memory caps, recorded with each run;
they do not claim to reproduce the official compute grant.

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
Production mode requires the organizer's configured judge and external outcomes and never
downgrades to smoke. The runner stops before inference when outcomes are undeclared or the
judge cannot be constructed.

Provenance includes the workspace source digest, Git revision and dirty state, Python,
installed scorer identity, and the actual shared-toolkit source digest. A package version
without a verified build stamp is identified as such. The Docker image ID identifies
participant runtime bytes separately from evaluator workspace bytes.

## Historical snapshots and evidence review

The development-only [historical builder](./historical.py) reads an external event specification
and retrieves Treasury observations from ALFRED. It requires `curl` for acquisition; cached
snapshots can subsequently be rebuilt offline. Every series is requested separately, and its
returned column must carry the exact requested vintage date. This matters because a multi-series
request can silently return the latest vintage for some columns. Downloaded bytes, source URLs,
vintages and retrieval times are retained outside the public repository.

The specification has `version: 1` and an `events` array. Each event declares `id`, `cutoff`,
`resolution`, and `split` (train, calibration or test). The builder refuses overlapping event
windows and split leakage before downloading. It creates classification, regression and ranking
views of each event, with the same event group: those views are correlated, not additional
independent samples. This is a single-domain rates benchmark, not a proxy for all hidden families.
The corpus is a deterministic extract of historical observations, not a financial forecast.
Only prediction inputs go under each unit; future snapshots and outcomes stay outside those units.

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

Toolkit tag `v2.4.0` still reports package version `2.3.1`; use the recorded source digest and
Git installation identity to distinguish it. The updated tag allows model-free `models: []`
descriptors but does not provide a production NLI judge.

## Frozen-Prediction Interval Calibration

[`calibration.py`](calibration.py) fits interval radii on earlier training-event residuals
and writes adjusted answers for later calibration events. This is an external development
workflow for the deterministic grounded baseline; it does not alter the submission defaults
or ship fitted outcome-derived parameters in the public image. Model reports, test splits,
overlapping windows, incomplete runs and mismatched prediction settings are refused.

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
