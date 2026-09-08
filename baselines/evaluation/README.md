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
Production mode requires the organizer's configured judge and external outcomes and never
downgrades to smoke. The runner stops before inference when outcomes are undeclared or the
judge cannot be constructed.

Provenance includes the workspace source digest, Git revision and dirty state, Python,
installed scorer identity, and the actual shared-toolkit source digest. A package version
without a verified build stamp is identified as such. The Docker image ID identifies
participant runtime bytes separately from evaluator workspace bytes.
