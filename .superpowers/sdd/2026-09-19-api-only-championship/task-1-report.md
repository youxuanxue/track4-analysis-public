## Executive summary (read this first)

Task 1 (N1) is implemented as an API-only participant contract. Active participant documentation now names `category = "api"` and the organizer House endpoint as the only Track 4 model path; local numerical artifacts remain permitted under the artifact policy, while developer-local model diagnostics are explicitly non-submission tooling. A document guard was added test-first to reject affirmative BYO categories, adapter packaging, and participant-provided language-model weights. The requested document tests and scorer/judge/firewall checks pass; the exemplar smoke command reaches the smoke verifier but remains inadmissible because its output is schema-invalid, which is recorded as a residual concern.

## Changes

- `README.md`: replaced the active two-mode/BYO contract with the API-only contract; retained permitted non-neural artifact language and clarified developer-local diagnostics.
- `SUBMISSION_CLI.md`: removed Track 4 BYO categories and the adapter-packaging section; documented only `category = "api"`, House model identity, and API-only reproducibility/request rules.
- `docs/TRAINING-POLICY.md`: narrowed offline fitting language to eligible non-neural artifacts and explicitly prohibited participant fine-tuning, language-model weights, LoRA, and adapters as submission paths.
- `docs/ARTIFACT-POLICY.md`: preserved permitted local numerical artifacts and made the API-only boundary explicit.
- `baselines/README.md`: updated the RAG baseline and submission guidance to House API-only; offline model references are developer-only and not submission artifacts.
- `baselines/strong_rag_baseline/README.md`: updated the official category section and local GGUF wording to API-only/non-submission diagnostics.
- `baselines/strong_rag_baseline/config.py`: changed the local/BYO ceiling comment to a developer-local, House-neutral description.
- `baselines/tests/test_docs_match_artifacts.py`: added fail-closed active-contract document checks and self-controls for `byo-small`, `byo-large`, adapter packaging, and participant model weights.

`docs/CHAMPIONSHIP-PLAN.md` was not edited. No `units/*/corpus/`, `units/*/manifest.json`, or private directory was edited.

## TDD evidence

1. Added the document guard before rewriting the contract.
2. Initial required command could not resolve `.venv/bin/python` in this worktree, so the existing public-worktree environment was temporarily referenced to run the same command. The new guard then failed as intended with 5 offenders: `README.md` BYO category plus `SUBMISSION_CLI.md` adapter and full-weight paths.
3. Rewrote the active contract and reran the guard; it passed with 13 tests.
4. Removed the temporary untracked `.venv` symlink before final verification.

## Verification commands and outputs

- `.venv/bin/python -m pytest baselines/tests/test_docs_match_artifacts.py -q` (using the existing public-worktree Python environment because this worktree has no local `.venv`): `13 passed in 0.04s`.
- `.venv/bin/python -m pytest scoring/ faithfulness/ -q` (same environment): `254 passed, 1 skipped in 15.73s`.
- `.venv/bin/python -m pytest baselines/tests/test_units_carry_no_answer_material.py -q` (same environment): `4 passed in 0.09s`.
- `git diff --check`: passed with no output.
- `qfbench2 manifest assert-public-safe units/t4-EXAMPLE-eps-beat`: `OK public-safe units/t4-EXAMPLE-eps-beat`.
- `PYTHONPATH=. qfbench2-smoke units/t4-EXAMPLE-eps-beat /tmp/t4-n1-smoke --track analysis`: reached `factory=build_smoke_verifier`, then returned `admissible=False score=None labels=['shared.schema.invalid_output']` and exit code 1.
- Active-doc residual search: remaining BYO/adapter/model-weight matches in the six guarded documents are prohibition language or developer-only diagnostics. Historical/scope references remain in `docs/superpowers/specs/`, `docs/superpowers/plans/`, `docs/CHAMPIONSHIP-PLAN.md` (controller-owned and intentionally untouched), `THIRD-PARTY-NOTICES.md`, and unrelated evaluation documentation; none is an affirmative Track 4 submission path.

## Residual concerns

- The exemplar smoke run is still inadmissible due to `shared.schema.invalid_output`. This is outside the N1 documentation change and was not modified because the task scope is the participant contract.
- The requested worktree has no native `.venv`; verification used the existing adjacent public-worktree environment by absolute path. No environment symlink or generated environment file remains in the worktree.
