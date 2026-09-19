## Executive summary (read this first)

This plan executes the approved N1–N4 design in isolated, reviewable tasks. It uses test-first changes, keeps private evaluation evidence outside the public repository, and treats toolkit migration and API-only policy as one atomic contract update. A new hosted submission is made only after the candidate passes local, container, and descriptor gates. G3-local runs only when the inventory audit proves the new batch is independent and meets the existing policy minima.

# API-only championship implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete reachable N1–N4 work, validate one attributable API-mode candidate, and open a fork pull request without exposing private evaluation material.

**Architecture:** Update the public contract and toolkit pin atomically, add diagnostics-driven target-semantics coverage to the existing strong RAG agent, and add a fail-closed generic inventory audit. Private migrations and batch evidence are generated outside the public worktree and are never committed.

**Tech Stack:** Python 3.13, pytest, stdlib HTTP test server, qfbench2-common v2.4.3, Docker/GHCR, GitHub CLI, Codabench.

**Spec:** `docs/superpowers/specs/2026-09-19-api-only-championship-design.md`

## Global constraints

- Never add resolved outcomes, answer material, source snapshots, sealed-event lists, private reports, or Codabench unit scores to the public repository.
- Preserve all historical private evidence; record supersession in new files rather than editing digest-bound files.
- Track 4 submission category is `api`; no participant language-model weights, fine-tuning, LoRA, adapters, or model server enter a candidate.
- Do not change scoring math, gate names, metric bounds, or acceptance-policy thresholds.
- Test all three target types: classification, regression, and ranking.
- Use ordinary `batch register/run/decide` for G3-local; never call `confirmations.seal` without the production judge contract.
- Push GHCR and upload Codabench only after the candidate gate passes.

---

### Task 1: API-only participant contract (N1)

**Files:**
- Modify: `README.md`
- Modify: `SUBMISSION_CLI.md`
- Modify: `docs/TRAINING-POLICY.md`
- Modify: `docs/ARTIFACT-POLICY.md`
- Modify: `baselines/README.md`
- Modify: `baselines/strong_rag_baseline/README.md`
- Modify: `baselines/strong_rag_baseline/config.py`
- Test: `baselines/tests/test_docs_match_artifacts.py`

**Interfaces:**
- Consumes: issue #8 final API-only ruling and existing artifact policy.
- Produces: one participant-facing contract in which `api` is the only Track 4 category and developer-only local diagnostics are not submission paths.

- [ ] Add failing document tests that reject affirmative `byo-small`, `byo-large`, adapter packaging, and participant-provided language-model weights in active contract sections.
- [ ] Run `.venv/bin/python -m pytest baselines/tests/test_docs_match_artifacts.py -q` and confirm the new assertions fail on current text.
- [ ] Rewrite the listed active documentation to API-only language while preserving the permitted non-neural artifact boundary.
- [ ] Change the `local/BYO` source comment in `config.py` to a developer-local/House-neutral description.
- [ ] Rerun the document test and confirm it passes.
- [ ] Search active docs for residual affirmative BYO wording and classify any remaining match as historical, prohibited, or unrelated terminology.

### Task 2: Toolkit 2.4.3 public migration (N2-public)

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `SUBMISSION_CLI.md`
- Modify: `baselines/requirements.txt`
- Modify: `faithfulness/judge.py`
- Modify: `docs/CHAMPIONSHIP-PLAN.md`
- Test: `baselines/tests/test_docs_match_artifacts.py`
- Test: `tests/test_scorer_version.py`

**Interfaces:**
- Consumes: qfbench2-common tag `v2.4.3` and its API-only descriptor schema.
- Produces: synchronized operational pins and tests that fail on version drift.

- [ ] Add or update version-drift assertions so current operational pins must be `2.4.3`.
- [ ] Run the focused tests and confirm they fail while pins remain `2.4.2`.
- [ ] Update every current CI/install/help pin to v2.4.3; do not rewrite immutable private reports.
- [ ] Run focused version and document tests.
- [ ] Create a fresh Python 3.13 environment outside the repository, install v2.4.3, and record version plus package-tree digest in the private migration record.
- [ ] Run public pytest, unit validation, and exemplar smoke in the fresh environment.

### Task 3: Private descriptor and evidence migration (N1/N2-private)

**Files outside public repository:**
- Create: `/Users/feng/Codes/challenge/agenthon2026/t4-evaluation/API-ONLY-SUPERSESSION-20260919.md`
- Create: `/Users/feng/Codes/challenge/agenthon2026/t4-evaluation/toolkit-243-migration-20260919/`
- Replace active local descriptor: `/Users/feng/Codes/challenge/agenthon2026/t4-submission/submission.json`
- Replace examples: `/Users/feng/Codes/challenge/agenthon2026/t4-submission/submission.draft.json`
- Replace examples: `/Users/feng/Codes/challenge/agenthon2026/t4-submission/analysis_dev.fixture.json`

**Interfaces:**
- Consumes: immutable historical receipts and the current hosted API descriptor.
- Produces: append-only supersession record, 2.4.3-valid API descriptors, and same-environment before/after migration evidence.

- [ ] Write the supersession record stating that local model experiments are historical diagnostics and cannot enter active candidates, budgets, or acceptance plans.
- [ ] Generate API descriptors from the v2.4.3 contract; set `models` according to actual House use and recompute descriptor digests with toolkit tooling.
- [ ] Prove all active descriptors parse under 2.4.3 and legacy BYO descriptors are rejected.
- [ ] Rerun the frozen before and after candidates in the same 2.4.3 environment without overwriting r8.
- [ ] Compare same-role 2.4.2 and 2.4.3 outputs; document every difference and block promotion on unexplained differences.

### Task 4: House-compatible diagnostics tests (N3-observability)

**Files:**
- Modify: `baselines/strong_rag_baseline/agent.py`
- Modify: `baselines/strong_rag_baseline/cli.py`
- Modify: `baselines/strong_rag_baseline/validation.py`
- Test: `baselines/strong_rag_baseline/tests/test_agent.py`
- Test: `baselines/strong_rag_baseline/tests/test_cli.py`
- Test: `baselines/strong_rag_baseline/tests/test_client.py`

**Interfaces:**
- Consumes: existing `ModelClient` result and fallback-stage names.
- Produces: stable per-run diagnostics that distinguish request, JSON, evidence, prediction, timeout, and budget rejection without exposing chain-of-thought.

- [ ] Add failing tests for diagnostics emitted on accepted model output and each fallback stage across all three target types.
- [ ] Run the focused tests and confirm failure before implementation.
- [ ] Implement the smallest diagnostics propagation needed by the tests; do not change scoring or retry limits.
- [ ] Run focused tests and existing client tests.
- [ ] Use a local OpenAI-compatible stub through the real client path and save only aggregate diagnostics in the private evaluation directory.

### Task 5: One attributable target-semantics improvement (N3-candidate)

**Files:**
- Modify only the dominant measured component, expected candidates:
  - `baselines/strong_rag_baseline/prompts.py`, or
  - `baselines/strong_rag_baseline/validation.py`
- Test: corresponding files under `baselines/strong_rag_baseline/tests/`

**Interfaces:**
- Consumes: aggregate rejection measurements from Task 4.
- Produces: one mechanism change with a measurable reduction in the selected fallback stage and no admissibility regression.

- [ ] Select the highest-frequency rejection stage and record the hypothesis privately before editing.
- [ ] Add failing tests that reproduce that stage, including rate-curve basis-point-change semantics if it is selected.
- [ ] Run the focused test and confirm it fails for the intended reason.
- [ ] Implement one minimal prompt or validation change; do not combine retrieval, interval, retry-budget, and fallback-heuristic changes.
- [ ] Run focused and full strong-RAG tests.
- [ ] Repeat the stub measurement and verify the predefined rejection metric improves.

### Task 6: Fail-closed disjoint inventory audit (N4)

**Files:**
- Create: `baselines/evaluation/inventory.py`
- Create: `baselines/evaluation/tests/test_inventory.py`
- Modify: `baselines/evaluation/__main__.py`
- Modify: `baselines/evaluation/README.md`

**Interfaces:**
- Consumes: candidate manifest and one or more consumed manifests/registrations.
- Produces: JSON audit with group overlap, input-digest overlap, per-domain counts, per-target-type counts, and `eligible` boolean; exits nonzero on overlap or policy-minimum failure.

- [ ] Add failing tests for group overlap, renamed-but-identical input overlap, insufficient domains, insufficient per-domain groups, insufficient target-type groups, and a passing disjoint manifest.
- [ ] Run `pytest baselines/evaluation/tests/test_inventory.py -q` and confirm failures.
- [ ] Implement deterministic parsing and audit logic using existing manifest conventions; do not copy scoring math.
- [ ] Add CLI wiring and document the exact command and exit semantics.
- [ ] Run inventory tests and the complete evaluation test suite.
- [ ] Audit the private proposed inventory against r8; if it is not eligible, create an append-only `BLOCKED_INVENTORY` record and do not register a batch.
- [ ] If it is eligible, execute ordinary `batch register/run/decide` and retain all outputs privately.

### Task 7: Candidate gate and conditional hosted submission

**Files:**
- Public repository: no new private output files.
- Private evidence: new candidate directory under `/Users/feng/Codes/challenge/agenthon2026/t4-evaluation/`.

**Interfaces:**
- Consumes: clean commit, Tasks 1–6 tests, candidate aggregate diagnostics, API descriptor.
- Produces: either a rejected local candidate or one immutable GHCR/Codabench candidate receipt.

- [ ] Run `scripts/preflight.sh` in the isolated worktree.
- [ ] Run all public tests, firewall, unit validation, all-unit smoke, and Docker smoke.
- [ ] Confirm git is clean except planned public files and no private artifact appears in `git status`.
- [ ] Build linux/amd64 image from the clean commit and exercise the real container interface with no network.
- [ ] Compare aggregate diagnostics to the frozen incumbent gate; stop without push if the gate fails.
- [ ] If the gate passes, push the immutable GHCR digest and verify anonymous pull plus interface label.
- [ ] Seal and pack a v2.4.3 API descriptor against that digest.
- [ ] Upload once to Codabench and retain the receipt privately.
- [ ] Promote only if every hosted unit is admissible and the predefined hosted metric improves; otherwise retain the hosted incumbent.

### Task 8: Integration review, commit, push, and pull request

**Files:**
- Review all public changes in the isolated worktree.

**Interfaces:**
- Consumes: reviewed Tasks 1–7.
- Produces: one branch and pull request targeting fork `origin/main`.

- [ ] Dispatch a requirements reviewer to compare the diff against the approved design and repository firewall.
- [ ] Dispatch a code-quality reviewer after requirements review passes.
- [ ] Fix findings with focused tests, then rerun the full verification gate.
- [ ] Commit public changes with coherent messages; never add `.venv`, private directories, receipts, or generated reports.
- [ ] Push `feat/api-only-243-championship` to `origin`.
- [ ] Create a pull request targeting `youxuanxue/track4-analysis-public:main`, reporting only public tests and non-sensitive outcomes.
