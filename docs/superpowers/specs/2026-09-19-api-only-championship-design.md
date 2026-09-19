## Executive summary (read this first)

This design moves Track 4 to the final API-only competition contract, pins the public workflow to toolkit 2.4.3, and runs one attributable House-compatible agent improvement. Public code, documentation, and generic safeguards belong in this repository; private outcomes, sealed-batch evidence, submission receipts, and source logs remain outside every public worktree. A Development upload is permitted only after the candidate passes the predefined local and container checks. G3-local is attempted only with genuinely disjoint events; otherwise the result is a verifiable inventory blocker, never a fabricated pass.

# API-only championship execution design

## Scope

The work delivers the reachable portions of N1–N4:

- N1 removes active BYO, LoRA, adapter, and participant-language-model paths while preserving historical diagnostics.
- N2 aligns current operational pins and descriptor validation with `qfbench2-common` v2.4.3.
- N3 measures House-compatible model-output acceptance, fixes one attributable loss mechanism, and conditionally creates one Development submission.
- N4 prevents reuse of consumed events and produces a machine-checkable disjointness gate; a quality batch runs only if sufficient fresh inventory exists.

G3-official remains out of scope until the production judge, equivalence contract, and artifact-eligibility contract are published.

## Repository and evidence boundaries

The public repository contains only participant-facing contracts, agent code, generic tests, and generic batch-safety tools. Private evaluation reports, immutable historical evidence, local descriptor inputs, resolved targets, prediction reports, Codabench unit scores, source snapshots, and sealed-event lists stay outside the public branch and pull request.

Historical private files are append-only evidence. New supersession or migration records may point to them, but existing receipts, registrations, reports, descriptors, and digest-bound plans are not rewritten.

## N1: API-only contract

Participant-facing documentation states that Track 4 uses `category = "api"` and the organizer House endpoint. Participant-provided language-model weights, fine-tuning, LoRA, adapters, and model servers are not submission paths. Permitted non-neural numerical artifacts remain governed by `docs/ARTIFACT-POLICY.md`.

Developer-only local model diagnostics may remain when they are clearly marked as non-submission tooling. Tests reject affirmative reintroduction of `byo-small`, `byo-large`, adapter packaging, or participant language-model weights in active contract documentation.

## N2: toolkit 2.4.3 migration

All current installation commands, CI jobs, requirements, and user-visible version references move together to v2.4.3. Validation uses a fresh Python 3.13 environment and records the package version and package-tree digest. API descriptors must parse under 2.4.3; legacy BYO descriptors must fail.

The frozen before and after candidates are rerun together under one 2.4.3 environment. Old 2.4.2 evidence remains immutable and cannot be combined with 2.4.3 reports for promotion.

## N3: attributable API improvement

The first step is observability, not prompt churn. A House-compatible test server drives deterministic replies through the real client and agent path. Diagnostics classify request, JSON, evidence, and prediction rejection, and tests cover classification, regression, ranking, malformed responses, timeouts, and request-budget exhaustion.

Only one dominant measured rejection mechanism is changed. The preferred hypothesis is target-semantics rejection on rate-curve tasks: the model must distinguish a yield-level observation from the requested basis-point change. If measurements identify a different dominant stage, implementation follows that stage while retaining the one-variable rule.

The candidate gate requires a clean commit, all public tests, all-unit smoke, offline fallback, container execution, and a measured reduction in the selected rejection mechanism without admissibility regression. Only then may the workflow build and push an immutable linux/amd64 GHCR image, reseal an API descriptor, and spend one Codabench Development upload. The hosted incumbent remains unless the candidate is admissible on every unit and improves the predefined hosted metric.

## N4: disjoint G3-local inventory

A generic audit compares a candidate roster manifest's event groups and input digests against one or more consumed roster manifests. Any overlap fails closed with a nonzero exit. A future private adapter may convert registrations into roster manifests; registration handling is not part of this public CLI. The audit also checks the internal policy minima: at least 60 independent events, at least three domains with 20 events each, and at least 20 groups for each target type.

If the private source audit does not produce a qualifying fresh manifest, N4 closes this iteration as `BLOCKED_INVENTORY` with an append-only private record. It must not invoke `confirmations.seal`, rename old groups, reinterpret seeds as events, or claim G3-local PASS.

## Testing and release

Agents work in separate scopes and do not edit the same files concurrently. Each scope receives test-first review, followed by cross-scope integration review. The final public gate is the repository preflight, scorer and judge tests, firewall, unit validation, smoke, and container test. The pull request targets the fork's `origin/main` and contains no private evidence.
