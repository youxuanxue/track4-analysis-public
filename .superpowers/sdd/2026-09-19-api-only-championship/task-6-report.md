## Executive summary (read this first)

Task 6 adds a fail-closed inventory audit for disjoint evaluation manifests. The public code and synthetic tests cover event-group overlap, input-digest overlap after case renaming, coverage minimums, deterministic JSON, and command-line exit statuses. The proposed r8 inventory was not available in this public worktree, so no private evidence was read and no batch was registered. This report records that blocked state rather than inventing an eligibility result.

## Implementation

- Added `baselines/evaluation/inventory.py` with deterministic JSON audit output.
- Added synthetic tests in `baselines/evaluation/tests/test_inventory.py`.
- Added `python -m baselines.evaluation inventory` dispatch in `baselines/evaluation/__main__.py`.
- Documented policy, command, and exit semantics in `baselines/evaluation/README.md`.
- The audit compares groups and staged input digests, counts domains and target types, and fails closed on malformed inputs, overlap, or policy minimum failures.
- The implementation does not import or call `confirmations.seal`, and does not implement scoring math.

## BLOCKED_INVENTORY

- Status: BLOCKED
- Reason: the private proposed r8 inventory and its consumed-registration evidence are not present in the public worktree and were not read or copied.
- Public synthetic audit: PASS for a disjoint roster under the default policy.
- Private r8 eligibility: UNMEASURED; no claim is made.
- Batch registration: NOT RUN.
- Required private follow-up: run the inventory command against the private candidate and consumed manifests outside every public worktree; retain the audit and any batch artifacts privately. If the audit is ineligible, append the valid JSON audit and a new blocked record to the private evaluation log without registering a batch.

## Verification

- `.venv/bin/python -m pytest baselines/evaluation/tests/test_inventory.py -q` — 15 passed.
- `.venv/bin/python -m pytest baselines/evaluation -q` — 162 passed, 18 skipped.
- `.venv/bin/python -m pytest baselines/tests/test_units_carry_no_answer_material.py -q` — 4 passed.
- The combined `scoring/ faithfulness/` command was attempted but collection is blocked in this checkout because the external `qfbench2_common` package is not installed (`ModuleNotFoundError`).

## Repair review

The repair removes caller-supplied `expected_runs` records, reads the authoritative policy from `acceptance-policy.json`, requires non-empty domains and the three allowed target types, and rejects duplicate or inconsistently mapped group identities across the candidate plus every consumed manifest. Synthetic coverage now uses 60 event groups with 20 groups per target type and an even three-domain distribution. Subprocess tests cover exit statuses 0, 1, and 2, malformed input, output failure, and multiple consumed manifests. No private r8 data was accessed and no batch or confirmation sealing was run.

Repair verification:

- `.venv/bin/python -m pytest baselines/evaluation/tests/test_inventory.py -q` — 19 passed.
- `.venv/bin/python -m pytest baselines/evaluation -q` — 166 passed, 18 skipped.
- `.venv/bin/python -m pytest baselines/tests/test_units_carry_no_answer_material.py -q` — 4 passed.
- Temporary installation of `qfbench2-common==2.4.3` was attempted via the configured package index; the package is unavailable in this environment, so `scoring/ faithfulness/` remains blocked at collection with `ModuleNotFoundError: qfbench2_common`.
