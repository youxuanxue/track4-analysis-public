## Executive summary (read this first)

Task 5 selects the measured target-semantics failure mode: rate and yield-change tasks can confuse a current level with the requested basis-point change. The candidate changes only the model user prompt, adding an explicit `change_bps` instruction to subtract the current level from the projected level and report the difference in basis points. No scoring, retry limits, fallback prediction, retrieval, or validation behavior changed. Existing diagnostics tests continue to verify model acceptance and grounded fallback behavior.

## Selection

Task 4 diagnostics expose `model_prediction` as the target-semantic rejection stage. Existing semantic regression tests already cover the corresponding rate-curve failure (`test_future_yield_change_does_not_copy_current_yield`) and conversion behavior (`test_declared_yield_projection_converts_percent_to_bps`). The preferred rate-curve hypothesis was therefore selected rather than introducing a broader target-aware validator.

## Change

- `baselines/strong_rag_baseline/prompts.py`: add one `change_bps` target-semantics instruction.
- `baselines/strong_rag_baseline/tests/test_model_failures.py`: assert the prompt distinguishes level from basis-point change and states the subtraction operation.

The focused acceptance/fallback contract remains unchanged: accepted model outputs retain `model_accepted=true`, while invalid model outputs remain grounded fallback with the existing stage and fallback reason. No fallback heuristic or retry budget was modified.

## Verification

- RED attempt: the new prompt test was added before implementation; the repository environment initially lacked pytest (`.venv` had no pytest), so the intended test command could not execute until pytest was installed. After installing pytest into the existing `.venv`, the focused suite passed.
- Focused strong-RAG tests: `.venv/bin/python -m pytest baselines/strong_rag_baseline/tests/test_model_failures.py -q` — **49 passed**.
- Full strong-RAG tests: `.venv/bin/python -m pytest baselines/strong_rag_baseline/tests -q` — **346 passed**.
- `git diff --check` — passed.
- Python bytecode compilation — passed.

## Metric and concerns

The available public worktree contains deterministic diagnostics and semantic tests but no Task 4 aggregate rate-curve measurement artifact. The review follow-up now exercises the real `agent.run_entity` request path for classification, regression, and ranking; ranking uses two entities and asserts every entity prompt. The same fixture has four samples total (one classification, one regression, two ranking), all four admissible after the change, with zero observed `model_prediction` fallback stages in the deterministic acceptance stub. A before/after reduction cannot be established because the stub returns a canned valid response and does not model prompt-sensitive behavior; the recorded measurement is therefore `candidate_gate=FAIL`, not a promotion claim. The JSON measurement is recorded at `.superpowers/sdd/2026-09-19-api-only-championship/task-5-measurement.json` and is intentionally not added to public Git.
