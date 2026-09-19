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

The available public worktree contains deterministic diagnostics and semantic tests but no Task 4 aggregate rate-curve measurement artifact. The measurable proxy is the model acceptance/fallback test contract, which remains green across all target types; this change specifically strengthens the prompt contract for the selected rejection mode. A first ad-hoc synthetic CLI probe used an invalid evidence identifier and correctly fell back, so it was not treated as evidence for improvement. No claim is made that hosted rates/FOMC rejection rates improved without the private House batch rerun.
