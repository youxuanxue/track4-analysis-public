"""The local pre-check must ask the judge the question the REAL gate asks.

Finding H1 of the prose-drift sweep of 2026-08-27, rewritten against post-#28 `main`.

Two generations of defect are pinned here, because the fix for the first one was written against
behaviour #28 had already deleted:

*Generation 1.* ``python faithfulness/judge.py --answer …`` resolved the premise as
``claim.get("_resolved_span", "") or claim_text``, and nothing ever set ``_resolved_span``, so
every claim was scored against itself.

*Generation 2.* The premise was then resolved from ``corpus/<doc_id>.json`` — but the **hypothesis**
stayed ``claim["text"]``, the string the participant wrote. #28 deleted exactly that rule from the
scorer: the hypothesis is now the canonical sentence derived from the submitted prediction
(:func:`qfbench2_track_analysis.hypothesis.canonical_hypothesis`). Measured on the synthetic unit
below, an answer with ``label="miss"``, ``point_forecast=-99.0`` and interval ``[-100, -98]`` whose
prose accurately described a real corpus passage scored **faithfulness 1.0 / gate_pass True**
against the old local check and **0.0** against the real gate. A pre-check that says PASS where the
gate says FAIL is worse than no pre-check at all, so every test here asserts the local number
*equals* what the scorer produces for the same answer and the same judge.

The judges come from ``scoring/tests/synthetic.py`` and need no model weights.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from faithfulness.judge import build_unit_context, check_answer
from qfbench2_track_analysis.codes import T4ParticipantFailure, T4Reason
from qfbench2_track_analysis.corpus import CorpusIndex
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import score_unit
from scoring.tests.synthetic import (
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    HypothesisAwareJudge,
    answer_for,
    build_unit,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

ROSTER = ("SYN-A",)

#: Prose that accurately describes `SUPPORTING_TEXT`. Under the deleted rule this string WAS the
#: hypothesis, so a judge that genuinely entails the passage scored the submission 1.0 no matter
#: what the submission predicted.
ACCURATE_PROSE = (
    "Synthetic Issuer A reported quarterly earnings above the published consensus."
)


def _judge() -> HypothesisAwareJudge:
    """Entails the supporting passage only when the hypothesis describes that passage.

    The canonical hypothesis never contains "above the published consensus" — it is built from
    the submitted label, forecast and interval — so this judge separates "asked about the prose"
    from "asked about the prediction" without a model.
    """
    return HypothesisAwareJudge(
        SUPPORTING_TEXT, required_in_hypothesis="above the published consensus"
    )


def _wrong_prediction_with_accurate_prose() -> dict[str, Any]:
    """label='miss', point_forecast=-99.0, interval [-100, -98] — and truthful prose."""
    return answer_for(
        entities=ROSTER,
        label="miss",
        point_forecast=-99.0,
        lo=-100.0,
        hi=-98.0,
        claim_text=ACCURATE_PROSE,
    )


# --------------------------------------------------------------------------------------------
# 1. The hypothesis is the PREDICTION, not the participant's prose.
# --------------------------------------------------------------------------------------------
def test_accurate_prose_does_not_rescue_a_wrong_prediction(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path, entities=ROSTER)
    judge = _judge()

    result = check_answer(
        _wrong_prediction_with_accurate_prose(), build_unit_context(unit), judge
    )

    assert result.faithfulness == 0.0
    assert result.gate_pass is False
    # The question actually asked. This is the assertion the old check could not make: it asked
    # about `ACCURATE_PROSE`, which is why it answered 1.0.
    asked = [hypothesis for _, hypothesis in judge.calls]
    assert ACCURATE_PROSE not in asked
    assert all("is miss." in h and "-100 to -98" in h for h in asked)


def test_a_supported_prediction_still_passes(tmp_path: pathlib.Path) -> None:
    """Positive control: the gate is not simply refusing everything."""
    unit = build_unit(tmp_path, entities=ROSTER)
    judge = HypothesisAwareJudge(SUPPORTING_TEXT, required_in_hypothesis="is beat")

    result = check_answer(
        answer_for(entities=ROSTER, label="beat"), build_unit_context(unit), judge
    )

    assert result.faithfulness == 1.0
    assert result.gate_pass is True


def _score_with(
    unit: pathlib.Path, answer: dict[str, Any], judge: object, tmp_path: pathlib.Path
) -> Any:
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=judge,
        judge_provenance=provenance,
    )


def test_local_number_equals_the_scorer_on_a_passing_answer(
    tmp_path: pathlib.Path,
) -> None:
    """The anti-drift test: the pre-check and `score_unit` must agree, or the pre-check lies."""
    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True)
    answer = answer_for(entities=ROSTER, label="beat")

    local = check_answer(
        answer,
        build_unit_context(unit),
        HypothesisAwareJudge(SUPPORTING_TEXT, required_in_hypothesis="is beat"),
    )
    outcome = _score_with(
        unit,
        answer,
        HypothesisAwareJudge(SUPPORTING_TEXT, required_in_hypothesis="is beat"),
        tmp_path,
    )

    assert outcome.state == "participant_success"
    assert local.gate_pass is True
    assert local.faithfulness == outcome.diagnostics["faithfulness"] == 1.0


def test_local_verdict_equals_the_scorer_on_the_prose_exploit(
    tmp_path: pathlib.Path,
) -> None:
    """The same submission must fail in both places, and for the same recorded reason."""
    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True)
    answer = _wrong_prediction_with_accurate_prose()

    local = check_answer(answer, build_unit_context(unit), _judge())
    outcome = _score_with(unit, answer, _judge(), tmp_path)

    assert local.gate_pass is False
    assert local.faithfulness == 0.0
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics["reason"] == T4Reason.EVIDENCE_UNSUPPORTED.value


# --------------------------------------------------------------------------------------------
# 2. The denominator is the trusted roster, not the number of claims written.
# --------------------------------------------------------------------------------------------
def test_padding_claims_cannot_move_the_local_denominator(
    tmp_path: pathlib.Path,
) -> None:
    """Seven copies of one claim on a one-entity roster: the old check reported 7, the gate 1."""
    unit = build_unit(tmp_path, entities=ROSTER)
    answer = _wrong_prediction_with_accurate_prose()
    one = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [dict(one) for _ in range(7)]

    result = check_answer(answer, build_unit_context(unit), _judge())

    assert result.roster_count == 1
    assert len(result.predictions) == 1
    assert result.faithfulness == 0.0


# --------------------------------------------------------------------------------------------
# 3. Citations resolve through the trusted CorpusIndex — no path is built from a doc_id.
# --------------------------------------------------------------------------------------------
def test_an_undeclared_file_in_corpus_is_not_a_citable_document(
    tmp_path: pathlib.Path,
) -> None:
    """A file dropped into corpus/ but absent from the manifest resolved, and scored 1.0."""
    unit = build_unit(tmp_path, entities=ROSTER)
    undeclared = "SYNTHDOC_UNDECLARED"
    (unit / "corpus" / f"{undeclared}.json").write_text(
        json.dumps(
            {"doc_id": undeclared, "doc_date": "2026-02-01", "text": SUPPORTING_TEXT}
        ),
        encoding="utf-8",
    )
    answer = answer_for(entities=ROSTER, doc_id=undeclared, label="beat")
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SUPPORTING_TEXT)

    with pytest.raises(T4ParticipantFailure) as excinfo:
        check_answer(answer, build_unit_context(unit), _judge())
    assert excinfo.value.reason is T4Reason.CITATION_UNRESOLVED


def test_a_traversing_doc_id_cannot_read_outside_the_corpus_directory(
    tmp_path: pathlib.Path,
) -> None:
    """THE serious one. `corpus_dir / f"{doc_id}.json"` read `../../oracle` — one level ABOVE
    the unit directory, where a sealed reference would sit. A dict lookup cannot traverse."""
    unit = build_unit(tmp_path, entities=ROSTER)
    planted = tmp_path / "oracle.json"
    planted.write_text(
        json.dumps(
            {
                "doc_id": "oracle",
                "doc_date": "2026-02-01",
                "text": "SEALED ANSWER: SYN-A is beat.",
            }
        ),
        encoding="utf-8",
    )
    assert (
        planted.is_file()
    )  # control: the file the old lookup succeeded in reading exists

    ctx = build_unit_context(unit)
    index = ctx["_corpus"]
    assert isinstance(index, CorpusIndex)
    assert index.doc_ids == tuple(sorted(index.doc_ids))
    assert "../../oracle" not in index.doc_ids

    with pytest.raises(T4ParticipantFailure) as direct:
        index.resolve("../../oracle")
    assert direct.value.reason is T4Reason.CITATION_UNRESOLVED

    answer = answer_for(entities=ROSTER, doc_id="../../oracle", label="beat")
    with pytest.raises(T4ParticipantFailure) as viacheck:
        check_answer(answer, ctx, _judge())
    assert viacheck.value.reason is T4Reason.CITATION_UNRESOLVED


def test_resolved_span_is_not_an_escape_hatch(tmp_path: pathlib.Path) -> None:
    """`_resolved_span` used to override the corpus lookup, so a claim could be its own premise.

    The field is not in `analysis.schema.json`, nothing in this repo or the toolkit sets it, and
    the override let a submission supply BOTH halves of the entailment pair. Measured against the
    branch that has been deleted: a claim whose `_resolved_span` equalled its own text scored
    faithfulness 1.0 / gate_pass True. It must now be inert — the premise comes from the trusted
    corpus and from nowhere else.
    """
    unit = build_unit(tmp_path, entities=ROSTER)
    answer = _wrong_prediction_with_accurate_prose()
    claim = answer["entity_predictions"][0]["claims"][0]
    # Span deliberately wider than the corpus document: Python slicing clamps it, so the
    # premise is the whole cited document either way — and, under the deleted override, the
    # whole participant-supplied string.
    claim["span_end"] = 400
    claim["_resolved_span"] = (
        "The eps outcome of Synthetic Issuer A (SYN-A) is miss. The 90% prediction interval "
        "for the eps outcome of Synthetic Issuer A (SYN-A) is -100 to -98."
    )

    judge = _SelfEntailingJudge()
    result = check_answer(answer, build_unit_context(unit), judge)

    assert result.faithfulness == 0.0
    assert result.gate_pass is False
    # The premise offered to the judge is always corpus text, never the participant's string.
    assert claim["_resolved_span"] not in [premise for premise, _ in judge.calls]


class _SelfEntailingJudge:
    """Entails whenever the premise contains the hypothesis. A claim used as its own premise
    scores 1.0 under it; a real corpus span about something else does not."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return 1.0 if hypothesis and hypothesis in premise else 0.0


# --------------------------------------------------------------------------------------------
# 4. A post-cutoff citation fails locally, exactly as it fails the gate.
# --------------------------------------------------------------------------------------------
def test_post_cutoff_citation_fails_locally(tmp_path: pathlib.Path) -> None:
    unit = build_unit(tmp_path, entities=ROSTER)
    answer = answer_for(entities=ROSTER, doc_id="SYNTHDOC_POST_20260301", label="beat")

    with pytest.raises(T4ParticipantFailure) as excinfo:
        check_answer(answer, build_unit_context(unit), _judge())
    assert excinfo.value.reason is T4Reason.CITATION_POST_CUTOFF


# --------------------------------------------------------------------------------------------
# 5. The CLI fails closed without a unit — and for that reason specifically.
# --------------------------------------------------------------------------------------------
def test_cli_refuses_answer_without_corpus(tmp_path: pathlib.Path) -> None:
    """--answer without --unit must be refused by ARGUMENT PARSING, before anything else.

    Asserting only ``returncode != 0`` was not enough: on a machine without model weights the
    script exits 1 down the judge-unavailable path, so the old assertion passed even with the
    requirement removed. Argparse's usage error is exit code 2 and its message names the option,
    and both are asserted here so the test can only pass for the right reason.
    """
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(json.dumps(answer_for(entities=ROSTER)), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "faithfulness" / "judge.py"),
            "--answer",
            str(answer_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    combined = proc.stdout + proc.stderr
    assert "--unit is required with --answer" in combined
    # And it must NOT have got as far as reporting a gate verdict.
    assert "GATE:" not in combined


def test_cli_runs_as_a_script_from_the_repo_root(tmp_path: pathlib.Path) -> None:
    """`python faithfulness/judge.py` puts faithfulness/ on sys.path, not the repo root.

    Without the bootstrap in the __main__ block the check dies on
    ``ModuleNotFoundError: qfbench2_track_analysis`` the moment it is given a unit — which is
    every documented invocation.
    """
    unit = build_unit(tmp_path, entities=ROSTER)
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps(answer_for(entities=ROSTER, doc_id=PRE_CUTOFF_DOC)), encoding="utf-8"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "faithfulness" / "judge.py"),
            "--answer",
            str(answer_path),
            "--unit",
            str(unit),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    combined = proc.stdout + proc.stderr
    assert "ModuleNotFoundError" not in combined, combined
    # The banner is printed from the HYDRATED unit, so seeing it proves the repo's own
    # qfbench2_track_analysis package imported and CorpusIndex.from_unit ran.
    assert "Scoring 1 prediction(s)" in combined, combined
    assert "the trusted roster: 1" in combined, combined
    # Whether the judge is installed decides what happens next; either way the run must not
    # end in a green exit without a verdict.
    assert proc.returncode != 0
    assert "GATE:" in combined


def test_transformers_without_torch_degrades_instead_of_crashing(
    tmp_path: pathlib.Path,
) -> None:
    """`transformers` imports without `torch` and then dies inside its own internals.

    This is not hypothetical: CI installs `transformers` and not `torch`, and the judge's
    availability probe imported only `pipeline`. The run therefore took the judge-IS-available
    path, printed the roster banner, and crashed with a bare
    ``NameError: name 'torch' is not defined`` raised from `hasattr(torch, dtype)` inside
    transformers -- after the banner and before any verdict, so the CLI produced no ``GATE:``
    line at all. Locally, where transformers is simply absent, the graceful path ran and the
    whole class was invisible.

    Importable is not the same as usable. The probe now requires the tensor library too, and
    this test pins the degraded path with a stub that reproduces the CI shape exactly: a
    `transformers` module that imports fine and raises the real NameError when used.
    """
    shim = tmp_path / "shim"
    (shim / "transformers").mkdir(parents=True)
    (shim / "transformers" / "__init__.py").write_text(
        "def pipeline(*a, **k):\n    raise NameError(\"name 'torch' is not defined\")\n",
        encoding="utf-8",
    )
    # The stub must actually be importable, or this test passes for the wrong reason.
    probe = subprocess.run(
        [sys.executable, "-c", "import transformers; print(transformers.__file__)"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(shim)},
    )
    assert (
        str(shim) in probe.stdout
    ), f"stub transformers was not imported: {probe.stdout!r}"

    unit = build_unit(tmp_path, entities=ROSTER)
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps(answer_for(entities=ROSTER, doc_id=PRE_CUTOFF_DOC)), encoding="utf-8"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "faithfulness" / "judge.py"),
            "--answer",
            str(answer_path),
            "--unit",
            str(unit),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(shim)},
    )
    combined = proc.stdout + proc.stderr
    assert "NameError" not in combined, combined
    assert "GATE:" in combined, combined
    assert proc.returncode != 0
