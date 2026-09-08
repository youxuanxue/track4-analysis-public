"""Development data boundaries use independent synthetic files."""

import json
import pytest

from baselines.evaluation.dataset import load_cases, require_external, stage_inputs


def task(root, cutoff="2020-01-01", resolution="2020-02-01"):
    root.mkdir(parents=True)
    (root / "task.json").write_text(
        json.dumps({"cutoff_date": cutoff, "resolution_date": resolution})
    )
    return root


def roster(tmp_path, rows):
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps({"version": 1, "cases": rows}))
    return manifest


def test_load_public_units_without_outcomes(tmp_path):
    task(tmp_path / "a")
    task(tmp_path / "b")
    cases = load_cases(units=tmp_path, manifest=None)
    assert [case.case_id for case in cases] == ["a", "b"]
    assert all(case.truth_path is None and case.split == "public-dev" for case in cases)


def test_private_split_requires_earlier_outcomes_before_later_inputs(tmp_path):
    task(tmp_path / "early", resolution="2020-02-01")
    task(tmp_path / "later", cutoff="2020-02-02", resolution="2020-03-01")
    rows = [
        {"id": "a", "unit_dir": "early", "split": "train", "group": "a"},
        {"id": "b", "unit_dir": "later", "split": "test", "group": "b"},
    ]
    manifest = roster(tmp_path, rows)
    assert [case.split for case in load_cases(units=None, manifest=manifest)] == [
        "train",
        "test",
    ]
    (tmp_path / "later/task.json").write_text(
        json.dumps({"cutoff_date": "2020-02-01", "resolution_date": "2020-03-01"})
    )
    with pytest.raises(ValueError, match="overlap in time"):
        load_cases(units=None, manifest=manifest)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"id": "../escape"}, "safe directory"),
        ({"split": "unknown"}, "split must"),
        ({"id": "a"}, "duplicate case"),
        ({"group": "a"}, "event group"),
        ({"unit_dir": "early"}, "duplicate case"),
    ],
)
def test_private_roster_rejects_ambiguous_cases(tmp_path, mutation, message):
    task(tmp_path / "early")
    task(tmp_path / "later", cutoff="2021-01-01", resolution="2021-02-01")
    row = {"id": "b", "unit_dir": "later", "split": "test", "group": "b"}
    row.update(mutation)
    manifest = roster(
        tmp_path,
        [{"id": "a", "unit_dir": "early", "split": "train", "group": "a"}, row],
    )
    with pytest.raises(ValueError, match=message):
        load_cases(units=None, manifest=manifest)


def test_truth_cannot_be_inside_any_input_unit(tmp_path):
    unit = task(tmp_path / "unit")
    (unit / "truth.json").write_text("{}")
    manifest = roster(
        tmp_path,
        [
            {
                "id": "a",
                "unit_dir": "unit",
                "split": "test",
                "group": "a",
                "truth_path": "unit/truth.json",
            }
        ],
    )
    with pytest.raises(ValueError, match="separately"):
        load_cases(units=None, manifest=manifest)


def test_external_path_guard_resolves_symlinks(tmp_path):
    repo = tmp_path / "public"
    repo.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="outside public"):
        require_external(alias / "outputs", [repo])
    require_external(tmp_path / "private", [repo])


def test_staging_excludes_truth_and_undeclared_files(tmp_path):
    unit = task(tmp_path / "unit")
    (unit / "corpus").mkdir()
    (unit / "corpus/source.json").write_text('{"text":"Synthetic pre-cutoff evidence"}')
    (unit / "corpus/unlisted.json").write_text(
        '{"private_marker":"must not reach predictor"}'
    )
    (unit / "corpus/ancillary.json").write_text(
        '{"text":"private_marker: declared metadata is not citable evidence"}'
    )
    (unit / "corpus/manifest.json").write_text('{"private_marker":"corpus index"}')
    (unit / "reference").mkdir()
    (unit / "reference/secret.json").write_text(
        '{"private_marker":"must not reach predictor"}'
    )
    (unit / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "corpus/source.json", "role": "corpus"},
                    {"path": "corpus/ancillary.json", "role": "metadata"},
                    {"path": "corpus/manifest.json", "role": "corpus"},
                ]
            }
        )
    )
    dest = tmp_path / "staged"
    digest = stage_inputs(unit, dest)
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == [
        "corpus/source.json",
        "task.json",
    ]
    assert "private_marker" not in "".join(p.read_text() for p in dest.rglob("*.json"))
    (unit / "corpus/source.json").write_text('{"text":"Changed observation"}')
    assert stage_inputs(unit, tmp_path / "staged-again") != digest


@pytest.mark.parametrize("role", ["metadata", None])
def test_staging_requires_declared_citable_evidence(tmp_path, role):
    unit = task(tmp_path / "unit")
    (unit / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "corpus/source.json", "role": role},
                    {"path": "corpus/manifest.json", "role": "corpus"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="no JSON evidence"):
        stage_inputs(unit, tmp_path / "staged")


def test_staging_rejects_corpus_symlink(tmp_path):
    unit = task(tmp_path / "unit")
    (unit / "corpus").mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text("{}")
    (unit / "corpus/source.json").symlink_to(secret)
    (unit / "manifest.json").write_text(
        json.dumps({"files": [{"path": "corpus/source.json", "role": "corpus"}]})
    )
    with pytest.raises(ValueError, match="link outside"):
        stage_inputs(unit, tmp_path / "staged")
