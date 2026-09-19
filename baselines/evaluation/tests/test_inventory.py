import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.evaluation import __main__ as evaluation_main
from baselines.evaluation import inventory


def make_case(root: Path, name: str, *, group: str, domain: str, target_type: str):
    unit = root / name
    (unit / "corpus").mkdir(parents=True)
    task = {
        "task_id": name,
        "cutoff_date": "2026-01-01",
        "resolution_date": "2026-01-02",
        "target": {"type": target_type},
        "entities": [{"entity_id": "e1"}],
    }
    (unit / "task.json").write_text(json.dumps(task))
    corpus = unit / "corpus" / "evidence.json"
    corpus.write_text(json.dumps({"text": name}))
    (unit / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "2.0",
                "unit_id": name,
                "files": [
                    {
                        "path": "corpus/evidence.json",
                        "role": "corpus",
                        "sha256": __import__("hashlib").sha256(corpus.read_bytes()).hexdigest(),
                        "bytes": corpus.stat().st_size,
                    }
                ],
            }
        )
    )
    return {"id": name, "unit_dir": str(unit), "split": "test", "group": group, "domain": domain}


def write_roster(path: Path, cases):
    path.write_text(json.dumps({"version": 1, "cases": cases}))
    return path


def policy():
    return {
        "min_event_groups": 60,
        "min_domains": 3,
        "min_groups_per_domain": 20,
        "min_groups_per_target_type": 20,
        "target_types": ["classification", "regression", "ranking"],
    }


def synthetic_policy():
    return {
        "min_event_groups": 1,
        "min_domains": 1,
        "min_groups_per_domain": 1,
        "min_groups_per_target_type": 1,
        "target_types": ["classification"],
    }


def passing_cases(root: Path):
    return [
        make_case(
            root,
            f"case-{kind}-{index}",
            group=f"group-{kind}-{index}",
            domain=f"domain-{(index + (0 if kind == 'classification' else 1 if kind == 'regression' else 2)) % 3}",
            target_type=kind,
        )
        for kind in ("classification", "regression", "ranking")
        for index in range(20)
    ]


def test_group_overlap_is_reported_and_ineligible(tmp_path):
    cases = [make_case(tmp_path, "candidate", group="g1", domain="d1", target_type="classification")]
    candidate = write_roster(tmp_path / "candidate.json", cases)
    consumed = write_roster(tmp_path / "consumed.json", [dict(cases[0], id="old")])
    result = inventory.audit(candidate, [consumed], policy=synthetic_policy())
    assert result["overlap"]["groups"] == ["g1"]
    assert result["eligible"] is False


def test_renamed_case_with_identical_input_is_overlap(tmp_path):
    first = make_case(tmp_path, "same-input", group="g1", domain="d1", target_type="classification")
    second = dict(first, id="renamed", unit_dir=first["unit_dir"], group="other")
    candidate = write_roster(tmp_path / "candidate.json", [first])
    consumed = write_roster(tmp_path / "consumed.json", [second])
    result = inventory.audit(candidate, [consumed], policy=synthetic_policy())
    assert result["overlap"]["input_digests"]
    assert result["eligible"] is False


def test_insufficient_domains_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group=f"g{i}", domain="only", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], policy={**synthetic_policy(), "min_domains": 2})
    assert result["counts"]["domains"] == {"only": 3}
    assert "domains" in result["policy_failures"]


def test_insufficient_per_domain_groups_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group=f"g{i}", domain=f"d{i}", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], policy={**synthetic_policy(), "min_groups_per_domain": 2})
    assert "groups_per_domain" in result["policy_failures"]


def test_insufficient_target_type_groups_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group=f"g{i}", domain="d", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], policy={**synthetic_policy(), "min_groups_per_target_type": 4})
    assert "groups_per_target_type" in result["policy_failures"]


def test_disjoint_manifest_passes_and_json_is_deterministic(tmp_path):
    candidate = write_roster(tmp_path / "candidate.json", passing_cases(tmp_path))
    result = inventory.audit(candidate, [], policy=policy())
    assert result["eligible"] is True
    assert result["counts"]["groups"] == 60
    assert inventory.dumps(result) == inventory.dumps(inventory.audit(candidate, [], policy=policy()))


def test_cli_valid_but_ineligible_returns_one(tmp_path, capsys):
    case = make_case(tmp_path, "one", group="g", domain="d", target_type="classification")
    candidate = write_roster(tmp_path / "candidate.json", [case])
    code = inventory.main(["--candidate", str(candidate)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["eligible"] is False


def test_cli_parse_failure_returns_two_and_json(tmp_path, capsys):
    candidate = tmp_path / "bad.json"
    candidate.write_text("{")
    code = inventory.main(["--candidate", str(candidate)])
    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["eligible"] is False and "error" in output


def test_cli_output_failure_returns_two_without_traceback(tmp_path, capsys):
    candidate = write_roster(tmp_path / "candidate.json", [make_case(tmp_path, "one", group="g", domain="d", target_type="classification")])
    code = inventory.main(["--candidate", str(candidate), "--out", str(tmp_path / "missing" / "audit.json")])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["eligible"] is False


def test_cli_accepts_multiple_consumed_manifests(tmp_path, capsys):
    candidate = write_roster(tmp_path / "candidate.json", [make_case(tmp_path, "one", group="g", domain="d", target_type="classification")])
    consumed_a = write_roster(tmp_path / "a.json", [make_case(tmp_path, "a", group="a", domain="a", target_type="classification")])
    consumed_b = write_roster(tmp_path / "b.json", [make_case(tmp_path, "b", group="b", domain="b", target_type="classification")])
    code = inventory.main(["--candidate", str(candidate), "--consumed", str(consumed_a), str(consumed_b)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["consumed"] == sorted([str(consumed_a.resolve()), str(consumed_b.resolve())])


def test_unknown_target_type_and_inconsistent_group_are_rejected(tmp_path):
    unknown = make_case(tmp_path, "unknown", group="g", domain="d", target_type="unknown")
    with pytest.raises(ValueError, match="unknown target_type"):
        inventory.audit(write_roster(tmp_path / "unknown.json", [unknown]), [], policy=synthetic_policy())
    first = make_case(tmp_path, "first", group="same", domain="d1", target_type="classification")
    second = make_case(tmp_path, "second", group="same", domain="d2", target_type="classification")
    with pytest.raises(ValueError, match="group maps inconsistently"):
        inventory.audit(write_roster(tmp_path / "inconsistent.json", [first, second]), [], policy=synthetic_policy())


def test_empty_domain_and_duplicate_case_are_rejected(tmp_path):
    missing = make_case(tmp_path, "missing", group="g", domain="d", target_type="classification")
    missing["domain"] = ""
    with pytest.raises(ValueError, match="domain"):
        inventory.audit(write_roster(tmp_path / "missing.json", [missing]), [], policy=synthetic_policy())
    first = make_case(tmp_path, "duplicate-a", group="g1", domain="d", target_type="classification")
    second = make_case(tmp_path, "duplicate-b", group="g2", domain="d", target_type="classification")
    first["id"] = second["id"] = "duplicate"
    with pytest.raises(ValueError, match="duplicate"):
        inventory.audit(write_roster(tmp_path / "duplicate.json", [first, second]), [], policy=synthetic_policy())


def test_arbitrary_expected_runs_are_rejected(tmp_path):
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"version": 1, "expected_runs": [{"group": "g", "input_digest": "fake"}]}))
    with pytest.raises(ValueError, match="expected_runs"):
        inventory.audit(path, [], policy=synthetic_policy())


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "baselines.evaluation", "inventory", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_subprocess_cli_has_defined_exit_statuses(tmp_path):
    eligible = write_roster(tmp_path / "eligible.json", passing_cases(tmp_path))
    valid = run_cli("--candidate", eligible)
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["eligible"] is True
    invalid = write_roster(tmp_path / "invalid.json", [make_case(tmp_path, "invalid", group="g", domain="d", target_type="classification")])
    rejected = run_cli("--candidate", invalid)
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["eligible"] is False
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    failed = run_cli("--candidate", malformed)
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["eligible"] is False
    assert "Traceback" not in failed.stderr


def test_package_cli_dispatches_inventory_command(tmp_path, capsys):
    case = make_case(tmp_path, "one", group="g", domain="d", target_type="classification")
    candidate = write_roster(tmp_path / "candidate.json", [case])
    code = evaluation_main.main(["inventory", "--candidate", str(candidate)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
