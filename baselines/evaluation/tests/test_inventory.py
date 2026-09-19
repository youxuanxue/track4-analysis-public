import json
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
    return {"min_domains": 3, "min_groups_per_domain": 2, "min_groups_per_target_type": 2}


def test_group_overlap_is_reported_and_ineligible(tmp_path):
    cases = [make_case(tmp_path, "candidate", group="g1", domain="d1", target_type="classification")]
    candidate = write_roster(tmp_path / "candidate.json", cases)
    consumed = write_roster(tmp_path / "consumed.json", [dict(cases[0], id="old")])
    result = inventory.audit(candidate, [consumed], minimums=policy())
    assert result["overlap"]["groups"] == ["g1"]
    assert result["eligible"] is False


def test_renamed_case_with_identical_input_is_overlap(tmp_path):
    first = make_case(tmp_path, "same-input", group="g1", domain="d1", target_type="classification")
    second = dict(first, id="renamed", unit_dir=first["unit_dir"], group="other")
    candidate = write_roster(tmp_path / "candidate.json", [first])
    consumed = write_roster(tmp_path / "consumed.json", [second])
    result = inventory.audit(candidate, [consumed], minimums={"min_domains": 1, "min_groups_per_domain": 1, "min_groups_per_target_type": 1})
    assert result["overlap"]["input_digests"]
    assert result["eligible"] is False


def test_insufficient_domains_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group=f"g{i}", domain="only", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], minimums=policy())
    assert result["counts"]["domains"] == {"only": 3}
    assert "domains" in result["policy_failures"]


def test_insufficient_per_domain_groups_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group="same", domain=f"d{i}", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], minimums=policy())
    assert "groups_per_domain" in result["policy_failures"]


def test_insufficient_target_type_groups_are_ineligible(tmp_path):
    cases = [make_case(tmp_path, f"c{i}", group="same", domain=f"d{i}", target_type="classification") for i in range(3)]
    result = inventory.audit(write_roster(tmp_path / "candidate.json", cases), [], minimums=policy())
    assert "groups_per_target_type" in result["policy_failures"]


def test_disjoint_manifest_passes_and_json_is_deterministic(tmp_path):
    cases = [
        make_case(tmp_path, "c", group="gc", domain="c", target_type="classification"),
        make_case(tmp_path, "r", group="gr", domain="r", target_type="regression"),
        make_case(tmp_path, "k", group="gk", domain="k", target_type="ranking"),
        make_case(tmp_path, "c2", group="gc2", domain="c", target_type="classification"),
        make_case(tmp_path, "r2", group="gr2", domain="r", target_type="regression"),
        make_case(tmp_path, "k2", group="gk2", domain="k", target_type="ranking"),
    ]
    candidate = write_roster(tmp_path / "candidate.json", cases)
    result = inventory.audit(candidate, [], minimums=policy())
    assert result["eligible"] is True
    assert inventory.dumps(result) == inventory.dumps(inventory.audit(candidate, [], minimums=policy()))


def test_cli_returns_nonzero_and_writes_audit_on_policy_failure(tmp_path, capsys):
    case = make_case(tmp_path, "one", group="g", domain="d", target_type="classification")
    candidate = write_roster(tmp_path / "candidate.json", [case])
    code = inventory.main(["--candidate", str(candidate), "--minimum-domains", "2"])
    assert code != 0
    output = json.loads(capsys.readouterr().out)
    assert output["eligible"] is False


def test_package_cli_dispatches_inventory_command(tmp_path, capsys):
    case = make_case(tmp_path, "one", group="g", domain="d", target_type="classification")
    candidate = write_roster(tmp_path / "candidate.json", [case])
    code = evaluation_main.main(["inventory", "--candidate", str(candidate)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
