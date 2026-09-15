"""Lifecycle effects follow reverified evidence, not a caller-supplied PASS."""

import copy
import json

import pytest

from baselines.evaluation import batch, lifecycle
from baselines.evaluation.acceptance import digest, load_policy
from baselines.evaluation.tests.test_batch import completed_report
from baselines.evaluation.tests.test_runner_integration import build_unit
from scoring.tests.synthetic import outcome_for


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    unit = build_unit(tmp_path / "inputs")
    truth = tmp_path / "truth.json"
    batch.write_new(truth, outcome_for())
    manifest = tmp_path / "manifest.json"
    batch.write_new(
        manifest,
        {
            "version": 1,
            "cases": [
                {
                    "id": "synthetic",
                    "unit_dir": str(unit),
                    "truth_path": str(truth),
                    "group": "synthetic-event",
                    "domain": "synthetic-domain",
                    "split": "test",
                }
            ],
        },
    )
    versions = {
        role: {"repo": role, "python": "synthetic", "image": None}
        for role in ("before", "after")
    }
    identity = {
        "git_dirty": False,
        "source_digest": "source",
        "toolkit_source_digest": "toolkit",
    }

    def snapshot(spec):
        return {**copy.deepcopy(identity), "git_commit": spec["repo"]}

    monkeypatch.setattr(batch, "snapshot", snapshot)
    lifecycle.initialize(root, versions["before"])
    directory = batch.register(root, manifest, versions, [1])
    lifecycle.attach(root, directory)
    return root, directory, identity


def complete(directory):
    record = batch.read_registration(directory)
    for role in ("before", "after"):
        completed_report(directory, role, record)
    return batch.decide(directory)


def test_missing_acceptance_keeps_incumbent_and_closes_batch(selected):
    root, directory, _ = selected
    complete(directory)
    state = lifecycle.resolve(root)
    assert state["incumbent"]["identity"]["git_commit"] == "before"
    assert state["incumbent"]["qualification"] == "bootstrap"
    assert state["production_candidate"] is None
    assert state["pending"] is None
    assert state["consumed"] == [directory.name]
    with pytest.raises(ValueError, match="no pending"):
        lifecycle.resolve(root)
    with pytest.raises(ValueError, match="before either"):
        lifecycle.attach(root, directory)


def test_edited_pass_decision_cannot_promote(selected):
    root, directory, _ = selected
    decision = complete(directory)
    decision["goals"]["G1"]["status"] = "PASS"
    decision["goals"]["G2"]["status"] = "PASS"
    (directory / "decision.json").write_text(json.dumps(decision))
    old_head = lifecycle.status(root)["journal_head"]
    with pytest.raises(ValueError, match="saved decision differs"):
        lifecycle.resolve(root)
    assert lifecycle.status(root)["journal_head"] == old_head


@pytest.mark.parametrize("change", ["report", "source", "policy"])
def test_changed_evidence_never_changes_incumbent(selected, monkeypatch, change):
    root, directory, identity = selected
    complete(directory)
    if change == "report":
        path = directory / "after/report.json"
        report = json.loads(path.read_text())
        report["runs"][0]["assessment"]["development_score"] = 0.8
        path.write_text(json.dumps(report))
    elif change == "source":
        identity["source_digest"] = "changed"
    else:
        policy = load_policy()
        policy["min_mean_gain"] = 0.02
        monkeypatch.setattr(lifecycle, "load_policy", lambda: policy)
    with pytest.raises(ValueError):
        lifecycle.resolve(root)
    assert lifecycle.status(root)["pending"]["batch_id"] == directory.name


def test_interrupted_batch_can_close_but_not_reuse_events(selected):
    root, directory, _ = selected
    batch.write_new(
        directory / "before.started.json", {"registration_digest": directory.name}
    )
    state = lifecycle.abandon(root, "process exited before receipt")
    assert state["pending"] is None and directory.name in state["consumed"]
    assert state["incumbent"]["identity"]["git_commit"] == "before"
    with pytest.raises(FileExistsError):
        batch.run(directory, "before")
    assert batch.read_registration(directory)["expected_runs"]


def test_selection_must_precede_results_and_match_current_incumbent(selected):
    root, directory, _ = selected
    # Use another lifecycle with the same batch root only after removing its
    # synthetic journal; this models two independent invalid selection attempts.
    import shutil

    shutil.rmtree(root / "lifecycle")
    record = batch.read_registration(directory)
    lifecycle.initialize(root, record["versions"]["after"]["spec"])
    with pytest.raises(ValueError, match="current incumbent"):
        lifecycle.attach(root, directory)
    batch.write_new(
        directory / "after.started.json", {"registration_digest": directory.name}
    )
    with pytest.raises(ValueError, match="before either"):
        lifecycle.attach(root, directory)


def test_dirty_checkout_cannot_initialize(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "snapshot", lambda _: {"git_dirty": True})
    with pytest.raises(ValueError, match="clean immutable"):
        lifecycle.initialize(tmp_path / "registry", {"repo": "synthetic"})


def test_corrupt_journal_refuses_further_transitions(selected):
    root, _, _ = selected
    path = root / "lifecycle/00000000.json"
    record = json.loads(path.read_text())
    record["event"]["data"]["version"]["identity"]["source_digest"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash chain"):
        lifecycle.abandon(root, "diagnostic")


def test_failed_atomic_append_has_no_transition_and_can_resume(selected, monkeypatch):
    root, _, _ = selected
    previous = lifecycle.status(root)["journal_head"]
    original = batch.os.link

    def fail(*args):
        raise OSError("publication interrupted")

    monkeypatch.setattr(batch.os, "link", fail)
    with pytest.raises(OSError, match="interrupted"):
        lifecycle.abandon(root, "interrupted")
    assert lifecycle.status(root)["journal_head"] == previous
    monkeypatch.setattr(batch.os, "link", original)
    assert lifecycle.abandon(root, "cancelled")["pending"] is None


def test_atomic_record_never_overwrites_existing_decision(tmp_path):
    path = tmp_path / "decision.json"
    batch.write_new(path, {"accepted": False})
    with pytest.raises(FileExistsError):
        batch.write_new(path, {"accepted": True})
    assert json.loads(path.read_text()) == {"accepted": False}


def test_full_synthetic_g2_promotes_then_restores_frozen_baseline(
    tmp_path, monkeypatch
):
    from baselines.evaluation.tests.test_acceptance import reports, checks_by_name
    from baselines.evaluation.tests.test_faults import synthetic_suite

    root = tmp_path / "registry"
    fault_path, runtime, _ = synthetic_suite(tmp_path / "recovery")
    policy = load_policy()
    # Keep all sample/gain thresholds; constant paired gains need few bootstrap draws here.
    policy["bootstrap_samples"] = 100
    monkeypatch.setattr(batch, "load_policy", lambda: policy)
    monkeypatch.setattr(lifecycle, "load_policy", lambda: policy)
    versions = {
        role: {
            "spec": {"repo": role, "python": "synthetic", "image": "synthetic"},
            "identity": {
                **runtime,
                "git_commit": role,
                "git_dirty": False,
                "source_digest": role,
                "toolkit_source_digest": "same",
            },
        }
        for role in ("before", "after")
    }
    monkeypatch.setattr(
        batch,
        "snapshot",
        lambda spec: copy.deepcopy(versions[spec["repo"]]["identity"]),
    )
    lifecycle.initialize(root, versions["before"]["spec"])
    before, after = reports()
    fields = (
        "case_id",
        "seed",
        "group",
        "domain",
        "split",
        "target_type",
        "input_digest",
        "truth_digest",
    )
    record = {
        "policy_digest": digest(policy),
        "versions": versions,
        "mode": "grounded",
        "profile": "smoke",
        "expected_runs": [{k: row[k] for k in fields} for row in before["runs"]],
        "faults": {
            role: {
                "path": str(fault_path),
                "sha256": batch.hashlib.sha256(fault_path.read_bytes()).hexdigest(),
            }
            for role in versions
        },
    }
    directory = root / "batches" / digest(record)
    directory.mkdir(parents=True)
    batch.write_new(directory / "registration.json", record)
    (root / "events").mkdir()
    for key in ("group", "input_digest"):
        for value in {row[key] for row in record["expected_runs"]}:
            batch.write_new(
                root / "events" / digest({key: value}), {"batch_id": directory.name}
            )
    lifecycle.attach(root, directory)
    for role, report in (("before", before), ("after", after)):
        report.update(
            provenance=versions[role]["identity"], mode="grounded", profile="smoke"
        )
        for row in report["runs"]:
            row["resource_contract"] = {
                "timeout_s": 60,
                "cpus": 1,
                "memory_bytes": 1024**3,
                "network": "none",
            }
            row["timeout_s"] = 60
            row["execution"].update(
                isolation="docker-network-none",
                process_elapsed_s=1,
                resource_observation={
                    "image_id": runtime["image_id"],
                    "cpus": 1,
                    "memory_bytes": 1024**3,
                    "network": "none",
                    "oom_killed": False,
                },
            )
            row["diagnostics"] = {
                "request_ledger": {
                    "version": 1,
                    "source": "offline",
                    "attempts": [],
                    "attempts_used": 0,
                }
            }
            folder = directory / role / row["case_id"] / f"seed-{row['seed']}"
            folder.mkdir(parents=True)
            batch.write_new(folder / "diagnostics.json", row["diagnostics"])
            batch.write_new(folder / "result.json", row)
        report_path = directory / role / "report.json"
        batch.write_new(report_path, report)
        batch.write_new(
            directory / f"{role}.started.json", {"registration_digest": directory.name}
        )
        batch.write_new(
            directory / f"{role}.receipt.json",
            {
                "registration_digest": directory.name,
                "report_path": str(report_path.resolve()),
                "report_digest": digest(report),
                "returncode": 0,
            },
        )
    decision = batch.decide(directory)
    assert decision["goals"]["G1"]["status"] == "PASS"
    assert decision["goals"]["G2"]["status"] == "PASS"
    assert (
        checks_by_name(decision["goals"]["G3"]["checks"])["quality_acceptance"][
            "status"
        ]
        == "PASS"
    )
    assert decision["goals"]["G3"]["status"] == "UNMEASURED"
    state = lifecycle.resolve(root)
    assert state["incumbent"]["identity"] == versions["after"]["identity"]
    assert state["incumbent"]["qualification"] == "G2"
    assert state["rollback"][0]["identity"] == versions["before"]["identity"]
    assert state["production_candidate"] is None
    assert lifecycle.status(root)["incumbent"] == state["incumbent"]
    state = lifecycle.rollback(root, "synthetic post-promotion regression")
    assert state["incumbent"]["identity"] == versions["before"]["identity"]
    assert state["incumbent"]["qualification"] == "bootstrap"
    assert state["rollback"] == []
    with pytest.raises(ValueError, match="no retained"):
        lifecycle.rollback(root, "already restored")
