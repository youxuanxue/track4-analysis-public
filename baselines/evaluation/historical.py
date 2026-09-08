"""Build an external, vintage-bound ALFRED development benchmark.

Each economic event has three target views, not three independent observations.
Downloaded snapshots and realized values never belong in the submission image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import subprocess
import time
import tomllib
import urllib.parse
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import REPO, load_cases, public_roots, require_external

SERIES = {f"DGS{years}": years for years in (1, 2, 3, 5, 7, 10, 20, 30)}
KINDS = ("classification", "regression", "ranking")
SOURCE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def _download(url: str) -> bytes:
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "30",
            "--retry",
            "1",
            "--retry-delay",
            "1",
            "--max-filesize",
            "8000000",
            url,
        ],
        check=False,
        capture_output=True,
        timeout=70,
    )
    if result.returncode:
        raise RuntimeError(
            f"ALFRED download failed ({result.returncode}): {result.stderr.decode(errors='replace')[:500]}"
        )
    return result.stdout


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def parse_snapshot(payload: bytes, series: str, vintage: date) -> dict[date, Decimal]:
    """The returned column must attest the requested vintage, not today's data."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    column = f"{series}_{vintage:%Y%m%d}"
    if reader.fieldnames != ["observation_date", column]:
        raise ValueError("ALFRED returned the wrong series/vintage or multiple columns")
    rows: dict[date, Decimal] = {}
    seen: set[date] = set()
    for row in reader:
        if set(row) != {"observation_date", column} or row[column] is None:
            raise ValueError("malformed ALFRED CSV row")
        day = date.fromisoformat(row["observation_date"])
        if day in seen or day > vintage:
            raise ValueError("duplicate or post-vintage ALFRED observation")
        seen.add(day)
        if row[column] in ("", "."):
            continue
        value = Decimal(row[column])
        if not value.is_finite() or not math.isfinite(float(value)):
            raise ValueError("nonfinite ALFRED observation")
        rows[day] = value
    if not rows:
        raise ValueError("ALFRED snapshot has no observations")
    return rows


def snapshot(
    cache: Path, series: str, vintage: date, *, offline: bool
) -> tuple[dict, dict]:
    if series not in SERIES:
        raise ValueError("unsupported Treasury series")
    params = {
        "id": series,
        "cosd": (vintage - timedelta(days=65)).isoformat(),
        "coed": vintage.isoformat(),
        "vintage_date": vintage.isoformat(),
    }
    url = SOURCE + "?" + urllib.parse.urlencode(params)
    path = cache / f"{series}-{vintage}.csv"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        payload = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("url") != url
            or meta.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("cached ALFRED snapshot identity/checksum mismatch")
    else:
        if offline:
            raise ValueError(f"missing frozen snapshot: {path.name}")
        payload = _download(url)
        if len(payload) > 8_000_000:
            raise ValueError("ALFRED response exceeds download budget")
        parse_snapshot(payload, series, vintage)
        meta = {
            "url": url,
            "series": series,
            "vintage": vintage.isoformat(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": "public-domain US government observations; cite FRED/ALFRED and Federal Reserve H.15",
        }
        path.write_bytes(payload)
        write_json(meta_path, meta)
        time.sleep(0.2)
    return parse_snapshot(payload, series, vintage), meta


def _toml(data: dict, prefix: str = "") -> str:
    """Serialize this card's scalar/list tables without editing TOML text."""
    lines = [f"[{prefix}]"] if prefix else []
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {json.dumps(value, allow_nan=False)}")
    for key, value in data.items():
        if isinstance(value, dict):
            lines += ["", _toml(value, f"{prefix}.{key}" if prefix else key)]
    return "\n".join(lines) + "\n"


def build_event(out: Path, event: dict, snapshots: dict) -> list[dict]:
    cutoff = date.fromisoformat(event["cutoff"])
    resolution = date.fromisoformat(event["resolution"])
    if resolution <= cutoff or (resolution - cutoff).days > 60:
        raise ValueError("event needs a positive horizon of at most 60 days")
    common_start = set.intersection(*(set(snapshots[(s, cutoff)][0]) for s in SERIES))
    common_end = set.intersection(*(set(snapshots[(s, resolution)][0]) for s in SERIES))
    start, end = max(common_start), max(common_end)
    if start > cutoff or end > resolution or end <= cutoff:
        raise ValueError("event snapshots do not bound a future observation")
    if (cutoff - start).days > 7 or (resolution - end).days > 7:
        raise ValueError("stale snapshot: no common recent curve")
    entities, outcomes, documents, provenance = [], [], {}, []
    for series, years in SERIES.items():
        before, before_meta = snapshots[(series, cutoff)]
        after, after_meta = snapshots[(series, resolution)]
        entity_id = f"UST{years}Y"
        history = sorted(d for d in before if cutoff - timedelta(days=35) <= d <= start)
        # Keep the most recent observations, with a short earlier context, in a bounded table.
        selected = sorted(set(history[::5] + history[-5:]))
        doc_id = f"ALFRED_{series}_{cutoff:%Y%m%d}"
        title = f"{years}-Year U.S. Treasury constant-maturity yield"
        text = (
            f"{entity_id}: {title}. {series} observations in percent, available in the "
            f"ALFRED vintage dated {cutoff}. This is a historical table, not a future forecast.\n"
            f"date | {series}\n"
            + "\n".join(f"{day} | {before[day]}" for day in selected)
        )
        entities.append(
            {
                "entity_id": entity_id,
                "name": title,
                "series_fred": series,
                "maturity_years": years,
                "start_yield_pct": float(before[start]),
                "as_of": start.isoformat(),
                "unit": "bps_change",
            }
        )
        change = float((after[end] - before[start]) * 100)
        outcomes.append(
            {
                "entity_id": entity_id,
                "y": change,
                "true_label": "up" if change > 5 else "down" if change < -5 else "flat",
            }
        )
        documents[doc_id] = {
            "doc_id": doc_id,
            "doc_date": cutoff.isoformat(),
            "source": before_meta["url"],
            "text": text,
        }
        provenance.append(
            {"series": series, "input": before_meta, "outcome": after_meta}
        )
    cases = []
    for kind in KINDS:
        case_id = f"t4-local-curve-{cutoff:%Y%m%d}-{kind}"
        unit = out / "inputs" / case_id
        (unit / "corpus").mkdir(parents=True)
        prompt = (
            "Using only the frozen table and corpus, predict the cross-section of Treasury "
            "yield changes in basis points. The target is 100 * (the latest common yield "
            f"available in the ALFRED vintage of {resolution} minus start_yield_pct). "
            "Each starting yield is supplied. Do not treat historical observations as known future values. "
            "Provide a numeric point_forecast, a 90% interval on the basis-point change scale, and citations. "
        )
        if kind == "classification":
            prompt += (
                "Labels: up if change > 5 bps, down if change < -5 bps, flat otherwise."
            )
        elif kind == "ranking":
            prompt += "Rank larger yield changes higher; point_forecast is the predicted change, not a rank integer."
        target = {"name": "yield_change_bps", "type": kind, "unit": "bps_change"}
        if kind == "classification":
            target["labels"] = ["down", "flat", "up"]
        task = {
            "schema_version": "3",
            "task_id": case_id,
            "family": "local_curve_change",
            "cutoff_date": cutoff.isoformat(),
            "resolution_date": resolution.isoformat(),
            "target": target,
            "interval_level": 0.9,
            "prompt": prompt,
            "entities": entities,
        }
        write_json(unit / "task.json", task)
        for doc_id, doc in documents.items():
            write_json(unit / "corpus" / f"{doc_id}.json", doc)
        card = tomllib.loads((REPO / "units/t4-EXAMPLE-eps-beat/card.toml").read_text())
        card["task"].update(
            id=case_id,
            title="Independent local Treasury cross-section",
            split="public-dev",
            family="local_curve_change",
            prompt=prompt,
            cutoff_date=cutoff.isoformat(),
            resolution_date=resolution.isoformat(),
            target_type=kind,
            adversarial=False,
            faithfulness_rubric="Citations must support the submitted prediction using only the declared vintage corpus; local smoke does not test production NLI.",
        )
        card["metadata"].update(
            author_name="Local development",
            author_email="local@example.invalid",
            category="local_curve_change",
            tags=["analysis", "local-development", "rates"],
        )
        card["provenance"].update(
            data_source="ALFRED vintage snapshots",
            data_cutoff=cutoff.isoformat(),
            license="public-domain",
        )
        card["contamination"]["canary_guid"] = str(
            uuid.UUID(
                bytes=hashlib.sha256(("local-evaluation/" + case_id).encode()).digest()[
                    :16
                ],
                version=4,
            )
        )
        card["scoring"]["params"]["target_type"] = kind
        (unit / "card.toml").write_text(_toml(card))
        files = []
        for path in sorted([unit / "task.json", *unit.glob("corpus/*.json")]):
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(unit).as_posix(),
                    "role": "task" if path.name == "task.json" else "corpus",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "source": "ALFRED vintage snapshots",
                    "license": "public-domain",
                    "redistributable": True,
                    "pii_stripped": True,
                    "cutoff": cutoff.isoformat(),
                }
            )
        write_json(
            unit / "manifest.json",
            {"manifest_version": "2.0", "unit_id": case_id, "files": files},
        )
        truth = out / "truth" / f"{case_id}.json"
        write_json(
            truth,
            {
                "unit_id": case_id,
                "cutoff_date": cutoff.isoformat(),
                "target_type": kind,
                "outcomes": outcomes,
            },
        )
        cases.append(
            {
                "id": case_id,
                "unit_dir": f"inputs/{case_id}",
                "truth_path": f"truth/{case_id}.json",
                "split": event["split"],
                "group": event["id"],
            }
        )
    write_json(
        out / "provenance" / f"{event['id']}.json",
        {
            "event": event,
            "start_observation": start.isoformat(),
            "end_observation": end.isoformat(),
            "sources": provenance,
            "independent_events": 1,
            "limitations": "One rates domain with three correlated target views; not representative of hidden T4 families. Model training contamination is not certified.",
        },
    )
    return cases


def build(spec_path: Path, cache: Path, out: Path, *, offline: bool = False) -> Path:
    roots = public_roots()
    for path in (spec_path, cache, out):
        require_external(path, roots)
    spec = json.loads(spec_path.read_text())
    events = spec.get("events")
    if spec.get("version") != 1 or not isinstance(events, list) or not events:
        raise ValueError("expected version 1 and a nonempty events roster")
    seen = set()
    periods: dict[str, list[tuple[date, date]]] = {}
    for event in events:
        if set(event) != {"id", "cutoff", "resolution", "split"}:
            raise ValueError("invalid event fields")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", event["id"])
            or event["id"] in seen
        ):
            raise ValueError("invalid or duplicate event id")
        seen.add(event["id"])
        if event["split"] not in ("train", "calibration", "test"):
            raise ValueError("invalid event split")
        cutoff, resolution = (
            date.fromisoformat(event["cutoff"]),
            date.fromisoformat(event["resolution"]),
        )
        if not 0 < (resolution - cutoff).days <= 60:
            raise ValueError("event needs a positive horizon of at most 60 days")
        if resolution >= date.today():
            raise ValueError("historical event must already be resolved")
        periods.setdefault(event["split"], []).append((cutoff, resolution))
    splits = ("train", "calibration", "test")
    for i, earlier in enumerate(splits):
        for later in splits[i + 1 :]:
            if (
                earlier in periods
                and later in periods
                and max(end for _, end in periods[earlier])
                >= min(start for start, _ in periods[later])
            ):
                raise ValueError("earlier outcomes must precede later cutoffs")
    ordered = sorted(
        (date.fromisoformat(e["cutoff"]), date.fromisoformat(e["resolution"]))
        for e in events
    )
    if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("event windows must not overlap")
    if out.exists():
        raise ValueError("benchmark output must be new")
    cache.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for event in events:
        for key in ("cutoff", "resolution"):
            vintage = date.fromisoformat(event[key])
            missing = [
                series for series in SERIES if (series, vintage) not in snapshots
            ]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {
                    series: pool.submit(
                        snapshot, cache, series, vintage, offline=offline
                    )
                    for series in missing
                }
                for series, job in jobs.items():
                    snapshots[(series, vintage)] = job.result()
            print(f"Validated vintage {vintage}", flush=True)
    out.mkdir(parents=True)
    for folder in ("truth", "provenance"):
        (out / folder).mkdir()
    cases = [case for event in events for case in build_event(out, event, snapshots)]
    manifest = out / "manifest.json"
    write_json(manifest, {"version": 1, "cases": cases})
    load_cases(units=None, manifest=manifest)
    from qfbench2_common import manifest as manifests, taskcard

    for case in cases:
        unit = out / case["unit_dir"]
        _, errors = taskcard.load_and_validate(unit)
        errors += manifests.verify_manifest(unit)
        errors += manifests.assert_public_safe(unit)
        if errors:
            raise ValueError(
                f"generated unit failed validation: {case['id']}: {errors}"
            )
    write_json(
        out / "build.json",
        {
            "version": 1,
            "complete": True,
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "events": len(events),
            "cases": len(cases),
            "entities_per_case": len(SERIES),
            "official_score": None,
            "domain": "Treasury yield cross-sections only",
        },
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(build(args.spec, args.cache, args.out, offline=args.offline))
        return 0
    except Exception as exc:
        print(f"Historical build aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
