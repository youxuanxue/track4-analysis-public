"""The version a participant resolves must be the version the package ships.

Track 2 previously declared 2.1.0 in `pyproject.toml` and 2.0.0 in the package, so a participant who
found a version could not trust it. The four scorers now share one number, bumped together, and this
pins the two declarations in THIS repository to each other -- a cross-repo check is not possible from
here, but drift within a repo is what actually shipped.
"""

from __future__ import annotations

import pathlib
import re

from qfbench2_track_analysis.scoring import SCORER_VERSION, scorer_identity

REPO = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
OPERATIONAL_TOOLKIT_FILES = (
    ".github/workflows/ci.yml",
    "README.md",
    "SUBMISSION_CLI.md",
    "baselines/requirements.txt",
    "faithfulness/judge.py",
    "docs/CHAMPIONSHIP-PLAN.md",
    "pyproject.toml",
)


def _declared() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
    assert m, "pyproject.toml declares no version"
    return m.group(1)


def test_pyproject_and_scorer_version_agree():
    assert _declared() == SCORER_VERSION, (
        f"pyproject.toml says {_declared()} and SCORER_VERSION says {SCORER_VERSION}; "
        "a participant who found one of them could not trust it"
    )


def test_the_shared_version_is_what_the_owner_set():
    assert SCORER_VERSION == "3.1.0", (
        "all four track scorers share one version, bumped together (owner ruling 2026-09-11). "
        "Changing it here alone reintroduces exactly the drift this replaced."
    )


def test_current_operational_toolkit_pins_are_243():
    offenders = []
    for relative in OPERATIONAL_TOOLKIT_FILES:
        text = (REPO / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"qfbench2-common[^\n]*?(?:@v|==)(2\.4\.\d+)", text):
            if match.group(1) != "2.4.3":
                offenders.append(f"{relative}: {match.group(1)}")
        for match in re.finditer(r"Agenthon2026-public/blob/v(2\.4\.\d+)", text):
            if match.group(1) != "2.4.3":
                offenders.append(f"{relative}: {match.group(1)}")
    assert not offenders, "operational toolkit pins must be 2.4.3: " + ", ".join(offenders)


def test_each_operational_surface_contains_an_explicit_243_pin():
    expected = {
        ".github/workflows/ci.yml": 'Agenthon2026-public.git@v2.4.3#subdirectory=common',
        "README.md": 'Agenthon2026-public.git@v2.4.3#subdirectory=common',
        "SUBMISSION_CLI.md": "Agenthon2026-public/blob/v2.4.3/",
        "baselines/requirements.txt": 'Agenthon2026-public.git@v2.4.3#subdirectory=common',
        "faithfulness/judge.py": 'Agenthon2026-public.git@v2.4.3#subdirectory=common',
        "docs/CHAMPIONSHIP-PLAN.md": "toolkit 2.4.3",
        "pyproject.toml": "Agenthon2026-public.git@v2.4.3#subdirectory=common",
    }
    missing = [
        f"{relative}: {needle!r}"
        for relative, needle in expected.items()
        if needle not in (REPO / relative).read_text(encoding="utf-8")
    ]
    assert not missing, "each operational surface must carry an explicit v2.4.3 pin: " + ", ".join(missing)


def test_scorer_identity_carries_the_version():
    ident = scorer_identity()
    assert ident["scorer_version"] == SCORER_VERSION
    assert ident["scorer_package"].endswith(".scoring")


def test_the_package_itself_exposes_the_version():
    """A participant reads `qfbench2_track_analysis.__version__`, not the scoring module's constant.

    Track 4 shipped `SCORER_VERSION` in `scoring` and re-exported it, but exposed no
    `__version__` and no `scorer_identity` on the package -- and the tests above could not see
    that, because they import from `.scoring` directly and bypass the re-export entirely.
    """
    import qfbench2_track_analysis

    assert qfbench2_track_analysis.__version__ == SCORER_VERSION
    assert qfbench2_track_analysis.scorer_identity() == scorer_identity()
