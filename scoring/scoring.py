"""Back-compat shim — the canonical module is `qfbench2_track_analysis.scoring`.

Kept so existing references (docs, tests, `sys.path` imports of `scoring.scoring`) keep working
unchanged. All public and underscore-prefixed symbols are re-exported.
"""

from __future__ import annotations

import pathlib
import sys

# Allow running from a repo checkout without installing the package.
_repo_root = str(pathlib.Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qfbench2_track_analysis import scoring as _impl  # noqa: E402
from qfbench2_track_analysis.scoring import (  # noqa: E402
    LEADERBOARD_SORT as LEADERBOARD_SORT,
    build_smoke_verifier as build_smoke_verifier,
    build_verifier as build_verifier,
    score_unit as score_unit,
)

# The globals().update() below re-exports everything at RUNTIME, which mypy cannot follow. The
# gates are underscore-prefixed and scoring/tests/test_g3_gates.py imports one directly, so a
# purely dynamic shim breaks `mypy --strict` on a test that was passing before. Named explicitly
# rather than silenced: the alternative is a type: ignore that hides the next one too.
from qfbench2_track_analysis.scoring import (  # noqa: E402
    _g0_integrity as _g0_integrity,
    _g1_schema as _g1_schema,
    _g2_cutoff_resource as _g2_cutoff_resource,
    _g3_domain_semantics as _g3_domain_semantics,
    hydrate as hydrate,
)

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
