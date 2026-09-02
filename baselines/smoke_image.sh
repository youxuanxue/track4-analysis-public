#!/usr/bin/env bash
# Smoke the submission image contract. Prefers `docker build` + `--network=none`;
# if Docker is not on the machine, runs the same entrypoint locally and says so.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT="$ROOT/units/t4-EXAMPLE-eps-beat"
OUT_DIR="${1:-/tmp/t4-analyze-smoke}"
IMAGE="${T4_IMAGE:-t4-analyze:latest}"
DOCKERFILE="$ROOT/baselines/Dockerfile"
ENTRYPOINT="$ROOT/baselines/analyze.py"

die() { echo "smoke_image: $*" >&2; exit 1; }

test -f "$DOCKERFILE" || die "missing $DOCKERFILE"
test -f "$ENTRYPOINT" || die "missing $ENTRYPOINT"
test -f "$UNIT/task.json" || die "missing exemplar task.json"

grep -q 'FROM python:3.13' "$DOCKERFILE" || die "Dockerfile must FROM python:3.13"
grep -q 'LABEL qfbench2.interface_version="2.0"' "$DOCKERFILE" || die "missing interface_version LABEL"
grep -q 'ENTRYPOINT \["python", "analyze.py"\]' "$DOCKERFILE" || die "ENTRYPOINT must be python analyze.py"
grep -q 'llama-server' "$DOCKERFILE" || die "Dockerfile must build llama-server"
grep -q 'Qwen2.5-7B-Instruct-Q4_K_M' "$DOCKERFILE" || die "Dockerfile must pin the Qwen Q4_K_M GGUF"
grep -q 'ensure_gguf' "$DOCKERFILE" || die "Dockerfile must fetch or copy the GGUF"
if grep -Eiq '(^|[^A-Za-z0-9_])(torch|transformers|tensorflow)([^A-Za-z0-9_]|$)' "$DOCKERFILE"; then
  die "submission image must not bake torch/transformers/tensorflow"
fi

mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/answer.json"
rm -f "$OUT_JSON"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "smoke_image: docker is available — building $IMAGE"
  docker build -f "$DOCKERFILE" -t "$IMAGE" "$ROOT/baselines"
  docker run --rm --network=none \
    -v "$UNIT":/input:ro \
    -v "$OUT_DIR":/output \
    "$IMAGE" \
    analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json
  echo "smoke_image: docker --network=none run exited 0"
else
  echo "smoke_image: docker is not available on this machine; running analyze.py locally"
  python3 "$ENTRYPOINT" analyze \
    --task "$UNIT/task.json" \
    --corpus "$UNIT/corpus" \
    --out "$OUT_JSON"
  echo "smoke_image: local entrypoint exited 0 (image was not built here)"
fi

python3 - "$OUT_JSON" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
assert path.is_file(), path
answer = json.loads(path.read_text(encoding="utf-8"))
assert answer.get("task_id") == "t4-EXAMPLE-eps-beat", answer.get("task_id")
preds = answer.get("entity_predictions") or []
assert preds, "no entity_predictions"
assert preds[0].get("claims"), "expected grounded claims"
print(f"smoke_image: ok — {path} has {len(preds)} entit(y/ies)")
PY
