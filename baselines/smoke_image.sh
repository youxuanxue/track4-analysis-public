#!/usr/bin/env bash
# Smoke the real submission image with Docker and the offline fallback.
# A local CLI run is a separate check and cannot validate a container.
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
if grep -Eiq 'llama-server|ensure_gguf|Qwen2.5-7B-Instruct-Q4_K_M' "$DOCKERFILE"; then
  die "submission image must not bake a localhost model server or GGUF"
fi
if grep -Eiq 'torch|transformers|tensorflow|cuda' "$DOCKERFILE"; then
  die "submission image must not bake a GPU/LLM stack"
fi

command -v docker >/dev/null 2>&1 || die "Docker is unavailable; container validation not performed. Run baselines/analyze.py separately for a local CLI check."
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable; container validation not performed. Run baselines/analyze.py separately for a local CLI check."

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
OUT_JSON="$OUT_DIR/answer.json"
rm -f "$OUT_JSON"

echo "smoke_image: building $IMAGE"
docker build --platform linux/amd64 -f "$DOCKERFILE" -t "$IMAGE" "$ROOT/baselines"
CONTAINER_ID="$(docker create --platform linux/amd64 --network=none \
  -v "$UNIT":/input:ro \
  "$IMAGE" \
  analyze --task /input/task.json --corpus /input/corpus --out /tmp/answer.json --mock)"
trap 'docker rm -f "$CONTAINER_ID" >/dev/null' EXIT
docker start --attach "$CONTAINER_ID"
# Copy through Docker: host /tmp may not be mounted by Docker's Linux VM.
docker cp "$CONTAINER_ID:/tmp/answer.json" "$OUT_JSON"
echo "smoke_image: docker --network=none run exited 0"

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
print(f"smoke_image: container fallback output verified: {path}, {len(preds)} entities")
print("smoke_image: production endpoint, NLI admission and predictive quality were not tested")
PY
