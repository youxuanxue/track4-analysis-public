#!/usr/bin/env bash
# Place Qwen2.5-7B-Instruct Q4_K_M into DEST_DIR.
# Used at image build (DEST_DIR=/opt/models) and locally (DEST_DIR=baselines/models).
# The GGUF is gitignored; this script downloads it from Hugging Face when absent.
set -euo pipefail

NAME="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
URL="https://huggingface.co/${REPO}/resolve/main/${NAME}"
DEST_DIR="${1:-/opt/models}"

mkdir -p "$DEST_DIR"
DEST="$DEST_DIR/$NAME"

if [[ -f "$DEST" && -s "$DEST" ]]; then
  echo "ensure_gguf: already present at $DEST ($(wc -c < "$DEST") bytes)"
  exit 0
fi

# A pre-copied file under the build context / a local models/ dir wins.
for candidate in \
  "/app/models/${NAME}" \
  "${DEST_DIR}/${NAME}" \
  "./models/${NAME}"; do
  if [[ -f "$candidate" && -s "$candidate" && "$candidate" != "$DEST" ]]; then
    echo "ensure_gguf: copying $candidate -> $DEST"
    cp -f "$candidate" "$DEST"
    exit 0
  fi
done

echo "ensure_gguf: downloading $URL"
tmp="${DEST}.partial"
curl -fL --retry 5 --retry-delay 8 --retry-all-errors \
  -A "t4-analyze-gguf-fetch" \
  -o "$tmp" \
  "$URL"
if [[ ! -s "$tmp" ]]; then
  echo "ensure_gguf: download produced an empty file" >&2
  rm -f "$tmp"
  exit 1
fi
mv -f "$tmp" "$DEST"
echo "ensure_gguf: wrote $DEST ($(wc -c < "$DEST") bytes)"
