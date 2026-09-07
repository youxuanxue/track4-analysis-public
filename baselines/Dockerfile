# Submission image: Track 4 house-endpoint agent (stdlib + this package).
# The harness supplies MODEL_ENDPOINT and MODEL_NAME; absent an endpoint,
# analyze uses the offline reasoner. Local GGUF experiments are opt-in.
#
# Build from the repository root:
#   docker build --platform linux/amd64 -f baselines/Dockerfile -t t4-analyze:latest baselines
#
# The harness then runs:
#   docker run --rm --network none|qfb2-eval \
#     -v <unit>:/input:ro -v <run>/output:/output \
#     t4-analyze:latest \
#     analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
FROM python:3.13-slim

# Required on every submission image (SUBMISSION_CLI.md); gate g0_integrity.
LABEL qfbench2.interface_version="2.0"

WORKDIR /app

# Comments-only today; keep the install so a future stdlib-adjacent pin is one line.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runtime modules only. Test fixtures must never enter the submission image.
COPY strong_rag_baseline/*.py ./strong_rag_baseline/
COPY analyze.py .

# The harness appends `analyze --task --corpus --out`. The verb is argv[1].
ENTRYPOINT ["python", "analyze.py"]
CMD ["analyze", "--help"]
