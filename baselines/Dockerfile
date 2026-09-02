# Submission image: extract-then-predict Track 4 agent (stdlib + this package).
# No in-image model server, no GGUF bake. Official analyze is network-none
# safe. A developer-machine GGUF path lives behind --local-llama (default OFF)
# and is not this ENTRYPOINT.
#
# Build from the repository root:
#   docker build -f baselines/Dockerfile -t t4-analyze:latest baselines
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

# Runtime needs the package plus the public-unit lock JSON the reasoner pins to.
COPY strong_rag_baseline ./strong_rag_baseline
COPY analyze.py .

# The harness appends `analyze --task --corpus --out`. The verb is argv[1].
ENTRYPOINT ["python", "analyze.py"]
CMD ["analyze", "--help"]
