# Submission image: local llama.cpp (Qwen2.5-7B-Instruct Q4_K_M) on 127.0.0.1.
# analyze starts llama-server, then POSTs to http://127.0.0.1:<port>/v1.
# It does not call $MODEL_ENDPOINT or a vendor API. Missing weights / a
# failed health check fall back to extract-then-predict (network-none safe).
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

# Runtime C++ bits + build toolchain (purged after llama-server is linked).
# libgomp1 stays: llama.cpp's CPU kernels need it. No torch / transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pin llama.cpp so the image is reproducible. GGML_NATIVE=OFF so the binary
# runs on the eval CPU, not only the build host. CPU kernels only: the image
# must still finish when no accelerator is attached.
ARG LLAMA_CPP_REF=v0.3.0
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DGGML_NATIVE=OFF \
    && cmake --build /tmp/llama.cpp/build --target llama-server -j"$(nproc)" \
    && cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server \
    && rm -rf /tmp/llama.cpp \
    && apt-get purge -y build-essential cmake git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Comments-only today; keep the install so a future stdlib-adjacent pin is one line.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional pre-copied GGUF from the build context (gitignored). ensure_gguf.sh
# copies it when present, otherwise downloads from Hugging Face at build time.
COPY models /opt/models
COPY scripts/ensure_gguf.sh /usr/local/bin/ensure_gguf.sh
RUN chmod +x /usr/local/bin/ensure_gguf.sh \
    && /usr/local/bin/ensure_gguf.sh /opt/models

# Runtime needs the package plus the public-unit lock JSON the reasoner pins to.
COPY strong_rag_baseline ./strong_rag_baseline
COPY analyze.py .

# The harness appends `analyze --task --corpus --out`. The verb is argv[1].
ENTRYPOINT ["python", "analyze.py"]
CMD ["analyze", "--help"]
