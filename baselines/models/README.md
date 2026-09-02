## Executive summary (read this first)

This directory is the optional local home for the Qwen2.5-7B-Instruct Q4_K_M
GGUF that the submission image serves on 127.0.0.1. The file is about 4.5 GB
and is gitignored — do not commit it. At `docker build` time,
`scripts/ensure_gguf.sh` copies a file already sitting here, or downloads it
from Hugging Face if the directory is empty. `analyze` then starts llama.cpp
bound to loopback and posts to that local OpenAI-compatible API; it does not
call `$MODEL_ENDPOINT` or a vendor host. Official scoring may attach an
accelerator, but the same image still has to finish when only the CPU is
available, and a failed local server falls back to extract-then-predict.

## Download the GGUF from Hugging Face

Pinned file: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from
`bartowski/Qwen2.5-7B-Instruct-GGUF` (Qwen2.5-7B-Instruct, Q4_K_M quant).

From the repository root:

```bash
# huggingface-cli (preferred)
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --local-dir baselines/models

# or the same fetch the image build uses
bash baselines/scripts/ensure_gguf.sh baselines/models
```

Direct URL (same object the script pulls):

```
https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

The expected path after either command is
`baselines/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`.

## How the container boots the local API

1. The image compiles `llama-server` (llama.cpp `v0.3.0`) and places the GGUF
   at `/opt/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
2. The harness runs `analyze --task --corpus --out`. That is the first argv.
3. `analyze` starts `llama-server -m <gguf> --host 127.0.0.1` and waits on
   `http://127.0.0.1:<port>/health`.
4. Each entity is a POST to `http://127.0.0.1:<port>/v1/chat/completions`.
5. If the binary, the GGUF, or the health check is missing, the agent writes
   the extract-then-predict answer instead. Public-dev rows stay pinned to
   `baselines/strong_rag_baseline/tests/locks/official_gate_d4d0584.json`.
