## Executive summary (read this first)

This directory is an optional local-dev home for a Qwen2.5-7B-Instruct Q4_K_M
GGUF. The file is about 4.5 GB and is gitignored — do not commit it. Official
`analyze` does not read this directory, does not start llama.cpp, and does not
bake the weights into the submission image. Use it only with `--local-llama`
(or `T4_LOCAL_LLAMA=1`) on a developer machine. A missing file just falls
back to extract-then-predict.

## Download the GGUF from Hugging Face

Pinned file: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from
`bartowski/Qwen2.5-7B-Instruct-GGUF`.

From the repository root:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --local-dir baselines/models

bash baselines/scripts/ensure_gguf.sh baselines/models
```

Direct URL:

```
https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Then:

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --local-llama \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json
```
