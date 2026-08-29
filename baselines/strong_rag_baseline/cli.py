"""``analyze`` CLI for the strong RAG baseline.

Usage — exactly the argv the scoring harness issues (see SUBMISSION_CLI.md)::

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

The leading ``analyze`` is the container command and is accepted here; it is
optional when running the module by hand::

    python -m baselines.strong_rag_baseline.cli \
        --task   units/t4-EXAMPLE-eps-beat/task.json \
        --corpus units/t4-EXAMPLE-eps-beat/corpus \
        --out    /tmp/answer.json

Requires an OpenAI-compatible model server at ``$MODEL_ENDPOINT`` (injected by
the harness at scoring time; locally use ollama/llama.cpp or ``--mock`` for a
network-free smoke run with a canned model reply).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import run_entity
from .client import HTTPModelClient, MockModelClient, ModelClient
from .config import Config
from .formatter import build_answer
from .indexer import build_index
from .retriever import BM25Index

_MOCK_REPLY = json.dumps(
    {
        "label": None,
        "point_forecast": 0.0,
        "interval": {"level": 0.90, "lo": -1.0, "hi": 1.0},
        "evidence": [],
    }
)


def run(
    task_path: Path,
    corpus_dir: Path,
    out_path: Path,
    client: ModelClient,
    top_k: int,
) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    results = [
        run_entity(task, entity, index, corpus, client, top_k)
        for entity in task.get("entities", [])
    ]
    answer = build_answer(task, results, corpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(answer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The harness passes the verb as the container command, so it arrives as argv[0].
    # Optional, so hand-invocation without it keeps working. See baseline_agent/cli.py.
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a canned model reply (no network) — wiring smoke runs only.",
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    client: ModelClient = (
        MockModelClient(reply=_MOCK_REPLY) if args.mock else HTTPModelClient(config)
    )
    answer = run(args.task, args.corpus, args.out, client, config.top_k)
    n_claims = sum(len(e["claims"]) for e in answer["entity_predictions"])
    print(
        f"wrote {args.out} — {len(answer['entity_predictions'])} entities, "
        f"{n_claims} grounded claims"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
