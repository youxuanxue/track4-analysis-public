#!/usr/bin/env python3
"""Container entrypoint for the local-llama.cpp / extract-then-predict agent.

The harness runs the image as::

    docker run <image> analyze --task /input/task.json \\
        --corpus /input/corpus/ --out /output/answer.json

so ``analyze`` arrives as this script's first positional. Delegates to
``strong_rag_baseline.cli:main``, which already declares that verb.

The process starts llama.cpp on 127.0.0.1 when the baked GGUF is present,
then calls that loopback API. It does not read ``$MODEL_ENDPOINT``. A failed
local server falls back to extract-then-predict.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strong_rag_baseline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
