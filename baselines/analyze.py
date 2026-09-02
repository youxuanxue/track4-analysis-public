#!/usr/bin/env python3
"""Container entrypoint for the extract-then-predict agent.

The harness runs the image as::

    docker run <image> analyze --task /input/task.json \\
        --corpus /input/corpus/ --out /output/answer.json

so ``analyze`` arrives as this script's first positional. Delegates to
``strong_rag_baseline.cli:main``, which already declares that verb.

Official analyze is extract-then-predict. It does not start a localhost
model server and does not read ``$MODEL_ENDPOINT``. A developer-machine
GGUF experiment path is ``--local-llama`` (default OFF).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strong_rag_baseline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
