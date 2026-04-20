#!/usr/bin/env python3
"""
Day 3 recognizable entrypoint.

This script delegates execution to the Day 4 modular pipeline runner while
keeping the same command surface and output behavior.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
