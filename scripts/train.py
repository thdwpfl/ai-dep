#!/usr/bin/env python3
"""Train one baseline or DEE-regularized cold-start model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.config import load_config  # noqa: E402
from coldstart_rec.training import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    summary = train_model(
        load_config(args.config),
        args.output_dir,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

