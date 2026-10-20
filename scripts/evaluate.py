#!/usr/bin/env python3
"""Evaluate a checkpoint on genuine user-disjoint cold-start episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.checkpoint import load_model_checkpoint  # noqa: E402
from coldstart_rec.data import load_processed_data  # noqa: E402
from coldstart_rec.evaluation import evaluate_model  # noqa: E402
from coldstart_rec.utils import resolve_device, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--support-sizes", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    device = resolve_device("cuda")
    data = load_processed_data(args.data)
    model, payload = load_model_checkpoint(args.checkpoint, device)
    if payload["dataset_fingerprint"] != data.manifest["artifact"]["sha256"]:
        raise ValueError("Checkpoint and processed data fingerprints differ")
    result = evaluate_model(
        model,
        data,
        split=args.split,
        support_sizes=args.support_sizes,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_users=args.max_users,
        threshold=args.threshold,
        bandwidth=args.bandwidth,
        seed=args.seed,
        device=device,
    )
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["checkpoint_epoch"] = payload["epoch"]
    write_json(result, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

