#!/usr/bin/env python3
"""Download MovieLens 1M and build deterministic cold-start data artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.data import download_ml1m, preprocess_ml1m  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Parent of raw/ and processed/ directories (default: project data/).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--sha256",
        default=None,
        help="Optional expected SHA-256 for the downloaded ml-1m.zip.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload and atomically replace an existing raw extraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run download and preprocessing."""

    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    raw_path = download_ml1m(
        data_dir / "raw",
        expected_sha256=args.sha256,
        force=args.force_download,
    )
    artifact_path, manifest_path = preprocess_ml1m(
        raw_path,
        data_dir / "processed",
        seed=args.seed,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    print(f"Raw dataset: {raw_path}")
    print(f"Processed artifact: {artifact_path}")
    print(f"Split manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
