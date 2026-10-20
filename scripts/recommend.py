#!/usr/bin/env python3
"""Print recommendations for a new user from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.inference import MovieRecommender  # noqa: E402
from coldstart_rec.utils import write_json  # noqa: E402


def parse_rating(value: str) -> tuple[int, float]:
    try:
        movie_id, rating = value.split(":", maxsplit=1)
        return int(movie_id), float(rating)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratings must use MOVIE_ID:SCORE") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/fair_best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--gender", default="unknown")
    parser.add_argument("--rating", action="append", type=parse_rating, default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = MovieRecommender(args.checkpoint, args.data)
    recommendations = engine.recommend(
        args.gender, dict(args.rating), top_k=args.top_k
    )
    rows = [row.to_dict() for row in recommendations]
    if args.output:
        write_json(rows, args.output)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
