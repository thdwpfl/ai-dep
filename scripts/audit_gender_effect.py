#!/usr/bin/env python3
"""Audit recommendation changes caused only by counterfactual M/F gender."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.checkpoint import load_model_checkpoint  # noqa: E402
from coldstart_rec.data import (  # noqa: E402
    GENRES,
    MovieLens1MData,
    load_processed_data,
)
from coldstart_rec.episodes import make_episode_batch, user_batches  # noqa: E402
from coldstart_rec.evaluation import select_split_users  # noqa: E402
from coldstart_rec.fairness import compute_dee  # noqa: E402
from coldstart_rec.groups import (  # noqa: E402
    PAPER_GROUP_NAMES,
    paper_genre_membership,
)
from coldstart_rec.metrics import exposure_by_gender  # noqa: E402
from coldstart_rec.model import ColdStartSetEncoder  # noqa: E402
from coldstart_rec.utils import (  # noqa: E402
    resolve_device,
    seed_everything,
    utc_now,
    write_json,
)


GENDER_NAMES = {0: "male", 1: "female", 2: "unknown"}


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: Tensor | Sequence[float]) -> dict[str, float | int]:
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError("Cannot summarize an empty audit population")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Audit population contains a non-finite value")
    quantiles = torch.quantile(
        tensor, torch.tensor([0.5, 0.95, 0.99], dtype=tensor.dtype)
    )
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "median": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "max": float(tensor.max().item()),
    }


def _mean_std(values: Tensor) -> dict[str, float | int]:
    tensor = values.to(dtype=torch.float64).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError("Cannot summarize an empty score population")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Score population contains a non-finite value")
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def _rate(count: int, total: int) -> dict[str, float | int]:
    return {
        "count": int(count),
        "fraction": float(count / total) if total else 0.0,
    }


def _validate_inputs(
    *,
    support_sizes: Sequence[int],
    top_k: int,
    batch_size: int,
    max_users: int | None,
    thresholds: Sequence[float],
    bandwidth: float,
) -> tuple[list[int], list[float]]:
    supports = list(dict.fromkeys(int(value) for value in support_sizes))
    if not supports or any(value < 0 for value in supports):
        raise ValueError("support-sizes must contain non-negative integers")
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if max_users is not None and max_users <= 0:
        raise ValueError("max-users must be positive when supplied")
    audited_thresholds = list(dict.fromkeys(float(value) for value in thresholds))
    if not audited_thresholds or any(
        not math.isfinite(value) for value in audited_thresholds
    ):
        raise ValueError("thresholds must contain finite values")
    if not math.isfinite(float(bandwidth)) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and strictly positive")
    return supports, audited_thresholds


def _actual_gender_dee(
    scores: Tensor,
    genders: Tensor,
    item_groups: Tensor,
    *,
    thresholds: Sequence[float],
    bandwidth: float,
) -> dict[str, Any]:
    score_statistics = _mean_std(scores)
    score_by_gender: dict[str, dict[str, float | int]] = {}
    for gender in (0, 1):
        mask = genders == gender
        if bool(mask.any()):
            score_by_gender[GENDER_NAMES[gender]] = _mean_std(scores[mask])

    results: dict[str, Any] = {}
    for threshold in thresholds:
        dee = compute_dee(
            scores,
            genders,
            item_groups,
            threshold=float(threshold),
            bandwidth=float(bandwidth),
            huber_delta=0.01,
            expected_user_groups=[0, 1],
            expected_item_groups=[0, 1],
        )
        cell_probabilities = {
            f"{GENDER_NAMES.get(int(user), str(int(user)))}|group_{int(item)}": float(
                probability
            )
            for user, item, probability in zip(
                dee.cell_user_groups.tolist(),
                dee.cell_item_groups.tolist(),
                dee.conditional_probabilities.tolist(),
                strict=True,
            )
        }
        cell_counts = {
            f"{GENDER_NAMES.get(int(user), str(int(user)))}|group_{int(item)}": int(
                count
            )
            for user, item, count in zip(
                dee.cell_user_groups.tolist(),
                dee.cell_item_groups.tolist(),
                dee.cell_counts.tolist(),
                strict=True,
            )
        }
        threshold_key = str(float(threshold))
        results[threshold_key] = {
            "threshold": float(threshold),
            "bandwidth": float(bandwidth),
            "raw_dee": float(dee.raw_dee.item()),
            "global_probability": float(dee.global_probability.item()),
            "cell_probabilities": cell_probabilities,
            "cell_counts": cell_counts,
            "score_count": score_statistics["count"],
            "score_mean": score_statistics["mean"],
            "score_std": score_statistics["std"],
            "score_by_actual_gender": score_by_gender,
        }
    return results


@torch.inference_mode()
def audit_support_size(
    model: ColdStartSetEncoder,
    data: MovieLens1MData,
    users: np.ndarray,
    *,
    support_size: int,
    top_k: int,
    batch_size: int,
    thresholds: Sequence[float],
    bandwidth: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Audit one chronological support size over identical M/F counterfactuals."""

    model.eval()
    rng = np.random.default_rng(seed)
    all_catalog = torch.arange(data.num_movies, dtype=torch.long, device=device)
    paper_membership = paper_genre_membership(data.movie_genres, GENRES)
    fair_items = np.flatnonzero(paper_membership.sum(axis=1) == 1)
    group_0_items = np.flatnonzero(paper_membership[:, 0] == 1)
    group_1_items = np.flatnonzero(paper_membership[:, 1] == 1)
    fair_labels = paper_membership[fair_items].argmax(axis=1).astype(np.int64)
    fair_items_tensor = torch.as_tensor(fair_items, dtype=torch.long, device=device)
    group_0_tensor = torch.as_tensor(
        group_0_items, dtype=torch.long, device=device
    )
    group_1_tensor = torch.as_tensor(
        group_1_items, dtype=torch.long, device=device
    )
    fair_labels_tensor = torch.as_tensor(
        fair_labels, dtype=torch.long, device=device
    )

    signed_score_parts: list[Tensor] = []
    group_0_parts: list[Tensor] = []
    group_1_parts: list[Tensor] = []
    actual_score_parts: list[Tensor] = []
    actual_gender_parts: list[Tensor] = []
    actual_item_group_parts: list[Tensor] = []
    male_recommendations: list[np.ndarray] = []
    female_recommendations: list[np.ndarray] = []
    overlap_fractions: list[float] = []
    jaccards: list[float] = []
    exact_set_count = 0
    exact_order_count = 0
    top_1_same_count = 0
    support_counts: list[int] = []
    effective_top_ks: list[int] = []

    for batch_users in user_batches(users, batch_size, rng, shuffle=False):
        batch = make_episode_batch(
            data,
            batch_users,
            int(support_size),
            query_size=None,
            rng=rng,
            device=device,
        )
        batch_length = len(batch_users)
        male_genders = torch.full(
            (batch_length,), model.MALE, dtype=torch.long, device=device
        )
        female_genders = torch.full(
            (batch_length,), model.FEMALE, dtype=torch.long, device=device
        )
        male_user = model.encode_user(
            batch.support_items,
            batch.support_ratings,
            batch.support_mask,
            male_genders,
        )
        female_user = model.encode_user(
            batch.support_items,
            batch.support_ratings,
            batch.support_mask,
            female_genders,
        )
        catalog_items = all_catalog.unsqueeze(0).expand(batch_length, -1)
        male_scores = model.decode(male_user, catalog_items).float()
        female_scores = model.decode(female_user, catalog_items).float()

        available = torch.ones_like(male_scores, dtype=torch.bool)
        if batch.support_mask.numel():
            support_rows = (
                torch.arange(batch_length, device=device)
                .unsqueeze(1)
                .expand_as(batch.support_items)
            )
            available[
                support_rows[batch.support_mask],
                batch.support_items[batch.support_mask],
            ] = False

        signed_difference = male_scores - female_scores
        signed_score_parts.append(signed_difference[available].cpu())
        group_0_available = available[:, group_0_tensor]
        group_1_available = available[:, group_1_tensor]
        group_0_parts.append(
            signed_difference[:, group_0_tensor][group_0_available].cpu()
        )
        group_1_parts.append(
            (-signed_difference[:, group_1_tensor])[group_1_available].cpu()
        )

        actual_scores = torch.where(
            batch.genders.unsqueeze(1) == model.MALE,
            male_scores,
            female_scores,
        )
        fair_available = available[:, fair_items_tensor]
        actual_fair_scores = actual_scores[:, fair_items_tensor]
        actual_score_parts.append(actual_fair_scores[fair_available].cpu())
        expanded_genders = batch.genders.unsqueeze(1).expand_as(actual_fair_scores)
        actual_gender_parts.append(expanded_genders[fair_available].cpu())
        expanded_item_groups = fair_labels_tensor.unsqueeze(0).expand(
            batch_length, -1
        )
        actual_item_group_parts.append(
            expanded_item_groups[fair_available].cpu()
        )

        male_available_scores = male_scores.masked_fill(~available, -torch.inf)
        female_available_scores = female_scores.masked_fill(~available, -torch.inf)
        row_support_counts = batch.support_mask.sum(dim=1).cpu().tolist()
        support_counts.extend(int(value) for value in row_support_counts)
        for row, row_support_count in enumerate(row_support_counts):
            effective_k = min(top_k, data.num_movies - int(row_support_count))
            if effective_k <= 0:
                raise ValueError(
                    "No recommendation candidate remains after excluding support"
                )
            male_top = (
                torch.topk(male_available_scores[row], k=effective_k, sorted=True)
                .indices.cpu()
                .numpy()
            )
            female_top = (
                torch.topk(female_available_scores[row], k=effective_k, sorted=True)
                .indices.cpu()
                .numpy()
            )
            support = (
                batch.support_items[row, batch.support_mask[row]].cpu().numpy()
            )
            if np.intersect1d(male_top, support).size or np.intersect1d(
                female_top, support
            ).size:
                raise RuntimeError("A rated support item leaked into counterfactual top-k")

            male_recommendations.append(male_top)
            female_recommendations.append(female_top)
            effective_top_ks.append(effective_k)
            overlap = int(np.intersect1d(male_top, female_top).size)
            union = int(np.union1d(male_top, female_top).size)
            overlap_fractions.append(overlap / effective_k)
            jaccards.append(overlap / union)
            exact_set_count += int(overlap == effective_k)
            exact_order_count += int(np.array_equal(male_top, female_top))
            top_1_same_count += int(male_top[0] == female_top[0])

    signed_scores = torch.cat(signed_score_parts)
    group_0_scores = torch.cat(group_0_parts)
    group_1_scores = torch.cat(group_1_parts)
    actual_scores = torch.cat(actual_score_parts).float()
    actual_genders = torch.cat(actual_gender_parts).long()
    actual_item_groups = torch.cat(actual_item_group_parts).long()
    signed_summary = _distribution(signed_scores)
    absolute_summary = _distribution(signed_scores.abs())
    group_0_summary = _distribution(group_0_scores)
    group_1_summary = _distribution(group_1_scores)

    exposure = exposure_by_gender(
        male_recommendations + female_recommendations,
        [model.MALE] * len(male_recommendations)
        + [model.FEMALE] * len(female_recommendations),
        data.movie_genres,
        GENRES,
        top_k,
    )
    male_exposure = exposure["male"]
    female_exposure = exposure["female"]
    exposure_gap = {
        genre: abs(male_exposure[genre] - female_exposure[genre])
        for genre in GENRES
    }
    gap_values = list(exposure_gap.values())
    user_count = len(male_recommendations)

    return {
        "users": int(user_count),
        "actual_support_count": {
            "min": int(min(support_counts)),
            "mean": float(np.mean(support_counts)),
            "max": int(max(support_counts)),
        },
        "score_difference_male_minus_female": {
            "signed": signed_summary,
            "absolute": absolute_summary,
        },
        "counterfactual_top_k": {
            "requested_k": int(top_k),
            "effective_k_min": int(min(effective_top_ks)),
            "effective_k_max": int(max(effective_top_ks)),
            "overlap_fraction": _distribution(overlap_fractions),
            "jaccard": _distribution(jaccards),
            "exact_set": _rate(exact_set_count, user_count),
            "exact_order": _rate(exact_order_count, user_count),
            "top_1_same": _rate(top_1_same_count, user_count),
        },
        "paper_group_counterfactual": {
            "group_0": {
                "name": PAPER_GROUP_NAMES[0],
                "male_minus_female": group_0_summary,
            },
            "group_1": {
                "name": PAPER_GROUP_NAMES[1],
                "female_minus_male": group_1_summary,
            },
            "interaction_difference_in_differences": float(
                group_0_summary["mean"] + group_1_summary["mean"]
            ),
            "formula": "(male-female on group_0) - (male-female on group_1)",
        },
        "counterfactual_top_k_genre_exposure": {
            "male": male_exposure,
            "female": female_exposure,
            "absolute_gap": exposure_gap,
            "absolute_gap_summary": {
                "mean": float(np.mean(gap_values)),
                "max": float(np.max(gap_values)),
                "l1_sum": float(np.sum(gap_values)),
            },
        },
        "actual_gender_dee_by_threshold": _actual_gender_dee(
            actual_scores,
            actual_genders,
            actual_item_groups,
            thresholds=thresholds,
            bandwidth=bandwidth,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test"
    )
    parser.add_argument(
        "--support-sizes", type=int, nargs="+", default=[1, 3, 5, 10]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-users", type=int)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[3.0, 3.5, 4.0]
    )
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    support_sizes, thresholds = _validate_inputs(
        support_sizes=args.support_sizes,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_users=args.max_users,
        thresholds=args.thresholds,
        bandwidth=args.bandwidth,
    )
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise RuntimeError(
            f"Exactly one visible GPU is required; torch sees {visible_gpu_count}. "
            "Set CUDA_VISIBLE_DEVICES to a single index."
        )
    device = resolve_device(args.device, require_single_gpu=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data = load_processed_data(data_path)
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    checkpoint_dataset_fingerprint = str(checkpoint["dataset_fingerprint"])
    data_fingerprint = str(data.manifest["artifact"]["sha256"])
    if checkpoint_dataset_fingerprint != data_fingerprint:
        raise ValueError("Checkpoint and processed data fingerprints differ")

    users = select_split_users(data, args.split, args.max_users, args.seed)
    selected_genders = set(int(value) for value in data.user_genders[users])
    if not selected_genders.issubset({model.MALE, model.FEMALE}):
        raise ValueError("Gender counterfactual audit requires binary M/F split users")
    actual_gender_user_counts = {
        GENDER_NAMES[index]: int(np.count_nonzero(data.user_genders[users] == index))
        for index in (model.MALE, model.FEMALE)
    }

    started = time.perf_counter()
    support_results: dict[str, Any] = {}
    for support_size in support_sizes:
        support_results[str(support_size)] = audit_support_size(
            model,
            data,
            users,
            support_size=support_size,
            top_k=args.top_k,
            batch_size=args.batch_size,
            thresholds=thresholds,
            bandwidth=args.bandwidth,
            seed=args.seed,
            device=device,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "audit": "same-support gender counterfactual full-catalog audit",
        "created_at": utc_now(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "format_version": int(checkpoint["format_version"]),
            "model_class": str(checkpoint["model_class"]),
            "model_config": dict(checkpoint["model_config"]),
            "epoch": int(checkpoint["epoch"]),
            "dee_coefficient": float(
                checkpoint.get("config", {}).get("train", {}).get(
                    "dee_coefficient", 0.0
                )
            ),
            "dataset_fingerprint": checkpoint_dataset_fingerprint,
        },
        "data": {
            "path": str(data_path),
            "dataset": str(data.manifest.get("dataset", "unknown")),
            "dataset_fingerprint": data_fingerprint,
            "split": args.split,
            "selected_users": int(len(users)),
            "actual_gender_user_counts": actual_gender_user_counts,
            "catalog_movies": int(data.num_movies),
        },
        "runtime": {
            "requested_device": args.device,
            "device": str(device),
            "visible_gpu_count": int(visible_gpu_count),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "parameters": {
            "support_sizes": support_sizes,
            "top_k": int(args.top_k),
            "batch_size": int(args.batch_size),
            "max_users": args.max_users,
            "thresholds": thresholds,
            "bandwidth": float(args.bandwidth),
            "seed": int(args.seed),
        },
        "protocol": {
            "users": f"user-disjoint {args.split} split",
            "support": "earliest timestamp-ordered ratings",
            "counterfactual": (
                "identical support movie indices, ratings, and timestamps; "
                "only the M/F gender input changes"
            ),
            "score_candidates": "full catalog excluding rated support movies",
            "top_k_candidates": "full catalog excluding rated support movies",
            "dee_candidates": (
                "unambiguous paper group-0/group-1 movies excluding rated support"
            ),
            "paper_group_0": PAPER_GROUP_NAMES[0],
            "paper_group_1": PAPER_GROUP_NAMES[1],
            "genre_exposure": (
                "multi-label genre memberships per top-k position; rates need "
                "not sum to one"
            ),
        },
        "support_sizes": support_results,
    }
    write_json(result, output_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
