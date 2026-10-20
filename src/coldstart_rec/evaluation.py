"""Cold-start validation and full test evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .data import GENRES, MovieLens1MData
from .episodes import make_episode_batch, user_batches
from .fairness import compute_dee
from .groups import paper_genre_membership
from .metrics import (
    exposure_by_gender,
    mean_metric_dict,
    ranking_metrics_for_user,
    regression_metrics,
)
from .model import ColdStartSetEncoder


def select_split_users(
    data: MovieLens1MData,
    split: str,
    max_users: int | None,
    seed: int,
) -> np.ndarray:
    """Select a deterministic gender-stratified subset of a user split."""
    users = data.split_user_indices(split)
    if max_users is None or max_users >= len(users):
        return users
    if max_users <= 0:
        raise ValueError("max_users must be positive or None")
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    genders = np.unique(data.user_genders[users])
    remaining = max_users
    for position, gender in enumerate(genders):
        candidates = users[data.user_genders[users] == gender]
        if position == len(genders) - 1:
            take = remaining
        else:
            take = int(round(max_users * len(candidates) / len(users)))
            take = min(max(take, 1), remaining - (len(genders) - position - 1))
        pieces.append(rng.choice(candidates, size=min(take, len(candidates)), replace=False))
        remaining -= len(pieces[-1])
    selected = np.concatenate(pieces)
    if len(selected) < max_users:
        unused = np.setdiff1d(users, selected, assume_unique=False)
        selected = np.concatenate(
            (selected, rng.choice(unused, size=max_users - len(selected), replace=False))
        )
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


@torch.inference_mode()
def quick_validate(
    model: ColdStartSetEncoder,
    data: MovieLens1MData,
    *,
    users: Sequence[int] | np.ndarray,
    support_size: int,
    query_size: int,
    batch_size: int,
    threshold: float,
    bandwidth: float,
    huber_delta: float,
    fairness_candidate_size: int,
    seed: int,
    device: torch.device,
    stress_margin: float = 0.0,
) -> dict[str, float]:
    """Efficient validation used for epoch checkpointing and bounded tuning."""
    model.eval()
    rng = np.random.default_rng(seed)
    paper_membership = paper_genre_membership(data.movie_genres, GENRES)
    # Validation selection must not depend on a noisy item sample: use the
    # exact dense cross-product over every unambiguous paper-group movie.
    candidates = np.flatnonzero(paper_membership.sum(axis=1) == 1)
    item_groups = paper_membership[candidates].argmax(axis=1).astype(np.int64)
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []
    ranking_rows: list[dict[str, float]] = []
    fairness_scores: list[Tensor] = []
    fairness_genders: list[Tensor] = []
    fairness_items: list[Tensor] = []
    counterfactual_group0: list[Tensor] = []
    counterfactual_group1: list[Tensor] = []

    for batch_users in user_batches(users, batch_size, rng, shuffle=False):
        batch = make_episode_batch(
            data,
            batch_users,
            support_size,
            query_size=query_size,
            rng=rng,
            device=device,
        )
        user = model.encode_user(
            batch.support_items,
            batch.support_ratings,
            batch.support_mask,
            batch.genders,
        )
        query_predictions = model.decode(user, batch.query_items)
        all_predictions.append(query_predictions[batch.query_mask].cpu())
        all_targets.append(batch.query_ratings[batch.query_mask].cpu())
        for row in range(len(batch_users)):
            mask = batch.query_mask[row]
            ranking_rows.append(
                ranking_metrics_for_user(
                    query_predictions[row, mask].float().cpu().numpy(),
                    batch.query_ratings[row, mask].float().cpu().numpy(),
                    k=10,
                )
            )

        candidate_tensor = torch.as_tensor(candidates, device=device).unsqueeze(0)
        candidate_tensor = candidate_tensor.expand(len(batch_users), -1)
        scores = model.decode(user, candidate_tensor)
        fairness_scores.append(scores.reshape(-1).float().cpu())
        fairness_genders.append(
            batch.genders.unsqueeze(1).expand_as(scores).reshape(-1).cpu()
        )
        fairness_items.append(
            torch.as_tensor(item_groups)
            .unsqueeze(0)
            .expand(len(batch_users), -1)
            .reshape(-1)
        )
        if model.config.architecture == "explicit_gender_genre":
            male_user = model.with_gender(
                user, torch.full_like(batch.genders, model.MALE)
            )
            female_user = model.with_gender(
                user, torch.full_like(batch.genders, model.FEMALE)
            )
            male_scores = model.decode(male_user, candidate_tensor)
            female_scores = model.decode(female_user, candidate_tensor)
            labels = torch.as_tensor(item_groups, dtype=torch.long, device=device)
            counterfactual_group0.append(
                (male_scores[:, labels == 0] - female_scores[:, labels == 0])
                .reshape(-1)
                .cpu()
            )
            counterfactual_group1.append(
                (female_scores[:, labels == 1] - male_scores[:, labels == 1])
                .reshape(-1)
                .cpu()
            )

    regression = regression_metrics(
        torch.cat(all_predictions), torch.cat(all_targets)
    )
    dee = compute_dee(
        torch.cat(fairness_scores),
        torch.cat(fairness_genders),
        torch.cat(fairness_items),
        threshold=threshold,
        bandwidth=bandwidth,
        huber_delta=huber_delta,
        expected_user_groups=[0, 1],
        expected_item_groups=[0, 1],
    )
    output = {
        **regression,
        **mean_metric_dict(ranking_rows),
        "dee": float(dee.raw_dee.item()),
        "dee_huber": float(dee.loss.item()),
        "users": int(len(users)),
    }
    if counterfactual_group0:
        group0_gap = float(torch.cat(counterfactual_group0).mean().item())
        group1_gap = float(torch.cat(counterfactual_group1).mean().item())
        output.update(
            {
                "cf_group0_male_minus_female": group0_gap,
                "cf_group1_female_minus_male": group1_gap,
                "cf_interaction_gap": group0_gap + group1_gap,
                "cf_stress_hinge": max(0.0, stress_margin - group0_gap)
                + max(0.0, stress_margin - group1_gap),
            }
        )
    return output


def _ranking_dee(
    recommended_items: Sequence[np.ndarray],
    genders: Sequence[int],
    paper_membership: np.ndarray,
) -> float:
    """Compute the paper's top-K DEE extension over the two genre groups."""
    valid_items = np.flatnonzero(paper_membership.sum(axis=1) == 1)
    labels = paper_membership[valid_items].argmax(axis=1)
    item_to_valid = {int(item): position for position, item in enumerate(valid_items)}
    indicators = np.zeros((len(recommended_items), len(valid_items)), dtype=np.float32)
    for user_row, items in enumerate(recommended_items):
        for item in items:
            position = item_to_valid.get(int(item))
            if position is not None:
                indicators[user_row, position] = 1.0
    global_rate = float(indicators.mean())
    result = 0.0
    gender_array = np.asarray(genders)
    for gender in (0, 1):
        user_mask = gender_array == gender
        for item_group in (0, 1):
            item_mask = labels == item_group
            if user_mask.any() and item_mask.any():
                result += abs(global_rate - float(indicators[user_mask][:, item_mask].mean()))
    return result


@torch.inference_mode()
def evaluate_model(
    model: ColdStartSetEncoder,
    data: MovieLens1MData,
    *,
    split: str,
    support_sizes: Sequence[int],
    top_k: int,
    batch_size: int,
    max_users: int | None,
    threshold: float,
    bandwidth: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate full timestamp queries and full-catalog recommendations."""
    model.eval()
    users = select_split_users(data, split, max_users, seed)
    paper_membership = paper_genre_membership(data.movie_genres, GENRES)
    fair_items = np.flatnonzero(paper_membership.sum(axis=1) == 1)
    fair_items_tensor = torch.as_tensor(fair_items, dtype=torch.long, device=device)
    fair_labels = paper_membership[fair_items].argmax(axis=1).astype(np.int64)
    all_catalog = torch.arange(data.num_movies, dtype=torch.long, device=device)
    results: dict[str, Any] = {
        "protocol": {
            "split": split,
            "user_disjoint": True,
            "support": "earliest timestamp-ordered ratings",
            "query": "all later rated movies",
            "ranking_candidates": "the held-out rated query set",
            "recommendation_candidates": "full catalog excluding support movies",
            "positive_rating_threshold": 4.0,
            "top_k": top_k,
        },
        "num_users": int(len(users)),
        "support_sizes": {},
    }
    rng = np.random.default_rng(seed)

    for support_size in support_sizes:
        prediction_parts: list[Tensor] = []
        target_parts: list[Tensor] = []
        by_gender_predictions: dict[int, list[Tensor]] = defaultdict(list)
        by_gender_targets: dict[int, list[Tensor]] = defaultdict(list)
        ranking_rows: list[dict[str, float]] = []
        ranking_rows_by_gender: dict[int, list[dict[str, float]]] = defaultdict(list)
        recommendations: list[np.ndarray] = []
        recommendation_genders: list[int] = []
        dee_scores: list[Tensor] = []
        dee_genders: list[Tensor] = []
        dee_items: list[Tensor] = []

        for batch_users in user_batches(users, batch_size, rng, shuffle=False):
            batch = make_episode_batch(
                data,
                batch_users,
                int(support_size),
                query_size=None,
                rng=rng,
                device=device,
            )
            user = model.encode_user(
                batch.support_items,
                batch.support_ratings,
                batch.support_mask,
                batch.genders,
            )
            query_predictions = model.decode(user, batch.query_items)
            prediction_parts.append(query_predictions[batch.query_mask].float().cpu())
            target_parts.append(batch.query_ratings[batch.query_mask].float().cpu())

            catalog_items = all_catalog.unsqueeze(0).expand(len(batch_users), -1)
            catalog_scores = model.decode(user, catalog_items).float()
            for row in range(len(batch_users)):
                query_mask = batch.query_mask[row]
                row_predictions = query_predictions[row, query_mask].float().cpu()
                row_targets = batch.query_ratings[row, query_mask].float().cpu()
                gender = int(batch.genders[row].item())
                by_gender_predictions[gender].append(row_predictions)
                by_gender_targets[gender].append(row_targets)
                ranking = ranking_metrics_for_user(
                    row_predictions.numpy(),
                    row_targets.numpy(),
                    k=top_k,
                )
                ranking_rows.append(ranking)
                ranking_rows_by_gender[gender].append(ranking)

                available_scores = catalog_scores[row].clone()
                rated_support = batch.support_items[row, batch.support_mask[row]]
                available_scores[rated_support] = -torch.inf
                recommended = torch.topk(
                    available_scores, k=min(top_k, data.num_movies - len(rated_support))
                ).indices.cpu().numpy()
                recommendations.append(recommended)
                recommendation_genders.append(gender)

            fair_score_batch = catalog_scores[:, fair_items_tensor]
            dee_scores.append(fair_score_batch.reshape(-1).cpu())
            dee_genders.append(
                batch.genders.unsqueeze(1)
                .expand(-1, len(fair_items))
                .reshape(-1)
                .cpu()
            )
            dee_items.append(
                torch.as_tensor(fair_labels)
                .unsqueeze(0)
                .expand(len(batch_users), -1)
                .reshape(-1)
            )

        regression = regression_metrics(
            torch.cat(prediction_parts), torch.cat(target_parts)
        )
        dee = compute_dee(
            torch.cat(dee_scores),
            torch.cat(dee_genders),
            torch.cat(dee_items),
            threshold=threshold,
            bandwidth=bandwidth,
            huber_delta=0.01,
            expected_user_groups=[0, 1],
            expected_item_groups=[0, 1],
        )
        gender_names = {0: "male", 1: "female", 2: "unknown"}
        gender_metrics: dict[str, Any] = {}
        for gender, parts in by_gender_predictions.items():
            gender_metrics[gender_names.get(gender, str(gender))] = {
                **regression_metrics(
                    torch.cat(parts), torch.cat(by_gender_targets[gender])
                ),
                **mean_metric_dict(ranking_rows_by_gender[gender]),
                "users": len(ranking_rows_by_gender[gender]),
            }
        cell_probabilities = {
            f"{gender_names.get(int(user), str(int(user)))}|group_{int(item)}": float(prob)
            for user, item, prob in zip(
                dee.cell_user_groups.tolist(),
                dee.cell_item_groups.tolist(),
                dee.conditional_probabilities.tolist(),
                strict=True,
            )
        }
        results["support_sizes"][str(support_size)] = {
            **regression,
            **mean_metric_dict(ranking_rows),
            "dee": float(dee.raw_dee.item()),
            "dee_huber": float(dee.loss.item()),
            "preference_probability": float(dee.global_probability.item()),
            "preference_probability_by_gender_item_group": cell_probabilities,
            "dee_ranking": _ranking_dee(
                recommendations, recommendation_genders, paper_membership
            ),
            "metrics_by_gender": gender_metrics,
            "genre_exposure_by_gender": exposure_by_gender(
                recommendations,
                recommendation_genders,
                data.movie_genres,
                GENRES,
                top_k,
            ),
        }
    return results
