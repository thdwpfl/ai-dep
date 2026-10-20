"""Focused tests for the paper-faithful DEE implementation."""

from __future__ import annotations

import math

import pytest
import torch

from coldstart_rec.fairness import (
    DEERegularizer,
    compute_dee,
    dee_metric,
    gaussian_kde_survival,
)


def test_gaussian_kde_survival_matches_threshold_definition() -> None:
    scores = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    probabilities = gaussian_kde_survival(
        scores, threshold=4.0, bandwidth=0.5
    )

    assert probabilities[1].item() == pytest.approx(0.5)
    assert probabilities[0] < 0.5 < probabilities[2]
    assert probabilities[0].item() == pytest.approx(
        1.0 - probabilities[2].item(), abs=1e-12
    )


def test_dee_is_zero_for_group_independent_score_distributions() -> None:
    # Every joint gender--genre cell has exactly the same score distribution.
    cell_scores = torch.tensor([2.0, 3.5, 4.0, 5.0], dtype=torch.float64)
    scores = cell_scores.repeat(4)
    user_groups = torch.tensor(
        [0] * 4 + [0] * 4 + [1] * 4 + [1] * 4
    )
    item_groups = torch.tensor(
        [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    )

    result = compute_dee(
        scores,
        user_groups,
        item_groups,
        threshold=4.0,
        bandwidth=0.3,
        expected_user_groups=[0, 1],
        expected_item_groups=[0, 1],
    )

    assert result.raw_dee.item() == pytest.approx(0.0, abs=1e-12)
    assert result.loss.item() == pytest.approx(0.0, abs=1e-12)
    assert result.cell_counts.tolist() == [4, 4, 4, 4]


def test_dee_is_positive_for_intentionally_biased_predictions() -> None:
    # Each joint cell is pushed to a different side of the preference threshold.
    scores = torch.tensor([5.0] * 8 + [1.0] * 8)
    user_groups = torch.tensor([0] * 8 + [1] * 8)
    item_groups = torch.tensor([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)

    result = compute_dee(
        scores,
        user_groups,
        item_groups,
        threshold=3.0,
        bandwidth=0.2,
        huber_delta=0.01,
        expected_user_groups=[0, 1],
        expected_item_groups=[0, 1],
    )

    # KDE probabilities are essentially 1 and 0, with global probability 0.5;
    # four cells therefore give a raw DEE close to 4 * 0.5 = 2.
    assert result.global_probability.item() == pytest.approx(0.5, abs=1e-6)
    assert result.raw_dee.item() > 1.9
    assert dee_metric(
        scores,
        user_groups,
        item_groups,
        threshold=3.0,
        bandwidth=0.2,
    ).item() == pytest.approx(result.raw_dee.item())


def test_huber_dee_has_finite_nonzero_gradients() -> None:
    scores = torch.tensor(
        [4.8, 4.4, 3.7, 3.4, 2.6, 2.9, 3.3, 3.6],
        dtype=torch.float64,
        requires_grad=True,
    )
    user_groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    item_groups = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    regularizer = DEERegularizer(
        threshold=3.5,
        bandwidth=0.7,
        huber_delta=0.05,
        expected_user_groups=[0, 1],
        expected_item_groups=[0, 1],
    )

    loss = regularizer(scores, user_groups, item_groups)
    loss.backward()

    assert loss.isfinite()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_multihot_genres_expand_each_active_membership() -> None:
    scores = torch.tensor([5.0, 1.0, 3.0], dtype=torch.float64)
    user_groups = torch.tensor([0, 1, 1])
    # Movie 0 belongs to genres 0 and 1; movie 1 to genre 1; movie 2 has no
    # active genre and must not enter either the global or cell distribution.
    genres = torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )

    result = compute_dee(
        scores,
        user_groups,
        genres,
        threshold=3.0,
        bandwidth=0.4,
        expected_user_groups=[0, 1, 2],
        expected_item_groups=[0, 1, 2],
    )

    # Expansion is (score 5, u0, g0), (score 5, u0, g1),
    # (score 1, u1, g1). Empty expected intersections are skipped.
    assert result.cell_user_groups.tolist() == [0, 0, 1]
    assert result.cell_item_groups.tolist() == [0, 1, 1]
    assert result.cell_counts.tolist() == [1, 1, 1]
    high = gaussian_kde_survival(
        torch.tensor(5.0, dtype=torch.float64),
        threshold=3.0,
        bandwidth=0.4,
    )
    low = gaussian_kde_survival(
        torch.tensor(1.0, dtype=torch.float64),
        threshold=3.0,
        bandwidth=0.4,
    )
    expected_global = (2.0 * high + low) / 3.0
    assert result.global_probability.item() == pytest.approx(
        expected_global.item(), abs=1e-12
    )
    expected_raw = (
        (expected_global - high).abs() * 2
        + (expected_global - low).abs()
    )
    assert result.raw_dee.item() == pytest.approx(expected_raw.item(), abs=1e-12)


def test_empty_multihot_batch_returns_differentiable_zero() -> None:
    scores = torch.tensor([2.0, 4.0], requires_grad=True)
    result = compute_dee(
        scores,
        torch.tensor([0, 1]),
        torch.zeros((2, 3), dtype=torch.long),
        expected_user_groups=[0, 1],
        expected_item_groups=[0, 1, 2],
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.raw_dee.item() == 0.0
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))


@pytest.mark.parametrize("bandwidth", [0.0, -0.1, math.inf])
def test_invalid_bandwidth_is_rejected(bandwidth: float) -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        gaussian_kde_survival(torch.tensor([3.0]), bandwidth=bandwidth)
