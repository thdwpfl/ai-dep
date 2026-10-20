"""Differentiable equal-experience (DEE) fairness for recommenders.

This module implements Definition 3.2 and Equations (12), (14), (15),
(20), and (21) of Cho, Choi, and Suh, *Equal Experience in Recommender
Systems* (arXiv:2210.05936).

For a predicted score ``s`` and preference threshold ``tau``, the paper
defines ``Y_tilde = 1{s >= tau}``.  With a Gaussian KDE of bandwidth ``h``,
its plug-in preference probability is the Gaussian survival probability

    F_k((tau - s) / h) = Phi((s - tau) / h).

DEE is the *sum* (not the average) over joint user/item groups of the
absolute difference between the global and conditional preference
probabilities.  For optimization, the absolute value is replaced with the
paper's standard Huber loss.  The raw absolute DEE remains available for
evaluation.

Movie genres are naturally multi-label.  A multi-hot ``item_groups`` matrix
is therefore expanded into one observation per active genre: a movie with
three genres contributes its score to three gender--genre cells and three
times to the global expanded population.  This is the direct multi-label
analogue of the mutually exclusive item groups in the paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


GroupValues = Tensor | Sequence[int] | Sequence[float]


@dataclass(frozen=True)
class DEEResult:
    """Full result of a DEE computation.

    Attributes:
        loss: Differentiable Huber surrogate used during training (Eq. 21).
        raw_dee: Unsmoothed, absolute plug-in DEE used for reporting (Eq. 12).
        global_probability: KDE estimate of ``P(Y_tilde=1)``.
        conditional_probabilities: KDE estimates for non-empty joint cells.
        deviations: ``global_probability - conditional_probability`` per cell.
        cell_user_groups: User-group label corresponding to each cell.
        cell_item_groups: Item-group label/index corresponding to each cell.
        cell_counts: Expanded observation count in each non-empty cell.
    """

    loss: Tensor
    raw_dee: Tensor
    global_probability: Tensor
    conditional_probabilities: Tensor
    deviations: Tensor
    cell_user_groups: Tensor
    cell_item_groups: Tensor
    cell_counts: Tensor

    @property
    def metric(self) -> Tensor:
        """Alias for the raw, unsmoothed DEE."""

        return self.raw_dee


def gaussian_kde_survival(
    scores: Tensor,
    *,
    threshold: float = 3.0,
    bandwidth: float = 0.2,
) -> Tensor:
    """Return each score's Gaussian-KDE preference contribution.

    This is ``F_k((tau - score) / h)`` from Equation (15), where ``F_k`` is
    the survival function of a standard Gaussian kernel.  The result has the
    same shape as ``scores`` and stays differentiable with respect to it.
    """

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not math.isfinite(float(bandwidth)) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and strictly positive")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if not scores.is_floating_point():
        scores = scores.to(dtype=torch.get_default_dtype())

    # 0.5 * erfc((tau-s)/(h*sqrt(2))) == Phi((s-tau)/h).
    z = (float(threshold) - scores) / (float(bandwidth) * math.sqrt(2.0))
    return 0.5 * torch.erfc(z)


def _expected_or_observed(
    expected: GroupValues | None,
    observed: Tensor,
    *,
    name: str,
) -> Tensor:
    if expected is None:
        return torch.unique(observed, sorted=True)
    if isinstance(expected, Tensor):
        groups = expected
    else:
        try:
            groups = torch.as_tensor(list(expected))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric categorical labels") from exc
    if groups.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    groups = groups.to(device=observed.device, dtype=observed.dtype)
    # Sorting and de-duplicating makes audit output independent of input order.
    return torch.unique(groups, sorted=True)


def _expand_item_memberships(
    scores: Tensor,
    user_groups: Tensor,
    item_groups: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Expand a single-label vector or multi-hot membership matrix."""

    if item_groups.ndim == 1:
        if item_groups.numel() != scores.numel():
            raise ValueError("item_groups must have one entry per score")
        return scores, user_groups, item_groups.reshape(-1)

    if item_groups.ndim != 2:
        raise ValueError("item_groups must be [N] labels or an [N, G] multi-hot matrix")
    if item_groups.shape[0] != scores.numel():
        raise ValueError("item_groups must have one row per score")

    # Every positive entry is an active membership. Values are not used as
    # weights; this intentionally implements categorical multi-hot expansion.
    membership = (item_groups > 0).nonzero(as_tuple=False)
    if membership.numel() == 0:
        empty_scores = scores[:0]
        empty_users = user_groups[:0]
        empty_items = torch.empty(0, dtype=torch.long, device=scores.device)
        return empty_scores, empty_users, empty_items
    rows, genre_indices = membership.unbind(dim=1)
    return scores[rows], user_groups[rows], genre_indices


def _huber_absolute(values: Tensor, delta: float) -> Tensor:
    """Standard Huber loss stated immediately after Equation (21)."""

    absolute = values.abs()
    delta_tensor = values.new_tensor(delta)
    return torch.where(
        absolute <= delta_tensor,
        0.5 * values.square(),
        delta_tensor * (absolute - 0.5 * delta_tensor),
    )


def compute_dee(
    scores: Tensor,
    user_groups: Tensor,
    item_groups: Tensor,
    *,
    threshold: float = 3.0,
    bandwidth: float = 0.2,
    huber_delta: float = 0.01,
    expected_user_groups: GroupValues | None = None,
    expected_item_groups: GroupValues | None = None,
) -> DEEResult:
    """Compute differentiable and raw DEE from a batch of predictions.

    ``user_groups`` contains numeric categorical labels (for example, 0/1
    gender labels). ``item_groups`` is either one label per score or an
    ``[N, G]`` multi-hot genre matrix.  For the latter, item-group labels are
    column indices.

    Explicit expected groups make the set and ordering of audited cells
    deterministic across batches.  As required by the probability definition,
    intersections with no observations are skipped rather than assigned an
    invented probability.
    """

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not isinstance(user_groups, Tensor) or not isinstance(item_groups, Tensor):
        raise TypeError("user_groups and item_groups must be torch.Tensor objects")
    if not math.isfinite(float(huber_delta)) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and strictly positive")

    flat_scores = scores.reshape(-1)
    flat_users = user_groups.reshape(-1).to(device=flat_scores.device)
    if flat_users.numel() != flat_scores.numel():
        raise ValueError("user_groups must have one entry per score")
    item_groups = item_groups.to(device=flat_scores.device)

    expanded_scores, expanded_users, expanded_items = _expand_item_memberships(
        flat_scores, flat_users, item_groups
    )

    if item_groups.ndim == 2:
        default_items = torch.arange(
            item_groups.shape[1], device=flat_scores.device, dtype=torch.long
        )
    else:
        default_items = expanded_items
    users_to_audit = _expected_or_observed(
        expected_user_groups, expanded_users, name="expected_user_groups"
    )
    items_to_audit = _expected_or_observed(
        expected_item_groups,
        default_items,
        name="expected_item_groups",
    )

    if expanded_scores.numel() == 0:
        # A differentiable zero is safer for a sparse mini-batch than NaN.
        zero = flat_scores.sum() * 0.0
        empty_float = flat_scores.new_empty((0,))
        empty_users = flat_users.new_empty((0,))
        empty_items = expanded_items.new_empty((0,))
        empty_counts = torch.empty(0, dtype=torch.long, device=flat_scores.device)
        return DEEResult(
            loss=zero,
            raw_dee=zero,
            global_probability=zero,
            conditional_probabilities=empty_float,
            deviations=empty_float,
            cell_user_groups=empty_users,
            cell_item_groups=empty_items,
            cell_counts=empty_counts,
        )

    preference = gaussian_kde_survival(
        expanded_scores, threshold=threshold, bandwidth=bandwidth
    )
    global_probability = preference.mean()

    conditional: list[Tensor] = []
    deviations: list[Tensor] = []
    cell_users: list[Tensor] = []
    cell_items: list[Tensor] = []
    counts: list[Tensor] = []
    for user_group in users_to_audit:
        for item_group in items_to_audit:
            mask = (expanded_users == user_group) & (expanded_items == item_group)
            count = mask.sum()
            if not bool(count):
                continue
            cell_probability = preference[mask].mean()
            conditional.append(cell_probability)
            deviations.append(global_probability - cell_probability)
            cell_users.append(user_group)
            cell_items.append(item_group)
            counts.append(count)

    if not deviations:
        zero = flat_scores.sum() * 0.0
        empty_float = flat_scores.new_empty((0,))
        return DEEResult(
            loss=zero,
            raw_dee=zero,
            global_probability=global_probability,
            conditional_probabilities=empty_float,
            deviations=empty_float,
            cell_user_groups=flat_users.new_empty((0,)),
            cell_item_groups=expanded_items.new_empty((0,)),
            cell_counts=torch.empty(0, dtype=torch.long, device=flat_scores.device),
        )

    conditional_tensor = torch.stack(conditional)
    deviation_tensor = torch.stack(deviations)
    raw_dee = deviation_tensor.abs().sum()
    loss = _huber_absolute(deviation_tensor, float(huber_delta)).sum()
    return DEEResult(
        loss=loss,
        raw_dee=raw_dee,
        global_probability=global_probability,
        conditional_probabilities=conditional_tensor,
        deviations=deviation_tensor,
        cell_user_groups=torch.stack(cell_users),
        cell_item_groups=torch.stack(cell_items),
        cell_counts=torch.stack(counts).to(dtype=torch.long),
    )


def dee_loss(
    scores: Tensor,
    user_groups: Tensor,
    item_groups: Tensor,
    **kwargs: object,
) -> Tensor:
    """Convenience wrapper returning only the differentiable Huber DEE."""

    return compute_dee(scores, user_groups, item_groups, **kwargs).loss


def dee_metric(
    scores: Tensor,
    user_groups: Tensor,
    item_groups: Tensor,
    **kwargs: object,
) -> Tensor:
    """Convenience wrapper returning the raw absolute KDE plug-in DEE."""

    return compute_dee(scores, user_groups, item_groups, **kwargs).raw_dee


class DEERegularizer(nn.Module):
    """Reusable ``nn.Module`` wrapper for the differentiable DEE loss."""

    def __init__(
        self,
        *,
        threshold: float = 3.0,
        bandwidth: float = 0.2,
        huber_delta: float = 0.01,
        expected_user_groups: GroupValues | None = None,
        expected_item_groups: GroupValues | None = None,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.bandwidth = bandwidth
        self.huber_delta = huber_delta
        self.expected_user_groups = expected_user_groups
        self.expected_item_groups = expected_item_groups

    def details(
        self,
        scores: Tensor,
        user_groups: Tensor,
        item_groups: Tensor,
    ) -> DEEResult:
        """Return training loss, raw metric, and per-cell audit details."""

        return compute_dee(
            scores,
            user_groups,
            item_groups,
            threshold=self.threshold,
            bandwidth=self.bandwidth,
            huber_delta=self.huber_delta,
            expected_user_groups=self.expected_user_groups,
            expected_item_groups=self.expected_item_groups,
        )

    def forward(
        self,
        scores: Tensor,
        user_groups: Tensor,
        item_groups: Tensor,
    ) -> Tensor:
        return self.details(scores, user_groups, item_groups).loss


__all__ = [
    "DEEResult",
    "DEERegularizer",
    "compute_dee",
    "dee_loss",
    "dee_metric",
    "gaussian_kde_survival",
]
