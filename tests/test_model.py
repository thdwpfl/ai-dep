from __future__ import annotations

import torch

from coldstart_rec.checkpoint import load_model_checkpoint, save_checkpoint
from coldstart_rec.model import ColdStartSetEncoder, ModelConfig


def make_model() -> ColdStartSetEncoder:
    genres = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float32
    )
    years = torch.tensor([-1.0, -0.2, 0.5, 1.0])
    return ColdStartSetEncoder(
        genres,
        years,
        ModelConfig(
            item_embedding_dim=8,
            hidden_dim=12,
            context_dim=7,
            gender_embedding_dim=3,
            dropout=0.0,
            gender_dropout=0.0,
        ),
    )


def test_forward_shapes_range_and_gradients() -> None:
    model = make_model()
    support_items = torch.tensor([[0, 1], [2, 0]])
    support_ratings = torch.tensor([[5.0, 2.0], [4.0, 0.0]])
    support_mask = torch.tensor([[True, True], [True, False]])
    genders = torch.tensor([0, 1])
    query = torch.tensor([[2, 3], [1, 3]])
    predictions = model(
        support_items, support_ratings, support_mask, genders, query
    )
    assert predictions.shape == (2, 2)
    assert torch.all((predictions >= 1.0) & (predictions <= 5.0))
    predictions.sum().backward()
    assert model.item_id_embedding.weight.grad is not None
    assert torch.isfinite(model.item_id_embedding.weight.grad).all()


def test_zero_support_and_unknown_gender() -> None:
    model = make_model().eval()
    prediction = model(
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0)),
        torch.empty((1, 0), dtype=torch.bool),
        torch.tensor([ColdStartSetEncoder.UNKNOWN]),
        torch.tensor([[0, 1, 2, 3]]),
    )
    assert prediction.shape == (1, 4)
    assert torch.isfinite(prediction).all()


def test_support_order_invariance() -> None:
    model = make_model().eval()
    items = torch.tensor([[0, 1, 2]])
    ratings = torch.tensor([[5.0, 2.0, 4.0]])
    mask = torch.ones_like(items, dtype=torch.bool)
    gender = torch.tensor([0])
    query = torch.tensor([[3]])
    first = model(items, ratings, mask, gender, query)
    permutation = torch.tensor([2, 0, 1])
    second = model(
        items[:, permutation], ratings[:, permutation], mask, gender, query
    )
    torch.testing.assert_close(first, second)


def test_explicit_gender_genre_head_is_directional_neutral_and_portable(
    tmp_path,
) -> None:
    genres = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float32
    )
    model = ColdStartSetEncoder(
        genres,
        torch.tensor([-1.0, -0.2, 0.5, 1.0]),
        ModelConfig(
            item_embedding_dim=8,
            hidden_dim=12,
            context_dim=7,
            gender_embedding_dim=3,
            dropout=0.0,
            gender_dropout=0.0,
            architecture="explicit_gender_genre",
            gender_genre_scale=1.0,
        ),
    ).eval()
    with torch.no_grad():
        # Group-0-like genre favors male; group-1-like genre favors female.
        model.gender_genre_effect.copy_(torch.tensor([-1.0, 1.0, 0.0]))
    support_items = torch.tensor([[2], [2], [2]])
    support_ratings = torch.tensor([[4.0], [4.0], [4.0]])
    support_mask = torch.ones_like(support_items, dtype=torch.bool)
    genders = torch.tensor([model.MALE, model.FEMALE, model.UNKNOWN])
    query = torch.tensor([[0, 1, 2]]).expand(3, -1)
    scores = model(
        support_items, support_ratings, support_mask, genders, query
    )
    assert scores[0, 0] > scores[1, 0]
    assert scores[1, 1] > scores[0, 1]
    torch.testing.assert_close(scores[0, 2], scores[1, 2])
    assert scores[1, 0] < scores[2, 0] < scores[0, 0]
    assert scores[0, 1] < scores[2, 1] < scores[1, 1]

    checkpoint = tmp_path / "explicit.pt"
    save_checkpoint(checkpoint, model, {"model": model.config.to_dict()}, {}, 1, "x")
    restored, payload = load_model_checkpoint(checkpoint)
    assert payload["model_config"]["architecture"] == "explicit_gender_genre"
    torch.testing.assert_close(
        restored(support_items, support_ratings, support_mask, genders, query),
        scores,
    )

    restored.train()
    restored.zero_grad(set_to_none=True)
    restored(support_items, support_ratings, support_mask, genders, query).sum().backward()
    assert restored.gender_genre_effect.grad is not None
    assert restored.gender_genre_effect.grad.abs().sum() > 0
