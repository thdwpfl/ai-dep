"""TaNP-inspired deterministic conditional set encoder for unseen users.

The model has learned item embeddings because the catalog is shared between
training and inference, but it deliberately has no learned user-ID table. A new
user state is produced only from gender and an arbitrary-size support set of
(movie, rating) pairs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions for :class:`ColdStartSetEncoder`."""

    item_embedding_dim: int = 64
    hidden_dim: int = 128
    context_dim: int = 96
    gender_embedding_dim: int = 8
    dropout: float = 0.1
    gender_dropout: float = 0.1
    architecture: str = "implicit"
    gender_genre_scale: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in keys if key in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_activation:
        layers.extend([nn.GELU(), nn.LayerNorm(output_dim)])
    return nn.Sequential(*layers)


class ColdStartSetEncoder(nn.Module):
    """Amortized few-shot recommender with permutation-invariant conditioning."""

    MALE = 0
    FEMALE = 1
    UNKNOWN = 2

    def __init__(
        self,
        movie_genres: Tensor,
        movie_years: Tensor,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        if movie_genres.ndim != 2:
            raise ValueError("movie_genres must have shape [num_movies, num_genres]")
        if movie_years.ndim not in (1, 2):
            raise ValueError("movie_years must have shape [num_movies] or [num_movies, 1]")
        if movie_years.shape[0] != movie_genres.shape[0]:
            raise ValueError("movie feature row counts differ")

        self.config = cfg
        if cfg.architecture not in {"implicit", "explicit_gender_genre"}:
            raise ValueError(
                "architecture must be 'implicit' or 'explicit_gender_genre'"
            )
        self.num_movies = int(movie_genres.shape[0])
        self.num_genres = int(movie_genres.shape[1])
        dim = cfg.item_embedding_dim
        self.register_buffer("movie_genres", movie_genres.float().clone())
        self.register_buffer("movie_years", movie_years.float().reshape(-1, 1).clone())

        self.item_id_embedding = nn.Embedding(self.num_movies, dim)
        self.genre_projection = nn.Linear(self.num_genres, dim, bias=False)
        self.year_projection = nn.Linear(1, dim)
        self.item_encoder = _mlp(dim * 3, cfg.hidden_dim, dim, cfg.dropout, True)
        self.item_bias = nn.Embedding(self.num_movies, 1)

        support_input = dim * 2 + 1
        self.support_encoder = _mlp(
            support_input, cfg.hidden_dim, cfg.context_dim, cfg.dropout, True
        )
        self.null_context = nn.Parameter(torch.zeros(cfg.context_dim))
        if cfg.architecture == "implicit":
            self.gender_embedding = nn.Embedding(3, cfg.gender_embedding_dim)
            user_input_dim = cfg.context_dim + cfg.gender_embedding_dim + 1
        else:
            # The wide head is the only direct gender path. Its centered sign
            # makes the genre effect identifiable and unknown exactly neutral.
            self.gender_genre_effect = nn.Parameter(torch.zeros(self.num_genres))
            user_input_dim = cfg.context_dim + 1
        self.user_encoder = _mlp(
            user_input_dim, cfg.hidden_dim, dim, cfg.dropout, True
        )
        self.decoder = _mlp(dim * 3, cfg.hidden_dim, 1, cfg.dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_id_embedding.weight, std=0.03)
        nn.init.zeros_(self.item_bias.weight)
        if self.config.architecture == "implicit":
            nn.init.normal_(self.gender_embedding.weight, std=0.03)
        nn.init.normal_(self.null_context, std=0.03)

    def encode_items(self, item_indices: Tensor) -> Tensor:
        """Encode any tensor of internal movie indices."""
        ids = item_indices.long()
        id_features = self.item_id_embedding(ids)
        genres = self.genre_projection(self.movie_genres[ids])
        years = self.year_projection(self.movie_years[ids])
        return self.item_encoder(torch.cat((id_features, genres, years), dim=-1))

    def encode_user(
        self,
        support_items: Tensor,
        support_ratings: Tensor,
        support_mask: Tensor,
        genders: Tensor,
    ) -> Tensor:
        """Compute a user representation without a user identifier."""
        if support_items.ndim != 2:
            raise ValueError("support_items must be [batch, support]")
        if support_ratings.shape != support_items.shape:
            raise ValueError("support rating shape differs from item shape")
        if support_mask.shape != support_items.shape:
            raise ValueError("support mask shape differs from item shape")

        pooled, counts = self._encode_support(
            support_items, support_ratings, support_mask
        )
        gender_ids = genders.long().clamp(min=self.MALE, max=self.UNKNOWN)
        if self.training and self.config.gender_dropout > 0:
            unknown = torch.rand_like(gender_ids.float()) < self.config.gender_dropout
            gender_ids = torch.where(
                unknown, torch.full_like(gender_ids, self.UNKNOWN), gender_ids
            )
        count_feature = torch.log1p(counts) / torch.log(
            torch.tensor(11.0, device=counts.device, dtype=counts.dtype)
        )
        if self.config.architecture == "implicit":
            gender = self.gender_embedding(gender_ids)
            return self.user_encoder(torch.cat((pooled, gender, count_feature), dim=-1))

        taste = self.user_encoder(torch.cat((pooled, count_feature), dim=-1))
        gender_sign = self._gender_sign(gender_ids, dtype=taste.dtype).unsqueeze(-1)
        return torch.cat((taste, gender_sign), dim=-1)

    def _encode_support(
        self,
        support_items: Tensor,
        support_ratings: Tensor,
        support_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode the permutation-invariant support set without demographics."""
        if support_items.ndim != 2:
            raise ValueError("support_items must be [batch, support]")
        if support_ratings.shape != support_items.shape:
            raise ValueError("support rating shape differs from item shape")
        if support_mask.shape != support_items.shape:
            raise ValueError("support mask shape differs from item shape")

        batch_size, support_size = support_items.shape
        mask = support_mask.to(dtype=self.movie_genres.dtype)
        if support_size:
            item = self.encode_items(support_items)
            normalized_rating = ((support_ratings.float() - 3.0) / 2.0).unsqueeze(-1)
            pair = torch.cat((item, normalized_rating, item * normalized_rating), dim=-1)
            encoded = self.support_encoder(pair)
            counts = mask.sum(dim=1, keepdim=True)
            pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1.0)
            pooled = torch.where(
                counts > 0,
                pooled,
                self.null_context.unsqueeze(0).expand(batch_size, -1),
            )
        else:
            counts = torch.zeros((batch_size, 1), device=support_items.device)
            pooled = self.null_context.unsqueeze(0).expand(batch_size, -1)
        return pooled, counts

    def _gender_sign(self, genders: Tensor, *, dtype: torch.dtype) -> Tensor:
        """Return centered M/F signs and exact zero for unknown gender."""
        signs = torch.zeros_like(genders, dtype=dtype)
        signs = torch.where(genders == self.MALE, signs.new_tensor(-0.5), signs)
        return torch.where(genders == self.FEMALE, signs.new_tensor(0.5), signs)

    def with_gender(self, user: Tensor, genders: Tensor) -> Tensor:
        """Counterfactually replace gender on an explicit-architecture user state."""
        if self.config.architecture != "explicit_gender_genre":
            raise RuntimeError("with_gender requires explicit_gender_genre architecture")
        if user.ndim != 2 or genders.reshape(-1).numel() != user.shape[0]:
            raise ValueError("genders must have one entry per user state")
        signs = self._gender_sign(
            genders.reshape(-1).long().clamp(min=self.MALE, max=self.UNKNOWN),
            dtype=user.dtype,
        ).unsqueeze(-1)
        return torch.cat((user[:, : self.config.item_embedding_dim], signs), dim=-1)

    def decode(self, user: Tensor, query_items: Tensor) -> Tensor:
        """Decode ratings for `[batch, candidates]` internal movie indices."""
        if user.ndim != 2 or query_items.ndim != 2:
            raise ValueError("user and query_items must be rank 2")
        item = self.encode_items(query_items)
        if self.config.architecture == "explicit_gender_genre":
            taste = user[:, : self.config.item_embedding_dim]
        else:
            taste = user
        expanded_user = taste.unsqueeze(1).expand_as(item)
        raw = self.decoder(
            torch.cat((expanded_user, item, expanded_user * item), dim=-1)
        ).squeeze(-1)
        raw = raw + self.item_bias(query_items.long()).squeeze(-1)
        if self.config.architecture == "explicit_gender_genre":
            genres = self.movie_genres[query_items.long()]
            genre_count = genres.sum(dim=-1).clamp_min(1.0)
            genre_effect = (genres * self.gender_genre_effect).sum(dim=-1) / genre_count
            gender_sign = user[:, self.config.item_embedding_dim].unsqueeze(-1)
            raw = raw + (
                float(self.config.gender_genre_scale) * gender_sign * genre_effect
            )
        return 3.0 + 2.0 * torch.tanh(raw / 2.0)

    def forward(
        self,
        support_items: Tensor,
        support_ratings: Tensor,
        support_mask: Tensor,
        genders: Tensor,
        query_items: Tensor,
    ) -> Tensor:
        user = self.encode_user(
            support_items, support_ratings, support_mask, genders
        )
        return self.decode(user, query_items)
