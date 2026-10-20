"""Training loop for episodic cold-start recommendation and DEE."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import load_model_checkpoint, save_checkpoint
from .config import save_config
from .data import GENRES, MovieLens1MData, load_processed_data
from .episodes import make_episode_batch, user_batches
from .evaluation import quick_validate, select_split_users
from .fairness import compute_dee
from .groups import balanced_group_item_sample, paper_genre_membership
from .model import ColdStartSetEncoder, ModelConfig
from .utils import (
    append_jsonl,
    parameter_count,
    resolve_device,
    seed_everything,
    utc_now,
    write_json,
)


def build_model(data: MovieLens1MData, model_config: Mapping[str, Any]) -> ColdStartSetEncoder:
    """Construct a set encoder from processed catalog features."""
    return ColdStartSetEncoder(
        movie_genres=torch.from_numpy(data.movie_genres.astype(np.float32)),
        movie_years=torch.from_numpy(data.movie_years_normalized.astype(np.float32)),
        config=ModelConfig.from_dict(dict(model_config)),
    )


def _training_users(
    data: MovieLens1MData,
    split: str,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    return select_split_users(data, split, maximum, seed)


def train_model(
    config: Mapping[str, Any],
    output_directory: str | Path,
    *,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one configured trial, saving its best checkpoint and all logs."""
    cfg = dict(config)
    seed = int(cfg["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    train_cfg = dict(cfg["train"])
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    data = load_processed_data(cfg["data"]["processed_path"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "config.yaml")
    log_path = output / "training.jsonl"
    if log_path.exists():
        log_path.unlink()

    if initial_checkpoint is None:
        model = build_model(data, cfg["model"])
        initialization = "random"
    else:
        model, source_payload = load_model_checkpoint(initial_checkpoint, device="cpu")
        if model.num_movies != data.num_movies:
            raise ValueError("Initial checkpoint and processed catalog sizes differ")
        initialization = str(Path(initial_checkpoint))
        source_fingerprint = source_payload.get("dataset_fingerprint")
        current_fingerprint = data.manifest["artifact"]["sha256"]
        if source_fingerprint != current_fingerprint:
            raise ValueError("Initial checkpoint was trained on a different data artifact")
        requested_model_config = ModelConfig.from_dict(dict(cfg["model"]))
        if model.config != requested_model_config:
            raise ValueError(
                "Initial checkpoint model configuration differs from requested config"
            )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    query_size = int(train_cfg["query_size"])
    support_choices = np.asarray(train_cfg["support_sizes"], dtype=np.int64)
    dee_coefficient = float(train_cfg.get("dee_coefficient", 0.0))
    threshold = float(train_cfg.get("preference_threshold", 3.0))
    bandwidth = float(train_cfg.get("dee_bandwidth", 0.5))
    validation_bandwidth = float(
        cfg.get("evaluation", {}).get("dee_bandwidth", 0.5)
    )
    huber_delta = float(train_cfg.get("dee_huber_delta", 0.01))
    fairness_candidate_size = int(train_cfg.get("fairness_candidate_size", 96))
    stress_coefficient = float(train_cfg.get("stress_coefficient", 0.0))
    stress_margin = float(train_cfg.get("stress_margin", 0.0))
    if stress_coefficient < 0 or stress_margin < 0:
        raise ValueError("stress_coefficient and stress_margin must be non-negative")
    if stress_coefficient > 0 and model.config.architecture != "explicit_gender_genre":
        raise ValueError(
            "Controlled gender stress requires explicit_gender_genre architecture"
        )
    rng = np.random.default_rng(seed)

    train_users = _training_users(
        data, "train", train_cfg.get("max_train_users"), seed
    )
    val_users = _training_users(
        data, "val", train_cfg.get("max_val_users"), seed + 1
    )
    paper_membership = paper_genre_membership(data.movie_genres, GENRES)
    dataset_fingerprint = str(data.manifest["artifact"]["sha256"])
    checkpoint_path = output / "best.pt"
    best_objective = math.inf
    best_epoch = -1
    best_validation: dict[str, float] = {}
    started = time.perf_counter()

    append_jsonl(
        {
            "event": "start",
            "timestamp": utc_now(),
            "seed": seed,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "visible_gpu_count": torch.cuda.device_count(),
            "parameters": parameter_count(model),
            "train_users": len(train_users),
            "val_users": len(val_users),
            "initialization": initialization,
        },
        log_path,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mse = 0.0
        epoch_dee_loss = 0.0
        epoch_dee_raw = 0.0
        epoch_stress_loss = 0.0
        epoch_stress_group0_gap = 0.0
        epoch_stress_group1_gap = 0.0
        epoch_examples = 0
        fairness_batches = 0
        stress_batches = 0
        batches = user_batches(train_users, batch_size, rng, shuffle=True)
        for batch_users in batches:
            supports = rng.choice(support_choices, size=len(batch_users), replace=True)
            batch = make_episode_batch(
                data,
                batch_users,
                supports,
                query_size=query_size,
                rng=rng,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            user = model.encode_user(
                batch.support_items,
                batch.support_ratings,
                batch.support_mask,
                batch.genders,
            )
            predictions = model.decode(user, batch.query_items)
            errors = predictions[batch.query_mask] - batch.query_ratings[batch.query_mask]
            mse = errors.square().mean()
            dee_loss = predictions.sum() * 0.0
            dee_raw = predictions.sum() * 0.0
            stress_loss = predictions.sum() * 0.0
            stress_group0_gap = predictions.sum() * 0.0
            stress_group1_gap = predictions.sum() * 0.0

            # DEE and the explicitly labeled stress objective both use a balanced
            # dense user/catalog cross-product rather than sparse observed ratings.
            if dee_coefficient > 0 or stress_coefficient > 0:
                candidates = balanced_group_item_sample(
                    paper_membership, fairness_candidate_size, rng
                )
                candidate_items = (
                    torch.as_tensor(candidates, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(len(batch_users), -1)
                )
                item_labels = torch.as_tensor(
                    paper_membership[candidates].argmax(axis=1),
                    dtype=torch.long,
                    device=device,
                )

                if stress_coefficient > 0:
                    male_user = model.with_gender(
                        user, torch.full_like(batch.genders, model.MALE)
                    )
                    female_user = model.with_gender(
                        user, torch.full_like(batch.genders, model.FEMALE)
                    )
                    male_predictions = model.decode(male_user, candidate_items)
                    female_predictions = model.decode(female_user, candidate_items)
                    group0 = item_labels == 0
                    group1 = item_labels == 1
                    stress_group0_gap = (
                        male_predictions[:, group0] - female_predictions[:, group0]
                    ).mean()
                    stress_group1_gap = (
                        female_predictions[:, group1] - male_predictions[:, group1]
                    ).mean()
                    stress_loss = F.relu(stress_margin - stress_group0_gap) + F.relu(
                        stress_margin - stress_group1_gap
                    )
                    stress_batches += 1

                # A sparse last batch without both observed genders is a safe
                # fallback for the four-cell DEE probability definition.
                if dee_coefficient > 0 and torch.unique(batch.genders).numel() >= 2:
                    fairness_predictions = model.decode(user, candidate_items)
                    user_groups = (
                        batch.genders.unsqueeze(1)
                        .expand_as(fairness_predictions)
                        .reshape(-1)
                    )
                    item_groups = (
                        item_labels.unsqueeze(0)
                        .expand(len(batch_users), -1)
                        .reshape(-1)
                    )
                    result = compute_dee(
                        fairness_predictions.reshape(-1).float(),
                        user_groups,
                        item_groups,
                        threshold=threshold,
                        bandwidth=bandwidth,
                        huber_delta=huber_delta,
                        expected_user_groups=[0, 1],
                        expected_item_groups=[0, 1],
                    )
                    if len(result.deviations) == 4:
                        dee_loss = result.loss
                        dee_raw = result.raw_dee
                        fairness_batches += 1

            loss = (
                mse
                + dee_coefficient * dee_loss
                + stress_coefficient * stress_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            examples = int(batch.query_mask.sum().item())
            epoch_examples += examples
            epoch_mse += float(mse.detach().item()) * examples
            epoch_dee_loss += float(dee_loss.detach().item())
            epoch_dee_raw += float(dee_raw.detach().item())
            epoch_stress_loss += float(stress_loss.detach().item())
            epoch_stress_group0_gap += float(stress_group0_gap.detach().item())
            epoch_stress_group1_gap += float(stress_group1_gap.detach().item())

        validation = quick_validate(
            model,
            data,
            users=val_users,
            support_size=5,
            query_size=min(query_size, 64),
            batch_size=batch_size,
            threshold=threshold,
            bandwidth=validation_bandwidth,
            huber_delta=huber_delta,
            fairness_candidate_size=max(fairness_candidate_size, 128),
            stress_margin=stress_margin,
            seed=seed + 10_000 + epoch,
            device=device,
        )
        validation_objective = float(validation["rmse"]) + (
            dee_coefficient * float(validation["dee_huber"])
        ) + stress_coefficient * float(validation.get("cf_stress_hinge", 0.0))
        record = {
            "event": "epoch",
            "timestamp": utc_now(),
            "epoch": epoch,
            "train_mse": epoch_mse / max(epoch_examples, 1),
            "train_rmse": math.sqrt(epoch_mse / max(epoch_examples, 1)),
            "train_dee_huber": epoch_dee_loss / max(fairness_batches, 1),
            "train_dee": epoch_dee_raw / max(fairness_batches, 1),
            "fairness_batches": fairness_batches,
            "train_stress_loss": epoch_stress_loss / max(stress_batches, 1),
            "train_stress_group0_gap": epoch_stress_group0_gap
            / max(stress_batches, 1),
            "train_stress_group1_gap": epoch_stress_group1_gap
            / max(stress_batches, 1),
            "stress_batches": stress_batches,
            "validation_objective": validation_objective,
            "validation": validation,
        }
        append_jsonl(record, log_path)
        if validation_objective < best_objective:
            best_objective = validation_objective
            best_epoch = epoch
            best_validation = validation
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                validation,
                epoch,
                dataset_fingerprint,
            )

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "status": "completed",
        "best_checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "best_validation": best_validation,
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "visible_gpu_count": torch.cuda.device_count(),
        "parameters": parameter_count(model),
        "train_users": int(len(train_users)),
        "val_users": int(len(val_users)),
        "initialization": initialization,
        "dee_coefficient": dee_coefficient,
        "stress_coefficient": stress_coefficient,
        "stress_margin": stress_margin,
        "dee_bandwidth": bandwidth,
        "validation_dee_bandwidth": validation_bandwidth,
    }
    write_json(summary, output / "summary.json")
    return summary
