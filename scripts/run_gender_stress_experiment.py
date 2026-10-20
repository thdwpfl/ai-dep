#!/usr/bin/env python3
"""Run a bounded natural/stress/DEE experiment without using test for selection."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from coldstart_rec.config import apply_overrides, load_config, save_config  # noqa: E402
from coldstart_rec.training import train_model  # noqa: E402
from coldstart_rec.utils import utc_now, write_json  # noqa: E402


def run_trial(
    name: str,
    config: dict[str, Any],
    output: Path,
    initial_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run one trial and retain explicit failure evidence."""
    trial_directory = output / name
    try:
        summary = train_model(
            config, trial_directory, initial_checkpoint=initial_checkpoint
        )
        return {
            "name": name,
            "status": "completed",
            "config": config,
            "summary": summary,
        }
    except Exception as exc:
        failure = {
            "name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config": config,
        }
        trial_directory.mkdir(parents=True, exist_ok=True)
        write_json(failure, trial_directory / "failure.json")
        return failure


def copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def validation(trial: dict[str, Any]) -> dict[str, float]:
    return trial["summary"]["best_validation"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-config", type=Path, default=Path("configs/explicit_natural.yaml"))
    parser.add_argument("--stress-config", type=Path, default=Path("configs/explicit_stress.yaml"))
    parser.add_argument("--fair-config", type=Path, default=Path("configs/explicit_fair.yaml"))
    parser.add_argument("--search-config", type=Path, default=Path("configs/gender_stress_search.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/gender_stress"))
    args = parser.parse_args()

    natural_config = load_config(args.natural_config)
    stress_base = load_config(args.stress_config)
    fair_base = load_config(args.fair_config)
    search = load_config(args.search_config)
    output = args.output_root
    trials: list[dict[str, Any]] = []

    natural = run_trial("natural", natural_config, output, None)
    trials.append(natural)
    if natural["status"] != "completed":
        raise RuntimeError("Natural explicit baseline failed")
    natural_checkpoint = Path(natural["summary"]["best_checkpoint"])
    natural_metrics = validation(natural)

    stress_trials: list[dict[str, Any]] = []
    for index, specification in enumerate(search["stress_trials"]):
        config = apply_overrides(
            stress_base,
            {
                "train.stress_margin": specification["stress_margin"],
                "train.stress_coefficient": specification["stress_coefficient"],
            },
        )
        config["seed"] = int(stress_base["seed"]) + index
        trial = run_trial(
            f"stress/{specification['name']}", config, output, natural_checkpoint
        )
        trials.append(trial)
        if trial["status"] == "completed":
            stress_trials.append(trial)
    if not stress_trials:
        raise RuntimeError("All controlled-bias stress trials failed")

    rules = search["selection"]

    def stress_qualified(trial: dict[str, Any]) -> bool:
        metrics = validation(trial)
        return (
            float(metrics["rmse"])
            <= float(natural_metrics["rmse"])
            * (1.0 + float(rules["stress_max_rmse_increase_fraction"]))
            and float(metrics["ndcg@10"])
            >= float(natural_metrics["ndcg@10"])
            * float(rules["stress_min_ndcg_retention"])
            and float(metrics["cf_interaction_gap"])
            >= float(rules["stress_min_interaction_gap"])
        )

    qualified_stress = [trial for trial in stress_trials if stress_qualified(trial)]
    stress_pool = qualified_stress or stress_trials
    selected_stress = max(
        stress_pool,
        key=lambda trial: (
            float(validation(trial)["cf_interaction_gap"]),
            -float(validation(trial)["rmse"]),
        ),
    )
    stress_checkpoint = Path(selected_stress["summary"]["best_checkpoint"])
    stress_metrics = validation(selected_stress)

    control_config = copy.deepcopy(fair_base)
    control_config["seed"] = int(fair_base["seed"]) + 500
    control_config["train"]["dee_coefficient"] = 0.0
    control_config["train"]["stress_coefficient"] = 0.0
    control = run_trial("mse_recovery_control", control_config, output, stress_checkpoint)
    trials.append(control)

    fair_trials: list[dict[str, Any]] = []
    for index, specification in enumerate(search["fair_trials"]):
        config = apply_overrides(
            fair_base,
            {
                "train.dee_coefficient": specification["dee_coefficient"],
                "train.dee_bandwidth": specification["dee_bandwidth"],
                "train.stress_coefficient": 0.0,
            },
        )
        config["seed"] = int(fair_base["seed"]) + index
        trial = run_trial(
            f"fair/{specification['name']}", config, output, stress_checkpoint
        )
        trials.append(trial)
        if trial["status"] == "completed":
            fair_trials.append(trial)
    if not fair_trials:
        raise RuntimeError("All DEE mitigation trials failed")

    def fair_qualified(trial: dict[str, Any]) -> bool:
        metrics = validation(trial)
        return (
            float(metrics["rmse"])
            <= float(stress_metrics["rmse"])
            * (1.0 + float(rules["fair_max_rmse_increase_fraction"]))
            and float(metrics["ndcg@10"])
            >= float(stress_metrics["ndcg@10"])
            * float(rules["fair_min_ndcg_retention"])
            and float(metrics["dee"])
            <= float(stress_metrics["dee"])
            * (1.0 - float(rules["fair_min_dee_improvement_fraction"]))
            and abs(float(metrics["cf_interaction_gap"]))
            <= abs(float(stress_metrics["cf_interaction_gap"]))
            * (1.0 - float(rules["fair_min_interaction_improvement_fraction"]))
        )

    qualified_fair = [trial for trial in fair_trials if fair_qualified(trial)]
    fair_pool = qualified_fair or fair_trials
    selected_fair = min(
        fair_pool,
        key=lambda trial: (
            float(validation(trial)["dee"]) / max(float(stress_metrics["dee"]), 1e-8)
            + abs(float(validation(trial)["cf_interaction_gap"]))
            / max(abs(float(stress_metrics["cf_interaction_gap"])), 1e-8),
            float(validation(trial)["rmse"]),
        ),
    )

    checkpoint_root = Path("artifacts/checkpoints")
    selected = {
        "explicit_natural_best.pt": natural,
        "stress_baseline_best.pt": selected_stress,
        "stress_fair_best.pt": selected_fair,
    }
    if control["status"] == "completed":
        selected["stress_control_best.pt"] = control
    for filename, trial in selected.items():
        copy_checkpoint(Path(trial["summary"]["best_checkpoint"]), checkpoint_root / filename)

    save_config(natural["config"], Path("configs/selected_explicit_natural.yaml"))
    save_config(selected_stress["config"], Path("configs/selected_stress_baseline.yaml"))
    save_config(selected_fair["config"], Path("configs/selected_stress_fair.yaml"))
    if control["status"] == "completed":
        save_config(control["config"], Path("configs/selected_stress_control.yaml"))

    summary = {
        "timestamp": utc_now(),
        "bounded_search": True,
        "selection_split": "validation only",
        "scenario": "controlled gender-by-genre bias stress test",
        "natural": natural,
        "selected_stress": selected_stress,
        "stress_met_constraints": bool(qualified_stress),
        "mse_recovery_control": control,
        "selected_fair": selected_fair,
        "fair_met_constraints": bool(qualified_fair),
        "trial_count": len(trials),
        "failed_trials": [trial for trial in trials if trial["status"] == "failed"],
    }
    write_json(summary, output / "tuning_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
