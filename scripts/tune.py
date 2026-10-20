#!/usr/bin/env python3
"""Run a bounded structured search for baseline and DEE models."""

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


def _run_trial(
    kind: str,
    name: str,
    config: dict[str, Any],
    tuning_root: Path,
    initial_checkpoint: Path | None,
) -> dict[str, Any]:
    trial_dir = tuning_root / kind / name
    try:
        summary = train_model(
            config, trial_dir, initial_checkpoint=initial_checkpoint
        )
        return {
            "kind": kind,
            "name": name,
            "status": "completed",
            "config": config,
            "summary": summary,
        }
    except Exception as exc:
        failure = {
            "kind": kind,
            "name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config": config,
        }
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(failure, trial_dir / "failure.json")
        return failure


def _copy_selected(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--fair-config", type=Path, default=Path("configs/fair.yaml"))
    parser.add_argument("--search-config", type=Path, default=Path("configs/search.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    baseline_base = load_config(args.baseline_config)
    fair_base = load_config(args.fair_config)
    search = load_config(args.search_config)
    output_root = args.output_root
    tuning_root = output_root / "tuning"
    trials: list[dict[str, Any]] = []

    for index, specification in enumerate(search["baseline_trials"]):
        config = apply_overrides(baseline_base, specification["overrides"])
        config["seed"] = int(baseline_base["seed"]) + index
        trials.append(
            _run_trial(
                "baseline",
                specification["name"],
                config,
                tuning_root,
                initial_checkpoint=None,
            )
        )
    completed_baselines = [
        trial for trial in trials if trial["kind"] == "baseline" and trial["status"] == "completed"
    ]
    if not completed_baselines:
        raise RuntimeError("All baseline tuning trials failed")

    # A single bounded fallback is automatically added when every initial
    # baseline has clearly poor five-shot validation RMSE.
    best_baseline = min(
        completed_baselines,
        key=lambda trial: trial["summary"]["best_validation"]["rmse"],
    )
    if float(best_baseline["summary"]["best_validation"]["rmse"]) > 1.25:
        fallback = apply_overrides(
            baseline_base,
            {
                "train.learning_rate": 0.0005,
                "train.weight_decay": 0.000001,
                "model.hidden_dim": 192,
                "train.epochs": min(int(baseline_base["train"]["epochs"]) + 4, 12),
            },
        )
        fallback["seed"] = int(baseline_base["seed"]) + 100
        trial = _run_trial(
            "baseline", "automatic_poor_rmse_retry", fallback, tuning_root, None
        )
        trials.append(trial)
        if trial["status"] == "completed":
            completed_baselines.append(trial)
            best_baseline = min(
                completed_baselines,
                key=lambda candidate: candidate["summary"]["best_validation"]["rmse"],
            )

    baseline_checkpoint = Path(best_baseline["summary"]["best_checkpoint"])
    best_model_overrides = {
        f"model.{key}": value for key, value in best_baseline["config"]["model"].items()
    }
    for index, specification in enumerate(search["fair_trials"]):
        overrides = {
            **best_model_overrides,
            "train.dee_coefficient": specification["dee_coefficient"],
            "train.dee_bandwidth": specification["dee_bandwidth"],
            "train.learning_rate": min(
                float(best_baseline["config"]["train"]["learning_rate"]), 0.001
            ),
        }
        config = apply_overrides(fair_base, overrides)
        config["seed"] = int(fair_base["seed"]) + 1000 + index
        trials.append(
            _run_trial(
                "fair",
                specification["name"],
                config,
                tuning_root,
                initial_checkpoint=baseline_checkpoint,
            )
        )
    completed_fair = [
        trial for trial in trials if trial["kind"] == "fair" and trial["status"] == "completed"
    ]
    if not completed_fair:
        raise RuntimeError("All DEE tuning trials failed")

    base_metrics = best_baseline["summary"]["best_validation"]
    requirements = search["selection"]

    def is_qualified(trial: dict[str, Any]) -> bool:
        metrics = trial["summary"]["best_validation"]
        ndcg_key = "ndcg@10"
        return (
            float(metrics["rmse"])
            <= float(base_metrics["rmse"])
            * (1.0 + float(requirements["max_rmse_increase_fraction"]))
            and float(metrics[ndcg_key])
            >= float(base_metrics[ndcg_key])
            * float(requirements["min_ndcg_retention"])
            and float(metrics["dee"])
            <= float(base_metrics["dee"])
            * (1.0 - float(requirements["minimum_dee_improvement_fraction"]))
        )

    qualified = [trial for trial in completed_fair if is_qualified(trial)]
    # One final, bounded stronger retry if the initial grid did not materially
    # improve fairness under the requested accuracy-retention constraints.
    if not qualified:
        strongest = min(
            completed_fair,
            key=lambda trial: trial["summary"]["best_validation"]["dee"],
        )
        if float(strongest["summary"]["best_validation"]["dee"]) >= float(base_metrics["dee"]):
            retry_config = copy.deepcopy(strongest["config"])
            retry_config["seed"] = int(fair_base["seed"]) + 2000
            retry_config["train"]["dee_coefficient"] = min(
                float(strongest["config"]["train"]["dee_coefficient"]) * 2.5, 250.0
            )
            retry_config["train"]["learning_rate"] = 0.0005
            retry = _run_trial(
                "fair",
                "automatic_weak_fairness_retry",
                retry_config,
                tuning_root,
                initial_checkpoint=baseline_checkpoint,
            )
            trials.append(retry)
            if retry["status"] == "completed":
                completed_fair.append(retry)
                if is_qualified(retry):
                    qualified.append(retry)

    if qualified:
        best_fair = min(
            qualified,
            key=lambda trial: (
                trial["summary"]["best_validation"]["dee"],
                trial["summary"]["best_validation"]["rmse"],
            ),
        )
        selection_reason = "lowest validation DEE among accuracy-retaining trials"
    else:
        # Honest bounded fallback: choose the smallest relative fairness/accuracy
        # score and record that the requested target was not reached.
        base_dee = max(float(base_metrics["dee"]), 1e-8)
        base_rmse = max(float(base_metrics["rmse"]), 1e-8)
        best_fair = min(
            completed_fair,
            key=lambda trial: (
                trial["summary"]["best_validation"]["dee"] / base_dee
                + trial["summary"]["best_validation"]["rmse"] / base_rmse
            ),
        )
        selection_reason = "bounded Pareto fallback; no trial met every retention target"

    checkpoint_root = output_root / "checkpoints"
    selected_baseline_path = checkpoint_root / "baseline_best.pt"
    selected_fair_path = checkpoint_root / "fair_best.pt"
    _copy_selected(Path(best_baseline["summary"]["best_checkpoint"]), selected_baseline_path)
    _copy_selected(Path(best_fair["summary"]["best_checkpoint"]), selected_fair_path)
    save_config(best_baseline["config"], Path("configs/selected_baseline.yaml"))
    save_config(best_fair["config"], Path("configs/selected_fair.yaml"))

    summary = {
        "timestamp": utc_now(),
        "bounded_search": True,
        "trial_count": len(trials),
        "trials": trials,
        "selected_baseline": {
            "trial": best_baseline["name"],
            "checkpoint": str(selected_baseline_path.resolve()),
            "validation": best_baseline["summary"]["best_validation"],
        },
        "selected_fair": {
            "trial": best_fair["name"],
            "checkpoint": str(selected_fair_path.resolve()),
            "validation": best_fair["summary"]["best_validation"],
            "selection_reason": selection_reason,
            "met_accuracy_fairness_constraints": bool(qualified),
        },
        "failed_trials": [
            {
                "kind": trial["kind"],
                "name": trial["name"],
                "error_type": trial.get("error_type"),
                "error": trial.get("error"),
            }
            for trial in trials
            if trial["status"] == "failed"
        ],
    }
    write_json(summary, output_root / "tuning_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

