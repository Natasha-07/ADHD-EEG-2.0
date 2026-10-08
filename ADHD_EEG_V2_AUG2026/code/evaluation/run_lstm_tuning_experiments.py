"""Run a focused LSTM tuning sweep for combined and cross-age scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from deep_model_utils import load_epoch_dataset
from lstm_experiment_utils import LSTMExperimentConfig, run_lstm_experiment


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "Results"
RESULT_TABLES_DIR = RESULTS_DIR / "Result Tables"
TUNING_DIR = RESULTS_DIR / "LSTM_Tuning"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--epoch_seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=["all", "combined_holdout", "adult_to_child", "child_to_adult"], default="all")
    parser.add_argument("--save_models", action="store_true")
    return parser.parse_args()


def combined_configs() -> list[LSTMExperimentConfig]:
    return [
        LSTMExperimentConfig(
            name="combined_balanced_threshold_acc",
            class_weight_mode="balanced",
            normalization_mode="train_channel_standard",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_no_class_weight_threshold_acc",
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_no_class_weight_fixed_threshold",
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="none",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_small_no_class_weight",
            lstm_units_1=32,
            lstm_units_2=16,
            dense_units=16,
            dense_dropout=0.4,
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_large_no_class_weight_low_lr",
            lstm_units_1=96,
            lstm_units_2=48,
            dense_units=48,
            dropout=0.2,
            dense_dropout=0.4,
            learning_rate=5e-4,
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_epoch_zscore_no_class_weight",
            class_weight_mode="none",
            normalization_mode="epoch_zscore",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_epoch_zscore_fixed_threshold",
            class_weight_mode="none",
            normalization_mode="epoch_zscore",
            threshold_metric="none",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name="combined_epoch_zscore_layer_norm",
            learning_rate=5e-4,
            class_weight_mode="none",
            normalization_mode="epoch_zscore",
            threshold_metric="balanced_accuracy",
            use_layer_norm=True,
            epochs=8,
        ),
    ]


def cross_age_configs(prefix: str) -> list[LSTMExperimentConfig]:
    return [
        LSTMExperimentConfig(
            name=f"{prefix}_balanced_threshold_acc",
            class_weight_mode="balanced",
            normalization_mode="train_channel_standard",
            threshold_metric="balanced_accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name=f"{prefix}_no_class_weight",
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name=f"{prefix}_no_class_weight_fixed_threshold",
            class_weight_mode="none",
            normalization_mode="train_channel_standard",
            threshold_metric="none",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name=f"{prefix}_epoch_zscore_no_class_weight",
            class_weight_mode="none",
            normalization_mode="epoch_zscore",
            threshold_metric="accuracy",
            epochs=8,
        ),
        LSTMExperimentConfig(
            name=f"{prefix}_epoch_zscore_layer_norm",
            learning_rate=5e-4,
            class_weight_mode="none",
            normalization_mode="epoch_zscore",
            threshold_metric="accuracy",
            use_layer_norm=True,
            epochs=8,
        ),
    ]


def scenario_configs(scenario: str) -> list[LSTMExperimentConfig]:
    if scenario == "combined_holdout":
        return combined_configs()
    if scenario == "adult_to_child":
        return cross_age_configs("adult_to_child")
    if scenario == "child_to_adult":
        return cross_age_configs("child_to_adult")
    raise ValueError(f"Unsupported scenario: {scenario}")


def selected_scenarios(scenario_arg: str) -> list[str]:
    if scenario_arg == "all":
        return ["combined_holdout", "adult_to_child", "child_to_adult"]
    return [scenario_arg]


def main():
    args = parse_args()
    scenarios = selected_scenarios(args.scenario)
    dataset = load_epoch_dataset(args.csv, epoch_seconds=args.epoch_seconds, overlap=args.overlap)

    all_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        configs = scenario_configs(scenario)
        print(f"\n{'=' * 80}")
        print(f"Tuning scenario={scenario} with {len(configs)} configurations")
        print(f"{'=' * 80}")

        for config in configs:
            trial_dir = TUNING_DIR / scenario / config.name
            print(f"Running {config.name} ...")
            row = run_lstm_experiment(
                dataset=dataset,
                scenario=scenario,
                config=config,
                seed=args.seed,
                epoch_seconds=args.epoch_seconds,
                overlap=args.overlap,
                test_size=args.test_size,
                val_size=args.val_size,
                output_dir=trial_dir,
                save_model=args.save_models,
            )
            all_rows.append(row)

            scenario_csv = RESULT_TABLES_DIR / f"lstm_tuning_{scenario}_trials.csv"
            scenario_df = pd.DataFrame([r for r in all_rows if r["Scenario"] == scenario])
            scenario_df.to_csv(scenario_csv, index=False)
            print(
                f"  test accuracy={row['Accuracy']:.4f}, "
                f"f1={row['F1-score']:.4f}, "
                f"roc_auc={row['ROC-AUC']:.4f}"
            )

    RESULT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    trial_frames = []
    for path in sorted(RESULT_TABLES_DIR.glob("lstm_tuning_*_trials.csv")):
        if path.name == "lstm_tuning_all_trials.csv":
            continue
        trial_frames.append(pd.read_csv(path))
    if not trial_frames:
        raise ValueError("No scenario trial CSVs were found to aggregate.")

    all_df = pd.concat(trial_frames, ignore_index=True)
    all_df = all_df.sort_values(by=["Scenario", "Accuracy", "F1-score"], ascending=[True, False, False], kind="stable")
    all_trials_path = RESULT_TABLES_DIR / "lstm_tuning_all_trials.csv"
    all_df.to_csv(all_trials_path, index=False)

    best_df = (
        all_df.sort_values(by=["Scenario", "Accuracy", "F1-score"], ascending=[True, False, False], kind="stable")
        .groupby("Scenario", sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_path = RESULT_TABLES_DIR / "lstm_tuning_best_by_scenario.csv"
    best_df.to_csv(best_path, index=False)

    summary_path = RESULT_TABLES_DIR / "lstm_tuning_manifest.json"
    manifest = {
        "csv": args.csv,
        "epoch_seconds": args.epoch_seconds,
        "overlap": args.overlap,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "scenarios": scenarios,
        "combined_configs": [config.name for config in combined_configs()] if "combined_holdout" in scenarios else [],
        "cross_age_configs": [config.name for config in cross_age_configs("template")] if any(s != "combined_holdout" for s in scenarios) else [],
    }
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved all LSTM tuning trials to: {all_trials_path}")
    print(f"Saved best-by-scenario LSTM results to: {best_path}")


if __name__ == "__main__":
    main()
