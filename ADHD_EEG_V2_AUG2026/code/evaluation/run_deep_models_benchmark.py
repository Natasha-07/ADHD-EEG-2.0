"""Run the clean deep-model benchmark across combined and cross-cohort scenarios."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from deep_model_utils import SCENARIO_DETAILS


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "Results"
RESULT_TABLES_DIR = RESULTS_DIR / "Result Tables"
SUBJECT_LEVEL_DIR = RESULTS_DIR / "reviewer_revision" / "subject_level"
BENCHMARK_DIR = SUBJECT_LEVEL_DIR / "deep_models"

MODELS = [
    {
        "name": "CNN",
        "script": ROOT / "Results" / "CNN" / "CNN_EpochLevelSplit.py",
        "metrics_file": "cnn_metrics.csv",
    },
    {
        "name": "LSTM",
        "script": ROOT / "Results" / "LSTM" / "LSTM_EpochLevelSplit.py",
        "metrics_file": "lstm_metrics.csv",
    },
    {
        "name": "Bi-LSTM",
        "script": ROOT / "Results" / "Bi-LSTM" / "BiLSTM_EpochLevelSplit.py",
        "metrics_file": "bilstm_metrics.csv",
    },
    {
        "name": "Temporal Transformer Baseline",
        "script": ROOT / "Results" / "EEGFormer" / "EEGFormer_EpochLevelSplit.py",
        "metrics_file": "temporal_transformer_metrics.csv",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--epoch_seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--scenario", choices=["all", *SCENARIO_DETAILS.keys()], default="all")
    parser.add_argument("--collect_only", action="store_true")
    return parser.parse_args()


def selected_scenarios(scenario_arg: str) -> list[str]:
    if scenario_arg == "all":
        return ["combined_holdout", "child_to_adult", "adult_to_child"]
    return [scenario_arg]


def run_model(config: dict[str, str], args, scenario: str) -> None:
    output_dir = BENCHMARK_DIR / scenario / config["name"]
    command = [
        sys.executable,
        str(config["script"]),
        "--csv",
        args.csv,
        "--epoch_seconds",
        str(args.epoch_seconds),
        "--overlap",
        str(args.overlap),
        "--test_size",
        str(args.test_size),
        "--val_size",
        str(args.val_size),
        "--scenario",
        scenario,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--out",
        str(output_dir),
    ]
    print(f"\n{'=' * 90}")
    print(f"Running {config['name']} for scenario={scenario}")
    print(f"{'=' * 90}")
    subprocess.run(command, cwd=ROOT, check=True)


def load_metrics(config: dict[str, str], scenario: str) -> pd.Series:
    metrics_path = BENCHMARK_DIR / scenario / config["name"] / config["metrics_file"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    subject_rows = df[df["Evaluation_Level"].isin(["Subject", "Subject_Default_0.5"])]
    if len(subject_rows) != 1:
        raise ValueError(f"Expected one subject-level metrics row in {metrics_path}")
    row = subject_rows.iloc[0].copy()
    expected_protocol = SCENARIO_DETAILS[scenario]["protocol"]
    if row.get("Split_Type") != "Subject-Wise":
        raise ValueError(f"{config['name']} did not report a subject-wise split.")
    if row.get("Protocol") != expected_protocol:
        raise ValueError(f"{config['name']} did not report protocol {expected_protocol}.")
    if row.get("Scenario") != scenario:
        raise ValueError(f"{config['name']} did not report scenario {scenario}.")
    return row


def save_protocol_manifest(args, scenarios: list[str]) -> Path:
    RESULT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "benchmark": "deep_models_clean_benchmark_v1",
        "csv": args.csv,
        "epoch_seconds": args.epoch_seconds,
        "overlap": args.overlap,
        "test_size_combined_holdout": args.test_size,
        "val_size_training_pool": args.val_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "models": [config["name"] for config in MODELS],
        "scenarios": {scenario: SCENARIO_DETAILS[scenario] for scenario in scenarios},
        "preprocessing": {
            "epoching": "2-second windows with 50% overlap and child recordings resampled to 256 Hz equivalent length",
            "normalization": "StandardScaler fit on training epochs only, then applied unchanged to validation and test epochs",
            "leakage_guard": "No subject_id overlap between train, validation, and test",
        },
    }
    manifest_path = SUBJECT_LEVEL_DIR / "deep_models_subject_primary_protocol.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main():
    args = parse_args()
    scenarios = selected_scenarios(args.scenario)
    manifest_path = save_protocol_manifest(args, scenarios)

    if not args.collect_only:
        for scenario in scenarios:
            for config in MODELS:
                run_model(config, args, scenario)

    rows = []
    for scenario in scenarios:
        for config in MODELS:
            rows.append(load_metrics(config, scenario))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "Model",
            "Scenario",
            "Split_Type",
            "Protocol",
            "Evaluation_Level",
            "Train_Age_Group",
            "Test_Age_Group",
            "SMOTE",
            "Accuracy",
            "Balanced_Accuracy",
            "Specificity",
            "Sensitivity",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "TN",
            "FP",
            "FN",
            "TP",
            "Confusion_Matrix",
            "Threshold",
            "Threshold_Source",
            "Seed",
            "Epoch_Seconds",
            "Overlap",
            "Test_Size",
            "Val_Size",
            "Train_Subjects",
            "Val_Subjects",
            "Test_Subjects",
        ]
    ]
    summary_df = summary_df.sort_values(by=["Scenario", "Model"], kind="stable").reset_index(drop=True)

    SUBJECT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUBJECT_LEVEL_DIR / "neural_subject_primary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(SUBJECT_LEVEL_DIR / "neural_subject_primary_and_secondary_results.csv", index=False)

    for scenario in scenarios:
        scenario_df = summary_df[summary_df["Scenario"] == scenario].reset_index(drop=True)
        scenario_df.to_csv(SUBJECT_LEVEL_DIR / f"{scenario}_subject_primary_results.csv", index=False)

    print(f"\nSaved protocol manifest to: {manifest_path}")
    print(f"Saved clean deep benchmark summary to: {summary_path}")
    print(f"Saved subject-level primary results to: {SUBJECT_LEVEL_DIR}")


if __name__ == "__main__":
    main()
