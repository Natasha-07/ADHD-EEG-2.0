"""Configurable leakage-safe LSTM benchmark for ADHD classification."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_model_utils import load_epoch_dataset
from lstm_experiment_utils import LSTMExperimentConfig, run_lstm_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--epoch_seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--scenario", choices=["combined_holdout", "child_to_adult", "adult_to_child"], default="combined_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lstm_units_1", type=int, default=64)
    parser.add_argument("--lstm_units_2", type=int, default=32)
    parser.add_argument("--dense_units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense_dropout", type=float, default=0.5)
    parser.add_argument("--class_weight_mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--normalization_mode",
        choices=["train_channel_standard", "epoch_zscore", "epoch_zscore_plus_train_channel_standard"],
        default="train_channel_standard",
    )
    parser.add_argument("--threshold_metric", choices=["none", "youden_j", "balanced_accuracy"], default="none")
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--out", default="Results/LSTM")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--experiment_name", default="manual_run")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_epoch_dataset(args.csv, epoch_seconds=args.epoch_seconds, overlap=args.overlap)
    config = LSTMExperimentConfig(
        name=args.experiment_name,
        lstm_units_1=args.lstm_units_1,
        lstm_units_2=args.lstm_units_2,
        dense_units=args.dense_units,
        dropout=args.dropout,
        dense_dropout=args.dense_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        class_weight_mode=args.class_weight_mode,
        normalization_mode=args.normalization_mode,
        threshold_metric=args.threshold_metric,
        use_layer_norm=args.use_layer_norm,
    )
    results = run_lstm_experiment(
        dataset=dataset,
        scenario=args.scenario,
        config=config,
        seed=args.seed,
        epoch_seconds=args.epoch_seconds,
        overlap=args.overlap,
        test_size=args.test_size,
        val_size=args.val_size,
        output_dir=args.out,
        save_model=args.save_model,
    )

    outdir = Path(args.out)

    print("\n" + "=" * 60)
    print(f"LSTM RESULTS ({args.scenario})")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Subject balanced accuracy: {results['Balanced_Accuracy']:.4f}")
    print(f"Precision:  {results['Precision']:.4f}")
    print(f"Sensitivity:{results['Sensitivity']:.4f}; Specificity: {results['Specificity']:.4f}")
    print(f"F1-score:   {results['F1-score']:.4f}")
    print(f"ROC-AUC:    {results['ROC-AUC']:.4f}")
    print(f"Threshold:  {results['Threshold']:.4f} ({results['Threshold_Source']})")
    print(f"Confusion Matrix: {results['Confusion_Matrix']}")
    print("=" * 60)
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
