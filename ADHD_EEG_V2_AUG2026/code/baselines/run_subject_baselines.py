"""Subject-safe EEG baselines: always-ADHD, logistic regression, and random forest.

The learned models receive the same epochs, subject-disjoint partitions, training-only
normalization, and subject-level probability aggregation as the deep benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deep_model_utils import (
    build_dual_level_results, build_experiment_split, load_epoch_dataset,
    split_epoch_dataset, write_split_manifest,
)


def spectral_summary(X: np.ndarray, fs: int = 256) -> np.ndarray:
    """Extract reproducible sensor-space summary features for classical models."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    freqs = np.fft.rfftfreq(X.shape[1], d=1 / fs)
    power = np.abs(np.fft.rfft(X, axis=1)) ** 2
    bands = ((1, 4), (4, 8), (8, 13), (13, 30))
    band_features = [power[:, (freqs >= low) & (freqs < high), :].mean(axis=1) for low, high in bands]
    return np.concatenate([mean, std, *band_features], axis=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--scenario", choices=["combined_holdout", "child_to_adult", "adult_to_child"], default="combined_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epoch_seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out", default="Results/ClassicalBaselines")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_epoch_dataset(args.csv, args.epoch_seconds, args.overlap)
    split = build_experiment_split(dataset, args.scenario, args.test_size, args.val_size, args.seed)
    write_split_manifest(outdir, dataset, split, args.seed, args.epoch_seconds, args.overlap, args.val_size, args.test_size)
    X_train, _, X_test, y_train, _, y_test = split_epoch_dataset(dataset, split)
    train_features, test_features = spectral_summary(X_train), spectral_summary(X_test)
    test_subject_ids = dataset.subject_ids[split.test_idx]

    baseline_models = {
        "Always-ADHD majority baseline": None,
        "Logistic regression (spectral summary)": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        ),
        "Random forest (spectral summary)": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=args.seed, n_jobs=-1
        ),
    }
    rows = []
    for name, estimator in baseline_models.items():
        if estimator is None:
            test_prob = np.ones(len(y_test), dtype=float)
        else:
            estimator.fit(train_features, y_train)
            test_prob = estimator.predict_proba(test_features)[:, 1]
        epoch_row, subject_row, sensitivity_row, subject_predictions = build_dual_level_results(
            name, y_test, test_prob, test_subject_ids, split, args.seed, args.epoch_seconds,
            args.overlap, args.val_size, args.test_size,
            dataset=dataset.age_groups[split.test_idx],
        )
        rows.extend((subject_row, sensitivity_row, epoch_row))
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        subject_predictions.to_csv(outdir / f"{safe_name}_subject_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(outdir / "classical_baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()
