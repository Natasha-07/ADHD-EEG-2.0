"""Leakage-safe CNN benchmark for ADHD classification."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_model_utils import (
    build_experiment_split,
    build_dual_level_results,
    fit_channel_standardizer,
    load_epoch_dataset,
    set_global_seed,
    split_epoch_dataset,
    transform_epochs,
    write_split_manifest,
)


def build_cnn(input_shape):
    """CNN optimized for raw EEG epoch classification."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--epoch_seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--scenario", choices=["combined_holdout", "child_to_adult", "adult_to_child"], default="combined_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="Results/CNN")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_epoch_dataset(args.csv, epoch_seconds=args.epoch_seconds, overlap=args.overlap)
    split = build_experiment_split(
        dataset,
        scenario=args.scenario,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    write_split_manifest(
        outdir,
        dataset,
        split,
        seed=args.seed,
        epoch_seconds=args.epoch_seconds,
        overlap=args.overlap,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_epoch_dataset(dataset, split)
    print(f"Scenario: {split.scenario}")
    print(f"Train: {len(X_train)} epochs from {len(split.train_subjects)} subjects")
    print(f"Validation: {len(X_val)} epochs from {len(split.val_subjects)} subjects")
    print(f"Test: {len(X_test)} epochs from {len(split.test_subjects)} subjects")

    scaler = fit_channel_standardizer(X_train)
    X_train = transform_epochs(X_train, scaler)
    X_val = transform_epochs(X_val, scaler)
    X_test = transform_epochs(X_test, scaler)

    tf.keras.backend.clear_session()
    model = build_cnn(X_train.shape[1:])
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.5,
        min_lr=1e-6,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=2,
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    epoch_results, subject_results, sensitivity_results, subject_predictions = build_dual_level_results(
        "CNN", y_test, y_prob, dataset.subject_ids[split.test_idx], split,
        seed=args.seed,
        epoch_seconds=args.epoch_seconds,
        overlap=args.overlap,
        val_size=args.val_size,
        test_size=args.test_size,
        dataset=dataset.age_groups[split.test_idx],
    )

    results = subject_results
    pd.DataFrame([subject_results, sensitivity_results, epoch_results]).to_csv(outdir / "cnn_metrics.csv", index=False)
    pd.DataFrame({"subject_id": dataset.subject_ids[split.test_idx], "y_true": y_test, "y_prob": y_prob}).to_csv(outdir / "epoch_predictions.csv", index=False)
    subject_predictions.to_csv(outdir / "subject_level_predictions.csv", index=False)
    subject_predictions.to_csv(outdir / "subject_predictions.csv", index=False)
    model.save(outdir / "cnn_model.keras")

    print("\n" + "=" * 60)
    print(f"CNN RESULTS ({split.scenario})")
    print("=" * 60)
    print(f"Subject balanced accuracy: {results['Balanced_Accuracy']:.4f}")
    print(f"Precision:  {results['Precision']:.4f}")
    print(f"Sensitivity:{results['Sensitivity']:.4f}; Specificity: {results['Specificity']:.4f}")
    print(f"F1-score:   {results['F1-score']:.4f}")
    print(f"ROC-AUC:    {results['ROC-AUC']:.4f}")
    print(f"Confusion Matrix: {results['Confusion_Matrix']}")
    print("=" * 60)
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
