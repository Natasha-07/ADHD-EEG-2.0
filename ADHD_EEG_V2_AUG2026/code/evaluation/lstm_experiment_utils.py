"""Utilities for configurable, leakage-safe LSTM experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from deep_model_utils import (
    ExperimentSplit,
    EpochDataset,
    build_experiment_split,
    build_dual_level_results,
    fit_channel_standardizer,
    set_global_seed,
    split_epoch_dataset,
    transform_epochs,
    write_split_manifest,
)
from subject_level_eval_utils import aggregate_subject_probabilities, compute_binary_metrics, select_threshold


@dataclass(frozen=True)
class LSTMExperimentConfig:
    name: str
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 32
    dropout: float = 0.3
    dense_dropout: float = 0.5
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 8
    patience: int = 3
    class_weight_mode: str = "balanced"
    normalization_mode: str = "train_channel_standard"
    threshold_metric: str = "none"
    use_layer_norm: bool = False


def build_lstm(input_shape, config: LSTMExperimentConfig):
    """Build a configurable LSTM classifier."""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(config.lstm_units_1, return_sequences=True)(inputs)
    if config.use_layer_norm:
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(config.dropout)(x)

    x = layers.LSTM(config.lstm_units_2)(x)
    if config.use_layer_norm:
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(config.dropout)(x)

    x = layers.Dense(config.dense_units, activation="relu")(x)
    if config.use_layer_norm:
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(config.dense_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_lstm_experiment(
    dataset: EpochDataset,
    scenario: str,
    config: LSTMExperimentConfig,
    seed: int = 42,
    epoch_seconds: int = 2,
    overlap: float = 0.5,
    test_size: float = 0.2,
    val_size: float = 0.1,
    output_dir: str | Path | None = None,
    save_model: bool = False,
) -> dict[str, object]:
    """Train and evaluate one LSTM configuration on one scenario."""
    set_global_seed(seed)
    split = build_experiment_split(
        dataset,
        scenario=scenario,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )

    outdir = Path(output_dir) if output_dir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        write_split_manifest(
            outdir,
            dataset,
            split,
            seed=seed,
            epoch_seconds=epoch_seconds,
            overlap=overlap,
            val_size=val_size,
            test_size=test_size,
        )

    X_train, X_val, X_test, y_train, y_val, y_test = split_epoch_dataset(dataset, split)
    X_train, X_val, X_test = _normalize_epochs(X_train, X_val, X_test, config.normalization_mode)

    tf.keras.backend.clear_session()
    model = build_lstm(X_train.shape[1:], config)

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "validation_data": (X_val, y_val),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "callbacks": [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=max(1, config.patience - 1),
                factor=0.5,
                min_lr=1e-6,
                verbose=0,
            ),
        ],
        "verbose": 0,
    }

    class_weight = _class_weight_dict(y_train, config.class_weight_mode)
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(**fit_kwargs)

    val_prob = model.predict(X_val, verbose=0).ravel()
    test_prob = model.predict(X_test, verbose=0).ravel()
    val_subject = aggregate_subject_probabilities(y_val, val_prob, dataset.subject_ids[split.val_idx])
    optimized_threshold, threshold_source, threshold_metric_value = _select_threshold(
        val_subject["y_true"].to_numpy(), val_subject["y_prob"].to_numpy(), config.threshold_metric
    )
    val_metrics = compute_binary_metrics(val_subject["y_true"], val_subject["y_prob"], optimized_threshold)
    default_epoch, default_subject, default_sensitivity, default_predictions = build_dual_level_results(
        "LSTM", y_test, test_prob, dataset.subject_ids[split.test_idx], split,
        seed=seed,
        epoch_seconds=epoch_seconds,
        overlap=overlap,
        val_size=val_size,
        test_size=test_size,
        threshold=0.5,
        threshold_source="fixed_0.5_prespecified",
        dataset=dataset.age_groups[split.test_idx],
    )
    default_subject["Evaluation_Level"] = "Subject_Default_0.5"
    default_sensitivity["Evaluation_Level"] = "Subject_Default_0.5_Sensitivity"
    default_epoch["Evaluation_Level"] = "Epoch_Default_0.5"
    result_rows = [default_subject, default_sensitivity, default_epoch]
    subject_predictions = default_predictions
    if config.threshold_metric not in ("none", "", None):
        optimized_epoch, optimized_subject, optimized_sensitivity, _ = build_dual_level_results(
            "LSTM", y_test, test_prob, dataset.subject_ids[split.test_idx], split,
            seed=seed, epoch_seconds=epoch_seconds, overlap=overlap, val_size=val_size,
            test_size=test_size, threshold=optimized_threshold,
            threshold_source=threshold_source + "_validation_subject", dataset=dataset.age_groups[split.test_idx],
        )
        optimized_subject["Evaluation_Level"] = "Subject_Validation_Optimized"
        optimized_sensitivity["Evaluation_Level"] = "Subject_Validation_Optimized_Sensitivity"
        optimized_epoch["Evaluation_Level"] = "Epoch_Validation_Optimized"
        result_rows.extend([optimized_subject, optimized_sensitivity, optimized_epoch])
    results = default_subject
    results.update(
        {
            "Experiment_Name": config.name,
            "Normalization_Mode": config.normalization_mode,
            "Class_Weight_Mode": config.class_weight_mode,
            "Threshold": 0.5,
            "Threshold_Source": "fixed_0.5_prespecified",
            "Validation_Optimized_Threshold": round(float(optimized_threshold), 6),
            "Validation_Optimized_Threshold_Source": threshold_source,
            "Threshold_Metric": config.threshold_metric,
            "Threshold_Metric_Value": round(float(threshold_metric_value), 4) if not np.isnan(threshold_metric_value) else np.nan,
            "Val_Subject_Accuracy": val_metrics["Accuracy"],
            "Val_Subject_Balanced_Accuracy": val_metrics["Balanced_Accuracy"],
            "Val_Precision": val_metrics["Precision"],
            "Val_Recall": val_metrics["Recall"],
            "Val_F1-score": val_metrics["F1-score"],
            "Val_ROC-AUC": val_metrics["ROC-AUC"],
            "LSTM_Units_1": config.lstm_units_1,
            "LSTM_Units_2": config.lstm_units_2,
            "Dense_Units": config.dense_units,
            "Dropout": config.dropout,
            "Dense_Dropout": config.dense_dropout,
            "Learning_Rate": config.learning_rate,
            "Batch_Size": config.batch_size,
            "Max_Epochs": config.epochs,
            "Use_Layer_Norm": int(config.use_layer_norm),
            "Best_Val_Loss": round(float(np.min(history.history["val_loss"])), 6),
            "Best_Val_Accuracy_Raw": round(float(np.max(history.history["val_accuracy"])), 6),
        }
    )

    if outdir is not None:
        for row in result_rows:
            row.update({"Experiment_Name": config.name, "Threshold_Metric": config.threshold_metric})
        pd.DataFrame(result_rows).to_csv(outdir / "lstm_metrics.csv", index=False)
        val_subject.to_csv(outdir / "val_subject_predictions.csv", index=False)
        pd.DataFrame({"subject_id": dataset.subject_ids[split.test_idx], "y_true": y_test, "y_prob": test_prob}).to_csv(outdir / "epoch_predictions.csv", index=False)
        subject_predictions.to_csv(outdir / "subject_level_predictions.csv", index=False)
        subject_predictions.to_csv(outdir / "subject_predictions.csv", index=False)
        pd.DataFrame([asdict(config)]).to_csv(outdir / "config.csv", index=False)
        if save_model:
            model.save(outdir / "lstm_model.keras")

    return results


def _normalize_epochs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one of the supported normalization strategies."""
    if normalization_mode == "train_channel_standard":
        scaler = fit_channel_standardizer(X_train)
        return (
            transform_epochs(X_train, scaler),
            transform_epochs(X_val, scaler),
            transform_epochs(X_test, scaler),
        )

    if normalization_mode == "epoch_zscore":
        return _per_epoch_zscore(X_train), _per_epoch_zscore(X_val), _per_epoch_zscore(X_test)

    if normalization_mode == "epoch_zscore_plus_train_channel_standard":
        X_train_norm = _per_epoch_zscore(X_train)
        X_val_norm = _per_epoch_zscore(X_val)
        X_test_norm = _per_epoch_zscore(X_test)
        scaler = fit_channel_standardizer(X_train_norm)
        return (
            transform_epochs(X_train_norm, scaler),
            transform_epochs(X_val_norm, scaler),
            transform_epochs(X_test_norm, scaler),
        )

    raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")


def _per_epoch_zscore(X: np.ndarray) -> np.ndarray:
    """Normalize each epoch independently, channel by channel."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32)


def _class_weight_dict(y_train: np.ndarray, class_weight_mode: str) -> dict[int, float] | None:
    """Return class weights or None."""
    if class_weight_mode == "none":
        return None
    if class_weight_mode != "balanced":
        raise ValueError(f"Unsupported class_weight_mode: {class_weight_mode}")

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def _select_threshold(y_val: np.ndarray, val_prob: np.ndarray, threshold_metric: str):
    """Choose a validation threshold or keep the default 0.5."""
    if threshold_metric in ("none", "", None):
        return 0.5, "fixed_0.5", float("nan")

    selection = select_threshold(y_val, val_prob, metric_name=threshold_metric)
    return selection.threshold, f"validation_subject_{selection.metric_name}", selection.metric_value
