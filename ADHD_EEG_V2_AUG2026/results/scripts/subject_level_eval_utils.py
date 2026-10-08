"""Utilities for subject-level evaluation from epoch-level predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metric_name: str
    metric_value: float
    sweep: pd.DataFrame


def filter_by_age_group(df: pd.DataFrame, age_group_filter: str) -> pd.DataFrame:
    """Return the requested age-group slice."""
    if age_group_filter in ("all", "", None):
        return df.copy()

    filtered = df[df["age_group"] == age_group_filter].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for age_group={age_group_filter!r}.")
    return filtered


def aggregate_subject_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Aggregate epoch probabilities into one probability per subject."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    subject_ids = np.asarray(subject_ids)

    if not (len(y_true) == len(y_prob) == len(subject_ids)):
        raise ValueError("y_true, y_prob, and subject_ids must have the same length.")

    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true,
            "y_prob": y_prob,
        }
    )
    grouped = frame.groupby("subject_id", sort=False)

    if aggregation == "mean":
        prob_series = grouped["y_prob"].mean()
    elif aggregation == "median":
        prob_series = grouped["y_prob"].median()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    label_lists = grouped["y_true"].agg(list)
    inconsistent = label_lists[label_lists.map(lambda values: any(v != values[0] for v in values[1:]))]
    if not inconsistent.empty:
        raise ValueError(f"Inconsistent subject labels detected: {inconsistent.index.tolist()}")

    subject_df = pd.DataFrame(
        {
            "subject_id": prob_series.index.to_numpy(),
            "y_true": label_lists.map(lambda values: values[0]).to_numpy(),
            "y_prob": prob_series.to_numpy(),
            "n_epochs": grouped.size().to_numpy(),
        }
    )
    return subject_df


def subject_probability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    dataset: np.ndarray | str | None = None,
    primary_aggregation: str = "mean",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return exactly one row per subject with mean and median epoch probabilities.

    Mean probability is the prespecified primary aggregation. Median probability is
    retained for a subject-level sensitivity analysis; neither method averages
    metrics across epochs.
    """
    if primary_aggregation not in {"mean", "median"}:
        raise ValueError("primary_aggregation must be 'mean' or 'median'.")
    frame = pd.DataFrame({"subject_id": np.asarray(subject_ids), "true_label": np.asarray(y_true), "epoch_probability": np.asarray(y_prob, dtype=float)})
    if dataset is None:
        frame["dataset"] = "unknown"
    elif isinstance(dataset, str):
        frame["dataset"] = dataset
    else:
        dataset_values = np.asarray(dataset)
        if len(dataset_values) != len(frame):
            raise ValueError("dataset must be a scalar or have one value per epoch.")
        frame["dataset"] = dataset_values

    grouped = frame.groupby("subject_id", sort=False)
    label_nunique = grouped["true_label"].nunique()
    dataset_nunique = grouped["dataset"].nunique()
    if (label_nunique != 1).any() or (dataset_nunique != 1).any():
        raise ValueError("Each subject must have exactly one true label and dataset value.")
    result = grouped.agg(
        dataset=("dataset", "first"),
        true_label=("true_label", "first"),
        number_of_epochs=("epoch_probability", "size"),
        mean_probability=("epoch_probability", "mean"),
        median_probability=("epoch_probability", "median"),
    ).reset_index()
    primary_probability = f"{primary_aggregation}_probability"
    result["predicted_label"] = (result[primary_probability] >= threshold).astype(int)
    result["primary_aggregation"] = primary_aggregation
    result["primary_threshold"] = threshold
    return result


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | str]:
    """Compute binary classification metrics from probabilities and a threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Accuracy": round(acc, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "Specificity": round(specificity, 4),
        "Sensitivity": round(rec, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(roc, 4) if not np.isnan(roc) else np.nan,
        "PR-AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Confusion_Matrix": f"[[{tn} {fp}],[{fn} {tp}]]",
    }


def evaluate_subject_aggregations(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    dataset: np.ndarray | str | None,
    threshold: float,
    primary_aggregation: str = "mean",
) -> tuple[dict[str, float | str], dict[str, float | str], pd.DataFrame]:
    """Evaluate one observation per subject using primary and sensitivity scores."""
    table = subject_probability_table(
        y_true, y_prob, subject_ids, dataset=dataset,
        primary_aggregation=primary_aggregation, threshold=threshold,
    )
    primary_column = f"{primary_aggregation}_probability"
    sensitivity_aggregation = "median" if primary_aggregation == "mean" else "mean"
    sensitivity_column = f"{sensitivity_aggregation}_probability"
    primary_metrics = compute_binary_metrics(table["true_label"], table[primary_column], threshold)
    sensitivity_metrics = compute_binary_metrics(table["true_label"], table[sensitivity_column], threshold)
    return primary_metrics, sensitivity_metrics, table


def evaluate_epoch_and_subject_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    threshold: float,
    aggregation: str = "mean",
) -> tuple[dict[str, float | str], dict[str, float | str], pd.DataFrame]:
    """Evaluate predictions at epoch and subject levels using identical thresholding."""
    epoch_metrics = compute_binary_metrics(y_true, y_prob, threshold)
    subject_df = aggregate_subject_probabilities(y_true, y_prob, subject_ids, aggregation)
    subject_metrics = compute_binary_metrics(subject_df["y_true"].to_numpy(), subject_df["y_prob"].to_numpy(), threshold)
    return epoch_metrics, subject_metrics, subject_df


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "f1",
) -> ThresholdSelection:
    """Select a decision threshold from validation probabilities."""
    metric_name = metric_name.lower()
    metric_column = {
        "balanced_accuracy": "Balanced_Accuracy",
        "youden_j": "Youden_J",
    }.get(metric_name)
    if metric_column is None:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")

    unique_probs = np.unique(np.asarray(y_prob, dtype=float))
    if unique_probs.size == 1:
        candidates = np.array([0.5], dtype=float)
    else:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
        candidates = np.concatenate(([0.0], midpoints, [1.0]))

    rows = []
    best_row = None
    best_sort_key = None

    for threshold in candidates:
        metrics = compute_binary_metrics(y_true, y_prob, float(threshold))
        row = {
            "Threshold": round(float(threshold), 6),
            **metrics,
        }
        row["Youden_J"] = row["Sensitivity"] + row["Specificity"] - 1.0
        rows.append(row)

        sort_key = (
            row[metric_column],
            row["Balanced_Accuracy"],
            row["Youden_J"],
            row["Accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_row = row

    sweep = pd.DataFrame(rows).sort_values(by=["Threshold"], kind="stable").reset_index(drop=True)
    return ThresholdSelection(
        threshold=float(best_row["Threshold"]),
        metric_name=metric_name,
        metric_value=float(best_row[metric_column]),
        sweep=sweep,
    )
