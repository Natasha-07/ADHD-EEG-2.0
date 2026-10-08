"""Shared utilities for leakage-safe deep EEG benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from subject_level_eval_utils import compute_binary_metrics, evaluate_subject_aggregations
from subject_split_utils import (
    cross_age_subject_train_val_test_indices,
    subject_metadata_table,
    subject_train_val_test_indices,
)


CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]

SCENARIO_DETAILS = {
    "combined_holdout": {
        "protocol": "deep_subject_safe_combined_v1",
        "train_age_group": "all",
        "test_age_group": "all",
    },
    "child_to_adult": {
        "protocol": "deep_subject_safe_child_to_adult_v1",
        "train_age_group": "child",
        "test_age_group": "adult",
    },
    "adult_to_child": {
        "protocol": "deep_subject_safe_adult_to_child_v1",
        "train_age_group": "adult",
        "test_age_group": "child",
    },
}


@dataclass(frozen=True)
class EpochDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    age_groups: np.ndarray


@dataclass(frozen=True)
class ExperimentSplit:
    scenario: str
    protocol: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    test_subjects: np.ndarray
    train_age_group: str
    test_age_group: str


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def epochify(x: np.ndarray, fs: int, seconds: int = 2, overlap: float = 0.5, target_len: int | None = None):
    """Create overlapping epochs and resample them to a common length if needed."""
    win = int(seconds * fs)
    step = int(win * (1 - overlap))
    total_samples = x.shape[0]
    if total_samples < win:
        return []

    epochs = []
    for start in range(0, total_samples - win + 1, step):
        epoch = x[start : start + win].astype(np.float32)
        if target_len and epoch.shape[0] != target_len:
            epoch = resample(epoch, target_len, axis=0).astype(np.float32)
        epochs.append(epoch)
    return epochs


def load_epoch_dataset(csv_path: str | Path, epoch_seconds: int = 2, overlap: float = 0.5) -> EpochDataset:
    """Load the combined CSV and convert each subject recording into epochs."""
    csv_path = Path(csv_path)
    target_len = epoch_seconds * 256

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    subjects = df["subject_id"].unique()
    print(f"Found {len(subjects)} unique subjects")

    print(f"Creating {epoch_seconds}s epochs with overlap={overlap}...")
    X_list, y_list, subject_list, age_group_list = [], [], [], []

    for i, subject_id in enumerate(subjects):
        subject_df = df[df["subject_id"] == subject_id]
        label = 1 if subject_df["label"].iloc[0] == "ADHD" else 0
        age_group = subject_df["age_group"].iloc[0]
        fs = 512 if age_group == "child" else 256
        eeg = subject_df[CHANNELS].values.astype(np.float32)

        epochs = epochify(
            eeg,
            fs=fs,
            seconds=epoch_seconds,
            overlap=overlap,
            target_len=target_len,
        )
        for epoch in epochs:
            X_list.append(epoch)
            y_list.append(label)
            subject_list.append(subject_id)
            age_group_list.append(age_group)

        if (i + 1) % 20 == 0 or i + 1 == len(subjects):
            print(f"  Processed {i + 1}/{len(subjects)} subjects")

    if not X_list:
        raise ValueError("No epochs were created from the input CSV.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    subject_ids = np.asarray(subject_list)
    age_groups = np.asarray(age_group_list)

    print(f"Total epochs: {len(X)}, shape: {X.shape}")
    print(f"Class distribution: Control={int(np.sum(y == 0))}, ADHD={int(np.sum(y == 1))}")
    return EpochDataset(X=X, y=y, subject_ids=subject_ids, age_groups=age_groups)


def build_experiment_split(
    dataset: EpochDataset,
    scenario: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> ExperimentSplit:
    """Build the requested leakage-safe subject split."""
    if scenario not in SCENARIO_DETAILS:
        raise ValueError(f"Unsupported scenario: {scenario}")

    details = SCENARIO_DETAILS[scenario]
    if scenario == "combined_holdout":
        train_idx, val_idx, test_idx, train_subjects, val_subjects, test_subjects = subject_train_val_test_indices(
            dataset.y,
            dataset.subject_ids,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    else:
        train_idx, val_idx, test_idx, train_subjects, val_subjects, test_subjects = cross_age_subject_train_val_test_indices(
            dataset.y,
            dataset.subject_ids,
            dataset.age_groups,
            train_age_group=details["train_age_group"],
            test_age_group=details["test_age_group"],
            val_size=val_size,
            random_state=random_state,
        )

    _validate_split(dataset, train_subjects, val_subjects, test_subjects)
    return ExperimentSplit(
        scenario=scenario,
        protocol=details["protocol"],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_age_group=details["train_age_group"],
        test_age_group=details["test_age_group"],
    )


def fit_channel_standardizer(X_train: np.ndarray) -> StandardScaler:
    """Fit a per-channel StandardScaler using training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_epochs(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted per-channel scaler to epoch data."""
    transformed = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return transformed.astype(np.float32)


def split_epoch_dataset(dataset: EpochDataset, split: ExperimentSplit):
    """Slice a full epoch dataset into train/val/test arrays."""
    X_train = dataset.X[split.train_idx]
    X_val = dataset.X[split.val_idx]
    X_test = dataset.X[split.test_idx]
    y_train = dataset.y[split.train_idx]
    y_val = dataset.y[split.val_idx]
    y_test = dataset.y[split.test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_epoch_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | str]:
    """Compute standard epoch-level binary metrics."""
    return compute_binary_metrics(y_true, y_prob, threshold)


def build_results_row(
    model_name: str,
    metrics: dict[str, float | str],
    split: ExperimentSplit,
    seed: int,
    epoch_seconds: int,
    overlap: float,
    val_size: float,
    test_size: float,
) -> dict[str, object]:
    """Create a standardized metrics row."""
    scenario_uses_holdout = split.scenario == "combined_holdout"
    return {
        "Model": model_name,
        "Scenario": split.scenario,
        "Split_Type": "Subject-Wise",
        "Protocol": split.protocol,
        "Evaluation_Level": "Epoch",
        "Train_Age_Group": split.train_age_group,
        "Test_Age_Group": split.test_age_group,
        "SMOTE": 0,
        "Accuracy": metrics["Accuracy"],
        "Balanced_Accuracy": metrics["Balanced_Accuracy"],
        "Specificity": metrics["Specificity"],
        "Sensitivity": metrics["Sensitivity"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "ROC-AUC": metrics["ROC-AUC"],
        "PR-AUC": metrics["PR-AUC"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "TP": metrics["TP"],
        "Confusion_Matrix": metrics["Confusion_Matrix"],
        "Threshold": 0.5,
        "Threshold_Source": "fixed_0.5",
        "Seed": seed,
        "Epoch_Seconds": epoch_seconds,
        "Overlap": overlap,
        "Test_Size": test_size if scenario_uses_holdout else np.nan,
        "Val_Size": val_size,
        "Train_Subjects": len(split.train_subjects),
        "Val_Subjects": len(split.val_subjects),
        "Test_Subjects": len(split.test_subjects),
    }


def build_dual_level_results(
    model_name: str,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    test_subject_ids: np.ndarray,
    split: ExperimentSplit,
    seed: int,
    epoch_seconds: int,
    overlap: float,
    val_size: float,
    test_size: float,
    threshold: float = 0.5,
    threshold_source: str = "fixed_0.5",
    dataset: np.ndarray | str | None = None,
    primary_aggregation: str = "mean",
) -> tuple[dict[str, object], dict[str, object], dict[str, object], pd.DataFrame]:
    """Return epoch, primary-subject, sensitivity-subject rows and subject data.

    The primary endpoint is one mean epoch probability per held-out subject. The
    median aggregation is reported as a subject-level sensitivity analysis.
    """
    epoch_metrics = compute_binary_metrics(y_test, y_prob, threshold)
    subject_metrics, sensitivity_metrics, subject_predictions = evaluate_subject_aggregations(
        y_test, y_prob, test_subject_ids, dataset=dataset, threshold=threshold,
        primary_aggregation=primary_aggregation,
    )
    common = dict(
        model_name=model_name, split=split, seed=seed, epoch_seconds=epoch_seconds,
        overlap=overlap, val_size=val_size, test_size=test_size,
    )
    epoch_row = build_results_row(metrics=epoch_metrics, **common)
    subject_row = build_results_row(metrics=subject_metrics, **common)
    subject_row["Evaluation_Level"] = "Subject"
    subject_row["Aggregation"] = primary_aggregation
    sensitivity_row = build_results_row(metrics=sensitivity_metrics, **common)
    sensitivity_row["Evaluation_Level"] = "Subject_Sensitivity"
    sensitivity_row["Aggregation"] = "median" if primary_aggregation == "mean" else "mean"
    for row in (epoch_row, subject_row, sensitivity_row):
        row["Threshold"] = round(float(threshold), 6)
        row["Threshold_Source"] = threshold_source
        row["Test_Epochs_Control"] = int((np.asarray(y_test) == 0).sum())
        row["Test_Epochs_ADHD"] = int((np.asarray(y_test) == 1).sum())
        row["Test_Subjects_Control"] = int((subject_predictions["true_label"].to_numpy() == 0).sum())
        row["Test_Subjects_ADHD"] = int((subject_predictions["true_label"].to_numpy() == 1).sum())
    return epoch_row, subject_row, sensitivity_row, subject_predictions


def split_class_counts(dataset: EpochDataset, split: ExperimentSplit) -> dict[str, int]:
    """Audit class counts at subject and epoch level for every partition."""
    subject_df = subject_metadata_table(dataset.subject_ids, dataset.y, dataset.age_groups)
    results: dict[str, int] = {}
    for name, indices, subject_ids in (
        ("Train", split.train_idx, split.train_subjects),
        ("Val", split.val_idx, split.val_subjects),
        ("Test", split.test_idx, split.test_subjects),
    ):
        labels = dataset.y[indices]
        subject_labels = subject_df.set_index("subject_id").loc[subject_ids, "label"].to_numpy()
        results[f"{name}_Epochs_Control"] = int((labels == 0).sum())
        results[f"{name}_Epochs_ADHD"] = int((labels == 1).sum())
        results[f"{name}_Subjects_Control"] = int((subject_labels == 0).sum())
        results[f"{name}_Subjects_ADHD"] = int((subject_labels == 1).sum())
    return results


def write_split_manifest(
    output_dir: str | Path,
    dataset: EpochDataset,
    split: ExperimentSplit,
    seed: int,
    epoch_seconds: int,
    overlap: float,
    val_size: float,
    test_size: float,
) -> Path:
    """Persist the exact subject split for auditability."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_df = subject_metadata_table(dataset.subject_ids, dataset.y, dataset.age_groups)
    subject_records = subject_df.set_index("subject_id")[["label", "age_group"]].to_dict(orient="index")

    def _subject_entries(subject_ids):
        return [
            {
                "subject_id": subject_id,
                "label": "ADHD" if int(subject_records[subject_id]["label"]) == 1 else "Control",
                "age_group": subject_records[subject_id]["age_group"],
            }
            for subject_id in subject_ids.tolist()
        ]

    manifest = {
        "scenario": split.scenario,
        "protocol": split.protocol,
        "seed": seed,
        "epoch_seconds": epoch_seconds,
        "overlap": overlap,
        "test_size": test_size if split.scenario == "combined_holdout" else None,
        "val_size": val_size,
        "train_age_group": split.train_age_group,
        "test_age_group": split.test_age_group,
        "train_subjects": _subject_entries(split.train_subjects),
        "val_subjects": _subject_entries(split.val_subjects),
        "test_subjects": _subject_entries(split.test_subjects),
        "class_counts": split_class_counts(dataset, split),
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _validate_split(
    dataset: EpochDataset,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> None:
    """Verify that the split is disjoint and each subject has one age group."""
    subject_df = subject_metadata_table(dataset.subject_ids, dataset.y, dataset.age_groups)
    known_subjects = set(subject_df["subject_id"].tolist())

    for name, subject_ids in {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }.items():
        unknown = sorted(set(np.asarray(subject_ids).tolist()) - known_subjects)
        if unknown:
            raise ValueError(f"{name} split contains unknown subjects: {unknown}")
