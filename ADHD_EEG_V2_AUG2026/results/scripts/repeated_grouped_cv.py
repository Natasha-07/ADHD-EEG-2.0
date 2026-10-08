"""Reusable repeated, stratified, subject-grouped cross-validation framework.

Outer folds are shared by every model. Each outer-training partition is split a
second time at subject level into training and validation subjects, so an optional
threshold can be selected without accessing the outer held-out subjects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.model_selection import StratifiedGroupKFold

from deep_model_utils import EpochDataset
from subject_level_eval_utils import evaluate_subject_aggregations, select_threshold
from subject_split_utils import split_subject_ids, subject_metadata_table


PRIMARY_AGGREGATION = "mean"
METRICS = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1-score", "ROC-AUC", "PR-AUC"]
RESULT_COLUMNS = [
    "Model", "Cohort", "Repeat_Seed", "Fold", "Threshold_Type", "Threshold", "Threshold_Source",
    "Evaluation_Level", *METRICS, "TN", "FP", "FN", "TP", "Confusion_Matrix",
    "Train_Subjects", "Validation_Subjects", "Test_Subjects",
]


@dataclass(frozen=True)
class GroupedFold:
    cohort: str
    repeat_seed: int
    fold: int
    train_subjects: np.ndarray
    validation_subjects: np.ndarray
    test_subjects: np.ndarray


def maximum_valid_folds(subject_labels: np.ndarray, target_folds: int = 5) -> int:
    """Return the largest feasible outer-fold count while preserving inner validation.

    A class needs at least three subjects: one can be held out in an outer fold,
    then the remaining two can occupy inner training and validation. The result is
    never hard-coded from manuscript counts.
    """
    _, counts = np.unique(np.asarray(subject_labels), return_counts=True)
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Repeated grouped CV with an inner validation split requires at least 3 subjects in each class.")
    return int(min(target_folds, counts.min()))


def _assert_disjoint(fold: GroupedFold) -> None:
    sets = [set(fold.train_subjects), set(fold.validation_subjects), set(fold.test_subjects)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError(f"Subject overlap detected in cohort={fold.cohort}, seed={fold.repeat_seed}, fold={fold.fold}.")


def generate_grouped_folds(
    subjects: pd.DataFrame,
    cohort: str,
    seeds: list[int],
    target_folds: int = 5,
    validation_fraction: float = 0.1,
) -> list[GroupedFold]:
    """Generate deterministic repeated outer CV + subject-disjoint inner validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    required = {"subject_id", "label", "age_group"}
    if missing := required.difference(subjects.columns):
        raise ValueError(f"Subject table missing {sorted(missing)}")
    frame = subjects if cohort == "all" else subjects[subjects["age_group"] == cohort]
    if frame.empty:
        raise ValueError(f"No subjects for cohort={cohort!r}.")
    n_folds = maximum_valid_folds(frame["label"].to_numpy(), target_folds)
    subject_ids, labels = frame["subject_id"].to_numpy(), frame["label"].to_numpy()
    folds: list[GroupedFold] = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # Groups are explicitly subject IDs. There is one row per subject here,
        # which makes the grouping invariant clear and auditable.
        for fold_index, (outer_train, outer_test) in enumerate(splitter.split(np.zeros((len(frame), 1)), labels, groups=subject_ids), start=1):
            outer_train_ids, outer_test_ids = subject_ids[outer_train], subject_ids[outer_test]
            outer_train_labels = labels[outer_train]
            train_ids, val_ids = split_subject_ids(
                outer_train_ids, outer_train_labels, test_size=validation_fraction, random_state=seed * 1000 + fold_index
            )
            fold = GroupedFold(cohort, seed, fold_index, train_ids, val_ids, outer_test_ids)
            _assert_disjoint(fold)
            folds.append(fold)
    return folds


def fold_assignments(subjects: pd.DataFrame, folds: list[GroupedFold]) -> pd.DataFrame:
    metadata = subjects.set_index("subject_id")
    rows = []
    for fold in folds:
        for split_name, ids in (("train", fold.train_subjects), ("validation", fold.validation_subjects), ("test", fold.test_subjects)):
            for subject_id in ids:
                record = metadata.loc[subject_id]
                rows.append({
                    "Cohort": fold.cohort, "Repeat_Seed": fold.repeat_seed, "Fold": fold.fold,
                    "Split": split_name, "subject_id": subject_id, "label": int(record["label"]),
                    "label_name": "ADHD" if int(record["label"]) == 1 else "Control", "age_group": record["age_group"],
                    "n_epochs": int(record["n_epochs"]) if "n_epochs" in record.index else np.nan,
                })
    result = pd.DataFrame(rows)
    duplicate = result.groupby(["Cohort", "Repeat_Seed", "Fold", "subject_id"])["Split"].nunique()
    if (duplicate > 1).any():
        raise AssertionError("A subject occurs in more than one split within a CV fold.")
    distribution = result.groupby(["Cohort", "Repeat_Seed", "Fold", "Split"], as_index=False).agg(
        Split_Subjects=("subject_id", "nunique"),
        Split_ADHD_Subjects=("label", lambda labels: int((labels == 1).sum())),
        Split_Control_Subjects=("label", lambda labels: int((labels == 0).sum())),
        Split_Epochs=("n_epochs", "sum"),
        Split_ADHD_Epochs=("n_epochs", lambda epochs: int(epochs[result.loc[epochs.index, "label"] == 1].sum())),
        Split_Control_Epochs=("n_epochs", lambda epochs: int(epochs[result.loc[epochs.index, "label"] == 0].sum())),
    )
    return result.merge(distribution, on=["Cohort", "Repeat_Seed", "Fold", "Split"], how="left", validate="many_to_one")


def _epoch_indices(dataset: EpochDataset, subject_ids: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(dataset.subject_ids, subject_ids))


Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def evaluate_model_on_folds(
    dataset: EpochDataset,
    folds: list[GroupedFold],
    model_name: str,
    predictor: Predictor,
    threshold_method: str = "none",
) -> pd.DataFrame:
    """Run a model callback on shared folds and score held-out subjects.

    ``predictor`` receives training/validation/test epoch tensors and labels, then
    returns validation and test ADHD probabilities. It may fit scalers/models only
    on its training inputs. A test threshold is never selected here.
    """
    if threshold_method not in {"none", "youden_j", "balanced_accuracy"}:
        raise ValueError("threshold_method must be none, youden_j, or balanced_accuracy.")
    rows = []
    for fold in folds:
        train_idx, val_idx, test_idx = (_epoch_indices(dataset, ids) for ids in (fold.train_subjects, fold.validation_subjects, fold.test_subjects))
        val_prob, test_prob = predictor(dataset.X[train_idx], dataset.y[train_idx], dataset.X[val_idx], dataset.y[val_idx], dataset.X[test_idx], fold.repeat_seed)
        val_primary, _, _ = evaluate_subject_aggregations(dataset.y[val_idx], val_prob, dataset.subject_ids[val_idx], dataset.age_groups[val_idx], threshold=0.5, primary_aggregation=PRIMARY_AGGREGATION)
        threshold, source = 0.5, "fixed_0.5_prespecified"
        if threshold_method != "none":
            val_table = pd.DataFrame({"subject_id": dataset.subject_ids[val_idx], "y": dataset.y[val_idx], "p": val_prob}).groupby("subject_id", sort=False).agg(y=("y", "first"), p=("p", "mean"))
            selection = select_threshold(val_table["y"], val_table["p"], threshold_method)
            threshold, source = selection.threshold, f"validation_subject_{threshold_method}"
        primary, sensitivity, _ = evaluate_subject_aggregations(dataset.y[test_idx], test_prob, dataset.subject_ids[test_idx], dataset.age_groups[test_idx], threshold=threshold, primary_aggregation=PRIMARY_AGGREGATION)
        for level, metrics in (("Subject", primary), ("Subject_Sensitivity", sensitivity)):
            rows.append({
                "Model": model_name, "Cohort": fold.cohort, "Repeat_Seed": fold.repeat_seed, "Fold": fold.fold,
                "Threshold_Type": "optimized" if threshold_method != "none" else "default",
                "Threshold": threshold, "Threshold_Source": source, "Evaluation_Level": level,
                **metrics, "Train_Subjects": len(fold.train_subjects), "Validation_Subjects": len(fold.validation_subjects), "Test_Subjects": len(fold.test_subjects),
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize_cv_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level subject metrics with t-based 95% confidence intervals."""
    columns = ["Model", "Cohort", "Threshold_Type", "Evaluation_Level", "Metric", "n_folds", "mean", "std", "ci95_low", "ci95_high"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    keys = ["Model", "Cohort", "Threshold_Type", "Evaluation_Level"]
    for key, frame in results.groupby(keys, dropna=False):
        n = len(frame)
        critical = student_t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        for metric in METRICS:
            values = frame[metric].dropna().astype(float)
            count = len(values)
            mean = values.mean() if count else np.nan
            std = values.std(ddof=1) if count > 1 else np.nan
            margin = critical * std / np.sqrt(count) if count > 1 else np.nan
            rows.append(dict(zip(keys, key)) | {"Metric": metric, "n_folds": count, "mean": mean, "std": std, "ci95_low": mean - margin if count > 1 else np.nan, "ci95_high": mean + margin if count > 1 else np.nan})
    return pd.DataFrame(rows, columns=columns)
