"""Helpers for subject-wise EEG dataset splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def subject_metadata_table(subject_ids, labels, age_groups=None):
    """Return one validated row per subject."""
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels)

    if subject_ids.shape[0] != labels.shape[0]:
        raise ValueError("subject_ids and labels must have the same length.")

    subject_df = pd.DataFrame({"subject_id": subject_ids, "label": labels})
    if age_groups is not None:
        age_groups = np.asarray(age_groups)
        if subject_ids.shape[0] != age_groups.shape[0]:
            raise ValueError("subject_ids and age_groups must have the same length.")
        subject_df["age_group"] = age_groups

    grouped = subject_df.groupby("subject_id", sort=False)
    label_lists = grouped["label"].agg(list)

    inconsistent_labels = label_lists[label_lists.map(lambda values: any(value != values[0] for value in values[1:]))]
    if not inconsistent_labels.empty:
        bad_subjects = inconsistent_labels.index.tolist()
        raise ValueError(f"Found subjects with inconsistent labels: {bad_subjects}")

    data = {
        "subject_id": label_lists.index.to_numpy(),
        "label": label_lists.map(lambda values: values[0]).to_numpy(),
    }

    if age_groups is not None:
        age_lists = grouped["age_group"].agg(list)
        inconsistent_age = age_lists[age_lists.map(lambda values: any(value != values[0] for value in values[1:]))]
        if not inconsistent_age.empty:
            bad_subjects = inconsistent_age.index.tolist()
            raise ValueError(f"Found subjects with inconsistent age groups: {bad_subjects}")
        data["age_group"] = age_lists.map(lambda values: values[0]).to_numpy()

    return pd.DataFrame(data)


def _subject_table(subject_ids, labels):
    """Return one row per subject with a single, validated label."""
    return subject_metadata_table(subject_ids, labels)[["subject_id", "label"]]


def _can_stratify(labels, test_size):
    """Return True when a stratified subject split is feasible."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return False

    if isinstance(test_size, float):
        n_test = int(np.ceil(len(labels) * test_size))
    else:
        n_test = int(test_size)

    n_train = len(labels) - n_test
    return n_test >= classes.size and n_train >= classes.size


def split_subject_ids(subject_ids, labels, test_size=0.2, random_state=42):
    """Split unique subjects into train/test sets."""
    subjects_df = _subject_table(subject_ids, labels)
    stratify = subjects_df["label"].to_numpy() if _can_stratify(subjects_df["label"], test_size) else None

    train_subjects, test_subjects = train_test_split(
        subjects_df["subject_id"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    assert_no_subject_overlap(train_subjects, test_subjects)
    return np.asarray(train_subjects), np.asarray(test_subjects)


def subject_holdout_indices(y, subject_ids, test_size=0.2, random_state=42):
    """Return epoch indices for a subject-wise train/test split."""
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    train_subjects, test_subjects = split_subject_ids(
        subject_ids, y, test_size=test_size, random_state=random_state
    )

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Subject-wise split produced an empty train or test set.")

    return train_idx, test_idx, train_subjects, test_subjects


def subject_train_val_test_indices(y, subject_ids, test_size=0.2, val_size=0.1, random_state=42):
    """
    Return epoch indices for subject-wise train/val/test splits.

    `val_size` is the fraction of the remaining training subjects reserved for validation.
    """
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in the range [0, 1).")

    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)

    train_val_idx, test_idx, train_val_subjects, test_subjects = subject_holdout_indices(
        y, subject_ids, test_size=test_size, random_state=random_state
    )

    if val_size == 0:
        return (
            train_val_idx,
            np.array([], dtype=int),
            test_idx,
            train_val_subjects,
            np.array([], dtype=train_val_subjects.dtype),
            test_subjects,
        )

    train_val_subject_ids = subject_ids[train_val_idx]
    train_val_labels = y[train_val_idx]
    train_subjects, val_subjects = split_subject_ids(
        train_val_subject_ids,
        train_val_labels,
        test_size=val_size,
        random_state=random_state + 1,
    )

    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)

    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Subject-wise split produced an empty train or validation set.")

    assert_no_subject_overlap(train_subjects, val_subjects)
    assert_no_subject_overlap(train_subjects, test_subjects)
    assert_no_subject_overlap(val_subjects, test_subjects)

    return train_idx, val_idx, test_idx, train_subjects, val_subjects, test_subjects


def cross_age_subject_train_val_test_indices(
    y,
    subject_ids,
    age_groups,
    train_age_group,
    test_age_group,
    val_size=0.1,
    random_state=42,
):
    """Return epoch indices for train/val on one age group and test on the other."""
    if train_age_group == test_age_group:
        raise ValueError("train_age_group and test_age_group must differ.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in the range [0, 1).")

    subject_df = subject_metadata_table(subject_ids, y, age_groups)
    train_pool = subject_df[subject_df["age_group"] == train_age_group].copy()
    test_pool = subject_df[subject_df["age_group"] == test_age_group].copy()

    if train_pool.empty:
        raise ValueError(f"No training subjects found for age_group={train_age_group!r}.")
    if test_pool.empty:
        raise ValueError(f"No test subjects found for age_group={test_age_group!r}.")

    train_pool_subjects = train_pool["subject_id"].to_numpy()
    train_pool_labels = train_pool["label"].to_numpy()

    if val_size == 0:
        train_subjects = train_pool_subjects
        val_subjects = np.array([], dtype=train_pool_subjects.dtype)
    else:
        train_subjects, val_subjects = split_subject_ids(
            train_pool_subjects,
            train_pool_labels,
            test_size=val_size,
            random_state=random_state,
        )

    test_subjects = test_pool["subject_id"].to_numpy()

    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Cross-age split produced an empty train or test set.")
    if val_size > 0 and val_idx.size == 0:
        raise ValueError("Cross-age split produced an empty validation set.")

    assert_no_subject_overlap(train_subjects, val_subjects)
    assert_no_subject_overlap(train_subjects, test_subjects)
    assert_no_subject_overlap(val_subjects, test_subjects)

    return train_idx, val_idx, test_idx, train_subjects, val_subjects, test_subjects


def assert_no_subject_overlap(train_subjects, test_subjects):
    """Raise if train and test subject sets overlap."""
    overlap = np.intersect1d(np.asarray(train_subjects), np.asarray(test_subjects))
    if overlap.size:
        raise ValueError(f"Data leakage detected: overlapping subjects {overlap.tolist()}")
