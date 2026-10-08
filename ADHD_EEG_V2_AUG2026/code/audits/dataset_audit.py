"""Audit the EEG dataset and saved subject-wise split manifests.

This module deliberately derives all counts from the supplied combined EEG CSV and
the JSON split manifests.  It does not import, train, or modify any model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "CSV Combined and Individual" / "Single_CSV" / "combined_eeg.csv"
DEFAULT_SPLIT_ROOT = ROOT / "Results"
DEFAULT_OUT = ROOT / "Results" / "reviewer_revision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a data-derived EEG dataset and split audit.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Combined EEG CSV to audit.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Root to search for split_manifest.json files.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--epoch-seconds", type=float, default=2.0, help="Epoch duration used by the benchmark loader.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Epoch overlap used by the benchmark loader.")
    parser.add_argument("--chunksize", type=int, default=250_000, help="CSV rows per read chunk.")
    return parser.parse_args()


def epoch_count(n_samples: int, sampling_rate: int, epoch_seconds: float, overlap: float) -> int:
    """Match deep_model_utils.epochify: windows are counted before resampling."""
    window = int(epoch_seconds * sampling_rate)
    step = int(window * (1 - overlap))
    if window <= 0 or step <= 0:
        raise ValueError("epoch_seconds and overlap must yield positive window and step sizes.")
    return 0 if n_samples < window else 1 + (n_samples - window) // step


def load_subject_index(csv_path: Path, epoch_seconds: float, overlap: float, chunksize: int) -> pd.DataFrame:
    """Derive one validated row per subject directly from the raw combined CSV."""
    required = {"subject_id", "label", "age_group", "sample"}
    subject_rows: dict[str, int] = defaultdict(int)
    labels: dict[str, set[str]] = defaultdict(set)
    age_groups: dict[str, set[str]] = defaultdict(set)
    sample_min: dict[str, int] = {}
    sample_max: dict[str, int] = {}

    header = pd.read_csv(csv_path, nrows=0)
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    for chunk in pd.read_csv(csv_path, usecols=sorted(required), chunksize=chunksize):
        if chunk["subject_id"].isna().any() or chunk[["label", "age_group", "sample"]].isna().any().any():
            raise ValueError("The dataset index contains missing subject_id, label, age_group, or sample values.")
        chunk["subject_id"] = chunk["subject_id"].astype(str)
        chunk["label"] = chunk["label"].astype(str)
        chunk["age_group"] = chunk["age_group"].astype(str)
        chunk["sample"] = pd.to_numeric(chunk["sample"], errors="raise").astype(np.int64)

        for subject_id, group in chunk.groupby("subject_id", sort=False):
            subject_rows[subject_id] += len(group)
            labels[subject_id].update(group["label"].unique())
            age_groups[subject_id].update(group["age_group"].unique())
            low, high = int(group["sample"].min()), int(group["sample"].max())
            sample_min[subject_id] = min(sample_min.get(subject_id, low), low)
            sample_max[subject_id] = max(sample_max.get(subject_id, high), high)

    rows: list[dict[str, object]] = []
    for subject_id in sorted(subject_rows):
        if len(labels[subject_id]) != 1 or len(age_groups[subject_id]) != 1:
            raise ValueError(
                f"Subject {subject_id!r} has inconsistent labels {sorted(labels[subject_id])} "
                f"or age groups {sorted(age_groups[subject_id])}."
            )
        label = next(iter(labels[subject_id]))
        age_group = next(iter(age_groups[subject_id]))
        if label not in {"ADHD", "Control"}:
            raise ValueError(f"Unexpected label {label!r} for subject {subject_id!r}; expected ADHD or Control.")
        if age_group not in {"child", "adult"}:
            raise ValueError(f"Unexpected age_group {age_group!r} for subject {subject_id!r}; expected child or adult.")

        n_samples = subject_rows[subject_id]
        expected_samples = sample_max[subject_id] - sample_min[subject_id] + 1
        if n_samples != expected_samples:
            raise ValueError(
                f"Subject {subject_id!r} has {n_samples} CSV rows but sample range "
                f"{sample_min[subject_id]}..{sample_max[subject_id]}; index is not contiguous/unique."
            )
        fs = 512 if age_group == "child" else 256
        rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "label_binary": 1 if label == "ADHD" else 0,
                "age_group": age_group,
                "n_samples": n_samples,
                "sampling_rate_used": fs,
                "n_epochs": epoch_count(n_samples, fs, epoch_seconds, overlap),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No subject records were found in the input CSV.")
    return result


def dataset_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, frame in [("all", subjects), *subjects.groupby("age_group", sort=True)]:
        adhd = frame[frame["label"] == "ADHD"]
        control = frame[frame["label"] == "Control"]
        rows.append(
            {
                "cohort": cohort,
                "total_subjects": len(frame),
                "adhd_subjects": len(adhd),
                "control_subjects": len(control),
                "total_epochs": int(frame["n_epochs"].sum()),
                "adhd_epochs": int(adhd["n_epochs"].sum()),
                "control_epochs": int(control["n_epochs"].sum()),
                "mean_epochs_per_subject": frame["n_epochs"].mean(),
                "median_epochs_per_subject": frame["n_epochs"].median(),
                "min_epochs_per_subject": int(frame["n_epochs"].min()),
                "max_epochs_per_subject": int(frame["n_epochs"].max()),
                "mean_epochs_per_adhd_subject": adhd["n_epochs"].mean(),
                "mean_epochs_per_control_subject": control["n_epochs"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    all_row = summary.loc[summary["cohort"] == "all"].iloc[0]
    summary["subject_pct_of_all"] = 100 * summary["total_subjects"] / all_row["total_subjects"]
    summary["epoch_pct_of_all"] = 100 * summary["total_epochs"] / all_row["total_epochs"]
    return summary


def assert_no_subject_overlap(split_ids: dict[str, list[str]], manifest_path: Path) -> None:
    """Fail immediately when a saved manifest uses a subject in multiple partitions."""
    memberships: dict[str, list[str]] = defaultdict(list)
    for split_name, ids in split_ids.items():
        for subject_id in ids:
            memberships[subject_id].append(split_name)
    overlapping = {subject_id: parts for subject_id, parts in memberships.items() if len(parts) > 1}
    if overlapping:
        raise AssertionError(f"Subject overlap in {manifest_path}: {overlapping}")


def audit_split_manifests(subjects: pd.DataFrame, split_root: Path) -> pd.DataFrame:
    subject_index = subjects.set_index("subject_id")
    output_rows: list[dict[str, object]] = []
    manifests = sorted(split_root.rglob("split_manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_ids = {
            name: [str(item["subject_id"]) for item in manifest.get(f"{name}_subjects", [])]
            for name in ("train", "val", "test")
        }
        assert_no_subject_overlap(split_ids, manifest_path)
        all_ids = [subject_id for ids in split_ids.values() for subject_id in ids]
        unknown = sorted(set(all_ids).difference(subject_index.index))
        if unknown:
            raise ValueError(f"Unknown subject IDs in {manifest_path}: {unknown}")

        for split_name, ids in split_ids.items():
            frame = subject_index.loc[ids] if ids else subject_index.iloc[0:0]
            if isinstance(frame, pd.Series):
                frame = frame.to_frame().T
            n_subjects = len(frame)
            n_epochs = int(frame["n_epochs"].sum())
            adhd_subjects = int((frame["label"] == "ADHD").sum())
            control_subjects = int((frame["label"] == "Control").sum())
            adhd_epochs = int(frame.loc[frame["label"] == "ADHD", "n_epochs"].sum())
            control_epochs = int(frame.loc[frame["label"] == "Control", "n_epochs"].sum())
            output_rows.append(
                {
                    "manifest_path": str(manifest_path.relative_to(ROOT)),
                    "scenario": manifest.get("scenario", ""),
                    "protocol": manifest.get("protocol", ""),
                    "seed": manifest.get("seed", ""),
                    "split": split_name,
                    "n_subjects": n_subjects,
                    "adhd_subjects": adhd_subjects,
                    "control_subjects": control_subjects,
                    "n_epochs": n_epochs,
                    "adhd_epochs": adhd_epochs,
                    "control_epochs": control_epochs,
                    "adhd_subject_proportion": adhd_subjects / n_subjects if n_subjects else np.nan,
                    "adhd_epoch_proportion": adhd_epochs / n_epochs if n_epochs else np.nan,
                    "subject_overlap_detected": False,
                    "unique_subject_ids": json.dumps(ids),
                }
            )
    if not output_rows:
        raise FileNotFoundError(f"No split_manifest.json files found under {split_root}")
    return pd.DataFrame(output_rows)


def markdown_report(summary: pd.DataFrame, split_summary: pd.DataFrame, args: argparse.Namespace) -> str:
    display_summary = summary.copy()
    display_summary["subject_pct_of_all"] = display_summary["subject_pct_of_all"].map(lambda x: f"{x:.2f}%")
    display_summary["epoch_pct_of_all"] = display_summary["epoch_pct_of_all"].map(lambda x: f"{x:.2f}%")
    for column in display_summary.columns:
        if column.startswith("mean_") or column.startswith("median_"):
            display_summary[column] = display_summary[column].map(lambda x: f"{x:.2f}")
    split_counts = split_summary.groupby(["scenario", "split"], sort=True).size().reset_index(name="saved_manifests")

    def as_markdown_table(frame: pd.DataFrame) -> str:
        """Render a small Markdown table without requiring pandas' tabulate extra."""
        columns = [str(column) for column in frame.columns]
        values = frame.fillna("").astype(str).values.tolist()
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(row) + " |" for row in values]
        return "\n".join([header, divider, *body])

    return "\n".join(
        [
            "# Dataset and split audit",
            "",
            "## Method",
            "",
            f"- Source CSV: `{args.csv}`",
            f"- Epoch rule: {args.epoch_seconds:g}-second windows with {args.overlap:.0%} overlap, matching `deep_model_utils.epochify`.",
            "- All counts are derived from CSV index columns (`subject_id`, `label`, `age_group`, `sample`), not manuscript numbers.",
            "- Every saved `split_manifest.json` was checked for pairwise train/validation/test subject overlap. The audit raises `AssertionError` before writing outputs if overlap is detected.",
            "",
            "## Dataset/cohort summary",
            "",
            as_markdown_table(display_summary),
            "",
            "The `subject_pct_of_all` and `epoch_pct_of_all` columns quantify cohort contribution. Their difference demonstrates the potential dominance of cohorts with longer recordings in epoch-weighted analyses.",
            "",
            "## Saved split-manifest coverage",
            "",
            as_markdown_table(split_counts),
            "",
            "`split_summary.csv` contains one row per manifest partition, including class-specific subject/epoch counts, class proportions, the full JSON-encoded list of unique subject IDs, and the overlap result.",
            "",
            "## Output files",
            "",
            "- `dataset_summary.csv`: overall and cohort-level counts and epoch-distribution statistics.",
            "- `subject_epoch_counts.csv`: one data-derived row per subject.",
            "- `split_summary.csv`: manifest-level train/validation/test audit rows.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    args.csv = args.csv.resolve()
    args.split_root = args.split_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    subjects = load_subject_index(args.csv, args.epoch_seconds, args.overlap, args.chunksize)
    summary = dataset_summary(subjects)
    splits = audit_split_manifests(subjects, args.split_root)

    subjects.to_csv(args.out / "subject_epoch_counts.csv", index=False)
    summary.to_csv(args.out / "dataset_summary.csv", index=False, float_format="%.6f")
    splits.to_csv(args.out / "split_summary.csv", index=False, float_format="%.6f")
    (args.out / "dataset_audit.md").write_text(markdown_report(summary, splits, args), encoding="utf-8")

    print("\nPublication-ready dataset summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nAudited {len(splits) // 3} saved split manifests; no subject overlap detected.")
    print(f"Saved audit files to: {args.out}")


if __name__ == "__main__":
    main()
