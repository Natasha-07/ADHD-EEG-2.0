"""Generate shared repeated subject-grouped CV assignments and result templates."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deep_model_utils import load_epoch_dataset
from repeated_grouped_cv import RESULT_COLUMNS, fold_assignments, generate_grouped_folds, summarize_cv_results
from subject_split_utils import subject_metadata_table


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results" / "reviewer_revision" / "cross_validation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 31415, 27182])
    parser.add_argument("--target-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_epoch_dataset(args.csv)
    subjects = subject_metadata_table(dataset.subject_ids, dataset.y, dataset.age_groups)
    subjects["n_epochs"] = subjects["subject_id"].map(pd.Series(dataset.subject_ids).value_counts())
    all_folds = []
    for cohort in ("all", "child", "adult"):
        all_folds.extend(generate_grouped_folds(subjects, cohort, args.seeds, args.target_folds, args.validation_fraction))
    assignments = fold_assignments(subjects, all_folds)
    assignments.to_csv(OUT / "fold_assignments.csv", index=False)
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(OUT / "cv_results_all_folds.csv", index=False)
    summarize_cv_results(pd.DataFrame(columns=RESULT_COLUMNS)).to_csv(OUT / "cv_summary.csv", index=False)
    fold_counts = assignments.groupby("Cohort").agg(repeats=("Repeat_Seed", "nunique"), folds_per_repeat=("Fold", "nunique"), subjects=("subject_id", "nunique")).reset_index()
    print("Shared repeated subject-grouped CV folds generated:")
    print(fold_counts.to_string(index=False))
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
