# Dataset and split audit

## Method

- Source CSV: `D:\RESEARCH\ADHD_EEG\CSV Combined and Individual\Single_CSV\combined_eeg.csv`
- Epoch rule: 2-second windows with 50% overlap, matching `deep_model_utils.epochify`.
- All counts are derived from CSV index columns (`subject_id`, `label`, `age_group`, `sample`), not manuscript numbers.
- Every saved `split_manifest.json` was checked for pairwise train/validation/test subject overlap. The audit raises `AssertionError` before writing outputs if overlap is detected.

## Dataset/cohort summary

| cohort | total_subjects | adhd_subjects | control_subjects | total_epochs | adhd_epochs | control_epochs | mean_epochs_per_subject | median_epochs_per_subject | min_epochs_per_subject | max_epochs_per_subject | mean_epochs_per_adhd_subject | mean_epochs_per_control_subject | subject_pct_of_all | epoch_pct_of_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 137 | 69 | 68 | 13983 | 7212 | 6771 | 102.07 | 33.00 | 14 | 657 | 104.52 | 99.57 | 100.00% | 100.00% |
| adult | 16 | 8 | 8 | 9931 | 4943 | 4988 | 620.69 | 619.00 | 610 | 657 | 617.88 | 623.50 | 11.68% | 71.02% |
| child | 121 | 61 | 60 | 4052 | 2269 | 1783 | 33.49 | 31.00 | 14 | 83 | 37.20 | 29.72 | 88.32% | 28.98% |

The `subject_pct_of_all` and `epoch_pct_of_all` columns quantify cohort contribution. Their difference demonstrates the potential dominance of cohorts with longer recordings in epoch-weighted analyses.

## Saved split-manifest coverage

| scenario | split | saved_manifests |
| --- | --- | --- |
| adult_to_child | test | 9 |
| adult_to_child | train | 9 |
| adult_to_child | val | 9 |
| child_to_adult | test | 9 |
| child_to_adult | train | 9 |
| child_to_adult | val | 9 |
| combined_holdout | test | 12 |
| combined_holdout | train | 12 |
| combined_holdout | val | 12 |

`split_summary.csv` contains one row per manifest partition, including class-specific subject/epoch counts, class proportions, the full JSON-encoded list of unique subject IDs, and the overlap result.

## Output files

- `dataset_summary.csv`: overall and cohort-level counts and epoch-distribution statistics.
- `subject_epoch_counts.csv`: one data-derived row per subject.
- `split_summary.csv`: manifest-level train/validation/test audit rows.
