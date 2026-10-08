"""Create a cautious small-sample audit for the adult EEG cohort."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REVISION = ROOT / "Results" / "reviewer_revision"
CV_ASSIGNMENTS = REVISION / "cross_validation" / "fold_assignments.csv"
OUT_REPORT = REVISION / "adult_small_sample_analysis.md"
OUT_PROBABILITIES = REVISION / "adult_available_subject_probabilities.csv"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def load_available_adult_probabilities(adults: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    source = REVISION / "subject_level" / "baselines" / "child_to_adult"
    files = sorted(source.glob("*_subject_level_predictions.csv"))
    if not files:
        return adults[["subject_id", "label", "n_epochs"]].copy(), []
    output = adults[["subject_id", "label", "n_epochs"]].copy()
    used = []
    for path in files:
        model = path.stem.removesuffix("_subject_level_predictions")
        frame = pd.read_csv(path)
        frame = frame[["subject_id", "mean_probability", "median_probability", "predicted_label"]].rename(columns={
            "mean_probability": f"{model}_mean_probability",
            "median_probability": f"{model}_median_probability",
            "predicted_label": f"{model}_predicted_label",
        })
        output = output.merge(frame, on="subject_id", how="left", validate="one_to_one")
        used.append(model)
    return output, used


def main() -> None:
    subjects = pd.read_csv(REVISION / "subject_epoch_counts.csv")
    adults = subjects[subjects["age_group"] == "adult"].sort_values("subject_id").reset_index(drop=True)
    if adults.empty:
        raise ValueError("No adult subjects found in subject_epoch_counts.csv.")
    n_adhd = int((adults["label"] == "ADHD").sum())
    n_control = int((adults["label"] == "Control").sum())
    assignments = pd.read_csv(CV_ASSIGNMENTS)
    adult_assignments = assignments[assignments["Cohort"] == "adult"].copy()
    fold_sizes = adult_assignments.groupby(["Repeat_Seed", "Fold", "Split"], as_index=False).agg(
        subjects=("subject_id", "nunique"), adhd=("label", lambda x: int((x == 1).sum())),
        control=("label", lambda x: int((x == 0).sum())), epochs=("n_epochs", "sum"),
    )
    test_sizes = fold_sizes[fold_sizes["Split"] == "test"]
    appearances = adult_assignments[adult_assignments["Split"] == "test"].groupby("subject_id").size()
    probabilities, models = load_available_adult_probabilities(adults)
    probabilities.to_csv(OUT_PROBABILITIES, index=False)

    examples = []
    for correct, total, name in ((4, 8, "Sensitivity/specificity at 4/8"), (8, 8, "Sensitivity/specificity at 8/8"), (8, 16, "Accuracy at 8/16"), (12, 16, "Accuracy at 12/16")):
        low, high = wilson_interval(correct, total)
        examples.append(f"- {name}: observed {correct/total:.3f}; Wilson 95% CI [{low:.3f}, {high:.3f}].")

    probability_columns = [column for column in probabilities.columns if column.endswith("_mean_probability")]
    probability_table = probabilities.to_markdown(index=False) if False else probabilities.to_csv(index=False)
    # Avoid a tabulate dependency while retaining a compact human-readable table.
    preview = probabilities.head(16).to_string(index=False, float_format=lambda value: f"{value:.3f}")
    report = "\n".join([
        "# Adult cohort: small-sample statistical audit", "",
        "## Independent sample size", "",
        f"- Adult subjects: **{len(adults)}** ({n_adhd} ADHD, {n_control} Control).",
        f"- Adult epochs: **{int(adults['n_epochs'].sum())}** ({int(adults.loc[adults['label'] == 'ADHD', 'n_epochs'].sum())} ADHD-labelled epochs; {int(adults.loc[adults['label'] == 'Control', 'n_epochs'].sum())} Control-labelled epochs).",
        f"- Adult epochs per subject: mean={adults['n_epochs'].mean():.2f}, median={adults['n_epochs'].median():.2f}, range={int(adults['n_epochs'].min())}-{int(adults['n_epochs'].max())}.",
        "",
        "The independent unit for diagnostic classification is the **subject**, not the epoch. The thousands of correlated epochs are repeated measurements within only 16 people; they do not create thousands of independent samples or justify epoch-level confidence intervals.",
        "",
        "## Appropriate grouped validation", "",
        "The adult cohort has eight subjects in each class. The reusable framework correctly selects five outer `StratifiedGroupKFold` folds (the requested maximum) and retains a subject-disjoint inner validation split. Each repeat therefore evaluates held-out folds of 3-4 adults, with all 16 adults appearing once in outer test per repeat.",
        f"Generated design: {adult_assignments['Repeat_Seed'].nunique()} deterministic repeats; outer test-fold size range={int(test_sizes['subjects'].min())}-{int(test_sizes['subjects'].max())}; class counts per test fold are stored in `cross_validation/fold_assignments.csv`; each adult appears {int(appearances.min())}-{int(appearances.max())} times as outer test across repeats.",
        "",
        "## Uncertainty implications", "",
        "Subject-level performance has coarse resolution: one adult changes an all-subject accuracy by 1/16 = 6.25 percentage points; one member of a class changes sensitivity or specificity by 1/8 = 12.5 percentage points. Illustrative Wilson intervals show why point estimates require caution:",
        *examples,
        "Repeated folds provide a distribution of model estimates, but repeats reuse the same 16 people and are not 16 times the number of repeats independent evidence. Fold/seed variability should be reported descriptively with mean, SD, and t-based 95% CI from the grouped-CV framework, not treated as a new subject sample.",
        "",
        "## Available adult individual probabilities", "",
        "The table below is from the available **Child-to-Adult baseline** predictions. It is a cross-cohort transfer diagnostic, not an adult-only training/performance estimate. It is included solely to make subject-to-subject probability instability inspectable. Adult-only neural subject probabilities are not present in the existing artifacts and must be generated by rerunning the new grouped-CV pipeline.",
        "",
        "```text", preview, "```", "",
        f"Saved machine-readable table: `{OUT_PROBABILITIES.relative_to(ROOT)}`. Available predictor columns: {', '.join(probability_columns) if probability_columns else 'none'}.",
        "",
        "## Conclusion", "",
        "No adult-specific clinical or biological conclusion should be based on a single split, epoch-weighted result, or a narrow apparent confidence interval. The adult cohort supports a cautious repeated subject-grouped analysis, but its eight ADHD and eight Control participants leave substantial uncertainty and high sensitivity to individual subjects and fold assignment.",
    ]) + "\n"
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
