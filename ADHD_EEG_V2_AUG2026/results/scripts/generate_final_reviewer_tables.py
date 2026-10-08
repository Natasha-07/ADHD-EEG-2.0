"""Create final tables solely from reviewer-revision result artifacts."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
REV = ROOT / "Results" / "reviewer_revision"
OUT = REV / "tables"
METRICS = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC"]


def markdown(frame: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    text = f"# {title}\n\n"
    if note:
        text += note + "\n\n"
    columns = [str(column) for column in rendered.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in rendered.fillna("NA").astype(str).itertuples(index=False, name=None)]
    text += "| " + " | ".join(columns) + " |\n"
    text += "| " + " | ".join("---" for _ in columns) + " |\n"
    text += "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_dataset = REV / "dataset_summary.csv"
    source_baseline = REV / "baselines" / "baseline_results_all_scenarios.csv"
    source_stats = REV / "statistics" / "model_comparisons.csv"
    source_ci = REV / "statistics" / "confidence_intervals.csv"
    source_cv = REV / "cross_validation" / "cv_summary.csv"
    source_comparison = REV / "baselines" / "baseline_neural_comparison.csv"

    dataset = pd.read_csv(source_dataset)
    comparison = pd.read_csv(REV / "dataset_comparison_table.csv")
    rates = comparison.set_index("age_group")[[
        "native_sampling_rate_used_by_code_hz", "resampled_rate_or_equivalent_hz", "number_of_channels_benchmark_input"
    ]]
    table1 = dataset[dataset.cohort != "all"].copy()
    table1["channels"] = table1.cohort.map(rates.number_of_channels_benchmark_input)
    table1["native_sampling_rate_hz"] = table1.cohort.map(rates.native_sampling_rate_used_by_code_hz)
    table1["processed_sampling_rate_hz"] = table1.cohort.map(rates.resampled_rate_or_equivalent_hz)
    table1 = table1.rename(columns={"cohort": "Cohort", "total_subjects": "Total subjects", "adhd_subjects": "ADHD subjects", "control_subjects": "Control subjects", "total_epochs": "Total epochs", "adhd_epochs": "ADHD epochs", "control_epochs": "Control epochs", "channels": "Channels", "native_sampling_rate_hz": "Native sampling rate (Hz)", "processed_sampling_rate_hz": "Processed sampling rate (Hz)"})
    table1 = table1[["Cohort", "Total subjects", "ADHD subjects", "Control subjects", "Total epochs", "ADHD epochs", "Control epochs", "Channels", "Native sampling rate (Hz)", "Processed sampling rate (Hz)"]]
    table1.to_csv(OUT / "table1_dataset_characteristics.csv", index=False)
    markdown(table1, OUT / "table1_dataset_characteristics.md", "Table 1. Dataset characteristics", "All counts are derived from reviewer-revision dataset auditing; epochs are not independent subjects.")

    baseline = pd.read_csv(source_baseline)
    main = baseline[(baseline.Scenario == "combined_holdout") & (baseline.Evaluation_Level == "Subject")].copy()
    main["N subjects"] = main.Test_Subjects
    table2 = main.rename(columns={"Model": "Model", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Balanced_Accuracy": "Balanced Accuracy", "F1": "F1"})
    table2 = table2[["Model", "N subjects", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "PR-AUC"]]
    unavailable_neural = pd.DataFrame([{"Model": "CNN", "N subjects": pd.NA, "Accuracy": pd.NA, "Balanced Accuracy": pd.NA, "Sensitivity": pd.NA, "Specificity": pd.NA, "Precision": pd.NA, "F1": pd.NA, "ROC-AUC": pd.NA, "PR-AUC": pd.NA}, {"Model": "LSTM", "N subjects": pd.NA, "Accuracy": pd.NA, "Balanced Accuracy": pd.NA, "Sensitivity": pd.NA, "Specificity": pd.NA, "Precision": pd.NA, "F1": pd.NA, "ROC-AUC": pd.NA, "PR-AUC": pd.NA}, {"Model": "Bi-LSTM", "N subjects": pd.NA, "Accuracy": pd.NA, "Balanced Accuracy": pd.NA, "Sensitivity": pd.NA, "Specificity": pd.NA, "Precision": pd.NA, "F1": pd.NA, "ROC-AUC": pd.NA, "PR-AUC": pd.NA}, {"Model": "Temporal Transformer Baseline", "N subjects": pd.NA, "Accuracy": pd.NA, "Balanced Accuracy": pd.NA, "Sensitivity": pd.NA, "Specificity": pd.NA, "Precision": pd.NA, "F1": pd.NA, "ROC-AUC": pd.NA, "PR-AUC": pd.NA}])
    table2 = pd.concat([table2, unavailable_neural], ignore_index=True)
    table2.to_csv(OUT / "table2_main_subject_level_performance.csv", index=False)
    markdown(table2, OUT / "table2_main_subject_level_performance.md", "Table 2. Main subject-level model performance", "Rows marked NA are retained neural models, but no newly generated reviewer-revision subject-level neural results exist. Old CleanDeepBenchmark results were not used.")

    cv = pd.read_csv(source_cv)
    if cv.empty:
        table3 = pd.DataFrame([{"Model": "All models", "Metric": "All primary metrics", "Mean": pd.NA, "Standard deviation": pd.NA, "95% CI": "Not available", "Status": "Repeated grouped CV result file is an initialized schema with zero result rows."}])
    else:
        table3 = cv[cv.Evaluation_Level == "Subject"].rename(columns={"Model": "Model", "Metric": "Metric", "mean": "Mean", "std": "Standard deviation"})
        table3["95% CI"] = table3.apply(lambda row: f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}]", axis=1)
        table3["Status"] = "Available"
        table3 = table3[["Model", "Metric", "Mean", "Standard deviation", "95% CI", "Status"]]
    table3.to_csv(OUT / "table3_repeated_subject_grouped_cv.csv", index=False)
    markdown(table3, OUT / "table3_repeated_subject_grouped_cv.md", "Table 3. Repeated subject-grouped cross-validation", "No CV values are imputed or copied from holdout/epoch evaluations.")

    transfer = baseline[(baseline.Scenario.isin(["adult_to_child", "child_to_adult"])) & (baseline.Evaluation_Level == "Subject")].copy()
    transfer["Transfer direction"] = transfer.Scenario.map({"adult_to_child": "Adult dataset -> Child dataset", "child_to_adult": "Child dataset -> Adult dataset"})
    transfer["N target subjects"] = transfer.Test_Subjects
    table4 = transfer.rename(columns={"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Balanced_Accuracy": "Balanced Accuracy", "F1": "F1"})
    table4 = table4[["Transfer direction", "Model", "N target subjects", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "PR-AUC"]]
    table4.to_csv(OUT / "table4_cross_dataset_transfer.csv", index=False)
    markdown(table4, OUT / "table4_cross_dataset_transfer.md", "Table 4. Cross-cohort/cross-dataset transfer", "Subject-level target evaluation. These results do not isolate an age effect.")

    comparisons = pd.read_csv(source_stats)
    table5 = comparisons.rename(columns={"model_a": "Model A", "model_b": "Model B", "Metric": "Metric", "effect_difference_a_minus_b": "Difference (A - B)", "raw_p_value": "Raw p-value", "holm_corrected_p_value": "Holm-corrected p-value", "bootstrap_95ci_low": "95% CI low", "bootstrap_95ci_high": "95% CI high", "test_statistic": "Test statistic"})
    table5 = table5[["scenario", "Model A", "Model B", "Metric", "Difference (A - B)", "95% CI low", "95% CI high", "Test statistic", "Raw p-value", "Holm-corrected p-value", "statistically_significant_holm_0_05"]]
    table5.to_csv(OUT / "table5_statistical_model_comparisons.csv", index=False)
    formatted5 = table5[table5.Metric == "Balanced_Accuracy"].rename(columns={"scenario": "Scenario", "statistically_significant_holm_0_05": "Holm significant (0.05)"})
    markdown(formatted5, OUT / "table5_statistical_model_comparisons.md", "Table 5. Statistical model comparisons", "Formatted table shows the prespecified primary metric (balanced accuracy); the CSV contains every tested metric and comparison. All tests are paired at subject level.")

    confusion = baseline[baseline.Evaluation_Level == "Subject"].copy()
    confusion["N subjects"] = confusion.Test_Subjects
    confusion["Evaluation"] = confusion.Scenario.map({"combined_holdout": "Main combined holdout", "adult_to_child": "Adult dataset -> Child dataset", "child_to_adult": "Child dataset -> Adult dataset"})
    confusion = confusion.rename(columns={"Balanced_Accuracy": "Balanced Accuracy", "TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP", "Sensitivity": "Sensitivity", "Specificity": "Specificity"})
    confusion = confusion[["Evaluation", "Model", "N subjects", "TN", "FP", "FN", "TP", "Sensitivity", "Specificity", "Balanced Accuracy"]]
    confusion.to_csv(OUT / "confusion_matrix_summary.csv", index=False)
    markdown(confusion, OUT / "confusion_matrix_summary.md", "Subject-level confusion-matrix summary")

    report = f"""# Final reviewer-revision table generation report

## Scope

Tables were generated only from newly generated reviewer-revision artifacts. No manuscript source, legacy paper table, or `Results/CleanDeepBenchmark` metric was read for numerical table values.

| Output | Exact source artifact(s) | Notes |
|---|---|---|
| Table 1 | `{source_dataset.relative_to(ROOT)}`; `{(REV / 'dataset_comparison_table.csv').relative_to(ROOT)}` | Dataset counts plus audited channel/rate metadata. |
| Table 2 | `{source_baseline.relative_to(ROOT)}` filtered to `combined_holdout`, `Subject` | Six baselines have new subject-level results. Neural rows are explicitly NA because no eligible reviewer-revision neural output exists. |
| Table 3 | `{source_cv.relative_to(ROOT)}` | The source has headers only. Table reports unavailability rather than values. |
| Table 4 | `{source_baseline.relative_to(ROOT)}` filtered to transfer scenarios and `Subject` | Subject-level cross-cohort/cross-dataset transfer for the newly run baselines only. |
| Table 5 | `{source_stats.relative_to(ROOT)}` | Paired subject-level baseline comparisons, with bootstrap CIs and Holm correction. |
| Confusion summary | `{source_baseline.relative_to(ROOT)}` filtered to `Subject` | TN, FP, FN, TP and requested diagnostic measures. |

`{source_ci.relative_to(ROOT)}` was audited as the CI source for performance/XAI uncertainty but is not duplicated into a performance table because Table 2's requested columns specify point estimates. It remains the supporting reviewer-revision uncertainty artifact.

## Explicit exclusions

- `{source_comparison.relative_to(ROOT)}` was not used as a numerical source because it contains old `Results/CleanDeepBenchmark` neural results, including epoch-level rows.
- No `.tex`, `.docx`, abstract, title, or manuscript text was modified.
- Repeated grouped-CV and neural subject-level results must be generated before Tables 2 and 3 can be considered complete for all retained neural models.
"""
    (OUT / "table_generation_report.md").write_text(report, encoding="utf-8")
    print(f"Saved final reviewer-revision tables to {OUT}")


if __name__ == "__main__":
    main()
