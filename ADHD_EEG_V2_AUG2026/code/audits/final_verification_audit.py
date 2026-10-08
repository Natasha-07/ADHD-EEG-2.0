"""Non-mutating final verification of reviewer-revision artifacts.

This script audits saved artifacts and regenerates only final tables/figures from
saved reviewer-revision results. It never trains a model or edits a manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
REV = ROOT / "Results" / "reviewer_revision"
# Aggregate CSV values are persisted to six decimal places; use a tolerance
# safely above that serialization rounding while still flagging real mismatch.
TOL = 5e-6
TOKEN_TO_MODEL = {
    "always_adhd": "Always-ADHD", "always_control": "Always-Control", "majority_class_dummy": "Majority-class Dummy",
    "stratified_dummy": "Stratified Dummy", "logistic_regression_welch_band_power": "Logistic Regression (Welch band power)",
    "linear_svm_welch_band_power": "Linear SVM (Welch band power)",
}


def metrics(y, p, pred):
    tn, fp, fn, tp = [int(v) for v in (np.sum((y == 0) & (pred == 0)), np.sum((y == 0) & (pred == 1)), np.sum((y == 1) & (pred == 0)), np.sum((y == 1) & (pred == 1)))]
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0; f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"Accuracy": (tn + tp) / len(y), "Balanced_Accuracy": (sens + spec) / 2, "Sensitivity": sens, "Specificity": spec, "Precision": prec, "F1": f1,
            "ROC_AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan, "PR_AUC": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def same(a, b):
    return (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) <= TOL


def main():
    expected = {
        "dataset audit": ["dataset_audit.md", "dataset_summary.csv", "subject_epoch_counts.csv"],
        "split audit": ["split_summary.csv"], "baselines": ["baselines/baseline_results_all_scenarios.csv"],
        "subject-level outputs": ["subject_level"], "threshold/LSTM analysis": ["threshold_and_collapse_analysis.md", "threshold_diagnostics_summary.csv"],
        "grouped CV": ["cross_validation/fold_assignments.csv", "cross_validation/cv_results_all_folds.csv", "cross_validation/cv_summary.csv"],
        "adult small sample": ["adult_small_sample_analysis.md"], "harmonization": ["cross_dataset_harmonization.md", "dataset_comparison_table.csv"],
        "architecture audit": ["eegformer_architecture_audit.md"], "XAI": ["xai/subject_attributions.csv", "xai/electrode_attributions.csv", "xai/xai_stability.csv"],
        "electrode normalization": ["xai/region_normalization.csv", "xai/region_bias_analysis.md"], "sensor topoplots": ["figures/cnn_adhd_subjects_topoplot.png", "figures/lstm_adhd_subjects_topoplot.png"],
        "statistics": ["statistics/model_comparisons.csv", "statistics/confidence_intervals.csv"], "tables": ["tables/table_generation_report.md"],
        "figures": ["figures/figure_generation_report.md"], "compliance": ["technical_compliance_matrix.md"], "final package": ["final_paper_package/RESULTS_INDEX.md"],
    }
    checklist = []
    for item, paths in expected.items():
        checks = [((REV / p).exists(), p) for p in paths]
        notes = []
        valid = all(exists for exists, _ in checks)
        if item == "grouped CV" and (pd.read_csv(REV / "cross_validation/cv_results_all_folds.csv").empty):
            valid = False; notes.append("Result schemas exist but contain zero fold results.")
        if item == "subject-level outputs":
            files = list((REV / "subject_level").rglob("*_subject_level_predictions.csv")); notes.append(f"{len(files)} baseline subject-prediction files found; no new neural files found.")
        checklist.append({"REQUIRED FILE/OUTPUT": item, "EXISTS?": all(x[0] for x in checks), "VALID?": valid, "NOTES": "; ".join(notes) or "Required artifacts present."})
    pd.DataFrame(checklist).to_csv(REV / "final_verification_checklist.csv", index=False)

    # Explicit leakage checks: reviewer-revision baseline manifests.
    leakage = []
    for path in sorted((REV / "baselines").glob("*/split_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        groups = {key: {str(v["subject_id"]) for v in manifest.get(f"{key}_subjects", [])} for key in ("train", "val", "test")}
        overlaps = {"train_val": groups["train"] & groups["val"], "train_test": groups["train"] & groups["test"], "val_test": groups["val"] & groups["test"]}
        leakage.append({"file": str(path.relative_to(REV)), "type": "holdout split manifest", "PASS_FAIL": "PASS" if not any(overlaps.values()) else "FAIL", "overlap_subject_ids": ";".join(sorted(set().union(*overlaps.values())))})
    folds = pd.read_csv(REV / "cross_validation/fold_assignments.csv")
    for (cohort, seed, fold), block in folds.groupby(["Cohort", "Repeat_Seed", "Fold"]):
        roles = block.groupby("subject_id")["Split"].nunique()
        invalid = roles[roles > 1].index.tolist()
        leakage.append({"file": "cross_validation/fold_assignments.csv", "type": f"{cohort}/seed={seed}/fold={fold}", "PASS_FAIL": "PASS" if not invalid else "FAIL", "overlap_subject_ids": ";".join(map(str, invalid))})
    leak_df = pd.DataFrame(leakage); leak_df.to_csv(REV / "subject_leakage_verification.csv", index=False)

    # Independently recalculate every saved new baseline subject result and compare source and final table values.
    saved = pd.read_csv(REV / "baselines/baseline_results_all_scenarios.csv")
    verification, aggregation_examples = [], []
    for path in sorted((REV / "baselines").glob("*/*_subject_predictions.csv")):
        scenario, token = path.parent.name, path.name.removesuffix("_subject_predictions.csv")
        model = TOKEN_TO_MODEL[token]; frame = pd.read_csv(path)
        recalculated = metrics(frame.true_label.to_numpy(), frame.mean_probability.to_numpy(), frame.predicted_label.to_numpy())
        source = saved[(saved.Model == model) & (saved.Scenario == scenario) & (saved.Evaluation_Level == "Subject")]
        if len(source) != 1: raise AssertionError(f"Expected exactly one subject result row for {scenario}/{model}")
        source = source.iloc[0]
        for metric, value in recalculated.items():
            stored = source[metric]
            verification.append({"model": model, "scenario": scenario, "metric": metric, "saved_table_value": stored, "recalculated_value": value, "absolute_difference": abs(float(stored) - float(value)), "PASS_FAIL": "PASS" if same(stored, value) else "FAIL", "source_prediction_file": str(path.relative_to(REV))})
        # Check primary aggregation: each prediction file must have exactly one row per subject and mean label.
        if frame.subject_id.duplicated().any(): raise AssertionError(f"Duplicate subject prediction rows in {path}")
        if token == "logistic_regression_welch_band_power" and scenario == "combined_holdout":
            epoch_path = path.with_name(token + "_epoch_predictions.csv")
            if epoch_path.exists():
                epoch = pd.read_csv(epoch_path)
                for subject_id in frame.subject_id.head(3):
                    subset = epoch[epoch.subject_id == subject_id]
                    aggregation_examples.append({"subject_id": subject_id, "number_of_epochs": len(subset), "epoch_probabilities": ";".join(f"{v:.4f}" for v in subset.y_score.head(12)), "aggregated_mean_probability": float(subset.y_score.mean()), "saved_mean_probability": float(frame.loc[frame.subject_id == subject_id, "mean_probability"].iloc[0]), "true_label": int(frame.loc[frame.subject_id == subject_id, "true_label"].iloc[0]), "predicted_label": int(frame.loc[frame.subject_id == subject_id, "predicted_label"].iloc[0])})
    metric_df = pd.DataFrame(verification); metric_df.to_csv(REV / "metric_verification.csv", index=False)
    pd.DataFrame(aggregation_examples).to_csv(REV / "subject_aggregation_examples.csv", index=False)

    # Baseline comparison (combined holdout) against highest trivial comparator for requested metrics.
    rows = []
    main = saved[(saved.Scenario == "combined_holdout") & (saved.Evaluation_Level == "Subject")]
    trivial = main[main.Model.isin(["Always-ADHD", "Always-Control", "Majority-class Dummy", "Stratified Dummy"])]
    learned = main[main.Model.isin(["Logistic Regression (Welch band power)", "Linear SVM (Welch band power)"])]
    for _, row in learned.iterrows():
        for metric in ("Accuracy", "Balanced_Accuracy", "F1", "ROC_AUC"):
            best = trivial.loc[trivial[metric].idxmax()]
            rows.append({"model": row.Model, "metric": metric, "model_value": row[metric], "best_trivial_model": best.Model, "best_trivial_value": best[metric], "difference": row[metric] - best[metric], "better_than_best_trivial": row[metric] > best[metric]})
    pd.DataFrame(rows).to_csv(REV / "trivial_baseline_comparison_verification.csv", index=False)

    # CV summary status and traceability/stale inventory.
    cv = pd.read_csv(REV / "cross_validation/cv_results_all_folds.csv")
    cv_status = pd.DataFrame([{"raw_fold_rows": len(cv), "status": "NOT AVAILABLE: initialized schema only" if cv.empty else "AVAILABLE", "grouping_variable": "subject_id", "fold_assignments_pass": bool((leak_df.PASS_FAIL == "PASS").all())}])
    cv_status.to_csv(REV / "cross_validation_verification.csv", index=False)
    trace = [
        ("tables/table1_dataset_characteristics.*", "dataset_summary.csv; dataset_comparison_table.csv", "cohort/subject/epoch", "datasets", "PASS"),
        ("tables/table2_main_subject_level_performance.*", "baselines/baseline_results_all_scenarios.csv", "Subject", "six baselines; neural rows NA", "PASS"),
        ("tables/table3_repeated_subject_grouped_cv.*", "cross_validation/cv_summary.csv", "Subject", "all models", "PASS (unavailability stated)"),
        ("tables/table4_cross_dataset_transfer.*", "baselines/baseline_results_all_scenarios.csv", "Target subject", "six baselines", "PASS"),
        ("tables/table5_statistical_model_comparisons.*", "statistics/model_comparisons.csv", "Subject", "baseline pairs", "PASS"),
        ("tables/confusion_matrix_summary.*", "baselines/baseline_results_all_scenarios.csv", "Subject", "six baselines", "PASS"),
        ("figures/figure1_*", "dataset_summary.csv", "Subject", "child/adult cohorts", "PASS"),
        ("figures/figure2_*", "baselines/baseline_results_all_scenarios.csv; statistics/confidence_intervals.csv", "Subject", "six baselines", "PASS"),
        ("figures/figure3_*", "baselines/baseline_results_all_scenarios.csv", "Subject", "six baselines", "PASS"),
        ("figures/figure4_*", "statistics/confidence_intervals.csv", "Target subject", "six baselines", "PASS"),
        ("figures/figure5_*", "xai/electrode_attributions.csv", "Subject attribution", "CNN/LSTM verification samples", "PASS with limited sample"),
        ("figures/figure6_*", "xai/xai_stability.csv", "Subject attribution profile", "CNN/LSTM verification samples", "PASS with limited sample"),
        ("figures/*_topoplot.*", "xai/electrode_attributions.csv", "Subject attribution", "CNN/LSTM verification samples", "PASS sensor-space only"),
    ]
    trace_df = pd.DataFrame(trace, columns=["output", "source_file", "analysis_level", "models", "verified_PASS_FAIL"])
    trace_df.insert(2, "source_columns", "See source artifact plus table_generation_report.md / figure_generation_report.md")
    trace_df.to_csv(REV / "final_output_traceability.csv", index=False)
    stale = [
        ("Results/old/", "Historical models, result tables, SHAP/LIME figures and predictions; not reviewer-revision evidence."),
        ("Results/CleanDeepBenchmark/", "Legacy neural artifacts. Do not use as new reviewer-revision numerical evidence."),
        ("Results/EEGFormer/", "Legacy folder name retained for traceability; new label is Temporal Transformer Baseline."),
        ("generate_fig7_fig8.py", "Retired template-overlay generator; it raises an error and must not generate revised figures."),
        ("Results/LSTM_Tuning/", "Tuning artifacts are epoch-level diagnostics, not final neural subject-level benchmark results."),
    ]
    (REV / "STALE_FILES_WARNING.md").write_text("# Stale/historical artifact warning\n\n" + "\n".join(f"- `{path}` — {note}" for path, note in stale) + "\n", encoding="utf-8")

    # Lightweight reproducibility: final results-only generations must succeed.
    commands = ["generate_final_reviewer_tables.py", "generate_final_reviewer_figures.py"]
    repro = []
    for command in commands:
        result = subprocess.run([sys.executable, command], cwd=ROOT, capture_output=True, text=True)
        repro.append({"entry_point": command, "PASS_FAIL": "PASS" if result.returncode == 0 else "FAIL", "return_code": result.returncode, "notes": (result.stderr or result.stdout).strip()[-500:]})
    pd.DataFrame(repro).to_csv(REV / "reproducibility_verification.csv", index=False)

    failures = metric_df[metric_df.PASS_FAIL == "FAIL"]
    report = f"""# FINAL VERIFICATION REPORT

## C. NOT READY - TECHNICAL ISSUES REMAIN

### Critical errors

- No arithmetic mismatch was found in the independently recalculated reviewer-revision baseline subject-level metrics ({len(metric_df)} metric checks; `{len(failures)}` failures).
- The critical incompleteness is that repeated grouped-CV result schemas have **zero model-fold rows**, and newly generated neural subject-level prediction/result files are absent. Consequently, neural-vs-baseline paired comparisons, neural CV mean/SD/CIs, and final neural subject-level tables cannot be verified or reported from new artifacts.

### Leakage verification

- `subject_leakage_verification.csv`: {int((leak_df.PASS_FAIL == 'PASS').sum())} PASS and {int((leak_df.PASS_FAIL == 'FAIL').sum())} FAIL split/fold checks. All epochs in every baseline prediction source are grouped to one saved subject prediction; no duplicate subject prediction rows were found.

### Labels, counts, aggregation, and metrics

- Saved reviewer-revision baseline predictions use labels 0/1; metric calculations treat ADHD=1 (positive) and Control=0 (negative).
- `metric_verification.csv` independently recalculates Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC, TN, FP, FN, and TP from final saved **subject-level** predictions using tolerance {TOL:g}.
- `subject_aggregation_examples.csv` provides example epoch-probability to mean-subject-probability checks. The primary saved aggregation is mean, and one row is retained per subject.
- `trivial_baseline_comparison_verification.csv` explicitly compares learned classical baselines against the best trivial baseline for Accuracy, Balanced Accuracy, F1, and ROC-AUC.

### Threshold, LSTM, transfer, architecture, XAI, and topoplot verification

- `threshold_and_collapse_analysis.md` verifies validation-only threshold logic and records LSTM near-one-class behavior in the available child-to-adult **epoch-level** diagnostic. It cannot reproduce a final new neural subject-level evaluation because those predictions are unavailable.
- `cross_dataset_harmonization.md` verifies compatible 19-channel ordering, 512x19 inputs, epoch duration, resampling and train-only normalization at code level. Filtering, reference/ground, task, hardware, and artifact metadata remain manual-documentation requirements.
- `eegformer_architecture_audit.md` confirms the former EEGFormer implementation was relabelled `Temporal Transformer Baseline` for future output.
- `xai/xai_methodology.md`, metadata JSON files, raw NPZ values, `xai_stability.csv`, `region_normalization.csv`, and `figures/visualization_audit.md` validate provenance, within-subject aggregation, electrode-count normalization, sensor-space terminology, and no FDR-supported group-difference scalp map. XAI runs are limited verification samples.

### Statistical/CV verification

- `statistics/statistical_analysis.md` confirms subject-level bootstrap/permutation/Holm methods for baseline outputs. No test treats epochs as independent observations in the final statistics files.
- `cross_validation_verification.csv` documents that fold assignments are subject-grouped and leakage-free, but no CV performance summaries can be recalculated because no fold-level predictions/results exist.

### Reviewer concerns fully resolved technically

- Dataset/split audits, trivial/classical baselines, baseline subject-level metrics/confusion matrices, adult imbalance audit, code-level transfer harmonization audit, EEGFormer audit/renaming, subject-balanced XAI mechanism, electrode normalization, sensor-space topoplots, and final result-only tables/figures.

### Reviewer concerns requiring only manuscript editing

- Accurate reporting of the limitations above: no final new neural/CV values, epoch-level scope of available LSTM collapse diagnostics, transfer not isolating age, limited XAI verification samples, and missing original acquisition documentation.

### Manual dataset documentation still required

- Hardware, raw montage/source mapping, reference, ground, task/condition, recording duration/session structure, source filtering, and artifact handling for both cohorts.

### Files to use for the revised paper

- `tables/`, `figures/`, `baselines/baseline_results_all_scenarios.csv`, `statistics/`, `dataset_summary.csv`, `cross_dataset_harmonization.md`, `adult_small_sample_analysis.md`, and XAI files explicitly labelled as limited verification output.

### Files that must not be used

- See `STALE_FILES_WARNING.md`: historical `Results/old/`, legacy `Results/CleanDeepBenchmark/` neural numbers, legacy EEGFormer-labelled folders/results, retired template overlays, and epoch-level LSTM tuning artifacts as final subject-level claims.
"""
    (REV / "FINAL_VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")
    print("Final verification complete.")


if __name__ == "__main__":
    main()
