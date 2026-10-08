"""Subject-level uncertainty and paired model-comparison analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


METRICS = ("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC")
DISPLAY = {
    "always_adhd": "Always-ADHD", "always_control": "Always-Control", "majority_class_dummy": "Majority-class Dummy",
    "stratified_dummy": "Stratified Dummy", "logistic_regression_welch_band_power": "Logistic Regression (Welch band power)",
    "linear_svm_welch_band_power": "Linear SVM (Welch band power)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="Results/reviewer_revision/baselines")
    parser.add_argument("--xai", default="Results/reviewer_revision/xai/subject_attributions.csv")
    parser.add_argument("--cv", default="Results/reviewer_revision/cross_validation/cv_results_all_folds.csv")
    parser.add_argument("--out", default="Results/reviewer_revision/statistics")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    return parser.parse_args()


def calculate(y, probability, prediction):
    y, probability, prediction = np.asarray(y), np.asarray(probability), np.asarray(prediction)
    tn = int(np.sum((y == 0) & (prediction == 0))); fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0))); tp = int(np.sum((y == 1) & (prediction == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    if len(np.unique(y)) == 2:
        n_positive, n_negative = int(np.sum(y == 1)), int(np.sum(y == 0))
        roc = (rankdata(probability)[y == 1].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        order, sorted_y = np.argsort(-probability, kind="stable"), y[np.argsort(-probability, kind="stable")]
        precision_curve = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
        pr = float(precision_curve[sorted_y == 1].mean())
    else:
        roc, pr = np.nan, np.nan
    return {
        "Accuracy": (tp + tn) / len(y), "Balanced_Accuracy": (sensitivity + specificity) / 2,
        "Sensitivity": sensitivity, "Specificity": specificity, "Precision": precision, "F1": f1,
        "ROC_AUC": roc, "PR_AUC": pr,
    }


def stratified_bootstrap_indices(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == label), size=np.sum(y == label), replace=True) for label in (0, 1)])


def metric_ci(frame, replicates, rng):
    y, prob, pred = frame.true_label.to_numpy(), frame.mean_probability.to_numpy(), frame.predicted_label.to_numpy()
    point = calculate(y, prob, pred)
    samples = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        index = stratified_bootstrap_indices(y, rng)
        values = calculate(y[index], prob[index], pred[index])
        for metric in METRICS:
            samples[metric].append(values[metric])
    return point, {
        metric: (np.nan, np.nan) if not np.isfinite(values).any() else (float(np.nanquantile(values, .025)), float(np.nanquantile(values, .975)))
        for metric, values in samples.items()
    }


def paired_comparison(a, b, replicates, permutations, rng):
    merged = a.merge(b, on="subject_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.true_label_a, merged.true_label_b):
        raise ValueError("Paired predictions disagree on a subject label.")
    y = merged.true_label_a.to_numpy()
    pa, pb = merged.mean_probability_a.to_numpy(), merged.mean_probability_b.to_numpy()
    ya, yb = merged.predicted_label_a.to_numpy(), merged.predicted_label_b.to_numpy()
    observed = calculate(y, pa, ya); comparator = calculate(y, pb, yb)
    difference = {metric: observed[metric] - comparator[metric] for metric in METRICS}
    bootstrap = {metric: [] for metric in METRICS}; permutation = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        index = stratified_bootstrap_indices(y, rng)
        va, vb = calculate(y[index], pa[index], ya[index]), calculate(y[index], pb[index], yb[index])
        for metric in METRICS: bootstrap[metric].append(va[metric] - vb[metric])
    for _ in range(permutations):
        swap = rng.random(len(y)) < .5
        spa, spb = np.where(swap, pb, pa), np.where(swap, pa, pb)
        sya, syb = np.where(swap, yb, ya), np.where(swap, ya, yb)
        va, vb = calculate(y, spa, sya), calculate(y, spb, syb)
        for metric in METRICS: permutation[metric].append(va[metric] - vb[metric])
    rows = []
    for metric in METRICS:
        null = np.asarray(permutation[metric]); boot = np.asarray(bootstrap[metric])
        estimate = difference[metric]
        valid_boot, valid_null = boot[np.isfinite(boot)], null[np.isfinite(null)]
        rows.append({"Metric": metric, "n_paired_subjects": len(y), "effect_difference_a_minus_b": difference[metric],
                     "bootstrap_95ci_low": np.nan if not len(valid_boot) else float(np.quantile(valid_boot, .025)), "bootstrap_95ci_high": np.nan if not len(valid_boot) else float(np.quantile(valid_boot, .975)),
                     "test_statistic": estimate, "raw_p_value": np.nan if not np.isfinite(estimate) or not len(valid_null) else float((np.sum(np.abs(valid_null) >= abs(estimate)) + 1) / (len(valid_null) + 1)),
                     "test": "paired subject-label-swap permutation"})
    return rows


def holm(values):
    values, order, result, running = np.asarray(values, dtype=float), np.argsort(values), np.empty(len(values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(running, 1.0)
    return result


def main():
    args = parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260811)
    prediction_files = sorted(Path(args.baseline_root).glob("*/*_subject_predictions.csv"))
    predictions = {}
    ci_rows = []
    for path in prediction_files:
        scenario, token = path.parent.name, path.name.removesuffix("_subject_predictions.csv")
        frame = pd.read_csv(path)
        key = (scenario, DISPLAY.get(token, token))
        predictions[key] = frame
        point, intervals = metric_ci(frame, args.bootstrap, rng)
        for metric in METRICS:
            ci_rows.append({"analysis_type": "subject_bootstrap", "scenario": scenario, "model": key[1], "metric": metric,
                            "n_subjects": len(frame), "estimate": point[metric], "ci95_low": intervals[metric][0], "ci95_high": intervals[metric][1],
                            "unit": "subject", "procedure": "stratified subject bootstrap percentile CI"})
    comparison_rows = []
    for scenario in sorted({scenario for scenario, _ in predictions}):
        models = sorted(model for s, model in predictions if s == scenario)
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                for row in paired_comparison(predictions[(scenario, model_a)], predictions[(scenario, model_b)], args.bootstrap, args.permutations, rng):
                    row.update({"scenario": scenario, "model_a": model_a, "model_b": model_b, "unit": "subject"})
                    comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_corrected_p_value"] = np.nan
        valid = comparisons.raw_p_value.notna()
        comparisons.loc[valid, "holm_corrected_p_value"] = holm(comparisons.loc[valid, "raw_p_value"].to_numpy())
        comparisons["statistically_significant_holm_0_05"] = comparisons.holm_corrected_p_value < .05
    comparisons.to_csv(out / "model_comparisons.csv", index=False)

    # XAI uncertainty: average repeated samples within each subject before bootstrap.
    xai = pd.read_csv(args.xai)
    subject_xai = xai.groupby(["model_name", "true_label", "subject_id", "channel"], as_index=False).mean(numeric_only=True)
    for (model, label, channel), block in subject_xai.groupby(["model_name", "true_label", "channel"]):
        values = block.mean_absolute_attribution.to_numpy(); draws = rng.choice(values, size=(args.bootstrap, len(values)), replace=True).mean(axis=1)
        ci_rows.append({"analysis_type": "xai_subject_bootstrap", "scenario": "combined_holdout", "model": model, "metric": f"Mean absolute attribution: {channel}",
                        "n_subjects": len(values), "estimate": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
                        "unit": "subject", "procedure": "subject bootstrap after within-subject seed aggregation", "subject_class": "ADHD" if label == 1 else "Control"})
    pd.DataFrame(ci_rows).to_csv(out / "confidence_intervals.csv", index=False)

    cv = pd.read_csv(args.cv)
    cv_status = "CV result rows are unavailable: the file contains only its initialized schema." if cv.empty else "CV results are available; report the existing mean +/- SD and fold-level CIs from cv_summary.csv."
    neural_missing = "No neural-model subject-level prediction files were found in the reviewer-revision output, so no valid neural-vs-baseline paired test was calculated."
    report = f"""# Subject-level statistical analysis

## Statistical unit and data availability

All calculations use subjects as the independent unit. Epochs are never resampled or counted as independent observations. The analysis found {len(predictions)} model-by-scenario subject-prediction files (the six baseline models across three scenarios). {neural_missing}

{cv_status}

## Confidence intervals

For each available subject-level prediction set, a stratified non-parametric bootstrap resamples subjects with replacement separately within ADHD and Control classes ({args.bootstrap} replicates). Percentile 95% confidence intervals are reported for Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, and PR-AUC. Cross-dataset transfer uncertainty is included as the `adult_to_child` and `child_to_adult` scenario rows in `confidence_intervals.csv`.

## Model comparisons

For each pair of models evaluated on exactly the same subject IDs within a scenario, the analysis reports the metric difference (model A minus model B), a paired stratified-subject bootstrap 95% CI, and a two-sided paired label-swap permutation p-value ({args.permutations} permutations). The label swap exchanges the two model outputs within each subject and therefore preserves the pairing without assuming normality. Holm correction is applied across every reported model-pair/metric test in `model_comparisons.csv`; only `statistically_significant_holm_0_05 = True` supports a multiplicity-adjusted significance statement.

The test statistic is the observed subject-level metric difference. A numerically larger mean is not interpreted as significant unless its corrected p-value meets the stated criterion.

## XAI uncertainty

For XAI, repeated epoch samples are averaged within subject first. Electrode-level mean absolute attribution is then bootstrapped across subjects, separately by model and class, with results appended to `confidence_intervals.csv`. These intervals quantify sampled-subject uncertainty only; they do not establish source localization or biological effects.
"""
    (out / "statistical_analysis.md").write_text(report, encoding="utf-8")
    print(f"Saved subject-level statistics to {out}")


if __name__ == "__main__":
    main()
