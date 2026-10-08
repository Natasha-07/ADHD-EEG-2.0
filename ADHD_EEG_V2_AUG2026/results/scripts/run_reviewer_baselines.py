"""Run trivial and classical baselines on the benchmark's exact subject splits.

The module is intentionally separate from neural-model training. It uses the shared
``build_experiment_split`` implementation so all partitions match the benchmark for
the same CSV, scenario, seed, test size, and validation size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from deep_model_utils import (
    SCENARIO_DETAILS,
    build_experiment_split,
    load_epoch_dataset,
    split_epoch_dataset,
    write_split_manifest,
)
from subject_level_eval_utils import subject_probability_table


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "Results" / "reviewer_revision" / "baselines"
SUBJECT_LEVEL_OUT = ROOT / "Results" / "reviewer_revision" / "subject_level" / "baselines"
BANDS = (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-disjoint ADHD EEG baseline benchmark.")
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--scenario", choices=["all", *SCENARIO_DETAILS], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epoch-seconds", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def spectral_bandpower_features(X: np.ndarray, fs: int = 256) -> np.ndarray:
    """Return log absolute Welch band power for every channel and canonical band.

    Input epochs are already harmonized to 512 samples by ``load_epoch_dataset``.
    Each epoch yields 19 x 5 = 95 features. No labels or population statistics are
    used here; StandardScaler is fit later on training features only.
    """
    features = np.empty((len(X), X.shape[-1] * len(BANDS)), dtype=np.float32)
    for index, epoch in enumerate(X):
        freqs, psd = welch(epoch, fs=fs, axis=0, nperseg=min(fs, len(epoch)))
        values = []
        for _, low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            band_power = np.trapezoid(psd[mask], freqs[mask], axis=0)
            values.extend(np.log10(band_power + 1e-12))
        features[index] = values
    return features


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute complete binary metrics, retaining all confusion-matrix cells."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Confusion_Matrix": f"[[{tn} {fp}],[{fn} {tp}]]",
    }


def positive_scores(estimator, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an ADHD ranking score and its decision threshold for an estimator."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = estimator.classes_
        if 1 not in classes:
            return np.zeros(len(X), dtype=float), 0.5
        return probabilities[:, np.flatnonzero(classes == 1)[0]], 0.5
    # LinearSVC has no probability calibration. Its signed margin is valid for
    # ROC/PR ranking and has a natural 0 decision boundary.
    return estimator.decision_function(X).ravel(), 0.0


def estimator_definitions(seed: int) -> dict[str, object]:
    return {
        "Always-ADHD": DummyClassifier(strategy="constant", constant=1),
        "Always-Control": DummyClassifier(strategy="constant", constant=0),
        "Majority-class Dummy": DummyClassifier(strategy="most_frequent"),
        "Stratified Dummy": DummyClassifier(strategy="stratified", random_state=seed),
        "Logistic Regression (Welch band power)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "Linear SVM (Welch band power)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)),
        ]),
    }


def result_row(model: str, scenario: str, level: str, metrics: dict[str, float | int], threshold: float, split, aggregation: str = "not_applicable") -> dict[str, object]:
    return {
        "Model": model,
        "Model_Family": "Baseline",
        "Scenario": scenario,
        "Evaluation_Level": level,
        "Feature_Representation": "None (trivial classifier)" if "Dummy" in model or model.startswith("Always") else "95 log Welch band-power features (5 bands x 19 channels)",
        "Threshold": threshold,
        "Aggregation": aggregation,
        "Train_Subjects": len(split.train_subjects),
        "Val_Subjects": len(split.val_subjects),
        "Test_Subjects": len(split.test_subjects),
        **metrics,
    }


def run_scenario(dataset, args: argparse.Namespace, scenario: str) -> list[dict[str, object]]:
    outdir = args.out / scenario
    subject_outdir = SUBJECT_LEVEL_OUT / scenario
    outdir.mkdir(parents=True, exist_ok=True)
    subject_outdir.mkdir(parents=True, exist_ok=True)
    split = build_experiment_split(dataset, scenario, args.test_size, args.val_size, args.seed)
    write_split_manifest(outdir, dataset, split, args.seed, args.epoch_seconds, args.overlap, args.val_size, args.test_size)
    X_train, X_val, X_test, y_train, y_val, y_test = split_epoch_dataset(dataset, split)
    # Validation features are persisted for transparency but not used to select a
    # threshold or hyperparameter in this fixed baseline comparison.
    train_features = spectral_bandpower_features(X_train)
    val_features = spectral_bandpower_features(X_val)
    test_features = spectral_bandpower_features(X_test)
    np.savez_compressed(outdir / "feature_audit.npz", train_shape=train_features.shape, val_shape=val_features.shape, test_shape=test_features.shape)

    rows: list[dict[str, object]] = []
    test_subject_ids = dataset.subject_ids[split.test_idx]
    for name, estimator in estimator_definitions(args.seed).items():
        estimator.fit(train_features, y_train)
        test_scores, threshold = positive_scores(estimator, test_features)
        epoch = binary_metrics(y_test, test_scores, threshold)
        subject_predictions = subject_probability_table(
            y_test, test_scores, test_subject_ids,
            dataset=dataset.age_groups[split.test_idx], primary_aggregation="mean", threshold=threshold,
        )
        subject = binary_metrics(subject_predictions["true_label"], subject_predictions["mean_probability"], threshold)
        sensitivity = binary_metrics(subject_predictions["true_label"], subject_predictions["median_probability"], threshold)
        rows.extend((
            result_row(name, scenario, "Epoch", epoch, threshold, split),
            result_row(name, scenario, "Subject", subject, threshold, split, aggregation="mean"),
            result_row(name, scenario, "Subject_Sensitivity", sensitivity, threshold, split, aggregation="median"),
        ))
        pd.DataFrame({"subject_id": test_subject_ids, "y_true": y_test, "y_score": test_scores}).to_csv(
            outdir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_epoch_predictions.csv", index=False
        )
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        subject_predictions.to_csv(outdir / f"{safe_name}_subject_predictions.csv", index=False)
        subject_predictions.to_csv(subject_outdir / f"{safe_name}_subject_level_predictions.csv", index=False)
    scenario_results = pd.DataFrame(rows)
    scenario_results.to_csv(outdir / "baseline_metrics.csv", index=False, float_format="%.6f")
    return rows


def neural_rows() -> pd.DataFrame:
    """Load saved neural metrics without fabricating fields absent from artifacts."""
    rows = []
    root = ROOT / "Results" / "CleanDeepBenchmark"
    for path in sorted(root.glob("*/*/*_metrics.csv")):
        frame = pd.read_csv(path)
        for _, record in frame.iterrows():
            data = record.to_dict()
            cm = str(data.get("Confusion_Matrix", ""))
            # Saved artifacts do not carry PR-AUC or separate confusion cells.
            rows.append({
                "Model": data.get("Model", path.parent.name), "Model_Family": "Existing neural",
                "Scenario": data.get("Scenario", path.parents[1].name),
                "Evaluation_Level": data.get("Evaluation_Level", "Unavailable"),
                "Feature_Representation": "Raw EEG epochs", "Accuracy": data.get("Accuracy", np.nan),
                "Balanced_Accuracy": data.get("Balanced_Accuracy", np.nan),
                "Sensitivity": data.get("Sensitivity", data.get("Recall", np.nan)),
                "Specificity": data.get("Specificity", np.nan), "Precision": data.get("Precision", np.nan),
                "F1": data.get("F1", data.get("F1-score", np.nan)), "ROC_AUC": data.get("ROC_AUC", data.get("ROC-AUC", np.nan)),
                "PR_AUC": np.nan, "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan,
                "Threshold": data.get("Threshold", np.nan), "Source_Artifact": str(path.relative_to(ROOT)),
                "Notes": f"Saved neural artifact; PR-AUC and TN/FP/FN/TP unavailable. Confusion_Matrix={cm}",
            })
    return pd.DataFrame(rows)


def write_methodology(outdir: Path) -> None:
    outdir.joinpath("baseline_methodology.md").write_text(
        """# Reviewer-revision baselines

All baseline scenarios call `deep_model_utils.build_experiment_split` with the
same CSV, seed, validation size, test size, and scenario names as the deep
benchmark. Each output scenario includes its exact `split_manifest.json`.

Learned models use 95 features per epoch: log10 Welch absolute band power in
delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma
(30-45 Hz), calculated for each of the 19 harmonized channels after the shared
loader has converted inputs to 512 samples at the 256-Hz-equivalent length.
`StandardScaler` is fit only on training features inside the Logistic Regression
and Linear SVM pipelines; validation/test features are transformed with the
fitted training scaler. The fixed decision threshold is 0.5 for probability
models and 0.0 for the LinearSVC signed margin.

Subject-level scores are the mean of a held-out subject's epoch scores. Existing
neural artifacts are copied into the comparison CSV exactly as saved; unavailable
PR-AUC and separate TN/FP/FN/TP fields are left blank rather than inferred.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    dataset = load_epoch_dataset(args.csv, args.epoch_seconds, args.overlap)
    scenarios = list(SCENARIO_DETAILS) if args.scenario == "all" else [args.scenario]
    baseline_rows = [row for scenario in scenarios for row in run_scenario(dataset, args, scenario)]
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(args.out / "baseline_results_all_scenarios.csv", index=False, float_format="%.6f")
    write_methodology(args.out)

    comparison = pd.concat([baseline_df, neural_rows()], ignore_index=True, sort=False)
    comparison.to_csv(args.out / "baseline_neural_comparison.csv", index=False, float_format="%.6f")
    print("\nBaseline results (epoch and subject level; no results hidden)")
    display_columns = ["Scenario", "Model", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC", "TN", "FP", "FN", "TP"]
    print(baseline_df[["Evaluation_Level", *display_columns]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved baseline and neural comparison tables to: {args.out}")


if __name__ == "__main__":
    main()
