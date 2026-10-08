"""Diagnose threshold use and one-class collapse without selecting on test data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from subject_level_eval_utils import compute_binary_metrics, select_threshold


ROOT = Path(__file__).resolve().parent
TUNING_ROOT = ROOT / "Results" / "LSTM_Tuning"
OUT = ROOT / "Results" / "reviewer_revision"
FIG_DIR = OUT / "threshold_diagnostics"


def reference_trial(scenario_dir: Path) -> Path | None:
    """Use prespecified fixed-threshold trial when available; never test-rank trials."""
    candidates = sorted(path for path in scenario_dir.iterdir() if path.is_dir() and "fixed_threshold" in path.name)
    return candidates[0] if candidates else None


def plot_probabilities(y_true: np.ndarray, probabilities: np.ndarray, scenario: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(probabilities[y_true == 0], bins=25, range=(0, 1), alpha=0.7, label="Control (0)", color="#1f77b4")
    axis.hist(probabilities[y_true == 1], bins=25, range=(0, 1), alpha=0.7, label="ADHD (1)", color="#d62728")
    axis.axvline(0.5, color="black", linestyle="--", label="Default threshold 0.50")
    axis.set(xlabel="LSTM epoch probability for ADHD", ylabel="Epoch count", title=f"LSTM probability distribution: {scenario}")
    axis.legend()
    figure.tight_layout()
    path = FIG_DIR / f"lstm_{scenario}_probability_histogram.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_calibration(y_true: np.ndarray, probabilities: np.ndarray, scenario: str) -> tuple[Path, float]:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
    axis.plot(predicted, observed, marker="o", label="LSTM")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed ADHD frequency", title=f"LSTM calibration: {scenario}")
    axis.legend()
    figure.tight_layout()
    path = FIG_DIR / f"lstm_{scenario}_calibration.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path, float(brier_score_loss(y_true, probabilities))


def class_counts_from_manifest(trial: Path) -> dict[str, int | str]:
    manifest = json.loads((trial / "split_manifest.json").read_text(encoding="utf-8"))
    if "class_counts" in manifest:
        counts = manifest["class_counts"]
        return {
            "train": f"ADHD={counts['Train_Subjects_ADHD']}, Control={counts['Train_Subjects_Control']}; epochs ADHD={counts['Train_Epochs_ADHD']}, Control={counts['Train_Epochs_Control']}",
            "val": f"ADHD={counts['Val_Subjects_ADHD']}, Control={counts['Val_Subjects_Control']}; epochs ADHD={counts['Val_Epochs_ADHD']}, Control={counts['Val_Epochs_Control']}",
        }
    subject_counts = pd.read_csv(OUT / "subject_epoch_counts.csv").set_index("subject_id")

    def describe(partition: str) -> str:
        entries = manifest[f"{partition}_subjects"]
        ids = [entry["subject_id"] for entry in entries]
        frame = subject_counts.loc[ids]
        return (
            f"ADHD={(frame['label'] == 'ADHD').sum()}, Control={(frame['label'] == 'Control').sum()}; "
            f"epochs ADHD={frame.loc[frame['label'] == 'ADHD', 'n_epochs'].sum()}, "
            f"Control={frame.loc[frame['label'] == 'Control', 'n_epochs'].sum()}"
        )
    return {
        "train": describe("train"),
        "val": describe("val"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    narrative = [
        "# Threshold and LSTM-collapse analysis", "",
        "## Threshold protocol repair", "",
        "The active evaluation code now always reports the prespecified 0.50 threshold. Optional optimized thresholds are selected only from validation **subject** mean probabilities using either Youden J or maximum balanced accuracy, then frozen before test evaluation. Test F1, test precision, and all other test metrics are not threshold-selection inputs.",
        "",
        "## Available-artifact diagnosis", "",
        "The existing CleanDeepBenchmark neural folders contain metrics/checkpoints but no epoch-prediction CSVs. Therefore their probability histograms, calibration, and subject-level collapse diagnosis cannot be reconstructed honestly. Updated neural runs will write the required epoch and subject prediction files. The diagnostics below use the prespecified fixed-threshold LSTM tuning artifacts only; their saved predictions lack subject IDs, so these are epoch-level diagnostic plots, not subject-level estimates.",
        "",
    ]
    for scenario_dir in sorted(path for path in TUNING_ROOT.iterdir() if path.is_dir()):
        trial = reference_trial(scenario_dir)
        if trial is None:
            narrative.append(f"### {scenario_dir.name}\n\nNo fixed-threshold LSTM reference trial was found.\n")
            continue
        val = pd.read_csv(trial / "val_predictions.csv")
        test = pd.read_csv(trial / "test_predictions.csv")
        val_selection = select_threshold(val["y_true"], val["y_prob"], metric_name="balanced_accuracy")
        default_metrics = compute_binary_metrics(test["y_true"], test["y_prob"], threshold=0.5)
        frozen_metrics = compute_binary_metrics(test["y_true"], test["y_prob"], threshold=val_selection.threshold)
        predicted_adhd = float((test["y_prob"] >= 0.5).mean())
        histogram = plot_probabilities(test["y_true"].to_numpy(), test["y_prob"].to_numpy(), scenario_dir.name)
        calibration, brier = plot_calibration(test["y_true"].to_numpy(), test["y_prob"].to_numpy(), scenario_dir.name)
        config = pd.read_csv(trial / "config.csv").iloc[0].to_dict()
        counts = class_counts_from_manifest(trial)
        collapse = predicted_adhd >= 0.95 or predicted_adhd <= 0.05
        records.append({
            "scenario": scenario_dir.name, "reference_trial": trial.name,
            "default_threshold": 0.5, "validation_balanced_accuracy_threshold": val_selection.threshold,
            "test_predicted_adhd_pct_default": 100 * predicted_adhd,
            "test_predicted_control_pct_default": 100 * (1 - predicted_adhd),
            "default_sensitivity": default_metrics["Sensitivity"], "default_specificity": default_metrics["Specificity"],
            "default_balanced_accuracy": default_metrics["Balanced_Accuracy"], "default_confusion_matrix": default_metrics["Confusion_Matrix"],
            "frozen_validation_threshold_sensitivity": frozen_metrics["Sensitivity"], "frozen_validation_threshold_specificity": frozen_metrics["Specificity"],
            "frozen_validation_threshold_balanced_accuracy": frozen_metrics["Balanced_Accuracy"], "frozen_validation_threshold_confusion_matrix": frozen_metrics["Confusion_Matrix"],
            "brier_score": brier, "near_one_class_collapse_default": collapse,
        })
        narrative.extend([
            f"## LSTM: {scenario_dir.name}", "",
            f"- Reference artifact: `{trial.relative_to(ROOT)}` (chosen by fixed-threshold name, not test performance).",
            f"- Training balance: {counts['train']}",
            f"- Validation balance: {counts['val']}",
            f"- Config: class weights `{config.get('class_weight_mode')}`; normalization `{config.get('normalization_mode')}`; loss is binary cross-entropy; output activation is sigmoid; labels are ADHD=1 and Control=0.",
            f"- Default 0.50: predicted ADHD={100 * predicted_adhd:.2f}%, Control={100 * (1-predicted_adhd):.2f}%; sensitivity={default_metrics['Sensitivity']:.4f}; specificity={default_metrics['Specificity']:.4f}; balanced accuracy={default_metrics['Balanced_Accuracy']:.4f}; confusion matrix={default_metrics['Confusion_Matrix']}.",
            f"- Validation-only maximum-balanced-accuracy threshold={val_selection.threshold:.6f}; frozen-test balanced accuracy={frozen_metrics['Balanced_Accuracy']:.4f}; confusion matrix={frozen_metrics['Confusion_Matrix']}. This is reported diagnostically, not selected using test results.",
            f"- Brier score={brier:.4f}; figures: `{histogram.relative_to(ROOT)}`, `{calibration.relative_to(ROOT)}`.",
            f"- Collapse flag at default threshold: **{'YES' if collapse else 'NO'}** (defined as >=95% one predicted class).",
            "",
        ])
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "threshold_diagnostics_summary.csv", index=False)
    narrative.extend([
        "## Interpretation and safeguards", "",
        "A high ADHD-prediction percentage accompanied by near-zero specificity is evidence of a practical one-class collapse at the specified threshold, even when F1 is superficially high. Class weighting changes the fitted loss emphasis and can shift probability distributions; it must not be compensated by choosing a test-derived threshold. Calibration is assessed descriptively by Brier score and reliability plots, not corrected using test labels.",
        "",
        "No model architecture, loss, output activation, or class weight was changed in this analysis to improve test results. Any future change must be chosen from training/validation evidence and evaluated once on frozen test subjects.",
    ])
    (OUT / "threshold_and_collapse_analysis.md").write_text("\n".join(narrative) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"Saved: {OUT / 'threshold_and_collapse_analysis.md'}")


if __name__ == "__main__":
    main()
