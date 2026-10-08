"""Generate publication-quality figures only from reviewer-revision outputs."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REV = ROOT / "Results" / "reviewer_revision"
OUT = REV / "figures"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def subject_distribution():
    data = pd.read_csv(REV / "dataset_summary.csv")
    data = data[data.cohort != "all"].set_index("cohort").loc[["child", "adult"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.5), constrained_layout=True)
    positions = np.arange(len(data))
    ax.bar(positions, data.control_subjects, label="Control", color="#4C78A8")
    ax.bar(positions, data.adhd_subjects, bottom=data.control_subjects, label="ADHD", color="#E45756")
    for x, control, adhd in zip(positions, data.control_subjects, data.adhd_subjects):
        ax.text(x, control / 2, str(int(control)), ha="center", va="center", color="white", fontweight="bold")
        ax.text(x, control + adhd / 2, str(int(adhd)), ha="center", va="center", color="white", fontweight="bold")
        ax.text(x, control + adhd + 3, f"N={int(control + adhd)}", ha="center", fontweight="bold")
    ax.set_xticks(positions, ["Child cohort", "Adult cohort"])
    ax.set_ylabel("Subjects")
    ax.set_title("Subject distribution by cohort and diagnosis")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_ylim(0, 140)
    save(fig, "figure1_dataset_subject_distribution")


def main_performance():
    results = pd.read_csv(REV / "baselines" / "baseline_results_all_scenarios.csv")
    ci = pd.read_csv(REV / "statistics" / "confidence_intervals.csv")
    results = results[(results.Scenario == "combined_holdout") & (results.Evaluation_Level == "Subject")].copy()
    metric = "Balanced_Accuracy"
    ci = ci[(ci.analysis_type == "subject_bootstrap") & (ci.scenario == "combined_holdout") & (ci.metric == metric)]
    results = results.merge(ci[["model", "ci95_low", "ci95_high"]], left_on="Model", right_on="model", how="left")
    results = results.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.3, 3.8), constrained_layout=True)
    y = np.arange(len(results)); values = results[metric].to_numpy()
    err = np.vstack([values - results.ci95_low.to_numpy(), results.ci95_high.to_numpy() - values])
    ax.barh(y, values, color="#4C78A8", xerr=err, capsize=3, error_kw={"ecolor": "#222222", "lw": 1})
    ax.axvline(.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks(y, results.Model)
    ax.set_xlabel("Subject-level balanced accuracy (bootstrap 95% CI)")
    ax.set_xlim(0, 1)
    ax.set_title("Main subject-level baseline performance (combined holdout; N=28)")
    ax.text(.505, len(results) - .35, "Chance reference", color="#555555", fontsize=9, ha="left", va="center")
    save(fig, "figure2_main_subject_level_performance")


def confusion_matrices():
    data = pd.read_csv(REV / "baselines" / "baseline_results_all_scenarios.csv")
    data = data[(data.Scenario == "combined_holdout") & (data.Evaluation_Level == "Subject")]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("reviewblue", ["#FFFFFF", "#4C78A8"])
    for axis, (_, row) in zip(axes.flat, data.iterrows()):
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=float)
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=14)
        for (r, c), value in np.ndenumerate(matrix): axis.text(c, r, str(int(value)), ha="center", va="center", fontsize=12, fontweight="bold")
        axis.set_title(row.Model, fontsize=10, fontweight="bold")
        axis.set_xticks([0, 1], ["Control", "ADHD"]); axis.set_yticks([0, 1], ["Control", "ADHD"])
        axis.set_xlabel("Predicted"); axis.set_ylabel("True")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72, pad=.01); colorbar.set_label("Subjects")
    fig.suptitle("Subject-level confusion matrices: combined holdout (N=28)", fontsize=14, fontweight="bold")
    save(fig, "figure3_subject_level_confusion_matrices")


def transfer_performance():
    ci = pd.read_csv(REV / "statistics" / "confidence_intervals.csv")
    ci = ci[(ci.analysis_type == "subject_bootstrap") & (ci.metric == "Balanced_Accuracy") & (ci.scenario.isin(["adult_to_child", "child_to_adult"]))]
    order = ["Always-ADHD", "Always-Control", "Majority-class Dummy", "Stratified Dummy", "Logistic Regression (Welch band power)", "Linear SVM (Welch band power)"]
    ci["model"] = pd.Categorical(ci.model, order, ordered=True); ci = ci.sort_values("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.8), sharey=True, constrained_layout=True)
    mapping = [("adult_to_child", "Adult dataset -> Child dataset\n(target N=121)"), ("child_to_adult", "Child dataset -> Adult dataset\n(target N=16)")]
    for axis, (scenario, title) in zip(axes, mapping):
        block = ci[ci.scenario == scenario]; x = np.arange(len(block)); values = block.estimate.to_numpy()
        errors = np.vstack([values - block.ci95_low.to_numpy(), block.ci95_high.to_numpy() - values])
        axis.bar(x, values, color="#72B7B2", yerr=errors, capsize=2, error_kw={"ecolor": "#222", "lw": .8})
        axis.axhline(.5, color="#666", linestyle="--", linewidth=1)
        axis.set_xticks(x, ["A-ADHD", "A-Control", "Majority", "Stratified", "Logistic", "Linear SVM"], rotation=45, ha="right")
        axis.set_ylim(0, 1); axis.set_title(title); axis.set_ylabel("Subject-level balanced accuracy")
    fig.suptitle("Cross-cohort/cross-dataset transfer (bootstrap 95% CI)", fontsize=14, fontweight="bold")
    save(fig, "figure4_cross_dataset_transfer")


def electrode_attributions():
    data = pd.read_csv(REV / "xai" / "electrode_attributions.csv")
    channels = data.channel.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.0), sharey=False, constrained_layout=True)
    for axis, (model, block) in zip(axes, data.groupby("model_name", sort=True)):
        pivot = block.pivot(index="channel", columns="subject_class", values="mean_absolute_attribution").reindex(channels)
        low = block.pivot(index="channel", columns="subject_class", values="bootstrap_95ci_low").reindex(channels)
        high = block.pivot(index="channel", columns="subject_class", values="bootstrap_95ci_high").reindex(channels)
        x = np.arange(len(channels)); width = .38
        for offset, label, color in [(-width / 2, "Control", "#4C78A8"), (width / 2, "ADHD", "#E45756")]:
            values = pivot[label].to_numpy(); errors = np.vstack([values - low[label].to_numpy(), high[label].to_numpy() - values])
            axis.bar(x + offset, values, width, label=label, color=color, yerr=errors, capsize=1.5, error_kw={"lw": .65})
        axis.set_title(f"{model}: sensor-level attribution")
        axis.set_xticks(x, channels, rotation=90); axis.set_ylabel("Mean absolute Integrated Gradients")
        axis.legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Aggregate sensor-level electrode attribution (subject bootstrap 95% CI)", fontsize=13, fontweight="bold")
    save(fig, "figure5_aggregate_xai_electrode_attribution")


def xai_stability():
    data = pd.read_csv(REV / "xai" / "xai_stability.csv")
    data["label"] = data.apply(lambda row: f"{row.model_name} | {row.subject_class}\n{row.stability_type.replace('_', ' ')}", axis=1)
    fig, ax = plt.subplots(figsize=(7.3, 4.4), constrained_layout=True)
    colors = np.where(data.stability_type == "between_subjects", "#F58518", "#54A24B")
    ax.barh(np.arange(len(data)), data.mean_spearman_rho, color=colors)
    ax.set_yticks(np.arange(len(data)), data.label)
    ax.set_xlim(0, 1); ax.set_xlabel("Spearman correlation of 19-electrode attribution profiles")
    ax.set_title("XAI attribution stability across subjects and repeated samples")
    ax.axvline(.5, color="#666", linestyle="--", linewidth=1)
    ax.legend([plt.Rectangle((0,0),1,1,color="#F58518"), plt.Rectangle((0,0),1,1,color="#54A24B")], ["Between subjects", "Repeated epoch samples/seeds"], frameon=False, loc="lower right")
    save(fig, "figure6_xai_stability")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_distribution(); main_performance(); confusion_matrices(); transfer_performance(); electrode_attributions(); xai_stability()
    # Existing reviewer-revision sensor-space topoplots are valid generated outputs; retain them as figure set 7.
    sources = [
        ("figure1_dataset_subject_distribution", "dataset_summary.csv", "Child and adult cohorts", "Subject", "Subject-level", "None; descriptive counts."),
        ("figure2_main_subject_level_performance", "baselines/baseline_results_all_scenarios.csv; statistics/confidence_intervals.csv", "Six newly run baselines", "Subject", "Subject-level", "Stratified subject-bootstrap 95% CI."),
        ("figure3_subject_level_confusion_matrices", "baselines/baseline_results_all_scenarios.csv", "Six newly run baselines", "Subject", "Subject-level", "None; observed held-out subject counts."),
        ("figure4_cross_dataset_transfer", "statistics/confidence_intervals.csv", "Six newly run baselines", "Subject", "Subject-level", "Stratified subject-bootstrap 95% CI."),
        ("figure5_aggregate_xai_electrode_attribution", "xai/electrode_attributions.csv", "Saved CNN and LSTM attribution runs", "Subject after within-subject epoch/seed aggregation", "Subject-level attribution", "Subject-bootstrap 95% CI; small verification samples, not confirmatory."),
        ("figure6_xai_stability", "xai/xai_stability.csv", "Saved CNN and LSTM attribution runs", "Subject", "Subject-level attribution profile", "No error bars; each bar is the recorded mean pairwise Spearman stability and sample sizes are limited."),
        ("Sensor-space topoplots", "figures/*_topoplot.png/pdf; xai/electrode_attributions.csv", "Saved CNN and LSTM attribution runs", "Subject after within-subject aggregation", "Subject-level attribution", "Within each model, ADHD, Control, and overall maps use a consistent scale. No group-difference map was supported after FDR correction."),
    ]
    table = "| Figure | Data source | Models/cohorts included | Statistical unit | Analysis level | Uncertainty/error bars |\n|---|---|---|---|---|---|\n" + "\n".join("| " + " | ".join(row) + " |" for row in sources)
    report = """# Final reviewer-revision figure generation report

Only reviewer-revision outputs were used. No manuscript file was edited.

""" + table + """

## Figures not generated

- **Repeated grouped-CV performance:** not plotted because `cross_validation/cv_results_all_folds.csv` and `cv_summary.csv` contain initialized schemas, not fold results.
- **Neural-model comparison/error bars:** not plotted because no newly generated reviewer-revision neural subject-level performance records exist.
- **Neural probability distributions:** not plotted because the only available threshold-collapse diagnostic uses epoch-level data. This figure set contains no epoch-level performance/probability panel.
- **ADHD-minus-Control scalp maps:** not plotted because no electrode survived FDR-controlled permutation testing in `topoplot_difference_statistics.csv`.
- **Physical brain/source-space figures:** not generated. The existing topoplots are explicitly sensor/scalp-space displays, not source localization.
"""
    (OUT / "figure_generation_report.md").write_text(report, encoding="utf-8")
    print(f"Saved final figures to {OUT}")


if __name__ == "__main__":
    main()
