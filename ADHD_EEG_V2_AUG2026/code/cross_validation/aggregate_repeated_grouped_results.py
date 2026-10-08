"""Aggregate repeated subject-wise folds with uncertainty and paired model tests.

Each input CSV must contain one row per model and repeat/fold at Evaluation_Level=Subject.
This utility intentionally summarizes subjects, never the correlated epochs within them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["Balanced_Accuracy", "Specificity", "Sensitivity", "Precision", "F1-score", "ROC-AUC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Fold-level results CSV")
    parser.add_argument("--out", default="Results/Result Tables/repeated_grouped_summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["Evaluation_Level"].eq("Subject")].copy()
    required = {"Model", "Seed", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    summary = df.groupby("Model", dropna=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(args.out, index=False)
    # Paired Wilcoxon tests retain the repeat/fold pairing rather than treating epochs as independent.
    tests = []
    models = sorted(df["Model"].unique())
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            paired = df[df["Model"].eq(left)].merge(
                df[df["Model"].eq(right)], on="Seed", suffixes=("_left", "_right")
            )
            for metric in METRICS:
                values = paired[[f"{metric}_left", f"{metric}_right"]].dropna()
                if len(values) >= 5 and (values.iloc[:, 0] != values.iloc[:, 1]).any():
                    statistic, p_value = wilcoxon(values.iloc[:, 0], values.iloc[:, 1])
                else:
                    statistic, p_value = float("nan"), float("nan")
                tests.append({"Model_A": left, "Model_B": right, "Metric": metric,
                              "Paired_Folds": len(values), "Wilcoxon_Statistic": statistic, "P_Value": p_value})
    pd.DataFrame(tests).to_csv(Path(args.out).with_name("paired_model_tests.csv"), index=False)


if __name__ == "__main__":
    main()
