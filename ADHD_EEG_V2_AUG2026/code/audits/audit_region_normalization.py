"""Compare summed versus per-electrode-normalized sensor-group attributions."""
from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


REGION_MAP = {
    "sensor-level frontal electrode group": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central electrode group": ["C3", "C4", "Cz"],
    "temporal electrode group": ["T7", "T8"],
    "parietal electrode group": ["P3", "P4", "P7", "P8", "Pz"],
    "occipital electrode group": ["O1", "O2"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--electrode", default="Results/reviewer_revision/xai/electrode_attributions.csv")
    parser.add_argument("--out", default="Results/reviewer_revision/xai")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.electrode)
    attributions = pd.read_csv(input_path)
    required = {"model_name", "subject_class", "channel", "mean_absolute_attribution"}
    missing = required - set(attributions.columns)
    if missing:
        raise ValueError(f"Missing required attribution columns: {sorted(missing)}")

    mapped = [channel for channels in REGION_MAP.values() for channel in channels]
    observed = set(attributions.channel.unique())
    if len(mapped) != len(set(mapped)) or set(mapped) != observed:
        raise ValueError("The mapping must contain every observed electrode exactly once.")

    rows, report_sections = [], []
    for (model, subject_class), block in attributions.groupby(["model_name", "subject_class"], sort=True):
        by_channel = block.set_index("channel")["mean_absolute_attribution"]
        records = []
        for region, channels in REGION_MAP.items():
            values = by_channel.loc[channels]
            records.append({
                "model_name": model,
                "subject_class": subject_class,
                "region": region,
                "electrodes": ";".join(channels),
                "number_of_electrodes": len(channels),
                "total_sensor_level_attribution": float(values.sum()),
                "mean_attribution_per_electrode": float(values.mean()),
            })
        regional = pd.DataFrame(records)
        regional["total_attribution_percent"] = 100 * regional.total_sensor_level_attribution / regional.total_sensor_level_attribution.sum()
        regional["per_electrode_normalized_percent"] = 100 * regional.mean_attribution_per_electrode / regional.mean_attribution_per_electrode.sum()
        regional["total_attribution_rank"] = regional.total_sensor_level_attribution.rank(ascending=False, method="min").astype(int)
        regional["per_electrode_normalized_rank"] = regional.mean_attribution_per_electrode.rank(ascending=False, method="min").astype(int)
        regional["rank_change_after_normalization"] = (regional.total_attribution_rank - regional.per_electrode_normalized_rank).abs()
        # Pre-specified practical flag: either top group changes or any group moves >=2 places.
        total_top = regional.loc[regional.total_attribution_rank == 1, "region"].iloc[0]
        normalized_top = regional.loc[regional.per_electrode_normalized_rank == 1, "region"].iloc[0]
        substantial = bool(total_top != normalized_top or (regional.rank_change_after_normalization >= 2).any())
        regional["ranking_changes_substantially_after_normalization"] = substantial
        rows.extend(regional.to_dict("records"))

        total_order = " > ".join(regional.sort_values("total_attribution_rank")["region"].str.replace(" electrode group", "", regex=False))
        mean_order = " > ".join(regional.sort_values("per_electrode_normalized_rank")["region"].str.replace(" electrode group", "", regex=False))
        report_sections.append(
            f"### {model} - {subject_class} subjects\n\n"
            f"- Sum-ranked sensor groups: {total_order}.\n"
            f"- Per-electrode-normalized ranking: {mean_order}.\n"
            f"- Substantial ranking change: **{'yes' if substantial else 'no'}** (pre-specified: a changed top group or a movement of at least two ranks).\n"
        )

    result = pd.DataFrame(rows).sort_values(["model_name", "subject_class", "total_attribution_rank", "region"])
    result.to_csv(outdir / "region_normalization.csv", index=False)
    mapping = "\n".join(f"- {region}: {len(channels)} scalp electrodes ({', '.join(channels)})" for region, channels in REGION_MAP.items())
    report = f"""# Sensor-electrode group normalization audit

## Mapping and scope

This audit aggregates the subject-balanced, absolute Integrated Gradients values in `electrode_attributions.csv`. It concerns **scalp-electrode attribution**, not brain-source localization. Each listed group is a descriptive sensor grouping:\n\n{mapping}

The mapping contains 19 unique electrodes exactly once. No claim is made that a sensor-level frontal electrode group measures frontal-lobe activity, or that any group represents a localized neural source.

## Methods

For every model and subject class, this audit reports both:

1. **Total sensor-level attribution**: the sum across electrodes in a group. This answers how much total attribution is assigned to all sensors in that group, but increases mechanically with group size.
2. **Mean attribution per electrode**: total attribution divided by the number of electrodes. The `per_electrode_normalized_percent` column compares these group means after scaling them to sum to 100% across the five groups. This is the appropriate comparison for group importance independent of electrode count.

`ranking_changes_substantially_after_normalization` is true when the top-ranked group changes or any group moves at least two ranks. It is a descriptive diagnostic, not a statistical test.

## Results

{''.join(report_sections)}
## Audit of prior aggregation code

- `Results/CNN/CNN_Channel_Regional_Importance.py`, `update_adhd_sample_importance.py`, and `generate_fig7_fig8.py` sum channel importances within each group and then convert those sums to percentages. They do **not** divide by electrode count, so their regional comparisons are size-biased.
- `update_excel_summary.py` preserves these summed regional percentages in report tables.
- The historical scripts also use terms such as "brain region," "regional contribution," and "Brain Region." These should be replaced in future outputs with "sensor-level electrode group" and "scalp-electrode attribution." Historical artifacts are not changed by this audit.

Use `region_normalization.csv` for any regional comparison, show both total and per-electrode-normalized results, and report whenever the rankings differ. Do not infer brain-source contribution, regional activation, or causality from these scalp-sensor attributions.
"""
    (outdir / "region_bias_analysis.md").write_text(report, encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Saved {outdir / 'region_normalization.csv'} and {outdir / 'region_bias_analysis.md'}")


if __name__ == "__main__":
    main()
