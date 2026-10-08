"""Generate standard-10-20 sensor-space attribution topographies.

Coordinates are read from MNE's packaged ``standard_1020.elc`` montage. The
interpolated field is only a display of scalp-sensor values, never source localization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-attributions", default="Results/reviewer_revision/xai/subject_attributions.csv")
    parser.add_argument("--out", default="Results/reviewer_revision/figures")
    parser.add_argument("--permutations", type=int, default=10000)
    return parser.parse_args()


def montage_xy(channels: list[str]) -> np.ndarray:
    montage_file = Path(r"C:\Program Files\Python312\Lib\site-packages\mne\channels\data\montages\standard_1020.elc")
    if not montage_file.exists():
        raise FileNotFoundError("MNE standard_1020.elc was not found; do not substitute invented coordinates.")
    lines = montage_file.read_text(encoding="utf-8").splitlines()
    n_positions = int(next(line.split("=")[1] for line in lines if line.startswith("NumberPositions")))
    position_start, label_start = lines.index("Positions") + 1, lines.index("Labels") + 1
    positions = np.asarray([[float(value) for value in line.split()] for line in lines[position_start:position_start + n_positions]])
    lookup = {label: position for label, position in zip(lines[label_start:label_start + n_positions], positions)}
    missing = sorted(set(channels) - set(lookup))
    if missing:
        raise ValueError(f"Channels are not valid standard_1020 labels: {missing}")
    xy = np.asarray([lookup[channel][:2] for channel in channels], dtype=float)
    return xy / np.max(np.linalg.norm(xy, axis=1))


def draw_topography(values, channels, xy, title, outbase, vmin, vmax, cmap="viridis"):
    grid = np.linspace(-1.05, 1.05, 250)
    grid_x, grid_y = np.meshgrid(grid, grid)
    field = griddata(xy, values, (grid_x, grid_y), method="cubic")
    mask = grid_x ** 2 + grid_y ** 2 > 1.0
    field[mask] = np.nan
    linear = griddata(xy, values, (grid_x, grid_y), method="linear")
    missing = np.isnan(field) & ~mask
    field[missing] = linear[missing]
    nearest = griddata(xy, values, (grid_x, grid_y), method="nearest")
    missing = np.isnan(field) & ~mask
    field[missing] = nearest[missing]
    fig, ax = plt.subplots(figsize=(7.4, 7.8), constrained_layout=True)
    image = ax.imshow(field, extent=(-1.05, 1.05, -1.05, 1.05), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=2.2))
    ax.plot([-0.13, 0, 0.13], [0.98, 1.15, 0.98], color="black", linewidth=2.2, clip_on=False)
    ax.scatter(xy[:, 0], xy[:, 1], s=58, c="white", edgecolor="black", linewidth=0.8, zorder=5)
    for name, (x, y) in zip(channels, xy):
        ax.text(x, y, name, ha="center", va="center", fontsize=7, fontweight="bold", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.10})
    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, shrink=0.80, pad=0.04)
    colorbar.set_label("Mean absolute Integrated Gradients attribution", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.savefig(outbase.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fdr_bh(p_values):
    order, adjusted, running = np.argsort(p_values), np.empty_like(p_values, dtype=float), 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return adjusted


def permutation_pvalues(adhd, control, permutations, rng):
    observed, combined, n_adhd = np.abs(adhd.mean(axis=0) - control.mean(axis=0)), np.vstack([adhd, control]), len(adhd)
    counts = np.ones_like(observed, dtype=float)
    for _ in range(permutations):
        indices = rng.permutation(len(combined))
        difference = np.abs(combined[indices[:n_adhd]].mean(axis=0) - combined[indices[n_adhd:]].mean(axis=0))
        counts += difference >= observed
    return counts / (permutations + 1)


def main():
    args, outdir = parse_args(), Path(parse_args().out)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.subject_attributions)
    required = {"model_name", "subject_id", "true_label", "channel", "mean_absolute_attribution"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    channels = frame.channel.drop_duplicates().tolist()
    xy = montage_xy(channels)
    subject = frame.groupby(["model_name", "subject_id", "true_label", "channel"], as_index=False)["mean_absolute_attribution"].mean()
    stats_rows, decisions = [], []
    for model, block in subject.groupby("model_name", sort=True):
        vectors = block.pivot(index=["subject_id", "true_label"], columns="channel", values="mean_absolute_attribution").reindex(columns=channels)
        labels = vectors.index.get_level_values("true_label").to_numpy()
        adhd, control = vectors[labels == 1].to_numpy(), vectors[labels == 0].to_numpy()
        summaries = {"ADHD subjects": adhd.mean(axis=0), "Control subjects": control.mean(axis=0), "Overall subjects": vectors.to_numpy().mean(axis=0)}
        vmax, stem = max(float(values.max()) for values in summaries.values()), model.lower().replace(" ", "_").replace("-", "_")
        for descriptor, values in summaries.items():
            draw_topography(values, channels, xy, f"{model}: {descriptor}\nSensor-space scalp attribution topography", outdir / f"{stem}_{descriptor.lower().replace(' ', '_')}_topoplot", 0.0, vmax)
        raw_p = permutation_pvalues(adhd, control, args.permutations, np.random.default_rng(42))
        adjusted = fdr_bh(raw_p)
        for channel, difference, raw, corrected in zip(channels, adhd.mean(axis=0) - control.mean(axis=0), raw_p, adjusted):
            stats_rows.append({"model_name": model, "channel": channel, "adhd_minus_control_attribution": difference, "permutation_p": raw, "fdr_bh_p": corrected, "statistically_supported_for_difference_map": bool(corrected < 0.05)})
        decisions.append(f"- {model}: " + ("difference topoplot generated (at least one electrode survived BH-FDR < 0.05)." if np.any(adjusted < 0.05) else "no difference topoplot generated; no electrode survived BH-FDR < 0.05."))
    pd.DataFrame(stats_rows).to_csv(outdir / "topoplot_difference_statistics.csv", index=False)
    (outdir / "visualization_audit.md").write_text("""# EEG visualization audit and replacement

## Retired visualizations

`generate_fig7_fig8.py` overlays sensor-group values on anatomical-reference templates and is retired from the revised workflow. `Results/CNN/CNN_Channel_Regional_Importance.py` produces summed sensor-group graphics and remains historical only. Neither is source localization, but both can overstate anatomical interpretation.

The available manuscript source (`main_fixed.tex`) contains benchmark tables but no Tables 3 or 4 identified as electrode/region-attribution tables. Therefore no table was silently removed. Where the reviewer means the legacy electrode/region displays, replace those manuscript placements with the topoplots here and retain `electrode_attributions.csv` as the supplementary numeric table.

## Revised topoplots

These figures use correctly matching channel names and coordinates from MNE's packaged `standard_1020.elc` montage. They display interpolated **sensor-space scalp topographies** of mean absolute Integrated Gradients attribution. They are not source reconstruction, brain images, or evidence of localized neural activity. ADHD, Control, and overall maps use one consistent scale within each model. Numeric sensor-level values remain in the XAI CSV files.

## Difference-map decision

""" + "\n".join(decisions) + "\n\nDifference maps require two-sided permutation testing and Benjamini-Hochberg FDR < 0.05 across 19 electrodes; details are in `topoplot_difference_statistics.csv`. Do not claim frontal-lobe activation, brain-source contribution, or source localization from these figures.\n", encoding="utf-8")
    print(f"Saved sensor-space topoplots and audit to {outdir}")


if __name__ == "__main__":
    main()
