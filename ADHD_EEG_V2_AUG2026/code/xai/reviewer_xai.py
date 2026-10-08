"""Subject-balanced Integrated Gradients for saved raw-EEG Keras models.

This tool explains a saved benchmark model; it never trains a replacement or
converts raw epochs to spectral features.  It is intentionally model-agnostic
for Keras models with an input layout of (time, channels) and sigmoid output.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr

from deep_model_utils import CHANNELS, fit_channel_standardizer, load_epoch_dataset, transform_epochs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", default="Results/CleanDeepBenchmark/combined_holdout/CNN/cnn_model.keras")
    parser.add_argument("--manifest", default="Results/CleanDeepBenchmark/combined_holdout/CNN/split_manifest.json")
    parser.add_argument("--model-name", default="CNN")
    parser.add_argument("--dataset", default="combined_eeg")
    parser.add_argument("--csv", default="CSV Combined and Individual/Single_CSV/combined_eeg.csv")
    parser.add_argument("--out", default="Results/reviewer_revision/xai")
    parser.add_argument("--subjects-per-class", type=int, default=5)
    parser.add_argument("--epochs-per-subject", type=int, default=8)
    parser.add_argument("--seeds", default="42,31415", help="Comma-separated epoch-sampling seeds.")
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--append", action="store_true", help="Append CSV summaries for another model to an existing output directory.")
    return parser.parse_args()


def integrated_gradients(model: tf.keras.Model, epochs: np.ndarray, baseline: np.ndarray, steps: int) -> np.ndarray:
    """Batched attribution of the ADHD sigmoid output to raw EEG epochs."""
    x = tf.convert_to_tensor(epochs, dtype=tf.float32)
    base = tf.convert_to_tensor(baseline, dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    path = base[None, None, ...] + alphas[None, :, None, None] * (x[:, None, ...] - base[None, None, ...])
    flat_path = tf.reshape(path, [-1, tf.shape(x)[1], tf.shape(x)[2]])
    with tf.GradientTape() as tape:
        tape.watch(flat_path)
        output = model(flat_path, training=False)
        target = tf.reshape(output, [-1])
    gradients = tf.reshape(tape.gradient(target, flat_path), tf.shape(path))
    # `gradients` has shape (epochs, integration points, time, channels).
    average_gradient = (gradients[:, :-1] + gradients[:, 1:]) / 2.0
    return ((x - base[None, ...]) * tf.reduce_mean(average_gradient, axis=1)).numpy()


def ci_mean(values: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else np.nan


def write_csv(path: Path, frame: pd.DataFrame, append: bool) -> None:
    """Append compatible model-level outputs without replacing prior models."""
    if append and path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model_file, manifest_file = Path(args.model_file), Path(args.manifest)
    if not model_file.exists() or not manifest_file.exists():
        raise FileNotFoundError("Both --model-file and --manifest must identify existing benchmark artifacts.")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for required in ("train_subjects", "test_subjects", "seed", "epoch_seconds", "overlap"):
        if required not in manifest:
            raise ValueError(f"Split manifest lacks required field: {required}")

    dataset = load_epoch_dataset(args.csv, epoch_seconds=int(manifest["epoch_seconds"]), overlap=float(manifest["overlap"]))
    train_subject_ids = np.array([str(row["subject_id"]) for row in manifest["train_subjects"]])
    test_subject_rows = manifest["test_subjects"]
    train_mask = np.isin(dataset.subject_ids, train_subject_ids)
    scaler = fit_channel_standardizer(dataset.X[train_mask])
    X = transform_epochs(dataset.X, scaler)
    baseline = X[train_mask].mean(axis=0).astype(np.float32)

    model = tf.keras.models.load_model(model_file, compile=False)
    expected_shape = tuple(model.input_shape[1:])
    if tuple(X.shape[1:]) != expected_shape:
        raise ValueError(f"Model expects {expected_shape}; reconstructed epochs have {tuple(X.shape[1:])}.")
    if model.output_shape[-1] != 1:
        raise ValueError("This runner currently requires a binary model with one sigmoid output for ADHD = 1.")

    # The chosen held-out subjects are stratified by their manifest label and fixed
    # by manifest order; seeds resample epochs only, so subject comparisons are paired.
    selected = []
    for label_name, label_value in (("ADHD", 1), ("Control", 0)):
        candidates = [str(row["subject_id"]) for row in test_subject_rows if str(row["label"]) == label_name]
        if not candidates:
            raise ValueError(f"No held-out {label_name} subjects in manifest.")
        selected.extend((sid, label_value) for sid in candidates[: args.subjects_per_class])
    selected_subject_ids = [subject_id for subject_id, _ in selected]
    run_metadata = {
        "model_name": args.model_name,
        "model_file": str(model_file),
        "model_input_representation": "training-standardized raw EEG epoch (time x 19 channels)",
        "dataset": args.dataset,
        "subjects_explained": selected_subject_ids,
        "xai_method": "Integrated Gradients",
        "baseline_background_data": "mean of training epochs after training-only channel standardization",
        "output_class_explained": "ADHD positive class (sigmoid output = 1)",
        "split_manifest": str(manifest_file),
        "integration_steps": args.integration_steps,
        "epochs_per_subject": args.epochs_per_subject,
        "sampling_seeds": args.seeds,
    }
    safe_name = "".join(character.lower() if character.isalnum() else "_" for character in args.model_name).strip("_")
    (outdir / f"xai_run_metadata_{safe_name}.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 2:
        raise ValueError("Use at least two --seeds values to quantify sampling stability.")
    raw_attrs, raw_subjects, raw_labels, raw_seeds, raw_epoch_indices = [], [], [], [], []
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for subject_id, label in selected:
            epoch_indices = np.flatnonzero(dataset.subject_ids == subject_id)
            sampled = rng.choice(epoch_indices, size=min(args.epochs_per_subject, len(epoch_indices)), replace=False)
            epoch_attrs = integrated_gradients(model, X[sampled], baseline, args.integration_steps)
            probabilities = model(X[sampled], training=False).numpy().ravel()
            for index, attr in zip(sampled, epoch_attrs):
                raw_attrs.append(attr.astype(np.float32))
                raw_subjects.append(subject_id)
                raw_labels.append(label)
                raw_seeds.append(seed)
                raw_epoch_indices.append(int(index))
            subject_attr = np.mean(epoch_attrs, axis=0)  # epochs first: one vector per subject
            for channel_idx, channel in enumerate(CHANNELS):
                rows.append({
                    "model_name": args.model_name, "model_file": str(model_file),
                    "model_input_representation": "training-standardized raw EEG epoch (time x 19 channels)",
                    "dataset": args.dataset, "subjects_explained": subject_id, "subject_id": subject_id, "true_label": label,
                    "xai_method": "Integrated Gradients", "baseline_background_data": "mean of training epochs after training-only channel standardization",
                    "output_class_explained": "ADHD positive class (sigmoid output = 1)",
                    "sampling_seed": seed, "number_of_available_epochs": len(epoch_indices),
                    "number_of_explained_epochs": len(sampled), "channel": channel,
                    "mean_signed_attribution": float(subject_attr[:, channel_idx].mean()),
                    "mean_absolute_attribution": float(np.abs(subject_attr[:, channel_idx]).mean()),
                    "mean_epoch_probability": float(np.mean(probabilities)),
                })

    subject_df = pd.DataFrame(rows)
    write_csv(outdir / "subject_attributions.csv", subject_df, args.append)
    np.savez_compressed(
        outdir / f"raw_epoch_attributions_{safe_name}.npz", attributions=np.asarray(raw_attrs, dtype=np.float32),
        subject_ids=np.asarray(raw_subjects), true_labels=np.asarray(raw_labels), sampling_seeds=np.asarray(raw_seeds),
        dataset_epoch_indices=np.asarray(raw_epoch_indices), channel_names=np.asarray(CHANNELS),
        metadata_json=np.asarray(json.dumps(run_metadata)),
    )

    # Average seed replicates within subject before any cohort-level result.
    per_subject = subject_df.groupby(["subject_id", "true_label", "channel"], as_index=False).agg(
        mean_absolute_attribution=("mean_absolute_attribution", "mean"),
        mean_signed_attribution=("mean_signed_attribution", "mean"),
    )
    electrode_rows = []
    boot_rng = np.random.default_rng(int(manifest["seed"]))
    for label, label_name in ((0, "Control"), (1, "ADHD")):
        class_subjects = per_subject.loc[per_subject.true_label == label, "subject_id"].drop_duplicates().tolist()
        for channel in CHANNELS:
            values = per_subject[(per_subject.true_label == label) & (per_subject.channel == channel)]["mean_absolute_attribution"].to_numpy()
            mean, low, high = ci_mean(values, args.bootstrap_replicates, boot_rng)
            electrode_rows.append({
                "model_name": args.model_name, "model_file": str(model_file),
                "model_input_representation": "training-standardized raw EEG epoch (time x 19 channels)",
                "dataset": args.dataset, "subjects_explained": ";".join(class_subjects), "number_of_subjects_explained": int(len(values)), "xai_method": "Integrated Gradients",
                "baseline_background_data": "mean of training epochs after training-only channel standardization",
                "output_class_explained": "ADHD positive class (sigmoid output = 1)",
                "subject_class": label_name, "channel": channel, "mean_absolute_attribution": mean,
                "bootstrap_95ci_low": low, "bootstrap_95ci_high": high,
            })
    write_csv(outdir / "electrode_attributions.csv", pd.DataFrame(electrode_rows), args.append)

    stability_rows = []
    for label, label_name in ((0, "Control"), (1, "ADHD")):
        subset = per_subject[per_subject.true_label == label]
        matrix = subset.pivot(index="subject_id", columns="channel", values="mean_absolute_attribution").reindex(columns=CHANNELS)
        pairs = [correlation(matrix.loc[a].to_numpy(), matrix.loc[b].to_numpy()) for a, b in combinations(matrix.index, 2)]
        stability_rows.append({"stability_type": "between_subjects", "subject_class": label_name, "subjects_explained": ";".join(matrix.index.tolist()), "n_subjects": len(matrix), "n_pairs": len(pairs), "mean_spearman_rho": float(np.nanmean(pairs)) if pairs else np.nan, "sampling_seeds": args.seeds})
    seed_matrix = subject_df.pivot_table(index=["subject_id", "true_label", "sampling_seed"], columns="channel", values="mean_absolute_attribution").reindex(columns=CHANNELS)
    for subject_id, label in selected:
        vectors = [seed_matrix.loc[(subject_id, label, seed)].to_numpy() for seed in seeds]
        pairs = [correlation(a, b) for a, b in combinations(vectors, 2)]
        stability_rows.append({"stability_type": "across_repeated_epoch_samples", "subject_id": subject_id, "subject_class": "ADHD" if label else "Control", "subjects_explained": subject_id, "n_subjects": 1, "n_pairs": len(pairs), "mean_spearman_rho": float(np.nanmean(pairs)), "sampling_seeds": args.seeds})
    stability = pd.DataFrame(stability_rows)
    for field, value in {"model_name": args.model_name, "model_file": str(model_file), "model_input_representation": "training-standardized raw EEG epoch (time x 19 channels)", "dataset": args.dataset, "xai_method": "Integrated Gradients", "baseline_background_data": "mean of training epochs after training-only channel standardization", "output_class_explained": "ADHD positive class (sigmoid output = 1)"}.items():
        stability[field] = value
    write_csv(outdir / "xai_stability.csv", stability, args.append)
    print(f"Saved subject-balanced raw-EEG attributions to {outdir}")


if __name__ == "__main__":
    main()
