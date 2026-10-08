"""Document adult/child dataset comparability without inferring missing acquisition metadata."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REVISION = ROOT / "Results" / "reviewer_revision"
OUT_CSV = REVISION / "dataset_comparison_table.csv"
OUT_MD = REVISION / "cross_dataset_harmonization.md"
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
MISSING = "REQUIRES MANUAL DATASET DOCUMENTATION"


def cohort_row(cohort: str, frame: pd.DataFrame) -> dict[str, object]:
    adhd = frame[frame["label"] == "ADHD"]
    control = frame[frame["label"] == "Control"]
    child = cohort == "Child dataset"
    return {
        "cohort": cohort,
        "age_group": "child" if child else "adult",
        "adhd_subjects": len(adhd),
        "control_subjects": len(control),
        "total_subjects": len(frame),
        "total_epochs_after_benchmark_epoching": int(frame["n_epochs"].sum()),
        "EEG_hardware": MISSING,
        "electrode_montage_source": MISSING,
        "reference_electrode": MISSING,
        "ground_electrode": MISSING,
        "number_of_channels_benchmark_input": len(CHANNELS),
        "native_sampling_rate_used_by_code_hz": 512 if child else 256,
        "resampled_rate_or_equivalent_hz": 256,
        "recording_task_condition": MISSING,
        "recording_duration": MISSING,
        "session_structure": MISSING,
        "preprocessing_differences_documented_in_code": "2-s epoching, 50% overlap; child epochs are Fourier-resampled from 1024 to 512 samples" if child else "2-s epoching, 50% overlap; adult 512-sample epochs are not temporally resampled",
        "available_common_channels": ", ".join(CHANNELS),
    }


def main() -> None:
    subjects = pd.read_csv(REVISION / "subject_epoch_counts.csv")
    children = subjects[subjects["age_group"] == "child"]
    adults = subjects[subjects["age_group"] == "adult"]
    table = pd.DataFrame([cohort_row("Child dataset", children), cohort_row("Adult dataset", adults)])
    table.to_csv(OUT_CSV, index=False)
    common = ", ".join(CHANNELS)
    report = f"""# Cross-dataset harmonization audit: Adult-to-Child and Child-to-Adult transfer

## Interpretation rule

These experiments are **cross-cohort/cross-dataset transfer between adult and child datasets**. They do not isolate age: cohort membership is inseparable from dataset provenance and potentially from acquisition, task, session, hardware, and preprocessing differences. No generated project text containing the phrase "cross-age generalization" was found in the audit; the safer wording is used here and should be retained in future tables/captions.

## Dataset comparison

The machine-readable comparison is `Results/reviewer_revision/dataset_comparison_table.csv`. Fields marked `{MISSING}` were not found in the current project's CSV metadata, preprocessing scripts, or manuscript source and must be obtained from the original dataset documentation.

| Cohort | Age group | Subjects (ADHD / Control) | Benchmark epochs | Native rate used by code | Model-input rate/equivalent | Benchmark channels |
|---|---:|---:|---:|---:|---:|---:|
| Child dataset | child | {len(children)} ({(children['label'] == 'ADHD').sum()} / {(children['label'] == 'Control').sum()}) | {int(children['n_epochs'].sum())} | 512 Hz | 256 Hz equivalent | 19 |
| Adult dataset | adult | {len(adults)} ({(adults['label'] == 'ADHD').sum()} / {(adults['label'] == 'Control').sum()}) | {int(adults['n_epochs'].sum())} | 256 Hz | 256 Hz | 19 |

## Code-level harmonization audit

1. **Channel intersection:** `deep_model_utils.CHANNELS` selects the same ordered 19 columns from the already-harmonized combined CSV: `{common}`. This verifies the benchmark input intersection, but cannot verify the raw-source intersection or any source-channel mapping because those mappings are not present.
2. **Channel ordering:** both cohorts are indexed with the one shared `CHANNELS` list. This is compatible at model input, conditional on the combined CSV having been correctly constructed.
3. **Sampling-rate conversion:** `load_epoch_dataset()` sets child `fs=512` and adult `fs=256`. Two-second child windows have 1024 samples and are Fourier-resampled to 512 (`scipy.signal.resample`); adult two-second windows already have 512 samples. Both reach the model with shape `(512, 19)`. This is a code-level shape/time-grid verification, not proof of acquisition equivalence.
4. **Filtering:** no filtering step is implemented in the active dataset loader or deep benchmark preprocessing. Any source filtering is {MISSING}.
5. **Referencing:** no rereferencing step is implemented in the active loader. Source reference status is {MISSING}.
6. **Epoch duration:** both cohorts use 2-second windows with 50% overlap. Child windows are resampled after windowing.
7. **Normalization:** active deep models fit a per-channel `StandardScaler` on training epochs only and apply it unchanged to validation/test. The LSTM tuning path may instead use per-epoch channel z-score; this is a configuration difference that must be reported for each run.
8. **Artifact handling:** no artifact detection/rejection/correction is implemented in the active loader. Source artifact handling is {MISSING}.

## Compatibility conclusion

The current code mathematically delivers compatible **tensor dimensions** and a common nominal 256-Hz-equivalent time grid: child and adult inputs both enter neural models as 512 x 19 arrays, and normalization avoids train/test leakage. It does **not** establish physiological or acquisition comparability because hardware, source montage, reference/ground, task, duration/session design, raw filtering, artifact handling, and raw channel mapping are undocumented in this repository. Therefore a transfer failure must be reported as limited cross-cohort/cross-dataset transfer, not as an age effect.
"""
    OUT_MD.write_text(report, encoding="utf-8")
    print(table[["cohort", "age_group", "adhd_subjects", "control_subjects", "total_epochs_after_benchmark_epoching"]].to_string(index=False))
    print(f"Saved {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
