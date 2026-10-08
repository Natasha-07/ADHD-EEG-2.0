# Cross-dataset harmonization audit: Adult-to-Child and Child-to-Adult transfer

## Interpretation rule

These experiments are **cross-cohort/cross-dataset transfer between adult and child datasets**. They do not isolate age: cohort membership is inseparable from dataset provenance and potentially from acquisition, task, session, hardware, and preprocessing differences. No generated project text containing the phrase "cross-age generalization" was found in the audit; the safer wording is used here and should be retained in future tables/captions.

## Dataset comparison

The machine-readable comparison is `Results/reviewer_revision/dataset_comparison_table.csv`. Fields marked `REQUIRES MANUAL DATASET DOCUMENTATION` were not found in the current project's CSV metadata, preprocessing scripts, or manuscript source and must be obtained from the original dataset documentation.

| Cohort | Age group | Subjects (ADHD / Control) | Benchmark epochs | Native rate used by code | Model-input rate/equivalent | Benchmark channels |
|---|---:|---:|---:|---:|---:|---:|
| Child dataset | child | 121 (61 / 60) | 4052 | 512 Hz | 256 Hz equivalent | 19 |
| Adult dataset | adult | 16 (8 / 8) | 9931 | 256 Hz | 256 Hz | 19 |

## Code-level harmonization audit

1. **Channel intersection:** `deep_model_utils.CHANNELS` selects the same ordered 19 columns from the already-harmonized combined CSV: `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz`. This verifies the benchmark input intersection, but cannot verify the raw-source intersection or any source-channel mapping because those mappings are not present.
2. **Channel ordering:** both cohorts are indexed with the one shared `CHANNELS` list. This is compatible at model input, conditional on the combined CSV having been correctly constructed.
3. **Sampling-rate conversion:** `load_epoch_dataset()` sets child `fs=512` and adult `fs=256`. Two-second child windows have 1024 samples and are Fourier-resampled to 512 (`scipy.signal.resample`); adult two-second windows already have 512 samples. Both reach the model with shape `(512, 19)`. This is a code-level shape/time-grid verification, not proof of acquisition equivalence.
4. **Filtering:** no filtering step is implemented in the active dataset loader or deep benchmark preprocessing. Any source filtering is REQUIRES MANUAL DATASET DOCUMENTATION.
5. **Referencing:** no rereferencing step is implemented in the active loader. Source reference status is REQUIRES MANUAL DATASET DOCUMENTATION.
6. **Epoch duration:** both cohorts use 2-second windows with 50% overlap. Child windows are resampled after windowing.
7. **Normalization:** active deep models fit a per-channel `StandardScaler` on training epochs only and apply it unchanged to validation/test. The LSTM tuning path may instead use per-epoch channel z-score; this is a configuration difference that must be reported for each run.
8. **Artifact handling:** no artifact detection/rejection/correction is implemented in the active loader. Source artifact handling is REQUIRES MANUAL DATASET DOCUMENTATION.

## Compatibility conclusion

The current code mathematically delivers compatible **tensor dimensions** and a common nominal 256-Hz-equivalent time grid: child and adult inputs both enter neural models as 512 x 19 arrays, and normalization avoids train/test leakage. It does **not** establish physiological or acquisition comparability because hardware, source montage, reference/ground, task, duration/session design, raw filtering, artifact handling, and raw channel mapping are undocumented in this repository. Therefore a transfer failure must be reported as limited cross-cohort/cross-dataset transfer, not as an age effect.
