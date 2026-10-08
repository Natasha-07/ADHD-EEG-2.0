# FINAL VERIFICATION REPORT

## C. NOT READY - TECHNICAL ISSUES REMAIN

### Critical errors

- No arithmetic mismatch was found in the independently recalculated reviewer-revision baseline subject-level metrics (216 metric checks; `0` failures).
- The critical incompleteness is that repeated grouped-CV result schemas have **zero model-fold rows**, and newly generated neural subject-level prediction/result files are absent. Consequently, neural-vs-baseline paired comparisons, neural CV mean/SD/CIs, and final neural subject-level tables cannot be verified or reported from new artifacts.

### Leakage verification

- `subject_leakage_verification.csv`: 48 PASS and 0 FAIL split/fold checks. All epochs in every baseline prediction source are grouped to one saved subject prediction; no duplicate subject prediction rows were found.

### Labels, counts, aggregation, and metrics

- Saved reviewer-revision baseline predictions use labels 0/1; metric calculations treat ADHD=1 (positive) and Control=0 (negative).
- `metric_verification.csv` independently recalculates Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC, TN, FP, FN, and TP from final saved **subject-level** predictions using tolerance 5e-06.
- `subject_aggregation_examples.csv` provides example epoch-probability to mean-subject-probability checks. The primary saved aggregation is mean, and one row is retained per subject.
- `trivial_baseline_comparison_verification.csv` explicitly compares learned classical baselines against the best trivial baseline for Accuracy, Balanced Accuracy, F1, and ROC-AUC.

### Threshold, LSTM, transfer, architecture, XAI, and topoplot verification

- `threshold_and_collapse_analysis.md` verifies validation-only threshold logic and records LSTM near-one-class behavior in the available child-to-adult **epoch-level** diagnostic. It cannot reproduce a final new neural subject-level evaluation because those predictions are unavailable.
- `cross_dataset_harmonization.md` verifies compatible 19-channel ordering, 512x19 inputs, epoch duration, resampling and train-only normalization at code level. Filtering, reference/ground, task, hardware, and artifact metadata remain manual-documentation requirements.
- `eegformer_architecture_audit.md` confirms the former EEGFormer implementation was relabelled `Temporal Transformer Baseline` for future output.
- `xai/xai_methodology.md`, metadata JSON files, raw NPZ values, `xai_stability.csv`, `region_normalization.csv`, and `figures/visualization_audit.md` validate provenance, within-subject aggregation, electrode-count normalization, sensor-space terminology, and no FDR-supported group-difference scalp map. XAI runs are limited verification samples.

### Statistical/CV verification

- `statistics/statistical_analysis.md` confirms subject-level bootstrap/permutation/Holm methods for baseline outputs. No test treats epochs as independent observations in the final statistics files.
- `cross_validation_verification.csv` documents that fold assignments are subject-grouped and leakage-free, but no CV performance summaries can be recalculated because no fold-level predictions/results exist.

### Reviewer concerns fully resolved technically

- Dataset/split audits, trivial/classical baselines, baseline subject-level metrics/confusion matrices, adult imbalance audit, code-level transfer harmonization audit, EEGFormer audit/renaming, subject-balanced XAI mechanism, electrode normalization, sensor-space topoplots, and final result-only tables/figures.

### Reviewer concerns requiring only manuscript editing

- Accurate reporting of the limitations above: no final new neural/CV values, epoch-level scope of available LSTM collapse diagnostics, transfer not isolating age, limited XAI verification samples, and missing original acquisition documentation.

### Manual dataset documentation still required

- Hardware, raw montage/source mapping, reference, ground, task/condition, recording duration/session structure, source filtering, and artifact handling for both cohorts.

### Files to use for the revised paper

- `tables/`, `figures/`, `baselines/baseline_results_all_scenarios.csv`, `statistics/`, `dataset_summary.csv`, `cross_dataset_harmonization.md`, `adult_small_sample_analysis.md`, and XAI files explicitly labelled as limited verification output.

### Files that must not be used

- See `STALE_FILES_WARNING.md`: historical `Results/old/`, legacy `Results/CleanDeepBenchmark/` neural numbers, legacy EEGFormer-labelled folders/results, retired template overlays, and epoch-level LSTM tuning artifacts as final subject-level claims.
