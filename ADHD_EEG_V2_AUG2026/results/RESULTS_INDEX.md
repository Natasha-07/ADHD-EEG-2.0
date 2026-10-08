# Reproducible technical results index

## Package scope

This package contains copied final reviewer-revision technical artifacts only. It contains no manuscript text and is intended to support reproducible inspection of the revised analyses.

The `scripts/` directory contains the copied final generation/audit scripts used for dataset auditing, baselines, grouped-CV design, XAI, normalization, statistics, tables, figures, transfer/harmonization, and the retained temporal-transformer baseline. It is included to make the package reproducible rather than results-only.

## Dataset, splits, and cohort balance

| What was measured | Statistical unit | Package output | Corresponding table / figure | Reviewer concern addressed |
|---|---|---|---|---|
| Cohort subject/class counts, epoch counts, and epoch-per-subject distribution | Subject and epoch (reported separately) | `dataset_summary.csv`; `subject_epoch_counts.csv`; `dataset_audit.md` | `tables/table1_dataset_characteristics.*`; `figures/figure1_dataset_subject_distribution.*` | Dataset class distributions; adult/child subject and epoch imbalance; avoid treating epochs as independent. |
| Train/validation/test class composition and subject overlap checks | Subject, with epoch counts descriptive | `split_summary.csv` | Supporting audit artifact | Subject-disjoint splitting and leakage prevention. |
| Cross-cohort acquisition/preprocessing comparability | Cohort / dataset | `dataset_comparison_table.csv`; `cross_dataset_harmonization.md` | Table 1 rate/channel fields; Table 4 context | Cross-dataset harmonization; avoid interpreting transfer as isolated age effect. |

## Baselines and subject-level performance

| What was measured | Statistical unit | Package output | Corresponding table / figure | Reviewer concern addressed |
|---|---|---|---|---|
| Always-ADHD, Always-Control, majority/stratified dummy, Logistic Regression, and Linear SVM performance on subject-disjoint splits | Subject; one mean epoch probability per subject for learned models | `baselines/baseline_results_all_scenarios.csv`; `baselines/*/*_subject_predictions.csv`; `subject_level/baselines/` | `tables/table2_main_subject_level_performance.*`; `figures/figure2_main_subject_level_performance.*` | Trivial/classical baselines; subject-level aggregation; balanced accuracy; specificity. |
| Subject-level confusion counts and diagnostic measures | Subject | `tables/confusion_matrix_summary.*`; source data in `baselines/baseline_results_all_scenarios.csv` | `figures/figure3_subject_level_confusion_matrices.*` | Confusion matrices, TN/FP/FN/TP, sensitivity, specificity, balanced accuracy. |
| Adult-to-child and child-to-adult transfer performance | Target-dataset subject | `baselines/baseline_results_all_scenarios.csv`; `subject_level/baselines/adult_to_child/`; `subject_level/baselines/child_to_adult/` | `tables/table4_cross_dataset_transfer.*`; `figures/figure4_cross_dataset_transfer.*` | Cross-dataset transfer, not isolated age effect; transfer uncertainty. |
| LSTM threshold/collapse diagnostic | Epoch for the available tuning diagnostics; explicitly not subject-level | `threshold_and_collapse_analysis.md`; `threshold_diagnostics_summary.csv`; `threshold_diagnostics/` | Supporting diagnostic figures only | Near-one-class behavior, threshold selection, calibration, class balance. |

## Cross-validation and uncertainty

| What was measured | Statistical unit | Package output | Corresponding table / figure | Reviewer concern addressed |
|---|---|---|---|---|
| Repeated stratified subject-grouped fold design and overlap assertions | Subject | `cross_validation/fold_assignments.csv`; `cross_validation/README.md`; `cross_validation/cv_results_all_folds.csv`; `cross_validation/cv_summary.csv` | `tables/table3_repeated_subject_grouped_cv.*` | Fair paired grouped CV, common folds, subject disjointness. |
| Subject-bootstrap confidence intervals, paired permutation tests, and Holm correction | Subject | `statistics/confidence_intervals.csv`; `statistics/model_comparisons.csv`; `statistics/statistical_analysis.md` | `tables/table5_statistical_model_comparisons.*`; Figures 2 and 4 | Uncertainty estimates and model comparisons without treating higher point estimates as significance. |
| Adult-cohort independent sample size and uncertainty limitations | Subject, with epochs explicitly separated | `adult_small_sample_analysis.md`; `adult_available_subject_probabilities.csv` | Supporting small-sample analysis | Small adult sample; no inflation of sample size from epochs. |

**Important availability note:** `cross_validation/cv_results_all_folds.csv` and `cross_validation/cv_summary.csv` are initialized schemas without model-fold result rows. The grouped-CV *design* is reproducible, but model performance mean/SD/CI cannot be reported until all models are executed through it. New neural subject-level prediction files are likewise not present; no legacy neural metrics were substituted.

## Explainability and sensor-space visualization

| What was measured | Statistical unit | Package output | Corresponding table / figure | Reviewer concern addressed |
|---|---|---|---|---|
| Raw-EEG Integrated Gradients for saved CNN/LSTM benchmark artifacts with model/background/class metadata | Epoch attribution aggregated within subject, then across subjects | `xai/subject_attributions.csv`; `xai/raw_epoch_attributions_cnn.npz`; `xai/raw_epoch_attributions_lstm.npz`; `xai/xai_run_metadata_*.json`; `xai/xai_methodology.md` | Supporting numeric XAI artifacts | XAI linked to identifiable model and raw input representation. |
| Electrode-level subject-balanced attribution with bootstrap CIs | Subject | `xai/electrode_attributions.csv`; `xai/region_normalization.csv`; `xai/region_bias_analysis.md` | `figures/figure5_aggregate_xai_electrode_attribution.*` | Electrode-count normalization; aggregate XAI uncertainty. |
| Stability across subjects and repeated epoch samples/seeds | Subject attribution profile | `xai/xai_stability.csv` | `figures/figure6_xai_stability.*` | XAI stability. |
| Sensor/scalp-space topographic display | Subject-balanced sensor-level attribution | `figures/*_topoplot.png`; `figures/*_topoplot.pdf`; `figures/visualization_audit.md`; `figures/topoplot_difference_statistics.csv` | CNN/LSTM ADHD, Control, and overall scalp topoplots | Replace source-like visuals with sensor-space scalp topographies; no source localization. |

## Architecture, final tables, figures, and compliance audit

| What was measured | Statistical unit | Package output | Corresponding table / figure | Reviewer concern addressed |
|---|---|---|---|---|
| Comparison between the former EEGFormer label and cited EEGformer architecture | Not applicable: architecture/code audit | `eegformer_architecture_audit.md` | Supporting technical audit | EEGFormer naming/architecture validity. |
| Final reviewer-revision tables and their exact source provenance | As stated in each table | `tables/`; `tables/table_generation_report.md` | Tables 1–5 and confusion summary | Reproducible final numerical reporting. |
| Final reviewer-revision figures and provenance | As stated per figure | `figures/`; `figures/figure_generation_report.md` | Figures 1–6 and topoplots | Readable, uncertainty-aware, sensor-space-only visuals. |
| Technical status of all Reviewer 2/3/4 concerns | Not applicable: implementation/output audit | `technical_compliance_matrix.md` | Supporting compliance artifact | Explicit RESOLVED / PARTIALLY RESOLVED / NOT RESOLVED evidence. |

## Reproducibility constraints

- Paths in copied CSV/JSON provenance may refer to their original repository locations; the corresponding copied artifacts are retained in this package.
- Missing acquisition metadata remains marked `REQUIRES MANUAL DATASET DOCUMENTATION`; it is not inferred here.
- XAI runs are clearly identified as limited verification samples and must not be generalized to source localization or biological causality.

## Final verification artifacts

The package additionally includes `FINAL_VERIFICATION_REPORT.md`, `final_verification_checklist.csv`, `subject_leakage_verification.csv`, `metric_verification.csv`, `subject_aggregation_examples.csv`, `trivial_baseline_comparison_verification.csv`, `cross_validation_verification.csv`, `final_output_traceability.csv`, `STALE_FILES_WARNING.md`, and `reproducibility_verification.csv`. These provide the final independent arithmetic, leakage, traceability, stale-artifact, and lightweight regeneration checks. The corresponding audit script is `scripts/final_verification_audit.py`.
