# Technical reviewer compliance matrix

## Scope and status definitions

This is a **technical code/result audit only**. It does not assess or revise manuscript wording.

- **RESOLVED**: the requested technical mechanism exists and a relevant reviewer-revision output demonstrates it.
- **PARTIALLY RESOLVED**: the mechanism exists but is only partly executed, is limited to a subset of models, or essential external metadata/results remain unavailable.
- **NOT RESOLVED**: no adequate technical implementation/output was found.
- **MANUSCRIPT REVISION REQUIRED**: technical evidence exists but its accurate reporting or interpretation requires manuscript work; no such writing is performed here.

## Reviewer 2

| Item | Status | Evidence (code / output / table / figure) | Technical audit finding |
|---|---|---|---|
| Always-ADHD baseline | RESOLVED | `run_reviewer_baselines.py`; `baselines/baseline_results_all_scenarios.csv`; `tables/table2_main_subject_level_performance.csv` | Newly generated subject-level predictions/results are available for all scenarios. |
| Always-Control and majority baselines | RESOLVED | `run_reviewer_baselines.py`; `baselines/baseline_results_all_scenarios.csv`; Table 2 | Both trivial comparators were newly evaluated at subject level. |
| LSTM near-one-class behavior investigation | PARTIALLY RESOLVED | `lstm_experiment_utils.py`; `threshold_and_collapse_analysis.md`; `threshold_diagnostics/` | Threshold, loss, output activation, labels, balance, calibration and epoch probability behavior were audited. The documented `child_to_adult` LSTM collapse is epoch-level because saved neural subject predictions are unavailable. |
| Specificity | RESOLVED | `subject_level_eval_utils.py`; baseline results; `tables/confusion_matrix_summary.csv` | Specificity is computed and reported for newly generated baseline subject-level results. |
| Balanced accuracy | RESOLVED | `subject_level_eval_utils.py`; baseline results; Tables 2 and 4; `figures/figure2_main_subject_level_performance.*` | Included as a primary reported metric for new baseline outputs. |
| Confusion matrices | RESOLVED | `subject_level_eval_utils.py`; baseline results; `tables/confusion_matrix_summary.csv`; `figures/figure3_subject_level_confusion_matrices.*` | Subject-level TN/FP/FN/TP and visual matrices generated for new baselines. |
| Subject-level aggregation | PARTIALLY RESOLVED | `subject_level_eval_utils.py`; `deep_model_utils.py`; baseline subject prediction CSVs; `subject_level/README.md` | Mean/median subject aggregation is implemented and executed for baselines. New neural subject-level result files are absent. |
| Repeated grouped validation | PARTIALLY RESOLVED | `repeated_grouped_cv.py`; `cross_validation/fold_assignments.csv`; `cross_validation/README.md` | Shared subject-disjoint repeated folds and assertions exist, but `cv_results_all_folds.csv` and `cv_summary.csv` contain only initialized schemas. |
| Uncertainty estimates | PARTIALLY RESOLVED | `subject_level_statistics.py`; `statistics/confidence_intervals.csv`; `adult_small_sample_analysis.md` | Subject-bootstrap CIs are generated for baselines and XAI; CV uncertainty cannot be reported until folds are evaluated. |
| Adult/child subject and epoch imbalance | RESOLVED | `dataset_audit.py`; `dataset_summary.csv`; `subject_epoch_counts.csv`; `adult_small_sample_analysis.md`; Table 1 | Actual counts and the adult epoch dominance are quantified, with explicit independent-subject warning. |
| Cross-dataset harmonization audit | PARTIALLY RESOLVED | `audit_cross_dataset_harmonization.py`; `cross_dataset_harmonization.md`; `dataset_comparison_table.csv` | Code-level channel/order/time-grid/normalization checks are documented. Acquisition, reference, filtering and artifact metadata require original dataset documentation. |
| XAI linked to identifiable model | PARTIALLY RESOLVED | `reviewer_xai.py`; `xai/xai_run_metadata_cnn.json`; `xai/xai_run_metadata_lstm.json`; `xai/subject_attributions.csv` | Full provenance is recorded for CNN/LSTM verification runs. It has not yet been run for every retained model/scenario. |
| XAI stability | PARTIALLY RESOLVED | `reviewer_xai.py`; `xai/xai_stability.csv`; `figures/figure6_xai_stability.*` | Across-subject and repeated-sampling stability are computed, but on small verification samples only. |
| Electrode-count normalization | RESOLVED | `audit_region_normalization.py`; `xai/region_normalization.csv`; `xai/region_bias_analysis.md` | Total and per-electrode-normalized sensor-group results are reported and ranking changes are flagged. |
| EEGFormer architecture audit | RESOLVED | `Results/EEGFormer/EEGFormer_EpochLevelSplit.py`; `eegformer_architecture_audit.md`; `run_deep_models_benchmark.py` | Audit found a non-reproduction; future generated label is `Temporal Transformer Baseline`. |
| Classical/linear baselines | RESOLVED | `run_reviewer_baselines.py`; baseline subject prediction CSVs; Tables 2 and 4 | Logistic Regression and Linear SVM using documented Welch band-power features were newly evaluated on benchmark splits. |
| Comparable model evaluation | PARTIALLY RESOLVED | `run_reviewer_baselines.py`; `repeated_grouped_cv.py`; `cross_validation/fold_assignments.csv` | Split framework is shared; baselines were executed. Neural models have not generated new matched subject-level/CV outputs. |

## Reviewer 3

| Item | Status | Evidence (code / output / table / figure) | Technical audit finding |
|---|---|---|---|
| Dataset class distributions | RESOLVED | `dataset_audit.py`; `dataset_summary.csv`; Table 1; `figures/figure1_dataset_subject_distribution.*` | Actual subject/epoch class distributions are generated from data/index files. |
| Available EEG acquisition metadata collected | PARTIALLY RESOLVED | `audit_cross_dataset_harmonization.py`; `dataset_comparison_table.csv`; `cross_dataset_harmonization.md` | Available code/CSV metadata were extracted. Missing hardware, montage source, reference, ground, task, duration/session documentation is explicitly marked `REQUIRES MANUAL DATASET DOCUMENTATION`. |
| Readable replacement figures | RESOLVED | `generate_final_reviewer_figures.py`; `figures/figure1_*` through `figure6_*` PNG/PDF | New figures use 300-DPI PNG, vector PDF, large fonts, explicit labels, and reviewer-revision data only. |
| Sensor-space scalp topoplots | RESOLVED | `generate_sensor_space_topoplots.py`; `figures/*_topoplot.png/pdf`; `figures/visualization_audit.md` | Standard-10-20 coordinate based scalp-sensor topographies are generated and explicitly labelled sensor-space. |
| Invalid source-space visualizations removed from generated results | RESOLVED | `generate_fig7_fig8.py`; `figures/visualization_audit.md` | Legacy template-overlay generator now raises an error directing users to sensor-space topoplots. No source-space diagram is generated by the revised figure workflow. |
| Additional statistical analysis | PARTIALLY RESOLVED | `subject_level_statistics.py`; `statistics/model_comparisons.csv`; `statistics/confidence_intervals.csv`; Table 5 | Subject-bootstrap/permutation/Holm analysis exists for baseline outputs; neural and populated-CV comparisons remain unavailable. |

## Reviewer 4

| Item | Status | Evidence (code / output / table / figure) | Technical audit finding |
|---|---|---|---|
| Subject-level evaluation | PARTIALLY RESOLVED | `subject_level_eval_utils.py`; baseline subject prediction CSVs; Tables 2/4; Figure 3 | Executed for all new baselines. The neural result set has not yet been regenerated at subject level. |
| Small adult sample uncertainty analysis | RESOLVED | `adult_small_sample_audit.py`; `adult_small_sample_analysis.md`; `adult_available_subject_probabilities.csv` | Subject N=16, class counts, epoch non-independence, resolution and Wilson uncertainty are documented. |
| Grouped cross-validation | PARTIALLY RESOLVED | `repeated_grouped_cv.py`; `cross_validation/fold_assignments.csv`; `cross_validation/README.md` | Correct grouped, repeated design and folds are generated; model-fold results are not yet populated. |
| Error bars/confidence intervals | PARTIALLY RESOLVED | `subject_level_statistics.py`; `statistics/confidence_intervals.csv`; Figures 2, 4, 5 | Available new baseline/XAI figures use bootstrap CIs. CV/neural error bars cannot be supplied without their result rows. |
| Statistical model comparisons | PARTIALLY RESOLVED | `subject_level_statistics.py`; `statistics/model_comparisons.csv`; Table 5 | Paired subject permutation, bootstrap CIs and Holm correction were applied to baseline comparisons only. |
| Subject-level XAI aggregation where technically possible | PARTIALLY RESOLVED | `reviewer_xai.py`; `xai/subject_attributions.csv`; `xai/electrode_attributions.csv`; Figures 5/6/topoplots | Within-subject epoch then across-subject aggregation is implemented and run for small CNN/LSTM verification subsets, not all retained models/subjects. |

## Technical work still required before every item can be marked resolved

1. Rerun CNN, LSTM, Bi-LSTM, and the Temporal Transformer Baseline through `repeated_grouped_cv.py` using the shared assignments, then populate `cv_results_all_folds.csv` and `cv_summary.csv`.
2. Save new neural subject-level predictions/results under `Results/reviewer_revision/subject_level/` so model-vs-baseline paired statistics, subject-level confusion matrices, probability distributions, and full Table 2 can be generated without legacy artifacts.
3. Rerun `reviewer_xai.py` with a prespecified adequate subject sample for every neural model/scenario intended for claims.
4. Obtain the externally missing acquisition/preprocessing metadata from original dataset documentation; it cannot be recovered from current project files.

## Manuscript revision required

The following technical evidence must be accurately described/interpreted in the manuscript, but this audit does not edit manuscript text:

- The distinction between completed baseline results and unavailable new neural/CV results.
- The LSTM collapse finding is based on available **epoch-level** tuning diagnostics, not new neural subject-level output.
- Transfer is cross-cohort/cross-dataset and not an isolated age effect.
- XAI verification runs have limited subject samples; scalp-electrode attribution is not source localization.
- Acquisition fields marked `REQUIRES MANUAL DATASET DOCUMENTATION` must not be inferred.
