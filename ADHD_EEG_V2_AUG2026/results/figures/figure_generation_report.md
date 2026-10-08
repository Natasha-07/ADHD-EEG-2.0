# Final reviewer-revision figure generation report

Only reviewer-revision outputs were used. No manuscript file was edited.

| Figure | Data source | Models/cohorts included | Statistical unit | Analysis level | Uncertainty/error bars |
|---|---|---|---|---|---|
| figure1_dataset_subject_distribution | dataset_summary.csv | Child and adult cohorts | Subject | Subject-level | None; descriptive counts. |
| figure2_main_subject_level_performance | baselines/baseline_results_all_scenarios.csv; statistics/confidence_intervals.csv | Six newly run baselines | Subject | Subject-level | Stratified subject-bootstrap 95% CI. |
| figure3_subject_level_confusion_matrices | baselines/baseline_results_all_scenarios.csv | Six newly run baselines | Subject | Subject-level | None; observed held-out subject counts. |
| figure4_cross_dataset_transfer | statistics/confidence_intervals.csv | Six newly run baselines | Subject | Subject-level | Stratified subject-bootstrap 95% CI. |
| figure5_aggregate_xai_electrode_attribution | xai/electrode_attributions.csv | Saved CNN and LSTM attribution runs | Subject after within-subject epoch/seed aggregation | Subject-level attribution | Subject-bootstrap 95% CI; small verification samples, not confirmatory. |
| figure6_xai_stability | xai/xai_stability.csv | Saved CNN and LSTM attribution runs | Subject | Subject-level attribution profile | No error bars; each bar is the recorded mean pairwise Spearman stability and sample sizes are limited. |
| Sensor-space topoplots | figures/*_topoplot.png/pdf; xai/electrode_attributions.csv | Saved CNN and LSTM attribution runs | Subject after within-subject aggregation | Subject-level attribution | Within each model, ADHD, Control, and overall maps use a consistent scale. No group-difference map was supported after FDR correction. |

## Figures not generated

- **Repeated grouped-CV performance:** not plotted because `cross_validation/cv_results_all_folds.csv` and `cv_summary.csv` contain initialized schemas, not fold results.
- **Neural-model comparison/error bars:** not plotted because no newly generated reviewer-revision neural subject-level performance records exist.
- **Neural probability distributions:** not plotted because the only available threshold-collapse diagnostic uses epoch-level data. This figure set contains no epoch-level performance/probability panel.
- **ADHD-minus-Control scalp maps:** not plotted because no electrode survived FDR-controlled permutation testing in `topoplot_difference_statistics.csv`.
- **Physical brain/source-space figures:** not generated. The existing topoplots are explicitly sensor/scalp-space displays, not source localization.
