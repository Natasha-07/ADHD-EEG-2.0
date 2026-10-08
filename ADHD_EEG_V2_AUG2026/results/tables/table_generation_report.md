# Final reviewer-revision table generation report

## Scope

Tables were generated only from newly generated reviewer-revision artifacts. No manuscript source, legacy paper table, or `Results/CleanDeepBenchmark` metric was read for numerical table values.

| Output | Exact source artifact(s) | Notes |
|---|---|---|
| Table 1 | `Results\reviewer_revision\dataset_summary.csv`; `Results\reviewer_revision\dataset_comparison_table.csv` | Dataset counts plus audited channel/rate metadata. |
| Table 2 | `Results\reviewer_revision\baselines\baseline_results_all_scenarios.csv` filtered to `combined_holdout`, `Subject` | Six baselines have new subject-level results. Neural rows are explicitly NA because no eligible reviewer-revision neural output exists. |
| Table 3 | `Results\reviewer_revision\cross_validation\cv_summary.csv` | The source has headers only. Table reports unavailability rather than values. |
| Table 4 | `Results\reviewer_revision\baselines\baseline_results_all_scenarios.csv` filtered to transfer scenarios and `Subject` | Subject-level cross-cohort/cross-dataset transfer for the newly run baselines only. |
| Table 5 | `Results\reviewer_revision\statistics\model_comparisons.csv` | Paired subject-level baseline comparisons, with bootstrap CIs and Holm correction. |
| Confusion summary | `Results\reviewer_revision\baselines\baseline_results_all_scenarios.csv` filtered to `Subject` | TN, FP, FN, TP and requested diagnostic measures. |

`Results\reviewer_revision\statistics\confidence_intervals.csv` was audited as the CI source for performance/XAI uncertainty but is not duplicated into a performance table because Table 2's requested columns specify point estimates. It remains the supporting reviewer-revision uncertainty artifact.

## Explicit exclusions

- `Results\reviewer_revision\baselines\baseline_neural_comparison.csv` was not used as a numerical source because it contains old `Results/CleanDeepBenchmark` neural results, including epoch-level rows.
- No `.tex`, `.docx`, abstract, title, or manuscript text was modified.
- Repeated grouped-CV and neural subject-level results must be generated before Tables 2 and 3 can be considered complete for all retained neural models.
