# Subject-level statistical analysis

## Statistical unit and data availability

All calculations use subjects as the independent unit. Epochs are never resampled or counted as independent observations. The analysis found 18 model-by-scenario subject-prediction files (the six baseline models across three scenarios). No neural-model subject-level prediction files were found in the reviewer-revision output, so no valid neural-vs-baseline paired test was calculated.

CV result rows are unavailable: the file contains only its initialized schema.

## Confidence intervals

For each available subject-level prediction set, a stratified non-parametric bootstrap resamples subjects with replacement separately within ADHD and Control classes (1000 replicates). Percentile 95% confidence intervals are reported for Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, and PR-AUC. Cross-dataset transfer uncertainty is included as the `adult_to_child` and `child_to_adult` scenario rows in `confidence_intervals.csv`.

## Model comparisons

For each pair of models evaluated on exactly the same subject IDs within a scenario, the analysis reports the metric difference (model A minus model B), a paired stratified-subject bootstrap 95% CI, and a two-sided paired label-swap permutation p-value (1000 permutations). The label swap exchanges the two model outputs within each subject and therefore preserves the pairing without assuming normality. Holm correction is applied across every reported model-pair/metric test in `model_comparisons.csv`; only `statistically_significant_holm_0_05 = True` supports a multiplicity-adjusted significance statement.

The test statistic is the observed subject-level metric difference. A numerically larger mean is not interpreted as significant unless its corrected p-value meets the stated criterion.

## XAI uncertainty

For XAI, repeated epoch samples are averaged within subject first. Electrode-level mean absolute attribution is then bootstrapped across subjects, separately by model and class, with results appended to `confidence_intervals.csv`. These intervals quantify sampled-subject uncertainty only; they do not establish source localization or biological effects.
