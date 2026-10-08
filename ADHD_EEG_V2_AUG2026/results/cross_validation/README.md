# Repeated subject-grouped cross-validation

`fold_assignments.csv` contains the shared outer folds for all models. Each row
identifies a subject, its repeat seed, fold, and role (`train`, `validation`, or
`test`), plus subject- and epoch-level class distributions. The grouping variable
is always `subject_id`; generation raises an assertion if a subject is assigned to
more than one role inside a fold.

The assignments use three deterministic seeds (`42`, `31415`, `27182`) and select
the maximum valid number of stratified folds up to five from the actual class
counts. Every audited cohort supports five folds in the current data. The inner
validation partition is subject-disjoint from both outer training and outer test.

`repeated_grouped_cv.evaluate_model_on_folds()` is the common model interface for
CNN, LSTM, Bi-LSTM, EEGFormer, and classical baselines. A model callback receives
the exact same train/validation/test epoch tensors for every assigned fold. It
returns validation/test probabilities; the framework evaluates one mean
probability per test subject, chooses an optional threshold on validation subjects
only, and writes fold-level metrics. `summarize_cv_results()` reports mean, SD,
and t-based 95% confidence intervals.

The current `cv_results_all_folds.csv` and `cv_summary.csv` are initialized
schemas, not invented model results. They are populated only when the models are
rerun through the shared callback interface.
