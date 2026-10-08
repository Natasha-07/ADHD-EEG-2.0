# Subject-level primary evaluation

The prespecified primary endpoint is one observation per held-out `subject_id`.
For each model, epoch probabilities are grouped by subject and their **mean** is
the primary subject probability. The **median** probability is reported in a
`Subject_Sensitivity` metrics row, not substituted into or averaged with the
primary metric.

Every `subject_level_predictions.csv` has at least:

`subject_id`, `dataset`, `true_label`, `number_of_epochs`,
`mean_probability`, `median_probability`, and `predicted_label`.

`run_deep_models_benchmark.py` writes updated neural runs under
`deep_models/`; `run_reviewer_baselines.py` writes executed baseline subject
tables under `baselines/`. Existing neural checkpoints cannot be post-hoc
converted because their current saved folders do not contain epoch predictions.
