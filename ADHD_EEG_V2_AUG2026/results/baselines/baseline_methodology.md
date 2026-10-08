# Reviewer-revision baselines

All baseline scenarios call `deep_model_utils.build_experiment_split` with the
same CSV, seed, validation size, test size, and scenario names as the deep
benchmark. Each output scenario includes its exact `split_manifest.json`.

Learned models use 95 features per epoch: log10 Welch absolute band power in
delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma
(30-45 Hz), calculated for each of the 19 harmonized channels after the shared
loader has converted inputs to 512 samples at the 256-Hz-equivalent length.
`StandardScaler` is fit only on training features inside the Logistic Regression
and Linear SVM pipelines; validation/test features are transformed with the
fitted training scaler. The fixed decision threshold is 0.5 for probability
models and 0.0 for the LinearSVC signed margin.

Subject-level scores are the mean of a held-out subject's epoch scores. Existing
neural artifacts are copied into the comparison CSV exactly as saved; unavailable
PR-AUC and separate TN/FP/FN/TP fields are left blank rather than inferred.
