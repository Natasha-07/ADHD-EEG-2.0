# Threshold and LSTM-collapse analysis

## Threshold protocol repair

The active evaluation code now always reports the prespecified 0.50 threshold. Optional optimized thresholds are selected only from validation **subject** mean probabilities using either Youden J or maximum balanced accuracy, then frozen before test evaluation. Test F1, test precision, and all other test metrics are not threshold-selection inputs.

## Available-artifact diagnosis

The existing CleanDeepBenchmark neural folders contain metrics/checkpoints but no epoch-prediction CSVs. Therefore their probability histograms, calibration, and subject-level collapse diagnosis cannot be reconstructed honestly. Updated neural runs will write the required epoch and subject prediction files. The diagnostics below use the prespecified fixed-threshold LSTM tuning artifacts only; their saved predictions lack subject IDs, so these are epoch-level diagnostic plots, not subject-level estimates.

## LSTM: adult_to_child

- Reference artifact: `Results\LSTM_Tuning\adult_to_child\adult_to_child_no_class_weight_fixed_threshold` (chosen by fixed-threshold name, not test performance).
- Training balance: ADHD=7, Control=7; epochs ADHD=4327, Control=4368
- Validation balance: ADHD=1, Control=1; epochs ADHD=616, Control=620
- Config: class weights `none`; normalization `train_channel_standard`; loss is binary cross-entropy; output activation is sigmoid; labels are ADHD=1 and Control=0.
- Default 0.50: predicted ADHD=64.41%, Control=35.59%; sensitivity=0.6474; specificity=0.3601; balanced accuracy=0.5037; confusion matrix=[[642 1141],[800 1469]].
- Validation-only maximum-balanced-accuracy threshold=0.588019; frozen-test balanced accuracy=0.4957; confusion matrix=[[955 828],[1235 1034]]. This is reported diagnostically, not selected using test results.
- Brier score=0.2746; figures: `Results\reviewer_revision\threshold_diagnostics\lstm_adult_to_child_probability_histogram.png`, `Results\reviewer_revision\threshold_diagnostics\lstm_adult_to_child_calibration.png`.
- Collapse flag at default threshold: **NO** (defined as >=95% one predicted class).

## LSTM: child_to_adult

- Reference artifact: `Results\LSTM_Tuning\child_to_adult\child_to_adult_no_class_weight_fixed_threshold` (chosen by fixed-threshold name, not test performance).
- Training balance: ADHD=54, Control=54; epochs ADHD=1961, Control=1612
- Validation balance: ADHD=7, Control=6; epochs ADHD=308, Control=171
- Config: class weights `none`; normalization `train_channel_standard`; loss is binary cross-entropy; output activation is sigmoid; labels are ADHD=1 and Control=0.
- Default 0.50: predicted ADHD=99.98%, Control=0.02%; sensitivity=0.9996; specificity=0.0000; balanced accuracy=0.4998; confusion matrix=[[0 4988],[2 4941]].
- Validation-only maximum-balanced-accuracy threshold=0.557763; frozen-test balanced accuracy=0.4996; confusion matrix=[[8 4980],[12 4931]]. This is reported diagnostically, not selected using test results.
- Brier score=0.2587; figures: `Results\reviewer_revision\threshold_diagnostics\lstm_child_to_adult_probability_histogram.png`, `Results\reviewer_revision\threshold_diagnostics\lstm_child_to_adult_calibration.png`.
- Collapse flag at default threshold: **YES** (defined as >=95% one predicted class).

## LSTM: combined_holdout

- Reference artifact: `Results\LSTM_Tuning\combined_holdout\combined_epoch_zscore_fixed_threshold` (chosen by fixed-threshold name, not test performance).
- Training balance: ADHD=49, Control=49; epochs ADHD=3603, Control=4997
- Validation balance: ADHD=6, Control=5; epochs ADHD=1377, Control=735
- Config: class weights `none`; normalization `epoch_zscore`; loss is binary cross-entropy; output activation is sigmoid; labels are ADHD=1 and Control=0.
- Default 0.50: predicted ADHD=21.80%, Control=78.20%; sensitivity=0.2442; specificity=0.8383; balanced accuracy=0.5412; confusion matrix=[[871 168],[1687 545]].
- Validation-only maximum-balanced-accuracy threshold=0.702278; frozen-test balanced accuracy=0.5009; confusion matrix=[[1038 1],[2226 6]]. This is reported diagnostically, not selected using test results.
- Brier score=0.2660; figures: `Results\reviewer_revision\threshold_diagnostics\lstm_combined_holdout_probability_histogram.png`, `Results\reviewer_revision\threshold_diagnostics\lstm_combined_holdout_calibration.png`.
- Collapse flag at default threshold: **NO** (defined as >=95% one predicted class).

## Interpretation and safeguards

A high ADHD-prediction percentage accompanied by near-zero specificity is evidence of a practical one-class collapse at the specified threshold, even when F1 is superficially high. Class weighting changes the fitted loss emphasis and can shift probability distributions; it must not be compensated by choosing a test-derived threshold. Calibration is assessed descriptively by Brier score and reliability plots, not corrected using test labels.

No model architecture, loss, output activation, or class weight was changed in this analysis to improve test results. Any future change must be chosen from training/validation evidence and evaluated once on frozen test subjects.
