# Explainability audit and revised XAI methodology

## Audit of existing XAI outputs

| Output / generator | What was explained | Input representation | Status |
|---|---|---|---|
| `Results/old/SHAP/shap_channel_importance.*` / `SHAP_ChannelImportance.py` | A newly trained `RandomForestClassifier`, not a benchmark neural model | 95 relative Welch band-power features (19 channels x 5 bands); at most 30 epochs per subject | **Feature-model explanation only.** It must not be described as explaining CNN, LSTM, Bi-LSTM, or the Temporal Transformer Baseline. The historical output does not save the model file, background data identity, or complete subject list. |
| `Results/old/LIME/lime_channel_importance.*` / `LIME_ChannelImportance.py` | A separately trained `RandomForestClassifier`, not a benchmark neural model | The same 95 relative Welch band-power features | **Feature-model explanation only.** The same missing-provenance limitations apply. |
| `Results/old/Graphs/channel_importance.csv`, `regional_importance.csv`, and workbook / `CNN_Channel_Regional_Importance.py` | A CNN trained within the plotting script, not a saved benchmark CNN | Training-standardized raw EEG epochs | Raw-EEG gradient sensitivity, but not a reproducible explanation of a saved benchmark artifact. It includes a single selected epoch and an epoch-weighted test average; it lacks the required metadata and subject-balanced uncertainty analysis. |

No SHAP/LIME output in the project explains an actual raw-EEG benchmark CNN/LSTM/Bi-LSTM/Temporal Transformer Baseline. Historic feature-model figures must be labelled “SHAP/LIME explanation of a Random Forest trained on relative Welch band-power features.”

## Revised raw-EEG pipeline

`reviewer_xai.py` applies **Integrated Gradients** directly to a supplied saved Keras benchmark model. It reconstructs the epochs and fits the channel standardizer using *only the train subjects recorded in that model's split manifest*. The baseline is the resulting mean standardized training epoch. The scalar explained output is the ADHD sigmoid probability (positive class = 1).

For each held-out ADHD and Control subject, the program samples a fixed number of that subject's epochs for each of at least two deterministic seeds. It averages epoch attributions within subject before aggregating subjects. Thus a person with many epochs does not receive more weight than another person. The saved files are:

- `subject_attributions.csv`: subject, class, seed, model/provenance, channel-level signed and absolute attributions, and the number of epochs used.
- `raw_epoch_attributions_<model>.npz`: the unaggregated time-by-channel Integrated Gradients values with subject, label, seed, and dataset-epoch index arrays.
- `electrode_attributions.csv`: subject-balanced electrode summaries by class with non-parametric bootstrap 95% CIs.
- `xai_stability.csv`: pairwise Spearman stability across subjects and across repeated epoch samples/seeds.
- `xai_run_metadata_<model>.json`: complete run-level provenance; the same metadata is also embedded in each model's raw-attribution file.

The default invocation explains the saved combined-holdout CNN in `Results/CleanDeepBenchmark/combined_holdout/CNN`. It is not a substitute for rerunning this procedure for every neural model/scenario intended for reporting. Each invocation records model name/file, raw-EEG input representation, dataset, subjects, method, background data, and output class in every generated CSV.

### Generated verification outputs

The current files contain separate runs for the saved combined-holdout CNN (three ADHD and three Control subjects; four sampled epochs per subject; 16 integration steps) and LSTM (two ADHD and two Control subjects; two sampled epochs per subject; eight integration steps). Both use seeds `42` and `31415`. These are a provenance and pipeline verification set, not a basis for scientific channel-ranking claims; the model-specific JSON metadata files are authoritative for the exact run settings. Before manuscript reporting, rerun the desired models/scenarios using the planned sample size and the default or explicitly preregistered attribution settings.

## Interpretation limits

Attribution describes local model sensitivity under the selected baseline. EEG electrodes are sensors, not direct measurements of neuroanatomical sources. Do not make causal, regional, or neuroanatomical claims from these sensor-level values. Confidence intervals and stability quantify only variability within the selected held-out subjects and sampling procedure; they do not establish biological validity.
