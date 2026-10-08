# EEGformer architecture audit

## Scope and verdict

The project previously labelled `Results/EEGFormer/EEGFormer_EpochLevelSplit.py` and its generated results as **EEGFormer**. The exact cited reference is:

> Wan, Z., Li, M., Liu, S., Huang, J., Tan, H., & Duan, W. (2023). *EEGformer: A transformer-based brain activity classification method using EEG signal*. **Frontiers in Neuroscience, 17**, 1148855. https://doi.org/10.3389/fnins.2023.1148855

The cited article describes a distinct CNN-plus-three-specialized-transformer architecture. The project implementation is **not a faithful implementation or a close architectural variant** of that method. It must therefore not be presented as EEGformer or as a reproduction/benchmark of Wan et al.

No new model architecture has been substituted and no historical result has been overwritten. For all future generated output, the code now uses the neutral label **Temporal Transformer Baseline**. The historical file path `Results/EEGFormer/EEGFormer_EpochLevelSplit.py` remains solely for traceability; its filename does not identify the current model label.

## Evidence inspected

- Project implementation: `Results/EEGFormer/EEGFormer_EpochLevelSplit.py`.
- Project manuscript/reference material: `main_fixed.tex` (citation key `Wan2023`) and `pdf_review_extract.txt`.
- Cited method: Wan et al. (2023), DOI above. The paper describes a depth-wise 1D-CNN feature extractor, serial regional/synchronous/temporal transformer processing, and a CNN decoder.

An official EEGformer implementation repository, commit, or model-configuration file is **not cited or vendored in this project**. Exact paper hyperparameters that are not unambiguously specified in the paper are consequently marked as not determinable rather than inferred.

## Side-by-side architecture comparison

| Component | Cited EEGformer (Wan et al., 2023) | Project implementation previously called EEGFormer | Assessment |
|---|---|---|---|
| Input representation | A normalized/detrended raw EEG segment is represented over channels and time; the paper describes an input of `S` channels by `L` samples before CNN feature extraction. | Epoch tensor is `(time, channels)`; for the default data flow this is `(512, 19)`. A training-only channel standardizer is applied before the network. | Different tensor convention and preprocessing path; not inherently invalid, but not demonstrated equivalent. |
| Temporal convolution | Three depth-wise 1D convolution layers (`1 x 10`, valid convolution, stride 1; paper describes 120 filters) extract channel-wise temporal features. | **Absent.** The first learned operation is `Dense(128)` applied independently to each time sample. | Substantial required component missing. |
| Spatial convolution / channel feature treatment | CNN features retain a channel/spatial dimension for subsequent regional and synchronous modelling; the decoder also uses convolutions. | No spatial convolution, depth-wise convolution, channel grouping, or channel-specific attention. Channels are simply input features to a shared dense projection. | Substantial required component missing. |
| Attention / transformer blocks | Three distinct transformer modules model regional, synchronous, and temporal relationships serially. | Two generic Keras self-attention encoder blocks attend only over the time axis (`MultiHeadAttention(x, x)`). | Different number, purpose, and axes of transformer modules. |
| Positional encoding | Positional embeddings are part of the paper's transformer design; exact implementation details/configuration were not recoverable from project-cited material. | Fixed sinusoidal temporal positional encoding added after the dense projection. | Not established as equivalent; project implementation is generic temporal encoding only. |
| Normalization | Paper reports detrending/normalization before the model; exact correspondence to this project's fitted standardizer is not established. | Channel standardizer fitted on training epochs only; `LayerNormalization` after each attention and feed-forward residual path. | Leakage-safe project preprocessing, but not a verified reproduction. |
| Number of attention heads | Multi-head attention is used. The exact reference configuration is not established by material vendored/cited in this project. | 4 heads by default; `key_dim=32` for `d_model=128`. | No evidence of hyperparameter match. |
| Embedding / latent dimensions | CNN-derived feature maps feed the specialized transformers. Exact directly comparable `d_model` is not determinable from the local cited material. | Linear projection to `d_model=128`; feed-forward dimension 256. | Different feature source and unverified dimensions. |
| Pooling / decoder | A CNN decoder of three convolutional layers followed by a fully connected classifier. | `GlobalAveragePooling1D` followed by dense layers of 128 and 64 units with dropout. | Substantial required component missing. |
| Classifier / objective | Fully connected output with softmax; paper describes categorical cross-entropy for multi-class brain-activity classification. | Single sigmoid output with binary cross-entropy for ADHD classification. | Task adaptation is reasonable in isolation, but it is not the cited classifier/decoder. |

## Exact project architecture now labelled Temporal Transformer Baseline

1. Training-only per-channel standardization is applied outside the network.
2. Each time point's 19-channel vector is linearly projected to 128 dimensions.
3. Fixed sinusoidal temporal positional encoding is added.
4. Two standard transformer encoder blocks are applied: 4-head temporal self-attention, residual `LayerNormalization`, a 256-unit ReLU feed-forward layer, dropout, projection back to 128 dimensions, and residual `LayerNormalization`.
5. Global average pooling across time is followed by dense layers of 128 and 64 units (each with 0.5 dropout) and a sigmoid binary classifier.

This is accurately described as a **temporal transformer baseline with sinusoidal positional encoding**, not EEGformer.

## Consequences for labels and results

- `Results/EEGFormer/EEGFormer_EpochLevelSplit.py` now emits `Model = "Temporal Transformer Baseline"`, `temporal_transformer_metrics.csv`, and `temporal_transformer_baseline.keras` for newly run experiments.
- `run_deep_models_benchmark.py` registers the model as `Temporal Transformer Baseline` and reads its renamed metrics file.
- Existing files/results bearing the historic `EEGFormer` label are deliberately left unchanged. They must be treated as legacy artifacts and not relabelled retroactively without provenance.
- Manuscript tables or archived result files that still display `EEGFormer` require a separate, explicit result-table/manuscript revision after rerunning or clearly mapping the recorded configuration. This audit does not reinterpret old numerical results.

## Required reporting language

Use: “Temporal Transformer Baseline” and describe the components listed above.

Do not use: “EEGformer,” “Wan et al. EEGformer,” “EEGformer reproduction,” or wording that implies the results evaluate the cited architecture.

## Limitations that cannot be resolved from project files

- No official EEGformer source-code version is linked in the project, so an implementation-level parameter-for-parameter comparison is not possible.
- The project does not contain an experiment implementing the cited depth-wise CNN, regional transformer, synchronous transformer, temporal transformer, and CNN decoder together.
- No run manifest currently ties historical EEGFormer-labelled saved weights to a source-code hash. Historical results can therefore be traced only to their filenames/directories and should not be represented as a validated EEGformer reproduction.
