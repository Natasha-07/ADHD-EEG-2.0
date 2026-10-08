# Sensor-electrode group normalization audit

## Mapping and scope

This audit aggregates the subject-balanced, absolute Integrated Gradients values in `electrode_attributions.csv`. It concerns **scalp-electrode attribution**, not brain-source localization. Each listed group is a descriptive sensor grouping:

- sensor-level frontal electrode group: 7 scalp electrodes (Fp1, Fp2, F3, F4, F7, F8, Fz)
- central electrode group: 3 scalp electrodes (C3, C4, Cz)
- temporal electrode group: 2 scalp electrodes (T7, T8)
- parietal electrode group: 5 scalp electrodes (P3, P4, P7, P8, Pz)
- occipital electrode group: 2 scalp electrodes (O1, O2)

The mapping contains 19 unique electrodes exactly once. No claim is made that a sensor-level frontal electrode group measures frontal-lobe activity, or that any group represents a localized neural source.

## Methods

For every model and subject class, this audit reports both:

1. **Total sensor-level attribution**: the sum across electrodes in a group. This answers how much total attribution is assigned to all sensors in that group, but increases mechanically with group size.
2. **Mean attribution per electrode**: total attribution divided by the number of electrodes. The `per_electrode_normalized_percent` column compares these group means after scaling them to sum to 100% across the five groups. This is the appropriate comparison for group importance independent of electrode count.

`ranking_changes_substantially_after_normalization` is true when the top-ranked group changes or any group moves at least two ranks. It is a descriptive diagnostic, not a statistical test.

## Results

### CNN - ADHD subjects

- Sum-ranked sensor groups: sensor-level frontal > parietal > central > occipital > temporal.
- Per-electrode-normalized ranking: central > parietal > sensor-level frontal > occipital > temporal.
- Substantial ranking change: **yes** (pre-specified: a changed top group or a movement of at least two ranks).
### CNN - Control subjects

- Sum-ranked sensor groups: sensor-level frontal > parietal > central > occipital > temporal.
- Per-electrode-normalized ranking: central > sensor-level frontal > parietal > occipital > temporal.
- Substantial ranking change: **yes** (pre-specified: a changed top group or a movement of at least two ranks).
### LSTM - ADHD subjects

- Sum-ranked sensor groups: sensor-level frontal > parietal > central > occipital > temporal.
- Per-electrode-normalized ranking: central > occipital > temporal > sensor-level frontal > parietal.
- Substantial ranking change: **yes** (pre-specified: a changed top group or a movement of at least two ranks).
### LSTM - Control subjects

- Sum-ranked sensor groups: sensor-level frontal > central > parietal > occipital > temporal.
- Per-electrode-normalized ranking: central > occipital > parietal > sensor-level frontal > temporal.
- Substantial ranking change: **yes** (pre-specified: a changed top group or a movement of at least two ranks).

## Audit of prior aggregation code

- `Results/CNN/CNN_Channel_Regional_Importance.py`, `update_adhd_sample_importance.py`, and `generate_fig7_fig8.py` sum channel importances within each group and then convert those sums to percentages. They do **not** divide by electrode count, so their regional comparisons are size-biased.
- `update_excel_summary.py` preserves these summed regional percentages in report tables.
- The historical scripts also use terms such as "brain region," "regional contribution," and "Brain Region." These should be replaced in future outputs with "sensor-level electrode group" and "scalp-electrode attribution." Historical artifacts are not changed by this audit.

Use `region_normalization.csv` for any regional comparison, show both total and per-electrode-normalized results, and report whenever the rankings differ. Do not infer brain-source contribution, regional activation, or causality from these scalp-sensor attributions.
