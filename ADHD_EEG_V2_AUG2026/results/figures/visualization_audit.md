# EEG visualization audit and replacement

## Retired visualizations

`generate_fig7_fig8.py` overlays sensor-group values on anatomical-reference templates and is retired from the revised workflow. `Results/CNN/CNN_Channel_Regional_Importance.py` produces summed sensor-group graphics and remains historical only. Neither is source localization, but both can overstate anatomical interpretation.

The available manuscript source (`main_fixed.tex`) contains benchmark tables but no Tables 3 or 4 identified as electrode/region-attribution tables. Therefore no table was silently removed. Where the reviewer means the legacy electrode/region displays, replace those manuscript placements with the topoplots here and retain `electrode_attributions.csv` as the supplementary numeric table.

## Revised topoplots

These figures use correctly matching channel names and coordinates from MNE's packaged `standard_1020.elc` montage. They display interpolated **sensor-space scalp topographies** of mean absolute Integrated Gradients attribution. They are not source reconstruction, brain images, or evidence of localized neural activity. ADHD, Control, and overall maps use one consistent scale within each model. Numeric sensor-level values remain in the XAI CSV files.

## Difference-map decision

- CNN: no difference topoplot generated; no electrode survived BH-FDR < 0.05.
- LSTM: no difference topoplot generated; no electrode survived BH-FDR < 0.05.

Difference maps require two-sided permutation testing and Benjamini-Hochberg FDR < 0.05 across 19 electrodes; details are in `topoplot_difference_statistics.csv`. Do not claim frontal-lobe activation, brain-source contribution, or source localization from these figures.
