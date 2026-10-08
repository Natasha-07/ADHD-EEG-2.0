# Table 2. Main subject-level model performance

Rows marked NA are retained neural models, but no newly generated reviewer-revision subject-level neural results exist. Old CleanDeepBenchmark results were not used.

| Model | N subjects | Accuracy | Balanced Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always-ADHD | 28 | 0.500 | 0.500 | 1.000 | 0.000 | 0.500 | 0.667 | 0.500 | 0.500 |
| Always-Control | 28 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.500 |
| Majority-class Dummy | 28 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.500 |
| Stratified Dummy | 28 | 0.464 | 0.464 | 0.143 | 0.786 | 0.400 | 0.211 | 0.571 | 0.555 |
| Logistic Regression (Welch band power) | 28 | 0.607 | 0.607 | 0.857 | 0.357 | 0.571 | 0.686 | 0.648 | 0.619 |
| Linear SVM (Welch band power) | 28 | 0.607 | 0.607 | 0.857 | 0.357 | 0.571 | 0.686 | 0.633 | 0.595 |
| CNN | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| LSTM | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| Bi-LSTM | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| Temporal Transformer Baseline | NA | NA | NA | NA | NA | NA | NA | NA | NA |
