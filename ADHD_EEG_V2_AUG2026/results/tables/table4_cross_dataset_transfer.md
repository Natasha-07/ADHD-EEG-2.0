# Table 4. Cross-cohort/cross-dataset transfer

Subject-level target evaluation. These results do not isolate an age effect.

| Transfer direction | Model | N target subjects | Accuracy | Balanced Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Child dataset -> Adult dataset | Always-ADHD | 16 | 0.500 | 0.500 | 1.000 | 0.000 | 0.500 | 0.667 | 0.500 | 0.500 |
| Child dataset -> Adult dataset | Always-Control | 16 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.500 |
| Child dataset -> Adult dataset | Majority-class Dummy | 16 | 0.500 | 0.500 | 1.000 | 0.000 | 0.500 | 0.667 | 0.500 | 0.500 |
| Child dataset -> Adult dataset | Stratified Dummy | 16 | 0.500 | 0.500 | 1.000 | 0.000 | 0.500 | 0.667 | 0.094 | 0.365 |
| Child dataset -> Adult dataset | Logistic Regression (Welch band power) | 16 | 0.500 | 0.500 | 0.125 | 0.875 | 0.500 | 0.200 | 0.516 | 0.560 |
| Child dataset -> Adult dataset | Linear SVM (Welch band power) | 16 | 0.438 | 0.438 | 0.000 | 0.875 | 0.000 | 0.000 | 0.531 | 0.563 |
| Adult dataset -> Child dataset | Always-ADHD | 121 | 0.504 | 0.500 | 1.000 | 0.000 | 0.504 | 0.670 | 0.500 | 0.504 |
| Adult dataset -> Child dataset | Always-Control | 121 | 0.496 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.504 |
| Adult dataset -> Child dataset | Majority-class Dummy | 121 | 0.496 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.504 |
| Adult dataset -> Child dataset | Stratified Dummy | 121 | 0.579 | 0.579 | 0.557 | 0.600 | 0.586 | 0.571 | 0.562 | 0.547 |
| Adult dataset -> Child dataset | Logistic Regression (Welch band power) | 121 | 0.603 | 0.605 | 0.459 | 0.750 | 0.651 | 0.538 | 0.693 | 0.666 |
| Adult dataset -> Child dataset | Linear SVM (Welch band power) | 121 | 0.579 | 0.580 | 0.426 | 0.733 | 0.619 | 0.505 | 0.687 | 0.674 |
