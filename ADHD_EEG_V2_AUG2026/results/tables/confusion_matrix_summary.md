# Subject-level confusion-matrix summary

| Evaluation | Model | N subjects | TN | FP | FN | TP | Sensitivity | Specificity | Balanced Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main combined holdout | Always-ADHD | 28 | 0 | 14 | 0 | 14 | 1.000 | 0.000 | 0.500 |
| Main combined holdout | Always-Control | 28 | 14 | 0 | 14 | 0 | 0.000 | 1.000 | 0.500 |
| Main combined holdout | Majority-class Dummy | 28 | 14 | 0 | 14 | 0 | 0.000 | 1.000 | 0.500 |
| Main combined holdout | Stratified Dummy | 28 | 11 | 3 | 12 | 2 | 0.143 | 0.786 | 0.464 |
| Main combined holdout | Logistic Regression (Welch band power) | 28 | 5 | 9 | 2 | 12 | 0.857 | 0.357 | 0.607 |
| Main combined holdout | Linear SVM (Welch band power) | 28 | 5 | 9 | 2 | 12 | 0.857 | 0.357 | 0.607 |
| Child dataset -> Adult dataset | Always-ADHD | 16 | 0 | 8 | 0 | 8 | 1.000 | 0.000 | 0.500 |
| Child dataset -> Adult dataset | Always-Control | 16 | 8 | 0 | 8 | 0 | 0.000 | 1.000 | 0.500 |
| Child dataset -> Adult dataset | Majority-class Dummy | 16 | 0 | 8 | 0 | 8 | 1.000 | 0.000 | 0.500 |
| Child dataset -> Adult dataset | Stratified Dummy | 16 | 0 | 8 | 0 | 8 | 1.000 | 0.000 | 0.500 |
| Child dataset -> Adult dataset | Logistic Regression (Welch band power) | 16 | 7 | 1 | 7 | 1 | 0.125 | 0.875 | 0.500 |
| Child dataset -> Adult dataset | Linear SVM (Welch band power) | 16 | 7 | 1 | 8 | 0 | 0.000 | 0.875 | 0.438 |
| Adult dataset -> Child dataset | Always-ADHD | 121 | 0 | 60 | 0 | 61 | 1.000 | 0.000 | 0.500 |
| Adult dataset -> Child dataset | Always-Control | 121 | 60 | 0 | 61 | 0 | 0.000 | 1.000 | 0.500 |
| Adult dataset -> Child dataset | Majority-class Dummy | 121 | 60 | 0 | 61 | 0 | 0.000 | 1.000 | 0.500 |
| Adult dataset -> Child dataset | Stratified Dummy | 121 | 36 | 24 | 27 | 34 | 0.557 | 0.600 | 0.579 |
| Adult dataset -> Child dataset | Logistic Regression (Welch band power) | 121 | 45 | 15 | 33 | 28 | 0.459 | 0.750 | 0.605 |
| Adult dataset -> Child dataset | Linear SVM (Welch band power) | 121 | 44 | 16 | 35 | 26 | 0.426 | 0.733 | 0.580 |
