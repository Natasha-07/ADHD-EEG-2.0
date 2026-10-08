# Table 5. Statistical model comparisons

Formatted table shows the prespecified primary metric (balanced accuracy); the CSV contains every tested metric and comparison. All tests are paired at subject level.

| Scenario | Model A | Model B | Metric | Difference (A - B) | 95% CI low | 95% CI high | Test statistic | Raw p-value | Holm-corrected p-value | Holm significant (0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult_to_child | Always-ADHD | Always-Control | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| adult_to_child | Always-ADHD | Linear SVM (Welch band power) | Balanced_Accuracy | -0.080 | -0.162 | 0.003 | -0.080 | 0.247 | 1.000 | False |
| adult_to_child | Always-ADHD | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.105 | -0.187 | -0.022 | -0.105 | 0.149 | 1.000 | False |
| adult_to_child | Always-ADHD | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| adult_to_child | Always-ADHD | Stratified Dummy | Balanced_Accuracy | -0.079 | -0.161 | 0.004 | -0.079 | 0.202 | 1.000 | False |
| adult_to_child | Always-Control | Linear SVM (Welch band power) | Balanced_Accuracy | -0.080 | -0.162 | -0.005 | -0.080 | 0.165 | 1.000 | False |
| adult_to_child | Always-Control | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.105 | -0.187 | -0.014 | -0.105 | 0.069 | 1.000 | False |
| adult_to_child | Always-Control | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| adult_to_child | Always-Control | Stratified Dummy | Balanced_Accuracy | -0.079 | -0.170 | 0.004 | -0.079 | 0.252 | 1.000 | False |
| adult_to_child | Linear SVM (Welch band power) | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.025 | -0.066 | 0.016 | -0.025 | 0.348 | 1.000 | False |
| adult_to_child | Linear SVM (Welch band power) | Majority-class Dummy | Balanced_Accuracy | 0.080 | -0.003 | 0.162 | 0.080 | 0.182 | 1.000 | False |
| adult_to_child | Linear SVM (Welch band power) | Stratified Dummy | Balanced_Accuracy | 0.001 | -0.131 | 0.125 | 0.001 | 0.912 | 1.000 | False |
| adult_to_child | Logistic Regression (Welch band power) | Majority-class Dummy | Balanced_Accuracy | 0.105 | 0.022 | 0.180 | 0.105 | 0.061 | 1.000 | False |
| adult_to_child | Logistic Regression (Welch band power) | Stratified Dummy | Balanced_Accuracy | 0.026 | -0.098 | 0.158 | 0.026 | 0.620 | 1.000 | False |
| adult_to_child | Majority-class Dummy | Stratified Dummy | Balanced_Accuracy | -0.079 | -0.161 | 0.005 | -0.079 | 0.254 | 1.000 | False |
| child_to_adult | Always-ADHD | Always-Control | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-ADHD | Linear SVM (Welch band power) | Balanced_Accuracy | 0.062 | 0.000 | 0.188 | 0.062 | 1.000 | 1.000 | False |
| child_to_adult | Always-ADHD | Logistic Regression (Welch band power) | Balanced_Accuracy | 0.000 | -0.125 | 0.188 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-ADHD | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-ADHD | Stratified Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-Control | Linear SVM (Welch band power) | Balanced_Accuracy | 0.062 | 0.000 | 0.188 | 0.062 | 1.000 | 1.000 | False |
| child_to_adult | Always-Control | Logistic Regression (Welch band power) | Balanced_Accuracy | 0.000 | -0.188 | 0.127 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-Control | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Always-Control | Stratified Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Linear SVM (Welch band power) | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.062 | -0.188 | 0.000 | -0.062 | 1.000 | 1.000 | False |
| child_to_adult | Linear SVM (Welch band power) | Majority-class Dummy | Balanced_Accuracy | -0.062 | -0.188 | 0.000 | -0.062 | 1.000 | 1.000 | False |
| child_to_adult | Linear SVM (Welch band power) | Stratified Dummy | Balanced_Accuracy | -0.062 | -0.188 | 0.000 | -0.062 | 1.000 | 1.000 | False |
| child_to_adult | Logistic Regression (Welch band power) | Majority-class Dummy | Balanced_Accuracy | 0.000 | -0.188 | 0.127 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Logistic Regression (Welch band power) | Stratified Dummy | Balanced_Accuracy | 0.000 | -0.125 | 0.125 | 0.000 | 1.000 | 1.000 | False |
| child_to_adult | Majority-class Dummy | Stratified Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| combined_holdout | Always-ADHD | Always-Control | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| combined_holdout | Always-ADHD | Linear SVM (Welch band power) | Balanced_Accuracy | -0.107 | -0.250 | 0.036 | -0.107 | 0.451 | 1.000 | False |
| combined_holdout | Always-ADHD | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.107 | -0.250 | 0.036 | -0.107 | 0.483 | 1.000 | False |
| combined_holdout | Always-ADHD | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| combined_holdout | Always-ADHD | Stratified Dummy | Balanced_Accuracy | 0.036 | -0.107 | 0.179 | 0.036 | 1.000 | 1.000 | False |
| combined_holdout | Always-Control | Linear SVM (Welch band power) | Balanced_Accuracy | -0.107 | -0.250 | 0.071 | -0.107 | 0.684 | 1.000 | False |
| combined_holdout | Always-Control | Logistic Regression (Welch band power) | Balanced_Accuracy | -0.107 | -0.250 | 0.036 | -0.107 | 0.633 | 1.000 | False |
| combined_holdout | Always-Control | Majority-class Dummy | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| combined_holdout | Always-Control | Stratified Dummy | Balanced_Accuracy | 0.036 | -0.107 | 0.179 | 0.036 | 1.000 | 1.000 | False |
| combined_holdout | Linear SVM (Welch band power) | Logistic Regression (Welch band power) | Balanced_Accuracy | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | False |
| combined_holdout | Linear SVM (Welch band power) | Majority-class Dummy | Balanced_Accuracy | 0.107 | -0.036 | 0.250 | 0.107 | 0.665 | 1.000 | False |
| combined_holdout | Linear SVM (Welch band power) | Stratified Dummy | Balanced_Accuracy | 0.143 | -0.036 | 0.321 | 0.143 | 0.455 | 1.000 | False |
| combined_holdout | Logistic Regression (Welch band power) | Majority-class Dummy | Balanced_Accuracy | 0.107 | -0.071 | 0.250 | 0.107 | 0.657 | 1.000 | False |
| combined_holdout | Logistic Regression (Welch band power) | Stratified Dummy | Balanced_Accuracy | 0.143 | -0.036 | 0.321 | 0.143 | 0.460 | 1.000 | False |
| combined_holdout | Majority-class Dummy | Stratified Dummy | Balanced_Accuracy | 0.036 | -0.107 | 0.179 | 0.036 | 1.000 | 1.000 | False |
