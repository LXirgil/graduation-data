# Analysis notes

These are descriptive results from the fixed test set. No statistical significance test or minimum acceptable performance threshold was set.

## Mean Recall and F1 across class-ratio settings

### Logistic Regression

- n=500: Recall 0.244, F1 0.205 (averaged equally across the five ratios).
- n=1,000: Recall 0.253, F1 0.228 (averaged equally across the five ratios).
- n=2,000: Recall 0.254, F1 0.237 (averaged equally across the five ratios).
- n=5,000: Recall 0.256, F1 0.244 (averaged equally across the five ratios).

### Random Forest

- n=500: Recall 0.240, F1 0.198 (averaged equally across the five ratios).
- n=1,000: Recall 0.250, F1 0.208 (averaged equally across the five ratios).
- n=2,000: Recall 0.264, F1 0.227 (averaged equally across the five ratios).
- n=5,000: Recall 0.277, F1 0.243 (averaged equally across the five ratios).

## Mean Accuracy and Recall across training sizes

### Logistic Regression

- yes=50%: Accuracy 0.725, Recall 0.641, F1 0.354, PR-AUC 0.387 (averaged equally across the four training sizes).
- yes=30%: Accuracy 0.869, Recall 0.390, F1 0.412, PR-AUC 0.387 (averaged equally across the four training sizes).
- yes=10%: Accuracy 0.891, Recall 0.155, F1 0.248, PR-AUC 0.372 (averaged equally across the four training sizes).
- yes=5%: Accuracy 0.888, Recall 0.068, F1 0.123, PR-AUC 0.346 (averaged equally across the four training sizes).
- yes=1%: Accuracy 0.883, Recall 0.003, F1 0.006, PR-AUC 0.271 (averaged equally across the four training sizes).

### Random Forest

- yes=50%: Accuracy 0.732, Recall 0.669, F1 0.370, PR-AUC 0.390 (averaged equally across the four training sizes).
- yes=30%: Accuracy 0.865, Recall 0.442, F1 0.433, PR-AUC 0.399 (averaged equally across the four training sizes).
- yes=10%: Accuracy 0.890, Recall 0.141, F1 0.226, PR-AUC 0.376 (averaged equally across the four training sizes).
- yes=5%: Accuracy 0.885, Recall 0.036, F1 0.066, PR-AUC 0.342 (averaged equally across the four training sizes).
- yes=1%: Accuracy 0.883, Recall 0.000, F1 0.000, PR-AUC 0.251 (averaged equally across the four training sizes).

## Relative descriptive effects on Recall

The comparisons below use equal-weight averages over the other conditions; they are descriptive, not statistical significance tests.

- Logistic Regression: increasing training size from 500 to 5,000 changed mean Recall by +0.012; increasing yes ratio from 1% to 50% changed mean Recall by +0.638.
- Random Forest: increasing training size from 500 to 5,000 changed mean Recall by +0.037; increasing yes ratio from 1% to 50% changed mean Recall by +0.669.

## Recall-zero runs

Recall was zero in 19 of 200 individual model runs. The only model/size/ratio group with zero Recall across all five seeds was Random Forest, n=500, yes=1%. This is recorded as an observed result; the run itself had no skipped conditions.

## Interpretation note

Accuracy should be read together with yes Recall, F1, and PR-AUC. A high Accuracy alone does not show whether the model found the minority class. The project has not defined a minimum performance threshold, so choosing a minimum data size or ratio requires a decision with the supervisor.
