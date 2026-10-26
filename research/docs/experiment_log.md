# Experiment log

### Quick run — 2026-10-26T13:56:33+09:00

- Results file: `../results/quick_results_2026-10-26.csv`
- Result rows: 2
- Skipped conditions: 0
- Fixed split: train 31647 rows (3702 yes), test 13564 rows (1587 yes); 70/30, stratified, seed 42
- Models: logistic_regression, random_forest
- Training sizes: [500]
- Requested yes ratios: [0.1]
- Sampling seeds: [42]
- Metrics: accuracy, yes precision, yes recall, yes F1 at threshold 0.5, and PR-AUC (Average Precision)
- Recall-zero rows: 0/2
- Groups with recall zero for every seed: []
- Versions: pandas 3.1.0, scikit-learn 1.9.1, ucimlrepo 0.0.7
### Full experiment — 2026-10-26T13:56:49+09:00

- Results file: `../results/results_2026-10-26.csv`
- Result rows: 200
- Skipped conditions: 0
- Fixed split: train 31647 rows (3702 yes), test 13564 rows (1587 yes); 70/30, stratified, seed 42
- Models: logistic_regression, random_forest
- Training sizes: [500, 1000, 2000, 5000]
- Requested yes ratios: [0.5, 0.3, 0.1, 0.05, 0.01]
- Sampling seeds: [42, 43, 44, 45, 46]
- Metrics: accuracy, yes precision, yes recall, yes F1 at threshold 0.5, and PR-AUC (Average Precision)
- Recall-zero rows: 19/200
- Groups with recall zero for every seed: [('random_forest', 500, 0.01)]
- Versions: pandas 3.1.0, scikit-learn 1.9.1, ucimlrepo 0.0.7
