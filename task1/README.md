# Task 1 - Cheater Detection

Task 1 builds a binary classifier to predict whether a player is a cheater.

## Notebook

- `player-intelligence-system-cpe342-ml-task1.ipynb`

## What the Notebook Covers

- Loads `public_dataset/task1/train.csv` and `public_dataset/task1/test.csv`.
- Drops identifier columns such as `id` and `player_id` before modeling.
- Removes rows where the target `is_cheater` is missing.
- Trains a baseline XGBoost classifier.
- Evaluates with F2 score because recall is important for cheat detection.
- Uses feature importance to remove weak features.
- Runs Optuna hyperparameter search.
- Experiments with a tree-based stacking ensemble.
- Writes predictions into the `task1` column of the sample submission.

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn ydata-profiling
```

## Notes

The numbers and outputs shown in the notebook cells may not correlate directly with execution order because some cells were run multiple times or skipped during experimentation. The code order is the intended reading order.
