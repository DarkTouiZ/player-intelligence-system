# Task 2 - Player Segmentation

Task 2 predicts a player's segment using tabular profile, engagement, and behavioral features.

## Notebooks

- `player-intelligence-system-cpe342-ml-task2-part1.ipynb`
- `player-intelligence-system-cpe342-ml-task2-part2.ipynb`
- `player-intelligence-system-cpe342-ml.ipynb`

## What the Notebooks Cover

### Part 1

- Loads `public_dataset/task2/train.csv`.
- Drops identifier and noisy random metric columns.
- Encodes ordinal and nominal categorical features.
- Trains a simple XGBoost multiclass classifier.
- Evaluates with macro, micro, and weighted F1 score.
- Uses feature importance and Optuna tuning.
- Compares imputation and model variants, including LightGBM, CatBoost, and logistic regression.

### Part 2

- Performs exploratory data analysis.
- Experiments with a custom stacking ensemble.

### Feature Linkage Experiment

- Investigates feature relationships and linkage-style experiments used during model iteration.

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn ydata-profiling
```

## Notes

The numbers and outputs shown in notebook cells may not correlate directly with execution order because some cells were run multiple times or skipped during experimentation. The code order is the intended reading order.
