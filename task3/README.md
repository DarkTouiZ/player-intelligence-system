# Task 3 - Spending Prediction

Task 3 predicts each player's spending over the next 30 days. The target, `spending_30d`, is continuous, non-negative, right-skewed, and includes many zero values, so the task is treated as both a regression and zero-inflated prediction problem.

## Notebooks

Run these notebooks in order:

1. `EDA.ipynb`
2. `Data_preprocessing.ipynb`
3. `Tweedie Modeling.ipynb`
4. `Two-Stage Modeling.ipynb`

## Workflow

### 1. Exploratory Data Analysis

`EDA.ipynb` inspects missing values, categorical/ordinal features, log-transform candidates, target skew, and feature correlations.

Key observations captured in the notebook:

- Missing values appear broadly across features.
- Spending-related features such as `historical_spending`, `prev_month_spending`, and `avg_transaction_value` are log-transform candidates.
- Highly correlated features are reviewed to reduce redundancy.
- Seasonal spending can be represented with cyclical features.

### 2. Data Preprocessing

`Data_preprocessing.ipynb` performs type casting, missing-value imputation, feature engineering, categorical encoding, robust scaling, and high-correlation feature removal. It writes preprocessed CSV files used by the modeling notebooks.

### 3. Tweedie Modeling

`Tweedie Modeling.ipynb` models spending directly with a Tweedie-style objective, which is suitable for non-negative, skewed targets with many zero values.

### 4. Two-Stage Modeling

`Two-Stage Modeling.ipynb` separates the problem into:

- Stage 1: binary classification for whether a player will spend.
- Stage 2: regression for the expected spending amount among likely spenders.

The final prediction combines the probability of spending with the predicted amount.

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn scipy plotly ydata-profiling tqdm
```

## Data Expectations

The notebooks expect `train.csv`, `test.csv`, and generated preprocessed files such as `train_preprocessed.csv` or `train_preprocessed_v4.csv` in the working directory. Adjust paths if running outside the original notebook environment.
