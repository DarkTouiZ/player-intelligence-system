# Player Intelligence System

Machine learning notebooks for a multi-task player intelligence pipeline. The project explores player risk, segmentation, monetization, game-title recognition, and account-security monitoring using the `cpe342-karena` Kaggle dataset.

## Project Overview

The repository is organized as five independent modeling tasks. Each task contains one or more notebooks with exploratory analysis, preprocessing, model experiments, evaluation, and submission-generation code.

| Task | Problem | Main approach | Location |
| --- | --- | --- | --- |
| 1 | Cheater detection | XGBoost classifier and tree-based stacking ensemble | [`task1/`](task1/) |
| 2 | Player segmentation | Multiclass classification with XGBoost, LightGBM, CatBoost, and stacking experiments | [`task2/`](task2/) |
| 3 | 30-day spending prediction | Feature engineering, Tweedie regression, and two-stage classification/regression modeling | [`task3/`](task3/) |
| 4 | Game title detection from images | Transfer learning with `timm`/fastai vision models and test-time augmentation | [`task4/`](task4/) |
| 5 | Account security anomaly detection | Isolation Forest, One-Class SVM, PCA analysis, and ensemble anomaly scoring | [`task5/`](task5/) |

## Repository Structure

```text
.
|-- task1/
|   |-- README.md
|   `-- player-intelligence-system-cpe342-ml-task1.ipynb
|-- task2/
|   |-- README.md
|   |-- player-intelligence-system-cpe342-ml-task2-part1.ipynb
|   |-- player-intelligence-system-cpe342-ml-task2-part2.ipynb
|   `-- player-intelligence-system-cpe342-ml.ipynb
|-- task3/
|   |-- README.md
|   |-- EDA.ipynb
|   |-- Data_preprocessing.ipynb
|   |-- Tweedie Modeling.ipynb
|   `-- Two-Stage Modeling.ipynb
|-- task4/
|   |-- README.md
|   `-- task4.ipynb
`-- task5/
    |-- README.md
    `-- task5_anomaly_detection.ipynb
```

## Data

The notebooks expect the Kaggle competition dataset `cpe342-karena`. Most notebooks were developed in Kaggle or Google Colab, so some paths are environment-specific:

- Kaggle-style paths: `/kaggle/input/cpe342-karena/public_dataset/...`
- Colab-style paths: `/content/public_dataset/...`
- Local task files: `train.csv`, `test.csv`, and preprocessed CSVs beside the notebook

To run locally, download the dataset from Kaggle and either update the notebook paths or place each task dataset in the expected working directory.

For Task 4 in Colab, add a Kaggle API token to Colab secrets as `KAGGLE_API_TOKEN`; the notebook writes it to `/root/.kaggle/kaggle.json` before downloading the dataset.

## Environment

The notebooks use Python and common data science libraries. Install only the packages needed for the task you are running.

Core tabular modeling:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost optuna scipy plotly ydata-profiling tqdm
```

Task 4 computer vision:

```bash
pip install fastai jupyter timm torchtnt kaggle imagehash albumentations
pip install cjm_pandas_utils cjm_psl_utils cjm_pil_utils cjm_pytorch_utils cjm_torchvision_tfms
```

Task 4 was designed for GPU execution. The checked-in notes mention A100 usage for the strongest experiments.

## Recommended Run Order

Run each task independently. For Task 3, the notebooks are connected and should be run in this order:

1. `task3/EDA.ipynb`
2. `task3/Data_preprocessing.ipynb`
3. `task3/Tweedie Modeling.ipynb`
4. `task3/Two-Stage Modeling.ipynb`

Task 1, Task 2, Task 4, and Task 5 can be opened directly, provided the corresponding dataset files are available at the paths used in the notebook.

## Reproducibility Notes

- Several notebooks were used as experiment logs. Some cell outputs may not match the visible execution order because cells were run multiple times during modeling.
- Task 4 contains results from GPU-heavy experiments that may not be fully reproducible without the same compute budget.
- Generated files such as profiling reports, submissions, and intermediate preprocessed CSVs are created by the notebooks and are not all committed to the repository.

## Outputs

The notebooks generate model diagnostics, plots, feature-importance views, and submission-ready CSV columns for the relevant task. Where a task contributes to a combined submission file, the notebook updates the corresponding task column such as `task1`, `task3`, or `is_anomaly`.
