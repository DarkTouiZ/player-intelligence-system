# Task 4 - Game Title Detection

Task 4 classifies game-title screenshots into five classes using transfer learning.

## Notebook

- `task4.ipynb`

## What the Notebook Covers

- Installs fastai, `timm`, Kaggle, image hashing, and supporting utility packages.
- Downloads the `cpe342-karena` Kaggle dataset in Colab.
- Loads image metadata from `public_dataset/task4`.
- Checks train/validation class distributions.
- Searches for visually similar or duplicated images to avoid data leakage.
- Trains a transfer-learning model based on `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`.
- Uses augmentation strategies such as ColorJitter, CutMix, and RandomErasing.
- Applies early stopping, weight decay, learning-rate selection, and test-time augmentation.
- Visualizes prediction errors and prepares submission predictions.

## Dependencies

```bash
pip install fastai jupyter timm torchtnt kaggle imagehash albumentations
pip install cjm_pandas_utils cjm_psl_utils cjm_pil_utils cjm_pytorch_utils cjm_torchvision_tfms
```

## Colab Setup

The notebook reads a Kaggle API token from Colab secrets:

```python
from google.colab import userdata
kaggle_json_str = userdata.get("KAGGLE_API_TOKEN")
```

Before running the download cells, add your Kaggle token to Colab secrets with the name `KAGGLE_API_TOKEN`.

## Reproducibility Notes

The outputs shown in the notebook may not align perfectly with the current code. Several new approaches were tested and later rolled back to the strongest known version, but the original A100 GPU compute budget was exhausted. Treat the checked-in notebook as an experiment record plus implementation reference rather than a guaranteed one-click reproduction.
