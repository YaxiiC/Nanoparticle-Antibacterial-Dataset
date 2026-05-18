# Nanoparticle Antibacterial Dataset

This repository contains curated nanoparticle antibacterial activity datasets and a small baseline machine-learning script. The goal is to keep the project easy to run and easy to extend.

## Contents

```text
data/                  CSV datasets
scripts/train_models.py Baseline MIC/MBC model training
paths.py               Shared data and output paths
requirements.txt       Python dependencies
```

## Datasets

| File | Purpose |
| --- | --- |
| `data/single_features.csv` | MIC regression dataset. |
| `data/single_MBC_features.csv` | MBC regression dataset. |
| `data/single_features_with_class.csv` | MIC dataset with `strong`, `moderate`, and `weak` class labels. |
| `data/enumerate_ag.csv` | Candidate Ag nanoparticle combinations. |
| `data/enumerate_ag_ecoli.csv` | Candidate Ag nanoparticle combinations for E. coli. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Train all baseline models:

```bash
python scripts/train_models.py
```

Train one task:

```bash
python scripts/train_models.py --task mic_class
python scripts/train_models.py --task mic_regression
python scripts/train_models.py --task mbc_regression
```

The script writes trained models and metrics to `outputs/`.

## Notes

- The baseline script uses simple random-forest models with shared preprocessing for numeric and categorical columns.
- Regression targets are trained with a `log1p` transform and evaluated on the original concentration scale.
- Generated outputs are ignored by Git so the repository stays focused on source data and code.
