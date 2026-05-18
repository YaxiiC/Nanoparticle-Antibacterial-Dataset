# MIC Class Prediction (Nanoparticle Features)

Machine-learning pipelines to classify minimum inhibitory concentration (MIC) strength (`strong` / `moderate` / `weak`) from nanoparticle composition and morphology features.

## Repository layout

```
github_upload/
├── data/                    # Training & enumeration CSVs (see below)
├── outputs/                 # Generated figures (gitignored by default)
├── paths.py                 # Shared data/output paths
├── requirements.txt
└── scripts/
    ├── models/              # Classifier training (RF, XGBoost, LightGBM, …)
    ├── shap/                # SHAP explainability plots
    ├── analysis/            # Outlier checks, candidate enumeration
    └── visualization/       # 3D MIC-class scatter plots
```

## Setup

```bash
cd github_upload
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `single_features_with_class.csv` in `data/` if it is not already there (included when built from the local project).

## Run examples

From the repo root (`github_upload/`):

```bash
# Random forest + SHAP
python scripts/models/RF.py

# XGBoost (Oct 2024 tuning pipeline)
python scripts/models/XGBoost1004.py

# PyTorch MLP (file RF1004.py — not random forest)
python scripts/models/RF1004.py

# 3D visualization
python scripts/visualization/plot_3d_mic_classes.py enumerate_ag_ecoli.csv -o outputs/mic_3d.png
```

## Script notes

| Script | Description |
|--------|-------------|
| `RF.py` | Random Forest with grid search + SHAP |
| `XGBoost1004.py` | XGBoost with randomized search (recommended) |
| `XGBoost.py` | XGBoost + SHAP waterfall export |
| `RF1004.py` | **PyTorch MLP** (historical filename) |
| `enumerate_xgboost.py` | Score enumerated Ag candidates |
| `plot_3d_mic_classes.py` | 3D Shape × synthesis × size plot |

Older scripts (`ElasticNet.py`, `knn.py`, …) use legacy CSV names in `data/` (`single_MBC_features.csv`, `single_features.csv`).

## What was excluded from the parent folder

The working directory `model_classification/` also contained:

- `shap_*_old/` — thousands of regenerated PNG plots
- `unit_alpha_ai_segmentation/` — unrelated medical-imaging unit tests (accidentally nested under an old SHAP folder)
- Root-level `.png` outputs — regenerate via the scripts above

Those are **not** copied here to keep the GitHub repo small and focused on code.

## Upload to GitHub

```bash
cd github_upload
git init
git add .
git commit -m "Initial commit: MIC classification pipelines"
gh repo create mic-nanoparticle-classification --public --source=. --push
```

(Or create the repo on github.com and `git remote add origin …` then `git push -u origin main`.)
