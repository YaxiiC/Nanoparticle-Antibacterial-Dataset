# Nanoparticle Antibacterial Activity — ML Regression Models

Machine-learning pipeline for predicting the antibacterial activity (MIC / MBC) of nanoparticles. Seven regression models are benchmarked with GridSearchCV, SHAP interpretability analysis, and grouped cross-validation experiments designed to address Ag-dominance bias.

## Repository Structure

```
github_upload/
├── README.md
├── requirements.txt
├── .gitignore
├── paths.py                        # Centralised data / output paths
├── data/                           # Datasets (CSV)
│   ├── single_features.csv         # MIC regression dataset
│   ├── single_MBC_features.csv     # MBC regression dataset
│   ├── single_features_with_class.csv  # MIC with class labels
│   ├── enumerate_ag.csv            # Enumeration dataset (Ag)
│   └── enumerate_ag_ecoli.csv      # Enumeration dataset (E. coli)
├── outputs/                        # Generated results (git-ignored)
└── scripts/
    ├── models/                     # Regression model training
    │   ├── RF.py                   # Random Forest (MIC)
    │   ├── SVM.py                  # Support Vector Machine (MBC)
    │   ├── XGBoost.py              # XGBoost + classification metrics (MIC)
    │   ├── knn.py                  # K-Nearest Neighbours (MBC)
    │   ├── Lasso.py                # Lasso (MBC)
    │   ├── ElasticNet.py           # ElasticNet (MBC)
    │   └── mlp.py                  # Multi-Layer Perceptron (MIC)
    ├── shap/                       # SHAP explainability
    │   └── RF_shap.py              # RF + TreeExplainer analysis
    ├── analysis/                   # Outlier removal & advanced experiments
    │   ├── RF_outlier.py           # RF with IQR outlier removal
    │   ├── enumerate_XGBoost.py    # XGBoost + enumeration predictions
    │   └── grouped_cv.py           # 5 grouped CV experiments (reviewer response)
    └── visualization/
        └── plot_3d_mic_values.py   # 3D scatter + applicability domain
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a model (e.g. Random Forest for MIC)
python scripts/models/RF.py

# 5. Run SHAP analysis
python scripts/shap/RF_shap.py

# 6. Run grouped cross-validation experiments
python scripts/analysis/grouped_cv.py

# 7. 3D visualisation (requires an enumeration CSV with predictions)
python scripts/visualization/plot_3d_mic_values.py data/enumerate_ag.csv \
    --value-col "MIC_pred (ug/mL)" --out outputs/3d_scatter.png
```

## Datasets

| File | Description | Target |
|------|-------------|--------|
| `single_features.csv` | Main MIC regression dataset with Magpie descriptors | MIC (ug/mL) |
| `single_MBC_features.csv` | MBC regression dataset | MBC (ug/mL) |
| `single_features_with_class.csv` | MIC dataset with class labels (strong / moderate / weak) | MIC (ug/mL) + MIC_class |
| `enumerate_ag.csv` | Enumerated Ag nanoparticle combinations for prediction | — |
| `enumerate_ag_ecoli.csv` | Enumerated Ag nanoparticle combinations (E. coli) | — |

## Models

All models use **log1p** target transformation during training and report metrics on the original scale after inverse transformation (expm1).

| Model | Script | Hyperparameter Search |
|-------|--------|-----------------------|
| Random Forest | `scripts/models/RF.py` | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| XGBoost | `scripts/models/XGBoost.py` | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |
| SVM (SVR) | `scripts/models/SVM.py` | kernel, C, epsilon, gamma |
| KNN | `scripts/models/knn.py` | n_neighbors, weights, p |
| Lasso | `scripts/models/Lasso.py` | alpha |
| ElasticNet | `scripts/models/ElasticNet.py` | alpha, l1_ratio |
| MLP | `scripts/models/mlp.py` | hidden_layer_sizes, activation, alpha, learning_rate |

## Grouped Cross-Validation Experiments

`scripts/analysis/grouped_cv.py` runs five experiments to address Ag-dominance concerns:

1. **GroupKFold** — 5-fold CV grouped by Formula
2. **Leave-One-Material-Out (LOMO)** — hold out each material with n >= 5
3. **Leave-Ag-Out** — train on non-Ag, test on Ag
4. **Material-Family GroupKFold** — grouped by material families (pure metal, metal oxide, chalcogenide, alloy, carbon)
5. **Downsampled-Ag** — cap Ag samples to match 2nd-largest material, repeated over 10 seeds

Results are saved to `outputs/results_grouped_cv/`.

## Upload to GitHub

```bash
cd github_upload
git init
git add .
git commit -m "Initial commit: nanoparticle antibacterial ML pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
