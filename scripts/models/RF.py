import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import os
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from collections import Counter

# Ensure output folder exists
output_dir = str(OUTPUT_DIR / "shap_figures")
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(str(TRAINING_CSV), encoding='latin1')
df.columns = df.columns.str.strip()

df.reset_index(drop=True, inplace=True)
df['data_index'] = df.index  # Add unique index

# Drop unwanted columns
cols_to_drop = [
    'MagpieData range Number',
    #'MagpieData range Electronegativity',
    #'MagpieData range MeltingT',
    #'MagpieData mean MeltingT',
    #'MagpieData range AtomicRadius',
    #'MagpieData range AtomicVolume',
    #'MagpieData mean AtomicVolume',
    #'MagpieData range CovalentRadius',
    #'MagpieData range ThermalConductivity',
    #'MagpieData mean ThermalConductivity',
    #'MagpieData range Density',
    #'MagpieData range FusionEnthalpy',
    #'MagpieData mean FusionEnthalpy',
    'MagpieData range Row',
    'MagpieData range Column',
    #'duration',
    #'Shape_Polyhedral',
    #'Shape_nanosheets',
    #'Shape_quantum dots',
    #'Shape.1_Oval'
]


df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Remove outliers
print("Removing extreme (outlier) data points...")
num_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[num_cols].quantile(0.15)
Q3 = df[num_cols].quantile(0.85)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_counts = ((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).sum(axis=1)
df = df[outlier_counts < 2]

# Prepare X and y
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)', 'MIC_class'], errors='ignore')
data_indices = X['data_index'].values
X = X.drop(columns=['data_index'])
y = df['MIC_class']

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Class weights
class_sample_count = Counter(y_encoded)
n_classes = len(class_sample_count)
n_samples = len(y_encoded)
class_weights = {label: n_samples / (n_classes * count) for label, count in class_sample_count.items()}

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Hyperparameter search space
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__max_features': ['sqrt'],
    'classifier__bootstrap': [True],
    'classifier__class_weight': ['balanced']
}

results = []
importances_all = []
seeds = [3]  # Only one seed for SHAP plotting
shap_summary = None

preprocessor.fit(X)
cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

for seed in seeds:
    print(f"\n🚀 Running with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=seed))
    ])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_encoded, data_indices, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    importances = best_model.named_steps['classifier'].feature_importances_
    importances_all.append(importances)

    results.append({
        'seed': seed,
        'Accuracy': accuracy,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1_macro': f1_macro,
        'Best Params': grid_search.best_params_
    })

    # === SHAP: Compute and Save ===
    rf_model = best_model.named_steps['classifier']
    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # === SHAP Summary Plot ===
    # === SHAP Summary Plots ===
 