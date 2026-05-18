"""Shared paths for MIC classification scripts."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

TRAINING_CSV = DATA_DIR / "single_features_with_class.csv"
ENUMERATE_AG_ECOLI_CSV = DATA_DIR / "enumerate_ag_ecoli.csv"

# Legacy filenames (older script versions)
LEGACY_MBC_CSV = DATA_DIR / "single_MBC_features.csv"
LEGACY_FEATURES_CSV = DATA_DIR / "single_features.csv"
