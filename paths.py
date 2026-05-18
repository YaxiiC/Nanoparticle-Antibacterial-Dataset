"""Shared project paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

MIC_DATA = DATA_DIR / "single_features.csv"
MBC_DATA = DATA_DIR / "single_MBC_features.csv"
MIC_WITH_CLASS_DATA = DATA_DIR / "single_features_with_class.csv"
ENUMERATE_AG = DATA_DIR / "enumerate_ag.csv"
ENUMERATE_AG_ECOLI = DATA_DIR / "enumerate_ag_ecoli.csv"
