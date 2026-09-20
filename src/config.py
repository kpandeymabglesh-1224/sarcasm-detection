"""Shared paths and settings for both pipelines."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "Sarcasm_Headlines_Dataset.json"
SPLIT_FILE = ROOT / "data" / "split.json"     # shared train/test indices
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Same values as the original Colab code
SEED = 42
TEST_SIZE = 0.2
