from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

TARGET_CHEMBL_ID = "CHEMBL203"
TARGET_NAME = "EGFR"

ACTIVITY_TYPE = "IC50"

# Classification threshold.
# IC50 <= 1,000 nM => active
# IC50 > 1,000 nM => inactive
ACTIVE_THRESHOLD_NM = 1000.0

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048