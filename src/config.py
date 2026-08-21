from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

TARGET_CHEMBL_ID = "CHEMBL203"
TARGET_NAME = "EGFR"

ACTIVITY_TYPE = "IC50"
ACTIVITY_UNIT = "nM"

# Project classification rule.
# IC50 <= 1,000 nM -> ACTIVE
# IC50 >  1,000 nM -> INACTIVE
ACTIVE_THRESHOLD_NM = 1000.0


# ---------------------------------------------------------
# Molecular representation
# ---------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048


# ---------------------------------------------------------
# ML
# ---------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

RAW_ACTIVITY_FILE = (
    RAW_DATA_DIR / "egfr_ic50_raw.csv"
)

CURATED_DATA_FILE = (
    PROCESSED_DATA_DIR / "egfr_ic50_curated.csv"
)

FEATURE_FILE = (
    PROCESSED_DATA_DIR / "egfr_features.npz"
)

MODEL_METADATA_FILE = (
    PROCESSED_DATA_DIR / "egfr_model_metadata.csv"
)

MODEL_FILE = (
    MODELS_DIR / "bioactivity_model.joblib"
)

MODEL_COMPARISON_FILE = (
    REPORTS_DIR / "model_comparison.csv"
)

DATA_QUALITY_FILE = (
    REPORTS_DIR / "data_quality.json"
)