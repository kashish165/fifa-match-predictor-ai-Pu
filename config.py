"""
config.py
=========
Central configuration for the FIFA Match Outcome Prediction System.
All magic numbers, hyperparameter ranges, and categorical mappings live here.
No logic files should contain hardcoded values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# ---------------------------------------------------------------------------
# Random seed (reproducibility)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Tournament tier mappings
# ---------------------------------------------------------------------------
TIER1_KEYWORDS = [
    "fifa world cup",
    "uefa european championship",
    "uefa euro",
    "copa america",
    "africa cup of nations",
    "african cup of nations",
    "afc asian cup",
    "concacaf gold cup",
    "ofc nations cup",
]

TIER2_KEYWORDS = [
    "qualif",
    "qualification",
    "nations league",
    "confederations cup",
    "concacaf championship",
]

TIER_WEIGHTS: dict[int, int] = {1: 3, 2: 2, 3: 1}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FORM_WINDOWS: list[int] = [5, 10]           # rolling match windows
MIN_H2H_MATCHES: int = 1                    # min H2H matches to compute H2H features
EXPERIENCE_LOG_BASE: float = 2.718281828    # natural log for experience feature
WIN_POINTS: int = 3
DRAW_POINTS: int = 1
LOSS_POINTS: int = 0

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
N_CV_SPLITS: int = 5                        # TimeSeriesSplit folds
RANDOMIZED_SEARCH_ITER: int = 50
SCORING_METRIC: str = "f1_weighted"
TARGET_COLUMN: str = "result"
TARGET_CLASSES: list[str] = ["home_win", "draw", "away_win"]

# Logistic Regression
LR_MAX_ITER: int = 1000
LR_C_VALUES: list[float] = [0.01, 0.1, 1.0, 10.0]

# Random Forest hyperparameter search space
RF_PARAM_DIST: dict = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [6, 9, 12, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
}
RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int = 12

# Gradient Boosting hyperparameter search space
GB_PARAM_DIST: dict = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__max_depth": [3, 5, 7],
    "classifier__subsample": [0.7, 0.8, 1.0],
    "classifier__min_samples_split": [2, 5],
}
GB_N_ESTIMATORS: int = 200
GB_LEARNING_RATE: float = 0.1
GB_MAX_DEPTH: int = 5

# ---------------------------------------------------------------------------
# Updater / production system
# ---------------------------------------------------------------------------
UPDATER_LOG_WINDOW: int = 100               # last N predictions for rolling accuracy
MODEL_FILENAME: str = "best_model.pkl"

# ---------------------------------------------------------------------------
# Streamlit / App
# ---------------------------------------------------------------------------
APP_TITLE: str = "FIFA Match Outcome Predictor"
APP_THEME_PRIMARY: str = "#00FF87"          # green accent
APP_THEME_BG: str = "#0E1117"              # dark background

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
EDA_MIN_MATCHES_FOR_WIN_PCT: int = 50       # min matches for team win% ranking
EDA_MIN_SHOOTOUTS_FOR_RATE: int = 3         # min shootouts for shootout win%
EDA_TOP_N_TEAMS: int = 20                   # top N teams in comparisons
EDA_ROLLING_YEARS: int = 5                  # rolling window for form charts
