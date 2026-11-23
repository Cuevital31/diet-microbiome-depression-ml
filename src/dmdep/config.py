"""Analysis constants, feature names and file locations.

Locations are taken from environment variables so that no path is hard-coded:

    DMDEP_DATA_DIR   folder with the three input files (see data/README.md)       [default: ./data/input]
    DMDEP_WORK_DIR   folder for participant-level intermediate files (not shared)  [default: ./work]
    DMDEP_N_JOBS     parallel workers for the permutation tests                    [default: all cores - 2]
    DMDEP_RSCRIPT    Rscript executable used for the Firth regressions             [default: Rscript]
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DMDEP_DATA_DIR", REPO / "data" / "input")).resolve()
WORK_DIR = Path(os.environ.get("DMDEP_WORK_DIR", REPO / "work")).resolve()
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"
N_JOBS = int(os.environ.get("DMDEP_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))
RSCRIPT = os.environ.get("DMDEP_RSCRIPT", "Rscript")

# --- input files ---------------------------------------------------------------
PARTICIPANTS_FILE = "participants.tsv"        # OpenNeuro ds004648 participant metadata
TAXONOMY_FILE = "taxonomy_counts.xlsx"        # hierarchical taxonomic count table (Mothur)
DIET_FILE = "diet_daily_totals.csv"           # daily nutrient totals from the food-frequency questionnaire

# --- cohort and outcome -----------------------------------------------------------
N_PARTICIPANTS, N_DEPRESSIVE, N_CONTROL = 88, 15, 73
ZUNG_CUTOFF = 50.0                            # Zung index score (raw score x 1.25); depressive if >= 50
SAMPLE_PREFIX, PARTICIPANT_PREFIX = "c16S", "sub-"

# --- microbiome preprocessing -------------------------------------------------------
MIN_READS_PER_SAMPLE = 5000
MIN_PREVALENCE = 9                            # detected (count > 0) in >= 9 of 88 participants (about 10%)
MIN_TOTAL_COUNT = 10
CLR_PSEUDOCOUNT = 0.5
KINGDOM_RANK_ID = "0.2"                       # Bacteria

# --- dietary variables ------------------------------------------------------------------
DIET_RAW = ("Kcal", "Prot", "TF", "SatFat", "MUF", "Poly_InsFat", "Chol", "Carb", "Fiber")
NUTRIENTS = ("Prot", "TF", "SatFat", "MUF", "Poly_InsFat", "Chol", "Carb", "Fiber")
KCAL_PER_GRAM = {"Prot": 4.0, "Carb": 4.0, "TF": 9.0, "SatFat": 9.0, "MUF": 9.0, "Poly_InsFat": 9.0}
DIET_NAMES = {                                # energy-adjusted variables (PE = % of energy, PK = per 1,000 kcal)
    "Prot": "Prot_PE", "TF": "TF_PE", "SatFat": "SatFat_PE", "MUF": "MUF_PE",
    "Poly_InsFat": "Poly_InsFat_PE", "Chol": "Chol_PK", "Carb": "Carb_PE", "Fiber": "Fiber_PK",
}
CORR_THRESHOLD = 0.90

# --- confounders ---------------------------------------------------------------------------
CONFOUNDERS = ("Age", "BMI", "SQ_all", "PA")  # SQ_all = natural logarithm of the sympathovagal quotient

# --- validation scheme ---------------------------------------------------------------------
SEEDS = (42, 43, 44, 45, 46)
FS_FRACTION = 0.40                            # feature-selection subset
TEST_FRACTION_OF_REST = 1.0 / 3.0             # test subset = 1/3 of the remaining 60%
N_TEST, N_TEST_DEPRESSIVE, N_TEST_CONTROL = 18, 3, 15
CLASSIFIER_RANDOM_STATE = 42

# --- taxa feature selection ---------------------------------------------------------------------
ANOVA_P_THRESHOLD = 0.05
MI_PERCENTILE = 95
RF_TREES = 500
CONSENSUS_MIN_METHODS = 2

# --- feature sets and models ----------------------------------------------------------------------
MODELS = ("LR", "SVM", "BRF")
FEATURE_SETS = ("Confounders", "Diet", "Taxa", "Diet + Taxa", "Confounders + Diet + Taxa")
MAIN_CONFIG = ("Confounders + Diet + Taxa", "LR")
METRICS = ("ROC-AUC", "ACC", "BAL-ACC", "SEN", "SPE", "PRE", "F1")

# --- uncertainty, calibration, permutation ---------------------------------------------------------
CALIB_CV_FOLDS, CALIB_CV_SEED, CALIB_N_BINS = 3, 42, 4
BOOTSTRAP_B, BOOTSTRAP_SEED = 2000, 2026
PERM_B, PERM_BASE_SEED = 1000, 20260
PERM_FAMILIES = {                             # family label -> feature sets
    "15": FEATURE_SETS,
    "12": ("Confounders", "Diet", "Taxa", "Confounders + Diet + Taxa"),
    "9": ("Diet", "Taxa", "Diet + Taxa"),
}

# --- feature prioritisation and Firth regression ----------------------------------------------------
MIN_SPLITS_RETAINED = 2
REFERENCE_MIN_PREVALENCE_FRACTION = 0.20      # reference family: taxa with count > 0 in >= 18 of 88
FIRTH_MAXIT = 100
PCT_CHANGE_MIN_BETA = 0.1                     # percentage changes only when |reference beta| >= 0.1

# --- confounder-residualized sensitivity analysis -----------------------------------------------------
RESID_MODELS = ("LR", "SVM")
RESID_SETS = ("Diet_adj", "Taxa_adj", "Diet_adj + Taxa_adj", "Confounders + Diet_adj + Taxa_adj")

# --- prediction-level adjustment --------------------------------------------------------------------------
PL_BOOTSTRAP_B, PL_BOOTSTRAP_SEED, PL_CLIP = 2000, 2026, 1e-6


def configurations(feature_sets=FEATURE_SETS, models=MODELS):
    """(feature set, model) pairs in table order."""
    return [(fs, m) for fs in feature_sets for m in models]


def config_label(config) -> str:
    return f"{config[0]} | {config[1]}"
