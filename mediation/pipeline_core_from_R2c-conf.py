"""pipeline_core.py - single source of truth for the Revision-R2c-conf analyses.

Revision-R2c-conf: the ML setup requested by Zerrin Hoca on 14 Nov 2026, with
CONFOUNDERS as an additional feature set. Taxa table: ALL 249 taxa (the ten "uncultured" taxa kept), as in Revision-R2c.

Feature sets (5) x models (LR, SVM, BRF) = 15 configurations:
    Confounders | Diet | Taxa | Diet + taxa | Confounders + diet + taxa
Zerrin Hoca's setup = the four sets without "Diet + taxa" (12 configurations);
"Diet + taxa" is kept because it is the set of Revision-R2c and shows
what the confounders add. Confounder features: Age, BMI, SQ_all (= ln LF/HF),
PA (ordinal 1-3); second run without PA. They are never passed through the
taxa feature selection and are standardised inside the model pipeline
(training rows only). Everything else (corrected diet file, nutrient density,
Spearman screening, Zung_S >= 50, splits, feature selection, models) is the
code of Revision-R2b, unchanged.

Manuscript: "Dietary and Gut Microbiome Features Associated with Depressive
Status: An Exploratory Machine-Learning Study" (IJMS, ijms-4632259).

All scientific logic lives here. The run scripts (run_0_prepare_inputs, run_A_main,
run_B_outputs, run_C_permutation) only set paths/parameters, call these
functions and print the results. Nothing here writes outside Revision-R2c-conf.

Validation scheme (unchanged from the submitted analysis): five repeated
stratified splits (seeds 42-46); each split = 40% feature-selection set /
40% training set / 20% test set. These are repeated holdout splits, NOT
5-fold cross-validation, and the pooled 90 test predictions are NOT 90
independent individuals.

Sections
    1. Constants and parameters
    2. Paths and environment
    3. Data loading and cohort checks
    4. Splits
    5. Taxa feature selection
    6. Dietary variables: energy adjustment and correlation filter
    7. Models, metrics, calibration
    8. One split / one variant
    9. Baseline reproduction (Step 0)
   10. Leakage invariance checks
   11. Step 2 tables (summary, composition, bootstrap, coefficients,
       calibration, participant characteristics)
   12. Figures
   13. Permutation test (Step 3)
   14. Results manifest
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier


# =============================================================================
# 1. Constants and parameters
# =============================================================================

# --- validation scheme (do not change) ---------------------------------------
SEEDS = (42, 43, 44, 45, 46)
CLASSIFIER_RANDOM_STATE = 42
FS_FRACTION = 0.40                 # first split: 40% feature-selection set
TEST_FRACTION_OF_REST = 1.0 / 3.0  # second split: 1/3 of the rest = 20% test

# --- cohort facts (asserted on load) -----------------------------------------
N_PARTICIPANTS = 88
N_DEPRESSIVE = 15
N_CONTROL = 73
ZUNG_CUTOFF = 50.0                 # Zung INDEX score (raw x 1.25); outcome = >= 50
LABEL_RULE = "Zung_S >= 50 (index score); identical to > 50 in this cohort"
N_TAXA_SUBMITTED = 249             # leaf taxa of the submitted analysis
N_TEST = 18
N_TEST_DEPRESSIVE = 3
N_TEST_CONTROL = 15

SAMPLE_PREFIX = "c16S"
PARTICIPANT_PREFIX = "sub-"

# --- taxa feature selection ---------------------------------------------------
ANOVA_P_THRESHOLD = 0.05
MI_PERCENTILE = 95
RF_TREES = 500
CONSENSUS_MIN_METHODS = 2

# --- dietary variables --------------------------------------------------------
DIET_RAW = ("Kcal", "Prot", "TF", "SatFat", "MUF", "Poly_InsFat",
            "Chol", "Carb", "Fiber")
NUTRIENTS = ("Prot", "TF", "SatFat", "MUF", "Poly_InsFat",
             "Chol", "Carb", "Fiber")               # the 8 non-energy variables
PUBLISHED_FILTERED_DIET = ("Prot", "SatFat", "Poly_InsFat",
                           "Chol", "Carb", "Fiber")  # submitted manuscript
CORR_THRESHOLD = 0.90
KCAL_PER_GRAM = {"Prot": 4.0, "Carb": 4.0, "TF": 9.0, "SatFat": 9.0,
                 "MUF": 9.0, "Poly_InsFat": 9.0}
ENERGY_DENOMS = ("reported_Kcal", "computed_4P9F4C")
PRIMARY_ENERGY_DENOM = "reported_Kcal"
DENSITY_NAMES = {
    "Prot": "Prot_pctE", "TF": "TF_pctE", "SatFat": "SatFat_pctE",
    "MUF": "MUF_pctE", "Poly_InsFat": "Poly_InsFat_pctE",
    "Carb": "Carb_pctE", "Chol": "Chol_mg_per_1000kcal",
    "Fiber": "Fiber_g_per_1000kcal",
}
RESIDUAL_SUFFIX = "_resid"

# --- models and configurations ------------------------------------------------
MODELS = ("LR", "SVM", "BRF")
BASE_FEATURE_SETS = ("Diet", "Taxa", "Diet + taxa")     # the three sets of Revision-R2b / R2c
CONFOUNDER_ONLY_SET = "Confounders"
CONFOUNDER_FEATURE_SET = "Confounders + diet + taxa"
FEATURE_SETS = (CONFOUNDER_ONLY_SET,) + BASE_FEATURE_SETS + (CONFOUNDER_FEATURE_SET,)   # 15 configurations
ZERRIN_FEATURE_SETS = (CONFOUNDER_ONLY_SET, "Diet", "Taxa", CONFOUNDER_FEATURE_SET)     # 12 configurations
REFERENCE_CONFIG = (CONFOUNDER_FEATURE_SET, "SVM")      # full model: confounders + diet + taxa, SVM
DIET_TAXA_CONFIG = ("Diet + taxa", "SVM")               # reference configuration of Revision-R2b / R2c
CONFOUNDER_SETS = {"CONF4": ("Age", "BMI", "SQ_all", "PA"),     # main run
                   "CONF3": ("Age", "BMI", "SQ_all")}           # PA is optional: run without PA
METRICS = ("ROC-AUC", "ACC", "BAL-ACC", "SEN", "SPE", "PRE", "F1")
TABLE_METRICS = ("ROC-AUC", "BAL-ACC", "SEN", "SPE", "PRE", "F1")
SUBMITTED_TABLE_METRICS = ("ROC-AUC", "BAL-ACC", "SEN", "PRE", "F1")   # columns of the submitted Table 1
SD_DDOF = 1                                          # sample SD over the 5 splits

# --- calibration (R2.8) -------------------------------------------------------
CALIB_CV_FOLDS = 3        # inner CV inside the TRAINING set for Platt scaling
CALIB_CV_SEED = 42
CALIB_N_BINS = 4
CALIB_BIN_STRATEGY = "quantile"

# --- paired bootstrap (R2.3) --------------------------------------------------
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 2026

# --- coefficient rule (R2.5) --------------------------------------------------
TOP_FRACTION = 0.25
MIN_SPLITS_RETAINED = 2
OLD_TOP_N = 20

# --- confounders (R2.8) -------------------------------------------------------
CONFOUNDER_CANDIDATES = ("Age", "BMI", "WHR", "Parasite", "LoE", "SES", "PA",
                         "SQ_all", "Bristol", "ALI_7")
# Pre-specified confounder sensitivity sets (R2.8). SQ_all = ln(LF/HF), not "SQ".
SENSITIVITY_SETS = {
    "SENS_A": ("Age", "BMI", "WHR"),
    "SENS_B": ("Age", "BMI", "WHR", "PA", "SQ_all"),
    "SENS_C": CONFOUNDER_CANDIDATES,          # exploratory only (overfitting-prone)
}
# Binary recoding: column -> code that becomes 1 (all other codes become 0).
# Parasite: code 1 has 42 participants, matching the 42 (47.7%) with parasites
# in Miranda-Angulo et al. 2024; meaning still to be confirmed by Ana.
BINARY_RECODE = {"Parasite": 1}
CATEGORICAL_CHARACTERISTICS = ("Parasite", "LoE", "SES", "PA", "Bristol", "ALI_7")

# --- permutation test (R2.4) --------------------------------------------------
PERM_B = 1000
PERM_BASE_SEED = 20260

# --- variants -----------------------------------------------------------------
# key -> folder name inside 01_main/ (and 02_outputs/). The names say which
# diet file and which taxa table were used, so runs can never be mixed up.
FOLDER = "Revision-R2c-conf"
MAIN_TAXA = "all"                 # taxa table of every run in this folder
N_TAXA_MAIN = 249
REFERENCE_FOLDER = "Revision-R2c"    # folder without confounders (only READ, for identity checks)
REFERENCE_RUN = "V1_MAIN_density_newDiet_allTaxa"
VARIANTS = {
    "CONF4": "MAIN_conf_Age_BMI_SQ_PA_249taxa",       # main run of this folder
    "CONF3": "noPA_conf_Age_BMI_SQ_249taxa",          # PA removed (PA was declared optional)
}
PRIMARY = "CONF4"
# variant key -> (diet values, taxa table)
VARIANT_DATA = {"CONF4": ("new", MAIN_TAXA), "CONF3": ("new", MAIN_TAXA)}
VARIANT_ROLES = {
    "CONF4": "MAIN RUN: confounders Age, BMI, SQ_all, PA; nutrient density, corrected diet file, 249 taxa",
    "CONF3": "secondary run without PA: confounders Age, BMI, SQ_all; otherwise identical",
}

# --- submitted Table 1 (manuscript template.tex, Table 1) ---------------------
# (Input features, Model) -> (ROC-AUC, BAL-ACC, SEN, PRE, F1)
SUBMITTED_TABLE1 = {
    ("Diet", "LR"): (0.471, 0.487, 0.333, 0.169, 0.220),
    ("Diet", "SVM"): (0.489, 0.520, 0.400, 0.188, 0.250),
    ("Diet", "BRF"): (0.518, 0.493, 0.400, 0.159, 0.226),
    ("Taxa", "LR"): (0.569, 0.453, 0.200, 0.123, 0.152),
    ("Taxa", "SVM"): (0.516, 0.547, 0.400, 0.229, 0.283),
    ("Taxa", "BRF"): (0.491, 0.527, 0.400, 0.186, 0.253),
    ("Diet + taxa", "LR"): (0.555, 0.526, 0.333, 0.187, 0.239),
    ("Diet + taxa", "SVM"): (0.640, 0.560, 0.467, 0.209, 0.287),
    ("Diet + taxa", "BRF"): (0.467, 0.487, 0.333, 0.187, 0.230),
}
SUBMITTED_CONFUSION = {"TN": 49, "FP": 26, "FN": 8, "TP": 7}   # Figure 2b
OLD_MODEL_NAMES = {"LinearSVM": "SVM", "BalancedRF": "BRF", "LR": "LR", "RF": "RF"}
OLD_RUN_KEY = "filtered_diet"

# --- figure colours (validated colour-blind-safe categorical slots) -----------
COLOR_CONTROL = "#2a78d6"
COLOR_DEPRESSIVE = "#eb6834"
COLOR_THIRD = "#1baf7a"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#d9d8d2"
COLOR_NEUTRAL = "#9a9992"

WORDING_SPLITS = "five repeated stratified splits"


# =============================================================================
# 2. Paths and environment
# =============================================================================

@dataclass(frozen=True)
class Paths:
    """All input and output locations, derived from the Clean-Codes root."""

    root: Path

    @property
    def part(self) -> Path:                      # labels and OLD diet values
        return (self.root / "Preprocessed Files"
                / "participants_v3_corrected_depflag.tsv")

    @property
    def taxa(self) -> Path:                      # submitted CLR (249 leaves)
        return self.root / "Preprocessed Files" / "taxonomy_leaf_CLR_v2.xlsx"

    @property
    def old_results(self) -> Path:
        return self.root / "Razan-5Fold-Results"

    @property
    def r2(self) -> Path:
        return self.root / FOLDER

    @property
    def inputs(self) -> Path:
        return self.r2 / "inputs"

    @property
    def walter(self) -> Path:                    # participants.tsv (metadata)
        return self.inputs / "source_files" / "participants.tsv"

    @property
    def taxa_new(self) -> Path:                  # CLR without uncultured taxa
        return self.inputs / "taxonomy_leaf_CLR_v2_noUncultured.xlsx"

    @property
    def diet_new(self) -> Path:                  # corrected diet values
        return self.inputs / "diet_new_88.csv"

    @property
    def confounders(self) -> Path:
        return self.inputs / "confounders_88.csv"

    @property
    def crosschecks(self) -> Path:
        return self.inputs / "crosschecks"

    @property
    def baseline(self) -> Path:
        return self.r2 / "00_checks"

    @property
    def main(self) -> Path:
        return self.r2 / "01_main"

    @property
    def outputs(self) -> Path:
        return self.r2 / "02_outputs"

    @property
    def permutation(self) -> Path:
        return self.r2 / "03_permutation"

    @property
    def figures(self) -> Path:
        return self.r2 / "figures"

    def variant_dir(self, key: str) -> Path:
        """Folder of a variant run inside 01_main/."""
        return self.main / VARIANTS[key]

    def output_dir(self, key: str) -> Path:
        """Folder of the Step 2 outputs of a variant inside 02_outputs/."""
        return self.outputs / VARIANTS[key]

    def permutation_dir(self, key: str = None) -> Path:
        """Folder of the permutation test of a variant (default: the main run)."""
        return self.permutation / VARIANTS[key or PRIMARY]


def make_paths(root: str | os.PathLike) -> Paths:
    """Build the Paths object and create the Revision-R2c-conf output folders."""
    paths = Paths(Path(root))
    for folder in (paths.r2, paths.inputs, paths.baseline, paths.main,
                   paths.outputs, paths.permutation, paths.figures):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_of(path: str | os.PathLike) -> str:
    """Return the SHA-256 of a file (used to record which inputs were used)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_info() -> dict:
    """Package versions and platform, stored next to every result."""
    import scipy
    import joblib
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "joblib": joblib.__version__,
        "cpu_count": os.cpu_count(),
        "in_colab": "google.colab" in sys.modules,
    }


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload: dict) -> None:
    """Save a small dictionary as JSON (used by the notebooks)."""
    _write_json(Path(path), payload)


def read_json(path) -> dict:
    """Load a JSON file written by write_json."""
    return _read_json(Path(path))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


# =============================================================================
# 3. Data loading and cohort checks
# =============================================================================

@dataclass(frozen=True)
class Cohort:
    """Aligned analysis data for the 88 participants (row order = PART file)."""

    ids: np.ndarray                 # participant_id, e.g. "sub-MIC002"
    y: np.ndarray                   # 1 = depressive (Zung index >= 50), 0 = control
    zung: np.ndarray                # Zung index score
    diet_raw: pd.DataFrame          # 9 raw dietary variables (old or new file)
    x_taxa: np.ndarray              # 88 x n_taxa CLR matrix
    taxa_ids: tuple                 # rankID of the leaf taxa
    taxon_names: dict               # rankID -> taxon name
    walter: Optional[pd.DataFrame]  # participants.tsv metadata (or None)
    confounders_raw: Optional[pd.DataFrame]   # 10 candidate confounders (raw codes)
    diet_label: str                 # "old" or "new"
    taxa_label: str                 # "all" or "noUncultured"
    sources: dict                   # file paths and SHA-256 of the inputs

    @property
    def n_taxa(self) -> int:
        return self.x_taxa.shape[1]


def load_cohort(part_path, taxa_path, walter_path=None, diet_path=None,
                confounders_path=None, taxa_label="all") -> Cohort:
    """Load labels, diet, CLR taxa table and metadata.

    Outcome: Zung index score >= 50 (LABEL_RULE). Asserts: 88 participants,
    15 depressive / 73 controls, labels identical to dep_flag_corrected and to
    the previous "> 50" rule, and the same participant IDs in the same order in
    every input table.

    diet_path = None  -> OLD dietary values (columns of PART)
    diet_path = CSV   -> corrected values extracted by run_0_prepare_inputs.py
    """
    part_path, taxa_path = Path(part_path), Path(taxa_path)
    part = pd.read_csv(part_path, sep="\t")
    clr = pd.read_excel(taxa_path, dtype={"rankID": str})

    # --- participants -----------------------------------------------------------
    ids = part["participant_id"].astype(str).to_numpy()
    assert len(ids) == N_PARTICIPANTS, f"expected 88 participants, got {len(ids)}"
    assert len(set(ids)) == N_PARTICIPANTS, "duplicated participant IDs in PART"

    # --- outcome ------------------------------------------------------------------
    zung = part["Zung_S"].to_numpy(dtype=float)
    assert not np.isnan(zung).any(), "missing Zung_S values"
    y = (zung >= ZUNG_CUTOFF).astype(int)
    assert np.array_equal(y, (zung > ZUNG_CUTOFF).astype(int)), \
        "a participant scores exactly 50: >= 50 and > 50 differ"
    assert int(y.sum()) == N_DEPRESSIVE and int((y == 0).sum()) == N_CONTROL, \
        f"expected 15 depressive / 73 controls, got {y.sum()} / {(y == 0).sum()}"
    assert np.array_equal(y, part["dep_flag_corrected"].astype(int).to_numpy()), \
        "Zung_S >= 50 does not reproduce dep_flag_corrected"

    sources = {
        "PART": {"path": str(part_path), "sha256": sha256_of(part_path)},
        "TAXA": {"path": str(taxa_path), "sha256": sha256_of(taxa_path)},
    }

    # --- diet ---------------------------------------------------------------------
    diet_old = part.loc[:, list(DIET_RAW)].astype(float).copy()
    diet_old.index = ids
    if diet_path is None:
        diet_raw, diet_label = diet_old, "old"
        sources["DIET"] = {"path": str(part_path) + " (dietary columns)",
                           "sha256": sources["PART"]["sha256"]}
    else:
        diet_new = pd.read_csv(diet_path)
        assert diet_new["participant_id"].astype(str).tolist() == list(ids), \
            "participant IDs / order differ between PART and the new diet table"
        diet_raw = diet_new.loc[:, list(DIET_RAW)].astype(float).copy()
        diet_raw.index = ids
        diet_label = "new"
        sources["DIET"] = {"path": str(diet_path), "sha256": sha256_of(diet_path)}
    assert not diet_raw.isna().any().any(), "missing dietary values"
    assert (diet_raw > 0).all().all(), "non-positive dietary values"

    # --- taxa ---------------------------------------------------------------------
    sample_cols = [c for c in clr.columns if str(c).startswith(SAMPLE_PREFIX)]
    taxa_sample_ids = [str(c).replace(SAMPLE_PREFIX, PARTICIPANT_PREFIX, 1)
                       for c in sample_cols]
    assert taxa_sample_ids == list(ids), \
        "participant IDs / order differ between PART and the taxa table"
    assert clr["rankID"].is_unique, "duplicated rankID in the taxa table"
    x_taxa = clr.loc[:, sample_cols].to_numpy(dtype=float).T
    assert x_taxa.shape == (N_PARTICIPANTS, len(clr)) and len(clr) > 0
    assert np.isfinite(x_taxa).all(), "non-finite values in the taxa matrix"
    taxa_ids = tuple(clr["rankID"].astype(str))
    taxon_names = dict(zip(taxa_ids, clr["taxon"].astype(str)))
    has_uncultured = any("uncultured" in name.lower() for name in taxon_names.values())
    if taxa_label == "all":
        assert len(taxa_ids) == N_TAXA_SUBMITTED, "expected the 249 submitted leaf taxa"
    else:
        assert taxa_label == "noUncultured" and not has_uncultured, \
            "the taxa table still contains uncultured taxa"

    # --- metadata table (participants.tsv) ------------------------------------------
    walter = None
    if walter_path is not None and Path(walter_path).exists():
        walter = pd.read_csv(walter_path, sep="\t")
        w_ids = walter["participant_id"].astype(str).to_numpy()
        assert np.array_equal(w_ids, ids), \
            "participant IDs / order differ between PART and participants.tsv"
        assert np.allclose(walter["Zung_S"].to_numpy(dtype=float), zung), \
            "Zung_S differs between PART and participants.tsv"
        walter = walter.copy()
        walter.index = ids
        sources["METADATA"] = {"path": str(walter_path),
                               "sha256": sha256_of(walter_path)}

    # --- candidate confounders -------------------------------------------------------
    confounders_raw = None
    if confounders_path is not None and Path(confounders_path).exists():
        table = pd.read_csv(confounders_path)
        assert table["participant_id"].astype(str).tolist() == list(ids), \
            "participant IDs / order differ between PART and the confounder table"
        confounders_raw = table.loc[:, list(CONFOUNDER_CANDIDATES)].astype(float)
        confounders_raw.index = ids
        assert not confounders_raw.isna().any().any(), "missing confounder values"
        sources["CONFOUNDERS"] = {"path": str(confounders_path),
                                  "sha256": sha256_of(confounders_path)}

    return Cohort(ids=ids, y=y, zung=zung, diet_raw=diet_raw, x_taxa=x_taxa,
                  taxa_ids=taxa_ids, taxon_names=taxon_names, walter=walter,
                  confounders_raw=confounders_raw, diet_label=diet_label,
                  taxa_label=taxa_label, sources=sources)


def load_all_cohorts(paths: "Paths") -> dict:
    """The four combinations of diet file (old / new) and taxa table (all 249
    leaves / uncultured removed), keyed by (diet, taxa)."""
    cohorts = {}
    for diet in ("old", "new"):
        for taxa in ("all", "noUncultured"):
            cohorts[(diet, taxa)] = load_cohort(
                paths.part,
                paths.taxa if taxa == "all" else paths.taxa_new,
                paths.walter,
                diet_path=None if diet == "old" else paths.diet_new,
                confounders_path=paths.confounders, taxa_label=taxa)
    reference = cohorts[("new", "noUncultured")]
    for cohort in cohorts.values():
        assert np.array_equal(cohort.y, reference.y)
        assert np.array_equal(cohort.ids, reference.ids)
    return cohorts


def cohort_for(cohorts: dict, variant_key: str) -> Cohort:
    """The cohort (diet file, taxa table) that a variant uses."""
    return cohorts[VARIANT_DATA[variant_key]]


def diet_sanity_report(cohort: Cohort) -> dict:
    """Unit check (4*Prot + 9*TF + 4*Carb) / Kcal and FFQ plausibility summary
    (no participant is excluded)."""
    d = cohort.diet_raw
    ratio = (4 * d["Prot"] + 9 * d["TF"] + 4 * d["Carb"]) / d["Kcal"]
    kcal = d["Kcal"]
    high = d.index[kcal > 4200]
    report = {
        "diet_file": cohort.diet_label,
        "ratio_median": float(ratio.median()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "kcal_min": float(kcal.min()), "kcal_median": float(kcal.median()),
        "kcal_max": float(kcal.max()),
        "n_kcal_gt_4200": int(len(high)),
        "ids_kcal_gt_4200": list(high),
        "kcal_gt_4200": [float(v) for v in kcal[high]],
        "group_kcal_gt_4200": [int(v) for v in cohort.y[np.isin(cohort.ids, high)]],
        "n_kcal_lt_800": int((kcal < 800).sum()),
        "n_excluded": 0,
    }
    assert 0.95 < report["ratio_median"] < 1.05, "macronutrient energy / Kcal off"
    assert report["ratio_min"] > 0.75 and report["ratio_max"] < 1.10
    return report


def confounder_frame(cohort: Cohort, names: Sequence[str]) -> pd.DataFrame:
    """Numeric confounder features for the sensitivity models.

    LoE, SES, PA (1-3), Bristol and ALI-7 are used as ordinal numeric values;
    Parasite becomes 0/1 (code 1 = present). SQ_all is ln(LF/HF). They are
    standardised inside the model pipeline, i.e. on the training set only.
    """
    assert cohort.confounders_raw is not None, "confounder table not loaded"
    missing = [n for n in names if n not in cohort.confounders_raw.columns]
    assert not missing, f"unknown confounder columns: {missing}"
    frame = cohort.confounders_raw.loc[:, list(names)].copy()
    for column, positive_code in BINARY_RECODE.items():
        if column in frame.columns:
            frame[column] = (frame[column] == positive_code).astype(float)
    return frame


# =============================================================================
# 4. Splits
# =============================================================================

def make_split(y: np.ndarray, seed: int):
    """Return (fs_idx, train_idx, test_idx) exactly as in the submitted code.

    Step 1: StratifiedShuffleSplit(test_size=0.40) -> 40% feature-selection set.
    Step 2: StratifiedShuffleSplit(test_size=1/3) on the remaining 60%
            -> 40% training set and 20% test set. Both use the same seed.
    """
    y = np.asarray(y)
    dummy = np.zeros((len(y), 1))
    first = StratifiedShuffleSplit(n_splits=1, test_size=FS_FRACTION,
                                   random_state=seed)
    rest_idx, fs_idx = next(first.split(dummy, y))
    second = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION_OF_REST,
                                    random_state=seed)
    train_rel, test_rel = next(second.split(dummy[rest_idx], y[rest_idx]))
    return fs_idx, rest_idx[train_rel], rest_idx[test_rel]


def check_split(y, fs_idx, train_idx, test_idx) -> None:
    """Assert disjoint FS/train/test sets and 3 depressive + 15 controls in test."""
    fs, tr, te = set(fs_idx.tolist()), set(train_idx.tolist()), set(test_idx.tolist())
    assert not (fs & tr) and not (fs & te) and not (tr & te), \
        "FS / training / test sets overlap"
    assert len(fs) + len(tr) + len(te) == len(y) == N_PARTICIPANTS
    assert len(fs | tr | te) == N_PARTICIPANTS
    assert len(te) == N_TEST, f"test set has {len(te)} participants, expected 18"
    n_dep = int(np.asarray(y)[test_idx].sum())
    assert n_dep == N_TEST_DEPRESSIVE and len(te) - n_dep == N_TEST_CONTROL, \
        f"test set has {n_dep} depressive / {len(te) - n_dep} controls"
    # Both classes must be present wherever something is fitted.
    assert 0 < np.asarray(y)[fs_idx].sum() < len(fs_idx)
    assert 0 < np.asarray(y)[train_idx].sum() < len(train_idx)


# =============================================================================
# 5. Taxa feature selection (on the FS set only)
# =============================================================================

def select_taxa(x_fs, y_fs, seed, n_jobs=-1, allow_empty=False):
    """ANOVA (p < 0.05), mutual information (>= 95th percentile) and Random
    Forest importance (> 1/n_taxa); consensus = selected by >= 2 of 3 methods.

    The caller must pass ONLY the rows of the feature-selection set.
    Returns (consensus_indices, method_sets, details).
    """
    n_taxa = x_fs.shape[1]
    assert n_taxa > 0 and x_fs.shape[0] == len(y_fs)

    _, p_values = f_classif(x_fs, y_fs)
    p_values = np.nan_to_num(p_values, nan=1.0)
    anova = set(np.flatnonzero(p_values < ANOVA_P_THRESHOLD).tolist())

    mi_values = mutual_info_classif(x_fs, y_fs, random_state=seed)
    mi_threshold = float(np.percentile(mi_values, MI_PERCENTILE))
    mi = set(np.flatnonzero(mi_values >= mi_threshold).tolist())

    forest = RandomForestClassifier(n_estimators=RF_TREES,
                                    class_weight="balanced",
                                    random_state=seed, n_jobs=n_jobs)
    forest.fit(x_fs, y_fs)
    rf_threshold = 1.0 / n_taxa
    rf = set(np.flatnonzero(forest.feature_importances_ > rf_threshold).tolist())

    method_sets = {"ANOVA": anova, "MI": mi, "RF": rf}
    votes = np.zeros(n_taxa, dtype=int)
    for selected in method_sets.values():
        votes[list(selected)] += 1
    consensus = np.flatnonzero(votes >= CONSENSUS_MIN_METHODS).tolist()

    if not consensus and not allow_empty:
        raise RuntimeError(
            f"No taxa passed the two-of-three consensus rule (seed={seed}).")

    details = {"anova_p": p_values, "mi": mi_values,
               "rf_importance": forest.feature_importances_,
               "mi_threshold": mi_threshold, "rf_threshold": rf_threshold,
               "n_fs_rows": int(x_fs.shape[0])}
    return consensus, method_sets, details


# =============================================================================
# 6. Dietary variables: energy adjustment and correlation filter
# =============================================================================

def energy_denominator(diet_raw: pd.DataFrame, energy_denom: str) -> pd.Series:
    """Energy (kcal/day) used as the denominator of the nutrient densities."""
    assert energy_denom in ENERGY_DENOMS, f"unknown ENERGY_DENOM: {energy_denom}"
    if energy_denom == "reported_Kcal":
        return diet_raw["Kcal"].astype(float)
    return (4.0 * diet_raw["Prot"] + 9.0 * diet_raw["TF"]
            + 4.0 * diet_raw["Carb"]).astype(float)


def nutrient_density(diet_raw: pd.DataFrame, energy_denom: str) -> pd.DataFrame:
    """Nutrient-density variables (Willett et al. 1997).

    Macronutrients as % of energy (4 kcal/g protein and carbohydrate, 9 kcal/g
    fats); cholesterol (mg) and fibre (g) per 1000 kcal. Computed separately
    for each participant, so no information is shared between participants.
    """
    energy = energy_denominator(diet_raw, energy_denom)
    out = pd.DataFrame(index=diet_raw.index)
    for nutrient in NUTRIENTS:
        if nutrient in KCAL_PER_GRAM:
            out[DENSITY_NAMES[nutrient]] = (
                100.0 * KCAL_PER_GRAM[nutrient] * diet_raw[nutrient] / energy)
        else:
            out[DENSITY_NAMES[nutrient]] = 1000.0 * diet_raw[nutrient] / energy
    return out


def spearman_filter(table: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> dict:
    """Remove redundant variables until no pair has |Spearman rho| >= threshold.

    The submitted analysis computed the Spearman matrix once on all 88
    participants, before any split and without using the outcome, and removed
    variables after reviewing the flagged pairs by hand (Kcal, TF, MUF). The
    hand review is replaced here by a deterministic rule that reproduces that
    choice exactly on the raw data (asserted in build_diet_design):

        repeat: among the remaining variables, drop the one involved in the
        largest number of pairs with |rho| >= threshold; ties are broken by
        the larger mean |rho| with the other remaining variables (then by
        column order); stop when no pair is left.
    """
    corr = table.corr(method="spearman")
    remaining = list(table.columns)
    removed, steps = [], []
    while True:
        sub = corr.loc[remaining, remaining].abs()
        high = (sub.to_numpy() >= threshold) & ~np.eye(len(remaining), dtype=bool)
        counts = high.sum(axis=1)
        if counts.max() == 0:
            break
        mean_abs = (sub.to_numpy().sum(axis=1) - 1.0) / (len(remaining) - 1)
        candidates = np.flatnonzero(counts == counts.max())
        drop_pos = int(candidates[np.argmax(mean_abs[candidates])])
        pairs = [(remaining[i], remaining[j], float(corr.loc[remaining[i], remaining[j]]))
                 for i in range(len(remaining)) for j in range(i + 1, len(remaining))
                 if high[i, j]]
        steps.append({
            "step": len(steps) + 1,
            "pairs_at_or_above_threshold": pairs,
            "dropped": remaining[drop_pos],
            "n_pairs_of_dropped": int(counts[drop_pos]),
            "mean_abs_rho_of_dropped": float(mean_abs[drop_pos]),
            "tied_candidates": [remaining[i] for i in candidates],
        })
        removed.append(remaining.pop(drop_pos))
    return {"corr": corr, "retained": tuple(remaining), "removed": tuple(removed),
            "steps": steps, "threshold": threshold}


@dataclass(frozen=True)
class DietDesign:
    """Dietary model inputs of one analysis variant."""

    variant: str                    # "V0", "V1", "V2" or "V3"
    label: str
    energy_denom: Optional[str]
    candidates: pd.DataFrame        # table on which the Spearman filter was run
    table: pd.DataFrame             # model-input table (V3: raw nutrients)
    feature_names: tuple            # names of the dietary model inputs
    retained: tuple
    removed: tuple
    corr: pd.DataFrame
    filter_steps: tuple
    residualize: bool               # V3: regress on Kcal inside each split
    kcal: np.ndarray
    notes: dict

    def matrix_for_split(self, train_idx: np.ndarray):
        """Return (88 x k dietary matrix, fit info) for one split.

        V0/V1/V2: the table is used as is (per-participant quantities).
        V3: each nutrient is regressed on Kcal using the TRAINING rows only;
        the fitted line is then applied to all rows (training and test).
        """
        values = self.table.to_numpy(dtype=float)
        if not self.residualize:
            return values, None
        design_train = np.column_stack([np.ones(len(train_idx)),
                                        self.kcal[train_idx]])
        beta, *_ = np.linalg.lstsq(design_train, values[train_idx], rcond=None)
        design_all = np.column_stack([np.ones(len(self.kcal)), self.kcal])
        residuals = values - design_all @ beta
        info = {"intercept": beta[0].copy(), "slope": beta[1].copy(),
                "n_fit_rows": int(len(train_idx))}
        return residuals, info


def build_diet_design(cohort: Cohort, variant: str,
                      energy_denom: str = PRIMARY_ENERGY_DENOM,
                      threshold: float = CORR_THRESHOLD) -> DietDesign:
    """Build the dietary inputs for variant V0, V1, V2 or V3.

    The Spearman |rho| >= 0.90 screening keeps the logic of the submitted
    analysis: computed once on all 88 participants, before the splits, without
    the outcome. It is therefore identical in every split and under every
    label permutation.

    V1 / V2: the screening is run on 9 variables (8 nutrient densities + Kcal).
    Kcal is never a model input in V1 (primary); V2 adds it as a feature.
    """
    raw = cohort.diet_raw
    kcal = raw["Kcal"].to_numpy(dtype=float)
    base_notes = {"diet_file": cohort.diet_label}

    if variant == "V0":
        result = spearman_filter(raw.loc[:, list(DIET_RAW)], threshold)
        if cohort.diet_label == "old":
            assert result["retained"] == PUBLISHED_FILTERED_DIET, (
                "correlation filter does not reproduce the submitted 6 variables: "
                f"{result['retained']}")
            assert set(result["removed"]) == {"Kcal", "TF", "MUF"}
        return DietDesign(
            variant="V0", label="V0: original raw dietary variables",
            energy_denom=None, candidates=raw.loc[:, list(DIET_RAW)],
            table=raw.loc[:, list(result["retained"])],
            feature_names=result["retained"], retained=result["retained"],
            removed=result["removed"], corr=result["corr"],
            filter_steps=tuple(result["steps"]), residualize=False, kcal=kcal,
            notes=dict(base_notes))

    if variant in ("V1", "V2"):
        density = nutrient_density(raw, energy_denom)
        nine = density.copy()
        nine["Kcal"] = raw["Kcal"].to_numpy()
        result = spearman_filter(nine, threshold)
        corr = result["corr"]
        kcal_rho = {c: float(corr.loc["Kcal", c]) for c in density.columns}
        kcal_in_pair = any("Kcal" in (a, b) for step in result["steps"]
                           for a, b, _ in step["pairs_at_or_above_threshold"])
        retained_density = tuple(c for c in result["retained"] if c != "Kcal")
        removed_density = tuple(c for c in result["removed"] if c != "Kcal")
        notes = dict(base_notes)
        notes.update({
            "screened_variables": list(nine.columns),
            "spearman_rho_Kcal_vs_densities": kcal_rho,
            "max_abs_rho_Kcal_vs_densities": max(abs(v) for v in kcal_rho.values()),
            "Kcal_in_a_pair_at_or_above_threshold": bool(kcal_in_pair),
            "Kcal_removed_by_rule": bool("Kcal" in result["removed"]),
            "Kcal_is_model_input": variant == "V2",
        })
        table = density.loc[:, list(retained_density)].copy()
        label = (f"V1: nutrient density (energy denominator: {energy_denom}; "
                 f"{cohort.diet_label} diet file)")
        if variant == "V2":
            table["Kcal"] = raw["Kcal"].to_numpy()
            label = (f"V2: nutrient density ({energy_denom}; {cohort.diet_label} "
                     "diet file) + Kcal as an additional feature")
        return DietDesign(
            variant=variant, label=label, energy_denom=energy_denom,
            candidates=nine, table=table, feature_names=tuple(table.columns),
            retained=retained_density, removed=removed_density,
            corr=corr, filter_steps=tuple(result["steps"]),
            residualize=False, kcal=kcal, notes=notes)

    if variant == "V3":
        # Screening decision only: residuals from a regression on all 88
        # participants (no outcome involved, same logic as the original filter).
        # The residuals used as MODEL INPUTS are re-fitted on the training set
        # of each split (see DietDesign.matrix_for_split).
        design_all = np.column_stack([np.ones(len(kcal)), kcal])
        nutrients = raw.loc[:, list(NUTRIENTS)].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(design_all, nutrients, rcond=None)
        resid_all = pd.DataFrame(nutrients - design_all @ beta, index=raw.index,
                                 columns=[n + RESIDUAL_SUFFIX for n in NUTRIENTS])
        result = spearman_filter(resid_all, threshold)
        kept = [name[: -len(RESIDUAL_SUFFIX)] for name in result["retained"]]
        notes = dict(base_notes)
        notes["filter_basis"] = ("residuals from a regression on all 88 "
                                 "participants (retention decision only)")
        return DietDesign(
            variant="V3",
            label=f"V3: energy-adjusted residuals ({cohort.diet_label} diet file; "
                  "regression on Kcal fitted on the training set of each split)",
            energy_denom="reported_Kcal", candidates=resid_all,
            table=raw.loc[:, kept], feature_names=result["retained"],
            retained=result["retained"], removed=result["removed"],
            corr=result["corr"], filter_steps=tuple(result["steps"]),
            residualize=True, kcal=kcal, notes=notes)

    raise ValueError(f"unknown variant: {variant}")


def describe_diet_design(design: DietDesign) -> str:
    """Human-readable report of the correlation filter for one variant."""
    lines = [design.label,
             f"  candidates ({design.candidates.shape[1]}): "
             f"{list(design.candidates.columns)}"]
    if not design.filter_steps:
        lines.append(f"  no pair with |rho| >= {CORR_THRESHOLD:.2f}")
    for step in design.filter_steps:
        pairs = ", ".join(f"{a}~{b} ({rho:+.3f})"
                          for a, b, rho in step["pairs_at_or_above_threshold"])
        lines.append(f"  step {step['step']}: pairs >= {CORR_THRESHOLD:.2f}: {pairs}")
        lines.append(f"           dropped {step['dropped']} "
                     f"(in {step['n_pairs_of_dropped']} pair(s), mean |rho| "
                     f"{step['mean_abs_rho_of_dropped']:.3f}; tied candidates: "
                     f"{step['tied_candidates']})")
    lines.append(f"  removed:  {list(design.removed)}")
    lines.append(f"  retained: {list(design.retained)}")
    lines.append(f"  model inputs ({len(design.feature_names)}): "
                 f"{list(design.feature_names)}")
    for key, value in design.notes.items():
        lines.append(f"  note - {key}: {value}")
    return "\n".join(lines)


# =============================================================================
# 7. Models, metrics, calibration
# =============================================================================

def build_models(n_jobs=-1) -> dict:
    """Fresh classifiers with the settings of the submitted analysis."""
    return {
        "LR": LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced",
                                 random_state=CLASSIFIER_RANDOM_STATE),
        "SVM": LinearSVC(C=1.0, max_iter=5_000, class_weight="balanced",
                         random_state=CLASSIFIER_RANDOM_STATE),
        "BRF": BalancedRandomForestClassifier(
            n_estimators=RF_TREES, random_state=CLASSIFIER_RANDOM_STATE,
            n_jobs=n_jobs),
    }


def make_model_pipeline(estimator) -> Pipeline:
    """StandardScaler + classifier; the scaler is fitted on the rows passed to fit."""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def continuous_scores(model, x_test) -> np.ndarray:
    """Decision scores for ROC-AUC (decision_function, else P(class 1))."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x_test), dtype=float)
    return np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]


def calculate_metrics(y_true, y_pred, scores) -> dict:
    """Metrics of the submitted analysis (depressive = positive class)."""
    return {
        "ROC-AUC": roc_auc_score(y_true, scores),
        "ACC": accuracy_score(y_true, y_pred),
        "BAL-ACC": balanced_accuracy_score(y_true, y_pred),
        "SEN": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "SPE": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "PRE": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def platt_probabilities(model_name, x_train, y_train, x_test, main_pipeline,
                        main_scores):
    """Platt scaling (sigmoid) fitted on the TRAINING set only.

    Out-of-fold decision scores are obtained by stratified CALIB_CV_FOLDS-fold
    cross-validation INSIDE the training set (scaler and classifier refitted
    in every inner fold); one sigmoid is fitted to those scores and then
    applied to the decision scores of the model fitted on the whole training
    set (ensemble=False). The test set is used only for prediction.

    Returns (probabilities for the test rows, sigmoid parameters a and b),
    with P(depressive) = 1 / (1 + exp(a * score + b)).
    """
    inner_cv = StratifiedKFold(n_splits=CALIB_CV_FOLDS, shuffle=True,
                               random_state=CALIB_CV_SEED)
    base = make_model_pipeline(build_models(n_jobs=1)[model_name])
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid",
                                        cv=inner_cv, ensemble=False)
    calibrated.fit(x_train, y_train)
    prob = calibrated.predict_proba(x_test)[:, 1]

    fitted = calibrated.calibrated_classifiers_[0]
    # The classifier inside the calibrator is the same model as the main one.
    assert np.allclose(fitted.estimator.named_steps["clf"].coef_,
                       main_pipeline.named_steps["clf"].coef_, atol=1e-8), \
        "calibrated estimator differs from the main model"
    assert fitted.estimator.named_steps["scaler"].n_samples_seen_ == len(y_train)
    sigmoid = fitted.calibrators[0]
    a, b = float(sigmoid.a_), float(sigmoid.b_)
    # Probabilities are exactly the training-set sigmoid of the main scores.
    assert np.allclose(prob, expit(-(a * np.asarray(main_scores) + b)), atol=1e-8), \
        "calibrated probabilities are not sigmoid(main decision scores)"
    return prob, a, b


# =============================================================================
# 8. One split / one variant
# =============================================================================

def all_configurations(feature_sets: Iterable[str] = FEATURE_SETS):
    """The (Input features, Model) configurations in table order."""
    return [(fs, model) for fs in feature_sets for model in MODELS]


def run_split(cohort: Cohort, y: np.ndarray, seed: int, split_no: int,
              design: DietDesign, *, confounders: Optional[pd.DataFrame] = None,
              n_jobs=-1, detail: bool = True, calibrate: bool = True,
              allow_empty_consensus: bool = False) -> dict:
    """Run one repeated stratified split for all feature sets and classifiers.

    Leakage rules enforced here:
      * taxa feature selection sees only the feature-selection rows;
      * the scaler, the classifiers, the Platt calibrator and (V3) the
        residual regression are fitted on the training rows only;
      * the test rows are used only for prediction.
    """
    y = np.asarray(y)
    fs_idx, train_idx, test_idx = make_split(y, seed)
    check_split(y, fs_idx, train_idx, test_idx)
    y_train, y_test = y[train_idx], y[test_idx]

    # --- taxa feature selection on the FS set only -------------------------------
    x_fs = cohort.x_taxa[fs_idx]
    consensus, method_sets, fs_details = select_taxa(
        x_fs, y[fs_idx], seed, n_jobs=n_jobs, allow_empty=allow_empty_consensus)
    assert fs_details["n_fs_rows"] == len(fs_idx)
    empty_consensus = len(consensus) == 0
    taxa_ids = [cohort.taxa_ids[i] for i in consensus]
    taxa_names = [cohort.taxon_names[t] for t in taxa_ids]
    x_taxa_sel = cohort.x_taxa[:, consensus]

    # --- dietary inputs (V3: residual regression fitted on the training set) -----
    x_diet, diet_info = design.matrix_for_split(train_idx)
    if design.residualize:
        assert diet_info["n_fit_rows"] == len(train_idx)
    diet_names = list(design.feature_names)

    # --- confounder features: always present, never passed through the taxa
    #     feature selection, standardised inside the model pipeline -----------------
    assert confounders is not None, "this pipeline always needs the confounder table"
    assert len(confounders) == len(y)
    conf_names = list(confounders.columns)
    x_conf = confounders.to_numpy(dtype=float)
    assert np.isfinite(x_conf).all()

    feature_sets = {
        CONFOUNDER_ONLY_SET: (x_conf, conf_names, conf_names,
                              ["Confounder"] * len(conf_names)),
        "Diet": (x_diet, diet_names, diet_names, ["Diet"] * len(diet_names)),
        "Taxa": (x_taxa_sel, taxa_ids, taxa_names, ["Taxa"] * len(taxa_ids)),
        "Diet + taxa": (np.hstack([x_diet, x_taxa_sel]), diet_names + taxa_ids,
                        diet_names + taxa_names,
                        ["Diet"] * len(diet_names) + ["Taxa"] * len(taxa_ids)),
        CONFOUNDER_FEATURE_SET: (
            np.hstack([x_diet, x_taxa_sel, x_conf]),
            diet_names + taxa_ids + conf_names,
            diet_names + taxa_names + conf_names,
            ["Diet"] * len(diet_names) + ["Taxa"] * len(taxa_ids)
            + ["Confounder"] * len(conf_names)),
    }
    assert tuple(feature_sets) == FEATURE_SETS

    out = {"split": split_no, "seed": seed, "aucs": {}, "metrics": [],
           "predictions": [], "coefficients": [], "selection": [],
           "membership": [], "calibration": [], "empty_consensus": empty_consensus,
           "fit_info": {"consensus": tuple(consensus), "scaler": {}, "coef": {},
                        "platt": {}, "resid": diet_info}}

    for fs_name, (matrix, feat_ids, feat_names, feat_groups) in feature_sets.items():
        n_features = matrix.shape[1]
        x_train, x_test = matrix[train_idx], matrix[test_idx]

        for model_name, estimator in build_models(n_jobs).items():
            fit_warning = ""
            prob_native = np.full(len(test_idx), np.nan)
            prob_platt = np.full(len(test_idx), np.nan)

            if n_features == 0:
                # Only possible under label permutation (allow_empty_consensus):
                # no taxon passed the consensus, so the taxa-only model carries
                # no information -> constant score (ROC-AUC = 0.5).
                scores = np.zeros(len(test_idx))
                y_pred = np.zeros(len(test_idx), dtype=int)
                pipeline = None
            else:
                pipeline = make_model_pipeline(estimator)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipeline.fit(x_train, y_train)
                fit_warning = "; ".join(sorted({type(w.message).__name__
                                                for w in caught
                                                if issubclass(w.category, ConvergenceWarning)}))
                scaler = pipeline.named_steps["scaler"]
                # Scaler fitted on the training rows only.
                assert scaler.n_samples_seen_ == len(train_idx)
                assert np.allclose(scaler.mean_, x_train.mean(axis=0))
                y_pred = pipeline.predict(x_test)
                scores = continuous_scores(pipeline, x_test)

            metrics = calculate_metrics(y_test, y_pred, scores)
            out["aucs"][(fs_name, model_name)] = float(metrics["ROC-AUC"])
            out["metrics"].append({
                "Split": split_no, "random_state": seed,
                "Input features": fs_name, "Model": model_name,
                "n selected taxa": len(consensus),
                "n diet features": len(diet_names),
                "n total features": n_features,
                **{k: float(v) for k, v in metrics.items()},
                "fit_warning": fit_warning,
            })
            if not detail or pipeline is None:
                continue

            # --- probabilities ---------------------------------------------------
            clf = pipeline.named_steps["clf"]
            if hasattr(pipeline, "predict_proba"):
                prob_native = pipeline.predict_proba(x_test)[:, 1]
            if calibrate and model_name in ("SVM", "LR"):
                prob_platt, a, b = platt_probabilities(
                    model_name, x_train, y_train, x_test, pipeline, scores)
                out["fit_info"]["platt"][(fs_name, model_name)] = (a, b)
                out["calibration"].append({
                    "Split": split_no, "random_state": seed,
                    "Input features": fs_name, "Model": model_name,
                    "platt_a": a, "platt_b": b,
                    "slope_inverted": bool(a > 0),
                    "n_calibration_rows": len(train_idx),
                    "inner_cv_folds": CALIB_CV_FOLDS,
                })

            for pos, row in enumerate(test_idx):
                out["predictions"].append({
                    "Split": split_no, "random_state": seed,
                    "Input features": fs_name, "Model": model_name,
                    "test_position": pos, "row_index": int(row),
                    "participant_id": cohort.ids[row],
                    "y_true": int(y_test[pos]), "y_pred": int(y_pred[pos]),
                    "y_score": float(scores[pos]),
                    "prob_native": float(prob_native[pos]),
                    "prob_platt": float(prob_platt[pos]),
                })

            # --- coefficients (LR, SVM) ------------------------------------------
            out["fit_info"]["scaler"][(fs_name, model_name)] = (
                scaler.mean_.copy(), scaler.scale_.copy())
            if hasattr(clf, "coef_"):
                coef = np.asarray(clf.coef_[0], dtype=float)
                out["fit_info"]["coef"][(fs_name, model_name)] = coef.copy()
                order = np.argsort(-np.abs(coef), kind="stable")
                ranks = np.empty(len(coef), dtype=int)
                ranks[order] = np.arange(1, len(coef) + 1)
                for j in range(len(coef)):
                    out["coefficients"].append({
                        "Split": split_no, "random_state": seed,
                        "Input features": fs_name, "Model": model_name,
                        "Feature ID": feat_ids[j], "Feature name": feat_names[j],
                        "Feature group": feat_groups[j],
                        "Coefficient": float(coef[j]),
                        "Absolute coefficient": float(abs(coef[j])),
                        "Rank by absolute coefficient": int(ranks[j]),
                        "n features in model": len(coef),
                    })

    if detail:
        consensus_set = set(consensus)
        for j, taxon_id in enumerate(cohort.taxa_ids):
            out["selection"].append({
                "Split": split_no, "random_state": seed, "rankID": taxon_id,
                "Taxon": cohort.taxon_names[taxon_id],
                "ANOVA": int(j in method_sets["ANOVA"]),
                "MI": int(j in method_sets["MI"]),
                "RF": int(j in method_sets["RF"]),
                "Consensus": int(j in consensus_set),
                "anova_p": float(fs_details["anova_p"][j]),
                "mi": float(fs_details["mi"][j]),
                "rf_importance": float(fs_details["rf_importance"][j]),
                "MI threshold": fs_details["mi_threshold"],
                "RF threshold": fs_details["rf_threshold"],
            })
        for role, idx in (("FS", fs_idx), ("train", train_idx), ("test", test_idx)):
            for row in idx:
                out["membership"].append({
                    "Split": split_no, "random_state": seed, "role": role,
                    "row_index": int(row), "participant_id": cohort.ids[row],
                    "y": int(y[row]),
                })
    return out


RUN_FILES = {
    "metrics": "metrics_per_split.csv",
    "summary": "summary_mean_sd.csv",
    "predictions": "predictions.csv",
    "coefficients": "coefficients.csv",
    "selection": "taxa_selection.csv",
    "membership": "split_membership.csv",
    "calibration": "calibration_params.csv",
    "corr": "diet_spearman_matrix.csv",
}
_ID_DTYPES = {"rankID": str, "Feature ID": str, "participant_id": str}


def load_run(out_dir) -> dict:
    """Load the saved outputs of one variant run."""
    out_dir = Path(out_dir)
    run = {}
    for key, name in RUN_FILES.items():
        path = out_dir / name
        if path.exists():
            index_col = 0 if key == "corr" else None
            run[key] = pd.read_csv(path, dtype=_ID_DTYPES, index_col=index_col,
                                   keep_default_na=True)
            if "fit_warning" in run[key].columns:
                run[key]["fit_warning"] = run[key]["fit_warning"].fillna("")
    run["config"] = _read_json(out_dir / "run_config.json")
    run["dir"] = out_dir
    return run


def run_variant(cohort: Cohort, design: DietDesign, out_dir, *,
                confounders: Optional[pd.DataFrame] = None, n_jobs=-1,
                force: bool = False, verbose: bool = True) -> dict:
    """Run the five repeated stratified splits for one variant and save everything.

    If the outputs already exist they are loaded instead of recomputed
    (set force=True to recompute).
    """
    out_dir = Path(out_dir)
    if not force and (out_dir / "run_config.json").exists():
        saved = load_run(out_dir)
        # A saved run is reused only if it was produced with the same design.
        expected = {
            "variant": design.variant, "energy_denom": design.energy_denom,
            "diet_model_inputs": list(design.feature_names),
            "confounders": (list(confounders.columns)
                            if confounders is not None else []),
        }
        for key, value in expected.items():
            assert saved["config"][key] == value, (
                f"saved run in {out_dir} used {key} = {saved['config'][key]!r}, "
                f"but {value!r} is requested now: set FORCE_RERUN = True")
        for name in ("PART", "TAXA", "DIET"):
            same = (saved["config"]["inputs"][name]["sha256"]
                    == cohort.sources[name]["sha256"])
            assert same, (f"saved run in {out_dir} used a different {name} file: "
                          "set FORCE_RERUN = True")
        if verbose:
            print(f"[load] {out_dir.name}: saved results reused "
                  f"(generated {saved['config']['generated']}; force=False)")
        return saved

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    collected = {k: [] for k in ("metrics", "predictions", "coefficients",
                                 "selection", "membership", "calibration")}
    resid_rows = []
    for split_no, seed in enumerate(SEEDS, start=1):
        res = run_split(cohort, cohort.y, seed, split_no, design,
                        confounders=confounders, n_jobs=n_jobs,
                        detail=True, calibrate=True)
        for key in collected:
            collected[key].extend(res[key])
        if res["fit_info"]["resid"] is not None:
            info = res["fit_info"]["resid"]
            for j, name in enumerate(design.feature_names):
                resid_rows.append({"Split": split_no, "random_state": seed,
                                   "Feature": name,
                                   "intercept": float(info["intercept"][j]),
                                   "slope_per_kcal": float(info["slope"][j]),
                                   "n_fit_rows": info["n_fit_rows"]})
        if verbose:
            n_taxa = res["metrics"][0]["n selected taxa"]
            auc = res["aucs"][REFERENCE_CONFIG]
            print(f"  split {split_no} (seed {seed}): consensus taxa = {n_taxa}, "
                  f"confounders+diet+taxa SVM ROC-AUC = {auc:.3f}, "
                  f"diet+taxa SVM ROC-AUC = {res['aucs'][DIET_TAXA_CONFIG]:.3f}")

    frames = {k: pd.DataFrame(v) for k, v in collected.items()}
    feature_sets = list(FEATURE_SETS)
    frames["summary"] = summarize_metrics(frames["metrics"], feature_sets)

    for key, frame in frames.items():
        frame.to_csv(out_dir / RUN_FILES[key], index=False)
    design.corr.to_csv(out_dir / RUN_FILES["corr"])
    if resid_rows:
        pd.DataFrame(resid_rows).to_csv(out_dir / "residual_regressions.csv",
                                        index=False)

    config = {
        "generated": _now(),
        "seconds": round(time.time() - started, 1),
        "wording": WORDING_SPLITS,
        "variant": design.variant, "label": design.label,
        "energy_denom": design.energy_denom,
        "diet_candidates": list(design.candidates.columns),
        "diet_removed": list(design.removed),
        "diet_retained": list(design.retained),
        "diet_model_inputs": list(design.feature_names),
        "diet_filter_steps": list(design.filter_steps),
        "diet_notes": design.notes,
        "confounders": list(confounders.columns) if confounders is not None else [],
        "binary_recode": BINARY_RECODE,
        "seeds": list(SEEDS), "classifier_random_state": CLASSIFIER_RANDOM_STATE,
        "anova_p_threshold": ANOVA_P_THRESHOLD, "mi_percentile": MI_PERCENTILE,
        "rf_trees": RF_TREES, "corr_threshold": CORR_THRESHOLD,
        "calibration": {"method": "sigmoid (Platt)", "inner_cv_folds": CALIB_CV_FOLDS,
                        "inner_cv_seed": CALIB_CV_SEED, "ensemble": False},
        "sd_ddof": SD_DDOF,
        "label_rule": LABEL_RULE,
        "labels_identical_to_dep_flag_corrected": True,
        "n_depressive": int(cohort.y.sum()), "n_control": int((cohort.y == 0).sum()),
        "diet_file": cohort.diet_label, "taxa_table": cohort.taxa_label,
        "n_taxa": int(cohort.n_taxa),
        "inputs": cohort.sources,
        "environment": environment_info(),
    }
    _write_json(out_dir / "run_config.json", config)
    if verbose:
        print(f"[saved] {out_dir}  ({config['seconds']} s)")
    return load_run(out_dir)


def summarize_metrics(metrics: pd.DataFrame,
                      feature_sets: Sequence[str] = FEATURE_SETS) -> pd.DataFrame:
    """Mean, SD (sample SD, n = 5 splits), min and max of every metric per
    configuration."""
    rows = []
    for fs_name in feature_sets:
        for model in MODELS:
            sub = metrics[(metrics["Input features"] == fs_name)
                          & (metrics["Model"] == model)]
            if sub.empty:
                continue
            assert len(sub) == len(SEEDS), "expected one row per split"
            row = {"Input features": fs_name, "Model": model, "n_splits": len(sub)}
            for metric in METRICS:
                values = sub[metric].to_numpy(dtype=float)
                row[f"{metric} mean"] = values.mean()
                row[f"{metric} SD"] = values.std(ddof=SD_DDOF)
                row[f"{metric} min"] = values.min()
                row[f"{metric} max"] = values.max()
            rows.append(row)
    return pd.DataFrame(rows)


def format_mean_sd(summary: pd.DataFrame,
                   metrics: Sequence[str] = TABLE_METRICS,
                   with_range: Sequence[str] = ("ROC-AUC",)) -> pd.DataFrame:
    """Display table: 'mean ± SD' per metric (and 'min-max' where requested)."""
    table = summary.loc[:, ["Input features", "Model"]].copy()
    for metric in metrics:
        table[f"{metric} (mean ± SD)"] = [
            f"{m:.3f} ± {s:.3f}" for m, s in zip(summary[f"{metric} mean"],
                                                 summary[f"{metric} SD"])]
        if metric in with_range:
            table[f"{metric} (min–max)"] = [
                f"{lo:.3f}–{hi:.3f}" for lo, hi in zip(summary[f"{metric} min"],
                                                      summary[f"{metric} max"])]
    return table


# =============================================================================
# 9. Baseline reproduction (Step 0)
# =============================================================================

def compare_with_submitted(run: dict, old_dir) -> dict:
    """Compare a V0 run with the submitted results (Razan-5Fold-Results and
    Table 1 of the manuscript). Returns comparison tables and pass/fail flags.
    """
    old_dir = Path(old_dir)
    metrics = run["metrics"]
    report = {}

    # --- (a) per-split metrics vs clf_results_all_runs_all_folds.csv --------------
    old = pd.read_csv(old_dir / "clf_results_all_runs_all_folds.csv")
    old = old[old["Run key"] == OLD_RUN_KEY].copy()
    old["Model"] = old["Model"].map(OLD_MODEL_NAMES)
    old = old[old["Model"].isin(MODELS)]
    merged = metrics.merge(old, left_on=["Split", "Input features", "Model"],
                           right_on=["Fold", "Input features", "Model"],
                           suffixes=("_new", "_old"))
    assert len(merged) == len(SEEDS) * len(FEATURE_SETS) * len(MODELS) == 45
    rows = []
    for metric in METRICS:
        diff = (merged[f"{metric}_new"] - merged[f"{metric}_old"]).abs()
        rows.append({"metric": metric, "n_compared": len(diff),
                     "max_abs_diff_vs_3dp_file": float(diff.max()),
                     "n_outside_rounding": int((diff > 0.0005 + 1e-9).sum())})
    report["per_split"] = pd.DataFrame(rows)
    report["per_split_merged"] = merged
    report["taxa_counts_match"] = bool(
        (merged["n selected taxa_new"] == merged["n selected taxa_old"]).all())
    report["per_split_ok"] = bool(
        report["per_split"]["n_outside_rounding"].sum() == 0
        and report["taxa_counts_match"])

    # --- (b) Table 1 means vs the manuscript --------------------------------------
    rows = []
    for (fs_name, model), target in SUBMITTED_TABLE1.items():
        sub = metrics[(metrics["Input features"] == fs_name)
                      & (metrics["Model"] == model)]
        for metric, submitted in zip(SUBMITTED_TABLE_METRICS, target):
            values = sub[metric].to_numpy(dtype=float)
            mean_exact = float(values.mean())
            # The submitted table averaged per-split values already rounded
            # to 3 decimals; reproduce that as well.
            mean_of_rounded = float(np.round(values, 3).mean())
            rows.append({
                "Input features": fs_name, "Model": model, "metric": metric,
                "submitted": submitted, "reproduced_mean": mean_exact,
                "reproduced_mean_of_3dp_splits": mean_of_rounded,
                "diff_exact": mean_exact - submitted,
                "diff_mean_of_3dp_splits": mean_of_rounded - submitted,
            })
    table = pd.DataFrame(rows)
    table["match_as_submitted"] = (
        table["diff_mean_of_3dp_splits"].abs() <= 0.0005 + 1e-9)
    table["match_within_0.001"] = table["diff_exact"].abs() <= 0.001 + 1e-9
    report["table1"] = table
    report["table1_ok"] = bool(table["match_as_submitted"].all()
                               and table["match_within_0.001"].all())

    # --- (c) consensus taxa vs fs_all_folds_detail.csv -----------------------------
    old_fs = pd.read_csv(old_dir / "fs_all_folds_detail.csv", dtype={"rankID": str})
    old_fs = old_fs[old_fs["Run key"] == OLD_RUN_KEY]
    sel = run["selection"].merge(old_fs, left_on=["Split", "rankID"],
                                 right_on=["Fold", "rankID"],
                                 suffixes=("_new", "_old"))
    assert len(sel) == N_TAXA_SUBMITTED * len(SEEDS)
    report["selection_mismatches"] = {
        "ANOVA": int((sel["ANOVA_new"] != sel["ANOVA_old"]).sum()),
        "MI": int((sel["MI"] != sel["MI95"]).sum()),
        "RF": int((sel["RF_new"] != sel["RF_old"]).sum()),
        "Consensus": int((sel["Consensus_new"] != sel["Consensus_old"]).sum()),
    }
    report["selection_ok"] = all(v == 0 for v in
                                 report["selection_mismatches"].values())

    # --- (d) SVM diet+taxa predictions and confusion matrix ------------------------
    old_pred = pd.read_csv(old_dir / "SVM_full_and_filtered_diet_plus_taxa_"
                                     "predictions_with_folds.csv")
    old_pred = old_pred[old_pred["Run_key"] == OLD_RUN_KEY].reset_index(drop=True)
    new_pred = run["predictions"]
    new_pred = new_pred[(new_pred["Input features"] == REFERENCE_CONFIG[0])
                        & (new_pred["Model"] == REFERENCE_CONFIG[1])]
    new_pred = new_pred.sort_values(["Split", "test_position"]).reset_index(drop=True)
    assert len(old_pred) == len(new_pred) == 90
    report["prediction_max_abs_score_diff"] = float(
        (old_pred["score"] - new_pred["y_score"]).abs().max())
    report["predictions_ok"] = bool(
        (old_pred["Fold"].to_numpy() == new_pred["Split"].to_numpy()).all()
        and (old_pred["y_true"].to_numpy() == new_pred["y_true"].to_numpy()).all()
        and (old_pred["y_pred"].to_numpy() == new_pred["y_pred"].to_numpy()).all()
        and report["prediction_max_abs_score_diff"] < 1e-6)
    report["confusion"] = confusion_counts(new_pred["y_true"], new_pred["y_pred"])
    report["confusion_ok"] = report["confusion"] == SUBMITTED_CONFUSION

    # --- (e) SVM diet+taxa coefficients ---------------------------------------------
    old_coef = pd.read_csv(old_dir / "SVM_coefficients_all_folds_full_and_filtered_"
                                     "diet_plus_taxa.csv", dtype={"Feature ID": str})
    old_coef = old_coef[old_coef["Run key"] == OLD_RUN_KEY]
    new_coef = run["coefficients"]
    new_coef = new_coef[(new_coef["Input features"] == REFERENCE_CONFIG[0])
                        & (new_coef["Model"] == REFERENCE_CONFIG[1])]
    coef = new_coef.merge(old_coef, left_on=["Split", "Feature ID"],
                          right_on=["Fold", "Feature ID"], suffixes=("_new", "_old"))
    report["coef_rows_new_old_merged"] = (len(new_coef), len(old_coef), len(coef))
    report["coef_max_abs_diff"] = float(
        (coef["Coefficient_new"] - coef["Coefficient_old"]).abs().max())
    report["coefficients_ok"] = bool(
        len(new_coef) == len(old_coef) == len(coef)
        and report["coef_max_abs_diff"] < 1e-6)

    report["all_ok"] = all(report[k] for k in (
        "per_split_ok", "table1_ok", "selection_ok", "predictions_ok",
        "confusion_ok", "coefficients_ok"))
    return report


def assert_baseline_reproduced(report: dict) -> None:
    """Stop if the baseline does not reproduce the submitted results."""
    problems = [key for key in ("per_split_ok", "table1_ok", "selection_ok",
                                "predictions_ok", "confusion_ok",
                                "coefficients_ok") if not report[key]]
    assert not problems, (
        "BASELINE NOT REPRODUCED - stop and inspect before going on: "
        + ", ".join(problems))


def confusion_counts(y_true, y_pred) -> dict:
    """Pooled confusion-matrix counts (depressive = positive)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"TN": int(((y_true == 0) & (y_pred == 0)).sum()),
            "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
            "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
            "TP": int(((y_true == 1) & (y_pred == 1)).sum())}


# =============================================================================
# 10. Leakage invariance checks
# =============================================================================

def leakage_invariance_check(cohort: Cohort, design: DietDesign, seed: int, *,
                             confounders: Optional[pd.DataFrame] = None,
                             n_jobs=-1, scramble_seed: int = 0) -> dict:
    """Functional leakage test for one split.

    Test A (feature selection uses only the FS set): the taxa values of every
    participant OUTSIDE the feature-selection set are replaced by random
    numbers; the selected taxa must not change.

    Test B (nothing is fitted on the test set): the taxa, dietary and
    confounder values of the TEST participants are replaced by random numbers;
    the selected taxa, scaler means/scales, model coefficients, Platt
    parameters and (V3) residual-regression coefficients must be identical,
    while the test predictions must change (proof that the scramble was
    effective).

    The retained dietary variable list is held fixed: as in the submitted
    analysis, the Spearman filter is computed once on all 88 participants
    before the splits and does not use the outcome.
    """
    rng = np.random.default_rng(scramble_seed)
    y = cohort.y
    fs_idx, train_idx, test_idx = make_split(y, seed)
    base = run_split(cohort, y, seed, 0, design, confounders=confounders,
                     n_jobs=n_jobs)

    # --- Test A ------------------------------------------------------------------
    non_fs = np.setdiff1d(np.arange(len(y)), fs_idx)
    x_a = cohort.x_taxa.copy()
    x_a[non_fs] = rng.normal(size=(len(non_fs), x_a.shape[1]))
    consensus_a, _, _ = select_taxa(x_a[fs_idx], y[fs_idx], seed, n_jobs=n_jobs)
    assert tuple(consensus_a) == base["fit_info"]["consensus"], \
        "LEAKAGE: taxa selection changed when non-FS rows were scrambled"

    # --- Test B ------------------------------------------------------------------
    x_b = cohort.x_taxa.copy()
    x_b[test_idx] = rng.normal(size=(len(test_idx), x_b.shape[1]))
    table_b = design.table.copy()
    table_b.iloc[test_idx] = np.abs(rng.normal(
        loc=table_b.to_numpy().mean(), scale=table_b.to_numpy().std(),
        size=(len(test_idx), table_b.shape[1])))
    kcal_b = design.kcal.copy()
    kcal_b[test_idx] = rng.uniform(1000, 6000, size=len(test_idx))
    design_b = dataclasses.replace(design, table=table_b, kcal=kcal_b)
    cohort_b = dataclasses.replace(cohort, x_taxa=x_b)
    conf_b = None
    if confounders is not None:
        conf_b = confounders.copy()
        conf_b.iloc[test_idx] = rng.normal(size=(len(test_idx), conf_b.shape[1]))
    other = run_split(cohort_b, y, seed, 0, design_b, confounders=conf_b,
                      n_jobs=n_jobs)

    a, b = base["fit_info"], other["fit_info"]
    assert a["consensus"] == b["consensus"], "LEAKAGE: taxa selection uses test rows"
    for key in a["scaler"]:
        assert np.allclose(a["scaler"][key][0], b["scaler"][key][0]) and \
            np.allclose(a["scaler"][key][1], b["scaler"][key][1]), \
            f"LEAKAGE: scaler of {key} depends on test rows"
    for key in a["coef"]:
        assert np.allclose(a["coef"][key], b["coef"][key], atol=1e-10), \
            f"LEAKAGE: coefficients of {key} depend on test rows"
    for key in a["platt"]:
        assert np.allclose(a["platt"][key], b["platt"][key], atol=1e-10), \
            f"LEAKAGE: Platt calibrator of {key} depends on test rows"
    if a["resid"] is not None:
        assert np.allclose(a["resid"]["slope"], b["resid"]["slope"]) and \
            np.allclose(a["resid"]["intercept"], b["resid"]["intercept"]), \
            "LEAKAGE: residual regression depends on test rows"

    base_scores = np.array([r["y_score"] for r in base["predictions"]])
    other_scores = np.array([r["y_score"] for r in other["predictions"]])
    assert not np.allclose(base_scores, other_scores), \
        "scramble had no effect on the test predictions (check is not valid)"

    return {"seed": seed, "variant": design.variant,
            "n_scalers_checked": len(a["scaler"]),
            "n_coefficient_vectors_checked": len(a["coef"]),
            "n_platt_calibrators_checked": len(a["platt"]),
            "residual_regression_checked": a["resid"] is not None,
            "confounders_included": confounders is not None,
            "passed": True}


# =============================================================================
# 11. Step 2 tables
# =============================================================================

# --- 2a: all 45 AUCs -----------------------------------------------------------

def auc_per_split_table(metrics: pd.DataFrame, metric: str = "ROC-AUC",
                        feature_sets: Sequence[str] = FEATURE_SETS) -> pd.DataFrame:
    """Wide table: one row per configuration, one column per split, plus
    mean, SD, min and max (the 45 ROC-AUC values for 9 configurations)."""
    wide = metrics.pivot_table(index=["Input features", "Model"], columns="Split",
                               values=metric, aggfunc="first")
    wide = wide.reindex([(fs, m) for fs in feature_sets for m in MODELS])
    split_cols = [f"Split {s} (seed {seed})" for s, seed in enumerate(SEEDS, 1)]
    wide.columns = split_cols
    values = wide.to_numpy(dtype=float)
    wide["Mean"] = values.mean(axis=1)
    wide["SD"] = values.std(axis=1, ddof=SD_DDOF)
    wide["Min"] = values.min(axis=1)
    wide["Max"] = values.max(axis=1)
    return wide.reset_index()


# --- 2b: test-set composition ---------------------------------------------------

def test_composition(membership: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """Class composition of every test set and how ROC-AUC was handled."""
    rows = []
    for (split, seed), sub in membership.groupby(["Split", "random_state"]):
        row = {"Split": split, "random_state": seed}
        for role in ("FS", "train", "test"):
            part = sub[sub["role"] == role]
            row[f"{role} n"] = len(part)
            row[f"{role} depressive"] = int(part["y"].sum())
            row[f"{role} controls"] = int((part["y"] == 0).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    assert (table["test n"] == N_TEST).all()
    assert (table["test depressive"] == N_TEST_DEPRESSIVE).all(), \
        "a test set does not contain exactly 3 depressive participants"
    assert (table["test controls"] == N_TEST_CONTROL).all()

    n_auc = int(metrics["ROC-AUC"].notna().sum())
    assert n_auc == len(metrics), "undefined ROC-AUC values found"
    test = membership[membership["role"] == "test"]
    appearances = test.groupby("participant_id").size()
    return {
        "table": table,
        "n_auc_values": n_auc,
        "n_auc_undefined_or_dropped": int(len(metrics) - n_auc),
        "pooled_test_instances": int(len(test)),
        "unique_test_participants": int(appearances.size),
        "participants_in_more_than_one_test_set": int((appearances > 1).sum()),
        "max_test_appearances": int(appearances.max()),
        "never_in_a_test_set": int(N_PARTICIPANTS - appearances.size),
        "appearance_distribution": appearances.value_counts().sort_index().to_dict(),
    }


def composition_statement(comp: dict) -> str:
    """Plain-language statement for R2.2b (class composition, AUC handling)."""
    return (
        f"Every test set of the {WORDING_SPLITS} contains {N_TEST} participants: "
        f"{N_TEST_DEPRESSIVE} depressive and {N_TEST_CONTROL} controls "
        "(stratified sampling; asserted in the code). No test set contains one or "
        "zero depressive participants, so ROC-AUC was always defined: "
        f"{comp['n_auc_values']} of {comp['n_auc_values']} ROC-AUC values were "
        f"computed and {comp['n_auc_undefined_or_dropped']} were dropped, undefined "
        "or set to 0.5. "
        f"The pooled {comp['pooled_test_instances']} test instances are accumulated "
        f"across the five repeated splits and come from "
        f"{comp['unique_test_participants']} distinct participants "
        f"({comp['participants_in_more_than_one_test_set']} appear in more than one "
        f"test set, at most {comp['max_test_appearances']} times; "
        f"{comp['never_in_a_test_set']} are never in a test set). With only "
        f"{N_TEST_DEPRESSIVE} depressive participants per test set, each per-split "
        f"ROC-AUC is a multiple of 1/{N_TEST_DEPRESSIVE * N_TEST_CONTROL} and is "
        "therefore noisy.")


# --- 2c: paired bootstrap ---------------------------------------------------------

def paired_bootstrap_auc(predictions: pd.DataFrame, metrics: pd.DataFrame,
                         reference=REFERENCE_CONFIG, n_boot: int = BOOTSTRAP_B,
                         seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap of the difference in mean ROC-AUC (R2.3).

    In every bootstrap replicate the participants of each test set are
    resampled with replacement within class (3 depressive and 15 controls stay
    fixed); the SAME resampled participants are used for all configurations,
    and the mean ROC-AUC over the five splits is recomputed. No model is
    refitted, so the interval reflects test-sample variability only, not
    training (or feature-selection) variability.

    Difference = mean AUC(reference) - mean AUC(comparator): positive values
    favour the reference (combined diet+taxa SVM).
    """
    configs = all_configurations()
    assert reference in configs
    splits = sorted(predictions["Split"].unique())
    assert len(splits) == len(SEEDS)

    # Pairwise comparison kernels: K[c][s][i, j] = 1 if depressive i scores
    # above control j, 0.5 if tied, 0 otherwise -> AUC = mean of the kernel.
    kernels, ids_per_split = {}, {}
    for split in splits:
        block = predictions[predictions["Split"] == split]
        for config in configs:
            sub = block[(block["Input features"] == config[0])
                        & (block["Model"] == config[1])].sort_values("test_position")
            ids = sub["participant_id"].to_numpy()
            y_true = sub["y_true"].to_numpy()
            if split in ids_per_split:
                assert np.array_equal(ids, ids_per_split[split][0]), \
                    "configurations do not share the same test participants"
            else:
                ids_per_split[split] = (ids, y_true)
            pos = sub["y_score"].to_numpy()[y_true == 1]
            neg = sub["y_score"].to_numpy()[y_true == 0]
            assert len(pos) == N_TEST_DEPRESSIVE and len(neg) == N_TEST_CONTROL
            kernel = ((pos[:, None] > neg[None, :]).astype(float)
                      + 0.5 * (pos[:, None] == neg[None, :]))
            kernels[(config, split)] = kernel
            observed = metrics[(metrics["Split"] == split)
                               & (metrics["Input features"] == config[0])
                               & (metrics["Model"] == config[1])]["ROC-AUC"].iloc[0]
            assert abs(kernel.mean() - observed) < 1e-9, "AUC kernel mismatch"

    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(configs)))
    for b in range(n_boot):
        for split in splits:
            pos_idx = rng.integers(0, N_TEST_DEPRESSIVE, N_TEST_DEPRESSIVE)
            neg_idx = rng.integers(0, N_TEST_CONTROL, N_TEST_CONTROL)
            for c, config in enumerate(configs):
                boot[b, c] += kernels[(config, split)][np.ix_(pos_idx, neg_idx)].mean()
    boot /= len(splits)

    ref_col = configs.index(reference)
    split_auc = {config: np.array([kernels[(config, s)].mean() for s in splits])
                 for config in configs}
    rows = []
    for c, config in enumerate(configs):
        if config == reference:
            continue
        # Mean AUCs are multiples of 1/225; rounding removes floating-point
        # noise so that an interval ending exactly at zero is treated as
        # containing zero.
        diff = np.round(boot[:, ref_col] - boot[:, c], 10)
        per_split = split_auc[reference] - split_auc[config]
        low, median, high = np.round(np.percentile(diff, [2.5, 50, 97.5]), 10)
        row = {"Comparator input features": config[0], "Comparator model": config[1],
               "Mean AUC reference": split_auc[reference].mean(),
               "Mean AUC comparator": split_auc[config].mean(),
               "Observed difference (ref - comp)": per_split.mean()}
        for s, value in zip(splits, per_split):
            row[f"Split {s} difference"] = value
        row.update({"Bootstrap median difference": median,
                    "95% CI low": low, "95% CI high": high,
                    "Interval crosses zero": bool(low <= 0.0 <= high),
                    "Share of replicates with difference <= 0": float((diff <= 0).mean())})
        rows.append(row)
    table = pd.DataFrame(rows)
    boot_frame = pd.DataFrame(boot, columns=[f"{fs} | {m}" for fs, m in configs])
    return {"table": table, "bootstrap_mean_auc": boot_frame,
            "reference": reference, "n_boot": n_boot, "seed": seed}


def bootstrap_note(boot: dict, metrics: pd.DataFrame) -> str:
    """Plain-language note for R2.3 (which intervals cross zero)."""
    table = boot["table"]
    ref = boot["reference"]
    means = {c: metrics[(metrics["Input features"] == c[0])
                        & (metrics["Model"] == c[1])]["ROC-AUC"].mean()
             for c in all_configurations()}
    top = max(means.values())
    tied = [c for c, v in means.items() if v >= top - 1e-9]
    lines = [f"Reference: {ref[0]} / {ref[1]} (mean ROC-AUC {means[ref]:.3f}). "
             "Difference = reference minus comparator; positive favours the reference.",
             f"Bootstrap: B = {boot['n_boot']}, seed {boot['seed']}, participants "
             "resampled within each test set, stratified by class "
             f"({N_TEST_DEPRESSIVE} depressive, {N_TEST_CONTROL} controls)."]
    if tied != [ref]:
        names = ", ".join(f"{c[0]} / {c[1]}" for c in tied)
        lines.append(f"NOTE: the highest mean ROC-AUC in this run ({top:.3f}) is reached by "
                     f"{len(tied)} configuration(s): {names}."
                     + ("" if ref in tied else " The reference is not among them."))
    crossing = table[table["Interval crosses zero"]]
    lines.append(f"NOTE: the 95% interval crosses zero for {len(crossing)} of "
                 f"{len(table)} comparisons" + (":" if len(crossing) else "."))
    for _, row in crossing.iterrows():
        lines.append(f"   - vs {row['Comparator input features']} / "
                     f"{row['Comparator model']}: median "
                     f"{row['Bootstrap median difference']:+.3f} "
                     f"[{row['95% CI low']:+.3f}, {row['95% CI high']:+.3f}]")
    for _, row in table[~table["Interval crosses zero"]].iterrows():
        lines.append(f"   interval excludes zero vs {row['Comparator input features']} / "
                     f"{row['Comparator model']}: median "
                     f"{row['Bootstrap median difference']:+.3f} "
                     f"[{row['95% CI low']:+.3f}, {row['95% CI high']:+.3f}]")
    lines.append("The interval reflects test-sample variability only (no model is "
                 "refitted), not training or feature-selection variability.")
    return chr(10).join(lines)


# --- 2d: coefficient rule -----------------------------------------------------------

def base_nutrient(feature_id: str) -> str:
    """Map an adjusted dietary variable name back to its nutrient (e.g.
    'Chol_mg_per_1000kcal' -> 'Chol')."""
    for nutrient, name in DENSITY_NAMES.items():
        if feature_id == name:
            return nutrient
    if feature_id.endswith(RESIDUAL_SUFFIX):
        return feature_id[: -len(RESIDUAL_SUFFIX)]
    return feature_id


def coefficient_stability(coefficients: pd.DataFrame, selection: pd.DataFrame,
                          feature_set: str = REFERENCE_CONFIG[0],
                          model: str = REFERENCE_CONFIG[1],
                          top_fraction: float = TOP_FRACTION,
                          min_splits: int = MIN_SPLITS_RETAINED,
                          old_top_n: int = OLD_TOP_N) -> dict:
    """Pre-specified proportional rule replacing 'top-20 coefficients' (R2.5).

    In each split the features of the model are ranked by |coefficient| and
    the top ceil(top_fraction * n_features) are kept; a feature is prioritised
    if it is kept in at least `min_splits` of the five splits. The old top-20
    rule is computed on the same coefficients for comparison, and the
    consensus-selection count of every taxon is attached.
    """
    sub = coefficients[(coefficients["Input features"] == feature_set)
                       & (coefficients["Model"] == model)].copy()
    assert not sub.empty, f"no coefficients for {feature_set} / {model}"
    sub["k"] = np.ceil(top_fraction * sub["n features in model"]).astype(int)
    sub["in_top_fraction"] = sub["Rank by absolute coefficient"] <= sub["k"]
    sub["in_old_top_n"] = sub["Rank by absolute coefficient"] <= old_top_n

    per_split = (sub.groupby("Split")
                 .agg(n_features=("Feature ID", "size"), k_top_fraction=("k", "first"),
                      n_kept=("in_top_fraction", "sum"),
                      n_in_old_top_n=("in_old_top_n", "sum"))
                 .reset_index())
    assert (per_split["n_kept"] == per_split["k_top_fraction"]).all()

    consensus_count = selection.groupby("rankID")["Consensus"].sum()
    rows = []
    for (fid, name, group), block in sub.groupby(
            ["Feature ID", "Feature name", "Feature group"], sort=False):
        coef_by_split = dict(zip(block["Split"], block["Coefficient"]))
        kept = block[block["in_top_fraction"]]
        row = {
            "Feature ID": fid, "Feature name": name, "Feature group": group,
            "Splits present in model": int(block["Split"].nunique()),
            "Splits in top fraction": int(block["in_top_fraction"].sum()),
            f"Splits in old top {old_top_n}": int(block["in_old_top_n"].sum()),
            "Splits selected by FS consensus": (
                int(consensus_count.get(fid, 0)) if group == "Taxa" else np.nan),
            "Mean coefficient": block["Coefficient"].mean(),
            "Mean absolute coefficient": block["Absolute coefficient"].mean(),
            "Positive coefficients (splits)": int((block["Coefficient"] > 0).sum()),
            "Negative coefficients (splits)": int((block["Coefficient"] < 0).sum()),
            "Positive when in top fraction": int((kept["Coefficient"] > 0).sum()),
            "Negative when in top fraction": int((kept["Coefficient"] < 0).sum()),
        }
        for split in sorted(sub["Split"].unique()):
            row[f"Coefficient split {split}"] = coef_by_split.get(split, np.nan)
            in_top = block.loc[block["Split"] == split, "in_top_fraction"]
            row[f"In top fraction split {split}"] = (
                bool(in_top.iloc[0]) if len(in_top) else np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["Prioritised (new rule)"] = table["Splits in top fraction"] >= min_splits
    table[f"Prioritised (old top-{old_top_n} rule)"] = (
        table[f"Splits in old top {old_top_n}"] >= min_splits)
    table = table.sort_values(
        ["Prioritised (new rule)", "Splits in top fraction",
         "Mean absolute coefficient"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return {"table": table, "per_split": per_split,
            "prioritised": table[table["Prioritised (new rule)"]].reset_index(drop=True),
            "feature_set": feature_set, "model": model,
            "top_fraction": top_fraction, "min_splits": min_splits}


def consensus_frequency(selection: pd.DataFrame) -> pd.DataFrame:
    """Number of splits in which each taxon passes the 2-of-3 consensus rule
    (and each single method)."""
    table = (selection.groupby(["rankID", "Taxon"], as_index=False)
             .agg(Splits_selected_by_consensus=("Consensus", "sum"),
                  Splits_ANOVA=("ANOVA", "sum"), Splits_MI=("MI", "sum"),
                  Splits_RF=("RF", "sum")))
    return (table[table["Splits_selected_by_consensus"] > 0]
            .sort_values(["Splits_selected_by_consensus", "Taxon"],
                         ascending=[False, True]).reset_index(drop=True))


def compare_with_submitted_top20(old_dir, stability_by_variant: dict,
                                 min_splits: int = MIN_SPLITS_RETAINED) -> pd.DataFrame:
    """Which features of the submitted 'top-20 in >= 2 splits' list are kept or
    dropped by the new top-25% rule (in V0 and in the primary variant V1)."""
    old = pd.read_csv(Path(old_dir) / "SVM_coefficient_table_filtered_diet_plus_taxa.csv",
                      dtype={"Feature ID": str})
    count_col = [c for c in old.columns if "top 20" in c][0]
    old = old.rename(columns={count_col: "Submitted: splits in top 20"})
    old = old[["Feature ID", "Feature name", "Feature group",
               "Submitted: splits in top 20"]]
    old["Submitted: prioritised"] = old["Submitted: splits in top 20"] >= min_splits
    old["key"] = old["Feature ID"].map(base_nutrient)

    merged = old.copy()
    for variant, stability in stability_by_variant.items():
        table = stability["table"][["Feature ID", "Feature name", "Feature group",
                                    "Splits present in model",
                                    "Splits in top fraction",
                                    "Prioritised (new rule)"]].copy()
        table["key"] = table["Feature ID"].map(base_nutrient)
        table = table.rename(columns={
            "Feature name": f"{variant}: feature name",
            "Feature group": f"{variant}: group",
            "Splits present in model": f"{variant}: splits in model",
            "Splits in top fraction": f"{variant}: splits in top 25%",
            "Prioritised (new rule)": f"{variant}: prioritised"})
        merged = merged.merge(table.drop(columns="Feature ID"), on="key", how="outer")
    merged["Feature name"] = merged["Feature name"].fillna(
        merged[[c for c in merged.columns if c.endswith(": feature name")]]
        .bfill(axis=1).iloc[:, 0])
    merged["Feature group"] = merged["Feature group"].fillna(
        merged[[c for c in merged.columns if c.endswith(": group")]]
        .bfill(axis=1).iloc[:, 0])
    merged["Submitted: prioritised"] = merged["Submitted: prioritised"].eq(True)
    for variant in stability_by_variant:
        col = f"{variant}: prioritised"
        merged[col] = merged[col].eq(True)
        merged[f"{variant}: status vs submitted"] = np.select(
            [merged["Submitted: prioritised"] & merged[col],
             merged["Submitted: prioritised"] & ~merged[col],
             ~merged["Submitted: prioritised"] & merged[col]],
            ["kept", "drops out", "new"], default="")
    keep = merged["Submitted: prioritised"]
    for variant in stability_by_variant:
        keep = keep | merged[f"{variant}: prioritised"]
    merged = merged[keep].drop(
        columns=[c for c in merged.columns
                 if c.endswith(": feature name") or c.endswith(": group")])
    merged = merged.rename(columns={"key": "Feature key"})
    return merged.sort_values(["Feature group", "Submitted: splits in top 20"],
                              ascending=[True, False]).reset_index(drop=True)


# --- 2e: calibration ------------------------------------------------------------------

def brier_score(y_true, prob) -> float:
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


def calibration_summary(predictions: pd.DataFrame, membership: pd.DataFrame,
                        feature_sets: Sequence[str] = FEATURE_SETS) -> dict:
    """Brier score of the pooled out-of-fold predictions (90 test instances
    accumulated across the five repeated stratified splits) against constant
    reference predictors."""
    prevalence = N_DEPRESSIVE / N_PARTICIPANTS          # 15 / 88, computed
    train = membership[membership["role"] == "train"]
    train_prev = train.groupby("Split")["y"].mean()

    rows = []
    kinds = (("SVM", "prob_platt", "Platt (sigmoid) on training set"),
             ("LR", "prob_native", "native predict_proba (balanced class weights)"),
             ("LR", "prob_platt", "Platt (sigmoid) on training set"),
             ("BRF", "prob_native", "native predict_proba (balanced bootstrap)"))
    for fs_name in feature_sets:
        for model, column, kind in kinds:
            sub = predictions[(predictions["Input features"] == fs_name)
                              & (predictions["Model"] == model)]
            if sub.empty or sub[column].isna().all():
                continue
            y_true = sub["y_true"].to_numpy()
            prob = sub[column].to_numpy()
            assert len(sub) == N_TEST * len(SEEDS) == 90 and not np.isnan(prob).any()
            brier = brier_score(y_true, prob)
            ref_cohort = brier_score(y_true, np.full(len(y_true), prevalence))
            ref_train = brier_score(y_true, sub["Split"].map(train_prev).to_numpy())
            rows.append({
                "Input features": fs_name, "Model": model, "Probability": kind,
                "column": column, "n instances": len(sub),
                "n depressive instances": int(y_true.sum()),
                "Brier": brier,
                "Reference Brier (constant 15/88)": ref_cohort,
                "Brier skill score vs constant 15/88": 1.0 - brier / ref_cohort,
                "Lower than reference": bool(brier < ref_cohort),
                "Reference Brier (training-set prevalence per split)": ref_train,
                "Mean predicted probability": float(prob.mean()),
                "Min predicted probability": float(prob.min()),
                "Max predicted probability": float(prob.max()),
            })
    return {"table": pd.DataFrame(rows),
            "prevalence_15_of_88": prevalence,
            "p_times_one_minus_p": prevalence * (1.0 - prevalence),
            "pooled_test_prevalence": 15.0 / 90.0,
            "training_prevalence_per_split": train_prev.to_dict()}


def calibration_statement(calib: dict, calibration_params: pd.DataFrame,
                          feature_set: str = REFERENCE_CONFIG[0]) -> str:
    """Plain-language statement for R2.8 (is the Brier score lower than the
    reference or not)."""
    table = calib["table"]
    lines = [f"Prevalence 15/88 = {calib['prevalence_15_of_88']:.4f}; "
             f"p(1-p) = {calib['p_times_one_minus_p']:.4f}. Reference Brier of the "
             "constant predictor 15/88 evaluated on the same 90 pooled test "
             f"instances = {table['Reference Brier (constant 15/88)'].iloc[0]:.4f}."]
    for _, row in table[table["Input features"] == feature_set].iterrows():
        verdict = "LOWER than" if row["Lower than reference"] else "NOT lower than"
        lines.append(f"  {row['Input features']} / {row['Model']} "
                     f"[{row['Probability']}]: Brier = {row['Brier']:.4f} -> "
                     f"{verdict} the reference "
                     f"(skill score {row['Brier skill score vs constant 15/88']:+.3f}).")
    sub = calibration_params[(calibration_params["Input features"] == feature_set)
                             & (calibration_params["Model"] == "SVM")]
    n_inv = int(sub["slope_inverted"].sum())
    lines.append(f"  Platt scaling of the {feature_set} SVM: in {n_inv} of {len(sub)} "
                 "splits the sigmoid fitted on the training set has an inverted "
                 "slope (the inner cross-validated scores were not positively "
                 "related to the outcome), i.e. the training set held no usable "
                 "calibration signal in those splits.")
    return chr(10).join(lines)


def calibration_bins(y_true, prob, n_bins: int = CALIB_N_BINS,
                     strategy: str = CALIB_BIN_STRATEGY) -> pd.DataFrame:
    """Calibration table with a few bins. 'quantile' = equal-count bins of the
    sorted predictions; 'uniform' = equal-width bins on [0, 1]."""
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    if strategy == "quantile":
        order = np.argsort(prob, kind="stable")
        groups = np.array_split(order, n_bins)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        which = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
        groups = [np.flatnonzero(which == b) for b in range(n_bins)]
    else:
        raise ValueError(strategy)
    rows = []
    for b, idx in enumerate(groups, start=1):
        if len(idx) == 0:
            continue
        rows.append({"bin": b, "n": int(len(idx)),
                     "depressive": int(y_true[idx].sum()),
                     "mean predicted probability": float(prob[idx].mean()),
                     "observed fraction": float(y_true[idx].mean()),
                     "min predicted": float(prob[idx].min()),
                     "max predicted": float(prob[idx].max())})
    return pd.DataFrame(rows)


# --- 2f: participant characteristics ---------------------------------------------------

def fisher_freeman_halton_exact(table) -> float:
    """Exact two-sided Fisher (2 x 2) / Fisher-Freeman-Halton (r x 2) test.

    All tables with the observed margins are enumerated; the p-value is the
    total probability (multivariate hypergeometric) of the tables that are no
    more likely than the observed one. Exact and deterministic (no simulation).
    """
    from scipy.special import gammaln

    table = np.asarray(table, dtype=int)
    assert table.ndim == 2 and table.shape[1] == 2, "expected an r x 2 table"
    rows = table.sum(axis=1)
    n_first = int(table[:, 0].sum())
    total = int(rows.sum())

    def log_choose(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    log_denominator = log_choose(total, n_first)

    def log_prob(first_column):
        return sum(log_choose(r, a) for r, a in zip(rows, first_column)) - log_denominator

    def enumerate_columns(level, remaining):
        if level == len(rows) - 1:
            if 0 <= remaining <= rows[level]:
                yield (remaining,)
            return
        for a in range(0, min(rows[level], remaining) + 1):
            for rest in enumerate_columns(level + 1, remaining - a):
                yield (a,) + rest

    log_observed = log_prob(tuple(table[:, 0]))
    probabilities = np.exp([log_prob(col) for col in enumerate_columns(0, n_first)])
    assert abs(probabilities.sum() - 1.0) < 1e-9
    return float(probabilities[probabilities <= np.exp(log_observed) * (1 + 1e-7)].sum())


def categorical_test(levels, groups):
    """Fisher's exact test for 2 x 2 tables; chi-square when every expected
    count is >= 5; otherwise the exact Fisher-Freeman-Halton test (r x 2)."""
    table = pd.crosstab(levels, groups).to_numpy()
    expected = stats.contingency.expected_freq(table)
    if table.shape == (2, 2):
        p_value = fisher_freeman_halton_exact(table)
        assert abs(p_value - stats.fisher_exact(table)[1]) < 1e-9
        return "Fisher exact", p_value
    if (expected >= 5).all():
        return "Chi-square", float(stats.chi2_contingency(table, correction=False)[1])
    return "Fisher-Freeman-Halton exact", fisher_freeman_halton_exact(table)


def characteristic_variables(cohort: Cohort,
                             energy_denom: str = PRIMARY_ENERGY_DENOM) -> dict:
    """Variables of the group-characteristics table: the ten candidate
    confounders (raw codes), the Zung index score, Kcal and the eight nutrient
    densities of the cohort's diet file."""
    assert cohort.confounders_raw is not None
    variables = {name: cohort.confounders_raw[name].to_numpy(dtype=float)
                 for name in CONFOUNDER_CANDIDATES}
    variables["Zung_S (index score)"] = cohort.zung
    variables["Kcal"] = cohort.diet_raw["Kcal"].to_numpy(dtype=float)
    density = nutrient_density(cohort.diet_raw, energy_denom)
    for column in density.columns:
        variables[column] = density[column].to_numpy(dtype=float)
    return variables


def group_comparison_table(cohort: Cohort, variables: dict) -> pd.DataFrame:
    """Controls (n = 73) vs depressive (n = 15): median (Q1-Q3) and two-sided
    Mann-Whitney U test (asymptotic, tie-corrected, with continuity
    correction) for every variable."""
    control, depressive = cohort.y == 0, cohort.y == 1
    assert control.sum() == N_CONTROL and depressive.sum() == N_DEPRESSIVE
    rows = []
    for name, values in variables.items():
        values = np.asarray(values, dtype=float)
        assert len(values) == N_PARTICIPANTS and not np.isnan(values).any(), name
        c, d = values[control], values[depressive]
        cq1, cmed, cq3 = np.percentile(c, [25, 50, 75])
        dq1, dmed, dq3 = np.percentile(d, [25, 50, 75])
        u_stat, p_value = stats.mannwhitneyu(d, c, alternative="two-sided",
                                             method="asymptotic", use_continuity=True)
        rows.append({
            "Variable": name,
            f"Controls (n={N_CONTROL}) median (Q1–Q3)": f"{cmed:.2f} ({cq1:.2f}–{cq3:.2f})",
            f"Depressive (n={N_DEPRESSIVE}) median (Q1–Q3)": f"{dmed:.2f} ({dq1:.2f}–{dq3:.2f})",
            "Mann-Whitney U (depressive vs control)": float(u_stat),
            "p (two-sided)": float(p_value),
            "P(depressive > control)": float(u_stat / (len(d) * len(c))),
            "controls_median": cmed, "controls_q1": cq1, "controls_q3": cq3,
            "depressive_median": dmed, "depressive_q1": dq1, "depressive_q3": dq3,
        })
    return pd.DataFrame(rows)


def compare_with_crosscheck(table: pd.DataFrame, crosscheck_csv,
                            tolerance: float = 0.005) -> pd.DataFrame:
    """Compare the Mann-Whitney p-values with the independent cross-check file."""
    check = pd.read_csv(crosscheck_csv)
    p_col = [c for c in check.columns if "two-sided p" in c][0]
    name_map = {"Age": "Age", "BMI": "BMI", "WHR": "WHR", "Parasite": "Parasite",
                "LoE": "LoE", "SES": "SES", "PA": "PA", "SQ": "SQ_all",
                "Bristol": "Bristol", "ALI-7": "ALI_7"}
    rows = []
    for _, row in check.iterrows():
        short = str(row["variable"]).split(" ")[0]
        assert short in name_map, f"unknown variable in the cross-check file: {row['variable']}"
        ours = table.loc[table["Variable"] == name_map[short], "p (two-sided)"].iloc[0]
        rows.append({"Variable": name_map[short], "p (this pipeline)": float(ours),
                     "p (cross-check file)": float(row[p_col]),
                     "abs difference": abs(float(ours) - float(row[p_col])),
                     f"difference > {tolerance}": bool(abs(float(ours) - float(row[p_col])) > tolerance)})
    return pd.DataFrame(rows)


def categorical_counts_table(cohort: Cohort,
                             names: Sequence[str] = CATEGORICAL_CHARACTERISTICS
                             ) -> pd.DataFrame:
    """n (%) per code in both groups for the coded variables, with an exact
    test (Fisher 2 x 2 / Fisher-Freeman-Halton r x 2, or chi-square when all
    expected counts are >= 5). Supplementary to the Mann-Whitney table."""
    control, depressive = cohort.y == 0, cohort.y == 1
    group_labels = np.where(depressive, "depressive", "control")
    rows = []
    for name in names:
        codes = cohort.confounders_raw[name].to_numpy()
        test_name, p_value = categorical_test(codes, group_labels)
        for i, code in enumerate(sorted(np.unique(codes))):
            n_c = int(((codes == code) & control).sum())
            n_d = int(((codes == code) & depressive).sum())
            rows.append({"Variable": name, "Code": int(code),
                         f"Controls (n={N_CONTROL})": f"{n_c} ({100 * n_c / N_CONTROL:.1f}%)",
                         f"Depressive (n={N_DEPRESSIVE})": f"{n_d} ({100 * n_d / N_DEPRESSIVE:.1f}%)",
                         "Test": test_name if i == 0 else "",
                         "p": float(p_value) if i == 0 else np.nan,
                         "controls_n": n_c, "depressive_n": n_d})
    return pd.DataFrame(rows)


def confounder_collinearity(cohort: Cohort) -> dict:
    """Spearman correlations and variance inflation factors of the ten
    candidate confounders (all 88 participants, no outcome)."""
    frame = confounder_frame(cohort, CONFOUNDER_CANDIDATES)
    corr = frame.corr(method="spearman")
    z = (frame - frame.mean()) / frame.std(ddof=0)
    inverse = np.linalg.inv(np.corrcoef(z.to_numpy().T))
    vif = pd.Series(np.diag(inverse), index=frame.columns, name="VIF")
    return {"spearman": corr, "vif": vif}


# =============================================================================
# 12. Figures
# =============================================================================

def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax, grid_axis="y"):
    """Recessive axes and grid; text in ink colours."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.xaxis.label.set_color(COLOR_INK)
    ax.yaxis.label.set_color(COLOR_INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out_dir, stem) -> list:
    """Save a figure as PDF and PNG (300 dpi) and return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    return written


def plot_zung_distribution(cohort: Cohort, out_dir, stem="figR1_1_zung_distribution"):
    """R1.1: distribution of the Zung index score in both groups, cut-off at 50."""
    plt = _plt()
    control, depressive = cohort.zung[cohort.y == 0], cohort.zung[cohort.y == 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    edges = np.arange(25.0, 67.5 + 0.01, 2.5)          # 50 is a bin edge
    assert cohort.zung.min() >= edges[0] and cohort.zung.max() <= edges[-1]
    ax.hist([control, depressive], bins=edges, stacked=True,
            color=[COLOR_CONTROL, COLOR_DEPRESSIVE], edgecolor="white",
            linewidth=1.2, zorder=2,
            label=[f"Control (n = {len(control)})",
                   f"Depressive (n = {len(depressive)})"])
    ax.axvline(ZUNG_CUTOFF, color=COLOR_INK, linestyle="--", linewidth=1.2, zorder=3,
               label="Cut-off: index score 50")
    ax.set_xlabel("Zung SDS index score (raw score × 1.25)")
    ax.set_ylabel("Number of participants")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("(a) Score distribution", fontsize=10, loc="left", color=COLOR_INK)
    _style(ax)

    ax = axes[1]
    rng = np.random.default_rng(0)
    for pos, (values, color) in enumerate(((control, COLOR_CONTROL),
                                           (depressive, COLOR_DEPRESSIVE))):
        ax.boxplot(values, positions=[pos], widths=0.5, showfliers=False,
                   patch_artist=True, zorder=2,
                   boxprops={"facecolor": "none", "edgecolor": COLOR_MUTED},
                   medianprops={"color": COLOR_INK, "linewidth": 1.6},
                   whiskerprops={"color": COLOR_MUTED},
                   capprops={"color": COLOR_MUTED})
        ax.scatter(pos + rng.uniform(-0.16, 0.16, len(values)), values, s=22,
                   color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(ZUNG_CUTOFF, color=COLOR_INK, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Control\n(n = {len(control)})",
                        f"Depressive\n(n = {len(depressive)})"])
    ax.set_ylabel("Zung SDS index score")
    ax.set_title("(b) Scores by group", fontsize=10, loc="left", color=COLOR_INK)
    _style(ax)

    fig.tight_layout()
    paths = _save(fig, out_dir, stem)
    summary = {
        "control": {"n": int(len(control)), "min": float(control.min()),
                    "median": float(np.median(control)), "max": float(control.max())},
        "depressive": {"n": int(len(depressive)), "min": float(depressive.min()),
                       "median": float(np.median(depressive)),
                       "max": float(depressive.max())},
        "n_exactly_50": int(np.isclose(cohort.zung, ZUNG_CUTOFF).sum()),
    }
    return fig, paths, summary


def mean_roc_curve(predictions: pd.DataFrame, config=REFERENCE_CONFIG,
                   n_grid: int = 1501) -> dict:
    """Per-split empirical ROC curves evaluated on a common FPR grid and their
    vertical (pointwise) mean.

    Each split's curve is the empirical step function TPR(f) = highest
    sensitivity reached at a false-positive rate <= f, so the area under every
    curve equals that split's ROC-AUC and the area under the mean curve equals
    the mean ROC-AUC that is printed in the figure (asserted below).

    Note: the submitted R script joined the ROC vertices by straight lines
    before averaging, which draws the curve slightly above the empirical step
    function; its area is reported as 'area_linear_construction' for
    comparison.
    """
    sub = predictions[(predictions["Input features"] == config[0])
                      & (predictions["Model"] == config[1])]
    grid = np.linspace(0.0, 1.0, n_grid)
    curves, linear_curves, aucs = {}, {}, {}
    for split, block in sub.groupby("Split"):
        fpr, tpr, _ = roc_curve(block["y_true"], block["y_score"],
                                drop_intermediate=False)
        frame = pd.DataFrame({"fpr": fpr, "tpr": tpr}).groupby("fpr")["tpr"].max()
        knots, heights = frame.index.to_numpy(), frame.to_numpy()
        position = np.searchsorted(knots - 1e-9, grid, side="right") - 1
        curves[int(split)] = heights[position]
        linear_curves[int(split)] = np.interp(grid, knots, heights)
        aucs[int(split)] = float(roc_auc_score(block["y_true"], block["y_score"]))
    mean_tpr = np.mean(list(curves.values()), axis=0)
    values = np.array(list(aucs.values()))

    def area(curve):
        return float(np.sum((curve[1:] + curve[:-1]) / 2.0 * np.diff(grid)))

    area_step = area(mean_tpr)
    assert abs(area_step - values.mean()) < 0.005, \
        "area under the mean ROC curve does not match the mean ROC-AUC"
    return {"grid": grid, "curves": curves, "mean_tpr": mean_tpr, "aucs": aucs,
            "mean_auc": float(values.mean()), "sd_auc": float(values.std(ddof=SD_DDOF)),
            "min_auc": float(values.min()), "max_auc": float(values.max()),
            "area_under_mean_curve": area_step,
            "area_linear_construction": area(np.mean(list(linear_curves.values()),
                                                     axis=0))}


def plot_figure2(predictions: pd.DataFrame, out_dir, config=REFERENCE_CONFIG,
                 stem_prefix="fig2", show_split_curves: bool = True):
    """R1.2: Figure 2a (mean ROC curve with the mean AUC labelled) and
    Figure 2b (confusion matrix of the pooled test predictions)."""
    plt = _plt()
    roc = mean_roc_curve(predictions, config)
    sub = predictions[(predictions["Input features"] == config[0])
                      & (predictions["Model"] == config[1])]
    counts = confusion_counts(sub["y_true"], sub["y_pred"])
    n_instances = len(sub)
    assert n_instances == sum(counts.values()) == N_TEST * len(SEEDS)

    def draw_roc(ax):
        if show_split_curves:
            for i, (split, curve) in enumerate(roc["curves"].items()):
                ax.plot(roc["grid"], curve, color=COLOR_NEUTRAL,
                        linewidth=0.9, alpha=0.8, zorder=2,
                        label="Individual splits" if i == 0 else None)
        ax.plot(roc["grid"], roc["mean_tpr"], color=COLOR_CONTROL,
                linewidth=2.2, zorder=3, label="Mean ROC curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_MUTED, linewidth=0.9,
                zorder=1, label="Chance")
        ax.text(0.97, 0.05,
                f"Mean AUC = {roc['mean_auc']:.3f}\n"
                f"SD = {roc['sd_auc']:.3f}; range {roc['min_auc']:.3f}–"
                f"{roc['max_auc']:.3f}\n({WORDING_SPLITS})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=COLOR_INK,
                bbox={"facecolor": "white", "edgecolor": COLOR_GRID,
                      "boxstyle": "round,pad=0.4"})
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, fontsize=8.5, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), ncol=3, handlelength=1.8,
                  columnspacing=1.2)
        _style(ax, grid_axis="both")

    def draw_cm(ax):
        matrix = np.array([[counts["TN"], counts["FP"]],
                           [counts["FN"], counts["TP"]]])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.25)
        row_tot = matrix.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                dark = matrix[i, j] > 0.6 * matrix.max()
                ax.text(j, i, f"{matrix[i, j]}\n{100 * matrix[i, j] / row_tot[i, 0]:.1f}%",
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if dark else COLOR_INK)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Depressive"])
        ax.set_yticklabels(["Control", "Depressive"])
        ax.set_xlabel(f"Predicted label\n(n = {n_instances} pooled test instances)")
        ax.set_ylabel("True label")
        ax.tick_params(colors=COLOR_MUTED, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_color(COLOR_MUTED)
            spine.set_linewidth(0.8)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label("Test instances", fontsize=9, color=COLOR_INK)
        bar.ax.tick_params(labelsize=8, colors=COLOR_MUTED)

    written = []
    fig_a, ax_a = plt.subplots(figsize=(5.2, 5.2))
    draw_roc(ax_a)
    fig_a.tight_layout()
    written += _save(fig_a, out_dir, f"{stem_prefix}_panel_a_svm_mean_roc_curve")

    fig_b, ax_b = plt.subplots(figsize=(5.4, 4.6))
    draw_cm(ax_b)
    fig_b.tight_layout()
    written += _save(fig_b, out_dir, f"{stem_prefix}_panel_b_svm_pooled_confusion_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_roc(axes[0])
    axes[0].text(-0.14, 1.06, "(a)", transform=axes[0].transAxes, fontsize=12,
                 color=COLOR_INK)
    draw_cm(axes[1])
    axes[1].text(-0.22, 1.06, "(b)", transform=axes[1].transAxes, fontsize=12,
                 color=COLOR_INK)
    fig.tight_layout()
    written += _save(fig, out_dir, f"{stem_prefix}_svm_roc_and_confusion_matrix")

    return {"figure": fig, "paths": written, "roc": roc, "confusion": counts,
            "n_instances": n_instances}


def figure2_legend_text(fig2: dict, composition: dict, variant_label: str,
                        feature_set_text: str = "combined confounder-diet-taxa") -> str:
    """Suggested legend for Figure 2 (plain text; the tex is edited by hand)."""
    roc, c = fig2["roc"], fig2["confusion"]
    per_split = ", ".join(f"{v:.3f}" for v in roc["aucs"].values())
    return (
        "SUGGESTED LEGEND - Figure 2 (do not paste blindly; edit as needed)\n"
        f"Source run: {variant_label}\n\n"
        "Figure 2. Internal test performance of the Linear SVM using the "
        f"{feature_set_text} feature set. "
        f"(a) Mean ROC curve across the {WORDING_SPLITS} (seeds 42-46; each split: "
        "40% feature-selection set, 40% training set, 20% test set); grey lines show "
        "the individual splits. "
        f"Mean ROC-AUC = {roc['mean_auc']:.3f} (SD {roc['sd_auc']:.3f}; range "
        f"{roc['min_auc']:.3f}-{roc['max_auc']:.3f}; per-split values: {per_split}). "
        f"(b) Confusion matrix of the pooled test predictions (TN = {c['TN']}, "
        f"FP = {c['FP']}, FN = {c['FN']}, TP = {c['TP']}; percentages are row "
        f"percentages). The {fig2['n_instances']} entries are test instances "
        f"accumulated across the five repeated splits (5 x {N_TEST} = "
        f"{fig2['n_instances']}; each test set: {N_TEST_DEPRESSIVE} depressive and "
        f"{N_TEST_CONTROL} control participants), not {fig2['n_instances']} "
        f"independent individuals: the cohort has {N_PARTICIPANTS} participants, "
        f"{composition['unique_test_participants']} of whom appear in at least one "
        f"test set and {composition['participants_in_more_than_one_test_set']} in "
        "more than one. All test sets are internal hold-out subsets of the same "
        "cohort; this is internal validation, not external validation. Because the "
        "splits overlap, the five results are correlated and should not be read as "
        "five independent experiments.\n")


def plot_auc_per_split(metrics: pd.DataFrame, out_dir,
                       stem="figS_auc_per_split_all_configurations", title=None):
    """All 75 ROC-AUC values (15 configurations x 5 splits) with their means."""
    plt = _plt()
    configs = all_configurations()
    fig, ax = plt.subplots(figsize=(15.0, 4.6))
    colors = {"LR": COLOR_CONTROL, "SVM": COLOR_DEPRESSIVE, "BRF": COLOR_THIRD}
    markers = {"LR": "o", "SVM": "s", "BRF": "^"}
    positions, labels = [], []
    for i, (fs_name, model) in enumerate(configs):
        x = i + (i // len(MODELS)) * 0.6
        positions.append(x)
        labels.append(model)
        values = metrics[(metrics["Input features"] == fs_name)
                         & (metrics["Model"] == model)].sort_values("Split")["ROC-AUC"]
        offsets = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(x + offsets, values, s=34, color=colors[model],
                   marker=markers[model], edgecolor="white", linewidth=0.7,
                   zorder=3, label=model if i < len(MODELS) else None)
        ax.hlines(values.mean(), x - 0.3, x + 0.3, color=COLOR_INK, linewidth=1.8,
                  zorder=4)
        ax.text(x, 1.04, f"{values.mean():.3f}", ha="center", va="bottom",
                fontsize=8.5, color=COLOR_INK)
    ax.axhline(0.5, color=COLOR_MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    for g, fs_name in enumerate(FEATURE_SETS):
        centre = np.mean(positions[g * len(MODELS):(g + 1) * len(MODELS)])
        ax.text(centre, -0.19, fs_name, ha="center", va="top", fontsize=10,
                color=COLOR_INK, transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("ROC-AUC (test set)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right",
              title="Points: individual splits; bar and number: mean",
              title_fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=COLOR_INK)
    _style(ax)
    fig.tight_layout()
    return fig, _save(fig, out_dir, stem)


def plot_calibration(predictions: pd.DataFrame, calib: dict, out_dir,
                     feature_set: str = REFERENCE_CONFIG[0],
                     stem="figR2_8_calibration", n_bins: int = CALIB_N_BINS,
                     strategy: str = CALIB_BIN_STRATEGY):
    """R2.8: calibration plot of the pooled out-of-fold predictions with a few
    bins. One panel per model (SVM with Platt scaling; LR native
    probabilities); each point is one bin, labelled 'depressive / n'."""
    plt = _plt()
    series = (("SVM", "prob_platt", "Linear SVM + Platt scaling", COLOR_CONTROL, "o"),
              ("LR", "prob_native", "Logistic regression (native probabilities)",
               COLOR_DEPRESSIVE, "s"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    bins = {}
    table = calib["table"]
    for ax, (model, column, label, color, marker) in zip(axes, series):
        sub = predictions[(predictions["Input features"] == feature_set)
                          & (predictions["Model"] == model)]
        frame = calibration_bins(sub["y_true"], sub[column], n_bins, strategy)
        bins[(model, column)] = frame
        row = table[(table["Input features"] == feature_set)
                    & (table["Model"] == model) & (table["column"] == column)].iloc[0]
        upper = max(frame["mean predicted probability"].max(),
                    frame["observed fraction"].max())
        limit = min(1.0, max(0.5, math.ceil(upper * 10 + 1) / 10))
        ax.plot([0, limit], [0, limit], linestyle="--", color=COLOR_MUTED,
                linewidth=0.9, zorder=1, label="Perfect calibration")
        ax.axhline(calib["pooled_test_prevalence"], color=COLOR_NEUTRAL,
                   linewidth=0.9, linestyle=":", zorder=1,
                   label="Observed fraction in the pooled test instances "
                         f"({calib['pooled_test_prevalence']:.3f})")
        ax.plot(frame["mean predicted probability"], frame["observed fraction"],
                color=color, marker=marker, markersize=8, linewidth=1.8,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                label=f"{n_bins} {strategy} bins of the {len(sub)} test instances")
        for _, point in frame.iterrows():
            above = point["observed fraction"] >= point["mean predicted probability"]
            ax.annotate(f"{int(point['depressive'])}/{int(point['n'])}",
                        (point["mean predicted probability"], point["observed fraction"]),
                        textcoords="offset points",
                        xytext=(0, 9 if above else -15), ha="center", fontsize=8.5,
                        color=COLOR_INK)
        ax.set_xlim(-0.02 * limit, limit)
        ax.set_ylim(-0.02 * limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted probability (per bin)")
        ax.set_ylabel("Observed fraction depressive (per bin)")
        ax.set_title(f"{label}\nBrier = {row['Brier']:.3f} "
                     f"(constant 15/88: {row['Reference Brier (constant 15/88)']:.3f})",
                     fontsize=9.5, loc="left", color=COLOR_INK)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        _style(ax, grid_axis="both")
    fig.tight_layout()
    return fig, _save(fig, out_dir, stem), bins


def plot_permutation_null(null_max: np.ndarray, observed: float, p_value: float,
                          out_dir, stem="figR2_4_permutation_null", n_configs: int = 15):
    """R2.4: null distribution of the maximum mean ROC-AUC over the configurations."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(null_max, bins=30, color=COLOR_CONTROL, edgecolor="white",
            linewidth=1.0, zorder=2)
    ax.axvline(observed, color=COLOR_DEPRESSIVE, linewidth=2, zorder=3)
    ax.text(observed, ax.get_ylim()[1] * 0.97,
            f"  observed best of {n_configs} = {observed:.3f}\n  p = {p_value:.4f} "
            f"(B = {len(null_max)})", ha="left", va="top", fontsize=9,
            color=COLOR_INK)
    ax.set_xlabel(f"Maximum over the {n_configs} configurations of the mean ROC-AUC "
                  "(labels permuted)")
    ax.set_ylabel("Number of permutations")
    _style(ax)
    fig.tight_layout()
    return fig, _save(fig, out_dir, stem)


# =============================================================================
# 13. Permutation test (Step 3)
# =============================================================================

PERM_CONFIG_COLUMNS = [f"{fs} | {m}" for fs, m in all_configurations()]


def pipeline_mean_aucs(cohort: Cohort, y: np.ndarray, variant: str = "V1",
                       energy_denom: str = PRIMARY_ENERGY_DENOM, n_jobs=1,
                       allow_empty_consensus: bool = True, *,
                       confounders: Optional[pd.DataFrame] = None) -> dict:
    """Run the ENTIRE pipeline for a given label vector and return the mean
    ROC-AUC over the five splits for each of the 15 configurations. The
    confounder values stay attached to the participants; only the outcome
    labels differ between calls.

    Everything is redone from the labels: correlation filter (label-free, so
    identical each time), stratified splits (seeds 42-46, stratified on the
    labels passed in), taxa feature selection (3 methods + consensus), model
    training and testing.
    """
    design = build_diet_design(cohort, variant, energy_denom)
    sums = {config: 0.0 for config in all_configurations()}
    n_empty = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for split_no, seed in enumerate(SEEDS, start=1):
            res = run_split(cohort, y, seed, split_no, design, n_jobs=n_jobs,
                            confounders=confounders, detail=False, calibrate=False,
                            allow_empty_consensus=allow_empty_consensus)
            n_empty += int(res["empty_consensus"])
            for config in sums:
                sums[config] += res["aucs"][config]
    means = {config: total / len(SEEDS) for config, total in sums.items()}
    return {"mean_auc": means, "max_auc": max(means.values()),
            "n_empty_consensus_splits": n_empty}


def permuted_labels(y: np.ndarray, perm_index: int,
                    base_seed: int = PERM_BASE_SEED) -> np.ndarray:
    """Shuffle the outcome labels across the 88 participants (the number of
    depressive participants stays 15). Seed = (base_seed, perm_index)."""
    rng = np.random.default_rng([base_seed, perm_index])
    permuted = rng.permutation(np.asarray(y))
    assert permuted.sum() == N_DEPRESSIVE and len(permuted) == N_PARTICIPANTS
    return permuted


def permutation_design(cohort: Cohort, variant: str, energy_denom: str,
                       base_seed: int, confounders: pd.DataFrame) -> dict:
    """Everything that defines a permutation null; stored next to the
    checkpoint and compared before resuming."""
    design = build_diet_design(cohort, variant, energy_denom)
    return {
        "variant": variant, "energy_denom": energy_denom,
        "confounders": list(confounders.columns),
        "diet_file": cohort.diet_label, "taxa_table": cohort.taxa_label,
        "n_taxa": int(cohort.n_taxa), "label_rule": LABEL_RULE,
        "base_seed": int(base_seed), "split_seeds": list(SEEDS),
        "diet_model_inputs": list(design.feature_names),
        "configurations": list(PERM_CONFIG_COLUMNS),
        "rf_trees": RF_TREES, "anova_p_threshold": ANOVA_P_THRESHOLD,
        "mi_percentile": MI_PERCENTILE,
        "inputs_sha256": {k: v["sha256"] for k, v in cohort.sources.items()
                          if k in ("PART", "TAXA", "DIET")},
    }


def _one_permutation(cohort, perm_index, variant, energy_denom, base_seed, confounders):
    started = time.time()
    y_perm = permuted_labels(cohort.y, perm_index, base_seed)
    res = pipeline_mean_aucs(cohort, y_perm, variant, energy_denom, n_jobs=1,
                             confounders=confounders)
    row = {"perm_index": perm_index, "base_seed": base_seed}
    for (fs_name, model), value in res["mean_auc"].items():
        row[f"{fs_name} | {model}"] = value
    row["max_auc"] = res["max_auc"]
    row["n_empty_consensus_splits"] = res["n_empty_consensus_splits"]
    row["seconds"] = round(time.time() - started, 2)
    return row


def run_permutations(cohort: Cohort, out_dir, *, n_perm: int = PERM_B,
                     variant: str = "V1", energy_denom: str = PRIMARY_ENERGY_DENOM,
                     base_seed: int = PERM_BASE_SEED, n_jobs: int = -1,
                     chunk_size: int = 20, verbose: bool = True,
                     confounders: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Max-statistic permutation null with checkpointing and resume support.

    Completed permutations are appended to perm_checkpoint.csv after every
    chunk; calling the function again continues with the missing permutation
    indices only. Each permutation has its own fixed seed, so the result does
    not depend on the number of workers or on interruptions.
    """
    from joblib import Parallel, delayed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "perm_checkpoint.csv"
    design_file = out_dir / "perm_design.json"
    assert confounders is not None
    design_now = permutation_design(cohort, variant, energy_denom, base_seed, confounders)
    done = pd.DataFrame()
    if checkpoint.exists():
        # Never resume a checkpoint that belongs to a different pipeline.
        assert design_file.exists(), \
            f"{checkpoint} exists without perm_design.json: cannot verify it"
        saved = _read_json(design_file)
        assert saved == design_now, (
            "the checkpoint in this folder was created with a different "
            f"pipeline design:\n saved: {saved}\n now:   {design_now}")
        done = pd.read_csv(checkpoint)
        assert (done["base_seed"] == base_seed).all(), \
            "checkpoint was created with a different base seed"
    else:
        _write_json(design_file, design_now)
    finished = set(done["perm_index"].tolist()) if len(done) else set()
    pending = [i for i in range(n_perm) if i not in finished]
    if verbose:
        print(f"permutations requested: {n_perm} | already done: {len(finished)} "
              f"| to run: {len(pending)}")

    started = time.time()
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_one_permutation)(cohort, i, variant, energy_denom, base_seed,
                                      confounders)
            for i in chunk)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        temp = checkpoint.with_suffix(".tmp")
        done.to_csv(temp, index=False)
        os.replace(temp, checkpoint)                     # atomic checkpoint
        if verbose:
            n_now = start + len(chunk)
            elapsed = time.time() - started
            eta = elapsed / n_now * (len(pending) - n_now)
            print(f"  {len(done)}/{n_perm} done | elapsed {elapsed / 60:.1f} min "
                  f"| remaining ~{eta / 60:.1f} min", flush=True)

    done = done.sort_values("perm_index").reset_index(drop=True)
    return done[done["perm_index"] < n_perm].reset_index(drop=True)


def permutation_result(null: pd.DataFrame, observed_means: dict, n_perm: int) -> dict:
    """Max-statistic permutation p-value:
    p = (1 + #{permuted max >= observed max}) / (1 + B)."""
    assert len(null) == n_perm and null["perm_index"].is_unique, \
        f"null distribution incomplete: {len(null)} of {n_perm} permutations"
    null_max = null["max_auc"].to_numpy(dtype=float)
    assert np.allclose(null_max, null[PERM_CONFIG_COLUMNS].max(axis=1))
    observed_max = max(observed_means.values())
    eps = 1e-9
    # mean AUCs are multiples of 1/225, so exact ties are possible: list them all
    tied = [c for c, v in observed_means.items() if v >= observed_max - eps]
    n_ge = int((null_max >= observed_max - eps).sum())
    rows = []
    for config, value in observed_means.items():
        column = f"{config[0]} | {config[1]}"
        own = null[column].to_numpy(dtype=float)
        rows.append({
            "Input features": config[0], "Model": config[1],
            "Observed mean ROC-AUC": value,
            "Null mean (same configuration)": float(own.mean()),
            "Unadjusted p (same configuration)":
                (1 + int((own >= value - eps).sum())) / (1 + n_perm),
            "Max-statistic adjusted p":
                (1 + int((null_max >= value - eps).sum())) / (1 + n_perm),
        })
    return {
        "B": n_perm, "observed_max": float(observed_max),
        "observed_best_configuration": "; ".join(f"{c[0]} | {c[1]}" for c in tied),
        "n_configurations_tied_at_maximum": len(tied),
        "n_perm_max_ge_observed": n_ge,
        "p_value": (1 + n_ge) / (1 + n_perm),
        "null_max_mean": float(null_max.mean()),
        "null_max_median": float(np.median(null_max)),
        "null_max_p95": float(np.percentile(null_max, 95)),
        "null_max_min": float(null_max.min()),
        "null_max_max": float(null_max.max()),
        "n_permutations_with_empty_consensus":
            int((null["n_empty_consensus_splits"] > 0).sum()),
        "per_configuration": pd.DataFrame(rows),
    }


def observed_mean_aucs(run: dict) -> dict:
    """Mean ROC-AUC per configuration of a saved run (the observed statistic)."""
    summary = run["summary"]
    return {(row["Input features"], row["Model"]): float(row["ROC-AUC mean"])
            for _, row in summary.iterrows()
            if row["Input features"] in FEATURE_SETS}


# =============================================================================
# 14. Results manifest
# =============================================================================

class Manifest:
    """Collects 'number / table / figure -> file, script, date' rows."""

    COLUMNS = ["item", "type", "value", "source_file", "script",
               "date_generated", "note"]

    def __init__(self, r2_dir):
        self.r2_dir = Path(r2_dir)
        self.rows = []

    def add(self, item, kind, value, source_file, script, note=""):
        path = Path(source_file)
        date = (_dt.datetime.fromtimestamp(path.stat().st_mtime)
                .strftime("%Y-%m-%d %H:%M") if path.exists() else "MISSING")
        try:
            rel = path.resolve().relative_to(self.r2_dir.resolve()).as_posix()
        except ValueError:
            rel = str(path)
        if isinstance(value, (float, np.floating)):
            value = f"{value:.6g}"
        self.rows.append({"item": item, "type": kind, "value": value,
                          "source_file": rel, "script": script,
                          "date_generated": date, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)

    def save(self, name="results_manifest.csv") -> Path:
        path = self.r2_dir / name
        self.frame().to_csv(path, index=False)
        return path


# File names of the Step 2 / Step 3 outputs (used by the notebooks and by the
# manifest so that both always agree).
OUT_FILES = {
    "table1": "table1_mean_sd_minmax.csv",
    "table1_formatted": "table1_formatted.csv",
    "auc45": "tableS_auc_all_45.csv",
    "spe45": "tableS_specificity_per_split.csv",
    "composition": "test_set_composition.csv",
    "composition_statement": "test_set_statement.txt",
    "bootstrap": "paired_bootstrap_vs_combined_svm.csv",
    "bootstrap_draws": "paired_bootstrap_draws.csv",
    "coef_all": "coef_rule_{model}_all_features.csv",
    "coef_prioritised": "coef_rule_{model}_prioritised.csv",
    "coef_per_split": "coef_rule_{model}_per_split.csv",
    "consensus": "taxa_consensus_frequency.csv",
    "brier": "brier_scores.csv",
    "calib_bins": "calibration_bins_{model}_{kind}.csv",
}
SHARED_OUT_FILES = {
    "top20_comparison": "coef_rule_comparison_with_submitted_top20.csv",
    "characteristics": "group_characteristics_mannwhitney.csv",
    "characteristics_crosscheck": "group_characteristics_p_vs_crosscheck.csv",
    "categorical_counts": "group_characteristics_coded_variables_counts.csv",
    "confounder_sensitivity": "confounder_sensitivity_SENS_A_B_C.csv",
    "confounder_spearman": "confounder_spearman_matrix.csv",
    "confounder_vif": "confounder_vif.csv",
    "decomposition": "effect_of_diet_file_and_uncultured_removal.csv",
    "spearman9_new": "diet_spearman_8densities_plus_Kcal_newDiet.csv",
    "spearman9_old": "diet_spearman_8densities_plus_Kcal_oldDiet.csv",
    "ffq_plausibility": "ffq_plausibility.json",
    "variant_comparison": "variant_comparison_mean_sd.csv",
    "zung_summary": "zung_distribution_summary.json",
}
PERM_FILES = {
    "checkpoint": "perm_checkpoint.csv",
    "null": "perm_null_distribution.csv",
    "result": "perm_result.json",
    "per_config": "perm_per_configuration.csv",
    "timing": "perm_timing.json",
}


def variant_comparison(runs: dict) -> pd.DataFrame:
    """Long table: mean, SD, min, max of every metric for every variant and
    configuration (runs = {label: loaded run})."""
    frames = []
    for label, run in runs.items():
        summary = run["summary"].copy()
        summary.insert(0, "Variant", label)
        summary.insert(1, "Diet model inputs",
                       ", ".join(run["config"]["diet_model_inputs"]))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def permutation_family_result(null: pd.DataFrame, observed_means: dict, n_perm: int,
                              feature_sets: Sequence[str], label: str) -> dict:
    """Max-statistic permutation p-value for a FAMILY (subset) of the
    configurations, computed from the same permutations:
    statistic = maximum mean ROC-AUC over the configurations of the family."""
    assert len(null) == n_perm
    configs = [(fs, model) for fs in feature_sets for model in MODELS]
    columns = [f"{fs} | {model}" for fs, model in configs]
    null_max = null[columns].max(axis=1).to_numpy(dtype=float)
    observed = {c: observed_means[c] for c in configs}
    observed_max = max(observed.values())
    eps = 1e-9
    tied = [c for c, v in observed.items() if v >= observed_max - eps]
    n_ge = int((null_max >= observed_max - eps).sum())
    return {
        "family": label, "feature_sets": list(feature_sets),
        "n_configurations": len(configs), "B": n_perm,
        "observed_max": float(observed_max),
        "observed_best_configuration": "; ".join(f"{c[0]} | {c[1]}" for c in tied),
        "n_perm_max_ge_observed": n_ge,
        "p_value": (1 + n_ge) / (1 + n_perm),
        "null_max_mean": float(null_max.mean()),
        "null_max_median": float(np.median(null_max)),
        "null_max_p95": float(np.percentile(null_max, 95)),
    }


def spearman_with_p(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlations with two-sided p-values (all rows of the
    frame, no outcome). Long table, one row per pair."""
    from scipy.stats import spearmanr
    rows = []
    names = list(frame.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho, p = spearmanr(frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float))
            rows.append({"Variable 1": a, "Variable 2": b, "Spearman rho": float(rho),
                         "p (two-sided)": float(p), "n": int(len(frame)),
                         "abs rho >= 0.30": bool(abs(rho) >= 0.30)})
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def selected_taxa_per_split(selection: pd.DataFrame) -> pd.DataFrame:
    """Taxa that pass the 2-of-3 consensus in each split (one row per split
    and taxon) with the number of splits in which the taxon is selected."""
    chosen = selection[selection["Consensus"] == 1].copy()
    count = chosen.groupby("rankID")["Consensus"].sum().rename("Splits_selected_by_consensus")
    chosen = chosen.merge(count, on="rankID")
    columns = ["Split", "random_state", "rankID", "Taxon", "ANOVA", "MI", "RF",
               "Splits_selected_by_consensus"]
    return chosen[columns].sort_values(["Split", "Taxon"]).reset_index(drop=True)


def selection_stability(selection: pd.DataFrame) -> dict:
    """Summary of how stable the consensus selection is across the splits."""
    chosen = selection[selection["Consensus"] == 1]
    sets = {int(s): set(block["rankID"]) for s, block in chosen.groupby("Split")}
    count = chosen.groupby("rankID").size()
    jaccard = []
    splits = sorted(sets)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            jaccard.append(len(sets[a] & sets[b]) / len(sets[a] | sets[b]))
    return {"taxa_per_split": {s: len(sets[s]) for s in splits},
            "distinct_taxa_selected": int(len(count)),
            "selected_in_at_least_2_splits": int((count >= 2).sum()),
            "selected_in_at_least_3_splits": int((count >= 3).sum()),
            "selected_in_all_5_splits": int((count >= 5).sum()),
            "mean_pairwise_jaccard": float(np.mean(jaccard))}


def build_manifest(paths: Paths) -> pd.DataFrame:
    """Write results_manifest.csv: every number, table and figure -> the file
    that holds it, the script that produced it and the date the file was
    written. Only existing files are listed."""
    man = Manifest(paths.r2)
    sc_0, sc_a, sc_b, sc_c = ("run_0_prepare_inputs.py", "run_A_main.py",
                              "run_B_outputs.py", "run_C_permutation.py")

    prep = paths.inputs / "prepare_report.json"
    if prep.exists():
        man.add("Inputs | preparation report (diet file, taxa tables, confounders)",
                "table", "", prep, sc_0)

    for path in sorted(paths.baseline.glob("*")):
        if path.is_file():
            man.add(f"Check | {path.stem}", "table", "", path, sc_a)

    for key, role in VARIANT_ROLES.items():
        folder = paths.variant_dir(key)
        summary_path = folder / RUN_FILES["summary"]
        if not summary_path.exists():
            continue
        config_path = folder / "run_config.json"
        config = _read_json(config_path)
        man.add(f"{key} | confounder features", "text",
                ", ".join(config["confounders"]), config_path, sc_a, role)
        man.add(f"{key} | dietary model inputs", "text",
                ", ".join(config["diet_model_inputs"]), config_path, sc_a, role)
        man.add(f"{key} | number of taxa entering feature selection", "number",
                config["n_taxa"], config_path, sc_a, role)
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            tag = f"{key} | {row['Input features']} | {row['Model']}"
            for metric in TABLE_METRICS:
                for stat in (("mean", "SD", "min", "max") if metric == "ROC-AUC"
                             else ("mean", "SD")):
                    man.add(f"{tag} | {metric} {stat}", "number",
                            float(row[f"{metric} {stat}"]), summary_path, sc_a, role)
        man.add(f"{key} | per-split metrics (all 15 configurations)", "table", "",
                folder / RUN_FILES["metrics"], sc_a, role)

        out = paths.output_dir(key)
        for path in sorted(out.glob("paired_bootstrap_ref_*.csv")):
            boot = pd.read_csv(path)
            ref = path.stem.replace("paired_bootstrap_ref_", "")
            for _, row in boot.iterrows():
                tag = (f"{key} | R2.3 reference {ref} minus "
                       f"{row['Comparator input features']} {row['Comparator model']}")
                man.add(f"{tag} | observed difference in mean AUC", "number",
                        float(row["Observed difference (ref - comp)"]), path, sc_b, role)
                man.add(f"{tag} | 95% interval", "text",
                        f"{row['95% CI low']:.4f} to {row['95% CI high']:.4f}",
                        path, sc_b, role)
                man.add(f"{tag} | interval crosses zero", "check",
                        bool(row["Interval crosses zero"]), path, sc_b, role)
        for path in sorted(out.glob("coef_rule_*_prioritised.csv")):
            table = pd.read_csv(path, dtype={"Feature ID": str})
            names = "; ".join(f"{n} ({int(c)}/5)" for n, c in
                              zip(table["Feature name"], table["Splits in top fraction"]))
            man.add(f"{key} | R2.5 {path.stem}: features in top 25% in >= 2 of 5 splits",
                    "text", names, path, sc_b, role)
        path = out / OUT_FILES["brier"]
        if path.exists():
            brier = pd.read_csv(path)
            for _, row in brier.iterrows():
                man.add(f"{key} | R2.8 Brier | {row['Input features']} | "
                        f"{row['Model']} | {row['Probability']}", "number",
                        float(row["Brier"]), path, sc_b, role)
            man.add(f"{key} | R2.8 reference Brier (constant 15/88)", "number",
                    float(brier["Reference Brier (constant 15/88)"].iloc[0]),
                    path, sc_b, role)
        if out.exists():
            for path in sorted(out.glob("*")):
                if path.is_file():
                    man.add(f"{key} | output file {path.name}",
                            "text" if path.suffix == ".txt" else "table", "", path, sc_b, role)

        perm_dir = paths.permutation_dir(key)
        path = perm_dir / PERM_FILES["result"]
        if path.exists():
            info = _read_json(path)
            for family in info["families"]:
                tag = f"{key} | R2.4 {family['family']}"
                man.add(f"{tag} | observed maximum mean ROC-AUC", "number",
                        family["observed_max"], path, sc_c, role)
                man.add(f"{tag} | best configuration", "text",
                        family["observed_best_configuration"], path, sc_c, role)
                man.add(f"{tag} | permutations with max >= observed", "number",
                        family["n_perm_max_ge_observed"], path, sc_c, role)
                man.add(f"{tag} | max-statistic permutation p-value", "number",
                        family["p_value"], path, sc_c, role)
            man.add(f"{key} | R2.4 number of permutations", "number", info["B"],
                    path, sc_c, role)
            man.add(f"{key} | R2.4 per-configuration p-values", "table", "",
                    perm_dir / PERM_FILES["per_config"], sc_c, role)
            man.add(f"{key} | R2.4 null distribution (one row per permutation)",
                    "table", "", perm_dir / PERM_FILES["null"], sc_c, role)

    for path in sorted(paths.outputs.glob("*")):
        if path.is_file():
            man.add(f"Shared output | {path.name}", "table", "", path, sc_b)
    for path in sorted(paths.main.glob("*.csv")):
        man.add(f"Main | {path.name}", "table", "", path, sc_a)
    for path in sorted(paths.figures.rglob("*")):
        if path.is_file() and path.suffix in (".pdf", ".txt", ".json"):
            kind = {".pdf": "figure", ".txt": "text", ".json": "table"}[path.suffix]
            script = sc_c if "permutation" in path.name else sc_b
            man.add(f"Figure | {path.relative_to(paths.figures).as_posix()}", kind, "",
                    path, script)
    for name in ("headline_results.csv", "comparison_239_vs_249_conf.csv"):
        if (paths.r2 / name).exists():
            man.add(f"Summary | {name}", "table", "", paths.r2 / name,
                    sc_b if name.startswith("headline") else "run_D_compare_239_vs_249.py")

    man.save()
    return man.frame()
