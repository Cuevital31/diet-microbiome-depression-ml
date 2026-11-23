"""Five repeated stratified holdout splits, taxa feature selection and the 15 model configurations.

Leakage rules (asserted where possible):
* taxa feature selection uses only the feature-selection subset of a split;
* the scaler, the classifiers and the Platt calibrator are fitted on the training subset only;
* the test subset is used only for prediction.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config as C


# ----------------------------------------------------------------------------- splits
def make_split(y, seed):
    """40% feature-selection subset, then the remaining 60% split 2:1 into training and test subsets
    (both stratified, same seed)."""
    y = np.asarray(y)
    dummy = np.zeros((len(y), 1))
    rest_idx, fs_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=C.FS_FRACTION,
                                                   random_state=seed).split(dummy, y))
    train_rel, test_rel = next(StratifiedShuffleSplit(n_splits=1, test_size=C.TEST_FRACTION_OF_REST,
                                                      random_state=seed).split(dummy[rest_idx], y[rest_idx]))
    fs_idx, train_idx, test_idx = fs_idx, rest_idx[train_rel], rest_idx[test_rel]
    assert not (set(fs_idx) & set(train_idx)) and not (set(fs_idx) & set(test_idx)) \
        and not (set(train_idx) & set(test_idx))
    assert len(fs_idx) + len(train_idx) + len(test_idx) == len(y) and len(test_idx) == C.N_TEST
    assert int(y[test_idx].sum()) == C.N_TEST_DEPRESSIVE
    return fs_idx, train_idx, test_idx


# ----------------------------------------------------------------------------- taxa selection
def select_taxa(x_fs, y_fs, seed, n_jobs=-1, allow_empty=False):
    """ANOVA F-test (p < 0.05), mutual information (>= 95th percentile) and random-forest importance
    (> 1 / number of taxa); a taxon is kept when selected by at least two of the three methods."""
    n_taxa = x_fs.shape[1]
    _, p_values = f_classif(x_fs, y_fs)
    p_values = np.nan_to_num(p_values, nan=1.0)
    anova = set(np.flatnonzero(p_values < C.ANOVA_P_THRESHOLD).tolist())
    mi_values = mutual_info_classif(x_fs, y_fs, random_state=seed)
    mi = set(np.flatnonzero(mi_values >= np.percentile(mi_values, C.MI_PERCENTILE)).tolist())
    forest = RandomForestClassifier(n_estimators=C.RF_TREES, class_weight="balanced", random_state=seed,
                                    n_jobs=n_jobs).fit(x_fs, y_fs)
    rf = set(np.flatnonzero(forest.feature_importances_ > 1.0 / n_taxa).tolist())
    methods = {"ANOVA": anova, "MI": mi, "RF": rf}
    votes = np.zeros(n_taxa, dtype=int)
    for chosen in methods.values():
        votes[list(chosen)] += 1
    consensus = np.flatnonzero(votes >= C.CONSENSUS_MIN_METHODS).tolist()
    if not consensus and not allow_empty:
        raise RuntimeError(f"no taxon passed the consensus rule (seed {seed})")
    return consensus, methods


# ----------------------------------------------------------------------------- models
def build_models(n_jobs=-1) -> dict:
    return {
        "LR": LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced",
                                 random_state=C.CLASSIFIER_RANDOM_STATE),
        "SVM": LinearSVC(C=1.0, max_iter=5_000, class_weight="balanced", random_state=C.CLASSIFIER_RANDOM_STATE),
        "BRF": BalancedRandomForestClassifier(n_estimators=C.RF_TREES, random_state=C.CLASSIFIER_RANDOM_STATE,
                                              n_jobs=n_jobs),
    }


def model_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def decision_scores(model, x):
    """Decision scores for LR and SVM, predicted probabilities for BRF."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return np.asarray(model.predict_proba(x), dtype=float)[:, 1]


def metrics(y_true, y_pred, scores) -> dict:
    return {"ROC-AUC": roc_auc_score(y_true, scores), "ACC": accuracy_score(y_true, y_pred),
            "BAL-ACC": balanced_accuracy_score(y_true, y_pred),
            "SEN": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "SPE": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "PRE": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0)}


def platt_probabilities(model_name, x_train, y_train, x_test, fitted_pipeline, test_scores):
    """Sigmoid (Platt) calibration fitted inside the training subset (3-fold stratified cross-validation,
    ensemble=False) and applied to the decision scores of the model fitted on the whole training subset."""
    inner = StratifiedKFold(n_splits=C.CALIB_CV_FOLDS, shuffle=True, random_state=C.CALIB_CV_SEED)
    calibrated = CalibratedClassifierCV(estimator=model_pipeline(build_models(n_jobs=1)[model_name]),
                                        method="sigmoid", cv=inner, ensemble=False).fit(x_train, y_train)
    prob = calibrated.predict_proba(x_test)[:, 1]
    fitted = calibrated.calibrated_classifiers_[0]
    assert np.allclose(fitted.estimator.named_steps["clf"].coef_, fitted_pipeline.named_steps["clf"].coef_, atol=1e-8)
    sigmoid = fitted.calibrators[0]
    a, b = float(sigmoid.a_), float(sigmoid.b_)
    assert np.allclose(prob, expit(-(a * np.asarray(test_scores) + b)), atol=1e-8)
    return prob, a, b


def fit_and_evaluate(matrix, train_idx, test_idx, y, model_name, n_jobs, detail, calibrate):
    """Fit one configuration; returns metrics, scores, predictions, probabilities and coefficients."""
    x_train, x_test = matrix[train_idx], matrix[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    out = {"prob_native": np.full(len(test_idx), np.nan), "prob_platt": np.full(len(test_idx), np.nan),
           "coef": None, "fit_warning": ""}
    if matrix.shape[1] == 0:                     # possible only under permuted labels: no taxon selected
        out.update(scores=np.zeros(len(test_idx)), y_pred=np.zeros(len(test_idx), dtype=int))
    else:
        pipe = model_pipeline(build_models(n_jobs)[model_name])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(x_train, y_train)
        out["fit_warning"] = "; ".join(sorted({type(w.message).__name__ for w in caught
                                               if issubclass(w.category, ConvergenceWarning)}))
        assert pipe.named_steps["scaler"].n_samples_seen_ == len(train_idx)
        out.update(scores=decision_scores(pipe, x_test), y_pred=pipe.predict(x_test))
        if detail:
            if hasattr(pipe, "predict_proba"):
                out["prob_native"] = pipe.predict_proba(x_test)[:, 1]
            if calibrate and model_name in ("LR", "SVM"):
                out["prob_platt"], out["platt_a"], out["platt_b"] = platt_probabilities(
                    model_name, x_train, y_train, x_test, pipe, out["scores"])
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "coef_"):
                out["coef"] = np.asarray(clf.coef_[0], dtype=float)
    out["metrics"] = metrics(y_test, out["y_pred"], out["scores"])
    return out


def feature_set_matrices(x_diet, diet_names, x_taxa_sel, taxa_ids, x_conf, conf_names):
    """The five feature sets (column order: diet, selected taxa, confounders)."""
    d, t, c = ["Diet"] * len(diet_names), ["Taxa"] * len(taxa_ids), ["Confounder"] * len(conf_names)
    return {
        "Confounders": (x_conf, list(conf_names), c),
        "Diet": (x_diet, list(diet_names), d),
        "Taxa": (x_taxa_sel, list(taxa_ids), t),
        "Diet + Taxa": (np.hstack([x_diet, x_taxa_sel]), list(diet_names) + list(taxa_ids), d + t),
        "Confounders + Diet + Taxa": (np.hstack([x_diet, x_taxa_sel, x_conf]),
                                      list(diet_names) + list(taxa_ids) + list(conf_names), d + t + c),
    }


def run_split(cohort, y, seed, split_no, *, n_jobs=-1, detail=True, calibrate=True, allow_empty=False):
    """One split, all 15 configurations."""
    y = np.asarray(y)
    fs_idx, train_idx, test_idx = make_split(y, seed)
    consensus, methods = select_taxa(cohort.x_taxa[fs_idx], y[fs_idx], seed, n_jobs=n_jobs, allow_empty=allow_empty)
    taxa_ids = [cohort.taxa_ids[i] for i in consensus]
    sets = feature_set_matrices(cohort.diet.to_numpy(float), cohort.diet_names, cohort.x_taxa[:, consensus],
                                taxa_ids, cohort.confounders.to_numpy(float), list(cohort.confounders.columns))
    out = {"aucs": {}, "metrics": [], "predictions": [], "coefficients": [], "selection": [], "membership": [],
           "calibration": [], "empty_consensus": len(consensus) == 0}
    for fs_name in C.FEATURE_SETS:
        matrix, feat_ids, groups = sets[fs_name]
        for model_name in C.MODELS:
            res = fit_and_evaluate(matrix, train_idx, test_idx, y, model_name, n_jobs, detail, calibrate)
            out["aucs"][(fs_name, model_name)] = float(res["metrics"]["ROC-AUC"])
            if not detail:
                continue
            _collect(out, res, split_no, seed, fs_name, model_name, feat_ids, groups, test_idx, cohort.ids, y,
                     len(consensus), len(cohort.diet_names), matrix.shape[1])
    if detail:
        _collect_selection(out, cohort, split_no, seed, consensus, methods, fs_idx, train_idx, test_idx, y)
    return out


def _collect(out, res, split_no, seed, fs_name, model_name, feat_ids, groups, test_idx, ids, y, n_taxa, n_diet,
             n_features):
    key = {"Split": split_no, "random_state": seed, "Input features": fs_name, "Model": model_name}
    out["metrics"].append({**key, "n selected taxa": n_taxa, "n diet features": n_diet,
                           "n total features": n_features, **{k: float(v) for k, v in res["metrics"].items()},
                           "fit_warning": res["fit_warning"]})
    if "platt_a" in res:
        out["calibration"].append({**key, "platt_a": res["platt_a"], "platt_b": res["platt_b"]})
    for pos, row in enumerate(test_idx):
        out["predictions"].append({**key, "test_position": pos, "participant_id": ids[row],
                                   "y_true": int(y[row]), "y_pred": int(res["y_pred"][pos]),
                                   "y_score": float(res["scores"][pos]), "prob_native": float(res["prob_native"][pos]),
                                   "prob_platt": float(res["prob_platt"][pos])})
    coef = res["coef"]
    if coef is not None:
        for j in range(len(coef)):
            out["coefficients"].append({**key, "Feature ID": feat_ids[j], "Feature group": groups[j],
                                        "Coefficient": float(coef[j]), "Absolute coefficient": float(abs(coef[j])),
                                        "n features in model": len(coef)})


def _collect_selection(out, cohort, split_no, seed, consensus, methods, fs_idx, train_idx, test_idx, y):
    chosen = set(consensus)
    for j, taxon_id in enumerate(cohort.taxa_ids):
        out["selection"].append({"Split": split_no, "random_state": seed, "rankID": taxon_id,
                                 "Taxon": cohort.taxon_names[taxon_id],
                                 "ANOVA": int(j in methods["ANOVA"]), "MI": int(j in methods["MI"]),
                                 "RF": int(j in methods["RF"]), "Consensus": int(j in chosen)})
    for role, idx in (("FS", fs_idx), ("train", train_idx), ("test", test_idx)):
        for row in idx:
            out["membership"].append({"Split": split_no, "random_state": seed, "role": role,
                                      "participant_id": cohort.ids[row], "y": int(y[row])})


def run_all_splits(cohort, n_jobs=-1, split_fn=None, **kwargs) -> dict:
    """The five repeated stratified splits; returns data frames."""
    split_fn = split_fn or run_split
    keys = ("metrics", "predictions", "coefficients", "selection", "membership", "calibration")
    collected = {k: [] for k in keys}
    for split_no, seed in enumerate(C.SEEDS, start=1):
        res = split_fn(cohort, cohort.y, seed, split_no, n_jobs=n_jobs, **kwargs)
        for k in keys:
            collected[k].extend(res.get(k, []))
    return {k: pd.DataFrame(v) for k, v in collected.items()}


def mean_aucs(cohort, y, split_fn=None, n_jobs=1, **kwargs) -> dict:
    """Whole pipeline for one label vector: mean ROC-AUC over the five splits per configuration."""
    split_fn = split_fn or run_split
    sums, n_empty = {}, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for split_no, seed in enumerate(C.SEEDS, start=1):
            res = split_fn(cohort, y, seed, split_no, n_jobs=n_jobs, detail=False, calibrate=False,
                           allow_empty=True, **kwargs)
            n_empty += int(res["empty_consensus"])
            for config, auc in res["aucs"].items():
                sums[config] = sums.get(config, 0.0) + auc
    return {"mean_auc": {c: v / len(C.SEEDS) for c, v in sums.items()}, "n_empty_consensus_splits": n_empty}


def permuted_labels(y, perm_index, base_seed=C.PERM_BASE_SEED):
    """Outcome labels shuffled across all participants (15 depressive kept); seed = (base_seed, index)."""
    permuted = np.random.default_rng([base_seed, perm_index]).permutation(np.asarray(y))
    assert permuted.sum() == C.N_DEPRESSIVE
    return permuted
