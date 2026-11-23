"""Sensitivity analysis: confounder-residualized prediction.

Within each split, every dietary variable and every CLR-transformed taxon is regressed (ordinary least squares)
on age, BMI, SQ_all and PA using only the participants of the feature-selection and training subsets; the fitted
coefficients are applied to all participants, test participants included, without refitting. Taxa selection is
rerun on the residualized taxa of the feature-selection subset; LR and linear SVM are trained and evaluated as in
the main analysis.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .pipeline import _collect, _collect_selection, fit_and_evaluate, make_split, select_taxa


def residualize(values, conf, fit_idx):
    design_fit = np.column_stack([np.ones(len(fit_idx)), conf[fit_idx]])
    beta, *_ = np.linalg.lstsq(design_fit, values[fit_idx], rcond=None)
    return values - np.column_stack([np.ones(len(conf)), conf]) @ beta


def run_split_residualized(cohort, y, seed, split_no, *, n_jobs=-1, detail=True, calibrate=True,
                           allow_empty=False):
    y = np.asarray(y)
    fs_idx, train_idx, test_idx = make_split(y, seed)
    dev_idx = np.sort(np.concatenate([fs_idx, train_idx]))
    assert not set(test_idx) & set(dev_idx)
    conf = cohort.confounders.to_numpy(float)
    conf_names = list(cohort.confounders.columns)
    diet_names = cohort.diet_names
    x_diet = residualize(cohort.diet.to_numpy(float), conf, dev_idx)
    x_taxa = residualize(cohort.x_taxa, conf, dev_idx)
    consensus, methods = select_taxa(x_taxa[fs_idx], y[fs_idx], seed, n_jobs=n_jobs, allow_empty=allow_empty)
    taxa_ids = [cohort.taxa_ids[i] for i in consensus]
    x_sel = x_taxa[:, consensus]
    d, t, c = ["Diet"] * len(diet_names), ["Taxa"] * len(taxa_ids), ["Confounder"] * len(conf_names)
    sets = {
        "Diet_adj": (x_diet, diet_names, d),
        "Taxa_adj": (x_sel, taxa_ids, t),
        "Diet_adj + Taxa_adj": (np.hstack([x_diet, x_sel]), diet_names + taxa_ids, d + t),
        "Confounders + Diet_adj + Taxa_adj": (np.hstack([x_diet, x_sel, conf]), diet_names + taxa_ids + conf_names,
                                              d + t + c),
    }
    out = {"aucs": {}, "metrics": [], "predictions": [], "coefficients": [], "selection": [], "membership": [],
           "calibration": [], "empty_consensus": len(consensus) == 0}
    for fs_name in C.RESID_SETS:
        matrix, feat_ids, groups = sets[fs_name]
        for model_name in C.RESID_MODELS:
            res = fit_and_evaluate(matrix, train_idx, test_idx, y, model_name, n_jobs, detail, calibrate)
            out["aucs"][(fs_name, model_name)] = float(res["metrics"]["ROC-AUC"])
            if detail:
                _collect(out, res, split_no, seed, fs_name, model_name, feat_ids, groups, test_idx, cohort.ids, y,
                         len(consensus), len(diet_names), matrix.shape[1])
    if detail:
        _collect_selection(out, cohort, split_no, seed, consensus, methods, fs_idx, train_idx, test_idx, y)
    return out
