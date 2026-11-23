"""Prediction-level confounder adjustment (descriptive; Dinga et al., 2020).

For each configuration, the 90 pooled held-out predictions (from 63 distinct participants) are related to the
outcome by unpenalised logistic regression with the z-scored confounders alone and with the confounders plus the
logit of the predicted probability (Platt-calibrated for LR and SVM, native for BRF). Reported: odds ratio per SD
of the logit score and the change in explained deviance, with 95% intervals from a participant-level cluster
bootstrap (resampling within outcome class); replicates that do not converge or show separation are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import config as C


def irls(X, y, max_iter=100, tol=1e-10):
    """Unpenalised logistic regression by Newton-IRLS. Returns (beta, deviance, ok)."""
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = expit(X @ b)
        w = p * (1 - p)
        if np.any(w < 1e-12):
            return b, np.nan, False
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < tol:
            p = np.clip(expit(X @ b), 1e-300, 1 - 1e-16)
            dev = -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            return b, float(dev), bool(np.all(np.abs(b) < 30))
    return b, np.nan, False


def _fit(y, Z, s):
    one = np.ones(len(y))
    _, dev_null, ok0 = irls(one[:, None], y)
    _, dev_c, ok1 = irls(np.column_stack([one, Z]), y)
    b, dev_cs, ok2 = irls(np.column_stack([one, Z, s]), y)
    if not (ok0 and ok1 and ok2):
        return None
    sd = s.std(ddof=1)
    return {"coef": float(b[-1]), "or_per_sd": float(np.exp(b[-1] * sd)), "delta_d2": float((dev_c - dev_cs) / dev_null)}


def adjust(pooled: dict, confounders: pd.DataFrame, n_boot=C.PL_BOOTSTRAP_B, seed=C.PL_BOOTSTRAP_SEED) -> pd.DataFrame:
    """pooled: label -> (predictions of one configuration, probability column)."""
    zc = (confounders - confounders.mean()) / confounders.std(ddof=1)
    first = next(iter(pooled.values()))[0]
    ref_ids = first["participant_id"].to_numpy()
    for sub, _ in pooled.values():
        assert np.array_equal(sub["participant_id"].to_numpy(), ref_ids)
    y = first["y_true"].to_numpy(float)
    Z = zc.loc[ref_ids].to_numpy(float)
    uniq = pd.Series(y, index=ref_ids).groupby(level=0).first()
    dep, ctl = uniq[uniq == 1].index.to_numpy(), uniq[uniq == 0].index.to_numpy()
    rows_of = {pid: np.flatnonzero(ref_ids == pid) for pid in uniq.index}
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_boot):
        chosen = np.concatenate([rng.choice(dep, len(dep), replace=True), rng.choice(ctl, len(ctl), replace=True)])
        resamples.append(np.concatenate([rows_of[p] for p in chosen]))
    out = []
    for label, (sub, column) in pooled.items():
        s = logit(np.clip(sub[column].to_numpy(float), C.PL_CLIP, 1 - C.PL_CLIP))
        obs = _fit(y, Z, s)
        boots = [_fit(y[r], Z[r], s[r]) for r in resamples]
        arr = pd.DataFrame([b for b in boots if b is not None])
        row = {"Configuration": label, "Probability": column, "n rows": len(sub),
               "n distinct participants": len(uniq), **obs}
        for k in ("or_per_sd", "delta_d2"):
            row[f"{k} interval low"], row[f"{k} interval high"] = np.percentile(arr[k], [2.5, 97.5])
        row["Replicates excluded"] = sum(b is None for b in boots)
        out.append(row)
    return pd.DataFrame(out)
