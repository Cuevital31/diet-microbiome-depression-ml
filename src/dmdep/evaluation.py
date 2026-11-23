"""Performance summaries, paired stratified bootstrap, permutation p-values, Brier scores, calibration bins,
Benjamini-Hochberg q-values and taxa-selection stability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-9          # mean ROC-AUC values are multiples of 1/225, so exact ties are possible


# ----------------------------------------------------------------------------- summaries
def summarize(metrics: pd.DataFrame, configs) -> pd.DataFrame:
    """Mean, sample SD (n = 5 splits), minimum and maximum of every metric per configuration."""
    rows = []
    for fs, model in configs:
        sub = metrics[(metrics["Input features"] == fs) & (metrics["Model"] == model)]
        assert len(sub) == len(C.SEEDS)
        row = {"Input features": fs, "Model": model}
        for m in C.METRICS:
            v = sub[m].to_numpy(float)
            row.update({f"{m} mean": v.mean(), f"{m} SD": v.std(ddof=1), f"{m} min": v.min(), f"{m} max": v.max()})
        rows.append(row)
    return pd.DataFrame(rows)


def per_split_table(metrics: pd.DataFrame, configs, metric="ROC-AUC") -> pd.DataFrame:
    wide = metrics.pivot_table(index=["Input features", "Model"], columns="Split", values=metric, aggfunc="first")
    wide = wide.reindex(list(configs))
    wide.columns = [f"Split {s}" for s in wide.columns]
    v = wide.to_numpy(float)
    wide["Mean"], wide["SD"] = v.mean(axis=1), v.std(axis=1, ddof=1)
    return wide.reset_index()


# ----------------------------------------------------------------------------- paired bootstrap
def bootstrap_draws(pred_by_label: dict, n_boot=C.BOOTSTRAP_B, seed=C.BOOTSTRAP_SEED):
    """Mean ROC-AUC over the five splits in each bootstrap replicate. In every replicate and split, the 3
    depressive and the 15 control test observations are resampled with replacement within class; the same
    resamples are used for all configurations; no model is refitted."""
    labels = list(pred_by_label)
    splits = sorted(next(iter(pred_by_label.values()))["Split"].unique())
    kernels, ids_ref = {}, {}
    for lab in labels:
        p = pred_by_label[lab]
        for s in splits:
            sub = p[p["Split"] == s].sort_values("test_position")
            ids = sub["participant_id"].to_numpy()
            if s in ids_ref:
                assert np.array_equal(ids, ids_ref[s]), "configurations do not share the test participants"
            else:
                ids_ref[s] = ids
            yt = sub["y_true"].to_numpy()
            pos, neg = sub["y_score"].to_numpy()[yt == 1], sub["y_score"].to_numpy()[yt == 0]
            assert len(pos) == C.N_TEST_DEPRESSIVE and len(neg) == C.N_TEST_CONTROL
            kernels[(lab, s)] = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(labels)))
    for b in range(n_boot):
        for s in splits:
            pi = rng.integers(0, C.N_TEST_DEPRESSIVE, C.N_TEST_DEPRESSIVE)
            ni = rng.integers(0, C.N_TEST_CONTROL, C.N_TEST_CONTROL)
            for c, lab in enumerate(labels):
                boot[b, c] += kernels[(lab, s)][np.ix_(pi, ni)].mean()
    boot /= len(splits)
    observed = {lab: np.array([kernels[(lab, s)].mean() for s in splits]) for lab in labels}
    return pd.DataFrame(boot, columns=labels), observed


def bootstrap_difference(draws: pd.DataFrame, observed: dict, ref: str, comp: str) -> dict:
    """Difference = mean ROC-AUC of the reference minus the comparator."""
    diff = np.round(draws[ref].to_numpy() - draws[comp].to_numpy(), 10)
    low, med, high = np.round(np.percentile(diff, [2.5, 50, 97.5]), 10)
    return {"Reference": ref, "Comparator": comp, "Observed difference": float((observed[ref] - observed[comp]).mean()),
            "Bootstrap median difference": float(med), "95% interval low": float(low),
            "95% interval high": float(high), "Interval includes zero": bool(low <= 0.0 <= high)}


def percentile_ci(values) -> tuple:
    lo, hi = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return float(lo), float(hi)


def predictions_by_config(predictions: pd.DataFrame, configs) -> dict:
    return {C.config_label(c): predictions[(predictions["Input features"] == c[0]) & (predictions["Model"] == c[1])]
            for c in configs}


# ----------------------------------------------------------------------------- permutation p-values
def family_result(null: pd.DataFrame, observed: dict, columns, n_perm: int) -> dict:
    """Max-statistic p-value for a family: p = (1 + #{permuted maxima >= observed maximum}) / (1 + B)."""
    nmax = null[columns].max(axis=1).to_numpy(float)
    omax = max(observed[c] for c in columns)
    best = [c for c in columns if observed[c] >= omax - EPS]
    k = int((nmax >= omax - EPS).sum())
    return {"Observed maximum": omax, "Configuration at maximum": "; ".join(best), "k": k,
            "Null maximum median": float(np.median(nmax)), "Null maximum 95th percentile": float(np.percentile(nmax, 95)),
            "p": (1 + k) / (1 + n_perm), "null_max": nmax}


def unadjusted_p(null_values, observed_value, n_perm) -> float:
    return (1 + int((np.asarray(null_values) >= observed_value - EPS).sum())) / (1 + n_perm)


# ----------------------------------------------------------------------------- calibration
def brier(y_true, prob) -> float:
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    return float(np.mean((prob - y_true) ** 2))


def brier_table(predictions: pd.DataFrame, feature_sets, models=C.MODELS) -> pd.DataFrame:
    """Brier scores of the pooled held-out predictions against a constant prediction of 15/88."""
    prevalence = C.N_DEPRESSIVE / C.N_PARTICIPANTS
    kinds = (("SVM", "prob_platt", "Platt (training set)"), ("LR", "prob_native", "native"),
             ("LR", "prob_platt", "Platt (training set)"), ("BRF", "prob_native", "native"))
    rows = []
    for fs in feature_sets:
        for model, column, kind in kinds:
            if model not in models:
                continue
            sub = predictions[(predictions["Input features"] == fs) & (predictions["Model"] == model)]
            if sub.empty:
                continue
            assert len(sub) == C.N_TEST * len(C.SEEDS) and sub[column].notna().all()
            b = brier(sub["y_true"], sub[column])
            ref = brier(sub["y_true"], np.full(len(sub), prevalence))
            rows.append({"Input features": fs, "Model": model, "Probability": kind, "Brier score": b,
                         "Reference Brier score (15/88)": ref, "Skill score": 1.0 - b / ref,
                         "Lower than reference": bool(b < ref)})
    return pd.DataFrame(rows)


def calibration_bins(y_true, prob, n_bins=C.CALIB_N_BINS) -> pd.DataFrame:
    """Equal-count bins of the sorted predicted probabilities."""
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    groups = np.array_split(np.argsort(prob, kind="stable"), n_bins)
    return pd.DataFrame([{"Bin": b, "Instances": len(idx), "Depressive instances": int(y_true[idx].sum()),
                          "Mean predicted probability": float(prob[idx].mean()),
                          "Observed fraction": float(y_true[idx].mean())}
                         for b, idx in enumerate(groups, start=1)])


# ----------------------------------------------------------------------------- misc
def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def selection_summary(selection: pd.DataFrame) -> dict:
    chosen = selection[selection["Consensus"] == 1]
    count = chosen.groupby("rankID").size()
    per_split = chosen.groupby("Split").size()
    return {"taxa_per_split": [int(per_split.get(s, 0)) for s in range(1, len(C.SEEDS) + 1)],
            "distinct_taxa_selected": int(len(count)),
            "selected_in_at_least_2_splits": int((count >= 2).sum()),
            "selected_in_at_least_3_splits": int((count >= 3).sum()),
            "selected_in_all_5_splits": int((count >= len(C.SEEDS)).sum())}


def consensus_frequency(selection: pd.DataFrame) -> pd.DataFrame:
    t = (selection.groupby(["rankID", "Taxon"], as_index=False)
         .agg(**{"Splits selected by consensus": ("Consensus", "sum")}))
    return (t[t["Splits selected by consensus"] > 0]
            .sort_values(["Splits selected by consensus", "Taxon"], ascending=[False, True]).reset_index(drop=True))
