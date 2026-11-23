"""Coefficient-based prioritisation of candidate features in the main configuration.

In each split, the features of the fitted model are ranked jointly by the absolute value of their standardised
coefficients; a feature is retained in that split when its absolute coefficient is at or above the split median
(a feature absent from a split's model counts as not retained). Features retained in at least two of the five
splits are prioritised; the confounders are part of the ranking but serve as adjustment variables afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

GROUP_ORDER = {"Diet": 0, "Taxa": 1, "Confounder": 2}


def median_rule(coefficients: pd.DataFrame, feature_set: str, model: str = "LR") -> pd.DataFrame:
    sub = coefficients[(coefficients["Input features"] == feature_set) & (coefficients["Model"] == model)].copy()
    assert not sub.empty
    sub["split median"] = sub.groupby("Split")["Absolute coefficient"].transform("median")
    sub["retained"] = sub["Absolute coefficient"] >= sub["split median"]
    rows = []
    for (fid, group), b in sub.groupby(["Feature ID", "Feature group"], sort=False):
        row = {"Feature ID": fid, "Feature group": group, "Splits in model": int(b["Split"].nunique()),
               "Splits retained": int(b["retained"].sum()),
               "Positive coefficient": int((b["Coefficient"] > 0).sum()),
               "Negative coefficient": int((b["Coefficient"] < 0).sum()),
               "Mean coefficient": float(b["Coefficient"].mean()),
               "Mean absolute coefficient": float(b["Absolute coefficient"].mean())}
        for s in range(1, len(C.SEEDS) + 1):
            hit = b[b["Split"] == s]
            row[f"Coefficient split {s}"] = float(hit["Coefficient"].iloc[0]) if len(hit) else np.nan
            row[f"Retained split {s}"] = bool(hit["retained"].iloc[0]) if len(hit) else False
        rows.append(row)
    table = pd.DataFrame(rows)
    table["Prioritised"] = table["Splits retained"] >= C.MIN_SPLITS_RETAINED
    table["_g"] = table["Feature group"].map(GROUP_ORDER)
    return (table.sort_values(["_g", "Splits retained", "Mean absolute coefficient"], ascending=[True, False, False])
            .drop(columns="_g").reset_index(drop=True))


def mean_coefficients(coefficients: pd.DataFrame, feature_ids, feature_set: str, model: str = "LR") -> pd.DataFrame:
    sub = coefficients[(coefficients["Input features"] == feature_set) & (coefficients["Model"] == model)]
    rows = []
    for fid in feature_ids:
        f = sub[sub["Feature ID"] == fid]
        rows.append({"Feature ID": fid, "Splits in model": int(f["Split"].nunique()),
                     "Mean coefficient": float(f["Coefficient"].mean()) if len(f) else np.nan})
    return pd.DataFrame(rows)
