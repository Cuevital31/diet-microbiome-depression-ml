"""Firth penalised logistic regression of each prioritised dietary and microbial feature (R package logistf).

One feature at a time in all 88 participants; features z-scored (odds ratio per 1 SD), confounders on their
original scale. Models: unadjusted; adjusted for age, BMI, SQ_all and PA; four leave-one-confounder-out models.
Benjamini-Hochberg q-values across the prioritised features within each model. A reference family (all dietary
variables and the taxa with a raw count > 0 in at least 18 of the 88 participants) is fitted unadjusted and
adjusted to calibrate for selection.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import benjamini_hochberg

R_SCRIPT = C.REPO / "R" / "firth_logistf.R"


def model_covariates() -> dict:
    models = {"unadjusted": [], "adjusted": list(C.CONFOUNDERS)}
    for c in C.CONFOUNDERS:
        models[f"without {c}"] = [x for x in C.CONFOUNDERS if x != c]
    return models


def reference_taxa(cohort) -> list:
    min_prev = math.ceil(C.REFERENCE_MIN_PREVALENCE_FRACTION * len(cohort.ids))
    prevalence = (cohort.leaf_counts > 0).sum(axis=0)
    return [t for t, p in zip(cohort.taxa_ids, prevalence) if p >= min_prev], min_prev


def prevalence_of(cohort, taxon_id) -> int:
    return int((cohort.leaf_counts[:, cohort.taxa_ids.index(taxon_id)] > 0).sum())


def run(cohort, prioritised_ids, work_dir: Path) -> dict:
    """Write the data and specification files, run logistf and return all fits with q-values."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ref_taxa, min_prev = reference_taxa(cohort)
    diet = cohort.diet
    features = list(cohort.diet_names) + ref_taxa
    features += [f for f in prioritised_ids if f not in features]
    col = {f: f"f{j:03d}" for j, f in enumerate(features)}

    def zscore(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)

    data = {"y": cohort.y.astype(int)}
    for f in features:
        src = diet[f].to_numpy() if f in diet.columns else cohort.x_taxa[:, cohort.taxa_ids.index(f)]
        data[col[f]] = zscore(src)
    for c in C.CONFOUNDERS:
        data[c] = cohort.confounders[c].to_numpy(float)
    pd.DataFrame(data).to_csv(work_dir / "firth_data.csv", index=False)

    models = model_covariates()
    spec = [{"fit_id": f"main|{m}|{f}", "feature": col[f], "covariates": "+".join(cov)}
            for f in prioritised_ids for m, cov in models.items()]
    spec += [{"fit_id": f"ref|{m}|{f}", "feature": col[f], "covariates": "+".join(models[m])}
             for f in list(cohort.diet_names) + ref_taxa for m in ("unadjusted", "adjusted")]
    pd.DataFrame(spec).to_csv(work_dir / "firth_spec.csv", index=False)

    proc = subprocess.run([C.RSCRIPT, str(R_SCRIPT), str(work_dir / "firth_data.csv"), str(work_dir / "firth_spec.csv"),
                           str(work_dir / "firth_fits.csv")], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"logistf run failed:\n{proc.stdout}\n{proc.stderr}")
    fits = pd.read_csv(work_dir / "firth_fits.csv")
    assert len(fits) == len(spec)
    fits[["family", "model", "Feature ID"]] = fits["fit_id"].str.split("|", expand=True)
    fits["OR"], fits["OR low"], fits["OR high"] = np.exp(fits["beta"]), np.exp(fits["ci_low"]), np.exp(fits["ci_high"])
    fits["q"] = np.nan
    for _, idx in fits.groupby(["family", "model"]).groups.items():
        fits.loc[idx, "q"] = benjamini_hochberg(fits.loc[idx, "p"].to_numpy(float))
    line = proc.stdout.strip().splitlines()[-1]          # "logistf <v> from <library path> | R version ... | fits: n"
    (work_dir / "firth_software.txt").write_text(line + "\n", encoding="utf-8")
    software = line.split(" from ")[0] + " | " + line.split(" | ", 1)[1]   # without the local library path
    return {"fits": fits, "reference_taxa": ref_taxa, "min_prevalence": min_prev, "r_version": software}
