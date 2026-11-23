"""Step 6 - confounder-adjusted Firth logistic regression of the prioritised dietary and microbial features
(R package logistf, called through Rscript; see R/firth_logistf.R).

Outputs:
  results/firth/per_feature.csv          unadjusted, adjusted and leave-one-confounder-out estimates, q-values,
                                         changes with adjustment and agreement with the LR coefficient signs
  results/firth/model_summary.csv        smallest q-value per model
  results/firth/reference_family.csv     reference family: p, q and rank of every feature (unadjusted, adjusted)
  results/firth/reference_summary.csv    reference family summary
  results/firth/prioritised_in_reference_family.csv
The regression input data are participant-level and are written to the work folder only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep import firth  # noqa: E402
from dmdep.data import load_cohort, taxon_label  # noqa: E402
from dmdep.io_utils import read_csv, results_path, work_path, write_csv, write_json  # noqa: E402


def main():
    cohort = load_cohort()
    prio = read_csv(results_path("prioritization", "prioritised_features.csv"))
    prio = prio[prio["Feature group"] != "Confounder"].reset_index(drop=True)
    ids = list(prio["Feature ID"])
    res = firth.run(cohort, ids, work_path("firth"))
    fits = res["fits"]
    models = firth.model_covariates()
    main = fits[fits["family"] == "main"].set_index(["model", "Feature ID"])
    label = lambda f: taxon_label(cohort, f) if f in cohort.taxon_names else f  # noqa: E731

    per = prio[["Feature ID", "Feature group", "Splits retained", "Positive coefficient", "Negative coefficient"]].copy()
    per.insert(1, "Label", per["Feature ID"].map(label))
    per["Group"] = np.where(per["Feature group"] == "Diet", "Diet", "Taxon")
    for m in models:
        for col in ("beta", "OR", "OR low", "OR high", "p", "q", "converged"):
            per[f"{m}: {col}"] = [main.loc[(m, f), col] for f in ids]
    b_un, b_adj = per["unadjusted: beta"], per["adjusted: beta"]
    per["Change in log-OR"] = b_adj - b_un
    per["Change in log-OR (%)"] = np.where(b_un.abs() >= C.PCT_CHANGE_MIN_BETA, (b_adj - b_un) / b_un * 100, np.nan)
    loo = [f"without {c}" for c in C.CONFOUNDERS]
    changes = np.column_stack([per[f"{m}: beta"] - b_adj for m in loo])
    which = np.abs(changes).argmax(axis=1)
    per["Largest leave-one-out change (%)"] = [
        (changes[i, w] / b_adj[i] * 100) if abs(b_adj[i]) >= C.PCT_CHANGE_MIN_BETA else np.nan
        for i, w in enumerate(which)]
    per["Confounder omitted (largest change)"] = [C.CONFOUNDERS[w] for w in which]
    pos, neg = per["Positive coefficient"], per["Negative coefficient"]
    majority = np.where(pos > neg, "+", np.where(neg > pos, "-", "tie"))
    sign = np.where(b_adj > 0, "+", np.where(b_adj < 0, "-", "0"))
    per["Direction agreement"] = np.where(majority == "tie", "tie", np.where(majority == sign, "Yes", "No"))
    per = per.sort_values(["Group", "Splits retained"], ascending=[True, False], kind="stable").reset_index(drop=True)
    write_csv(per, results_path("firth", "per_feature.csv"))
    write_csv(pd.DataFrame([{"Model": m, "Smallest q": float(per[f"{m}: q"].min()),
                             "Features with p < 0.05": int((per[f"{m}: p"] < 0.05).sum()),
                             "Features with q < 0.10": int((per[f"{m}: q"] < 0.10).sum())} for m in models]),
              results_path("firth", "model_summary.csv"))

    ref = fits[fits["family"] == "ref"].copy()
    ref["Group"] = np.where(ref["Feature ID"].isin(cohort.diet_names), "Diet", "Taxon")
    ref["Rank of p"] = ref.groupby("model")["p"].rank(method="min").astype(int)
    ref["Label"] = ref["Feature ID"].map(label)
    write_csv(ref[["model", "Feature ID", "Label", "Group", "OR", "OR low", "OR high", "p", "q", "Rank of p", "converged"]],
              results_path("firth", "reference_family.csv"))
    summ = []
    for m in ("unadjusted", "adjusted"):
        t = ref[ref["model"] == m]
        summ.append({"Model": m, "Features": len(t), "Dietary variables": int((t["Group"] == "Diet").sum()),
                     "Taxa": int((t["Group"] == "Taxon").sum()), "Features with p < 0.05": int((t["p"] < 0.05).sum()),
                     "Features with q < 0.10": int((t["q"] < 0.10).sum()), "Smallest q": float(t["q"].min())})
    write_csv(pd.DataFrame(summ), results_path("firth", "reference_summary.csv"))
    rows = []
    adj_ref = ref[ref["model"] == "adjusted"].set_index("Feature ID")
    for f in per["Feature ID"]:
        in_family = f in adj_ref.index
        rows.append({"Feature ID": f, "Label": label(f),
                     "Prevalence": firth.prevalence_of(cohort, f) if f in cohort.taxon_names else np.nan,
                     "Adjusted p": main.loc[("adjusted", f), "p"],
                     "Rank in family": int(adj_ref.loc[f, "Rank of p"]) if in_family else "not in family",
                     "q in family": float(adj_ref.loc[f, "q"]) if in_family else "not in family",
                     "q among prioritised": main.loc[("adjusted", f), "q"]})
    write_csv(pd.DataFrame(rows), results_path("firth", "prioritised_in_reference_family.csv"))
    write_json(results_path("firth", "fit_info.json"),
               {"fits": int(len(fits)), "converged": int(fits["converged"].astype(bool).sum()),
                "fits_prioritised": int((fits["family"] == "main").sum()), "fits_reference": int((fits["family"] == "ref").sum()),
                "reference_min_prevalence": res["min_prevalence"], "software": res["r_version"]})
    print(per[["Label", "Splits retained", "unadjusted: OR", "unadjusted: q", "adjusted: OR", "adjusted: p",
               "adjusted: q", "Direction agreement"]].round(3).to_string())
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
