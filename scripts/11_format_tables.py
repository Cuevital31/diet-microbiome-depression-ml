"""Step 11 - manuscript and Supplement tables at the printed precision (results/tables/*.csv), built only from the
aggregate results of steps 1-9.

Table 1, Table 2, Supplementary Tables S1, S8-S19 (S2-S7 belong to the measured-SCFA and PICRUSt2 analyses,
which are not part of this repository).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.io_utils import read_csv, read_json, results_path, write_csv  # noqa: E402

R = lambda *p: read_csv(results_path(*p))  # noqa: E731
f2, f3, f4 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")
s3 = lambda v: "+0.000" if round(v, 3) == 0 else f"{v:+.3f}"  # noqa: E731  (no "-0.000" for rounding noise)
pm = lambda m, s: f"{m:.3f} ± {s:.3f}"  # noqa: E731
ci = lambda lo, hi: f"{lo:.3f}–{hi:.3f}"  # noqa: E731
orci = lambda o, lo, hi: f"{o:.2f} ({lo:.2f}–{hi:.2f})"  # noqa: E731


def out(frame, name):
    write_csv(frame, results_path("tables", name))


def main():
    perf = R("main", "performance_summary.csv")
    cis = R("bootstrap", "mean_auc_ci.csv").set_index("Configuration")
    lab = lambda r: f"{r['Input features']} | {r['Model']}"  # noqa: E731

    out(pd.DataFrame([{"Input features": r["Input features"], "Model": r["Model"],
                       "ROC-AUC": pm(r["ROC-AUC mean"], r["ROC-AUC SD"]),
                       "95% CI": ci(cis.loc[lab(r), "95% CI low"], cis.loc[lab(r), "95% CI high"]),
                       "BAL-ACC": pm(r["BAL-ACC mean"], r["BAL-ACC SD"]), "F1-Score": pm(r["F1 mean"], r["F1 SD"])}
                      for _, r in perf.iterrows()]), "table1.csv")

    s8 = R("main", "roc_auc_per_split.csv")
    out(s8.assign(**{c: s8[c].map(f3) for c in s8.columns if c not in ("Input features", "Model")}), "tableS8.csv")

    for tag, name in (("conf_diet_taxa_LR", "tableS9a.csv"), ("diet_taxa_LR", "tableS9b.csv"),
                      ("conf_diet_taxa_SVM", "tableS10a.csv"), ("diet_taxa_SVM", "tableS10b.csv")):
        b = R("bootstrap", f"paired_ref_{tag}.csv")
        out(pd.DataFrame({"Comparator": b["Comparator"], "Observed difference": b["Observed difference"].map(s3),
                          "Bootstrap median difference": b["Bootstrap median difference"].map(s3),
                          "95% interval": [f"[{s3(lo)}, {s3(hi)}]" for lo, hi in
                                           zip(b["95% interval low"], b["95% interval high"])]}), name)

    fam = R("permutation", "families.csv")
    out(pd.DataFrame({"Family (configurations)": fam["Family (configurations)"],
                      "Observed maximum": fam["Observed maximum"].map(f3),
                      "Best configuration": fam["Configuration at maximum"], "k": fam["k"],
                      "Null maximum: median": fam["Null maximum median"].map(f3),
                      "Null maximum: 95th percentile": fam["Null maximum 95th percentile"].map(f3),
                      "p": fam["p"].map(f3)}), "tableS11a.csv")
    per = R("permutation", "per_configuration.csv")
    out(pd.DataFrame({"Configuration": per["Configuration"], "Observed mean ROC-AUC": per["Observed mean ROC-AUC"].map(f3),
                      "Unadjusted p": per["Unadjusted p"].map(f3), "Adjusted p (15)": per["Adjusted p (15)"].map(f3),
                      "Adjusted p (12)": per["Adjusted p (12)"].map(lambda v: "--" if pd.isna(v) else f3(v))}),
        "tableS11b.csv")

    br = R("calibration", "brier_scores.csv")
    out(pd.DataFrame({"Input features": br["Input features"], "Model": br["Model"], "Probability": br["Probability"],
                      "Brier score": br["Brier score"].map(f4), "Skill score": br["Skill score"].map(s3),
                      "Lower than reference": br["Lower than reference"].map({True: "yes", False: "no"}),
                      "Reference": br["Reference Brier score (15/88)"].map(f4)}), "tableS12a.csv")
    bins = R("calibration", "bins_main_configuration.csv")
    out(bins.assign(**{"Mean predicted probability": bins["Mean predicted probability"].map(f3),
                       "Observed fraction": bins["Observed fraction"].map(f3)}), "tableS12b.csv")

    prio = R("prioritization", "prioritised_features.csv")
    cons = R("prioritization", "taxa_consensus_counts.csv").set_index("Feature ID")["Splits selected by consensus"]
    out(pd.DataFrame({"Feature": prio["Label"], "Group": prio["Feature group"],
                      "Selected by FS consensus": [str(cons[f]) if f in cons.index else "--" for f in prio["Feature ID"]],
                      "Times above median": prio["Splits retained"],
                      # coefficient shown only for the splits in which the feature met the median rule ("--" otherwise)
                      **{f"Split {s}": [f4(v) if kept else "--" for v, kept in
                                        zip(prio[f"Coefficient split {s}"], prio[f"Retained split {s}"])]
                         for s in range(1, 6)}}), "tableS1.csv")
    out(pd.DataFrame({"Feature": prio["Label"], "Group": prio["Feature group"], "Splits in model": prio["Splits in model"],
                      "Splits retained (k/5)": prio["Splits retained"], "Positive coefficient": prio["Positive coefficient"],
                      "Negative coefficient": prio["Negative coefficient"],
                      "Mean coefficient": prio["Mean coefficient"].map(s3)}), "tableS13.csv")

    fp = R("firth", "per_feature.csv")
    out(pd.DataFrame({"Feature": fp["Label"], "Group": fp["Group"], "Splits retained (k/5)": fp["Splits retained"].map(lambda k: f"{k}/5"),
                      "Unadjusted OR (95% CI)": [orci(*v) for v in fp[["unadjusted: OR", "unadjusted: OR low", "unadjusted: OR high"]].to_numpy()],
                      "Unadjusted q": fp["unadjusted: q"].map(f3),
                      "Adjusted OR (95% CI)": [orci(*v) for v in fp[["adjusted: OR", "adjusted: OR low", "adjusted: OR high"]].to_numpy()],
                      "Adjusted q": fp["adjusted: q"].map(f3)}), "table2.csv")
    out(pd.DataFrame({"Feature": fp["Label"], "Group": fp["Group"],
                      "Coefficient sign (+ / - splits)": [f"+{a} / -{b}" for a, b in zip(fp["Positive coefficient"], fp["Negative coefficient"])],
                      "Change in log-OR (absolute)": fp["Change in log-OR"].map(s3),
                      "Change in log-OR (%)": fp["Change in log-OR (%)"].map(lambda v: "n/a*" if pd.isna(v) else f"{v:+.0f}"),
                      "Direction agreement": fp["Direction agreement"]}), "tableS14.csv")
    s15 = {"Feature": fp["Label"]}
    for c in C.CONFOUNDERS:
        m = f"without {c}"
        s15[f"Without {c}"] = [orci(*v) for v in fp[[f"{m}: OR", f"{m}: OR low", f"{m}: OR high"]].to_numpy()]
    s15["Largest change (%)"] = [("n/a*" if pd.isna(v) else f"{v:+.0f}") + f" ({w})" for v, w in
                                 zip(fp["Largest leave-one-out change (%)"], fp["Confounder omitted (largest change)"])]
    out(pd.DataFrame(s15), "tableS15.csv")
    ms = R("firth", "model_summary.csv")
    out(ms.assign(**{"Smallest q": ms["Smallest q"].map(f3)}), "tableS15_smallest_q.csv")

    mc = R("prioritization", "mean_coefficients.csv")
    labels = dict(zip(prio["Feature ID"], prio["Label"]))
    groups = dict(zip(prio["Feature ID"], prio["Feature group"]))
    out(pd.DataFrame({"Feature": mc["Feature ID"].map(labels), "Group": mc["Feature ID"].map(groups),
                      "Splits in model (Diet + Taxa LR)": mc["Splits in model (Diet + Taxa LR)"].map(lambda k: f"{k}/5"),
                      "Mean coefficient (Diet + Taxa LR)": mc["Mean coefficient (Diet + Taxa LR)"].map(s3),
                      "Splits in model (Confounders + Diet + Taxa LR)": mc["Splits in model (Confounders + Diet + Taxa LR)"].map(lambda k: f"{k}/5"),
                      "Mean coefficient (Confounders + Diet + Taxa LR)": mc["Mean coefficient (Confounders + Diet + Taxa LR)"].map(s3)}),
        "tableS16.csv")

    rf = R("firth", "prioritised_in_reference_family.csv")
    out(pd.DataFrame({"Feature": rf["Label"], "Prevalence": rf["Prevalence"].map(lambda v: "--" if pd.isna(v) else f"{int(v)}/88"),
                      "Adjusted p": rf["Adjusted p"].map(f3), "Rank in family": rf["Rank in family"],
                      "q in family": rf["q in family"].map(lambda v: v if v == "not in family" else f3(float(v))),
                      "q among the 13": rf["q among prioritised"].map(f3)}), "tableS17.csv")
    rs = R("firth", "reference_summary.csv")
    info = read_json(results_path("firth", "fit_info.json"))
    out(rs.assign(**{"Smallest q": rs["Smallest q"].map(f3), "Fits converged": f"{info['converged']} of {info['fits']}",
                     "Fits (prioritised / reference)": f"{info['fits_prioritised']} / {info['fits_reference']}"}),
        "tableS17_summary.csv")

    rp = R("residualized", "performance.csv")
    out(pd.DataFrame({"Input features": rp["Input features"], "Model": rp["Model"], "Source": rp["Source"],
                      "ROC-AUC": [pm(a, b) for a, b in zip(rp["ROC-AUC mean"], rp["ROC-AUC SD"])],
                      "ROC-AUC 95% CI": [ci(a, b) for a, b in zip(rp["ROC-AUC 95% CI low"], rp["ROC-AUC 95% CI high"])],
                      "BAL-ACC": [pm(a, b) for a, b in zip(rp["BAL-ACC mean"], rp["BAL-ACC SD"])],
                      "SEN": [pm(a, b) for a, b in zip(rp["SEN mean"], rp["SEN SD"])],
                      "SPE": [pm(a, b) for a, b in zip(rp["SPE mean"], rp["SPE SD"])]}), "tableS18a.csv")
    rpf = R("residualized", "permutation_families.csv")
    out(pd.DataFrame({"Family": rpf["Family"], "Configurations": rpf["Configurations"],
                      "Observed maximum": rpf["Observed maximum"].map(f3), "Best configuration": rpf["Configuration at maximum"],
                      "k": rpf["k"], "p": rpf["p"].map(f3)}), "tableS18b.csv")

    pl = R("prediction_level", "adjustment.csv")
    out(pd.DataFrame({"Configuration": pl["Configuration"],
                      "OR per SD [95% interval]": [f"{o:.2f} [{lo:.2f}, {hi:.2f}]" for o, lo, hi in
                                                   zip(pl["or_per_sd"], pl["or_per_sd interval low"], pl["or_per_sd interval high"])],
                      "Delta D2": pl["delta_d2"].map(f3), "Replicates excluded": pl["Replicates excluded"],
                      "OR interval excludes 1": [(lo > 1) or (hi < 1) for lo, hi in
                                                 zip(pl["or_per_sd interval low"], pl["or_per_sd interval high"])]}),
        "tableS19.csv")
    print("tables written to", results_path("tables"))


if __name__ == "__main__":
    main()
