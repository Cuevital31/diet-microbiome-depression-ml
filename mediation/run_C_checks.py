"""run_C_checks.py - Revision-R2g: checks 1-7 and 9 of ANALYSIS_PLAN_R2g.md (check 8, protected
folders, is run_0_checksums.py after). Writes 00_checks/check*.csv and checks_summary.csv.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

import mediation_core as mc  # noqa: E402

PLAN_SHA = mc.check_plan()
CHK = HERE / "00_checks"
CHK.mkdir(exist_ok=True)
ctx = mc.context()
X, M, C, zung = ctx["X"], ctx["M"], ctx["C"], ctx["zung"]
N = len(ctx["ids"])
rd = lambda p: pd.read_csv(p, dtype={"M_feature_id": str})  # noqa: E731
P = rd(HERE / "01_family_P" / "results_per_pair.csv")
S = rd(HERE / "02_family_S" / "results_per_pair.csv")
S1 = rd(HERE / "03_sensitivity" / "S1_inverse_normal_zung_M1_results_per_pair.csv")
S2 = rd(HERE / "03_sensitivity" / "S2_no_covariates_M0_results_per_pair.csv")
summary = []


def record(no, name, ok, detail):
    summary.append({"check": no, "description": name, "pass": bool(ok), "detail": detail})
    print(f"check {no}: {'PASS' if ok else 'FAIL'} | {name} | {detail}")


# 1. c paths vs Revision-R2d diet family (M1 and M0), betas and HC3 p-values
rows = []
for model, res in (("M1", P), ("M0", S2)):
    ref = pd.read_csv(mc.INPUTS / "Revision-R2d" / "results" / f"family_D_{model}.csv")
    ref = ref.set_index("feature_id")
    for d in mc.DIET:
        sub = res[res["X"] == d]
        assert sub["c"].nunique() == 1 and sub["c_p"].nunique() == 1
        rows.append({"model": model, "diet": d, "c_R2g": sub["c"].iloc[0], "c_R2d": ref.loc[d, "beta"],
                     "p_R2g": sub["c_p"].iloc[0], "p_R2d": ref.loc[d, "p"],
                     "se_R2g": sub["c_se_HC3"].iloc[0], "se_R2d": ref.loc[d, "se_HC3"]})
c1 = pd.DataFrame(rows)
c1["max_abs_diff"] = np.maximum.reduce([(c1["c_R2g"] - c1["c_R2d"]).abs(), (c1["p_R2g"] - c1["p_R2d"]).abs(),
                                        (c1["se_R2g"] - c1["se_R2d"]).abs()])
c1["within_1e-8"] = c1["max_abs_diff"] < 1e-8
c1.to_csv(CHK / "check1_c_paths_vs_R2d.csv", index=False)
record(1, "c paths reproduce Revision-R2d diet family (M1 and M0): beta, HC3 SE, HC3 p",
       c1["within_1e-8"].all(), f"max abs diff {c1['max_abs_diff'].max():.1e} (tolerance 1e-8)")

# 2. a, b, c' for 20 random Family S pairs vs statsmodels formula API and a manual numpy HC3
rng = np.random.default_rng(mc.SEED)
pick = sorted(rng.choice(len(S), size=20, replace=False))


def hc3_manual(y, D, j):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ beta
    XtXi = np.linalg.inv(D.T @ D)
    h = np.einsum("ij,jk,ik->i", D, XtXi, D)
    V = XtXi @ (D.T * (e / (1 - h)) ** 2) @ D @ XtXi
    se = np.sqrt(V[j, j])
    p = 2 * stats.t.sf(abs(beta[j] / se), len(y) - D.shape[1])
    return beta[j], se, p


rows = []
for k in pick:
    r = S.iloc[k]
    df = pd.DataFrame({"y": zung, "x": X[r["X"]].to_numpy(), "m": M[r["M_feature_id"]].to_numpy(),
                       **{c: C[c].to_numpy() for c in mc.CONF}})
    cv = " + ".join(mc.CONF)
    fa = smf.ols(f"m ~ x + {cv}", df).fit(cov_type="HC3", use_t=True)
    fb = smf.ols(f"y ~ x + m + {cv}", df).fit(cov_type="HC3", use_t=True)
    one = np.ones(N)
    Da = np.column_stack([one, df["x"], df[mc.CONF]])
    Db = np.column_stack([one, df["x"], df["m"], df[mc.CONF]])
    a_m = hc3_manual(df["m"].to_numpy(), Da, 1)
    b_m = hc3_manual(zung, Db, 2)
    cp_m = hc3_manual(zung, Db, 1)
    vals = {"a": (r["a"], fa.params["x"], a_m[0]), "a_p": (r["a_p"], fa.pvalues["x"], a_m[2]),
            "b": (r["b"], fb.params["m"], b_m[0]), "b_p": (r["b_p"], fb.pvalues["m"], b_m[2]),
            "c_prime": (r["c_prime"], fb.params["x"], cp_m[0]), "c_prime_p": (r["c_prime_p"], fb.pvalues["x"], cp_m[2]),
            "a_se_HC3": (r["a_se_HC3"], fa.bse["x"], a_m[1]), "b_se_HC3": (r["b_se_HC3"], fb.bse["m"], b_m[1])}
    row = {"X": r["X"], "M_feature_id": r["M_feature_id"]}
    for key, (saved, sfm, man) in vals.items():
        row[f"{key}_saved"], row[f"{key}_statsmodels_formula"], row[f"{key}_numpy_manual"] = saved, sfm, man
    row["max_abs_diff"] = max(max(abs(s - f), abs(s - m)) for s, f, m in vals.values())
    rows.append(row)
c2 = pd.DataFrame(rows)
c2.to_csv(CHK / "check2_random_pairs_independent_fits.csv", index=False)
record(2, "a, b, c' (beta, HC3 SE, p) of 20 random Family S pairs = statsmodels formula fit = manual numpy HC3",
       (c2["max_abs_diff"] < 1e-8).all(), f"max abs diff {c2['max_abs_diff'].max():.1e}")

# 3. bootstrap ab for 3 pairs reproduced by an independent loop (full-design lstsq) on the same draws
cases = [("P", "01_family_P", P, "results_per_pair_ab_bootstrap.npy", "bootstrap_draws_B5000_seed2026.npy",
          (mc.FIBER, "0.2.10.2.2.11.26")),
         ("P", "01_family_P", P, "results_per_pair_ab_bootstrap.npy", "bootstrap_draws_B5000_seed2026.npy",
          ("Chol_mg_per_1000kcal", "0.2.10.2.2.11.26")),
         ("S", "02_family_S", S, "results_per_pair_ab_bootstrap.npy", "bootstrap_draws_B2000_seed2026.npy",
          ("Chol_mg_per_1000kcal", "0.2.10.2.2.14.5"))]
rows = []
for fam, folder, res, abfile, drawfile, (d, t) in cases:
    ab_saved_all = np.load(HERE / folder / abfile)
    idx = np.load(HERE / folder / drawfile)
    assert np.array_equal(idx, mc.draws(idx.shape[0], N)), "saved draws differ from seed 2026"
    k = int(np.flatnonzero((res["X"] == d).to_numpy() & (res["M_feature_id"] == t).to_numpy())[0])
    x, m, Cm = X[d].to_numpy(), M[t].to_numpy(), C.to_numpy()
    ab = np.empty(idx.shape[0])
    for r in range(idx.shape[0]):
        ii = idx[r]
        Da = np.column_stack([np.ones(N), x[ii], Cm[ii]])
        Db = np.column_stack([np.ones(N), x[ii], m[ii], Cm[ii]])
        a = np.linalg.lstsq(Da, m[ii], rcond=None)[0][1]
        b = np.linalg.lstsq(Db, zung[ii], rcond=None)[0][2]
        ab[r] = a * b
    diff = float(np.nanmax(np.abs(ab - ab_saved_all[:, k])))
    lo, hi = np.percentile(ab, [2.5, 97.5])
    rows.append({"family": fam, "X": d, "M_feature_id": t, "B": idx.shape[0], "max_abs_diff_ab": diff,
                 "ci_low_independent": lo, "ci_low_saved": res.iloc[k]["ab_boot_ci_low"],
                 "ci_high_independent": hi, "ci_high_saved": res.iloc[k]["ab_boot_ci_high"]})
c3 = pd.DataFrame(rows)
c3["ci_diff"] = np.maximum((c3["ci_low_independent"] - c3["ci_low_saved"]).abs(),
                           (c3["ci_high_independent"] - c3["ci_high_saved"]).abs())
c3.to_csv(CHK / "check3_bootstrap_independent_loop.csv", index=False)
record(3, "bootstrap ab of 3 pairs reproduced by an independent full-design loop on the same draws",
       (c3["max_abs_diff_ab"] < 1e-9).all() and (c3["ci_diff"] < 1e-9).all(),
       f"max abs diff ab {c3['max_abs_diff_ab'].max():.1e}, CI {c3['ci_diff'].max():.1e}")

# 4. n = 88 and no missing values
miss = {"X": int(X.isna().sum().sum()), "M": int(M.isna().sum().sum()), "C": int(C.isna().sum().sum()),
        "zung": int(np.isnan(zung).sum())}
ns = {k: sorted(v["n"].unique().tolist()) for k, v in (("P", P), ("S", S), ("S1", S1), ("S2", S2))}
nan_cols = {k: int(v[["a", "b", "c", "c_prime", "ab", "ab_boot_ci_low", "ab_boot_ci_high", "boot_p"]].isna().sum().sum())
            for k, v in (("P", P), ("S", S), ("S1", S1), ("S2", S2))}
dep = int(ctx["y_dep"].sum())
ok4 = N == 88 and dep == 15 and all(v == [88] for v in ns.values()) and sum(miss.values()) == 0 \
    and sum(nan_cols.values()) == 0
pd.DataFrame([{"n_participants": N, "n_depressive": dep, "missing_inputs": str(miss), "n_per_family": str(ns),
               "missing_results": str(nan_cols), "pass": ok4}]).to_csv(CHK / "check4_n_and_missing.csv", index=False)
record(4, "n = 88 (15 depressive), no missing input or result values", ok4, f"n per family {ns}")

# 5. z-scoring
rows = []
for name, frame in (("X diet", X), ("M taxa (249)", M), ("C confounders", C)):
    rows.append({"block": name, "max_abs_mean": float(frame.mean().abs().max()),
                 "max_abs_sd_minus_1": float((frame.std(ddof=1) - 1).abs().max())})
c5 = pd.DataFrame(rows)
c5.to_csv(CHK / "check5_zscoring.csv", index=False)
record(5, "z-scores have mean 0 and SD 1 (ddof = 1)",
       (c5["max_abs_mean"] < 1e-12).all() and (c5["max_abs_sd_minus_1"] < 1e-12).all(),
       f"max |mean| {c5['max_abs_mean'].max():.1e}, max |SD-1| {c5['max_abs_sd_minus_1'].max():.1e}")

# 6. prevalence count and input hashes
cp = pd.read_csv(CHK / "check_inputs_copied.csv")
src = ctx["cohort"].sources
hash_rows = [{"input": k, "sha256_loaded": src[k]["sha256"], "sha256_main_run": ctx["run_cfg"]["inputs"][k]["sha256"]}
             for k in ("PART", "TAXA", "DIET", "CONFOUNDERS")]
hr = pd.DataFrame(hash_rows)
hr["identical"] = hr["sha256_loaded"] == hr["sha256_main_run"]
hr.to_csv(CHK / "check6_input_hashes_vs_main_run.csv", index=False)
prev_ok = len(ctx["taxa_s"]) == 188 and ctx["min_prev"] == 18
pd.DataFrame([{"min_prevalence": ctx["min_prev"], "n_taxa_family_S": len(ctx["taxa_s"]),
               "family_P_taxa_in_family_S": int(sum(t in ctx["taxa_s"] for t in mc.TAXA_P))}]).to_csv(
    CHK / "check6_prevalence.csv", index=False)
record(6, "188 taxa with prevalence >= 18/88; input copies = originals; loaded inputs = main-run hashes",
       prev_ok and cp["identical"].all() and hr["identical"].all(),
       f"{len(ctx['taxa_s'])} taxa; {len(cp)} copies identical; 4 main-run hashes identical")

# 7. BH with two implementations
rows = []
for fam, res in (("P", P), ("S", S), ("S1", S1), ("S2", S2)):
    for qcol, pcol in (("boot_q_BH", "boot_p"), ("a_q_BH", "a_p"), ("b_q_BH", "b_p"), ("sobel_q_BH", "sobel_p")):
        rows.append({"family": fam, "q": qcol, "max_abs_diff_vs_manual":
                     float(np.abs(res[qcol].to_numpy() - mc.bh_manual(res[pcol])).max())})
    cu = res.drop_duplicates("X")
    rows.append({"family": fam, "q": "c_q_BH_7_diet", "max_abs_diff_vs_manual":
                 float(np.abs(cu["c_q_BH_7_diet"].to_numpy() - mc.bh_manual(cu["c_p"])).max())})
c7 = pd.DataFrame(rows)
c7.to_csv(CHK / "check7_bh_two_implementations.csv", index=False)
record(7, "BH q-values: statsmodels = manual implementation", (c7["max_abs_diff_vs_manual"] < 1e-12).all(),
       f"{len(c7)} q columns, max diff {c7['max_abs_diff_vs_manual'].max():.1e}")

# 9. plan hash
record(9, "plan hash unchanged", mc.check_plan() == PLAN_SHA, PLAN_SHA[:16] + "...")

pd.DataFrame(summary).to_csv(CHK / "checks_summary_1_7_9.csv", index=False)
assert all(r["pass"] for r in summary), "A CHECK FAILED"
print("run_C_checks.py finished: all checks passed")
