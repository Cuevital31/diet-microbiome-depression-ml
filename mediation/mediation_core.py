"""mediation_core.py - shared code of Revision-R2g (conventional mediation analysis).

* plan-hash guard
* data loading through an unchanged, hash-checked copy of Revision-R2c-conf/pipeline_core.py,
  reading only the input copies in Revision-R2g/inputs (original folder layout mirrored)
* OLS with HC3 standard errors (statsmodels), per-pair a / c / b / c' paths, Sobel z
* vectorised participant-level bootstrap of ab (Frisch-Waugh-Lovell for the b path)
* Benjamini-Hochberg (two implementations), formatting helpers
Follows ANALYSIS_PLAN_R2g.md.
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # Clean-Codes (originals: read-only)
INPUTS = HERE / "inputs"                             # mirrored copies (read by the scripts)
CHK = HERE / "00_checks"
MAIN_RUN = "MAIN_conf_Age_BMI_SQ_PA_249taxa"
SEED = 2026
B_P = 5000
B_S = 2000
ALPHA = 0.05
Q_THRESHOLD = 0.10
DEGENERATE_REL = 1e-12

DIET = ["Prot_pctE", "SatFat_pctE", "MUF_pctE", "Poly_InsFat_pctE",
        "Chol_mg_per_1000kcal", "Carb_pctE", "Fiber_g_per_1000kcal"]
CONF = ["Age", "BMI", "SQ_all", "PA"]
TAXA_P = ["0.2.10.2.2.11.26", "0.2.10.2.2.14.28", "0.2.10.2.2.2.1",
          "0.2.10.2.2.11.24", "0.2.17.1.6.5.1", "0.2.10.3.1.1.6"]
TAXA_P_NAMES = ["Lachnospiraceae_ND3007_group", "Ruminiclostridium_9", "Catabacter",
                "Lachnospiraceae_FCS020_group", "Rhizobiales_unclassified", "Dielma"]
FIBER = "Fiber_g_per_1000kcal"
NOTE_P = "exploratory; taxa were prioritised using the outcome; q-values are not confirmatory"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_plan() -> str:
    saved = (HERE / "plan_hash.txt").read_text(encoding="utf-8").split()[0]
    now = sha256(HERE / "ANALYSIS_PLAN_R2g.md")
    assert now == saved, "ANALYSIS_PLAN_R2g.md changed after its hash was saved: refusing to run"
    return now


def _load_core():
    copy = HERE / "pipeline_core_from_R2c-conf.py"
    orig = ROOT / "Revision-R2c-conf" / "pipeline_core.py"
    assert sha256(copy) == sha256(orig), "pipeline_core copy differs from Revision-R2c-conf"
    name = "pipeline_core_r2c_conf"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, copy)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                       # needed by dataclasses
    spec.loader.exec_module(module)
    return module


pc = _load_core()
assert pc.FOLDER == "Revision-R2c-conf"


def zscore(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=1)


def inverse_normal(v) -> np.ndarray:
    """Rank-based inverse-normal transform (Blom), average ranks for ties."""
    r = stats.rankdata(v, method="average")
    return stats.norm.ppf((r - 3.0 / 8.0) / (len(r) + 0.25))


_CTX = {}


def context() -> dict:
    """Cohort (CONF4: new diet file, 249 taxa), diet design, confounders, prevalence, labels.
    All read from the copies in Revision-R2g/inputs."""
    if _CTX:
        return _CTX
    paths = pc.Paths(INPUTS)                         # no make_paths: it would create folders
    cohorts = pc.load_all_cohorts(paths)
    cohort = pc.cohort_for(cohorts, "CONF4")
    run_cfg = json.loads((INPUTS / "Revision-R2c-conf" / "01_main" / MAIN_RUN / "run_config.json")
                         .read_text(encoding="utf-8"))
    design = pc.build_diet_design(cohort, "V1", run_cfg["energy_denom"])
    assert list(design.feature_names) == run_cfg["diet_model_inputs"]
    assert sorted(design.feature_names) == sorted(DIET)
    conf = pc.confounder_frame(cohort, pc.CONFOUNDER_SETS["CONF4"])
    assert list(conf.columns) == run_cfg["confounders"] == CONF
    assert cohort.n_taxa == 249 and cohort.taxa_label == "all"
    ids = list(cohort.ids)
    # input hashes recorded by pipeline_core must equal those of the main run
    for key in ("PART", "TAXA", "DIET", "CONFOUNDERS"):
        assert cohort.sources[key]["sha256"] == run_cfg["inputs"][key]["sha256"], key

    v5 = pd.read_excel(INPUTS / "Preprocessed Files" / "taxonomy_completo_paper_v5.xlsx",
                       dtype={"rankID": str})
    cnt_cols = [c for c in v5.columns if str(c).startswith(pc.SAMPLE_PREFIX)]
    assert [c.replace(pc.SAMPLE_PREFIX, pc.PARTICIPANT_PREFIX, 1) for c in cnt_cols] == ids
    v5i = v5.set_index("rankID")
    prevalence = (v5i.loc[list(cohort.taxa_ids), cnt_cols].astype(float) > 0).sum(axis=1)
    name_of = v5i["taxon"].astype(str).to_dict()
    min_prev = math.ceil(0.20 * len(ids))
    taxa_s = [t for t in cohort.taxa_ids if prevalence[t] >= min_prev]

    diet_raw = design.table.loc[ids, DIET].astype(float)
    taxa_raw = pd.DataFrame(cohort.x_taxa, index=ids, columns=list(cohort.taxa_ids))
    assert [cohort.taxon_names[t] for t in TAXA_P] == TAXA_P_NAMES

    def label(fid: str) -> str:
        if fid in DIET:
            return fid
        return f"{cohort.taxon_names[fid]} ({name_of.get(fid.rsplit('.', 1)[0], '')}); {fid}"

    X = pd.DataFrame({d: zscore(diet_raw[d]) for d in DIET}, index=ids)
    M = pd.DataFrame({t: zscore(taxa_raw[t]) for t in taxa_raw.columns}, index=ids)
    C = pd.DataFrame({c: zscore(conf[c]) for c in CONF}, index=ids)
    zung = np.asarray(cohort.zung, dtype=float)
    _CTX.update(paths=paths, cohort=cohort, design=design, conf=conf, run_cfg=run_cfg, ids=ids,
                diet_raw=diet_raw, taxa_raw=taxa_raw, X=X, M=M, C=C, zung=zung,
                y_dep=np.asarray(cohort.y, dtype=int), prevalence=prevalence, min_prev=min_prev,
                taxa_s=taxa_s, label=label)
    return _CTX


# =============================================================================
# Point estimates
# =============================================================================

def ols_hc3(y: np.ndarray, design: pd.DataFrame) -> dict:
    """OLS with HC3 SEs (t reference, residual df), as in Revision-R2d."""
    rob = sm.OLS(y, design).fit(cov_type="HC3", use_t=True)
    return {"params": rob.params, "bse": rob.bse, "pvalues": rob.pvalues, "df_resid": rob.df_resid,
            "nobs": int(rob.nobs)}


def _design(cols: dict, index) -> pd.DataFrame:
    return sm.add_constant(pd.DataFrame(cols, index=index), has_constant="add")


def pair_paths(x: np.ndarray, m: np.ndarray, y: np.ndarray, C: pd.DataFrame) -> dict:
    """a: M ~ X + C; c: Y ~ X + C; b, c': Y ~ X + M + C (all OLS, HC3)."""
    idx = C.index
    covs = {c: C[c].to_numpy() for c in C.columns}
    fa = ols_hc3(m, _design({"X": x, **covs}, idx))
    fc = ols_hc3(y, _design({"X": x, **covs}, idx))
    fb = ols_hc3(y, _design({"X": x, "M": m, **covs}, idx))
    a, sa = float(fa["params"]["X"]), float(fa["bse"]["X"])
    b, sb = float(fb["params"]["M"]), float(fb["bse"]["M"])
    sobel_z = a * b / math.sqrt(b * b * sa * sa + a * a * sb * sb)
    return {"n": fa["nobs"],
            "a": a, "a_se_HC3": sa, "a_p": float(fa["pvalues"]["X"]),
            "c": float(fc["params"]["X"]), "c_se_HC3": float(fc["bse"]["X"]),
            "c_p": float(fc["pvalues"]["X"]),
            "b": b, "b_se_HC3": sb, "b_p": float(fb["pvalues"]["M"]),
            "c_prime": float(fb["params"]["X"]), "c_prime_se_HC3": float(fb["bse"]["X"]),
            "c_prime_p": float(fb["pvalues"]["X"]),
            "ab": a * b, "sobel_z": sobel_z, "sobel_p": float(2 * stats.norm.sf(abs(sobel_z)))}


# =============================================================================
# Bootstrap
# =============================================================================

def draws(B: int, n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def _a_b(xcol: np.ndarray, Mmat: np.ndarray, y: np.ndarray, Cmat: np.ndarray):
    """a for every mediator column and b (FWL) for every mediator; also the residual SS of M."""
    D = np.column_stack([np.ones(len(xcol)), xcol, Cmat]) if Cmat.size else \
        np.column_stack([np.ones(len(xcol)), xcol])
    rank = np.linalg.matrix_rank(D)
    RHS = np.column_stack([Mmat, y])
    coef, *_ = np.linalg.lstsq(D, RHS, rcond=None)
    resid = RHS - D @ coef
    rM, rY = resid[:, :-1], resid[:, -1]
    ssM = (rM * rM).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (rM * rY[:, None]).sum(axis=0) / ssM
    return coef[1, :-1], b, ssM, rank == D.shape[1]


def bootstrap_ab(X: pd.DataFrame, M: pd.DataFrame, y: np.ndarray, C: pd.DataFrame,
                 pairs: list, idx_draws: np.ndarray) -> dict:
    """ab for every pair in every replicate; NaN for degenerate replicates."""
    diet_vars = list(dict.fromkeys(p[0] for p in pairs))
    Cmat = C.to_numpy() if C is not None and C.shape[1] else np.empty((len(y), 0))
    B = idx_draws.shape[0]
    out = np.full((B, len(pairs)), np.nan)
    pos = {p: k for k, p in enumerate(pairs)}
    for d in diet_vars:
        meds = [p[1] for p in pairs if p[0] == d]
        cols = [pos[(d, m)] for m in meds]
        Mmat = M[meds].to_numpy()
        xcol = X[d].to_numpy()
        _, _, ss_full, _ = _a_b(xcol, Mmat, y, Cmat)
        for r in range(B):
            ii = idx_draws[r]
            a, b, ssM, full_rank = _a_b(xcol[ii], Mmat[ii], y[ii], Cmat[ii] if Cmat.size else Cmat)
            ok = full_rank & (ssM > DEGENERATE_REL * ss_full)
            out[r, cols] = np.where(ok, a * b, np.nan)
    return {"ab": out, "pairs": pairs}


def bootstrap_summary(ab_draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(ab_draws.shape[1]):
        v = ab_draws[:, k]
        valid = v[~np.isnan(v)]
        Bv = len(valid)
        le, ge = float((valid <= 0).mean()), float((valid >= 0).mean())
        p = min(1.0, max(2 * min(le, ge), 1.0 / (Bv + 1)))
        lo, hi = np.percentile(valid, [2.5, 97.5])
        rows.append({"ab_boot_ci_low": float(lo), "ab_boot_ci_high": float(hi),
                     "ab_boot_mean": float(valid.mean()), "boot_p": p,
                     "B_valid": Bv, "n_degenerate_replicates": int(len(v) - Bv),
                     "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# =============================================================================
# Multiple testing, families
# =============================================================================

def bh(p) -> np.ndarray:
    return multipletests(np.asarray(p, dtype=float), method="fdr_bh")[1]


def bh_manual(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def run_family(pairs: list, y: np.ndarray, C, idx_draws: np.ndarray, family: str,
               outcome: str, covariates: str) -> tuple:
    """Point estimates, bootstrap, BH and Baron-Kenny flags for a list of (diet, taxon) pairs."""
    ctx = context()
    X, M = ctx["X"], ctx["M"]
    Cdf = C if C is not None else pd.DataFrame(index=X.index)
    rows = []
    for d, t in pairs:
        r = pair_paths(X[d].to_numpy(), M[t].to_numpy(), y, Cdf)
        rows.append({"family": family, "outcome": outcome, "covariates": covariates,
                     "X": d, "M_feature_id": t, "M": ctx["label"](t),
                     "M_prevalence_n_of_88": int(ctx["prevalence"][t]), **r})
    res = pd.DataFrame(rows)
    boot = bootstrap_ab(X, M, y, Cdf, pairs, idx_draws)
    summ = bootstrap_summary(boot["ab"])
    res = pd.concat([res, summ], axis=1)
    res["B"] = idx_draws.shape[0]
    res["boot_q_BH"] = bh(res["boot_p"])
    res["a_q_BH"] = bh(res["a_p"])
    res["b_q_BH"] = bh(res["b_p"])
    c_unique = res.drop_duplicates("X")[["X", "c_p"]]
    c_q = dict(zip(c_unique["X"], bh(c_unique["c_p"])))
    res["c_q_BH_7_diet"] = res["X"].map(c_q)
    res["sobel_q_BH"] = bh(res["sobel_p"])
    reduced = res["c_prime"].abs() < res["c"].abs()
    res["BK1_a_p05"] = res["a_p"] < ALPHA
    res["BK2_c_p05"] = res["c_p"] < ALPHA
    res["BK3_b_p05_and_c_prime_reduced"] = (res["b_p"] < ALPHA) & reduced
    res["BK_all_three_p05"] = res["BK1_a_p05"] & res["BK2_c_p05"] & res["BK3_b_p05_and_c_prime_reduced"]
    res["BK1_a_q10"] = res["a_q_BH"] < Q_THRESHOLD
    res["BK2_c_q10"] = res["c_q_BH_7_diet"] < Q_THRESHOLD
    res["BK3_b_q10_and_c_prime_reduced"] = (res["b_q_BH"] < Q_THRESHOLD) & reduced
    res["BK_all_three_q10"] = res["BK1_a_q10"] & res["BK2_c_q10"] & res["BK3_b_q10_and_c_prime_reduced"]
    res["proportion_mediated"] = [f"{ab / c:.3f}" if cp < ALPHA else "n/a"
                                  for ab, c, cp in zip(res["ab"], res["c"], res["c_p"])]
    return res, boot


def bk_summary(res: pd.DataFrame) -> pd.DataFrame:
    n = len(res)
    rows = []
    for label, col in (("(1) a path p < 0.05", "BK1_a_p05"), ("(2) c path p < 0.05", "BK2_c_p05"),
                       ("(3) b path p < 0.05 and |c'| < |c|", "BK3_b_p05_and_c_prime_reduced"),
                       ("all three criteria (nominal 0.05)", "BK_all_three_p05"),
                       ("(1) a path q < 0.10", "BK1_a_q10"),
                       ("(2) c path q < 0.10 (BH over 7 diet variables)", "BK2_c_q10"),
                       ("(3) b path q < 0.10 and |c'| < |c|", "BK3_b_q10_and_c_prime_reduced"),
                       ("all three criteria (BH q < 0.10)", "BK_all_three_q10"),
                       ("bootstrap 95% CI of ab excludes 0", "ci_excludes_0")):
        rows.append({"criterion": label, "n_pairs_meeting": int(res[col].sum()), "n_pairs": n})
    rows.append({"criterion": "bootstrap q < 0.10", "n_pairs_meeting": int((res["boot_q_BH"] < Q_THRESHOLD).sum()),
                 "n_pairs": n})
    rows.append({"criterion": "Sobel p < 0.05 (descriptive)", "n_pairs_meeting": int((res["sobel_p"] < ALPHA).sum()),
                 "n_pairs": n})
    return pd.DataFrame(rows)


def fmt_path(beta: float, p: float) -> str:
    return f"{beta:+.3f} (p = {p:.3f})"


def formatted_table(res: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "X (diet, per 1 SD)": res["X"], "M (taxon)": res["M"],
        "a: M ~ X + C (SD of M per SD of X)": [fmt_path(b, p) for b, p in zip(res["a"], res["a_p"])],
        "c: Y ~ X + C (Zung points)": [fmt_path(b, p) for b, p in zip(res["c"], res["c_p"])],
        "b: Y ~ X + M + C (Zung points per SD of M)": [fmt_path(b, p) for b, p in zip(res["b"], res["b_p"])],
        "c': Y ~ X + M + C (Zung points)": [fmt_path(b, p) for b, p in zip(res["c_prime"], res["c_prime_p"])],
        "ab (95% bootstrap CI)": [f"{ab:+.3f} ({lo:+.3f} to {hi:+.3f})"
                                  for ab, lo, hi in zip(res["ab"], res["ab_boot_ci_low"], res["ab_boot_ci_high"])],
        "bootstrap p": res["boot_p"].map(lambda v: f"{v:.3f}"),
        "bootstrap q (BH)": res["boot_q_BH"].map(lambda v: f"{v:.3f}"),
        "proportion mediated": res["proportion_mediated"],
    })


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def environment_info() -> dict:
    import platform

    import scipy
    import statsmodels
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "platform": platform.platform()}
