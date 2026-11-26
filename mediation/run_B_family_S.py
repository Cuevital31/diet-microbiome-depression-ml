"""run_B_family_S.py - Revision-R2g: Family S (secondary scan, descriptive): 7 diet variables x all
taxa with raw count > 0 in >= 18 of 88 participants (188 taxa) = 1316 pairs; model M1; B = 2000,
seed 2026. --resume: skip the computation if results_per_pair.csv exists.
"""
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mediation_core as mc  # noqa: E402

T0 = time.time()
mc.check_plan()
OUT = HERE / "02_family_S"
OUT.mkdir(exist_ok=True)
ctx = mc.context()
N = len(ctx["ids"])
TAXA = ctx["taxa_s"]
assert len(TAXA) == 188, len(TAXA)
PAIRS = [(d, t) for d in mc.DIET for t in TAXA]
path = OUT / "results_per_pair.csv"

if "--resume" in sys.argv and path.exists():
    res = pd.read_csv(path, dtype={"M_feature_id": str})
    print("loaded existing results_per_pair.csv (--resume)")
else:
    idx = mc.draws(mc.B_S, N)
    np.save(OUT / "bootstrap_draws_B2000_seed2026.npy", idx)
    res, boot = mc.run_family(PAIRS, ctx["zung"], ctx["C"], idx, "S", "Zung_S (index score)",
                              "Age + BMI + SQ_all + PA (z-scored)")
    res["in_family_P"] = res["M_feature_id"].isin(mc.TAXA_P)
    res["note"] = "secondary descriptive scan; many tests"
    res.to_csv(path, index=False)
    np.save(OUT / "results_per_pair_ab_bootstrap.npy", boot["ab"])

n = len(res)
summary = pd.DataFrame([
    {"quantity": "pairs", "value": n},
    {"quantity": "taxa (prevalence >= 18 of 88)", "value": len(TAXA)},
    {"quantity": "bootstrap replicates B", "value": mc.B_S},
    {"quantity": "degenerate bootstrap replicates (all pairs)", "value": int(res["n_degenerate_replicates"].sum())},
    {"quantity": "pairs with bootstrap 95% CI of ab excluding 0", "value": int(res["ci_excludes_0"].sum())},
    {"quantity": "expected under the null at 5% (0.05 x pairs)", "value": 0.05 * n},
    {"quantity": "pairs with bootstrap p < 0.05", "value": int((res["boot_p"] < 0.05).sum())},
    {"quantity": "pairs with a path p < 0.05", "value": int((res["a_p"] < 0.05).sum())},
    {"quantity": "pairs with b path p < 0.05", "value": int((res["b_p"] < 0.05).sum())},
    {"quantity": "pairs with both a and b p < 0.05", "value": int(((res["a_p"] < 0.05) & (res["b_p"] < 0.05)).sum())},
    {"quantity": "pairs with Sobel p < 0.05 (descriptive)", "value": int((res["sobel_p"] < 0.05).sum())},
    {"quantity": "diet variables with c path p < 0.05 (of 7)", "value": int((res.drop_duplicates("X")["c_p"] < 0.05).sum())},
    {"quantity": "Baron-Kenny all three criteria (nominal 0.05)", "value": int(res["BK_all_three_p05"].sum())},
    {"quantity": "Baron-Kenny all three criteria (BH q < 0.10)", "value": int(res["BK_all_three_q10"].sum())},
    {"quantity": "pairs with bootstrap q < 0.10", "value": int((res["boot_q_BH"] < 0.10).sum())},
    {"quantity": "minimum bootstrap q", "value": float(res["boot_q_BH"].min())},
    {"quantity": "minimum bootstrap p", "value": float(res["boot_p"].min())},
    {"quantity": "minimum a-path q", "value": float(res["a_q_BH"].min())},
    {"quantity": "minimum b-path q", "value": float(res["b_q_BH"].min())},
    {"quantity": "minimum Sobel q", "value": float(res["sobel_q_BH"].min())},
])
summary.to_csv(OUT / "summary.csv", index=False)
cols = ["X", "M", "M_feature_id", "M_prevalence_n_of_88", "a", "a_p", "b", "b_p", "c", "c_p", "c_prime",
        "c_prime_p", "ab", "ab_boot_ci_low", "ab_boot_ci_high", "boot_p", "boot_q_BH", "sobel_p", "in_family_P"]
top = res.sort_values(["boot_p", "sobel_p"]).head(20)[cols]
top.to_csv(OUT / "top20_by_bootstrap_p.csv", index=False)
mc.bk_summary(res).to_csv(OUT / "bk_flags_summary.csv", index=False)
print(summary.to_string(index=False))
print("\ntop 20 by bootstrap p:")
print(top.round(4).to_string(index=False))
print(f"run_B_family_S.py finished in {time.time() - T0:.1f} s")
