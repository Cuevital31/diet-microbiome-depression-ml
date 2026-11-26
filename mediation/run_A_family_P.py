"""run_A_family_P.py - Revision-R2g: Family P (7 diet variables x 6 prioritised taxa = 42 pairs),
primary model M1 (Zung_S ~ ... + Age + BMI + SQ_all + PA), the Fiber table, Baron-Kenny flag
summary, forest plot, and the sensitivity runs S1 (inverse-normal Zung, M1) and S2 (Zung, no
covariates). B = 5000, seed 2026, same draws for all runs. --resume: skip runs whose results exist.
"""
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mediation_core as mc  # noqa: E402

T0 = time.time()
mc.check_plan()
RESUME = "--resume" in sys.argv
OUT_P, OUT_S = HERE / "01_family_P", HERE / "03_sensitivity"
for d in (OUT_P, OUT_S):
    d.mkdir(exist_ok=True)

ctx = mc.context()
N = len(ctx["ids"])
PAIRS = [(d, t) for d in mc.DIET for t in mc.TAXA_P]
assert len(PAIRS) == 42
idx = mc.draws(mc.B_P, N)
np.save(OUT_P / "bootstrap_draws_B5000_seed2026.npy", idx)

RUNS = {"M1": (OUT_P / "results_per_pair.csv", ctx["zung"], ctx["C"],
               "Zung_S (index score)", "Age + BMI + SQ_all + PA (z-scored)"),
        "S1": (OUT_S / "S1_inverse_normal_zung_M1_results_per_pair.csv", mc.inverse_normal(ctx["zung"]),
               ctx["C"], "rank-based inverse-normal Zung_S (Blom)", "Age + BMI + SQ_all + PA (z-scored)"),
        "S2": (OUT_S / "S2_no_covariates_M0_results_per_pair.csv", ctx["zung"], None,
               "Zung_S (index score)", "none (M0)")}

results = {}
for key, (path, y, C, outcome, covs) in RUNS.items():
    if RESUME and path.exists():
        results[key] = pd.read_csv(path, dtype={"M_feature_id": str})
        print(f"{key}: loaded existing {path.name} (--resume)")
        continue
    t = time.time()
    res, boot = mc.run_family(PAIRS, y, C, idx, "P", outcome, covs)
    res.insert(3, "run", key)
    res["note"] = mc.NOTE_P
    res.to_csv(path, index=False)
    np.save(path.with_name(path.stem + "_ab_bootstrap.npy"), boot["ab"])
    results[key] = res
    print(f"{key}: 42 pairs, B = {mc.B_P}, {time.time() - t:.1f} s; degenerate replicates: "
          f"{int(res['n_degenerate_replicates'].sum())}")

# formatted tables, Fiber table, BK summaries
for key, res in results.items():
    tab = mc.formatted_table(res)
    target = OUT_P / "table_M1_formatted.csv" if key == "M1" else OUT_S / f"{key}_table_formatted.csv"
    tab.to_csv(target, index=False)
    bk = mc.bk_summary(res)
    (OUT_P / "bk_flags_summary.csv" if key == "M1" else OUT_S / f"{key}_bk_flags_summary.csv").write_text(
        bk.to_csv(index=False), encoding="utf-8")
m1 = results["M1"]
fib = mc.formatted_table(m1[m1["X"] == mc.FIBER])
fib.to_csv(OUT_P / "fiber_pairs.csv", index=False)
m1[m1["X"] == mc.FIBER].to_csv(OUT_P / "fiber_pairs_full_columns.csv", index=False)

# forest plot of ab (M1) with 95% bootstrap CI, grouped by diet variable
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
fig_h = 0.24 * len(m1) + 1.6
fig, ax = plt.subplots(figsize=(8.2, fig_h))
ypos, labels, y0 = [], [], 0
group_mid = {}
for d in mc.DIET:
    sub = m1[m1["X"] == d]
    start = y0
    for _, r in sub.iterrows():
        ypos.append(y0)
        labels.append(r["M"].split(";")[0].split(" (")[0])
        y0 += 1
    group_mid[d] = (start + y0 - 1) / 2
    y0 += 0.8
ypos = np.array(ypos)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.hlines(ypos, m1["ab_boot_ci_low"], m1["ab_boot_ci_high"], color=MUTED, lw=2, zorder=2)
ax.scatter(m1["ab"], ypos, s=40, color=INK, zorder=3, edgecolor="white", linewidth=1.5)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8, color=INK)
ax.invert_yaxis()
xmin = float(m1["ab_boot_ci_low"].min())
for d, mid in group_mid.items():
    ax.annotate(d, xy=(0, mid), xycoords=("axes fraction", "data"), xytext=(-150, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK,
                fontweight="bold")
ax.set_xlabel("Indirect effect ab (Zung points per 1 SD of the diet variable)\n"
              "point estimate with 95% percentile bootstrap CI", fontsize=8.5, color=INK)
fig.suptitle(f"Family P: diet -> taxon -> Zung_S, adjusted for Age, BMI, SQ_all and PA "
             f"(n = {N}, B = {mc.B_P})", fontsize=9.5, color=INK, x=0.02, ha="left")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.tick_params(axis="x", labelsize=8, colors=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.47, right=0.97, top=1 - 0.55 / fig_h, bottom=0.75 / fig_h)
for ext in ("png", "pdf"):
    fig.savefig(OUT_P / f"forest_plot_ab_family_P.{ext}", dpi=200)
plt.close(fig)

mc.write_json(OUT_P / "family_P_settings.json", {
    "plan_sha256": mc.check_plan(), "pairs": [list(p) for p in PAIRS], "B": mc.B_P, "seed": mc.SEED,
    "covariates_M1": mc.CONF, "outcome": "Zung_S", "n": N, "runs": {k: v[3] + " | " + v[4] for k, v in RUNS.items()},
    "environment": mc.environment_info(), "seconds": time.time() - T0})

print("\n--- M1 Baron-Kenny / bootstrap summary")
print(mc.bk_summary(m1).to_string(index=False))
print("\n--- Fiber pairs (M1)")
print(fib.to_string(index=False))
for k in ("S1", "S2"):
    r = results[k]
    print(f"{k}: CI excludes 0 in {int(r['ci_excludes_0'].sum())}/42; min boot q {r['boot_q_BH'].min():.3f}")
print(f"run_A_family_P.py finished in {time.time() - T0:.1f} s")
