"""Step 10 - figures: Figure 1 (split-level ROC-AUC), Figure 2 (ROC curve and pooled confusion matrix of the main
configuration), Figure 3 (LR coefficients of the 13 prioritised dietary and microbial features), Figure 5 (workflow
schematic) and Supplementary Figure S1 (Spearman correlations among the energy-adjusted dietary variables).

Outputs: figures/*.pdf (+ *_preview.png), results/figures/figure_values.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmdep import config as C  # noqa: E402
from dmdep import figures  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.io_utils import load_main_run, read_csv, results_path, write_json  # noqa: E402


def main():
    cohort = load_cohort()
    run = load_main_run()
    m = run["metrics"]
    auc = {(fs, mod): m[(m["Input features"] == fs) & (m["Model"] == mod)].sort_values("Split")["ROC-AUC"].to_numpy()
           for fs, mod in C.configurations()}
    figures.roc_auc_per_split(auc)

    p = run["predictions"]
    main = p[(p["Input features"] == C.MAIN_CONFIG[0]) & (p["Model"] == C.MAIN_CONFIG[1])]
    counts = figures.roc_curve_and_confusion(main)

    prio = read_csv(results_path("prioritization", "prioritised_features.csv"))
    prio = prio[prio["Feature group"] != "Confounder"].reset_index(drop=True)
    labels = {f: cohort.taxon_names.get(f, f) for f in prio["Feature ID"]}
    figures.coefficients_across_splits(prio, labels)
    figures.workflow()
    density = cohort.diet_info["density"]
    figures.diet_correlation_heatmap(density.corr(method="spearman"))
    n_span = sum(min(v) < 0.5 < max(v) for v in auc.values())
    write_json(results_path("figures", "figure_values.json"),
               {"pooled_confusion_main_configuration": counts,
                "configurations_with_split_values_on_both_sides_of_0.5": n_span,
                "figure3_filled_markers": int(prio[[f"Retained split {s}" for s in range(1, 6)]].to_numpy().sum()),
                "figure3_features": len(prio)})
    print(counts, "configurations spanning 0.5:", n_span)


if __name__ == "__main__":
    main()
