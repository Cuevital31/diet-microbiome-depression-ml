"""Step 1 - load and preprocess the inputs; write the preprocessing summary (aggregate values only).

Outputs: results/preprocessing_summary.json, results/diet_spearman_matrix.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.io_utils import results_path, write_csv, write_json  # noqa: E402


def main():
    cohort = load_cohort()
    info = cohort.diet_info
    conf_corr = cohort.confounders.corr(method="spearman")
    off_diagonal = [conf_corr.iloc[i, j] for i in range(len(conf_corr)) for j in range(i + 1, len(conf_corr))]
    summary = {
        "participants": len(cohort.ids), "depressive": int(cohort.y.sum()), "control": int((cohort.y == 0).sum()),
        "outcome_rule": f"Zung index >= {C.ZUNG_CUTOFF:g} (identical to > {C.ZUNG_CUTOFF:g} in this cohort)",
        "microbiome": cohort.taxa_summary,
        "diet": {"screened_variables": list(info["screen_corr"].columns),
                 "pairs_with_abs_rho_at_or_above_0.90": info["pairs"], "removed": info["removed"],
                 "model_inputs": cohort.diet_names},
        "confounders": {"names": list(C.CONFOUNDERS), "spearman_min": float(min(off_diagonal)),
                        "spearman_max": float(max(off_diagonal))},
    }
    write_json(results_path("preprocessing_summary.json"), summary)
    corr = info["screen_corr"].copy()
    corr.insert(0, "Variable", corr.index)
    write_csv(corr, results_path("diet_spearman_matrix.csv"))
    print(summary)


if __name__ == "__main__":
    main()
