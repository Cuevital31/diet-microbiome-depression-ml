"""Step 4 - stratified test-set bootstrap (B = 2,000, seed 2026) and probability calibration.

* 95% CI of the mean ROC-AUC of every configuration (2.5th-97.5th percentile of the bootstrap means);
* paired comparisons with four references: LR and SVM with confounders, diet and taxa; LR and SVM with diet
  and taxa (difference = reference minus comparator);
* Brier scores of the pooled held-out predictions versus a constant prediction of 15/88, and calibration bins
  (quartiles of the predicted probability) of the main configuration.

Outputs: results/bootstrap/*.csv, results/calibration/*.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.evaluation import (bootstrap_difference, bootstrap_draws, brier_table, calibration_bins,  # noqa: E402
                              percentile_ci, predictions_by_config)
from dmdep.io_utils import load_main_run, results_path, write_csv  # noqa: E402

REFERENCES = {"conf_diet_taxa_LR": ("Confounders + Diet + Taxa", "LR"), "diet_taxa_LR": ("Diet + Taxa", "LR"),
              "conf_diet_taxa_SVM": ("Confounders + Diet + Taxa", "SVM"), "diet_taxa_SVM": ("Diet + Taxa", "SVM")}


def main():
    run = load_main_run()
    configs = C.configurations()
    draws, observed = bootstrap_draws(predictions_by_config(run["predictions"], configs))
    ci = pd.DataFrame([{"Configuration": lab, "Mean ROC-AUC": observed[lab].mean(),
                        "95% CI low": percentile_ci(draws[lab])[0], "95% CI high": percentile_ci(draws[lab])[1]}
                       for lab in draws.columns])
    write_csv(ci, results_path("bootstrap", "mean_auc_ci.csv"))
    for tag, ref in REFERENCES.items():
        ref_label = C.config_label(ref)
        rows = [bootstrap_difference(draws, observed, ref_label, lab) for lab in draws.columns if lab != ref_label]
        write_csv(pd.DataFrame(rows), results_path("bootstrap", f"paired_ref_{tag}.csv"))

    write_csv(brier_table(run["predictions"], C.FEATURE_SETS), results_path("calibration", "brier_scores.csv"))
    pred = run["predictions"]
    main = pred[(pred["Input features"] == C.MAIN_CONFIG[0]) & (pred["Model"] == C.MAIN_CONFIG[1])]
    bins = []
    for column, kind in (("prob_native", "native"), ("prob_platt", "Platt (training set)")):
        b = calibration_bins(main["y_true"], main[column])
        b.insert(0, "Probability", kind)
        bins.append(b)
    write_csv(pd.concat(bins, ignore_index=True), results_path("calibration", "bins_main_configuration.csv"))
    print(ci.round(3).to_string())


if __name__ == "__main__":
    main()
