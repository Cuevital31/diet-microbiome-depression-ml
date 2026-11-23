"""Step 7 - sensitivity analysis: confounder-residualized prediction (LR and linear SVM; see
src/dmdep/residualized.py). The confounders-only rows are the models of the main analysis.

Outputs: results/residualized/performance.csv (mean, SD, bootstrap 95% CI of the mean ROC-AUC),
         results/residualized/roc_auc_per_split.csv, results/residualized/taxa_selection_summary.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.evaluation import (bootstrap_draws, per_split_table, percentile_ci, predictions_by_config,  # noqa: E402
                              selection_summary, summarize)
from dmdep.io_utils import load_main_run, results_path, work_path, write_csv, write_json  # noqa: E402
from dmdep.pipeline import mean_aucs, run_all_splits  # noqa: E402
from dmdep.residualized import run_split_residualized  # noqa: E402


def main():
    cohort = load_cohort()
    run = run_all_splits(cohort, n_jobs=-1, split_fn=run_split_residualized)
    for key, frame in run.items():
        write_csv(frame, work_path("residualized", f"{key}.csv"))
    main_run = load_main_run()
    adj_configs = C.configurations(C.RESID_SETS, C.RESID_MODELS)
    conf_configs = C.configurations(("Confounders",), C.RESID_MODELS)

    # the permutation code path must reproduce this run on the true labels
    true_path = mean_aucs(cohort, cohort.y, split_fn=run_split_residualized)["mean_auc"]
    summ_adj = summarize(run["metrics"], adj_configs)
    for _, r in summ_adj.iterrows():
        assert abs(true_path[(r["Input features"], r["Model"])] - r["ROC-AUC mean"]) < 1e-12

    preds = {**{"ORIG | " + k: v for k, v in predictions_by_config(main_run["predictions"], C.configurations()).items()},
             **{"ADJ | " + k: v for k, v in predictions_by_config(run["predictions"], adj_configs).items()}}
    draws, _ = bootstrap_draws(preds)
    main_draws, _ = bootstrap_draws(predictions_by_config(main_run["predictions"], C.configurations()))
    for c in C.configurations():   # same random stream: the original columns equal the main-analysis bootstrap
        assert np.abs(draws["ORIG | " + C.config_label(c)].to_numpy() - main_draws[C.config_label(c)].to_numpy()).max() < 1e-12

    rows = []
    summ_conf = summarize(main_run["metrics"], conf_configs)
    for source, summ, prefix in (("main analysis", summ_conf, "ORIG | "), ("residualized", summ_adj, "ADJ | ")):
        for _, r in summ.iterrows():
            lo, hi = percentile_ci(draws[prefix + C.config_label((r["Input features"], r["Model"]))])
            rows.append({"Input features": r["Input features"], "Model": r["Model"], "Source": source,
                         **{f"{m} {s}": r[f"{m} {s}"] for m in ("ROC-AUC", "BAL-ACC", "SEN", "SPE") for s in ("mean", "SD")},
                         "ROC-AUC 95% CI low": lo, "ROC-AUC 95% CI high": hi})
    write_csv(pd.DataFrame(rows), results_path("residualized", "performance.csv"))
    write_csv(per_split_table(run["metrics"], adj_configs), results_path("residualized", "roc_auc_per_split.csv"))
    write_json(results_path("residualized", "taxa_selection_summary.json"), selection_summary(run["selection"]))
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
