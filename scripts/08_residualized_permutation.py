"""Step 8 - max-statistic permutation test of the residualized pipeline (B = 1,000; the permuted label vectors
of the main analysis; residualization and taxa selection rerun in every permutation). Three families:
the 8 residualized configurations; LR only (5 main-analysis LR + 4 residualized LR); all 23 configurations
(15 main-analysis + 8 residualized), evaluated with the null of step 3 for the main-analysis configurations.

Outputs: results/residualized/permutation_families.csv, results/residualized/permutation_null_mean_auc.csv
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.evaluation import family_result  # noqa: E402
from dmdep.io_utils import read_csv, results_path, work_path, write_csv  # noqa: E402
from dmdep.permutation import residualized_worker, run_permutations  # noqa: E402


def main():
    adj_configs = C.configurations(C.RESID_SETS, C.RESID_MODELS)
    null_adj = run_permutations(residualized_worker, work_path("residualized_permutation", "checkpoint.csv"))
    adj_cols = [C.config_label(c) for c in adj_configs]
    write_csv(null_adj[["perm_index"] + adj_cols + ["n_empty_consensus_splits"]],
              results_path("residualized", "permutation_null_mean_auc.csv"))
    null_main = read_csv(results_path("permutation", "null_mean_auc.csv"))
    assert null_main["perm_index"].tolist() == null_adj["perm_index"].tolist()

    perf = read_csv(results_path("residualized", "performance.csv"))
    main_summary = read_csv(results_path("main", "performance_summary.csv"))
    observed = {"ADJ | " + f"{r['Input features']} | {r['Model']}": float(r["ROC-AUC mean"])
                for _, r in perf[perf["Source"] == "residualized"].iterrows()}
    observed.update({"ORIG | " + f"{r['Input features']} | {r['Model']}": float(r["ROC-AUC mean"])
                     for _, r in main_summary.iterrows()})
    null = pd.DataFrame({"ADJ | " + c: null_adj[c].to_numpy() for c in adj_cols})
    for c in C.configurations():
        null["ORIG | " + C.config_label(c)] = null_main[C.config_label(c)].to_numpy()

    families = {
        "8 residualized configurations": ["ADJ | " + c for c in adj_cols],
        "LR only: 5 main-analysis LR + 4 residualized LR":
            ["ORIG | " + C.config_label((fs, "LR")) for fs in C.FEATURE_SETS]
            + ["ADJ | " + C.config_label((fs, "LR")) for fs in C.RESID_SETS],
        "15 main-analysis + 8 residualized configurations":
            ["ORIG | " + C.config_label(c) for c in C.configurations()] + ["ADJ | " + c for c in adj_cols],
    }
    rows = []
    for name, cols in families.items():
        res = family_result(null, observed, cols, C.PERM_B)
        res.pop("null_max")
        rows.append({"Family": name, "Configurations": len(cols), **res})
    write_csv(pd.DataFrame(rows), results_path("residualized", "permutation_families.csv"))
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
