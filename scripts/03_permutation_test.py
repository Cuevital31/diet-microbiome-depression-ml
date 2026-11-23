"""Step 3 - max-statistic label-permutation test of the main analysis (B = 1,000, seed 20260).

In every permutation the outcome labels are shuffled across the 88 participants (15 depressive kept) and the
complete pipeline is repeated: stratified splitting, taxa feature selection, model training and testing of all
15 configurations. The statistic of a family of configurations is the maximum mean ROC-AUC over the family;
p = (1 + k) / (1 + B), where k is the number of permutation maxima at least as large as the observed maximum.
Checkpointed after every chunk; rerunning the script resumes.

Outputs: results/permutation/{families.csv, per_configuration.csv, null_mean_auc.csv}
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.evaluation import family_result, unadjusted_p  # noqa: E402
from dmdep.io_utils import read_csv, results_path, work_path, write_csv  # noqa: E402
from dmdep.permutation import main_worker, run_permutations  # noqa: E402
from dmdep.pipeline import mean_aucs  # noqa: E402

def main():
    cohort = load_cohort()
    summary = read_csv(results_path("main", "performance_summary.csv"))
    observed = {f"{r['Input features']} | {r['Model']}": float(r["ROC-AUC mean"]) for _, r in summary.iterrows()}
    true_path = mean_aucs(cohort, cohort.y)["mean_auc"]           # the permutation code path on the true labels
    assert all(abs(true_path[c] - observed[C.config_label(c)]) < 1e-12 for c in true_path)

    null = run_permutations(main_worker, work_path("permutation", "checkpoint.csv"))
    columns = [C.config_label(c) for c in C.configurations()]
    write_csv(null[["perm_index"] + columns + ["n_empty_consensus_splits"]], results_path("permutation", "null_mean_auc.csv"))

    fams, maxima = [], {}
    for name, sets in C.PERM_FAMILIES.items():
        cols = [C.config_label(c) for c in C.configurations(sets)]
        res = family_result(null, observed, cols, C.PERM_B)
        maxima[name] = res.pop("null_max")
        fams.append({"Family (configurations)": name, "Feature sets": ", ".join(sets), **res})
    write_csv(pd.DataFrame(fams), results_path("permutation", "families.csv"))

    per = []
    for c in columns:
        fs = c.split(" | ")[0]
        row = {"Configuration": c, "Observed mean ROC-AUC": observed[c],
               "Unadjusted p": unadjusted_p(null[c], observed[c], C.PERM_B),
               "Adjusted p (15)": unadjusted_p(maxima["15"], observed[c], C.PERM_B),
               "Adjusted p (12)": (unadjusted_p(maxima["12"], observed[c], C.PERM_B)
                                   if fs in C.PERM_FAMILIES["12"] else float("nan"))}
        per.append(row)
    write_csv(pd.DataFrame(per), results_path("permutation", "per_configuration.csv"))
    print(pd.DataFrame(fams).round(4).to_string())


if __name__ == "__main__":
    main()
