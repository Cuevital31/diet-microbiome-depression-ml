"""Step 2 - main analysis: five repeated stratified splits x 15 configurations.

Participant-level outputs (work folder): predictions, split membership, metrics, coefficients, taxa selection,
Platt parameters. Aggregate outputs (results/): performance summary, ROC-AUC per split, taxa selection summary,
test-set composition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.evaluation import consensus_frequency, per_split_table, selection_summary, summarize  # noqa: E402
from dmdep.io_utils import results_path, work_path, write_csv, write_json  # noqa: E402
from dmdep.pipeline import run_all_splits  # noqa: E402


def main():
    cohort = load_cohort()
    run = run_all_splits(cohort, n_jobs=-1)
    for key, frame in run.items():
        write_csv(frame, work_path("main", f"{key}.csv"))
    configs = C.configurations()
    write_csv(summarize(run["metrics"], configs), results_path("main", "performance_summary.csv"))
    write_csv(per_split_table(run["metrics"], configs), results_path("main", "roc_auc_per_split.csv"))
    write_csv(run["metrics"].drop(columns=["fit_warning"]), results_path("main", "metrics_per_split.csv"))
    write_csv(consensus_frequency(run["selection"]), results_path("main", "taxa_consensus_frequency.csv"))
    test = run["membership"][run["membership"]["role"] == "test"]
    composition = (run["membership"].groupby(["Split", "role"])["y"].agg(n="size", depressive="sum").reset_index())
    write_csv(composition, results_path("main", "split_composition.csv"))
    write_json(results_path("main", "taxa_selection_summary.json"),
               {**selection_summary(run["selection"]), "pooled_test_instances": int(len(test)),
                "distinct_test_participants": int(test["participant_id"].nunique()),
                "convergence_warnings": int((run["metrics"]["fit_warning"].fillna("") != "").sum())})
    print(summarize(run["metrics"], configs)[["Input features", "Model", "ROC-AUC mean", "ROC-AUC SD"]].round(3))


if __name__ == "__main__":
    main()
