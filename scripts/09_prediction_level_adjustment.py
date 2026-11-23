"""Step 9 - prediction-level confounder adjustment (descriptive; Dinga et al., 2020) for the 15 configurations
of the main analysis and the 8 residualized configurations (see src/dmdep/prediction_level.py).

Output: results/prediction_level/adjustment.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort  # noqa: E402
from dmdep.io_utils import load_main_run, read_csv, results_path, work_path, write_csv  # noqa: E402
from dmdep.prediction_level import adjust  # noqa: E402


def main():
    cohort = load_cohort()
    sources = {"Main analysis": (load_main_run()["predictions"], C.configurations()),
               "Residualized": (read_csv(work_path("residualized", "predictions.csv")),
                                C.configurations(C.RESID_SETS, C.RESID_MODELS))}
    pooled = {}
    for run_name, (pred, configs) in sources.items():
        for fs, m in configs:
            sub = (pred[(pred["Input features"] == fs) & (pred["Model"] == m)]
                   .sort_values(["Split", "test_position"]).reset_index(drop=True))
            column = "prob_native" if m == "BRF" else "prob_platt"
            assert len(sub) == C.N_TEST * len(C.SEEDS) and sub[column].notna().all()
            pooled[f"{run_name} | {fs} | {m}"] = (sub, column)
    table = adjust(pooled, cohort.confounders)
    write_csv(table, results_path("prediction_level", "adjustment.csv"))
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
