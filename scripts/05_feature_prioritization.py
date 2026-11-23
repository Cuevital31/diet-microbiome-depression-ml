"""Step 5 - coefficient-based prioritisation in the main configuration (LR with confounders, diet and taxa).

Outputs:
  results/prioritization/median_rule_all_features.csv   every feature of the main configuration
  results/prioritization/prioritised_features.csv       features retained in >= 2 of 5 splits (with split coefficients)
  results/prioritization/mean_coefficients.csv          mean coefficients in Diet + Taxa LR and in the main configuration
  results/prioritization/taxa_consensus_counts.csv      splits in which each prioritised taxon passed the consensus
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dmdep import config as C  # noqa: E402
from dmdep.data import load_cohort, taxon_label  # noqa: E402
from dmdep.io_utils import load_main_run, results_path, write_csv  # noqa: E402
from dmdep.prioritization import mean_coefficients, median_rule  # noqa: E402


def main():
    cohort = load_cohort()
    run = load_main_run()
    rule = median_rule(run["coefficients"], C.MAIN_CONFIG[0], C.MAIN_CONFIG[1])
    label = {fid: (taxon_label(cohort, fid) if fid in cohort.taxon_names else fid) for fid in rule["Feature ID"]}
    rule.insert(1, "Label", rule["Feature ID"].map(label))
    write_csv(rule, results_path("prioritization", "median_rule_all_features.csv"))
    prio = rule[rule["Prioritised"]].reset_index(drop=True)
    write_csv(prio, results_path("prioritization", "prioritised_features.csv"))

    features = list(prio.loc[prio["Feature group"] != "Confounder", "Feature ID"])
    a = mean_coefficients(run["coefficients"], features, "Diet + Taxa").add_suffix(" (Diet + Taxa LR)")
    b = mean_coefficients(run["coefficients"], features, C.MAIN_CONFIG[0]).add_suffix(" (Confounders + Diet + Taxa LR)")
    means = pd.concat([a.rename(columns={"Feature ID (Diet + Taxa LR)": "Feature ID"}),
                       b.drop(columns="Feature ID (Confounders + Diet + Taxa LR)")], axis=1)
    write_csv(means, results_path("prioritization", "mean_coefficients.csv"))

    sel = run["selection"]
    counts = sel.groupby("rankID")["Consensus"].sum()
    write_csv(pd.DataFrame({"Feature ID": [f for f in features if f in cohort.taxon_names],
                            "Splits selected by consensus": [int(counts[f]) for f in features if f in cohort.taxon_names]}),
              results_path("prioritization", "taxa_consensus_counts.csv"))
    print(prio[["Label", "Feature group", "Splits in model", "Splits retained", "Mean coefficient"]].round(3).to_string())
    print(f"prioritised: {len(prio)} features ({len(features)} without the confounders)")


if __name__ == "__main__":
    main()
