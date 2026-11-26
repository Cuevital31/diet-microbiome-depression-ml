# Mediation analysis (Supplementary Figure S3)

Exploratory single-mediator analysis: does a gut taxon statistically mediate the association between a dietary
variable and the continuous Zung index (diet → taxon → Zung score)? It follows the three-regression structure of
Bi et al. 2017 (Hum Brain Mapp 38:4088–4097) with a participant-level percentile bootstrap of the indirect effect
*ab* (Kenny, https://davidakenny.net/cm/mediate.htm). It is separate from the machine-learning pipeline in
`scripts/` and does not change Table 1. The data are cross-sectional, so the indirect effects are statistical, not
causal.

## Design
- n = 88 (15 participants with depressive symptoms). Outcome: Zung index score (Zung points).
- X: the 7 energy-adjusted dietary variables of the main run; M: CLR taxa; covariates: age, BMI, SQ_all, PA; all
  z-scored.
- Paths by OLS with HC3 standard errors: a (M ~ X + C), c (Y ~ X + C), b and c′ (Y ~ X + M + C), *ab* = a × b.
- **Family P** (primary): 7 diet variables × 6 prioritised taxa = 42 pairs, B = 5,000. The taxa were prioritised
  using the outcome, so the q-values are not confirmatory.
- **Family S** (descriptive): 7 diet variables × 188 taxa present in ≥ 18 of 88 participants = 1,316 pairs, B = 2,000.
- **Sensitivity** (Family P): S1 rank-based inverse-normal Zung score; S2 without covariates.
- Seed 2026; Benjamini–Hochberg q-values.

## Main results
- Family P: 0 of 42 bootstrap 95% intervals of *ab* exclude 0; minimum bootstrap q = 0.960.
- No dietary variable has a total effect c with p < 0.05.
- Sensitivity: 0 of 42 intervals exclude 0 in both S1 and S2.
- Family S: 0 of 1,316 intervals exclude 0.

All numbers are in `headline_results.csv`, each with its source file.

## Files
| Item | File(s) |
|---|---|
| Supplementary Figure S3 (forest plot of 42 indirect effects) | `01_family_P/forest_plot_ab_family_P.pdf` (`.png`) |
| Family P results per pair; formatted table | `01_family_P/results_per_pair.csv`, `table_M1_formatted.csv` |
| Fibre × 6 taxa | `01_family_P/fiber_pairs.csv`, `fiber_pairs_full_columns.csv` |
| Baron–Kenny flags | `bk_flags_summary.csv` in each results folder |
| Family S | `02_family_S/results_per_pair.csv`, `summary.csv`, `top20_by_bootstrap_p.csv` |
| Sensitivity analyses | `03_sensitivity/` |
| Consistency checks (rerun of the c paths, independent implementations, bootstrap loop, BH) | `00_checks/checks_summary_1_7_9.csv`, `REVIEW_CHECKLIST.md` |
| Analysis plan, written before any computation (SHA-256 in `plan_hash.txt`) | `ANALYSIS_PLAN_R2g.md` |

## Code
The scripts are included as they were run, for transparency. They are not part of `run_all.py`.

| Script | Purpose |
|---|---|
| `run_0_checksums.py` | Plan hash and checksums of the input folders |
| `run_1_copy_inputs.py` | Copies the inputs into `inputs/` and verifies each copy |
| `run_A_family_P.py` | Family P, Fibre table, sensitivity analyses S1 and S2, forest plot |
| `run_B_family_S.py` | Family S |
| `run_C_checks.py` | Consistency checks |
| `mediation_core.py` | Shared code (regressions, bootstrap, BH) |
| `pipeline_core_from_R2c-conf.py` | Unchanged copy of the data-loading module of the main run (an earlier version of the code in `src/dmdep/`) |

The scripts expect the folder layout of the original analysis (this folder next to `Preprocessed Files/`,
`Revision-R2c-conf/` and `Revision-R2d/`). Every script checks the hash of the analysis plan and refuses to run if it
changed.

## Not included
- The participant-level inputs (`inputs/`; see `../data/README.md`) and the main-run intermediates
  (`Revision-R2c-conf/`).
- The bootstrap draws (`*.npy`), which are resampling indices of the participants.

## Packages
Verified with Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3 and statsmodels 0.14.4 (matplotlib for the forest plot).
