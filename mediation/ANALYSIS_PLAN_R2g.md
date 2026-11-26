# ANALYSIS PLAN - Revision-R2g: conventional mediation analysis (diet -> gut taxon -> Zung score)

Written before any computation (2026-11-18). Its SHA-256 is saved in `plan_hash.txt`; every
script refuses to run if this file changes.

## Question and scope
Walter asks: "Does the gut microbiome mediate the association between diet and depressive
symptoms?" Answered with a conventional single-mediator analysis on the study data (n = 88),
following the structure of Bi et al. 2017 (Hum Brain Mapp 38:4088-4097, Figure 1:
x -> m, x -> y, x + m -> y; Baron-Kenny) and Kenny's current guidance
(https://davidakenny.net/cm/mediate.htm):

- the indirect effect ab is evaluated directly with a participant-level percentile bootstrap;
- the Sobel test is reported descriptively only;
- a significant total effect is not required.

This is an exploratory supplementary analysis. It does not change Table 1. It makes no claim
that ML is better or worse than mediation, because the two answer different questions. It makes
no causal claim, because the data are cross-sectional.

## Rules
- New folder `Clean-Codes/Revision-R2g` only. Protected (read-only) folders:
  - Revision-R2b, Revision-R2c, Revision-R2b-conf, Revision-R2c-conf;
  - Revision-R2d, Revision-R2e, Revision-R2f.

  SHA-256 of every file in them before any computation and after the last step; the two lists
  must be identical.
- No questions to the author: unspecified choices are taken as defaults and listed in the
  README under "Defaults chosen".
- Standalone scripts. No ML, no splits, no taxa selection. Every script supports `--resume`.
- Data loading reuses Revision-R2c-conf `pipeline_core.py`, copied unchanged as
  `pipeline_core_from_R2c-conf.py` with a hash check: `load_all_cohorts`,
  `cohort_for(..., "CONF4")`, `build_diet_design`, the CLR taxa matrix of the 249-taxa run and
  `confounder_frame`. This is the same loading used by Revision-R2e Part A and Revision-R2d.
- All inputs are copied into `Revision-R2g/inputs/`, which mirrors the original folder layout,
  and each copy is checked against its original by SHA-256. The scripts read only the copies.
  The copied files:
  - `Preprocessed Files/participants_v3_corrected_depflag.tsv`
  - `Preprocessed Files/taxonomy_leaf_CLR_v2.xlsx`
  - `Preprocessed Files/taxonomy_completo_paper_v5.xlsx` (raw counts for prevalence and parent names)
  - `Revision-R2c-conf/inputs/` (all files)
  - `Revision-R2c-conf/01_main/MAIN_conf_Age_BMI_SQ_PA_249taxa/run_config.json`
  - `Revision-R2d/results/family_D_M0.csv`, `family_D_M1.csv` (reference values for check 1)
  - `Revision-R2d/results/sensitivity_rankZung_D_M1.csv` (for reference only)

## Data and definitions
- **Sample:** n = 88; 15 depressive and 73 controls by Zung_S >= 50.
- **Outcome Y (primary):** Zung_S, the continuous index score, used unscaled. So a, c, c' and ab
  are in Zung points.
- **Exposures X:** the 7 nutrient densities of the R2c-conf main run, z-scored over the 88
  participants (sample SD, ddof = 1):
  - Prot_pctE, SatFat_pctE, MUF_pctE, Poly_InsFat_pctE;
  - Chol_mg_per_1000kcal, Carb_pctE, Fiber_g_per_1000kcal.
- **Mediators M:** CLR taxa of the 249-taxa table, z-scored over the 88 participants.
- **Covariates C (primary model M1):** Age, BMI, SQ_all and PA (numeric 1-3). They are z-scored
  as in Revision-R2d; this does not change any X or M coefficient.
- **Taxa labels:** genus (family); Feature ID, as in the R2e tables.
- **No re-standardisation inside bootstrap replicates:** the z-scores of the full sample are
  kept fixed.

## Pair families
- **Family P** (primary, pre-specified, 42 pairs): the 7 diet variables x the 6 prioritised taxa
  of R2e Part A:
  - Lachnospiraceae_ND3007_group 0.2.10.2.2.11.26
  - Ruminiclostridium_9 0.2.10.2.2.14.28
  - Catabacter 0.2.10.2.2.2.1
  - Lachnospiraceae_FCS020_group 0.2.10.2.2.11.24
  - Rhizobiales_unclassified 0.2.17.1.6.5.1
  - Dielma 0.2.10.3.1.1.6

  These taxa were prioritised using the outcome (ML coefficients), so Family P is exploratory
  and its q-values are not confirmatory. Fiber_g_per_1000kcal x each of the 6 taxa is reported
  in a separate small table (Walter's example).
- **Family S** (secondary scan, descriptive): the 7 diet variables x every taxon with raw count
  > 0 in at least 18 of the 88 participants (ceil(0.20 x 88); 188 taxa, as in R2e and R2d)
  = 1316 pairs.

## Models per pair (all 88 participants, OLS)
- a path: M ~ X + C
- c path (total effect): Y ~ X + C
- b and c' paths: Y ~ X + M + C
- Indirect effect ab = a x b.

**Reported per path (a, b, c, c'):** beta, HC3 SE and HC3 p (t reference distribution with the
residual df, as in R2d).

**Sobel test (descriptive):** z = ab / sqrt(b^2 SE_a^2 + a^2 SE_b^2) with the HC3 SEs; two-sided
normal p.

**Baron-Kenny flags (descriptive):**
- (1) a: p < 0.05;
- (2) c: p < 0.05;
- (3) b: p < 0.05 and |c'| < |c|.

Report the number of pairs meeting all three, at nominal 0.05 and at BH q < 0.10.

**BH for the flags:**
- for a and b, across the pairs of the family;
- for c, across the 7 unique diet variables, since c does not depend on the taxon.

**Bootstrap of ab:**
- B = 5000 for Family P and B = 2000 for Family S; seed 2026 (numpy default_rng).
- Participants are resampled with replacement: all 88 rows together, with the same index draws
  for all pairs of a family.
- Both regressions (a path and b path) are refitted in every replicate. The b path is computed
  through Frisch-Waugh-Lovell, which is identical to the OLS coefficient of M in Y ~ X + M + C.
- **Degenerate replicate:** the design [1, X, C] is rank deficient, or the residual variance of
  M after [1, X, C] is <= 1e-12 times its full-sample value. Such replicates are excluded for
  that pair and counted; the count should be 0.
- **Reported:**
  - the 95% percentile interval of ab (2.5 and 97.5 percentiles);
  - a two-sided bootstrap p = 2 x min(share of ab <= 0, share of ab >= 0), floored at 1/(B+1)
    and capped at 1;
  - the number of degenerate replicates;
  - BH q within each family on the bootstrap p.
- **Proportion mediated** ab/c is reported only when the c path has p < 0.05; otherwise "n/a".

## Sensitivity runs (Family P only)
- **S1:** Y = rank-based inverse-normal Zung score (Blom: qnorm((r - 3/8) / (n + 1/4)), average
  ranks for ties). Covariates M1.
- **S2:** no covariates (M0), Zung_S.

Same outputs, B = 5000, same seed and draws.

## Outputs
- `01_family_P/`:
  - results_per_pair.csv
  - table_M1_formatted.csv (one row per pair: X, M, a, c, b, c', ab with 95% bootstrap CI,
    bootstrap q)
  - fiber_pairs.csv
  - bk_flags_summary.csv
  - forest_plot_ab_family_P.png / .pdf
- `02_family_S/`:
  - results_per_pair.csv
  - summary.csv (pairs with CI excluding 0 vs the 5% nominal expectation; nominal p < 0.05
    counts for a, b and ab; min q)
  - top20_by_bootstrap_p.csv
- `03_sensitivity/`: S1 and S2 files.
- `00_checks/`:
  1. the c paths reproduce the R2d diet-family M1 betas and HC3 p-values (tolerance 1e-8); S2 c
     paths are also compared with R2d M0;
  2. a, b and c' for 20 random Family S pairs equal an independent statsmodels fit;
  3. the bootstrap ab for 3 pairs is reproduced by an independent loop (full-design lstsq) on
     the same draws;
  4. n = 88 and no missing values for all pairs;
  5. z-scoring check (mean 0, SD 1);
  6. taxa prevalence count = 188 and input hashes (copies = originals);
  7. BH reproduced with two implementations;
  8. protected folders unchanged;
  9. plan hash unchanged.
- `README.md`:
  - purpose and exact definitions;
  - results read from the CSV files (never typed by hand);
  - "Defaults chosen";
  - limitations: cross-sectional design; temporal ordering and no-unmeasured-confounding
    assumptions cannot be tested here; 88 participants / 15 depressive; low power for ab and c;
    taxa prioritised using the outcome; many pairs in Family S;
  - a neutral Methods paragraph and a neutral Results paragraph for the supplement;
  - a sentence stating that this analysis answers a different question from the ML models, so
    the two are not compared as methods;
  - no claims of significance unless the q-value supports them.
- `headline_results.csv` (every number with its source file) and `REVIEW_CHECKLIST.md`.
- `Clean-Codes/Revision-R2g.rar` (tested), plus a project memory update.

## Scripts (run in this order)
1. `run_0_checksums.py plan`, then `before`.
2. `run_1_copy_inputs.py`: copies and verifies the inputs, and copies the pipeline core.
3. `run_A_family_P.py`: Family P, the Fiber table, the BK flags, the forest plot and the S1 / S2
   sensitivity runs.
4. `run_B_family_S.py`: Family S.
5. `run_C_checks.py`: checks 1-7 and 9.
6. `run_0_checksums.py after`: check 8.
7. `run_D_write_readme.py`: README, headline_results.csv, REVIEW_CHECKLIST.md, run_config.json.
