# Dietary and gut microbiome features associated with depression status: analysis code

Code for the machine-learning and statistical analyses of the manuscript *"Dietary and Gut Microbiome Features
Associated with Depression Status: An Exploratory Machine-Learning Study"* (International Journal of Molecular
Sciences, manuscript ijms-4632259).

The pipeline relates dietary intake and gut microbiome composition to depressive symptom status (Zung index ≥ 50) in
88 Colombian men (15 depressive, 73 control). It includes:
- preprocessing;
- five repeated stratified holdout splits with split-specific consensus taxa selection;
- five feature sets × three classifiers;
- uncertainty analyses (stratified bootstrap, label-permutation tests, calibration);
- coefficient-based feature prioritisation with confounder-adjusted Firth logistic regression;
- two confounder sensitivity analyses;
- the figures of these analyses.

**Notebooks.** The measured short-chain fatty acid (SCFA) analyses (Supplementary Tables S2–S4, Figure 4,
Supplementary Figure S2) and the PICRUSt2/KEGG functional predictions (Supplementary Tables S5–S7) are provided as
notebooks in `notebooks/` (see `notebooks/README.md`). They are not part of `run_all.py`: they were written for Google
Colab, and the SCFA notebook needs participant-level intermediates of the main pipeline that are not shared.

**Mediation analysis.** The exploratory single-mediator analysis (diet → taxon → Zung score; Supplementary Figure S3) is in
`mediation/` (see `mediation/README.md`). Its scripts and aggregate results are included; like the notebooks, it is not
part of `run_all.py` and needs the participant-level inputs.

## Requirements

- **Python 3.13.5** with the exact package builds in `environment.yml`: Anaconda defaults channel, Windows 64-bit,
  NumPy 2.1.3 linked to MKL 2023.1.0, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, imbalanced-learn 0.13.0,
  joblib 1.4.2, matplotlib 3.10.0, openpyxl 3.1.5.
- **R 4.5.1** with logistf 1.26.1, used for the Firth regressions; `R/install_packages.R` installs it with its
  dependencies into a separate library.
- About 1 GB of disk space for the environments.
- Several CPU cores are recommended for the permutation tests.

### Numerical reproducibility

The taxa selection uses a nearest-neighbour mutual-information estimator. That estimator is sensitive to differences
in the last digits of its input, so the reported numbers are reproduced only when the floating-point inputs are
exactly those of the original analysis. Two steps in `src/dmdep/data.py` therefore recreate the stored
representation of the inputs; both change values only at the level of about 1e-15:
- CLR values are rounded to 16 significant digits;
- confounder values go through one CSV write/read with pandas' default parser.

Two further constraints apply:
- **MKL versus OpenBLAS.** NumPy linked to OpenBLAS (for example, pip wheels) instead of MKL changes the residualized
  sensitivity analysis (Supplementary Table S18).
- **Other hardware.** Other CPUs or operating systems may also change last digits.

All analyses were verified on Windows 11 (x86-64) with the environment in `environment.yml`.

## Data

The participant-level input data are **not** included (see `data/README.md`):
- the depression and dietary metadata are available from OpenNeuro (ds004648);
- the processed taxonomy table was provided by the cohort investigators;
- the daily nutrient totals are available from the authors.

Place the three input files in one folder and set `DMDEP_DATA_DIR` to it.

The pipeline writes two kinds of output:
- **Participant-level intermediates** (predictions, split membership, regression input data) go to `DMDEP_WORK_DIR`
  (default `work/`, ignored by git) and should not be shared.
- **Aggregate results** go to `results/` and figures to `figures/`. `run_all.py` checks that no participant
  identifier or participant-level prediction ends up in `results/`.

## How to run (Windows PowerShell)

Open an Anaconda PowerShell Prompt (any PowerShell in which `conda` is available) in the repository folder and run:

```powershell
conda create -y -p .\env --file environment-win64-explicit.txt
$env:R_LIBS_USER = "$PWD\r-lib"
& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" R\install_packages.R r-lib
$env:DMDEP_RSCRIPT = "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
$env:DMDEP_DATA_DIR = "C:\path\to\input-data"
.\env\python.exe run_all.py
```

Adjust the Rscript path to your R installation and `DMDEP_DATA_DIR` to the folder with the input files.

The two environment files differ:
- `environment-win64-explicit.txt` lists the exact packages of the verified runs.
- `environment.yml` gives the same main package builds in readable form (`conda env create -p .\env -f
  environment.yml`). The solver may then choose newer versions of secondary packages, which could affect the figure
  files but not the analysis packages.

Optional environment variables:
- `DMDEP_WORK_DIR`: folder for participant-level intermediates (default `work`);
- `DMDEP_N_JOBS`: workers of the permutation tests (default: all cores minus two).

Interrupted runs:
- the two permutation steps are checkpointed, and rerunning `run_all.py` resumes them;
- every step can also be run alone: `.\env\python.exe scripts\<step>.py`.

The fixed parameters (seeds, thresholds, numbers of replicates) are in `src/dmdep/config.py`.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_prepare_data.py` | Outcome; microbiome filtering and CLR (249 genus-level taxa); energy adjustment and Spearman filter of the dietary variables (7 model inputs); confounders |
| 2 | `scripts/02_main_analysis.py` | Five repeated stratified splits (seeds 42–46; 36 feature-selection / 34 training / 18 test participants, 3 depressive + 15 control per test set); taxa selection on the feature-selection subset (ANOVA F-test p < 0.05, mutual information ≥ 95th percentile, random-forest importance > 1/n; kept if selected by ≥ 2 methods); 5 feature sets × LR / linear SVM / balanced random forest; Platt calibration inside the training subset |
| 3 | `scripts/03_permutation_test.py` | Max-statistic label-permutation test, complete pipeline repeated (B = 1,000, seed 20260); families of 15, 12 and 9 configurations |
| 4 | `scripts/04_bootstrap_calibration.py` | Stratified test-set bootstrap (B = 2,000, seed 2026): 95% CI of the mean ROC-AUC and paired comparisons; Brier scores and calibration bins |
| 5 | `scripts/05_feature_prioritization.py` | Median rule in the main configuration (LR with confounders, diet and taxa): retained if \|coefficient\| ≥ split median in ≥ 2 of 5 splits |
| 6 | `scripts/06_firth_regression.py` (+ `R/firth_logistf.R`) | Firth logistic regression per prioritised feature: unadjusted, adjusted for age, BMI, SQ_all and PA, leave-one-confounder-out; Benjamini–Hochberg q-values; reference family of 195 features |
| 7 | `scripts/07_residualized_analysis.py` | Sensitivity analysis: diet and taxa residualized on the confounders within each split (fitted on the feature-selection and training subsets) |
| 8 | `scripts/08_residualized_permutation.py` | Permutation test of the residualized pipeline (B = 1,000) |
| 9 | `scripts/09_prediction_level_adjustment.py` | Prediction-level confounder adjustment (Dinga et al., 2020) with a participant-level cluster bootstrap (B = 2,000, seed 2026) |
| 10 | `scripts/10_figures.py` | Figures |
| 11 | `scripts/11_format_tables.py` | Manuscript and Supplement tables at the printed precision |

The main configuration (LR with confounders, diet and taxa) was designated during the analysis for its
interpretability; it was not pre-specified. All 15 configurations are reported.

## Manuscript items and their outputs

| Manuscript item | Output file(s) | Script |
|---|---|---|
| Preprocessing numbers (850 → 422 → 421 → 249 taxa; ρ = 0.945; confounder correlations) | `results/preprocessing_summary.json`, `results/diet_spearman_matrix.csv` | 01 |
| Table 1 | `results/tables/table1.csv` | 02, 04, 11 |
| Table 2 | `results/tables/table2.csv` | 06, 11 |
| Figure 1 | `figures/figure1_roc_auc_per_split.pdf` | 10 |
| Figure 2 (a, b) | `figures/figure2a_roc_curve.pdf`, `figures/figure2b_confusion_matrix.pdf` | 10 |
| Figure 3 | `figures/figure3_lr_coefficients.pdf` | 10 |
| Figure 5 | `figures/figure5_workflow.pdf` (schematic, no data) | 10 |
| Supplementary Figure S1 | `figures/figureS1_diet_spearman_heatmap.pdf` | 10 |
| Supplementary Table S1 | `results/tables/tableS1.csv` | 05, 11 |
| Supplementary Table S8 | `results/tables/tableS8.csv` | 02, 11 |
| Supplementary Tables S9, S10 | `results/tables/tableS9a.csv`, `tableS9b.csv`, `tableS10a.csv`, `tableS10b.csv` | 04, 11 |
| Supplementary Table S11 | `results/tables/tableS11a.csv`, `tableS11b.csv` | 03, 11 |
| Supplementary Table S12 | `results/tables/tableS12a.csv`, `tableS12b.csv` | 04, 11 |
| Supplementary Table S13 | `results/tables/tableS13.csv` | 05, 11 |
| Supplementary Tables S14, S15, S17 | `results/tables/tableS14.csv`, `tableS15.csv` (+ `tableS15_smallest_q.csv`), `tableS17.csv` (+ `tableS17_summary.csv`) | 06, 11 |
| Supplementary Table S16 | `results/tables/tableS16.csv` | 05, 11 |
| Supplementary Table S18 | `results/tables/tableS18a.csv`, `tableS18b.csv` | 07, 08, 11 |
| Supplementary Table S19 | `results/tables/tableS19.csv` | 09, 11 |
| Supplementary Tables S2, S3 (SCFA logistic-regression models) | `notebooks/scfa/scfa_analysis.ipynb` | notebook |
| Supplementary Table S4 (SCFA group comparisons) | `notebooks/scfa/scfa_analysis.ipynb` | notebook |
| Figure 4, Supplementary Figure S2 (SCFA correlations) | `notebooks/scfa/scfa_analysis.ipynb` | notebook |
| Supplementary Tables S5–S7 (PICRUSt2 KO occurrence) | `notebooks/picrust2/PICRUSt2_9ASV_S5_S7.ipynb` | notebook |
| Supplementary Figure S3 (mediation: indirect effects, 42 pairs) | `mediation/01_family_P/forest_plot_ab_family_P.pdf` | `mediation/run_A_family_P.py` |
| Confusion counts in the Results text (4 of 15 depressive, 57 of 75 control) | `results/figures/figure_values.json` | 10 |

Other aggregate outputs (performance of every metric, ROC-AUC per split, taxa selection frequency, permutation null
distributions of the mean ROC-AUC, bootstrap intervals, Firth estimates of the reference family) are in the
subfolders of `results/`.

## Runtime

Two complete runs on the same 24-thread Windows workstation (22 permutation workers) took 37 and 83 minutes; the
difference reflects machine load.
- The main permutation test took 30–48 minutes.
- The residualized permutation test took 6–32 minutes.
- All other steps took between a few seconds and about two minutes.

Creating the environments takes a few minutes in addition.

## Seeds

- Split seeds: 42, 43, 44, 45, 46.
- Classifier and random-forest random state: 42.
- Inner Platt cross-validation: seed 42.
- Bootstraps: seed 2026.
- Permutations: seed (20260, index).

All seeds are fixed in `src/dmdep/config.py`.
