# SCFA analyses

`scfa_analysis.ipynb` analyses the measured short-chain fatty acids (acetic, propionic, butyric and isobutyric acid)
of the 88 participants (SCFA values available for 87).

| Notebook section | Manuscript item |
|---|---|
| Merging of SCFA and clinical data; Mann–Whitney comparisons (BH-adjusted) | Supplementary Table S4 |
| Logistic regression with confounders, diet and taxa, plus each SCFA and all four SCFAs; paired bootstrap of the ROC-AUC differences (B = 2,000, seed 2026) | Supplementary Tables S2, S3 |
| Spearman and confounder-adjusted (partial) Spearman correlations of the SCFAs with the prioritised taxa, diet and confounders | Figure 4, Supplementary Figure S2 |
| Manuscript cross-check | assertions that stop the notebook if a reported value is not reproduced |

Run the sections from top to bottom.

## Inputs (not included)

| File | Content | Source |
|---|---|---|
| `New SCFA Walter.xlsx` | Measured SCFA concentrations (column `Code` = participant code) | Zenodo record 23045748 |
| `Colombian-metadata.xlsx` (sheets `Clinical-data`, `Meta-data`) | Participant-level clinical and sequencing metadata | Cohort investigators; clinical variables are in OpenNeuro ds004648 |
| `participants_v3_corrected_depflag.tsv`, `taxonomy_leaf_CLR_v2.xlsx`, `diet_new_88.csv`, `confounders_88.csv` | The participant-level inputs of the main pipeline | See `../../data/README.md` |
| Main-run folder `MAIN_conf_Age_BMI_SQ_PA_249taxa` (`run_config.json`, `taxa_selection.csv`, `split_membership.csv`, `metrics_per_split.csv`) | Split membership and consensus-selected taxa of the main run | Intermediates of the main pipeline; participant-level, not shared |

## Paths

- The project root is read from `COLOMBIA_PROJECT_ROOT` (default `/content/drive/MyDrive/project`) in the first and the
  correlation sections.
- The logistic-regression section sets `PROJECT = Path('/content/drive/MyDrive/project')` directly; change this line
  to the same folder.
- The expected sub-folders (`NEW/inputs/`, `NEW/Revision-R2c-conf/...`, `picrust2/...`) follow the layout of the
  original analysis. Adapt `PART`, `TAXA`, `DIET`, `CONF` and `RUN` if your files are elsewhere.
- Output goes to `picrust2/SCFA_analysis/` below the project root.

## Packages

Python with numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib and openpyxl.

## Notes

- The SCFA units are not given in the source file.
- The SCFA models are exploratory post-hoc additions: with 3 depressive participants per test set the ROC-AUC
  differences are not interpretable beyond the reported intervals.
