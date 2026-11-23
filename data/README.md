# Input data (not included)

The analysis uses participant-level data of the cross-sectional cohort of Miranda-Angulo et al. (2024): 88 Colombian
men aged 21–40 years. These data are **not** distributed with this repository. Authorized users place the three files
below in one folder and point the environment variable `DMDEP_DATA_DIR` to it (default: `data/input/`, which is
ignored by git). If a file is missing, the code stops with a message naming it.

| File | Content | Source |
|---|---|---|
| `participants.tsv` | Participant metadata, one row per participant (tab-separated) | OpenNeuro dataset ds004648 (participant-level depression and dietary metadata) |
| `taxonomy_counts.xlsx` | Hierarchical taxonomic count table (16S rRNA, V3–V4; Mothur, SILVA v138.1) | Processed taxonomy table provided by the cohort investigators (Ana L. Miranda-Angulo) |
| `diet_daily_totals.csv` | Daily nutrient totals from the semi-quantitative food-frequency questionnaire, one row per participant | Daily totals computed from the questionnaire answers; available from the authors on request |

## Expected layout

### `participants.tsv`

Columns used (other columns are ignored):

| Column | Meaning |
|---|---|
| `participant_id` | Identifier of the form `sub-MIC<code>` (88 unique values) |
| `Zung_S` | Zung Self-Rating Depression Scale index score (raw score × 1.25, as provided); depressive if ≥ 50 |
| `Age` | Age (years) |
| `BMI` | Body mass index (kg/m²) |
| `SQ` | Sympathovagal quotient as provided in the cohort dataset |
| `SQ_all` | Natural logarithm of `SQ` (checked by the code) |
| `PA` | Physical activity (ordinal code) |

### `taxonomy_counts.xlsx`

First sheet. One row per taxonomic entry of the hierarchy (root to genus), with the columns `taxlevel`, `rankID`
(e.g. `0.2.10.2.2.11.26`; `0.2` = Bacteria), `taxon`, `daughterlevels` (0 for leaves), `total`, and one count column
per sample named `c16S<code>`. The sample `c16SMIC<code>` belongs to participant `sub-MIC<code>`. All 88 participants must be
present.

### `diet_daily_totals.csv`

Comma-separated; columns `participant_id`, `Kcal` (energy, kcal/day), `Prot`, `TF` (total fat), `SatFat`, `MUF`
(monounsaturated fat), `Poly_InsFat` (polyunsaturated fat), `Carb`, `Fiber` (all g/day) and `Chol` (mg/day). All
values must be positive.

## What the code derives

- **Outcome.** Depressive status = Zung index ≥ 50: 15 depressive and 73 control participants. No participant scores
  exactly 50.
- **Microbiome.**
  - Kingdom Bacteria only. Samples with fewer than 5,000 reads would be excluded; none are.
  - Entries detected in at least 9 participants and with a total count of at least 10 are kept.
  - Internal nodes are recomputed from their remaining leaves.
  - CLR transformation with pseudocount 0.5 over all rows of the filtered hierarchy; the 249 genus-level leaves are
    the features.
- **Diet.**
  - Protein and carbohydrate (4 kcal/g) and the four fat variables (9 kcal/g) are expressed as percentages of energy
    (`*_PE`); cholesterol and fibre per 1,000 kcal (`*_PK`).
  - Total fat (`TF_PE`) is removed by the Spearman |ρ| ≥ 0.90 rule, leaving the 7 model inputs `Prot_PE`,
    `SatFat_PE`, `MUF_PE`, `Poly_InsFat_PE`, `Chol_PK`, `Carb_PE` and `Fiber_PK`.
- **Confounders.** `Age`, `BMI`, `SQ_all`, `PA`.

## Data availability

The participant-level depression and dietary metadata are available from OpenNeuro (ds004648). The processed
taxonomy table was provided directly by the cohort investigators. Redistribution of these data is not covered by this
repository; please contact the authors.
