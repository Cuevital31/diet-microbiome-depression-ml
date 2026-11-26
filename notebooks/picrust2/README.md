# PICRUSt2: nine-ASV KO analysis

`PICRUSt2_9ASV_S5_S7.ipynb` reproduces Supplementary Tables S5–S7 (predicted KEGG orthologue occurrence for the nine
ASVs assigned to the three prioritised taxa: *Lachnospiraceae* ND3007 group, 4 ASVs; *Dielma*, 3; unclassified
*Rhizobiales*, 2).

The notebook does **not** rerun PICRUSt2. It reads the existing Nephele/PICRUSt2 predictions, counts a KO as present
when its predicted abundance is > 0, writes the S5, S6 and S7 tables as CSV files, and stops with an error if a count
differs from the one in the manuscript.

## Inputs

| File | Included here | Content |
|---|---|---|
| `informative_taxa_KO_predicted.csv` | yes | Predicted KO abundances per ASV (output of PICRUSt2 via Nephele) |
| `ASV_dna_sequences_taxonomy.xlsx` (sheet `ASV_sequences`, 6,144 ASVs) | no | ASV sequences and taxonomy; the original file also holds per-sample abundance columns |
| `pre_aligned_sequences.fasta` | no (optional) | Checks the exact sequence-to-ASV matching |
| `PipelineResults_5677a27dd332.tar.gz` | no (optional, about 700 MB) | Original Nephele output; only used to verify the sample-weighted NSTI |

The taxonomy file comes from the cohort investigators. The notebook only uses the columns `DNA sequence`, `Order` and
`Genus`.

## Running

Put the input files in one folder and set it as `PICRUST2_BASE` (default
`/content/drive/MyDrive/project/picrust2`). Output goes to `PICRUSt2_9ASV_manuscript_outputs/` in that folder
(`PICRUST2_OUTPUT` overrides it). Individual files can be given with `PICRUST2_KO_CSV`, `PICRUST2_TAXONOMY_XLSX`,
`PICRUST2_FASTA` and `PICRUST2_NEPHELE_ARCHIVE`.

Packages: numpy, pandas and openpyxl.
