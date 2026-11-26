# Notebooks: SCFA and PICRUSt2 analyses

Two Jupyter/Colab notebooks complement the main pipeline (`scripts/`, `run_all.py`). They were written by Razan
Joumaa and are included **unchanged**.

| Folder | Manuscript items | Notebook |
|---|---|---|
| `scfa/` | Supplementary Tables S2–S4, Figure 4, Supplementary Figure S2 | `scfa_analysis.ipynb` |
| `picrust2/` | Supplementary Tables S5–S7 | `PICRUSt2_9ASV_S5_S7.ipynb` |

General notes:
- The notebooks were developed in Google Colab with the project on Google Drive. The default paths
  (`/content/drive/MyDrive/project/...`) therefore have to be adapted to your own folder; the individual READMEs say
  where.
- Participant-level input files are **not** included in this repository (see `../data/README.md`). The READMEs list
  where each input comes from.
- The SCFA notebook additionally needs the intermediate files of the main pipeline (`work/`), which are
  participant-level and are not shared.
