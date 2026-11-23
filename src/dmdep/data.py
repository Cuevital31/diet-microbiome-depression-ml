"""Loading and preprocessing of the three input files.

* Outcome: Zung depression index (raw score x 1.25, as provided) >= 50 -> depressive.
* Confounders: Age, BMI, SQ_all (natural logarithm of the sympathovagal quotient SQ), PA.
* Microbiome: kingdom Bacteria; samples with < 5,000 reads excluded; entries detected in >= 9 participants
  and with a total count >= 10 kept; internal nodes recomputed from their remaining leaves; CLR with
  pseudocount 0.5 over all rows of the filtered hierarchy; genus-level leaves used as features.
* Diet: nutrients expressed relative to reported energy intake (% of energy with 4 kcal/g for protein and
  carbohydrate and 9 kcal/g for fats; cholesterol and fibre per 1,000 kcal); redundant variables removed by a
  Spearman |rho| >= 0.90 rule computed once on all participants without the outcome.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class MissingInputError(FileNotFoundError):
    pass


def _input(name: str) -> Path:
    path = C.DATA_DIR / name
    if not path.exists():
        raise MissingInputError(
            f"Required input file not found: {path}\n"
            f"Set DMDEP_DATA_DIR to the folder that contains {C.PARTICIPANTS_FILE}, {C.TAXONOMY_FILE} and "
            f"{C.DIET_FILE} (see data/README.md for how to obtain them and their expected layout).")
    return path


# ----------------------------------------------------------------------------- participants
def load_participants() -> pd.DataFrame:
    table = pd.read_csv(_input(C.PARTICIPANTS_FILE), sep="\t")
    needed = ["participant_id", "Zung_S", "SQ", *C.CONFOUNDERS]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(f"{C.PARTICIPANTS_FILE} lacks columns {missing}")
    table["participant_id"] = table["participant_id"].astype(str)
    assert table["participant_id"].is_unique and len(table) == C.N_PARTICIPANTS, "expected 88 unique participants"
    assert table[needed].notna().all().all(), "missing values in the participant table"
    # SQ_all is the natural logarithm of the SQ column of the cohort dataset
    assert np.allclose(np.log(table["SQ"].to_numpy(float)), table["SQ_all"].to_numpy(float), rtol=0, atol=1e-12)
    return table


def outcome(participants: pd.DataFrame):
    zung = participants["Zung_S"].to_numpy(float)
    y = (zung >= C.ZUNG_CUTOFF).astype(int)
    assert np.array_equal(y, (zung > C.ZUNG_CUTOFF).astype(int)), "an index score equals 50: >= and > differ"
    assert int(y.sum()) == C.N_DEPRESSIVE and int((y == 0).sum()) == C.N_CONTROL
    return y, zung


# ----------------------------------------------------------------------------- microbiome
def preprocess_taxonomy(participant_ids) -> dict:
    """Filtered hierarchy -> CLR -> genus-level leaves. Returns matrices aligned to participant_ids."""
    raw = pd.read_excel(_input(C.TAXONOMY_FILE), dtype={"rankID": str})
    meta_cols = ["taxlevel", "rankID", "taxon", "daughterlevels", "total"]
    sample_cols = [c for c in raw.columns if str(c).startswith(C.SAMPLE_PREFIX)]
    ids = [c.replace(C.SAMPLE_PREFIX, C.PARTICIPANT_PREFIX, 1) for c in sample_cols]
    assert sorted(ids) == sorted(participant_ids), "taxonomy samples differ from the participant table"
    order = [sample_cols[ids.index(p)] for p in participant_ids]

    kingdom_mask = (raw["rankID"] == C.KINGDOM_RANK_ID) | raw["rankID"].str.startswith(C.KINGDOM_RANK_ID + ".")
    table = raw.loc[kingdom_mask, meta_cols + order].reset_index(drop=True)
    n_entries = len(table)
    depth = table.loc[table["rankID"] == C.KINGDOM_RANK_ID, order].iloc[0].astype(float)
    assert (depth >= C.MIN_READS_PER_SAMPLE).all(), "a sample has fewer reads than the minimum depth"

    counts = table[order].to_numpy(float)
    prevalence = (counts > 0).sum(axis=1)
    table = table[prevalence >= C.MIN_PREVALENCE].reset_index(drop=True)
    n_after_prevalence = len(table)
    table = table[table["total"].astype(float) >= C.MIN_TOTAL_COUNT].reset_index(drop=True)
    n_after_lowcount = len(table)

    # internal nodes = sum of their remaining leaf descendants
    table["daughterlevels"] = table["daughterlevels"].fillna(0).astype(int)
    leaf = table["daughterlevels"] == 0
    leaves = table[leaf]
    for idx in table.index[~leaf]:
        desc = leaves.loc[leaves["rankID"].str.startswith(table.at[idx, "rankID"] + "."), order]
        table.loc[idx, order] = desc.sum(axis=0).to_numpy(float) if len(desc) else 0.0

    values = table[order].to_numpy(float)
    # CLR per participant over all rows of the filtered hierarchy. The reported results were computed from a CLR
    # table stored with 16 significant digits; the values are rounded in the same way here. This matters because
    # the nearest-neighbour mutual-information estimator used in the taxa selection is sensitive to differences
    # in the last digits of its input (without the rounding, the selected taxa differ in 4 of the 5 splits).
    log_t = np.log((values + C.CLR_PSEUDOCOUNT).T)
    clr = (log_t - log_t.mean(axis=1, keepdims=True)).T
    clr = np.array([[float(f"{v:.16g}") for v in row] for row in clr])
    leaf = (table["daughterlevels"] == 0).to_numpy()
    rank_ids = table["rankID"].astype(str).tolist()
    names = dict(zip(rank_ids, table["taxon"].astype(str)))
    leaf_ids = [r for r, is_leaf in zip(rank_ids, leaf) if is_leaf]
    return {
        "taxa_ids": tuple(leaf_ids),
        "taxon_names": {r: names[r] for r in leaf_ids},
        "family_names": {r: names.get(r.rsplit(".", 1)[0], "") for r in leaf_ids},
        "x_taxa": clr[leaf].T,                              # participants x taxa
        "leaf_counts": values[leaf].T,                      # participants x taxa (raw counts)
        "summary": {"n_entries_kingdom_to_genus": n_entries, "n_after_prevalence_filter": n_after_prevalence,
                    "n_after_low_count_filter": n_after_lowcount, "n_genus_level_taxa": int(leaf.sum()),
                    "taxonomic_levels": sorted(int(v) for v in raw.loc[kingdom_mask, "taxlevel"].unique()),
                    "min_reads_per_sample": int(depth.min()), "n_samples_below_min_reads": 0},
    }


# ----------------------------------------------------------------------------- diet
def nutrient_density(diet_raw: pd.DataFrame) -> pd.DataFrame:
    energy = diet_raw["Kcal"].astype(float)
    out = pd.DataFrame(index=diet_raw.index)
    for nutrient in C.NUTRIENTS:
        if nutrient in C.KCAL_PER_GRAM:
            out[C.DIET_NAMES[nutrient]] = 100.0 * C.KCAL_PER_GRAM[nutrient] * diet_raw[nutrient] / energy
        else:
            out[C.DIET_NAMES[nutrient]] = 1000.0 * diet_raw[nutrient] / energy
    return out


def spearman_filter(table: pd.DataFrame, threshold: float = C.CORR_THRESHOLD) -> dict:
    """Repeatedly drop the variable involved in most pairs with |rho| >= threshold (ties: larger mean |rho|
    with the remaining variables, then column order) until no such pair is left."""
    corr = table.corr(method="spearman")
    remaining, removed, pairs_seen = list(table.columns), [], []
    while True:
        sub = corr.loc[remaining, remaining].abs()
        high = (sub.to_numpy() >= threshold) & ~np.eye(len(remaining), dtype=bool)
        counts = high.sum(axis=1)
        if counts.max() == 0:
            break
        pairs_seen += [(remaining[i], remaining[j], float(corr.loc[remaining[i], remaining[j]]))
                       for i in range(len(remaining)) for j in range(i + 1, len(remaining)) if high[i, j]]
        mean_abs = (sub.to_numpy().sum(axis=1) - 1.0) / (len(remaining) - 1)
        candidates = np.flatnonzero(counts == counts.max())
        removed.append(remaining.pop(int(candidates[np.argmax(mean_abs[candidates])])))
    return {"corr": corr, "retained": tuple(remaining), "removed": tuple(removed), "pairs": pairs_seen}


def load_diet(participant_ids) -> dict:
    diet = pd.read_csv(_input(C.DIET_FILE))
    diet["participant_id"] = diet["participant_id"].astype(str)
    assert sorted(diet["participant_id"]) == sorted(participant_ids), "diet table participants differ"
    diet = diet.set_index("participant_id").loc[list(participant_ids), list(C.DIET_RAW)].astype(float)
    assert diet.notna().all().all() and (diet > 0).all().all(), "missing or non-positive dietary values"
    density = nutrient_density(diet)
    screen = density.copy()
    screen["Kcal"] = diet["Kcal"].to_numpy()
    result = spearman_filter(screen)
    retained = [c for c in result["retained"] if c != "Kcal"]
    return {"table": density[retained], "density": density, "screen_corr": result["corr"],
            "removed": [c for c in result["removed"] if c != "Kcal"], "pairs": result["pairs"]}


# ----------------------------------------------------------------------------- cohort
@dataclass(frozen=True)
class Cohort:
    ids: np.ndarray
    y: np.ndarray
    zung: np.ndarray
    x_taxa: np.ndarray
    taxa_ids: tuple
    taxon_names: dict
    family_names: dict
    leaf_counts: np.ndarray
    diet: pd.DataFrame            # 7 energy-adjusted model inputs (rows = participants)
    diet_info: dict
    confounders: pd.DataFrame     # Age, BMI, SQ_all, PA
    taxa_summary: dict

    @property
    def diet_names(self):
        return list(self.diet.columns)


def load_cohort() -> Cohort:
    participants = load_participants()
    ids = participants["participant_id"].to_numpy()
    y, zung = outcome(participants)
    taxa = preprocess_taxonomy(list(ids))
    diet = load_diet(list(ids))
    conf = participants.set_index("participant_id").loc[list(ids), list(C.CONFOUNDERS)].astype(float)
    # The reported results used the confounder values after one CSV write/read with pandas' default float parser,
    # which changes some values in the last digit (about 1e-15). That representation is reproduced here, because
    # the mutual-information taxa selection of the residualized sensitivity analysis is sensitive to it.
    conf = pd.read_csv(io.StringIO(conf.to_csv()), index_col=0)
    conf.index = conf.index.astype(str)
    return Cohort(ids=ids, y=y, zung=zung, x_taxa=taxa["x_taxa"], taxa_ids=taxa["taxa_ids"],
                  taxon_names=taxa["taxon_names"], family_names=taxa["family_names"],
                  leaf_counts=taxa["leaf_counts"], diet=diet["table"], diet_info=diet,
                  confounders=conf, taxa_summary=taxa["summary"])


def taxon_label(cohort: Cohort, taxon_id: str) -> str:
    """'Genus (family); feature ID'; a family placeholder 'X_unclassified' is written as
    'order X, family unclassified'."""
    family = cohort.family_names[taxon_id]
    if family.endswith("_unclassified"):
        family = f"order {family[: -len('_unclassified')]}, family unclassified"
    return f"{cohort.taxon_names[taxon_id]} ({family}); {taxon_id}"
