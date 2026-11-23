"""Checkpointed, parallel label-permutation loops. Every permutation has its own fixed seed
(PERM_BASE_SEED, index), so results do not depend on the number of workers or on interruptions."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from . import config as C

_COHORT = None


def _cohort():
    global _COHORT
    if _COHORT is None:
        from .data import load_cohort
        _COHORT = load_cohort()
    return _COHORT


def main_worker(index: int) -> dict:
    from .pipeline import mean_aucs, permuted_labels
    cohort = _cohort()
    res = mean_aucs(cohort, permuted_labels(cohort.y, index))
    return {"perm_index": index, **{C.config_label(c): v for c, v in res["mean_auc"].items()},
            "n_empty_consensus_splits": res["n_empty_consensus_splits"]}


def residualized_worker(index: int) -> dict:
    from .pipeline import mean_aucs, permuted_labels
    from .residualized import run_split_residualized
    cohort = _cohort()
    res = mean_aucs(cohort, permuted_labels(cohort.y, index), split_fn=run_split_residualized)
    return {"perm_index": index, **{C.config_label(c): v for c, v in res["mean_auc"].items()},
            "n_empty_consensus_splits": res["n_empty_consensus_splits"]}


def run_permutations(worker, checkpoint: Path, n_perm: int = C.PERM_B, n_jobs: int = C.N_JOBS) -> pd.DataFrame:
    from joblib import Parallel, delayed
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    design_file = checkpoint.with_name(checkpoint.stem + "_design.json")
    design = {"worker": worker.__name__, "n_perm": n_perm, "base_seed": C.PERM_BASE_SEED, "split_seeds": list(C.SEEDS)}
    done = pd.DataFrame()
    if checkpoint.exists():
        assert json.loads(design_file.read_text()) == design, f"{checkpoint} belongs to a different design"
        done = pd.read_csv(checkpoint)
    else:
        design_file.write_text(json.dumps(design))
    finished = set(done["perm_index"]) if len(done) else set()
    pending = [i for i in range(n_perm) if i not in finished]
    print(f"[{worker.__name__}] permutations done {len(finished)}, to run {len(pending)}", flush=True)
    chunk, started = max(10, 2 * n_jobs), time.time()
    for start in range(0, len(pending), chunk):
        rows = Parallel(n_jobs=n_jobs)(delayed(worker)(i) for i in pending[start:start + chunk])
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        tmp = checkpoint.with_suffix(".tmp")
        done.to_csv(tmp, index=False)
        os.replace(tmp, checkpoint)
        print(f"  {len(done)}/{n_perm} | {(time.time() - started) / 60:.1f} min elapsed", flush=True)
    done = done.sort_values("perm_index").reset_index(drop=True)
    assert done["perm_index"].tolist() == list(range(n_perm))
    return done
