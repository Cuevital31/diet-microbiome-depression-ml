"""Output helpers. Aggregate results go to results/ (shareable); participant-level intermediates (predictions,
split membership, residualized data, regression input data) go to the work folder, which is not shared."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def results_path(*parts) -> Path:
    path = C.RESULTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def work_path(*parts) -> Path:
    path = C.WORK_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(type(value))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"rankID": str, "Feature ID": str, "participant_id": str})


def load_main_run() -> dict:
    keys = ("metrics", "predictions", "coefficients", "selection", "membership", "calibration")
    return {k: read_csv(work_path("main", f"{k}.csv")) for k in keys}
