"""Run the complete analysis in order (steps 01-11). Usage: python run_all.py

Inputs: see data/README.md (folder given by DMDEP_DATA_DIR). Participant-level intermediate files are written to
DMDEP_WORK_DIR (default ./work; not part of the repository); aggregate results to results/, figures to figures/.
The two permutation steps are checkpointed: if the run is interrupted, running this script again resumes them.
A log of every step is written to the work folder.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from dmdep import config as C  # noqa: E402

STEPS = ["01_prepare_data.py", "02_main_analysis.py", "03_permutation_test.py", "04_bootstrap_calibration.py",
         "05_feature_prioritization.py", "06_firth_regression.py", "07_residualized_analysis.py",
         "08_residualized_permutation.py", "09_prediction_level_adjustment.py", "10_figures.py", "11_format_tables.py"]
PARTICIPANT_LEVEL_COLUMNS = {"participant_id", "y_true", "y_pred", "y_score", "prob_native", "prob_platt"}


def check_results_are_aggregate():
    """No file in results/ may contain participant identifiers or participant-level predictions."""
    import pandas as pd
    for path in C.RESULTS_DIR.rglob("*"):
        if path.suffix == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
            assert not columns & PARTICIPANT_LEVEL_COLUMNS, f"participant-level columns in {path}"
        if path.suffix in (".csv", ".json"):
            assert "sub-" not in path.read_text(encoding="utf-8"), f"participant identifier in {path}"


def main():
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    log = C.WORK_DIR / "run_all.log"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))}
    started = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        for step in STEPS:
            t0 = time.time()
            print(f"=== {step}", flush=True)
            fh.write(f"\n=== {step}\n")
            fh.flush()
            proc = subprocess.run([sys.executable, str(REPO / "scripts" / step)], cwd=REPO, env=env,
                                  stdout=fh, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"{step} failed (exit {proc.returncode}); see {log}")
                sys.exit(proc.returncode)
            print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)
            fh.write(f"=== {step} done in {time.time() - t0:.1f} s\n")
    check_results_are_aggregate()
    print(f"all steps finished in {(time.time() - started) / 60:.1f} min; results in {C.RESULTS_DIR.name}/, "
          f"figures in {C.FIGURES_DIR.name}/")


if __name__ == "__main__":
    main()
