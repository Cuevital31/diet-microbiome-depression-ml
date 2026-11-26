"""run_1_copy_inputs.py - Revision-R2g: copy every input into Revision-R2g/inputs (original
folder layout mirrored) and the R2c-conf pipeline core, then verify each copy by SHA-256.
Read-only on the original folders. --resume: files already copied and identical are kept.
Writes 00_checks/check_inputs_copied.csv.
"""
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Clean-Codes
INPUTS = HERE / "inputs"
CHK = HERE / "00_checks"
MAIN_RUN = "MAIN_conf_Age_BMI_SQ_PA_249taxa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_plan():
    saved = (HERE / "plan_hash.txt").read_text(encoding="utf-8").split()[0]
    assert sha256(HERE / "ANALYSIS_PLAN_R2g.md") == saved, "ANALYSIS_PLAN_R2g.md changed: refusing to run"


def sources():
    files = [ROOT / "Preprocessed Files" / "participants_v3_corrected_depflag.tsv",
             ROOT / "Preprocessed Files" / "taxonomy_leaf_CLR_v2.xlsx",
             ROOT / "Preprocessed Files" / "taxonomy_completo_paper_v5.xlsx",
             ROOT / "Revision-R2c-conf" / "01_main" / MAIN_RUN / "run_config.json",
             ROOT / "Revision-R2d" / "results" / "family_D_M0.csv",
             ROOT / "Revision-R2d" / "results" / "family_D_M1.csv",
             ROOT / "Revision-R2d" / "results" / "sensitivity_rankZung_D_M1.csv"]
    files += sorted(p for p in (ROOT / "Revision-R2c-conf" / "inputs").rglob("*") if p.is_file())
    return files


def main():
    check_plan()
    resume = "--resume" in sys.argv
    rows = []
    for src in sources():
        dst = INPUTS / src.relative_to(ROOT)
        h_src = sha256(src)
        if not (resume and dst.exists() and sha256(dst) == h_src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        h_dst = sha256(dst)
        rows.append({"original": str(src.relative_to(ROOT)).replace("\\", "/"),
                     "copy": str(dst.relative_to(HERE)).replace("\\", "/"),
                     "sha256_original": h_src, "sha256_copy": h_dst, "identical": h_src == h_dst})
    core_src = ROOT / "Revision-R2c-conf" / "pipeline_core.py"
    core_dst = HERE / "pipeline_core_from_R2c-conf.py"
    if not (resume and core_dst.exists() and sha256(core_dst) == sha256(core_src)):
        shutil.copy2(core_src, core_dst)
    rows.append({"original": "Revision-R2c-conf/pipeline_core.py", "copy": core_dst.name,
                 "sha256_original": sha256(core_src), "sha256_copy": sha256(core_dst),
                 "identical": sha256(core_src) == sha256(core_dst)})
    CHK.mkdir(exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(CHK / "check_inputs_copied.csv", index=False)
    print(f"{len(t)} files copied; all identical: {bool(t['identical'].all())}")
    assert t["identical"].all()


if __name__ == "__main__":
    main()
