"""run_0_checksums.py - Revision-R2g: plan hash and SHA-256 of every file in the protected folders.

Usage:  python run_0_checksums.py plan     -> plan_hash.txt (only if it does not exist yet)
        python run_0_checksums.py before   -> 00_checks/protected_checksums_before.csv
        python run_0_checksums.py after    -> 00_checks/protected_checksums_after.csv
                                              and 00_checks/check_protected_folders_unchanged.csv
Read-only on the protected folders.
"""
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTECTED = ["Revision-R2b", "Revision-R2c", "Revision-R2b-conf", "Revision-R2c-conf",
             "Revision-R2d", "Revision-R2e", "Revision-R2f"]
CHK = HERE / "00_checks"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table() -> pd.DataFrame:
    rows = []
    for folder in PROTECTED:
        base = ROOT / folder
        assert base.is_dir(), f"missing protected folder {base}"
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rows.append({"folder": folder, "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                             "bytes": p.stat().st_size, "sha256": sha256(p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ("plan", "before", "after")
    if mode == "plan":
        out = HERE / "plan_hash.txt"
        assert not out.exists(), "plan_hash.txt already exists: the plan hash is written only once"
        h = sha256(HERE / "ANALYSIS_PLAN_R2g.md")
        out.write_text(f"{h}  ANALYSIS_PLAN_R2g.md  written {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                       encoding="utf-8")
        print(out.read_text(encoding="utf-8").strip())
        sys.exit(0)
    CHK.mkdir(exist_ok=True)
    t = table()
    t.to_csv(CHK / f"protected_checksums_{mode}.csv", index=False)
    print(f"{mode}: {len(t)} files hashed")
    if mode == "after":
        b = pd.read_csv(CHK / "protected_checksums_before.csv")
        m = b.merge(t, on="file", how="outer", suffixes=("_before", "_after"), indicator=True)
        summary = []
        for folder in PROTECTED:
            sub = m[m["file"].str.split("/").str[0] == folder]
            missing = int((sub["_merge"] == "left_only").sum())
            added = int((sub["_merge"] == "right_only").sum())
            both = sub[sub["_merge"] == "both"]
            changed = int((both["sha256_before"] != both["sha256_after"]).sum())
            summary.append({"folder": folder, "files_before": int((sub["_merge"] != "right_only").sum()),
                            "files_after": int((sub["_merge"] != "left_only").sum()),
                            "files_missing_after": missing, "files_added_after": added,
                            "files_changed": changed,
                            "identical": missing == 0 and added == 0 and changed == 0})
        s = pd.DataFrame(summary)
        s.to_csv(CHK / "check_protected_folders_unchanged.csv", index=False)
        print(s.to_string(index=False))
        assert s["identical"].all(), "PROTECTED FOLDERS CHANGED"
