"""Gate before the 128-step rerun: every earlier check must have passed in this run.

Each check file must exist, say PASS, and be newer than manifest_start.json (so a file left
over from a dry run or an earlier attempt cannot pass the gate). check_august.json must cover
all 128 of August's step files, and check_responses_smoke.json both smoke-run files.

Usage:
    python gate.py            # writes gate.json; exits 1 unless every check passed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
CHECK_FILES = [
    "check_diff.json",
    "check_tests.json",
    "check_config.json",
    "check_august.json",
    "check_responses_smoke.json",
    "check_smoke.json",
    "manifest_gate.json",
]


def main() -> int:
    start_time = (RUN_DIR / "manifest_start.json").stat().st_mtime
    status = {}
    for name in CHECK_FILES:
        path = RUN_DIR / name
        assert path.exists(), f"missing {name}"
        data = json.loads(path.read_text())
        fresh = path.stat().st_mtime >= start_time
        status[name] = {"PASS": data["PASS"] is True, "written_after_manifest_start": fresh}
    august = json.loads((RUN_DIR / "check_august.json").read_text())
    smoke = json.loads((RUN_DIR / "check_responses_smoke.json").read_text())
    coverage = {
        "august_files_checked": august["n_files_checked"],
        "august_files_total": august["n_files"],
        "smoke_files_checked": smoke["n_files_checked"],
        "smoke_files_total": smoke["n_files"],
    }
    coverage_ok = coverage["august_files_checked"] == coverage["august_files_total"] == 128 and (
        coverage["smoke_files_checked"] == coverage["smoke_files_total"] == 2
    )
    passed = coverage_ok and all(entry["PASS"] and entry["written_after_manifest_start"] for entry in status.values())
    for name, entry in status.items():
        print(f"{name}: {entry}")
    print(f"coverage: {coverage}")
    (RUN_DIR / "gate.json").write_text(json.dumps({"PASS": passed, "checks": status, "coverage": coverage}, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {RUN_DIR / 'gate.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
