"""Run the ported sycophancy task test in the code copy and record the outcome.

tests/test_multiple_choice_task.py checks the answer parser and the cue rendering
byte-for-byte against the domain baseline's files (260819). Run with run_env.sh sourced, so
the copy's package is the one imported (the script asserts it).

Usage:
    python check_tests.py [--out check_tests.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
COPY_ATLAS = Path("/workspace/sycophancy-rl/obfuscation-atlas")
TEST_FILE = "tests/test_multiple_choice_task.py"
EXPECTED_TESTS = 9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "check_tests.json")
    args = parser.parse_args()

    imported_from = subprocess.run(
        [sys.executable, "-c", "import obfuscation_atlas; print(obfuscation_atlas.__file__)"],
        cwd=COPY_ATLAS, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert imported_from.startswith(str(COPY_ATLAS)), f"obfuscation_atlas imported from {imported_from}, not the copy"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST_FILE],
        cwd=COPY_ATLAS, capture_output=True, text=True,
    )
    summary = completed.stdout.strip().splitlines()[-1]
    print(completed.stdout[-2000:])
    passed = completed.returncode == 0 and summary.startswith(f"{EXPECTED_TESTS} passed")
    result = {"PASS": passed, "exit_code": completed.returncode, "summary": summary, "imported_from": imported_from}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {args.out}")


if __name__ == "__main__":
    main()
