"""Check that the code copy is the release plus the sycophancy commit and nothing else.

The copy at /workspace/sycophancy-rl started as the release's tracked files
(/workspace/probes-as-training-signals) minus seven public-release files. The training
package then received private commit 12631d5 (the MULTIPLE_CHOICE task) by three-way merge,
and the launcher (training/) got sycophancy support. This script checks:

  1. the copy's file set is the release's tracked files minus EXCLUDED_FILES plus NEW_FILES;
  2. NEW_FILES are byte-identical to their 12631d5 versions;
  3. every file outside PORTED_FILES and LAUNCHER_FILES is byte-identical to the release;
  4. in PORTED_FILES, the changes release -> copy equal the changes 7c9b848 -> 12631d5
     (the commit's parent -> the commit), hunk for hunk with line numbers ignored, except for
     the hand-merged hunks listed in expected_diff_hunks.json, which must differ exactly as
     listed.

Usage:
    python check_diff.py [--out check_diff.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("/workspace")
RELEASE = WORKSPACE / "probes-as-training-signals"
COPY = WORKSPACE / "sycophancy-rl"
FORK_COMMIT = "7c9b848"  # parent of the sycophancy commit; the release started from this code
SYCOPHANCY_COMMIT = "12631d5"

EXCLUDED_FILES = {
    "README.md",
    "NOTICE",
    "LICENSE",
    "obfuscation-atlas/README.md",
    "obfuscation-atlas/Makefile",
    "obfuscation-atlas/pyproject.toml",
    "obfuscation-atlas/uv.lock",
}
NEW_FILES = {
    "obfuscation-atlas/obfuscation_atlas/tasks/multiple_choice.py",
    "obfuscation-atlas/tests/test_multiple_choice_task.py",
}
PORTED_FILES = [
    "obfuscation-atlas/obfuscation_atlas/config.py",
    "obfuscation-atlas/obfuscation_atlas/detectors/custom_dataset.py",
    "obfuscation-atlas/obfuscation_atlas/scripts/train_grpo_model_against_probe.py",
    "obfuscation-atlas/obfuscation_atlas/training/grpo_trainer.py",
    "obfuscation-atlas/obfuscation_atlas/training/obfuscation_training.py",
]
LAUNCHER_FILES = {
    "training/common.py",
    "training/grpo_launcher.py",
    "training/run_grpo.py",
    "training/extract_run_results.py",
}


def run(command: list[str], *, check: bool = True) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=check).stdout


def copy_file_set() -> set[str]:
    """Files git would commit under sycophancy-rl/ (tracked or untracked, ignored ones left out)."""
    listed = run(["git", "-C", str(WORKSPACE), "ls-files", "--cached", "--others", "--exclude-standard", "--", "sycophancy-rl"])
    return {line.removeprefix("sycophancy-rl/") for line in listed.splitlines()}


def hunks(unified_diff: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """(removed lines, added lines) of each hunk of a -U0 unified diff, line numbers dropped."""
    result = []
    current = None
    for line in unified_diff.splitlines():
        if line.startswith("@@"):
            current = ([], [])
            result.append(current)
        elif current is not None and line.startswith("-") and not line.startswith("---"):
            current[0].append(line[1:])
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            current[1].append(line[1:])
    return [(tuple(removed), tuple(added)) for removed, added in result]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "check_diff.json")
    args = parser.parse_args()

    release_files = set(run(["git", "-C", str(RELEASE), "ls-files"]).split())
    assert run(["git", "-C", str(RELEASE), "status", "--porcelain", "--untracked-files=no"]) == "", "release has uncommitted changes"
    copy_files = copy_file_set()
    expected_files = (release_files - EXCLUDED_FILES) | NEW_FILES
    file_set_ok = copy_files == expected_files
    print(f"file set: copy {len(copy_files)}, expected {len(expected_files)}, equal={file_set_ok}")
    if not file_set_ok:
        print(f"  missing from copy: {sorted(expected_files - copy_files)}")
        print(f"  unexpected in copy: {sorted(copy_files - expected_files)}")

    new_files_identical = {
        path: (COPY / path).read_bytes()
        == subprocess.run(["git", "-C", str(WORKSPACE), "show", f"{SYCOPHANCY_COMMIT}:{path}"], capture_output=True, check=True).stdout
        for path in sorted(NEW_FILES)
    }
    print(f"new files identical to {SYCOPHANCY_COMMIT}: {new_files_identical}")

    other_files = sorted(copy_files - NEW_FILES - set(PORTED_FILES) - LAUNCHER_FILES)
    changed_other_files = [path for path in other_files if (COPY / path).read_bytes() != (RELEASE / path).read_bytes()]
    print(f"other files identical to the release: {len(other_files) - len(changed_other_files)}/{len(other_files)}")
    for path in changed_other_files:
        print(f"  CHANGED: {path}")

    unmatched = []
    per_file = {}
    for path in PORTED_FILES:
        original = hunks(run(["git", "-C", str(WORKSPACE), "diff", "-U0", FORK_COMMIT, SYCOPHANCY_COMMIT, "--", path]))
        port = hunks(run(["git", "diff", "--no-index", "-U0", str(RELEASE / path), str(COPY / path)], check=False))
        original_counts, port_counts = collections.Counter(original), collections.Counter(port)
        per_file[path] = {
            "original_hunks": len(original),
            "port_hunks": len(port),
            "matched_hunks": sum((original_counts & port_counts).values()),
        }
        print(f"{path}: {per_file[path]}")
        for side, counts in (("original_only", original_counts - port_counts), ("port_only", port_counts - original_counts)):
            for (removed, added), count in counts.items():
                for _ in range(count):
                    unmatched.append({"file": path, "side": side, "removed": list(removed), "added": list(added)})

    expected_unmatched = json.loads((RUN_DIR / "expected_diff_hunks.json").read_text())

    def key(hunk: dict) -> str:
        return json.dumps(hunk, sort_keys=True)

    unexpected = [hunk for hunk in unmatched if key(hunk) not in {key(h) for h in expected_unmatched}]
    missing_expected = [hunk for hunk in expected_unmatched if key(hunk) not in {key(h) for h in unmatched}]
    print(f"unmatched hunks: {len(unmatched)} (expected {len(expected_unmatched)}); unexpected {len(unexpected)}, expected but absent {len(missing_expected)}")
    for hunk in unexpected:
        print(f"  UNEXPECTED {hunk['side']} in {hunk['file']}: -{len(hunk['removed'])} +{len(hunk['added'])} {(hunk['added'] or hunk['removed'])[0][:100]!r}")

    passed = (
        file_set_ok
        and all(new_files_identical.values())
        and not changed_other_files
        and not unexpected
        and not missing_expected
        and len(unmatched) == len(expected_unmatched)
    )
    result = {
        "PASS": passed,
        "file_set_ok": file_set_ok,
        "n_copy_files": len(copy_files),
        "new_files_identical": new_files_identical,
        "other_files_changed": changed_other_files,
        "per_ported_file": per_file,
        "unexpected_hunks": unexpected,
        "expected_hunks_absent": missing_expected,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {args.out}")


if __name__ == "__main__":
    main()
