"""Record a SHA-256 of every file of the code copy that git would commit, and optionally check
that nothing changed since an earlier record. The rerun runs on uncommitted code, so these
manifests are the record of exactly which code ran; at commit time the committed files are
checked against manifest_start.json.

Usage:
    python manifest.py --out manifest_start.json
    python manifest.py --out manifest_gate.json --compare manifest_start.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

WORKSPACE = Path("/workspace")
COPY_RELATIVE = "sycophancy-rl"


def file_hashes() -> dict[str, str]:
    listed = subprocess.run(
        ["git", "-C", str(WORKSPACE), "ls-files", "--cached", "--others", "--exclude-standard", "--", COPY_RELATIVE],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {path: hashlib.sha256((WORKSPACE / path).read_bytes()).hexdigest() for path in sorted(listed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None, help="an earlier manifest that must match exactly")
    args = parser.parse_args()

    hashes = file_hashes()
    result: dict = {"n_files": len(hashes), "files": hashes}
    if args.compare is not None:
        earlier = json.loads(args.compare.read_text())["files"]
        changed = sorted(path for path in hashes.keys() & earlier.keys() if hashes[path] != earlier[path])
        added = sorted(hashes.keys() - earlier.keys())
        removed = sorted(earlier.keys() - hashes.keys())
        result.update({"compared_to": str(args.compare), "changed": changed, "added": added, "removed": removed})
        result["PASS"] = not (changed or added or removed)
        print(f"compared with {args.compare}: changed {changed}, added {added}, removed {removed}")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(hashes)} files hashed{'' if args.compare is None else ('; PASS' if result['PASS'] else '; FAIL')}: wrote {args.out}")


if __name__ == "__main__":
    main()
