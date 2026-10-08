"""Resumable single-file JSONL output, shared by the two generators.

Records are appended after every batch. A re-run loads the file, skips the keys
already present and, at the end, asserts the key set equals the expected set
(260819's finalize, without the per-category cache dirs). The file is rewritten
sorted by key once complete.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    """Every record in the file. A hard kill during an append can leave a last
    line without its newline; that partial line is cut off (and reported)
    before parsing, so the next append starts on a clean line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        print(f"  WARNING: {path} ends in a partial line ({len(data) - keep} bytes); cutting it off")
        with path.open("r+b") as f:
            f.truncate(keep)
        data = data[:keep]
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def load_done_keys(path: Path, record_key: Callable[[dict], tuple]) -> set[tuple]:
    records = read_records(path)
    keys = [record_key(record) for record in records]
    assert len(set(keys)) == len(keys), f"{len(keys) - len(set(keys))} duplicate keys in {path}"
    return set(keys)


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def finalize(path: Path, record_key: Callable[[dict], tuple], expected_keys: set[tuple]) -> list[dict]:
    """Assert the file holds exactly the expected keys; rewrite it sorted by key."""
    records = read_records(path)
    keys = [record_key(record) for record in records]
    assert len(set(keys)) == len(keys), f"{len(keys) - len(set(keys))} duplicate keys in {path}"
    extra = set(keys) - expected_keys
    assert not extra, f"{len(extra)} records in {path} are outside the expected key set, e.g. {sorted(extra)[:3]}"
    missing = expected_keys - set(keys)
    assert not missing, f"{len(missing)} expected records missing from {path}, e.g. {sorted(missing)[:3]}"
    records.sort(key=record_key)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    print(f"  CHECK: {path} holds exactly the {len(expected_keys)} expected records (rewritten sorted by key)")
    return records
