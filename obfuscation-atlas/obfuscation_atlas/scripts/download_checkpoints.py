"""Download Obfuscation Atlas checkpoints from HuggingFace Hub.

Downloads LoRA adapter checkpoints and the base pretrained model to the local
HF cache (controlled by HF_HOME). Discovers available seeds automatically by
querying the HuggingFace API. Skips checkpoints already present in the local
cache (use --force-redownload to override).

Usage:
    # Dry run: download 1 new checkpoint, verify 1 cached skip, print full plan
    python -m obfuscation_atlas.scripts.download_checkpoints --dry-run

    # Download all target checkpoints (skips already-cached ones)
    python -m obfuscation_atlas.scripts.download_checkpoints

    # Download only det=3 checkpoints with specific KL values
    python -m obfuscation_atlas.scripts.download_checkpoints --det 3 --kl 0.01 0.1

    # Force re-download everything (ignore cache)
    python -m obfuscation_atlas.scripts.download_checkpoints --force-redownload

    # Skip the large base model download
    python -m obfuscation_atlas.scripts.download_checkpoints --skip-base-model

    # Download only specific probes
    python -m obfuscation_atlas.scripts.download_checkpoints --probe diverse_deception_probe mbpp_probe
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

HF_ORG = "AlignmentResearch"
REPO_PREFIX = "obfuscation-atlas"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SUFFIX = "Meta-Llama-3-8B-Instruct"

DEFAULT_KL_COEFS = [0.0001, 0.001, 0.01, 0.1, 1.0]
DEFAULT_DET_COEFS = [0, 1, 3, 10]
DEFAULT_PROBES = ["diverse_deception_probe", "mbpp_probe"]

REPO_PATTERN = re.compile(
    rf"^{re.escape(REPO_PREFIX)}-{re.escape(MODEL_SUFFIX)}"
    r"-kl(?P<kl>[\d.e+-]+)-det(?P<det>[\d.]+)-seed(?P<seed>\d+)"
    r"(?:-(?P<probe>\w+))?$"
)


@dataclass
class CheckpointSpec:
    kl: float
    det: float
    seed: int
    probe: str | None

    @property
    def repo_id(self) -> str:
        kl_str = f"{self.kl:g}"
        det_str = f"{self.det:g}"
        base = f"{HF_ORG}/{REPO_PREFIX}-{MODEL_SUFFIX}-kl{kl_str}-det{det_str}-seed{self.seed}"
        if self.probe is not None:
            return f"{base}-{self.probe}"
        return base

    def __str__(self) -> str:
        probe_str = f", probe={self.probe}" if self.probe else ""
        return f"kl={self.kl:g}, det={self.det:g}, seed={self.seed}{probe_str}"


def get_cached_repo_ids() -> set[str]:
    """Return repo IDs already present in the local HF cache (no network calls)."""
    try:
        cache_info = scan_cache_dir()
        return {repo.repo_id for repo in cache_info.repos}
    except Exception as e:
        print(f"WARNING: Could not scan HF cache: {e}")
        return set()


def discover_available_repos(api: HfApi) -> list[dict]:
    """List all obfuscation-atlas repos for the target model and parse their metadata."""
    print(f"Discovering repos from {HF_ORG} matching '{REPO_PREFIX}-{MODEL_SUFFIX}'...")
    models = list(api.list_models(
        author=HF_ORG,
        search=f"{REPO_PREFIX}-{MODEL_SUFFIX}",
    ))

    parsed = []
    for model in models:
        name = model.modelId.split("/")[-1]
        m = REPO_PATTERN.match(name)
        if m:
            parsed.append({
                "repo_id": model.modelId,
                "kl": float(m.group("kl")),
                "det": float(m.group("det")),
                "seed": int(m.group("seed")),
                "probe": m.group("probe"),
            })

    kl_vals = sorted({r["kl"] for r in parsed})
    det_vals = sorted({r["det"] for r in parsed})
    seeds = sorted({r["seed"] for r in parsed})
    probes = sorted({r["probe"] for r in parsed if r["probe"]})
    print(f"Found {len(parsed)} matching repos")
    print(f"  KL values: {kl_vals}")
    print(f"  det values: {det_vals}")
    print(f"  seeds: {seeds}")
    print(f"  probes: {probes}")
    return parsed


def build_download_list(
    available: list[dict],
    kl_coefs: list[float],
    det_coefs: list[float],
    probes: list[str],
    seeds: list[int] | None,
) -> tuple[list[CheckpointSpec], list[CheckpointSpec]]:
    """Build the list of checkpoints to download, returning (found, missing)."""
    available_set = {r["repo_id"] for r in available}
    available_seeds = sorted({r["seed"] for r in available}) if seeds is None else seeds

    wanted: list[CheckpointSpec] = []
    for kl in kl_coefs:
        for det in det_coefs:
            for seed in available_seeds:
                if det > 0:
                    for probe in probes:
                        wanted.append(CheckpointSpec(kl=kl, det=det, seed=seed, probe=probe))
                else:
                    wanted.append(CheckpointSpec(kl=kl, det=det, seed=seed, probe=None))

    found = [spec for spec in wanted if spec.repo_id in available_set]
    missing = [spec for spec in wanted if spec.repo_id not in available_set]

    return found, missing


def verify_download(spec: CheckpointSpec, download_path: str) -> bool:
    """Verify a downloaded LoRA checkpoint has the expected files."""
    adapter_config = os.path.join(download_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        print(f"  WARNING: {spec.repo_id} missing adapter_config.json at {download_path}")
        return False
    return True


def download_checkpoint(spec: CheckpointSpec) -> tuple[bool, str | None]:
    """Download a single checkpoint. Returns (success, path)."""
    repo_id = spec.repo_id
    try:
        path = snapshot_download(repo_id=repo_id)
        print(f"  Downloaded: {repo_id} -> {path}")
        return True, path
    except Exception as e:
        print(f"  FAILED: {repo_id} -- {e}")
        return False, None


def download_base_model(dry_run: bool = False) -> bool:
    """Download the base pretrained model."""
    if dry_run:
        print(f"  [DRY-RUN] Would download base model: {BASE_MODEL}")
        return True

    try:
        print(f"  Downloading base model: {BASE_MODEL} (this may take a while)...")
        path = snapshot_download(repo_id=BASE_MODEL)
        print(f"  Downloaded base model: {BASE_MODEL} -> {path}")
        return True
    except Exception as e:
        print(f"  FAILED to download base model: {BASE_MODEL} -- {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Obfuscation Atlas checkpoints from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kl", type=float, nargs="+", default=DEFAULT_KL_COEFS,
        help=f"KL coefficients to download (default: {DEFAULT_KL_COEFS})",
    )
    parser.add_argument(
        "--det", type=float, nargs="+", default=DEFAULT_DET_COEFS,
        help=f"Detector coefficients to download (default: {DEFAULT_DET_COEFS})",
    )
    parser.add_argument(
        "--probe", type=str, nargs="+", default=DEFAULT_PROBES,
        help=f"Probe types to download for det>0 (default: {DEFAULT_PROBES})",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=None,
        help="Specific seeds to download (default: all available)",
    )
    parser.add_argument(
        "--skip-base-model", action="store_true",
        help="Skip downloading the base pretrained model",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validation smoke test: download 1 new checkpoint, verify 1 cached "
             "skip, then print full plan without downloading the rest",
    )
    parser.add_argument(
        "--force-redownload", action="store_true",
        help="Ignore local cache and download all checkpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = HfApi()

    # --- Step 1: Discover available repos on HF ---
    print("=== Discovery ===")
    available = discover_available_repos(api)
    if not available:
        print("ERROR: No matching repos found. Check your HF_TOKEN and network connection.")
        return 1

    # --- Step 2: Build download list ---
    found, missing = build_download_list(
        available, args.kl, args.det, args.probe, args.seed,
    )

    # --- Step 3: Scan local cache ---
    print("\n=== Cache Scan ===")
    cached_repo_ids = get_cached_repo_ids()
    if args.force_redownload:
        print("--force-redownload: ignoring local cache")
        already_cached = []
        to_download = found
    else:
        already_cached = [spec for spec in found if spec.repo_id in cached_repo_ids]
        to_download = [spec for spec in found if spec.repo_id not in cached_repo_ids]
    print(f"Local HF cache contains {len(cached_repo_ids)} repos total")
    print(f"  Target checkpoints on HF: {len(found)}")
    print(f"  Already cached (will skip): {len(already_cached)}")
    print(f"  New (will download): {len(to_download)}")
    if missing:
        print(f"  Not found on HF: {len(missing)}")

    # --- Step 4: Print download plan ---
    print(f"\n=== Download Plan ===")
    if already_cached:
        print(f"Skipping {len(already_cached)} cached checkpoints:")
        for spec in already_cached:
            print(f"  CACHED: {spec}")
    if to_download:
        print(f"Will download {len(to_download)} new checkpoints:")
        for spec in to_download:
            print(f"  NEW: {spec}")
    if missing:
        print(f"Not found on HF ({len(missing)}):")
        for spec in missing:
            print(f"  NOT FOUND: {spec} ({spec.repo_id})")
    if not args.skip_base_model:
        base_cached = BASE_MODEL in cached_repo_ids
        print(f"Base model: {BASE_MODEL} ({'cached' if base_cached else 'will download'})")

    if not to_download and (args.skip_base_model or BASE_MODEL in cached_repo_ids):
        print("\nNothing new to download.")
        return 0

    # --- Step 5: Dry-run validation mode ---
    if args.dry_run:
        print(f"\n=== Dry-Run Validation ===")
        verify_fail = False

        # Test 1: verify a cached checkpoint is correctly identified
        if already_cached:
            test_cached = already_cached[0]
            print(f"[SKIP TEST] Verifying cached checkpoint is correctly skipped:")
            print(f"  Repo: {test_cached.repo_id}")
            assert test_cached.repo_id in cached_repo_ids, (
                f"BUG: {test_cached.repo_id} should be in cache but isn't"
            )
            print(f"  PASS: correctly identified as cached")
        else:
            print("[SKIP TEST] No cached checkpoints to test skip logic (all are new)")

        # Test 2: actually download one new checkpoint
        if to_download:
            test_new = to_download[0]
            print(f"[DOWNLOAD TEST] Downloading 1 new checkpoint:")
            print(f"  Repo: {test_new.repo_id}")
            success, path = download_checkpoint(test_new)
            if success and path:
                if verify_download(test_new, path):
                    print(f"  PASS: downloaded and verified (adapter_config.json present)")
                else:
                    print(f"  FAIL: downloaded but verification failed")
                    verify_fail = True
            else:
                print(f"  FAIL: download failed")
                verify_fail = True
        else:
            print("[DOWNLOAD TEST] No new checkpoints to test download (all cached)")

        remaining = len(to_download) - (1 if to_download else 0)
        print(f"\n=== Dry-Run Summary ===")
        print(f"Cached (skipped): {len(already_cached)}")
        print(f"Downloaded in test: {1 if to_download else 0}")
        print(f"Remaining to download (full run): {remaining}")
        if missing:
            print(f"Not found on HF: {len(missing)}")
        if verify_fail:
            print("WARNING: Verification failed -- check output above")
            return 1
        return 0

    # --- Step 6: Download all new checkpoints ---
    print(f"\n=== Downloading LoRA Adapters ===")
    success_count = 0
    fail_count = 0
    verify_warn_count = 0
    total = len(to_download)

    for i, spec in enumerate(to_download, 1):
        print(f"  [{i}/{total}] {spec}")
        success, path = download_checkpoint(spec)
        if success:
            success_count += 1
            if path and not verify_download(spec, path):
                verify_warn_count += 1
        else:
            fail_count += 1

    # --- Step 7: Base model ---
    if not args.skip_base_model:
        base_cached = BASE_MODEL in cached_repo_ids
        if base_cached and not args.force_redownload:
            print(f"\n=== Base Model ===")
            print(f"  Base model already cached, skipping: {BASE_MODEL}")
        else:
            print(f"\n=== Downloading Base Model ===")
            if not download_base_model():
                fail_count += 1

    # --- Step 8: Final summary ---
    print(f"\n=== Summary ===")
    print(f"Skipped (cached): {len(already_cached)}")
    print(f"Downloaded: {success_count}/{total}")
    if verify_warn_count:
        print(f"Verification warnings: {verify_warn_count}")
    if fail_count:
        print(f"Failures: {fail_count}")
    if missing:
        print(f"Not found on HF: {len(missing)}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
