#!/usr/bin/env python3
"""CLI entry point for ingest.fetch: downloads the pinned kernel.org
tarball from kernel_manifest.yaml, verifies it, and extracts the
configured source paths into .kernel_src/.

Usage: python scripts/fetch_kernel_source.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linuxgpt.ingest.fetch import fetch, load_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download and re-extract even if already fetched"
    )
    args = parser.parse_args()

    manifest = load_manifest()
    print(f"Fetching {manifest.tarball_name} ({manifest.kernel_version})...")
    dest = fetch(manifest, force=args.force)
    print(f"Extracted to {dest}")


if __name__ == "__main__":
    main()
