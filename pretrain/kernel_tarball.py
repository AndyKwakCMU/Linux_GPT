"""Shared access to the pinned kernel tarball for pretrain/: loading piece
manifests, verifying the tarball's sha256, and streaming its members.

Reads the same cached tarball the RAG side downloads
(.cache/linux-6.12.tar.xz, written by `linuxgpt ingest`), straight out of
the .tar.xz -- nothing gets extracted to disk. Deliberately does not import
anything from src/linuxgpt/ (pretrain/ is standalone, own venv); the
checksum and path-matching logic below mirror
src/linuxgpt/ingest/fetch.py's _expected_sha256 / _member_matches, so keep
them in sync if either changes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

PRETRAIN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PRETRAIN_ROOT.parent
PIECES_DIR = PRETRAIN_ROOT / "pieces"
DEFAULT_TARBALL_DIR = REPO_ROOT / ".cache"


@dataclass(frozen=True)
class PathSpec:
    type: str  # "dir" | "file" | "glob"
    path: str
    pattern: str | None = None


@dataclass(frozen=True)
class Piece:
    name: str
    kernel_version: str
    tarball_name: str
    checksums_url: str
    paths: list[PathSpec]


def load_piece(name: str) -> Piece:
    path = PIECES_DIR / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in PIECES_DIR.glob("*.yaml"))
        raise SystemExit(f"No piece manifest at {path}. Available pieces: {available}")
    data = yaml.safe_load(path.read_text())
    return Piece(
        name=data["name"],
        kernel_version=data["kernel_version"],
        tarball_name=data["tarball_name"],
        checksums_url=data["checksums_url"],
        paths=[PathSpec(**p) for p in data["paths"]],
    )


def member_matches(rel: str, spec: PathSpec) -> bool:
    """rel is the tar member path with the 'linux-X.Y/' top-level dir
    already stripped, e.g. 'fs/f2fs/super.c'."""
    if spec.type == "dir":
        return rel == spec.path or rel.startswith(spec.path.rstrip("/") + "/")
    if spec.type == "file":
        return rel == spec.path
    if spec.type == "glob":
        rel_dir, _, rel_name = rel.rpartition("/")
        return rel_dir == spec.path and fnmatch.fnmatch(rel_name, spec.pattern or "*")
    raise ValueError(f"unknown path spec type: {spec.type}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha256(checksums_url: str, tarball_name: str) -> str:
    with urllib.request.urlopen(checksums_url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == tarball_name:
            return parts[0].lower()
    raise ValueError(f"{tarball_name} not found in checksums file at {checksums_url}")


def load_kernel_pin() -> dict:
    """The repo-wide kernel pin (version, tarball, checksums URL) from
    kernel_manifest.yaml -- used where no single piece applies, e.g. the
    shared tokenizer."""
    return yaml.safe_load((REPO_ROOT / "kernel_manifest.yaml").read_text())


def verified_tarball(
    tarball_name: str, checksums_url: str, tarball_dir: Path = DEFAULT_TARBALL_DIR, *, skip_verify: bool = False
) -> tuple[Path, str]:
    """Returns (tarball path, its sha256), after checking the sha256 against
    kernel.org's published checksums. skip_verify only skips the network
    comparison -- the local hash is still computed and recorded."""
    tarball = tarball_dir / tarball_name
    if not tarball.is_file():
        raise SystemExit(
            f"No tarball at {tarball}. Run `linuxgpt ingest` from the repo root "
            f"(it downloads + verifies it), or copy {tarball_name} there from another machine."
        )
    actual = sha256_of(tarball)
    if skip_verify:
        print(f"WARNING: --skip-verify: not checking {tarball.name} against {checksums_url}")
        return tarball, actual
    expected = _expected_sha256(checksums_url, tarball_name)
    if actual != expected:
        raise SystemExit(f"sha256 mismatch for {tarball}: expected {expected}, got {actual}")
    return tarball, actual


def iter_members(tarball: Path) -> Iterator[tuple[str, bytes]]:
    """Streams (relative path, file bytes) for every regular file in the
    tarball, in archive order. Streaming mode ('r|xz') so a full pass over
    the ~1.4GB uncompressed tree never needs random access or extraction."""
    with tarfile.open(tarball, mode="r|xz") as tar:
        top_prefix = None
        for member in tar:
            if top_prefix is None:
                top_prefix = member.name.split("/", 1)[0] + "/"
            if not member.isfile() or not member.name.startswith(top_prefix):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            yield member.name[len(top_prefix):], f.read()
