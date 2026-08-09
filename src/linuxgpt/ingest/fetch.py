"""Downloads a pinned kernel.org release tarball, verifies its sha256
against kernel.org's published checksums file, and extracts only the
source paths listed in kernel_manifest.yaml into the local .kernel_src/
cache. Idempotent: re-running without --force is a no-op once a version
has been fetched."""

from __future__ import annotations

import fnmatch
import hashlib
import tarfile
from dataclasses import dataclass

import httpx
import yaml

from linuxgpt.config import settings


@dataclass(frozen=True)
class PathSpec:
    type: str  # "dir" | "file" | "glob"
    path: str
    pattern: str | None = None


@dataclass(frozen=True)
class Manifest:
    kernel_version: str
    tarball_url: str
    tarball_name: str
    checksums_url: str
    paths: list[PathSpec]


def load_manifest(manifest_path=None) -> Manifest:
    manifest_path = manifest_path or settings.manifest_path
    data = yaml.safe_load(manifest_path.read_text())
    paths = [PathSpec(**p) for p in data["paths"]]
    return Manifest(
        kernel_version=data["kernel_version"],
        tarball_url=data["tarball_url"],
        tarball_name=data["tarball_name"],
        checksums_url=data["checksums_url"],
        paths=paths,
    )


def _download(url: str, dest) -> None:
    """Downloads to a sibling .part file and atomically renames it onto
    dest only on success -- a prior version wrote straight into dest and
    an interrupted re-download (e.g. a network timeout) truncated an
    already-good cached tarball in place, corrupting it for no reason."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp_dest.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    tmp_dest.replace(dest)


def _expected_sha256(checksums_url: str, tarball_name: str) -> str:
    resp = httpx.get(checksums_url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        line = line.strip()
        if line.endswith(tarball_name):
            parts = line.split()
            if len(parts) >= 2:
                return parts[0].lower()
    raise ValueError(f"{tarball_name} not found in checksums file at {checksums_url}")


def _sha256_of(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _member_matches(rel: str, spec: PathSpec) -> bool:
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


def _top_level_prefix(tar: tarfile.TarFile) -> str:
    top = tar.getmembers()[0].name.split("/", 1)[0]
    return top + "/"


def fetch(manifest: Manifest | None = None, *, force: bool = False):
    """Downloads + verifies the pinned tarball, extracts matching paths into
    settings.kernel_src_dir. Returns the extraction root. Idempotent unless
    force=True."""
    manifest = manifest or load_manifest()
    dest_root = settings.kernel_src_dir
    marker = dest_root / f".fetched-{manifest.kernel_version}"
    if marker.exists() and not force:
        return dest_root

    cache_dir = settings.kernel_src_dir.parent / ".cache"
    tarball_path = cache_dir / manifest.tarball_name

    if not tarball_path.exists() or force:
        _download(manifest.tarball_url, tarball_path)

    expected = _expected_sha256(manifest.checksums_url, manifest.tarball_name)
    actual = _sha256_of(tarball_path)
    if actual != expected:
        tarball_path.unlink(missing_ok=True)
        raise ValueError(
            f"sha256 mismatch for {manifest.tarball_name}: "
            f"expected {expected}, got {actual} (deleted corrupt download)"
        )

    dest_root.mkdir(parents=True, exist_ok=True)
    dest_root_resolved = dest_root.resolve()
    extracted = 0
    with tarfile.open(tarball_path, mode="r:xz") as tar:
        top_prefix = _top_level_prefix(tar)
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith(top_prefix):
                continue
            rel = member.name[len(top_prefix):]
            if not any(_member_matches(rel, spec) for spec in manifest.paths):
                continue
            target = (dest_root / rel).resolve()
            if dest_root_resolved not in target.parents:
                continue  # guard against path traversal in a malicious tar entry
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
            extracted += 1

    marker.write_text(f"kernel_version={manifest.kernel_version}\nfiles_extracted={extracted}\n")
    return dest_root
