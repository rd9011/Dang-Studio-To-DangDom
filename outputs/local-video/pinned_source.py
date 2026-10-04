#!/usr/bin/env python3
"""Verified, Git-free source snapshots for optional Dang Studio runtimes.

The public installers use immutable GitHub codeload archives as transport and
extract only the exact files needed at runtime.  The resulting tree must match
an embedded SHA-256 before it can be published, so a provenance JSON file alone
is never treated as authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, Sequence


PROVENANCE_NAME = ".dang-studio-source.json"
PROVENANCE_SCHEMA = "dang-studio-source/1"
_IGNORED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


class SourceInstallError(RuntimeError):
    """A pinned source snapshot could not be acquired or verified safely."""


@dataclass(frozen=True)
class SourcePin:
    name: str
    repository: str
    revision: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    archive_root: str
    tree_sha256: str
    includes: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise SourceInstallError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_ignored_runtime_artifact(relative: PurePosixPath) -> bool:
    return (
        any(part in _IGNORED_DIRECTORY_NAMES for part in relative.parts)
        or relative.suffix in _IGNORED_FILE_SUFFIXES
        or relative.as_posix() == PROVENANCE_NAME
    )


def source_tree_digest(
    root: Path, *, allowed_runtime_entries: Sequence[str] = ()
) -> str:
    """Hash a source tree's paths, executable bits, byte counts and contents.

    Python bytecode/cache products are excluded because merely importing a
    verified source tree may create them.  All other additions and mutations
    change the digest.  Symbolic links and non-regular filesystem entries are
    rejected rather than followed.
    """

    if root.is_symlink() or not root.is_dir():
        _fail(f"source snapshot is absent or unsafe: {root}")
    if (root / ".git").exists():
        _fail("source snapshot must not contain .git")
    allowed = set(allowed_runtime_entries)
    if (root / ".venv").exists() and ".venv" not in allowed:
        _fail("source snapshot must not contain a prebuilt .venv")
    records: list[tuple[str, int, int, str]] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for entry in entries:
            relative = PurePosixPath(entry.relative_to(root).as_posix())
            if relative.parts and relative.parts[0] in allowed:
                continue
            if _is_ignored_runtime_artifact(relative):
                continue
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                _fail(f"source snapshot contains a symbolic link: {relative}")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                _fail(f"source snapshot contains a non-regular entry: {relative}")
            records.append(
                (
                    relative.as_posix(),
                    1 if mode & 0o111 else 0,
                    entry.stat().st_size,
                    sha256_file(entry),
                )
            )
    except OSError as exc:
        raise SourceInstallError(f"cannot inspect source snapshot {root}: {exc}") from exc
    digest = hashlib.sha256()
    for relative, executable, size, content_hash in records:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(executable).encode("ascii") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(content_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def verify_source_tree(
    root: Path,
    pin: SourcePin,
    *,
    allowed_runtime_entries: Sequence[str] = (),
) -> tuple[bool, str]:
    try:
        actual = source_tree_digest(root, allowed_runtime_entries=allowed_runtime_entries)
        if actual != pin.tree_sha256:
            return False, f"{pin.name} source tree SHA-256 differs ({actual})"
        provenance_path = root / PROVENANCE_NAME
        if not provenance_path.is_file() or provenance_path.is_symlink():
            return False, f"{pin.name} source provenance is absent"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = _provenance(pin)
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != value for key, value in expected.items()
        ):
            return False, f"{pin.name} source provenance differs from its immutable pin"
    except (OSError, UnicodeError, json.JSONDecodeError, SourceInstallError) as exc:
        return False, f"cannot verify {pin.name} source: {exc}"
    return True, f"verified Git-free {pin.name} source at {pin.revision[:12]}"


def _provenance(pin: SourcePin) -> dict[str, object]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "name": pin.name,
        "repository": pin.repository,
        "revision": pin.revision,
        "archive_url": pin.archive_url,
        "archive_sha256": pin.archive_sha256,
        "archive_bytes": pin.archive_bytes,
        "tree_sha256": pin.tree_sha256,
        "includes": list(pin.includes),
    }


def _write_provenance(root: Path, pin: SourcePin) -> None:
    path = root / PROVENANCE_NAME
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_provenance(pin), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _included(relative: PurePosixPath, includes: Sequence[str]) -> bool:
    value = relative.as_posix()
    return any(value == item or (item.endswith("/") and value.startswith(item)) for item in includes)


def extract_verified_archive(archive: Path, destination: Path, pin: SourcePin) -> None:
    """Safely extract the selected source files from a hash-verified tarball."""

    if destination.exists() or destination.is_symlink():
        _fail(f"source extraction destination already exists: {destination}")
    if archive.is_symlink() or not archive.is_file():
        _fail(f"source archive is absent or unsafe: {archive}")
    if archive.stat().st_size != pin.archive_bytes or sha256_file(archive) != pin.archive_sha256:
        _fail(f"{pin.name} source archive failed its immutable byte/hash pin")
    destination.mkdir(parents=True, mode=0o700)
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as source:
            for member in source:
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
                    _fail(f"unsafe source archive member: {member.name}")
                if not member_path.parts or member_path.parts[0] != pin.archive_root:
                    _fail(f"unexpected source archive root: {member.name}")
                relative = PurePosixPath(*member_path.parts[1:])
                if not relative.parts:
                    if not member.isdir():
                        _fail(f"source archive root is not a directory: {member.name}")
                    continue
                # Reject links/devices even outside the reduced runtime subset;
                # the approved transport must be safe as a whole.
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    _fail(f"unsupported source archive member type: {member.name}")
                if not (member.isdir() or member.isfile()):
                    _fail(f"unsupported source archive member: {member.name}")
                if not _included(relative, pin.includes) or member.isdir():
                    continue
                relative_text = relative.as_posix()
                if relative_text in seen:
                    _fail(f"duplicate source archive member: {relative_text}")
                seen.add(relative_text)
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    _fail(f"duplicate or unsafe source target: {relative_text}")
                input_handle = source.extractfile(member)
                if input_handle is None:
                    _fail(f"cannot read source archive member: {relative_text}")
                with input_handle, target.open("xb") as output:
                    shutil.copyfileobj(input_handle, output, 1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    except (OSError, tarfile.TarError) as exc:
        raise SourceInstallError(f"cannot extract {pin.name} source archive: {exc}") from exc
    actual = source_tree_digest(destination)
    if actual != pin.tree_sha256:
        _fail(f"extracted {pin.name} source tree SHA-256 differs ({actual})")
    _write_provenance(destination, pin)


def download_verified_archive(pin: SourcePin, cache_root: Path) -> Path:
    if not pin.archive_url or not pin.archive_sha256 or pin.archive_bytes <= 0:
        _fail(f"{pin.name} does not define a public source archive transport")
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"{pin.name}-{pin.revision}.tar.gz"
    if archive.is_file() and not archive.is_symlink():
        if archive.stat().st_size == pin.archive_bytes and sha256_file(archive) == pin.archive_sha256:
            return archive
        archive.unlink()
    elif archive.exists() or archive.is_symlink():
        _fail(f"unsafe source archive cache path: {archive}")
    curl = shutil.which("curl")
    if not curl:
        _fail("secure curl is required to download the pinned source archive")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".partial", dir=cache_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        command = (
            curl,
            "--disable",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--tlsv1.2",
            "--location",
            "--fail",
            "--show-error",
            "--silent",
            "--retry",
            "10",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--output",
            str(temporary),
            pin.archive_url,
        )
        environment = dict(os.environ)
        for key in ("GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            environment.pop(key, None)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            _fail(f"{pin.name} source archive download exited with status {result.returncode}")
        if temporary.stat().st_size != pin.archive_bytes or sha256_file(temporary) != pin.archive_sha256:
            _fail(f"downloaded {pin.name} source archive failed its immutable byte/hash pin")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, archive)
    except OSError as exc:
        raise SourceInstallError(f"cannot download {pin.name} source archive: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def stage_pinned_source(
    destination: Path,
    pin: SourcePin,
    *,
    cache_root: Path,
) -> str:
    """Download and extract verified source into a new directory."""

    archive = download_verified_archive(pin, cache_root)
    extract_verified_archive(archive, destination, pin)
    return "archive"
