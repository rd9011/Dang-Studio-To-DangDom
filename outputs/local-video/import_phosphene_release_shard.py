#!/usr/bin/env python3
"""Authenticate and move one externally downloaded Phosphene release shard.

The normal installer owns the private staging format.  This helper only admits
an exact asset from one of its revision-pinned manifests, creates the same
identity sidecar, and moves an isolated APFS copy-on-write clone into staging
without allocating a second physical multi-gigabyte copy.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

import install_ltx25_advanced_q8 as advanced
import install_phosphene_ltx25_release as release


BROWSER_PARTIAL_SUFFIXES = (
    ".crdownload",
    ".download",
    ".partial",
    ".part",
    ".tmp",
)


class ShardImportError(release.InstallerError):
    """An external file could not be admitted to trusted private staging."""


@dataclass(frozen=True)
class AssetMatch:
    pin: release.PackPin
    manifest_digest: str
    file_name: str
    file_spec: Mapping[str, object]
    shard_index: int
    shard: Mapping[str, object]


@dataclass(frozen=True)
class ImportResult:
    pack: str
    asset: str
    file_name: str
    staged_path: Path
    bytes: int
    sha256: str


def _fail(message: str) -> NoReturn:
    raise ShardImportError(message)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _inode_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _linked_content_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    # Publishing a private clone through a temporary hard link legitimately
    # updates ctime/link-count. Device, inode, size, and mtime remain stable.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _absolute_source(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        _fail("source must be an absolute path")
    # Normalize ``..`` without dereferencing the final component before lstat.
    return Path(os.path.abspath(os.fspath(expanded)))


def _inspect_source(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ShardImportError(f"source is unavailable: {path} ({exc})") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail(f"refusing symlinked source: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail(f"refusing non-regular source: {path}")
    lowered = path.name.lower()
    if lowered.endswith(BROWSER_PARTIAL_SUFFIXES):
        _fail(
            f"refusing browser partial/in-progress filename: {path.name}; "
            "wait for the browser to publish the exact release-asset basename"
        )
    return info


def _supported_packs() -> dict[str, release.PackPin]:
    """Pins accepted by the importer without widening either installer's CLI."""
    return {
        **release.PACKS,
        **{pin.key: pin for pin in advanced.PINS},
    }


def _supported_pack_order() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*release.PACK_ORDER, *(pin.key for pin in advanced.PINS)))
    )


def _candidate_pack_keys(asset: str, pack_key: str | None) -> tuple[str, ...]:
    packs = _supported_packs()
    if pack_key is not None:
        if pack_key not in packs:
            _fail(
                f"unknown pack {pack_key!r}; expected one of "
                f"{', '.join(_supported_pack_order())}"
            )
        return (pack_key,)
    # A unique known prefix narrows which pinned manifest must be fetched, but
    # never authenticates the asset: _find_asset still requires an exact shard
    # entry in that byte-pinned manifest. Shared unprefixed assets retain the
    # exhaustive ambiguity check over every supported pack.
    prefixed = tuple(
        key for key in _supported_pack_order() if asset.startswith(f"{key}__")
    )
    return prefixed or _supported_pack_order()


def _importable_files(
    pin: release.PackPin,
    manifest: Mapping[str, object],
) -> dict[str, dict]:
    """Return files the corresponding local installer will actually consume."""
    files = release.pack_files(manifest)
    if pin.key == advanced.Q8_PIN.key:
        selected = frozenset((*advanced.Q8_METADATA, advanced.Q8_TRANSFORMER))
        return {name: spec for name, spec in files.items() if name in selected}
    if pin.key == advanced.HQ_PIN.key:
        return {
            name: spec
            for name, spec in files.items()
            if name == advanced.HQ_TRANSFORMER
        }
    return files


def _find_asset(
    asset: str,
    *,
    pack_key: str | None,
    opener: object,
    timeout: int,
    report: Callable[[str], None],
) -> AssetMatch:
    matches: list[AssetMatch] = []
    packs = _supported_packs()
    for key in _candidate_pack_keys(asset, pack_key):
        pin = packs[key]
        manifest, raw = release.fetch_pinned_manifest(
            pin,
            opener=opener,
            timeout=timeout,
        )
        digest = release.sha256_bytes(raw)
        report(
            f"[manifest] {key}: {len(raw):,} bytes, SHA-256 {digest} verified"
        )
        for file_name, file_spec in _importable_files(pin, manifest).items():
            shards = file_spec["shards"]
            assert isinstance(shards, list)
            for index, shard in enumerate(shards):
                assert isinstance(shard, dict)
                if shard["asset"] == asset:
                    matches.append(
                        AssetMatch(
                            pin=pin,
                            manifest_digest=digest,
                            file_name=file_name,
                            file_spec=file_spec,
                            shard_index=index,
                            shard=shard,
                        )
                    )
    if not matches:
        qualifier = f" in pack {pack_key}" if pack_key else ""
        _fail(f"{asset!r} is not an exact pinned Phosphene release asset{qualifier}")
    if len(matches) != 1:
        packs = ", ".join(match.pin.key for match in matches)
        _fail(f"asset basename is ambiguous across packs ({packs}); pass --pack")
    return matches[0]


def _ensure_directory(path: Path, *, mode: int) -> None:
    if _lexists(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            _fail(f"refusing unsafe directory path: {path}")
        return
    try:
        path.mkdir(mode=mode)
    except FileExistsError:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            _fail(f"refusing unsafe directory path: {path}")


def _prepare_stage_directory(models_root: Path, pin: release.PackPin) -> Path:
    models_root.mkdir(parents=True, exist_ok=True)
    if models_root.is_symlink() or not models_root.is_dir():
        _fail(f"models root is not a safe directory: {models_root}")
    stage_root = models_root / ".phosphene-release-staging"
    _ensure_directory(stage_root, mode=0o700)
    stage_dir = stage_root / pin.key
    _ensure_directory(stage_dir, mode=0o700)
    return stage_dir


def _refuse_path_collision(path: Path, *, label: str) -> None:
    if _lexists(path):
        _fail(f"refusing to overwrite existing {label}: {path}")


def _validate_assembly_position(
    models_root: Path,
    match: AssetMatch,
) -> None:
    assembly, metadata = release._stage_paths(
        models_root,
        match.pin,
        match.file_name,
    )
    assembly_exists = _lexists(assembly)
    metadata_exists = _lexists(metadata)
    if assembly_exists:
        release._regular_or_missing(assembly, label="assembly")
    if metadata_exists:
        release._regular_or_missing(metadata, label="assembly metadata")
    if not assembly_exists and not metadata_exists:
        return
    if not metadata_exists:
        _fail(f"untrusted assembly collision has no metadata: {assembly}")
    value = release._read_json(metadata)
    expected = release._assembly_identity(
        match.pin,
        match.manifest_digest,
        match.file_name,
        match.file_spec,
    )
    if not release._resume_metadata_matches(value, expected):
        _fail(f"untrusted assembly metadata collision: {metadata}")
    assert value is not None
    completed = value.get("shards_done")
    assembled_bytes = value.get("assembled_bytes")
    shards = match.file_spec["shards"]
    assert isinstance(shards, list)
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= len(shards)
    ):
        _fail(f"invalid shards_done in assembly metadata: {metadata}")
    expected_bytes = sum(int(shard["bytes"]) for shard in shards[:completed])
    if assembled_bytes != expected_bytes:
        _fail(f"invalid assembled_bytes in assembly metadata: {metadata}")
    if assembly_exists and assembly.stat().st_size != expected_bytes:
        _fail(f"assembly size disagrees with its metadata: {assembly}")
    if not assembly_exists and expected_bytes:
        _fail(f"assembly metadata claims bytes but its file is missing: {metadata}")
    if completed > match.shard_index:
        _fail(
            f"{match.shard['asset']} was already incorporated into the private assembly; "
            "refusing a duplicate"
        )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, release.CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _publish_metadata_exclusive(
    path: Path,
    payload: Mapping[str, object],
) -> tuple[int, int]:
    """Publish complete mode-0600 JSON at ``path`` without replacement."""
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.import-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o600)
        owned_descriptor = descriptor
        descriptor = -1
        with os.fdopen(owned_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_identity = _inode_identity(os.fstat(handle.fileno()))
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ShardImportError(
                f"refusing to overwrite existing shard metadata: {path}"
            ) from exc
        assert temporary_identity is not None
        published_identity = temporary_identity
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or _inode_identity(info) != published_identity
        ):
            _fail(f"metadata publication did not create a regular file: {path}")
        release._fsync_directory(path.parent)
        return published_identity
    except BaseException:
        if published_identity is not None:
            issue = _unlink_if_owned(path, published_identity)
            if issue:
                raise ShardImportError(
                    f"metadata publication failed and cleanup was incomplete: {issue}"
                )
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> str | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"could not inspect {path}: {exc}"
    if _inode_identity(info) != identity:
        return f"refused to clean replaced path {path}"
    try:
        path.unlink()
    except OSError as exc:
        return f"could not clean {path}: {exc}"
    return None


def _unlink_source(path: Path) -> None:
    path.unlink()


def _same_filesystem(source_info: os.stat_result, stage_dir: Path) -> bool:
    return source_info.st_dev == stage_dir.stat().st_dev


def _fsync_directory_strict(directory: Path) -> None:
    """Make a transaction boundary durable or fail before its sole source moves."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _trusted_partial_metadata(
    metadata_path: Path,
    identity: Mapping[str, object],
) -> tuple[dict, tuple[int, int]]:
    """Accept only the installer's exact identity and optional resume validator."""
    release._regular_or_missing(metadata_path, label="staged shard metadata")
    value = release._read_json(metadata_path)
    if value is None or not release._resume_metadata_matches(value, identity):
        _fail(f"existing partial has untrusted shard metadata: {metadata_path}")
    identity_keys = set(identity)
    extra = set(value) - identity_keys
    if extra:
        if (
            extra != {"validator_kind", "validator"}
            or release._metadata_range_validator(value) is None
        ):
            _fail(f"existing partial metadata has unexpected fields: {metadata_path}")
    info = metadata_path.lstat()
    if not stat.S_ISREG(info.st_mode):
        _fail(f"refusing non-regular staged shard metadata: {metadata_path}")
    return value, _inode_identity(info)


def _verify_partial_prefix(
    partial_path: Path,
    source_descriptor: int,
    *,
    expected_bytes: int,
) -> tuple[os.stat_result, int]:
    """Prove an incomplete staged file is the exact prefix of the verified source."""
    try:
        before = partial_path.lstat()
    except OSError as exc:
        raise ShardImportError(f"cannot inspect existing staged shard: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(f"refusing non-regular or symlinked staged shard: {partial_path}")
    if before.st_size >= expected_bytes:
        state = "full-size" if before.st_size == expected_bytes else "oversized"
        _fail(f"refusing to replace {state} staged shard: {partial_path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(partial_path, flags)
    except OSError as exc:
        raise ShardImportError(
            f"cannot securely open existing staged shard {partial_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before) or not stat.S_ISREG(opened.st_mode):
            _fail("existing staged shard changed while it was being opened")
        offset = 0
        while offset < opened.st_size:
            wanted = min(release.CHUNK_BYTES, opened.st_size - offset)
            partial_block = os.read(descriptor, wanted)
            source_block = os.pread(source_descriptor, wanted, offset)
            if len(partial_block) != wanted or partial_block != source_block:
                _fail(
                    f"existing staged shard is not an exact prefix of the verified source: "
                    f"{partial_path}"
                )
            offset += wanted
        after = os.fstat(descriptor)
        if _identity(opened) != _identity(after):
            _fail("existing staged shard changed while its prefix was verified")
        return after, opened.st_size
    finally:
        os.close(descriptor)


def _clone_candidate(
    source_descriptor: int,
    stage_dir: Path,
    asset: str,
) -> Path:
    """Create a distinct private APFS CoW clone from an already-open source."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fclonefileat = libc.fclonefileat
    except (OSError, AttributeError) as exc:
        raise ShardImportError(
            "this macOS runtime does not expose fclonefileat; refusing a mutable hard-link import"
        ) from exc
    fclonefileat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    fclonefileat.restype = ctypes.c_int

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_descriptor = os.open(stage_dir, directory_flags)
    candidate: Path | None = None
    try:
        for _attempt in range(16):
            candidate = stage_dir / (
                f".{asset}.external-replacement-{secrets.token_hex(8)}"
            )
            ctypes.set_errno(0)
            result = fclonefileat(
                source_descriptor,
                directory_descriptor,
                os.fsencode(candidate.name),
                0,
            )
            if result == 0:
                break
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                continue
            if error_number in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP}:
                _fail(
                    "source and staging must be on the same clone-capable APFS filesystem; "
                    "refusing a physical multi-gigabyte copy or mutable hard link"
                )
            raise ShardImportError(
                f"fclonefileat could not create a private CoW candidate: "
                f"{os.strerror(error_number)}"
            )
        else:
            _fail("could not allocate a collision-free private clone path")
    finally:
        os.close(directory_descriptor)

    assert candidate is not None
    return candidate


def _authenticate_clone(
    candidate: Path,
    source_info: os.stat_result,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> os.stat_result:
    """Full-hash a clone and prove it is stable and inode-isolated."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ShardImportError(f"cannot securely open private CoW candidate: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"private CoW candidate is not a regular file: {candidate}")
        if _inode_identity(before) == _inode_identity(source_info):
            _fail("clonefile returned the mutable source inode instead of a distinct CoW clone")
        if before.st_size != expected_bytes:
            _fail(
                f"private CoW candidate has wrong size: {before.st_size:,} != "
                f"{expected_bytes:,} bytes"
            )
        # clonefile copies the source mode. Restrict the private candidate
        # before the potentially long full-hash pass, even when an existing
        # staging directory is traversable by other local accounts.
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        protected = os.fstat(descriptor)
        if (
            _inode_identity(protected) != _inode_identity(before)
            or protected.st_size != before.st_size
            or protected.st_mtime_ns != before.st_mtime_ns
        ):
            _fail("private CoW candidate changed while its permissions were restricted")
        observed_sha256 = _sha256_descriptor(descriptor)
        after_hash = os.fstat(descriptor)
        if _identity(protected) != _identity(after_hash):
            _fail("private CoW candidate changed while it was being authenticated")
        if observed_sha256 != expected_sha256:
            _fail(
                f"private CoW candidate SHA-256 mismatch: {observed_sha256} != "
                f"{expected_sha256}"
            )
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            final.st_size != expected_bytes
            or final.st_mtime_ns != after_hash.st_mtime_ns
            or _inode_identity(final) != _inode_identity(after_hash)
        ):
            _fail("private CoW candidate changed during finalization")
        return final
    finally:
        os.close(descriptor)


def import_shard(
    source: Path,
    *,
    pack_key: str | None = None,
    models_root: Path = release.DEFAULT_MODELS_ROOT,
    opener: object | None = None,
    timeout: int = 60,
    replace_partial: bool = False,
    report: Callable[[str], None] = print,
) -> ImportResult:
    """Verify and move one exact official asset into installer staging."""
    if timeout <= 0:
        _fail("timeout must be positive")
    source = _absolute_source(source)
    source_lstat = _inspect_source(source)
    models_root = Path(models_root).expanduser().resolve()
    opener = opener if opener is not None else release.build_opener()
    match = _find_asset(
        source.name,
        pack_key=pack_key,
        opener=opener,
        timeout=timeout,
        report=report,
    )
    expected_bytes = int(match.shard["bytes"])
    expected_sha256 = str(match.shard["sha256"])
    if source_lstat.st_size != expected_bytes:
        _fail(
            f"wrong size for {source.name}: {source_lstat.st_size:,} != "
            f"{expected_bytes:,} bytes; source was left untouched"
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ShardImportError(f"cannot securely open source {source}: {exc}") from exc

    metadata_created: tuple[Path, tuple[int, int]] | None = None
    private_candidate: tuple[Path, tuple[int, int]] | None = None
    publication_committed = False
    source_removed = False
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _inode_identity(opened) != _inode_identity(source_lstat):
            _fail("source changed while it was being opened; refusing import")
        observed_sha256 = _sha256_descriptor(descriptor)
        after_hash = os.fstat(descriptor)
        if _identity(opened) != _identity(after_hash):
            _fail("source changed while it was being hashed; refusing import")
        if observed_sha256 != expected_sha256:
            _fail(
                f"SHA-256 mismatch for {source.name}: {observed_sha256} != "
                f"{expected_sha256}; source was left untouched"
            )
        report(
            f"[verify] {source.name}: {expected_bytes:,} bytes, "
            f"SHA-256 {observed_sha256} verified"
        )

        stage_dir = _prepare_stage_directory(models_root, match.pin)
        if not _same_filesystem(after_hash, stage_dir):
            _fail(
                "source and private staging are on different filesystems; refusing a hidden "
                "multi-gigabyte copy. Move the completed download onto the models volume first"
            )
        shard_path, metadata_path = release._stage_paths(
            models_root,
            match.pin,
            match.file_name,
            str(match.shard["asset"]),
        )
        with release._file_staging_lock(
            models_root,
            match.pin,
            match.file_name,
            blocking=False,
        ):
            destination_dir = models_root / match.pin.directory
            if _lexists(destination_dir):
                info = destination_dir.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    _fail(f"refusing unsafe pack destination: {destination_dir}")
            _refuse_path_collision(
                destination_dir / match.file_name,
                label="installed model file",
            )
            _validate_assembly_position(models_root, match)

            identity = release._shard_identity(
                match.pin,
                match.manifest_digest,
                match.shard,
            )
            sidecar = {
                **identity,
                # import_shard has already full-hashed the source.  This
                # journal lets disk planning credit the staged bytes without
                # reading multi-gigabyte shards yet again; install_file still
                # independently rehashes each shard before assembly.
                release.VERIFIED_SHARD_SHA256_FIELD: expected_sha256,
            }
            shard_exists = _lexists(shard_path)
            metadata_exists = _lexists(metadata_path)
            replacing = shard_exists or metadata_exists
            partial_info: os.stat_result | None = None
            partial_bytes = 0
            prior_metadata: dict | None = None
            prior_metadata_identity: tuple[int, int] | None = None
            if replacing:
                if not replace_partial:
                    collision = shard_path if shard_exists else metadata_path
                    _fail(
                        f"refusing to overwrite existing staged shard state: {collision}; "
                        "pass --replace-partial only for a trusted incomplete partial"
                    )
                if not shard_exists:
                    _fail(f"refusing replacement with missing staged shard: {shard_path}")
                if not metadata_exists:
                    _fail(
                        f"refusing replacement because the existing partial has no sidecar: "
                        f"{metadata_path}"
                    )
                prior_metadata, prior_metadata_identity = _trusted_partial_metadata(
                    metadata_path,
                    identity,
                )
                partial_info, partial_bytes = _verify_partial_prefix(
                    shard_path,
                    descriptor,
                    expected_bytes=expected_bytes,
                )
                report(
                    f"[replace] {source.name}: authenticated existing "
                    f"{partial_bytes:,}-byte prefix before replacement"
                )

            if not replacing:
                metadata_identity = _publish_metadata_exclusive(metadata_path, sidecar)
                metadata_created = (metadata_path, metadata_identity)

            # Clone from the already-open verified descriptor, not the source
            # pathname.  APFS gives the candidate its own CoW inode, so a
            # browser or other process that still has the source open cannot
            # mutate the staged result through the original inode.
            candidate = _clone_candidate(
                descriptor,
                stage_dir,
                str(match.shard["asset"]),
            )
            candidate_initial = candidate.lstat()
            private_candidate = (candidate, _inode_identity(candidate_initial))
            candidate_info = _authenticate_clone(
                candidate,
                after_hash,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
            private_candidate = (candidate, _inode_identity(candidate_info))
            _fsync_directory_strict(stage_dir)

            if replacing:
                assert partial_info is not None
                assert prior_metadata is not None
                assert prior_metadata_identity is not None
                current_partial = shard_path.lstat()
                if _identity(current_partial) != _identity(partial_info):
                    _fail("existing staged partial changed before replacement commit")
                current_metadata = metadata_path.lstat()
                if _inode_identity(current_metadata) != prior_metadata_identity:
                    _fail("existing staged metadata changed before replacement commit")
                if release._read_json(metadata_path) != prior_metadata:
                    _fail("existing staged metadata contents changed before replacement commit")
                os.replace(candidate, shard_path)
                private_candidate = None
                publication_committed = True
            else:
                try:
                    os.link(candidate, shard_path, follow_symlinks=False)
                except FileExistsError as exc:
                    raise ShardImportError(
                        f"refusing to overwrite existing staged shard: {shard_path}"
                    ) from exc
                publication_committed = True
                candidate.unlink()
                private_candidate = None

            _fsync_directory_strict(stage_dir)
            committed = shard_path.lstat()
            if (
                not stat.S_ISREG(committed.st_mode)
                or _linked_content_identity(committed)
                != _linked_content_identity(candidate_info)
            ):
                _fail("atomic publication did not retain the authenticated CoW clone")
            if replacing:
                assert prior_metadata is not None
                if release._read_json(metadata_path) != prior_metadata:
                    _fail("trusted staged metadata changed during replacement commit")
            elif release._read_json(metadata_path) != sidecar:
                _fail("staged shard metadata changed during publication")

            # The clone is now an independently authenticated inode.  Remove
            # only the original pathname; never unlink a replacement file that
            # appeared there while the import was running.
            try:
                current = source.lstat()
            except FileNotFoundError:
                source_removed = True
            else:
                if _inode_identity(current) == _inode_identity(after_hash):
                    _unlink_source(source)
                    source_removed = True
                else:
                    report(
                        f"[import] source pathname was replaced concurrently and was left "
                        f"untouched: {source}"
                    )
            if source_removed:
                _fsync_directory_strict(source.parent)

        result = ImportResult(
            pack=match.pin.key,
            asset=str(match.shard["asset"]),
            file_name=match.file_name,
            staged_path=shard_path,
            bytes=expected_bytes,
            sha256=expected_sha256,
        )
    except BaseException as exc:
        if source_removed or publication_committed:
            raise ShardImportError(
                f"the verified source was committed to staging at {shard_path}, but "
                f"finalization reported an error; the complete staged data was preserved: {exc}"
            ) from exc
        cleanup_errors: list[str] = []
        preserved_candidate: Path | None = None
        if private_candidate is not None:
            try:
                current_source = source.lstat()
            except OSError:
                current_source = None
            if current_source is not None and _identity(current_source) == _identity(after_hash):
                issue = _unlink_if_owned(*private_candidate)
                if issue:
                    cleanup_errors.append(issue)
            else:
                # The source name no longer proves that a complete original
                # remains. Keep the private clone as the recovery copy.
                preserved_candidate = private_candidate[0]
        if metadata_created is not None:
            issue = _unlink_if_owned(*metadata_created)
            if issue:
                cleanup_errors.append(issue)
        if preserved_candidate is not None:
            raise ShardImportError(
                f"{exc}; the source path was no longer the verified original, so the private "
                f"CoW recovery copy was preserved at {preserved_candidate}"
            ) from exc
        if cleanup_errors:
            raise ShardImportError(
                f"{exc}; failure cleanup was incomplete: {'; '.join(cleanup_errors)}"
            ) from exc
        raise
    finally:
        os.close(descriptor)

    try:
        source_note = (
            "external source removed"
            if source_removed
            else "verified clone staged; concurrently replaced source path left untouched"
        )
        report(
            f"[import] {result.pack}/{result.asset}: moved to {result.staged_path}; "
            f"{source_note}"
        )
    except Exception:
        # Reporting is outside the committed filesystem transaction.
        pass
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and move one externally downloaded official Phosphene shard "
            "into the pinned installer's private staging layout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="absolute path to a completed file whose basename exactly matches a release asset",
    )
    parser.add_argument(
        "--pack",
        choices=_supported_pack_order(),
        help="expected pack; omitted to detect it from the exact asset basename",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=release.DEFAULT_MODELS_ROOT,
        help="same models root used by install_phosphene_ltx25_release.py",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="timeout for fetching only the small pinned manifest",
    )
    parser.add_argument(
        "--replace-partial",
        action="store_true",
        help=(
            "replace an existing trusted incomplete shard only after the external "
            "full shard passes exact size and SHA-256 verification"
        ),
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import_shard(
            args.source,
            pack_key=args.pack,
            models_root=args.models_root,
            timeout=args.timeout,
            replace_partial=args.replace_partial,
        )
    except release.InstallerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
