#!/usr/bin/env python3
"""Securely install the pinned Phosphene LTX-2.5 GitHub release packs.

The downloader is stdlib-only, sends no credentials, resumes inside a private
staging directory, checks the release manifest plus every shard and assembled
file, and only then atomically publishes a file under its loader-visible name.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Mapping, NoReturn, Sequence

from studio_paths import PATHS


WORKSPACE = PATHS.workspace
DEFAULT_MODELS_ROOT = PATHS.models_root
RELEASE_REPOSITORY = "mrbizarro/Phosphene"
RELEASE_TAG = "weights-ltx25-v1"
RELEASE_BASE = (
    f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}"
)
USER_AGENT = "local-video-phosphene-installer/1"
MANIFEST_LIMIT = 1 << 20
CHUNK_BYTES = 8 << 20
PROGRESS_INTERVAL_SECONDS = 2.5
DEFAULT_RESERVE_BYTES = 2 * 2**30
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]*$")
VERIFIED_SHARD_SHA256_FIELD = "verified_sha256"


class InstallerError(RuntimeError):
    """The requested operation could not be completed safely."""


class ManifestError(InstallerError):
    """A release manifest did not match its immutable local pin or schema."""


class DownloadError(InstallerError):
    """A release asset failed transport or integrity validation."""


class VerificationError(InstallerError):
    """A local file did not match its pinned manifest entry."""


@dataclass(frozen=True)
class PackPin:
    key: str
    label: str
    directory: str
    manifest_asset: str
    manifest_bytes: int
    manifest_sha256: str

    @property
    def manifest_url(self) -> str:
        return f"{RELEASE_BASE}/{self.manifest_asset}"


@dataclass
class DownloadedRange:
    """One validated range held privately until the ordered writer consumes it."""

    start: int
    end: int
    validator: tuple[str, str]
    handle: BinaryIO
    path: Path

    def close(self) -> None:
        try:
            self.handle.close()
        finally:
            _unlink_internal(self.path)


# These hashes pin the exact release-manifest bytes published for the official
# release. A mutable tag or replaced GitHub asset therefore cannot silently
# redefine either pack or any of its shard hashes.
PACKS: dict[str, PackPin] = {
    "q4_25": PackPin(
        key="q4_25",
        label="LTX 2.5 Q4 base model",
        directory="ltx-2.5-mlx-q4",
        manifest_asset="q4_25__phosphene_release_manifest.json",
        manifest_bytes=9_430,
        manifest_sha256="0013bfab7524d61d7c0e3d95733e6453cbab9dde944eb5a9100db8bf531ab424",
    ),
    "gemma4_25": PackPin(
        key="gemma4_25",
        label="Gemma 4 12B Q4 text encoder",
        directory="gemma4-12b-ltx25-q4",
        manifest_asset="gemma4_25__phosphene_release_manifest.json",
        manifest_bytes=5_387,
        manifest_sha256="b995e479a8e3d2c08af35572fccf9412fad8b3c4f6ef3f2934d733f6596472d5",
    ),
}
PACK_ORDER = tuple(PACKS)

LICENSE_NAME = "LICENSE-LTX-2.x-Community-License.md"
STALE_LICENSE_SPEC: dict[str, object] = {
    "bytes": 30_938,
    "sha256": "4e7dd26d9c9294e1e0549587505dcf4ba40c7c82c50850559f846f92245f6120",
    "shards": [
        {
            "asset": LICENSE_NAME,
            "bytes": 30_938,
            "sha256": "4e7dd26d9c9294e1e0549587505dcf4ba40c7c82c50850559f846f92245f6120",
        }
    ],
}
# The publisher replaced this shared release asset on 2026-09-06 without
# regenerating either pack manifest. This explicit second trust root handles
# only that legal-text file. Weight entries never use this exception.
CURRENT_LICENSE_SPEC: dict[str, object] = {
    "bytes": 34_545,
    "sha256": "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654",
    "shards": [
        {
            "asset": LICENSE_NAME,
            "bytes": 34_545,
            "sha256": "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654",
        }
    ],
}
CURRENT_LICENSE_PIN = str(CURRENT_LICENSE_SPEC["sha256"])


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> NoReturn:
    raise InstallerError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_region(handle: BinaryIO, length: int) -> str:
    remaining = length
    digest = hashlib.sha256()
    while remaining:
        block = handle.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise VerificationError("staged assembly ended inside a declared shard")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _safe_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value):
        raise ManifestError(f"unsafe {field}: {value!r}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ManifestError(f"unsafe {field}: {value!r}")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} must be a positive integer")
    return value


def _sha256_value(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_manifest(manifest: object, pin: PackPin) -> dict:
    """Return a strictly validated manifest bound to ``pin``."""
    if not isinstance(manifest, dict):
        raise ManifestError(f"{pin.key}: manifest root is not an object")
    if manifest.get("schema") != "phosphene-release-manifest/1":
        raise ManifestError(f"{pin.key}: unsupported manifest schema")
    if manifest.get("repo_key") != pin.key:
        raise ManifestError(f"{pin.key}: manifest repo_key mismatch")
    if manifest.get("pack") != pin.directory:
        raise ManifestError(f"{pin.key}: manifest pack directory mismatch")
    release = manifest.get("release")
    if not isinstance(release, dict):
        raise ManifestError(f"{pin.key}: missing release identity")
    expected_release = {
        "release_repo": RELEASE_REPOSITORY,
        "tag": RELEASE_TAG,
        "asset_prefix": f"{pin.key}__",
    }
    for field, expected in expected_release.items():
        if release.get(field) != expected:
            raise ManifestError(f"{pin.key}: release {field} mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ManifestError(f"{pin.key}: manifest declares no files")
    seen_assets: set[str] = set()
    total_bytes = 0
    for raw_name, raw_spec in files.items():
        name = _safe_name(raw_name, field="file name")
        if not isinstance(raw_spec, dict):
            raise ManifestError(f"{pin.key}/{name}: file spec is not an object")
        file_bytes = _positive_int(raw_spec.get("bytes"), field=f"{name}.bytes")
        _sha256_value(raw_spec.get("sha256"), field=f"{name}.sha256")
        shards = raw_spec.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ManifestError(f"{pin.key}/{name}: no shards declared")
        shard_total = 0
        for index, raw_shard in enumerate(shards):
            if not isinstance(raw_shard, dict):
                raise ManifestError(f"{pin.key}/{name}: shard {index} is not an object")
            asset = _safe_name(raw_shard.get("asset"), field=f"{name}.shards[{index}].asset")
            if asset in seen_assets:
                raise ManifestError(f"{pin.key}: duplicate release asset {asset!r}")
            seen_assets.add(asset)
            shard_total += _positive_int(
                raw_shard.get("bytes"), field=f"{name}.shards[{index}].bytes"
            )
            _sha256_value(
                raw_shard.get("sha256"), field=f"{name}.shards[{index}].sha256"
            )
        if shard_total != file_bytes:
            raise ManifestError(
                f"{pin.key}/{name}: shard bytes {shard_total} != file bytes {file_bytes}"
            )
        total_bytes += file_bytes
    if total_bytes > 100_000_000_000:
        raise ManifestError(f"{pin.key}: unreasonable declared pack size")
    if files.get(LICENSE_NAME) != STALE_LICENSE_SPEC:
        raise ManifestError(
            f"{pin.key}: the license exception is permitted only for the exact known stale entry"
        )
    return manifest


def pack_files(manifest: Mapping[str, object]) -> dict[str, dict]:
    """Weight-pack files, excluding only the exact validated stale license."""
    files = manifest["files"]
    assert isinstance(files, dict)
    if files.get(LICENSE_NAME) != STALE_LICENSE_SPEC:
        raise ManifestError("refusing to apply the stale-license exception to an unknown entry")
    return {
        str(name): spec
        for name, spec in files.items()
        if name != LICENSE_NAME and isinstance(spec, dict)
    }


class HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow normal redirects but reject HTTPS-to-plaintext downgrades."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        if urllib.parse.urlsplit(resolved).scheme.lower() != "https":
            raise urllib.error.HTTPError(
                req.full_url, code, "refusing non-HTTPS redirect", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, resolved)


class _CaseInsensitiveHeaders(dict[str, str]):
    def get(self, key: str, default=None):  # noqa: ANN001
        return super().get(key.lower(), default)


class CurlResponse:
    """Small in-memory response used only for manifests and byte probes."""

    def __init__(self, body: bytes, status: int, headers: Mapping[str, str], url: str):
        import io

        self._body = io.BytesIO(body)
        self.status = status
        self.headers = _CaseInsensitiveHeaders(
            {str(key).lower(): str(value) for key, value in headers.items()}
        )
        self._url = url

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self._body.close()


def _parse_curl_headers(raw: bytes) -> tuple[int, _CaseInsensitiveHeaders]:
    normalized = raw.replace(b"\r\n", b"\n")
    blocks = [block for block in normalized.split(b"\n\n") if block.startswith(b"HTTP/")]
    if not blocks:
        raise DownloadError("curl returned no HTTP response headers")
    lines = blocks[-1].decode("iso-8859-1").splitlines()
    try:
        status = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise DownloadError("curl returned malformed HTTP status") from exc
    headers = _CaseInsensitiveHeaders()
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return status, headers


class CurlOpener:
    """curl transport using macOS system trust and no ambient curl config."""

    def __init__(self, executable: str, *, force_ipv6: bool = False):
        self.executable = executable
        self.force_ipv6 = force_ipv6

    def _base_command(self, timeout: int) -> list[str]:
        # --disable must be the first option so ~/.curlrc cannot inject auth,
        # alternate protocols, or other behavior into this installer.
        command = [
            self.executable,
            "--disable",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--location",
            "--fail",
            "--show-error",
            "--connect-timeout",
            str(timeout),
            "--speed-limit",
            "1",
            "--speed-time",
            str(timeout),
        ]
        if self.force_ipv6:
            command.append("--ipv6")
        return command

    def open(self, request: urllib.request.Request, timeout: int) -> CurlResponse:
        with tempfile.TemporaryDirectory(prefix="phosphene-curl-") as temporary:
            headers_path = Path(temporary) / "headers"
            command = self._base_command(timeout) + [
                "--silent",
                "--max-filesize",
                str(MANIFEST_LIMIT),
                "--dump-header",
                str(headers_path),
                "--output",
                "-",
            ]
            for name, value in request.header_items():
                command.extend(["--header", f"{name}: {value}"])
            command.extend(["--", request.full_url])
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                raise urllib.error.URLError(f"curl exited {result.returncode}")
            try:
                raw_headers = headers_path.read_bytes()
            except OSError as exc:
                raise urllib.error.URLError("curl did not save response headers") from exc
            status, headers = _parse_curl_headers(raw_headers)
            return CurlResponse(result.stdout, status, headers, request.full_url)

    def download(
        self,
        request: urllib.request.Request,
        destination: Path,
        *,
        offset: int,
        timeout: int,
    ) -> tuple[int, _CaseInsensitiveHeaders]:
        """Stream one asset directly to its private partial path."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phosphene-curl-") as temporary:
            headers_path = Path(temporary) / "headers"
            command = self._base_command(timeout) + [
                "--progress-bar",
                "--dump-header",
                str(headers_path),
                "--output",
                str(destination),
            ]
            if offset:
                command.extend(["--continue-at", str(offset)])
            for name, value in request.header_items():
                # curl creates the Range request itself for --continue-at.
                if name.lower() == "range":
                    continue
                command.extend(["--header", f"{name}: {value}"])
            command.extend(["--", request.full_url])
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise DownloadError(f"curl exited {result.returncode}")
            try:
                return _parse_curl_headers(headers_path.read_bytes())
            except OSError as exc:
                raise DownloadError("curl did not save response headers") from exc

    def download_range(
        self,
        request: urllib.request.Request,
        destination: BinaryIO,
        *,
        expected_bytes: int,
        timeout: int,
    ) -> tuple[int, _CaseInsensitiveHeaders]:
        """Write one explicitly bounded range into an already-open private file."""
        with tempfile.TemporaryDirectory(prefix="phosphene-curl-") as temporary:
            headers_path = Path(temporary) / "headers"
            command = self._base_command(timeout) + [
                "--silent",
                # Bounded ranges should finish quickly.  A trickling CDN
                # connection can stay above curl's low-speed floor forever,
                # so also cap the complete request and let the caller retry
                # the untouched private chunk through a fresh connection.
                "--max-time",
                str(timeout),
                "--max-filesize",
                str(expected_bytes),
                "--dump-header",
                str(headers_path),
                "--output",
                "-",
            ]
            for name, value in request.header_items():
                if name.lower() in {"authorization", "proxy-authorization", "cookie"}:
                    raise DownloadError(f"refusing credential-bearing {name} header")
                command.extend(["--header", f"{name}: {value}"])
            command.extend(["--", request.full_url])
            result = subprocess.run(command, stdout=destination, check=False)
            if result.returncode != 0:
                raise DownloadError(f"curl exited {result.returncode}")
            try:
                return _parse_curl_headers(headers_path.read_bytes())
            except OSError as exc:
                raise DownloadError("curl did not save response headers") from exc


def build_opener(*, force_ipv6: bool = False) -> CurlOpener:
    # Apple's curl uses the system trust configuration, including managed
    # enterprise roots. Python.org's bundled CA store commonly does not.
    executable = shutil.which("curl")
    if not executable:
        raise InstallerError("curl is required for system-trust HTTPS but was not found")
    return CurlOpener(executable, force_ipv6=force_ipv6)


def _status(response: object) -> int:
    value = getattr(response, "status", None)
    if value is None:
        value = response.getcode()  # type: ignore[attr-defined]
    return int(value)


def _header(response: object, name: str) -> str | None:
    value = response.headers.get(name)  # type: ignore[attr-defined]
    return str(value) if value is not None else None


def _response_url(response: object, fallback: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if getter is not None else fallback


def _assert_https_response(response: object, fallback_url: str) -> None:
    final_url = _response_url(response, fallback_url)
    if urllib.parse.urlsplit(final_url).scheme.lower() != "https":
        raise DownloadError("server redirected to a non-HTTPS URL")


def _safe_network_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return f"network error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def fetch_pinned_manifest(
    pin: PackPin,
    *,
    opener: object,
    timeout: int,
) -> tuple[dict, bytes]:
    request = urllib.request.Request(
        pin.manifest_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            _assert_https_response(response, pin.manifest_url)
            if _status(response) != 200:
                raise DownloadError(f"{pin.key}: manifest returned HTTP {_status(response)}")
            encoding = (_header(response, "Content-Encoding") or "identity").lower()
            if encoding not in {"", "identity"}:
                raise DownloadError(f"{pin.key}: manifest used unexpected content encoding")
            raw = response.read(MANIFEST_LIMIT + 1)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise DownloadError(f"{pin.key}: could not fetch manifest ({_safe_network_error(exc)})") from exc
    if len(raw) > MANIFEST_LIMIT:
        raise ManifestError(f"{pin.key}: manifest exceeds the safety limit")
    if len(raw) != pin.manifest_bytes:
        raise ManifestError(
            f"{pin.key}: manifest size mismatch ({len(raw)} != {pin.manifest_bytes})"
        )
    digest = sha256_bytes(raw)
    if digest != pin.manifest_sha256:
        raise ManifestError(
            f"{pin.key}: manifest SHA-256 mismatch ({digest} != {pin.manifest_sha256})"
        )
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{pin.key}: pinned manifest is not valid UTF-8 JSON") from exc
    return validate_manifest(decoded, pin), raw


def asset_url(asset: str) -> str:
    return f"{RELEASE_BASE}/{urllib.parse.quote(asset, safe='')}"


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = CONTENT_RANGE_RE.fullmatch(value.strip())
    if not match:
        return None
    return tuple(int(group) for group in match.groups())  # type: ignore[return-value]


def _validator(response: object) -> tuple[str, str] | None:
    etag = _header(response, "ETag")
    if etag:
        return "etag", etag
    modified = _header(response, "Last-Modified")
    if modified:
        return "last-modified", modified
    return None


def _request_headers(*, offset: int = 0, validator: tuple[str, str] | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if validator:
            headers["If-Range"] = validator[1]
    return headers


def _range_request_headers(
    *,
    start: int,
    end: int,
    validator: tuple[str, str] | None,
) -> dict[str, str]:
    headers = _request_headers()
    headers["Range"] = f"bytes={start}-{end}"
    if validator:
        headers["If-Range"] = validator[1]
    return headers


def _stable_range_validator(response: object) -> tuple[str, str] | None:
    """Return an If-Range-safe validator, preferring a strong ETag."""
    etag = _header(response, "ETag")
    if etag and not etag.lstrip().lower().startswith("w/"):
        return "etag", etag
    modified = _header(response, "Last-Modified")
    if modified:
        return "last-modified", modified
    return None


def _metadata_range_validator(
    metadata: Mapping[str, object] | None,
) -> tuple[str, str] | None:
    validator = _metadata_validator(metadata)
    if validator and validator[0] == "etag" and validator[1].lstrip().lower().startswith("w/"):
        return None
    return validator


def probe_small_asset(
    pin: PackPin,
    manifest: Mapping[str, object],
    *,
    opener: object,
    timeout: int,
) -> str:
    """Exercise the release CDN using the smallest declared asset only."""
    candidates: list[tuple[int, str, str]] = []
    files = manifest["files"]
    assert isinstance(files, dict)
    for spec in files.values():
        assert isinstance(spec, dict)
        shards = spec["shards"]
        assert isinstance(shards, list)
        for shard in shards:
            assert isinstance(shard, dict)
            candidates.append((int(shard["bytes"]), str(shard["asset"]), str(shard["sha256"])))
    expected_bytes, asset, expected_sha = min(candidates)
    if expected_bytes > 64 << 10:
        raise ManifestError(f"{pin.key}: no suitably small preflight asset")
    url = asset_url(asset)
    request = urllib.request.Request(url, headers=_request_headers(offset=0))
    request.add_header("Range", "bytes=0-0")
    try:
        with opener.open(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            _assert_https_response(response, url)
            status = _status(response)
            encoding = (_header(response, "Content-Encoding") or "identity").lower()
            if encoding not in {"", "identity"}:
                raise DownloadError(f"{pin.key}: probe used unexpected content encoding")
            if status == 206:
                content_range = _parse_content_range(_header(response, "Content-Range"))
                if content_range != (0, 0, expected_bytes):
                    raise DownloadError(
                        f"{pin.key}: unsafe probe Content-Range {_header(response, 'Content-Range')!r}"
                    )
                body = response.read(2)
                if len(body) != 1:
                    raise DownloadError(f"{pin.key}: probe did not return exactly one byte")
                return f"HTTP 206 range supported ({asset}, 1 byte)"
            if status == 200:
                body = response.read(expected_bytes + 1)
                if len(body) != expected_bytes or sha256_bytes(body) != expected_sha:
                    raise DownloadError(f"{pin.key}: range fallback asset failed integrity verification")
                return f"HTTP 200 range ignored; small fallback verified ({asset}, {expected_bytes} bytes)"
            raise DownloadError(f"{pin.key}: probe returned HTTP {status}")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise DownloadError(f"{pin.key}: CDN probe failed ({_safe_network_error(exc)})") from exc


def fetch_current_license(*, opener: object, timeout: int) -> bytes:
    """Fetch and verify the separately pinned current release license."""
    url = asset_url(LICENSE_NAME)
    request = urllib.request.Request(url, headers=_request_headers())
    try:
        with opener.open(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            _assert_https_response(response, url)
            if _status(response) != 200:
                raise DownloadError(f"current license returned HTTP {_status(response)}")
            raw = response.read(int(CURRENT_LICENSE_SPEC["bytes"]) + 1)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise DownloadError(f"could not fetch current license ({_safe_network_error(exc)})") from exc
    if len(raw) != CURRENT_LICENSE_SPEC["bytes"]:
        raise VerificationError(
            f"current license size mismatch ({len(raw)} != {CURRENT_LICENSE_SPEC['bytes']})"
        )
    digest = sha256_bytes(raw)
    if digest != CURRENT_LICENSE_PIN:
        raise VerificationError(
            f"current license SHA-256 mismatch ({digest} != {CURRENT_LICENSE_PIN})"
        )
    return raw


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    if temporary.is_symlink():
        raise InstallerError(f"refusing symlinked metadata path: {temporary}")
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.replace(temporary, path)


def _read_json(path: Path) -> dict | None:
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _unlink_internal(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            raise InstallerError(f"refusing to unlink unexpected staging directory: {path}")
        path.unlink()
    except FileNotFoundError:
        pass


def _regular_or_missing(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise InstallerError(f"refusing symlinked {label}: {path}")
    if path.exists() and not path.is_file():
        raise InstallerError(f"refusing non-file {label}: {path}")


def _shard_identity(
    pin: PackPin,
    manifest_digest: str,
    shard: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "local-video-phosphene-shard/1",
        "repo_key": pin.key,
        "manifest_sha256": manifest_digest,
        "asset": shard["asset"],
        "bytes": shard["bytes"],
        "sha256": shard["sha256"],
    }


def _resume_metadata_matches(
    metadata: Mapping[str, object] | None,
    identity: Mapping[str, object],
) -> bool:
    return bool(metadata) and all(metadata.get(key) == value for key, value in identity.items())


def _metadata_validator(metadata: Mapping[str, object] | None) -> tuple[str, str] | None:
    if not metadata:
        return None
    kind = metadata.get("validator_kind")
    value = metadata.get("validator")
    if kind in {"etag", "last-modified"} and isinstance(value, str) and value:
        return str(kind), value
    return None


def _validate_download_response(
    response: object,
    *,
    offset: int,
    expected_bytes: int,
    previous_validator: tuple[str, str] | None,
) -> tuple[str, int, tuple[str, str] | None]:
    status = _status(response)
    encoding = (_header(response, "Content-Encoding") or "identity").lower()
    if encoding not in {"", "identity"}:
        raise DownloadError("asset response used unexpected content encoding")
    current_validator = _validator(response)
    if offset and status == 206:
        expected_range = (offset, expected_bytes - 1, expected_bytes)
        actual_range = _parse_content_range(_header(response, "Content-Range"))
        if actual_range != expected_range:
            raise DownloadError(
                f"unsafe resume Content-Range {_header(response, 'Content-Range')!r}; "
                f"expected bytes {offset}-{expected_bytes - 1}/{expected_bytes}"
            )
        if previous_validator and current_validator and current_validator != previous_validator:
            raise DownloadError("resume validator changed during partial response")
        declared = _header(response, "Content-Length")
        if declared is not None and int(declared) != expected_bytes - offset:
            raise DownloadError("resume Content-Length does not match the pinned size")
        return "append", offset, current_validator or previous_validator
    if status == 200:
        declared = _header(response, "Content-Length")
        if declared is not None and int(declared) != expected_bytes:
            raise DownloadError("asset Content-Length does not match the pinned size")
        return "restart", 0, current_validator
    if not offset and status == 206:
        actual_range = _parse_content_range(_header(response, "Content-Range"))
        if actual_range != (0, expected_bytes - 1, expected_bytes):
            raise DownloadError("unexpected partial response for a new shard")
        return "restart", 0, current_validator
    raise DownloadError(f"asset returned unexpected HTTP {status}")


def _validate_exact_range_response(
    response: object,
    *,
    start: int,
    end: int,
    expected_bytes: int,
    previous_validator: tuple[str, str] | None,
) -> tuple[str, str]:
    if _status(response) != 206:
        raise DownloadError(f"range request returned HTTP {_status(response)} instead of 206")
    encoding = (_header(response, "Content-Encoding") or "identity").lower()
    if encoding not in {"", "identity"}:
        raise DownloadError("range response used unexpected content encoding")
    actual_range = _parse_content_range(_header(response, "Content-Range"))
    expected_range = (start, end, expected_bytes)
    if actual_range != expected_range:
        raise DownloadError(
            f"unsafe range Content-Range {_header(response, 'Content-Range')!r}; "
            f"expected bytes {start}-{end}/{expected_bytes}"
        )
    declared = _header(response, "Content-Length")
    if declared is None:
        raise DownloadError("range response omitted Content-Length")
    try:
        declared_bytes = int(declared)
    except ValueError as exc:
        raise DownloadError("range response used malformed Content-Length") from exc
    range_bytes = end - start + 1
    if declared_bytes != range_bytes:
        raise DownloadError(
            f"range Content-Length {declared_bytes} does not match requested {range_bytes}"
        )
    validator = _stable_range_validator(response)
    if validator is None:
        raise DownloadError("range response supplied no stable ETag or Last-Modified validator")
    if previous_validator and validator != previous_validator:
        raise DownloadError("range validator changed between chunks")
    return validator


def _stream_exact_range(
    opener: object,
    request: urllib.request.Request,
    destination: BinaryIO,
    *,
    expected_bytes: int,
    timeout: int,
) -> CurlResponse:
    """Stream one bounded response without placing it in the shard partial."""
    if isinstance(opener, CurlOpener):
        status, headers = opener.download_range(
            request,
            destination,
            expected_bytes=expected_bytes,
            timeout=timeout,
        )
        # CurlOpener permits only HTTPS for both the initial request and every
        # redirect. The response object retains the original HTTPS URL.
        return CurlResponse(b"", status, headers, request.full_url)

    with opener.open(request, timeout=timeout) as response:  # type: ignore[attr-defined]
        _assert_https_response(response, request.full_url)
        status = _status(response)
        headers = _CaseInsensitiveHeaders(
            {str(key).lower(): str(value) for key, value in response.headers.items()}
        )
        final_url = _response_url(response, request.full_url)
        remaining = expected_bytes + 1
        while remaining:
            block = response.read(min(CHUNK_BYTES, remaining))
            if not block:
                break
            destination.write(block)
            remaining -= len(block)
    return CurlResponse(b"", status, headers, final_url)


def _append_range_chunk(
    source: BinaryIO,
    download_path: Path,
    *,
    offset: int,
    chunk_bytes: int,
) -> None:
    """Append one already validated chunk and durably record its new boundary."""
    _regular_or_missing(download_path, label="shard partial")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        write_flags |= os.O_NOFOLLOW
    output_descriptor = os.open(download_path, write_flags, 0o600)
    with os.fdopen(output_descriptor, "ab") as output:
        if os.fstat(output.fileno()).st_size != offset:
            raise DownloadError("shard partial changed while a range was in flight")
        if os.fstat(source.fileno()).st_size != chunk_bytes:
            raise DownloadError("private range chunk changed before append")
        source.seek(0)
        shutil.copyfileobj(source, output, CHUNK_BYTES)
        output.flush()
        os.fsync(output.fileno())
        actual_size = os.fstat(output.fileno()).st_size
        if actual_size != offset + chunk_bytes:
            raise DownloadError(
                f"range append ended at {actual_size}; expected {offset + chunk_bytes}"
            )
    _fsync_directory(download_path.parent)


def _fetch_validated_range(
    asset: str,
    *,
    start: int,
    end: int,
    total_bytes: int,
    previous_validator: tuple[str, str] | None,
    opener: object,
    attempts: int,
    timeout: int,
    temporary_directory: Path,
    cancel_event: threading.Event,
    report: Callable[[str], None],
) -> DownloadedRange:
    """Fetch one range privately; the caller owns the returned open handle."""
    wanted = end - start + 1
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".phosphene-range-",
        suffix=".chunk",
        dir=temporary_directory,
    )
    os.fchmod(descriptor, 0o600)
    chunk_path = Path(temporary_name)
    handle: BinaryIO | None = None
    try:
        handle = os.fdopen(descriptor, "w+b")
        descriptor = -1
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            if cancel_event.is_set():
                raise DownloadError(f"{asset}: range batch cancelled")
            handle.seek(0)
            handle.truncate(0)
            request = urllib.request.Request(
                asset_url(asset),
                headers=_range_request_headers(
                    start=start,
                    end=end,
                    validator=previous_validator,
                ),
            )
            try:
                response = _stream_exact_range(
                    opener,
                    request,
                    handle,
                    expected_bytes=wanted,
                    timeout=timeout,
                )
                handle.flush()
                os.fsync(handle.fileno())
                validator = _validate_exact_range_response(
                    response,
                    start=start,
                    end=end,
                    expected_bytes=total_bytes,
                    previous_validator=previous_validator,
                )
                actual_chunk_bytes = os.fstat(handle.fileno()).st_size
                if actual_chunk_bytes != wanted:
                    raise DownloadError(
                        f"range body ended at {actual_chunk_bytes} bytes; expected {wanted}"
                    )
                return DownloadedRange(start, end, validator, handle, chunk_path)
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, InstallerError) as exc:
                last_error = _safe_network_error(exc) if not isinstance(exc, InstallerError) else str(exc)
                if attempt < attempts:
                    delay = min(2 * attempt, 10)
                    report(
                        f"[download] {asset}: range {start}-{end} attempt "
                        f"{attempt}/{attempts} failed ({last_error}); retrying in {delay}s"
                    )
                    if cancel_event.wait(delay):
                        raise DownloadError(f"{asset}: range batch cancelled")
        raise DownloadError(
            f"{asset}: range {start}-{end} failed after {attempts} attempt(s): {last_error}"
        )
    except BaseException:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)
        _unlink_internal(chunk_path)
        raise


def _fetch_range_batch(
    asset: str,
    ranges: Sequence[tuple[int, int]],
    *,
    total_bytes: int,
    previous_validator: tuple[str, str] | None,
    opener: object,
    attempts: int,
    timeout: int,
    temporary_directory: Path,
    range_workers: int,
    report: Callable[[str], None],
) -> list[DownloadedRange]:
    """Fetch one bounded batch concurrently without modifying the shard partial."""
    cancel_event = threading.Event()
    if len(ranges) == 1:
        start, end = ranges[0]
        return [
            _fetch_validated_range(
                asset,
                start=start,
                end=end,
                total_bytes=total_bytes,
                previous_validator=previous_validator,
                opener=opener,
                attempts=attempts,
                timeout=timeout,
                temporary_directory=temporary_directory,
                cancel_event=cancel_event,
                report=report,
            )
        ]
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(range_workers, len(ranges)),
        thread_name_prefix="phosphene-range",
    )
    futures: list[concurrent.futures.Future[DownloadedRange]] = []
    results: list[DownloadedRange] = []
    try:
        for start, end in ranges:
            futures.append(
                executor.submit(
                    _fetch_validated_range,
                    asset,
                    start=start,
                    end=end,
                    total_bytes=total_bytes,
                    previous_validator=previous_validator,
                    opener=opener,
                    attempts=attempts,
                    timeout=timeout,
                    temporary_directory=temporary_directory,
                    cancel_event=cancel_event,
                    report=report,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    except BaseException:
        cancel_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        owned = {id(result) for result in results}
        for future in futures:
            if future.cancelled() or not future.done():
                continue
            try:
                result = future.result()
            except BaseException:
                continue
            if id(result) not in owned:
                results.append(result)
                owned.add(id(result))
        for result in results:
            result.close()
        raise
    else:
        executor.shutdown(wait=True)
    return sorted(results, key=lambda result: result.start)


def _download_shard_in_ranges(
    shard: Mapping[str, object],
    download_path: Path,
    metadata_path: Path,
    identity: Mapping[str, object],
    *,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int,
    range_workers: int,
    report: Callable[[str], None],
) -> None:
    """Fetch exact ranges concurrently, then append them in strict offset order."""
    asset = str(shard["asset"])
    expected_bytes = int(shard["bytes"])
    expected_sha = str(shard["sha256"])
    while True:
        offset = download_path.stat().st_size if download_path.exists() else 0
        if offset >= expected_bytes:
            break
        metadata = _read_json(metadata_path)
        if offset and not _resume_metadata_matches(metadata, identity):
            raise DownloadError("range partial lost its pinned resume metadata")
        previous_validator = _metadata_range_validator(metadata)
        ranges: list[tuple[int, int]] = []
        cursor = offset
        while cursor < expected_bytes and len(ranges) < range_workers:
            end = min(cursor + range_chunk_bytes, expected_bytes) - 1
            ranges.append((cursor, end))
            cursor = end + 1

        chunks = _fetch_range_batch(
            asset,
            ranges,
            total_bytes=expected_bytes,
            previous_validator=previous_validator,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            temporary_directory=download_path.parent,
            range_workers=range_workers,
            report=report,
        )
        try:
            actual_ranges = [(chunk.start, chunk.end) for chunk in chunks]
            if actual_ranges != ranges:
                raise DownloadError("parallel range batch did not return every requested chunk")
            validators = {chunk.validator for chunk in chunks}
            if len(validators) != 1:
                raise DownloadError("range validator changed within parallel batch")
            validator = next(iter(validators))
            if previous_validator and validator != previous_validator:
                raise DownloadError("range validator changed between batches")

            persisted = dict(identity)
            persisted["validator_kind"], persisted["validator"] = validator
            _atomic_json(metadata_path, persisted)
            append_offset = offset
            for chunk in chunks:
                if chunk.start != append_offset:
                    raise DownloadError("parallel range chunks were not contiguous")
                wanted = chunk.end - chunk.start + 1
                _append_range_chunk(
                    chunk.handle,
                    download_path,
                    offset=append_offset,
                    chunk_bytes=wanted,
                )
                append_offset = chunk.end + 1
                report(
                    f"[download] {asset}: {append_offset} / {expected_bytes} bytes "
                    f"({100.0 * append_offset / expected_bytes:.1f}%), exact ranges"
                )
        finally:
            for chunk in chunks:
                chunk.close()

    actual_bytes = download_path.stat().st_size if download_path.exists() else 0
    if actual_bytes != expected_bytes:
        raise DownloadError(f"asset ended at {actual_bytes} bytes; expected {expected_bytes}")
    actual_sha = sha256_file(download_path)
    if actual_sha != expected_sha:
        _unlink_internal(download_path)
        _unlink_internal(metadata_path)
        raise VerificationError(f"shard SHA-256 mismatch ({actual_sha} != {expected_sha})")
    report(f"[download] {asset}: {expected_bytes} bytes, SHA-256 verified")


def download_shard(
    pin: PackPin,
    manifest_digest: str,
    shard: Mapping[str, object],
    download_path: Path,
    metadata_path: Path,
    *,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int = 0,
    range_workers: int = 1,
    report: Callable[[str], None] = log,
) -> None:
    """Resume and verify one shard without ever exposing it as a model file."""
    if range_chunk_bytes < 0:
        raise InstallerError("range chunk size cannot be negative")
    if range_workers < 1:
        raise InstallerError("range workers must be positive")
    asset = str(shard["asset"])
    expected_bytes = int(shard["bytes"])
    expected_sha = str(shard["sha256"])
    identity = _shard_identity(pin, manifest_digest, shard)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    _regular_or_missing(download_path, label="shard partial")
    _regular_or_missing(metadata_path, label="shard metadata")

    metadata = _read_json(metadata_path)
    if download_path.exists() and not _resume_metadata_matches(metadata, identity):
        report(f"[download] {asset}: discarding untrusted partial metadata")
        _unlink_internal(download_path)
        _unlink_internal(metadata_path)
        metadata = None
    if not download_path.exists() and metadata_path.exists():
        _unlink_internal(metadata_path)
        metadata = None
    if download_path.exists() and download_path.stat().st_size > expected_bytes:
        _unlink_internal(download_path)
        _unlink_internal(metadata_path)
        metadata = None
    if download_path.exists() and download_path.stat().st_size == expected_bytes:
        if sha256_file(download_path) == expected_sha:
            report(f"[download] {asset}: staged shard already verified")
            return
        _unlink_internal(download_path)
        _unlink_internal(metadata_path)
        metadata = None

    if range_chunk_bytes:
        _download_shard_in_ranges(
            shard,
            download_path,
            metadata_path,
            identity,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=range_workers,
            report=report,
        )
        return

    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        offset = download_path.stat().st_size if download_path.exists() else 0
        metadata = _read_json(metadata_path)
        if offset and not _resume_metadata_matches(metadata, identity):
            _unlink_internal(download_path)
            _unlink_internal(metadata_path)
            offset = 0
            metadata = None
        previous_validator = _metadata_validator(metadata)
        request = urllib.request.Request(
            asset_url(asset), headers=_request_headers(offset=offset, validator=previous_validator)
        )
        try:
            if isinstance(opener, CurlOpener):
                # Bind any bytes curl creates to this exact pinned shard before
                # starting the child process. An interrupted run can therefore
                # resume, but an unrelated partial can never be adopted.
                persisted = dict(identity)
                if previous_validator:
                    persisted["validator_kind"], persisted["validator"] = previous_validator
                _atomic_json(metadata_path, persisted)
                status, headers = opener.download(
                    request,
                    download_path,
                    offset=offset,
                    timeout=timeout,
                )
                response = CurlResponse(b"", status, headers, request.full_url)
                mode, _, validator = _validate_download_response(
                    response,
                    offset=offset,
                    expected_bytes=expected_bytes,
                    previous_validator=previous_validator,
                )
                if offset and mode != "append":
                    # curl should reject a server that ignores its resume range.
                    # If an unusual build accepts it, never guess how it wrote.
                    _unlink_internal(download_path)
                    _unlink_internal(metadata_path)
                    raise DownloadError("curl resume did not receive an exact partial response")
                if validator:
                    persisted["validator_kind"], persisted["validator"] = validator
                    _atomic_json(metadata_path, persisted)
            else:
                with opener.open(request, timeout=timeout) as response:  # type: ignore[attr-defined]
                    _assert_https_response(response, request.full_url)
                    mode, have, validator = _validate_download_response(
                        response,
                        offset=offset,
                        expected_bytes=expected_bytes,
                        previous_validator=previous_validator,
                    )
                    if mode == "restart" and offset:
                        report(f"[download] {asset}: server ignored resume; restarting this shard")
                    persisted = dict(identity)
                    if validator:
                        persisted["validator_kind"], persisted["validator"] = validator
                    _atomic_json(metadata_path, persisted)
                    flags = os.O_WRONLY | os.O_CREAT
                    flags |= os.O_APPEND if mode == "append" else os.O_TRUNC
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    descriptor = os.open(download_path, flags, 0o600)
                    with os.fdopen(descriptor, "ab" if mode == "append" else "wb") as handle:
                        last_report = 0.0
                        while True:
                            block = response.read(CHUNK_BYTES)
                            if not block:
                                break
                            have += len(block)
                            if have > expected_bytes:
                                raise DownloadError("asset exceeded its pinned byte size")
                            handle.write(block)
                            now = time.monotonic()
                            if now - last_report >= PROGRESS_INTERVAL_SECONDS:
                                report(
                                    f"[download] {asset}: {have / 1e9:.2f}/{expected_bytes / 1e9:.2f} GB "
                                    f"({100.0 * have / expected_bytes:.1f}%)"
                                )
                                last_report = now
                        handle.flush()
                        os.fsync(handle.fileno())
            actual_bytes = download_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise DownloadError(
                    f"asset ended at {actual_bytes} bytes; expected {expected_bytes}"
                )
            actual_sha = sha256_file(download_path)
            if actual_sha != expected_sha:
                _unlink_internal(download_path)
                _unlink_internal(metadata_path)
                raise VerificationError(
                    f"shard SHA-256 mismatch ({actual_sha} != {expected_sha})"
                )
            report(f"[download] {asset}: {expected_bytes} bytes, SHA-256 verified")
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, InstallerError) as exc:
            last_error = _safe_network_error(exc) if not isinstance(exc, InstallerError) else str(exc)
            if isinstance(exc, VerificationError):
                _unlink_internal(download_path)
                _unlink_internal(metadata_path)
            if isinstance(exc, DownloadError) and last_error in {"curl exited 33", "curl exited 36"}:
                # These are curl's range/resume errors. Restarting this private
                # shard is safe; appending an unconfirmed 200 response is not.
                _unlink_internal(download_path)
                _unlink_internal(metadata_path)
            if attempt < attempts:
                delay = min(2 * attempt, 10)
                report(
                    f"[download] {asset}: attempt {attempt}/{attempts} failed "
                    f"({last_error}); retrying in {delay}s"
                )
                time.sleep(delay)
    raise DownloadError(f"{asset}: failed after {attempts} attempt(s): {last_error}")


def _assembly_identity(
    pin: PackPin,
    manifest_digest: str,
    name: str,
    spec: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "local-video-phosphene-assembly/1",
        "repo_key": pin.key,
        "manifest_sha256": manifest_digest,
        "file": name,
        "bytes": spec["bytes"],
        "sha256": spec["sha256"],
    }


def recover_assembly(
    assembly_path: Path,
    metadata_path: Path,
    identity: Mapping[str, object],
    shards: Sequence[Mapping[str, object]],
) -> int:
    """Verify staged shard boundaries and return the safe completed count."""
    _regular_or_missing(assembly_path, label="assembly")
    _regular_or_missing(metadata_path, label="assembly metadata")
    metadata = _read_json(metadata_path)
    if assembly_path.exists() and not _resume_metadata_matches(metadata, identity):
        _unlink_internal(assembly_path)
        _unlink_internal(metadata_path)
        return 0
    if not assembly_path.exists():
        _unlink_internal(metadata_path)
        return 0

    actual_size = assembly_path.stat().st_size
    expected_total = sum(int(shard["bytes"]) for shard in shards)
    if actual_size > expected_total:
        _unlink_internal(assembly_path)
        _unlink_internal(metadata_path)
        return 0
    completed = 0
    verified_bytes = 0
    with assembly_path.open("rb") as handle:
        for shard in shards:
            shard_bytes = int(shard["bytes"])
            if actual_size - verified_bytes < shard_bytes:
                break
            digest = _sha256_region(handle, shard_bytes)
            if digest != shard["sha256"]:
                _unlink_internal(assembly_path)
                _unlink_internal(metadata_path)
                return 0
            completed += 1
            verified_bytes += shard_bytes
    if actual_size != verified_bytes:
        # A process can be killed while copying an already verified shard into
        # the assembly. Only the incomplete tail is discarded.
        with assembly_path.open("r+b") as handle:
            handle.truncate(verified_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    persisted = dict(identity)
    persisted.update({"shards_done": completed, "assembled_bytes": verified_bytes})
    _atomic_json(metadata_path, persisted)
    return completed


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_local_file(path: Path, spec: Mapping[str, object]) -> tuple[bool, str]:
    if path.is_symlink():
        return False, "unsafe symlink"
    if not path.exists():
        return False, "missing"
    if not path.is_file():
        return False, "not a regular file"
    expected_bytes = int(spec["bytes"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"size mismatch ({actual_bytes} != {expected_bytes})"
    actual_sha = sha256_file(path)
    if actual_sha != spec["sha256"]:
        return False, f"SHA-256 mismatch ({actual_sha} != {spec['sha256']})"
    return True, "size and SHA-256 verified"


def _stage_paths(
    models_root: Path,
    pin: PackPin,
    name: str,
    asset: str | None = None,
) -> tuple[Path, Path]:
    stage_dir = models_root / ".phosphene-release-staging" / pin.key
    stem = asset if asset is not None else name
    suffix = ".shard.partial" if asset is not None else ".assembling"
    partial = stage_dir / f"{stem}{suffix}"
    return partial, partial.with_name(partial.name + ".json")


@contextlib.contextmanager
def _file_staging_lock(
    models_root: Path,
    pin: PackPin,
    name: str,
    *,
    blocking: bool = True,
):
    """Serialize one file's download/assembly with external shard imports."""
    stage_dir = models_root / ".phosphene-release-staging" / pin.key
    stage_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stage_dir / f".{name}.staging.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise InstallerError(f"cannot open staging lock {lock_path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise InstallerError(f"refusing non-regular staging lock: {lock_path}")
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise InstallerError(
                f"another installer/importer owns staging for {pin.key}/{name}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _install_file_unlocked(
    pin: PackPin,
    manifest_digest: str,
    name: str,
    spec: Mapping[str, object],
    *,
    models_root: Path,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int = 0,
    range_workers: int = 1,
    report: Callable[[str], None] = log,
) -> bool:
    """Install one file; return True when bytes were promoted."""
    destination_dir = models_root / pin.directory
    target = destination_dir / name
    _regular_or_missing(target, label="destination")
    ready, detail = verify_local_file(target, spec)
    if ready:
        report(f"[install] {pin.key}/{name}: already {detail}; skipped")
        return False
    if target.exists():
        report(f"[install] {pin.key}/{name}: existing file is invalid ({detail}); preserving until replacement verifies")

    shards = spec["shards"]
    assert isinstance(shards, list)
    assembly_path, assembly_meta = _stage_paths(models_root, pin, name)
    assembly_path.parent.mkdir(parents=True, exist_ok=True)
    identity = _assembly_identity(pin, manifest_digest, name, spec)
    completed = recover_assembly(assembly_path, assembly_meta, identity, shards)
    if completed:
        report(f"[install] {pin.key}/{name}: resumed after {completed}/{len(shards)} verified shard(s)")
    elif not assembly_path.exists():
        _atomic_json(
            assembly_meta,
            {**identity, "shards_done": 0, "assembled_bytes": 0},
        )

    for index, shard in enumerate(shards):
        if index < completed:
            continue
        assert isinstance(shard, dict)
        asset = str(shard["asset"])
        shard_path, shard_meta = _stage_paths(models_root, pin, name, asset)
        download_shard(
            pin,
            manifest_digest,
            shard,
            shard_path,
            shard_meta,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=range_workers,
            report=report,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(assembly_path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as output, shard_path.open("rb") as source:
            shutil.copyfileobj(source, output, CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        completed = index + 1
        assembled_bytes = assembly_path.stat().st_size
        _atomic_json(
            assembly_meta,
            {**identity, "shards_done": completed, "assembled_bytes": assembled_bytes},
        )
        _unlink_internal(shard_path)
        _unlink_internal(shard_meta)
        report(f"[install] {pin.key}/{name}: assembled shard {completed}/{len(shards)}")

    expected_bytes = int(spec["bytes"])
    actual_bytes = assembly_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise VerificationError(
            f"{pin.key}/{name}: assembled size {actual_bytes} != {expected_bytes}"
        )
    actual_sha = sha256_file(assembly_path)
    if actual_sha != spec["sha256"]:
        raise VerificationError(
            f"{pin.key}/{name}: assembled SHA-256 {actual_sha} != {spec['sha256']}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    _regular_or_missing(target, label="destination")
    os.replace(assembly_path, target)
    _fsync_directory(destination_dir)
    _unlink_internal(assembly_meta)
    report(f"[install] {pin.key}/{name}: atomically installed; size and SHA-256 verified")
    return True


def install_file(
    pin: PackPin,
    manifest_digest: str,
    name: str,
    spec: Mapping[str, object],
    *,
    models_root: Path,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int = 0,
    range_workers: int = 1,
    report: Callable[[str], None] = log,
) -> bool:
    """Install one file while excluding a concurrent external shard import."""
    with _file_staging_lock(models_root, pin, name):
        return _install_file_unlocked(
            pin,
            manifest_digest,
            name,
            spec,
            models_root=models_root,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=range_workers,
            report=report,
        )


def pack_size(manifest: Mapping[str, object]) -> int:
    return sum(int(spec["bytes"]) for spec in pack_files(manifest).values()) + int(
        CURRENT_LICENSE_SPEC["bytes"]
    )


def pack_status(
    pin: PackPin,
    manifest: Mapping[str, object],
    *,
    models_root: Path,
) -> tuple[int, list[tuple[str, bool, str]]]:
    results: list[tuple[str, bool, str]] = []
    missing_bytes = 0
    for name, spec in pack_files(manifest).items():
        ready, detail = verify_local_file(models_root / pin.directory / name, spec)
        results.append((name, ready, detail))
        if not ready:
            missing_bytes += int(spec["bytes"])
    license_ready, license_detail = verify_local_file(
        models_root / pin.directory / LICENSE_NAME, CURRENT_LICENSE_SPEC
    )
    results.append((LICENSE_NAME, license_ready, f"current-license pin: {license_detail}"))
    if not license_ready:
        missing_bytes += int(CURRENT_LICENSE_SPEC["bytes"])
    return missing_bytes, results


def estimate_required_space(
    selected: Sequence[PackPin],
    manifests: Mapping[str, Mapping[str, object]],
    *,
    models_root: Path,
    reserve_bytes: int,
    jobs: int,
    range_chunk_bytes: int = 0,
    range_workers: int = 1,
    metadata_only: bool = False,
) -> int:
    missing = 0
    largest_shard = 0
    for pin in selected:
        manifest = manifests[pin.key]
        effective_files = {
            **pack_files(manifest),
            LICENSE_NAME: CURRENT_LICENSE_SPEC,
        }
        for name, spec in effective_files.items():
            if metadata_only and name.endswith(".safetensors"):
                continue
            ready, _ = verify_local_file(models_root / pin.directory / name, spec)
            if ready:
                continue
            missing += int(spec["bytes"])
            shards = spec["shards"]
            assert isinstance(shards, list)
            largest_shard = max(largest_shard, *(int(shard["bytes"]) for shard in shards))
    # Each worker can hold one fully downloaded shard while copying it into a
    # different file's assembly. Charging the largest shard for every worker is
    # deliberately conservative and keeps --jobs from hiding its disk cost.
    range_transient = min(range_chunk_bytes, largest_shard) if range_chunk_bytes else 0
    return missing + jobs * (
        largest_shard + range_workers * range_transient
    ) + reserve_bytes


def _receipt(pin: PackPin, manifest_digest: str, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "local-video-phosphene-install-receipt/1",
        "installed_utc": datetime.now(timezone.utc).isoformat(),
        "release_repository": RELEASE_REPOSITORY,
        "release_tag": RELEASE_TAG,
        "repo_key": pin.key,
        "pack": pin.directory,
        "manifest_asset": pin.manifest_asset,
        "manifest_sha256": manifest_digest,
        "license_exception": {
            "reason": "release asset replaced after manifest publication",
            "stale_manifest_sha256": STALE_LICENSE_SPEC["sha256"],
            "installed_sha256": CURRENT_LICENSE_PIN,
        },
        "files": len(pack_files(manifest)) + 1,
        "bytes": pack_size(manifest),
    }


def install_pack(
    pin: PackPin,
    manifest: Mapping[str, object],
    manifest_digest: str,
    *,
    models_root: Path,
    opener: object,
    attempts: int,
    timeout: int,
    jobs: int,
    range_chunk_bytes: int = 0,
    range_workers: int = 1,
    metadata_only: bool = False,
    report: Callable[[str], None] = log,
) -> None:
    declared_files = pack_files(manifest)
    files = {
        name: spec
        for name, spec in declared_files.items()
        if not metadata_only or not name.endswith(".safetensors")
    }
    report(
        f"[install] {pin.key}: {len(files) + 1} "
        f"{'metadata/license files' if metadata_only else 'files'}, "
        f"{(sum(int(spec['bytes']) for spec in files.values()) + int(CURRENT_LICENSE_SPEC['bytes'])) / 1e9:.2f} GB "
        f"-> {models_root / pin.directory}"
    )
    failures: list[str] = []

    def install_declared_file(name: str, spec: Mapping[str, object]) -> None:
        # Exactly one future owns each unique manifest name. Its shards remain
        # sequential inside install_file, preserving assembly order.
        install_file(
            pin,
            manifest_digest,
            name,
            spec,
            models_root=models_root,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=range_workers,
            report=report,
        )

    if jobs == 1:
        for name, spec in files.items():
            try:
                install_declared_file(name, spec)
            except InstallerError as exc:
                failures.append(name)
                report(f"[install] {pin.key}/{name}: FAILED - {exc}")
    else:
        report(f"[install] {pin.key}: using {jobs} independent file workers")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=jobs,
            thread_name_prefix=f"{pin.key}-fetch",
        ) as executor:
            future_names = {
                executor.submit(install_declared_file, name, spec): name
                for name, spec in files.items()
            }
            for future in concurrent.futures.as_completed(future_names):
                name = future_names[future]
                try:
                    future.result()
                except InstallerError as exc:
                    failures.append(name)
                    report(f"[install] {pin.key}/{name}: FAILED - {exc}")
                except Exception as exc:  # defensive: retain other verified progress
                    failures.append(name)
                    report(
                        f"[install] {pin.key}/{name}: FAILED - "
                        f"unexpected {type(exc).__name__}: {exc}"
                    )
    try:
        install_file(
            pin,
            CURRENT_LICENSE_PIN,
            LICENSE_NAME,
            CURRENT_LICENSE_SPEC,
            models_root=models_root,
            opener=opener,
            attempts=attempts,
            timeout=timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=range_workers,
            report=report,
        )
    except InstallerError as exc:
        failures.append(LICENSE_NAME)
        report(f"[install] {pin.key}/{LICENSE_NAME}: FAILED - {exc}")
    if failures:
        raise InstallerError(
            f"{pin.key}: {len(failures)} file(s) failed ({', '.join(failures)}); rerun to resume"
        )
    destination = models_root / pin.directory
    if metadata_only:
        report(
            f"[install] {pin.key}: metadata/license subset complete; no full-pack receipt was written"
        )
        return
    _atomic_json(destination / ".phosphene-install-receipt.json", _receipt(pin, manifest_digest, manifest))
    report(
        f"[install] {pin.key}: complete; every weight file and the separately pinned current license verified"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the pinned Phosphene LTX-2.5 Q4 and Gemma 4 packs from GitHub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="verify pinned manifests and tiny CDN ranges only")
    mode.add_argument("--plan", action="store_true", help="inspect local files and estimate required disk; download no weights")
    mode.add_argument("--verify", action="store_true", help="hash-check installed packs; download no weights")
    mode.add_argument("--install", action="store_true", help="download, verify, and install the selected packs")
    parser.add_argument(
        "--pack",
        action="append",
        choices=PACK_ORDER,
        help="pack to operate on; repeat as needed (default: both)",
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--timeout", type=int, default=60, help="per-request network timeout in seconds")
    parser.add_argument("--attempts", type=int, default=4, help="attempts per shard")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="independent file downloads to run in parallel (1-4)",
    )
    parser.add_argument(
        "--range-chunk-mib",
        type=int,
        default=0,
        help=(
            "download each shard as exact resumable HTTP ranges of this size; "
            "0 keeps the normal streaming downloader (enabled range: 1-1024 MiB)"
        ),
    )
    parser.add_argument(
        "--range-workers",
        type=int,
        default=1,
        help=(
            "exact-range requests per shard (1-16); values above 1 require "
            "--range-chunk-mib and multiply with --jobs"
        ),
    )
    parser.add_argument(
        "--ipv6",
        action="store_true",
        help="force curl's IPv6 route (use only after a successful IPv6 range probe)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "with --install, fetch only non-safetensors metadata/tokenizer files plus the license; "
            "never writes a full-pack receipt"
        ),
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=DEFAULT_RESERVE_BYTES / 2**30,
        help="free space to leave untouched during installation",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm acceptance of the LTX-2.x Community License (required with --install)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts <= 0 or args.reserve_gib < 0:
        parser.error("--timeout and --attempts must be positive; --reserve-gib cannot be negative")
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    if not 0 <= args.range_chunk_mib <= 1024:
        parser.error("--range-chunk-mib must be 0 or between 1 and 1024")
    if not 1 <= args.range_workers <= 16:
        parser.error("--range-workers must be between 1 and 16")
    if args.range_workers > 1 and args.range_chunk_mib == 0:
        parser.error("--range-workers above 1 requires --range-chunk-mib")
    if args.jobs * args.range_workers > 16:
        parser.error("--jobs multiplied by --range-workers cannot exceed 16")
    if args.install and not args.accept_license:
        parser.error("--install requires --accept-license after reviewing the LTX-2.x Community License")
    if args.metadata_only and not args.install:
        parser.error("--metadata-only requires --install")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    range_chunk_bytes = args.range_chunk_mib * 2**20
    selected = [PACKS[key] for key in (args.pack or PACK_ORDER)]
    # Stable de-duplication if a user repeats --pack.
    selected = list(dict.fromkeys(selected))
    models_root = args.models_root.expanduser().resolve()
    opener = build_opener(force_ipv6=args.ipv6)
    manifests: dict[str, dict] = {}
    digests: dict[str, str] = {}

    try:
        for pin in selected:
            manifest, raw = fetch_pinned_manifest(pin, opener=opener, timeout=args.timeout)
            manifests[pin.key] = manifest
            digests[pin.key] = sha256_bytes(raw)
            log(
                f"[manifest] {pin.key}: {len(raw)} bytes, SHA-256 {digests[pin.key]} verified"
            )

        if args.preflight:
            for pin in selected:
                result = probe_small_asset(
                    pin, manifests[pin.key], opener=opener, timeout=args.timeout
                )
                log(f"[preflight] {pin.key}: {result}")
            license_bytes = fetch_current_license(opener=opener, timeout=args.timeout)
            log(
                f"[preflight] current license: {len(license_bytes)} bytes, "
                f"SHA-256 {CURRENT_LICENSE_PIN} verified"
            )
            log("[preflight] PASS - GitHub release path works; no weight shard was downloaded")
            return 0

        if args.plan or args.verify:
            failures = 0
            total_missing = 0
            for pin in selected:
                missing_bytes, results = pack_status(
                    pin, manifests[pin.key], models_root=models_root
                )
                total_missing += missing_bytes
                for name, ready, detail in results:
                    marker = "OK" if ready else "NEEDED"
                    log(f"[{marker}] {pin.key}/{name}: {detail}")
                    failures += int(args.verify and not ready)
                log(
                    f"[status] {pin.key}: {(pack_size(manifests[pin.key]) - missing_bytes) / 1e9:.2f}/"
                    f"{pack_size(manifests[pin.key]) / 1e9:.2f} GB verified"
                )
            if args.plan:
                required = estimate_required_space(
                    selected,
                    manifests,
                    models_root=models_root,
                    reserve_bytes=int(args.reserve_gib * 2**30),
                    jobs=args.jobs,
                    range_chunk_bytes=range_chunk_bytes,
                    range_workers=args.range_workers,
                )
                free = shutil.disk_usage(models_root.parent if not models_root.exists() else models_root).free
                log(f"[plan] missing final bytes: {total_missing / 1e9:.2f} GB")
                log(f"[plan] conservative free-space requirement: {required / 1e9:.2f} GB")
                log(f"[plan] currently free: {free / 1e9:.2f} GB")
                return 0
            if failures:
                log(f"[verify] FAIL - {failures} file(s) missing or invalid")
                return 1
            log("[verify] PASS - all selected files match the pinned release manifests")
            return 0

        models_root.mkdir(parents=True, exist_ok=True)
        if models_root.is_symlink() or not models_root.is_dir():
            fail(f"unsafe models root: {models_root}")
        required = estimate_required_space(
            selected,
            manifests,
            models_root=models_root,
            reserve_bytes=int(args.reserve_gib * 2**30),
            jobs=args.jobs,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=args.range_workers,
            metadata_only=args.metadata_only,
        )
        free = shutil.disk_usage(models_root).free
        if free < required:
            fail(
                f"insufficient free space: need a conservative {required / 1e9:.2f} GB, "
                f"have {free / 1e9:.2f} GB"
            )
        log(
            f"[install] disk preflight passed: {free / 1e9:.2f} GB free, "
            f"{required / 1e9:.2f} GB conservative requirement"
        )
        for pin in selected:
            install_pack(
                pin,
                manifests[pin.key],
                digests[pin.key],
                models_root=models_root,
                opener=opener,
                attempts=args.attempts,
                timeout=args.timeout,
                jobs=args.jobs,
                range_chunk_bytes=range_chunk_bytes,
                range_workers=args.range_workers,
                metadata_only=args.metadata_only,
            )
        if args.metadata_only:
            log(
                "[install] PASS - selected metadata/license subsets are complete and "
                "cryptographically verified; weight files were intentionally skipped"
            )
        else:
            log("[install] PASS - selected packs are complete and cryptographically verified")
        return 0
    except KeyboardInterrupt:
        log("[install] interrupted; verified progress is staged and the next run will resume")
        return 130
    except InstallerError as exc:
        log(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
