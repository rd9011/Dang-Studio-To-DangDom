#!/usr/bin/env python3
"""Install pinned optional LTX-2.5 IC-LoRA adapters safely and sequentially.

Downloads go directly to hidden partial files beside their final destinations,
are verified before an atomic rename, and never touch the user's Downloads
folder. Authentication is read from the normal Hugging Face environment/token
files and is never accepted on the command line or printed. Motion and
Ingredients also have a separately staged, hash-verified ModelScope fallback;
Hugging Face credentials are never sent to that mirror. The verified Phosphene
base/Gemma packs are deliberately outside this installer's scope.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

from studio_paths import PATHS


CONTROL_ROOT = PATHS.models_root / "LTX-2.5-control-adapters"
RECEIPT_PATH = PATHS.models_root / ".ltx25-optional-assets.json"
RESERVE_BYTES = 6 * 2**30
CHUNK_BYTES = 8 << 20
PROGRESS_BYTES = 256 << 20
HUGGINGFACE = "huggingface"
MODELSCOPE = "modelscope"
AUTO = "auto"
SOURCE_CHOICES = (AUTO, HUGGINGFACE, MODELSCOPE)


class InstallerError(RuntimeError):
    """An optional asset could not be safely planned, verified, or installed."""


class VerificationError(InstallerError):
    """A local file did not match its immutable manifest entry."""


class DownloadError(InstallerError):
    """A network download failed without exposing credentials."""


@dataclass(frozen=True)
class Asset:
    key: str
    label: str
    repo: str
    revision: str
    filename: str
    destination: Path
    size: int
    sha256: str | None
    kind: str = "safetensors"
    gated: bool = False
    modelscope_repo: str | None = None

    @property
    def terms_url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


ASSETS: dict[str, Asset] = {
    "union": Asset(
        key="union",
        label="Union Control IC-LoRA",
        repo="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        revision="b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff",
        filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        destination=CONTROL_ROOT / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        size=654_465_352,
        sha256="a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18",
    ),
    "motion": Asset(
        key="motion",
        label="Motion Track IC-LoRA",
        repo="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        revision="572bb9c9a1ba3d8e8724cce69783ffc2422386db",
        filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        destination=CONTROL_ROOT / "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        size=327_309_314,
        sha256="e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca",
        modelscope_repo="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
    ),
    "ingredients": Asset(
        key="ingredients",
        label="Ingredients IC-LoRA",
        repo="Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        revision="12040e4091ac2008d3906a594e31a7fb1ab9d546",
        filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        destination=CONTROL_ROOT / "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        size=1_308_787_472,
        sha256="ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95",
        gated=True,
        modelscope_repo="Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
    ),
}

ASSET_ORDER = tuple(ASSETS)
BUNDLES: dict[str, tuple[str, ...]] = {
    "union": ("union",),
    "motion": ("motion",),
    "ingredients": ("ingredients",),
    "controls": ("union", "motion", "ingredients"),
    "all": ASSET_ORDER,
}


@dataclass(frozen=True)
class AssetState:
    asset: Asset
    ready: bool
    detail: str
    partial_bytes: int
    transfer_bytes: int
    disk_bytes: int


def _fail(message: str) -> NoReturn:
    raise InstallerError(message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and atomically install pinned optional LTX-2.5 IC-LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=tuple(BUNDLES),
        required=True,
        help="asset bundle to install; repeat to combine bundles",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=AUTO,
        help=(
            "download source policy; auto tries the pinned Hugging Face revision first, then an independently "
            "staged ModelScope mirror when one is declared"
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="show files, status, and disk requirement; download nothing",
    )
    parser.add_argument(
        "--replace-invalid",
        action="store_true",
        help="atomically replace an existing destination only after the new file verifies",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help=(
            "disable HTTPS certificate verification for a TLS-intercepting proxy; dangerous and never implicit. "
            "Pinned size and SHA-256 checks still run."
        ),
    )
    return parser.parse_args(argv)


def selected_assets(bundles: Sequence[str]) -> tuple[Asset, ...]:
    """Expand bundles into a stable, de-duplicated sequential install order."""
    selected: set[str] = set()
    for bundle in bundles:
        try:
            selected.update(BUNDLES[bundle])
        except KeyError:
            _fail(f"unknown bundle: {bundle}")
    return tuple(ASSETS[key] for key in ASSET_ORDER if key in selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_safetensors(path: Path, *, expected_size: int) -> None:
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError("missing 8-byte header length")
            header_size = int.from_bytes(raw_size, "little")
            if not 2 <= header_size <= 64 << 20:
                raise ValueError(f"unsafe header length {header_size}")
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError("truncated JSON header")
            header = json.loads(raw_header)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VerificationError(f"invalid safetensors structure: {path} ({exc})") from exc
    if not isinstance(header, dict):
        raise VerificationError(f"safetensors header is not an object: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise VerificationError(f"safetensors file contains no tensors: {path}")
    data_size = expected_size - 8 - header_size
    spans: list[tuple[int, int, str]] = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise VerificationError(f"invalid tensor entry {name!r}: {path}")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise VerificationError(f"invalid safetensors entry {name!r}: {path}")
        spans.append((offsets[0], offsets[1], name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise VerificationError(f"non-contiguous data before tensor {name!r}: {path}")
        cursor = end
    if cursor != data_size:
        raise VerificationError(f"safetensors data region is not fully indexed: {path}")


def _validate_upscaler_config(path: Path) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid upscaler config JSON: {path} ({exc})") from exc
    config = raw.get("config") if isinstance(raw, dict) else None
    expected = {
        "_class_name": "LatentUpsampler",
        "in_channels": 128,
        "mid_channels": 1024,
        "num_blocks_per_stage": 4,
        "dims": 3,
        "spatial_upsample": True,
        "temporal_upsample": False,
        "spatial_scale": 2.0,
        "rational_resampler": False,
    }
    if config != expected:
        raise VerificationError(f"unexpected LTX-2.5 spatial upscaler config schema: {path}")


def verify_asset(asset: Asset, path: Path, *, receipt_sha256: str | None = None) -> str:
    """Fully verify a file and return its computed SHA-256."""
    if not path.is_file():
        raise VerificationError(f"missing {asset.label}: {path}")
    actual_size = path.stat().st_size
    if actual_size != asset.size:
        raise VerificationError(
            f"wrong size for {asset.label}: {actual_size:,} != {asset.size:,} bytes ({path})"
        )
    if asset.kind == "safetensors":
        _validate_safetensors(path, expected_size=actual_size)
    elif asset.kind == "upscaler-config":
        _validate_upscaler_config(path)
    else:
        raise VerificationError(f"unknown validation kind {asset.kind!r} for {asset.key}")
    digest = _sha256(path)
    expected_digest = asset.sha256 or receipt_sha256
    if expected_digest and digest != expected_digest:
        raise VerificationError(
            f"SHA-256 mismatch for {asset.label}: {digest} != {expected_digest} ({path})"
        )
    return digest


def _load_receipts(path: Path = RECEIPT_PATH) -> dict[str, object]:
    if not path.is_file():
        return {"schema": 1, "assets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"schema": 1, "assets": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), dict):
        return {"schema": 1, "assets": {}}
    return payload


def _receipt_digest(asset: Asset, receipts: Mapping[str, object]) -> str | None:
    entries = receipts.get("assets")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(asset.key)
    if not isinstance(entry, dict):
        return None
    if (
        entry.get("repo") != asset.repo
        or entry.get("revision") != asset.revision
        or entry.get("filename") != asset.filename
        or entry.get("size") != asset.size
    ):
        return None
    digest = entry.get("sha256")
    return digest if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) else None


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def _record_receipt(asset: Asset, digest: str, *, path: Path = RECEIPT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    receipts = _load_receipts(path)
    entries = dict(receipts.get("assets", {}))
    entries[asset.key] = {
        "repo": asset.repo,
        "revision": asset.revision,
        "filename": asset.filename,
        "destination": str(asset.destination),
        "size": asset.size,
        "sha256": digest,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(path, {"schema": 1, "assets": entries})


def _source_candidates(asset: Asset, source: str) -> tuple[str, ...]:
    """Resolve a source policy without permitting an unpinned mirror download."""
    if source not in SOURCE_CHOICES:
        _fail(f"unknown download source: {source}")
    has_modelscope = (
        isinstance(asset.modelscope_repo, str)
        and re.fullmatch(r"Lightricks/[A-Za-z0-9][A-Za-z0-9._-]*", asset.modelscope_repo) is not None
        and asset.size > 0
        and isinstance(asset.sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", asset.sha256) is not None
    )
    if source == MODELSCOPE:
        if not has_modelscope:
            _fail(
                f"{asset.label} has no ModelScope fallback with a pinned size and SHA-256; "
                "use --source huggingface"
            )
        return (MODELSCOPE,)
    if source == HUGGINGFACE:
        return (HUGGINGFACE,)
    return (HUGGINGFACE, MODELSCOPE) if has_modelscope else (HUGGINGFACE,)


def _partial_paths(asset: Asset, source: str = HUGGINGFACE) -> tuple[Path, Path]:
    """Return source-isolated staging paths so fallback can never mix byte ranges."""
    if source not in (HUGGINGFACE, MODELSCOPE):
        _fail(f"partial paths require a concrete source, not {source!r}")
    source_suffix = "" if source == HUGGINGFACE else f".{source}"
    partial = asset.destination.with_name(
        f".{asset.filename}.optional-download{source_suffix}.part"
    )
    metadata = partial.with_name(partial.name + ".json")
    return partial, metadata


def _read_resume_metadata(
    path: Path,
    asset: Asset,
    source: str = HUGGINGFACE,
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = {
        "schema": 1,
        "repo": asset.repo,
        "revision": asset.revision,
        "filename": asset.filename,
        "size": asset.size,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    # Metadata written before source fallback existed is Hugging Face metadata.
    if payload.get("source", HUGGINGFACE) != source:
        return None
    validator = payload.get("if_range")
    if (
        not isinstance(validator, str)
        or not validator.strip()
        or validator.lstrip().lower().startswith("w/")
    ):
        return None
    return payload


def _write_resume_metadata(
    path: Path,
    asset: Asset,
    validator: str,
    source: str = HUGGINGFACE,
) -> None:
    _atomic_json(
        path,
        {
            "schema": 1,
            "repo": asset.repo,
            "revision": asset.revision,
            "filename": asset.filename,
            "size": asset.size,
            "source": source,
            "if_range": validator,
        },
    )


def discover_hf_token(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> str | None:
    """Read the standard Hugging Face token locations without ever logging it."""
    env = os.environ if environ is None else environ
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = env.get(name, "").strip()
        if value:
            return value
    base_home = Path.home() if home is None else home
    configured = env.get("HF_TOKEN_PATH")
    hf_home = Path(env.get("HF_HOME", base_home / ".cache" / "huggingface")).expanduser()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((hf_home / "token", base_home / ".huggingface" / "token"))
    for path in candidates:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward credentials or cookies to a different redirect origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        if urllib.parse.urlsplit(resolved_url).scheme.lower() != "https":
            raise urllib.error.HTTPError(resolved_url, code, "refusing non-HTTPS redirect", headers, fp)
        redirected = super().redirect_request(req, fp, code, msg, headers, resolved_url)
        if redirected is not None and _origin(req.full_url) != _origin(resolved_url):
            redirected.remove_header("Authorization")
            redirected.remove_header("Cookie")
        return redirected


def build_opener(*, insecure_tls: bool) -> urllib.request.OpenerDirector:
    context = ssl._create_unverified_context() if insecure_tls else ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        SafeRedirectHandler(),
        urllib.request.HTTPSHandler(context=context),
    )


def _download_url(asset: Asset, source: str = HUGGINGFACE) -> str:
    filename = urllib.parse.quote(asset.filename, safe="/")
    if source == HUGGINGFACE:
        repo = urllib.parse.quote(asset.repo, safe="/")
        revision = urllib.parse.quote(asset.revision, safe="")
        return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    if source == MODELSCOPE:
        _source_candidates(asset, MODELSCOPE)
        repo = urllib.parse.quote(str(asset.modelscope_repo), safe="/")
        return f"https://www.modelscope.cn/models/{repo}/resolve/master/{filename}"
    _fail(f"download URL requires a concrete source, not {source!r}")


def _content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    return tuple(int(group) for group in match.groups())  # type: ignore[return-value]


def _resume_validator(headers: Mapping[str, str]) -> str | None:
    """Return an If-Range validator, rejecting weak ETags."""
    etag = headers.get("ETag")
    if etag:
        etag = etag.strip()
        if etag and not etag.lower().startswith("w/"):
            return etag
    modified = headers.get("Last-Modified")
    if modified and modified.strip():
        return modified.strip()
    return None


def _http_failure(asset: Asset, status: int, source: str) -> DownloadError:
    if source == HUGGINGFACE and status in {401, 403}:
        return DownloadError(
            f"Hugging Face returned HTTP {status} for {asset.label}. Accept the repository terms at "
            f"{asset.terms_url}, then save an authorized token with `hf auth login` (or HF_TOKEN) and retry. "
            "The installer never accepts or prints a token argument."
        )
    source_label = "Hugging Face" if source == HUGGINGFACE else "ModelScope"
    return DownloadError(f"{source_label} returned HTTP {status} while downloading {asset.label}")


def download_asset(
    asset: Asset,
    partial: Path,
    metadata_path: Path,
    *,
    token: str | None,
    insecure_tls: bool,
    source: str = HUGGINGFACE,
    opener: urllib.request.OpenerDirector | None = None,
    report: Callable[[str], None] = print,
) -> None:
    """Download one asset, safely resuming only a validator-backed HTTP 206."""
    _source_candidates(asset, source)
    partial.parent.mkdir(parents=True, exist_ok=True)
    client = opener or build_opener(insecure_tls=insecure_tls)
    resume_metadata = (
        _read_resume_metadata(metadata_path, asset, source) if partial.is_file() else None
    )
    partial_size = partial.stat().st_size if partial.is_file() else 0

    if partial_size == asset.size:
        try:
            verify_asset(asset, partial)
            report(f"Using complete verified partial for {asset.label}")
            return
        except VerificationError:
            resume_metadata = None
    if partial_size > asset.size:
        resume_metadata = None

    resume_at = partial_size if resume_metadata is not None and 0 < partial_size < asset.size else 0
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "local-ltx25-optional-installer/1",
    }
    # A Hugging Face token must never leave the Hugging Face request path.
    if token and source == HUGGINGFACE:
        headers["Authorization"] = f"Bearer {token}"
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
        headers["If-Range"] = str(resume_metadata["if_range"])

    url = _download_url(asset, source)
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        response = client.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and resume_at:
            # The server rejected the range. Retry from zero; never append.
            resume_at = 0
            headers.pop("Range", None)
            headers.pop("If-Range", None)
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                response = client.open(request, timeout=60)
            except urllib.error.HTTPError as retry_exc:
                raise _http_failure(asset, retry_exc.code, source) from None
        else:
            raise _http_failure(asset, exc.code, source) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise DownloadError(
                f"TLS certificate verification failed for {asset.label}. Fix the corporate CA trust chain, "
                "or explicitly retry with --insecure-tls (credentials can then be intercepted)."
            ) from None
        raise DownloadError(f"network error while downloading {asset.label}: {reason}") from None

    with response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        response_headers = response.headers
        validator = _resume_validator(response_headers)

        mode = "wb"
        written = 0
        if resume_at:
            parsed_range = _content_range(response_headers.get("Content-Range"))
            response_validator = _resume_validator(response_headers)
            if (
                status == 206
                and parsed_range is not None
                and parsed_range[0] == resume_at
                and parsed_range[1] == asset.size - 1
                and parsed_range[2] == asset.size
                and response_validator == resume_metadata["if_range"]
            ):
                mode = "ab"
                written = resume_at
                report(f"Safely resuming {asset.label} at {resume_at / 2**30:.2f} GiB")
            elif status == 200:
                # If-Range deliberately falls back to a full 200 when the
                # validator changed or ranges are unsupported. Truncate only
                # our hidden partial; never append a full response.
                resume_at = 0
            else:
                raise DownloadError(
                    f"unsafe resume response for {asset.label}: HTTP {status}, "
                    f"Content-Range={response_headers.get('Content-Range')!r}; partial left untouched"
                )
        elif status != 200:
            raise DownloadError(f"unexpected HTTP {status} for a fresh {asset.label} download")

        content_length = response_headers.get("Content-Length")
        if content_length:
            try:
                advertised = int(content_length)
            except ValueError:
                raise DownloadError(f"invalid Content-Length for {asset.label}") from None
            expected_response = asset.size - written
            if advertised != expected_response:
                raise DownloadError(
                    f"server advertised {advertised:,} bytes for {asset.label}; expected {expected_response:,}"
                )

        if mode == "ab" and not validator:
            raise DownloadError(f"server omitted the resume validator for {asset.label}")

        last_report = written
        started = time.monotonic()
        try:
            with partial.open(mode) as handle:
                if mode == "wb":
                    # Truncate the old partial before publishing a validator
                    # for the new response. A crash can therefore never pair
                    # old bytes with new resume metadata.
                    try:
                        metadata_path.unlink(missing_ok=True)
                    except TypeError:  # pragma: no cover - Python <3.8 fallback
                        if metadata_path.exists():
                            metadata_path.unlink()
                    if validator:
                        _write_resume_metadata(metadata_path, asset, validator, source)
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if written - last_report >= PROGRESS_BYTES:
                        percent = 100.0 * written / asset.size
                        report(
                            f"  {asset.label}: {written / 2**30:.2f}/{asset.size / 2**30:.2f} GiB "
                            f"({percent:.1f}%)"
                        )
                        last_report = written
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TimeoutError, urllib.error.URLError, http.client.HTTPException):
            raise DownloadError(
                f"download interrupted for {asset.label} after {written:,}/{asset.size:,} bytes; "
                "the partial was kept and will resume only if its validator metadata is safe"
            ) from None
        elapsed = time.monotonic() - started
        report(f"Downloaded {asset.label}: {written:,} bytes total in {elapsed:.1f}s")


def inspect_assets(
    assets: Sequence[Asset],
    *,
    receipts_path: Path = RECEIPT_PATH,
    source: str = HUGGINGFACE,
) -> tuple[AssetState, ...]:
    receipts = _load_receipts(receipts_path)
    states: list[AssetState] = []
    for asset in assets:
        ready = False
        detail = "missing"
        receipt_digest = _receipt_digest(asset, receipts)
        if asset.destination.exists():
            try:
                verify_asset(asset, asset.destination, receipt_sha256=receipt_digest)
            except (OSError, VerificationError) as exc:
                detail = f"invalid existing file: {exc}"
            else:
                ready = True
                detail = "installed and verified"
                if asset.sha256 is None and receipt_digest is None:
                    detail += "; local SHA-256 receipt will be created during install"
        candidates = _source_candidates(asset, source)
        partial_bytes = 0
        resumable_bytes = 0
        complete_partial = False
        for candidate in candidates:
            partial, metadata = _partial_paths(asset, candidate)
            candidate_bytes = 0
            if not ready and partial.is_file():
                try:
                    candidate_bytes = min(partial.stat().st_size, asset.size)
                except OSError:
                    candidate_bytes = 0
                partial_bytes = max(partial_bytes, candidate_bytes)
                source_detail = f"{candidate} " if len(candidates) > 1 else ""
                if candidate_bytes == asset.size:
                    try:
                        verify_asset(asset, partial)
                    except (OSError, VerificationError):
                        detail += f"; {source_detail}complete partial is invalid and will restart"
                    else:
                        complete_partial = True
                        detail += f"; {source_detail}complete partial is verified and ready to promote"
                elif candidate_bytes and _read_resume_metadata(metadata, asset, candidate) is not None:
                    resumable_bytes = max(resumable_bytes, candidate_bytes)
                    detail += (
                        f"; {source_detail}resumable partial "
                        f"{candidate_bytes:,}/{asset.size:,} bytes"
                    )
                elif candidate_bytes:
                    detail += f"; {source_detail}partial has no safe resume validator and will restart"
        transfer = 0 if ready or complete_partial else asset.size - resumable_bytes
        disk_bytes = 0 if ready else max(0, asset.size - partial_bytes)
        states.append(AssetState(asset, ready, detail, partial_bytes, transfer, disk_bytes))
    return tuple(states)


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def print_plan(
    states: Sequence[AssetState],
    *,
    bundles: Sequence[str],
    free_bytes: int,
    source: str = HUGGINGFACE,
) -> int:
    print("Optional LTX-2.5 asset plan")
    print("Bundles: " + ", ".join(bundles))
    print(f"Source policy: {source}")
    total_transfer = 0
    additional_disk = 0
    for state in states:
        mark = "READY" if state.ready else ("PROMOTE" if state.transfer_bytes == 0 else "FETCH")
        print(f"[{mark}] {state.asset.label}")
        print(f"  source: {state.asset.repo}@{state.asset.revision}")
        print(f"  file:   {state.asset.filename} ({state.asset.size:,} bytes)")
        print(f"  target: {state.asset.destination}")
        print(f"  status: {state.detail}")
        total_transfer += state.transfer_bytes
        additional_disk += state.disk_bytes
    required = additional_disk + RESERVE_BYTES
    print(f"Transfer remaining: {_gib(total_transfer)}")
    print(f"Additional destination disk needed: {_gib(additional_disk)}")
    print(f"Required free including 6 GiB safety reserve: {_gib(required)}")
    print(f"Current free: {_gib(free_bytes)}")
    print("Disk preflight: " + ("READY" if free_bytes >= required else "INSUFFICIENT SPACE"))
    return total_transfer


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _discard_metadata(path: Path) -> None:
    """Best-effort cleanup for installer-owned resume metadata."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _source_label(source: str) -> str:
    return "Hugging Face" if source == HUGGINGFACE else "ModelScope"


def _cleanup_download_artifacts(asset: Asset, sources: Sequence[str]) -> None:
    """Remove only installer-owned staging files after a verified publish."""
    for source in sources:
        partial, metadata = _partial_paths(asset, source)
        for path in (partial, metadata):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def install_asset(
    asset: Asset,
    *,
    replace_invalid: bool,
    insecure_tls: bool,
    token: str | None,
    source: str = HUGGINGFACE,
    receipts_path: Path = RECEIPT_PATH,
    downloader: Callable[..., None] = download_asset,
    report: Callable[[str], None] = print,
) -> str:
    """Install one asset, preserving any existing destination until verification."""
    candidates = _source_candidates(asset, source)
    receipts = _load_receipts(receipts_path)
    if asset.destination.exists():
        try:
            digest = verify_asset(asset, asset.destination, receipt_sha256=_receipt_digest(asset, receipts))
        except VerificationError as exc:
            if not replace_invalid:
                raise InstallerError(
                    f"existing destination is invalid and was preserved: {exc}\n"
                    "Pass --replace-invalid to download and atomically replace it after verification."
                ) from None
        else:
            if asset.sha256 is None and _receipt_digest(asset, receipts) is None:
                _record_receipt(asset, digest, path=receipts_path)
                report(f"RECORDED {asset.label}: local SHA-256 receipt created")
            _cleanup_download_artifacts(asset, candidates)
            report(f"READY {asset.label}: already installed and verified")
            return digest

    asset.destination.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        partial, _metadata = _partial_paths(asset, candidate)
        if partial.is_file() and partial.stat().st_size == asset.size:
            try:
                digest = verify_asset(asset, partial)
            except VerificationError:
                pass
            else:
                report(
                    f"VERIFY {asset.label}: complete {_source_label(candidate)} partial is valid; "
                    "no network needed"
                )
                os.replace(partial, asset.destination)
                _sync_directory(asset.destination.parent)
                _record_receipt(asset, digest, path=receipts_path)
                _cleanup_download_artifacts(asset, candidates)
                return digest

    failures: list[tuple[str, str]] = []
    for index, candidate in enumerate(candidates):
        partial, metadata = _partial_paths(asset, candidate)
        report(
            f"FETCH {asset.label} ({_gib(asset.size)}) from {_source_label(candidate)}"
        )
        try:
            downloader(
                asset,
                partial,
                metadata,
                token=token if candidate == HUGGINGFACE else None,
                insecure_tls=insecure_tls,
                source=candidate,
                report=report,
            )
            digest = verify_asset(asset, partial)
        except (DownloadError, VerificationError) as exc:
            failures.append((candidate, str(exc)))
            if index + 1 < len(candidates):
                next_candidate = candidates[index + 1]
                report(
                    f"FALLBACK {asset.label}: {_source_label(candidate)} failed verification or transfer; "
                    f"its isolated partial was preserved. Trying {_source_label(next_candidate)}."
                )
                continue
            if len(failures) == 1:
                raise
            summary = "; ".join(
                f"{_source_label(failed_source)}: {message}"
                for failed_source, message in failures
            )
            raise DownloadError(f"all verified sources failed for {asset.label}: {summary}") from None

        report(
            f"VERIFIED {asset.label} from {_source_label(candidate)}: "
            "size, structure, and SHA-256 checks passed"
        )
        os.replace(partial, asset.destination)
        _sync_directory(asset.destination.parent)
        _record_receipt(asset, digest, path=receipts_path)
        _cleanup_download_artifacts(asset, candidates)
        report(f"INSTALLED {asset.label}: {asset.destination}")
        return digest

    raise DownloadError(f"no usable source was attempted for {asset.label}")  # pragma: no cover


def _warn_insecure(assets: Sequence[Asset]) -> None:
    print(
        "WARNING: --insecure-tls disables HTTPS certificate verification. The proxy can intercept both the "
        "saved Hugging Face token and downloaded bytes. Known assets will still be rejected unless their pinned "
        "size and SHA-256 match.",
        file=sys.stderr,
    )
    if any(asset.sha256 is None for asset in assets):
        print(
            "WARNING: At least one selected asset has no upstream-published SHA-256. Under --insecure-tls its "
            "first download is not cryptographically authenticated: the request names the pinned revision, but "
            "only exact size and safetensors structure can be enforced. A local SHA-256 receipt detects later "
            "changes, not interception of this first download.",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    assets = selected_assets(args.bundle)
    if args.insecure_tls:
        _warn_insecure(assets)

    try:
        states = inspect_assets(assets, source=args.source)
        free_bytes = shutil.disk_usage(PATHS.data_root).free
        print_plan(states, bundles=args.bundle, free_bytes=free_bytes, source=args.source)
        if args.plan:
            print("PLAN ONLY; no directories or downloads were created.")
            return 0

        invalid = [state for state in states if state.asset.destination.exists() and not state.ready]
        if invalid and not args.replace_invalid:
            names = ", ".join(state.asset.filename for state in invalid)
            raise InstallerError(
                f"invalid destination file(s) were preserved: {names}. Pass --replace-invalid to repair them."
            )
        required = sum(state.disk_bytes for state in states) + RESERVE_BYTES
        if free_bytes < required:
            raise InstallerError(
                f"disk preflight failed: {_gib(free_bytes)} free, but {_gib(required)} is required "
                "for additional destination data plus the 6 GiB safety reserve"
            )

        uses_huggingface = any(
            HUGGINGFACE in _source_candidates(asset, args.source) for asset in assets
        )
        token = discover_hf_token() if uses_huggingface else None
        if token is None and uses_huggingface and any(asset.gated for asset in assets):
            print(
                "WARNING: no saved Hugging Face token was found. Gated files will return 401/403 until their "
                "terms are accepted and `hf auth login` (or HF_TOKEN) is configured. Auto mode can still use "
                "a declared hash-verified ModelScope fallback.",
                file=sys.stderr,
            )
        for asset in assets:
            install_asset(
                asset,
                replace_invalid=args.replace_invalid,
                insecure_tls=args.insecure_tls,
                token=token,
                source=args.source,
            )
        print(f"OPTIONAL ASSETS READY: {len(assets)} pinned file(s)")
        return 0
    except (InstallerError, OSError) as exc:
        print(f"INSTALL FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
