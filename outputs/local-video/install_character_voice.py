#!/usr/bin/env python3
"""Install the isolated local Chatterbox Multilingual V3 voice runtime.

The installer deliberately keeps TTS separate from the LTX environment. Source
comes from an immutable hash-pinned HTTPS archive, Python dependencies come
from a hash-locked ``uv.lock``, and the five public model files are downloaded
from one immutable Hugging Face revision and verified before atomic
publication. A checkout at the same commit is the network fallback.

This file never accepts or forwards an access token.  The selected model is
public and inference is performed later with the Hub in strict offline mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, NoReturn, Sequence

from pinned_source import (
    PROVENANCE_NAME,
    SourceInstallError,
    SourcePin,
    stage_pinned_source,
    verify_source_tree,
)
from studio_paths import PATHS


HERE = PATHS.app_root
WORKSPACE = PATHS.workspace
LOCAL_VIDEO_ROOT = PATHS.data_root
TEMPLATE_ROOT = HERE / "character_voice_runtime"
RUNTIME_ROOT = PATHS.character_voice_runtime
RUNTIME_STAGING = PATHS.runtimes_root / ".character-voice-runtime-staging"
SOURCE_DIR = RUNTIME_ROOT / "source"
VENV_PYTHON = RUNTIME_ROOT / ".venv" / "bin" / "python"
MODEL_DIR = PATHS.models_root / "chatterbox-multilingual-v3"
MODEL_STAGING = PATHS.models_root / ".chatterbox-multilingual-v3-staging"
UV_CACHE_DIR = PATHS.cache_root / "uv-character-voice"
RECEIPT_PATH = RUNTIME_ROOT / "install-receipt.json"

SOURCE_REPO = "https://github.com/resemble-ai/chatterbox.git"
SOURCE_COMMIT = "65b18437192794391a0308a8f705b1e33e633948"
SOURCE_VERSION = "0.1.7"
SOURCE_PIN = SourcePin(
    name="chatterbox",
    repository=SOURCE_REPO,
    revision=SOURCE_COMMIT,
    archive_url=(
        "https://codeload.github.com/resemble-ai/chatterbox/tar.gz/"
        "65b18437192794391a0308a8f705b1e33e633948"
    ),
    archive_sha256="c7c0db5207ef7ba8f660b870b9c84e4796fb16de9f39867d12b27e1eef3144ac",
    archive_bytes=1_431_580,
    archive_root="chatterbox-65b18437192794391a0308a8f705b1e33e633948",
    tree_sha256="55d1d2ec850ec23b1a0e97804118c9df8991c67ff9e2dfeea7ccdba43c684143",
    includes=("pyproject.toml", "README.md", "LICENSE", "src/"),
)
MODEL_REPO = "ResembleAI/chatterbox"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
RESERVE_BYTES = 5 * 2**30


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str


# Only the files read by ChatterboxMultilingualTTS.from_local(..., t3_model="v3").
# The built-in voice is intentionally omitted: stable application identities
# must resolve to an explicit, consented canonical reference clip.
MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(
        "t3_mtl23ls_v3.safetensors",
        2_143_989_928,
        "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953",
    ),
    ModelFile(
        "s3gen.pt",
        1_057_165_844,
        "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3",
    ),
    ModelFile(
        "ve.pt",
        5_698_626,
        "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1",
    ),
    ModelFile(
        "grapheme_mtl_merged_expanded_v1.json",
        69_989,
        "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf",
    ),
    ModelFile(
        "Cangjie5_TC.json",
        1_920_163,
        "7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c",
    ),
)
MODEL_TOTAL_BYTES = sum(item.size for item in MODEL_FILES)

EXPECTED_PACKAGES = {
    "chatterbox-tts": SOURCE_VERSION,
    "setuptools": "84.0.0",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "transformers": "5.2.0",
    "diffusers": "0.29.0",
    "resemble-perth": "1.0.1",
    "s3tokenizer": "0.3.0",
    "safetensors": "0.5.3",
    "spacy-pkuseg": "1.0.1",
}


class InstallError(RuntimeError):
    """Installation cannot continue safely."""


def _fail(message: str) -> NoReturn:
    raise InstallError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest() -> str:
    payload = json.dumps(
        [item.__dict__ for item in MODEL_FILES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def template_digest() -> str:
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        path = TEMPLATE_ROOT / name
        if not path.is_file() or path.is_symlink():
            _fail(f"runtime template is missing or unsafe: {path}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        _fail(f"unsafe manifest path: {relative!r}")
    if root.is_symlink():
        _fail(f"unsafe symbolic-link model root: {root}")
    resolved_root = root.resolve(strict=False)
    candidate = root / relative_path
    resolved_parent = candidate.parent.resolve(strict=False)
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        _fail(f"unsafe manifest path: {relative!r}")
    cursor = root
    for part in relative_path.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"unsafe symbolic-link model directory in path: {relative!r}")
    return candidate


def verify_model(root: Path = MODEL_DIR, *, full_hash: bool = True) -> list[str]:
    errors: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return [f"model directory is missing or unsafe: {root}"]
    for item in MODEL_FILES:
        try:
            path = _safe_child(root, item.path)
        except InstallError as exc:
            errors.append(str(exc))
            continue
        try:
            if path.is_symlink() or not path.is_file():
                errors.append(f"missing or symlinked model file: {item.path}")
                continue
            actual_size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot inspect {item.path}: {exc}")
            continue
        if actual_size != item.size:
            errors.append(f"wrong size for {item.path}: {actual_size:,} != {item.size:,}")
            continue
        if full_hash:
            try:
                actual_hash = _sha256(path)
            except OSError as exc:
                errors.append(f"cannot hash {item.path}: {exc}")
                continue
            if actual_hash != item.sha256:
                errors.append(f"SHA-256 mismatch for {item.path}")
    return errors


def _git_head(source: Path) -> str | None:
    if not (source / ".git").is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_clean(source: Path) -> bool:
    if not (source / ".git").is_dir():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip()


def _runtime_versions(python: Path) -> tuple[dict[str, str], str]:
    script = (
        "import importlib.metadata as m,json,sys; "
        f"names={list(EXPECTED_PACKAGES)!r}; "
        "print(json.dumps({'python': '.'.join(map(str,sys.version_info[:3])), "
        "'packages': {n:m.version(n) for n in names}}, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        return {}, f"cannot inspect isolated dependency versions: {exc}"
    python_version = str(payload.get("python", ""))
    versions = payload.get("packages")
    if not isinstance(versions, dict):
        return {}, "isolated dependency report is malformed"
    if not python_version.startswith("3.11."):
        return {}, f"isolated Python is {python_version or 'unknown'}, expected 3.11.x"
    return {str(k): str(v) for k, v in versions.items()}, ""


def _runtime_import_probe(python: Path, source: Path) -> str:
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(source / 'src')!r}); "
        "import torch, torchaudio; "
        "from chatterbox.mtl_tts import ChatterboxMultilingualTTS, MULTILINGUAL_T3_MODELS; "
        "assert MULTILINGUAL_T3_MODELS['v3'] == 't3_mtl23ls_v3.safetensors'; "
        "assert len(ChatterboxMultilingualTTS.get_supported_languages()) == 23"
    )
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        environment.pop(key, None)
    try:
        subprocess.run(
            [str(python), "-I", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        return f"isolated Chatterbox import probe failed: {detail.strip()[-800:]}"
    return ""


def _validated_venv_python(root: Path) -> tuple[Path | None, str]:
    """Validate uv's macOS interpreter symlink against its pyvenv.cfg."""

    venv = root / ".venv"
    bin_dir = venv / "bin"
    python = bin_dir / "python"
    config = venv / "pyvenv.cfg"
    if (
        not venv.is_dir()
        or venv.is_symlink()
        or not bin_dir.is_dir()
        or bin_dir.is_symlink()
        or not config.is_file()
        or config.is_symlink()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        return None, "isolated Python environment is incomplete"
    try:
        values = {
            key.strip(): value.strip()
            for line in config.read_text(encoding="utf-8").splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        }
        home = Path(values["home"])
        version = values["version_info"]
        implementation = values["implementation"]
    except (OSError, UnicodeDecodeError, KeyError, ValueError):
        return None, "isolated Python configuration is malformed"
    if implementation != "CPython" or not version.startswith("3.11.") or not home.is_absolute():
        return None, "isolated Python configuration is not pinned to CPython 3.11"
    if python.is_symlink():
        expected = home / "python3.11"
        try:
            resolved = python.resolve(strict=True)
            expected_resolved = expected.resolve(strict=True)
        except (OSError, RuntimeError):
            return None, "isolated Python interpreter link is broken"
        if resolved != expected_resolved or not resolved.is_file() or not os.access(resolved, os.X_OK):
            return None, "isolated Python interpreter link does not match pyvenv.cfg"
    return python, ""


def runtime_ready(root: Path = RUNTIME_ROOT) -> tuple[bool, str]:
    if not root.is_dir() or root.is_symlink():
        return False, "isolated voice runtime is missing"
    source = root / "source"
    if (source / PROVENANCE_NAME).is_file() and not (source / ".git").exists():
        source_ok, source_note = verify_source_tree(source, SOURCE_PIN)
        if not source_ok:
            return False, source_note
    else:
        if _git_head(source) != SOURCE_COMMIT:
            return False, "Chatterbox source is missing or at the wrong commit"
        if not _git_clean(source):
            return False, "pinned Chatterbox source has local changes"
        source_note = f"Git source at {SOURCE_COMMIT[:12]}"
    try:
        for name in ("pyproject.toml", "uv.lock"):
            installed = root / name
            template = TEMPLATE_ROOT / name
            if installed.is_symlink() or not installed.is_file() or installed.read_bytes() != template.read_bytes():
                return False, f"installed {name} does not match the checked-in lock template"
    except OSError as exc:
        return False, f"cannot inspect runtime lock files: {exc}"
    python, python_error = _validated_venv_python(root)
    if python_error or python is None:
        return False, python_error
    versions, error = _runtime_versions(python)
    if error:
        return False, error
    for name, expected in EXPECTED_PACKAGES.items():
        if versions.get(name) != expected:
            return False, f"{name} is {versions.get(name, 'missing')}, expected {expected}"
    import_error = _runtime_import_probe(python, source)
    if import_error:
        return False, import_error
    return True, f"Chatterbox {SOURCE_VERSION}; {source_note}; locked Python 3.11 runtime"


def model_ready(*, full_hash: bool = False) -> tuple[bool, str]:
    errors = verify_model(full_hash=full_hash)
    if errors:
        return False, errors[0]
    qualifier = "SHA-256 verified" if full_hash else "verified-size"
    return True, f"Chatterbox Multilingual V3 ({MODEL_TOTAL_BYTES:,} {qualifier} bytes)"


def receipt_ready() -> tuple[bool, str]:
    if not RECEIPT_PATH.is_file() or RECEIPT_PATH.is_symlink():
        return False, "installation receipt is missing"
    try:
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "installation receipt is unreadable"
    expected = {
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": manifest_digest(),
        "runtime_template_sha256": template_digest(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            return False, f"installation receipt has stale {key}"
    return True, "revision and manifest receipt matches"


def readiness(*, full_hash: bool = False) -> dict[str, object]:
    runtime_ok, runtime_note = runtime_ready()
    model_ok, model_note = model_ready(full_hash=full_hash)
    receipt_ok, receipt_note = receipt_ready()
    return {
        "ready": runtime_ok and model_ok and receipt_ok,
        "runtime_ready": runtime_ok,
        "model_ready": model_ok,
        "receipt_ready": receipt_ok,
        "runtime_note": runtime_note,
        "model_note": model_note,
        "receipt_note": receipt_note,
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_bytes": MODEL_TOTAL_BYTES,
    }


def _scrubbed_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/usr/bin/false",
            "PYTHONNOUSERSITE": "1",
            "UV_CACHE_DIR": str(UV_CACHE_DIR),
        }
    )
    return environment


def _run(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(command[:3]) + (" …" if len(command) > 3 else "")
    print(f"[character-voice] Running: {rendered}", flush=True)
    try:
        subprocess.run(list(command), cwd=cwd, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InstallError(f"command failed: {rendered} ({exc})") from exc


def _install_pinned_source(staging: Path, uv: str, environment: dict[str, str]) -> None:
    """Install package metadata/code from a disposable copy of the pinned checkout."""

    source = staging / "source"
    python, python_error = _validated_venv_python(staging)
    if python_error or python is None:
        _fail(python_error)
    # setuptools writes build products into its input tree.  Build from a
    # private regular-file copy so the canonical verified source remains clean;
    # this also avoids requiring Git merely to make a local build checkout.
    with tempfile.TemporaryDirectory(prefix=".chatterbox-build-", dir=staging) as temporary:
        build_source = Path(temporary) / "source"
        try:
            shutil.copytree(
                source,
                build_source,
                symlinks=False,
                ignore=shutil.ignore_patterns(".git", ".venv", PROVENANCE_NAME, "__pycache__", "*.pyc", "*.pyo"),
            )
        except OSError as exc:
            raise InstallError(f"cannot make private Chatterbox build copy: {exc}") from exc
        _run(
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--no-build-isolation",
                "--python",
                str(python),
                str(build_source),
            ],
            env=environment,
        )


def _prepare_runtime_staging() -> Path:
    if RUNTIME_ROOT.exists():
        ok, note = runtime_ready()
        if ok:
            return RUNTIME_ROOT
        _fail(f"refusing to replace unexpected runtime at {RUNTIME_ROOT}: {note}")
    if RUNTIME_STAGING.is_symlink():
        _fail(f"refusing symbolic-link staging directory: {RUNTIME_STAGING}")
    RUNTIME_STAGING.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("pyproject.toml", "uv.lock"):
        source = TEMPLATE_ROOT / name
        destination = RUNTIME_STAGING / name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            _fail(f"stale runtime staging file differs from template: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    return RUNTIME_STAGING


def _install_runtime() -> None:
    staging = _prepare_runtime_staging()
    if staging == RUNTIME_ROOT:
        print("[character-voice] Pinned runtime is already installed.")
        return
    source = staging / "source"
    environment = _scrubbed_environment()
    if source.exists():
        if source.is_symlink():
            _fail(f"refusing symbolic-link source in staging: {source}")
        if (source / PROVENANCE_NAME).is_file() and not (source / ".git").exists():
            source_ok, source_note = verify_source_tree(source, SOURCE_PIN)
            if not source_ok:
                _fail(f"refusing unexpected source snapshot in staging: {source_note}")
        elif _git_head(source) != SOURCE_COMMIT or not _git_clean(source):
            _fail(f"refusing unexpected source checkout in staging: {source}")
    else:
        try:
            print("[character-voice] Fetching immutable Chatterbox source archive.", flush=True)
            stage_pinned_source(
                source,
                SOURCE_PIN,
                cache_root=PATHS.cache_root / "source-archives",
            )
        except SourceInstallError as archive_error:
            git = shutil.which("git")
            if not git:
                raise
            print(
                f"[character-voice] Archive transport unavailable; using pinned Git fallback: {archive_error}",
                flush=True,
            )
            if source.exists():
                # This is the installer's private staging directory created by
                # the failed archive attempt.
                shutil.rmtree(source)
            _run([git, "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPO, str(source)], env=environment)
            _run([git, "-C", str(source), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], env=environment)
            _run([git, "-C", str(source), "checkout", "--detach", SOURCE_COMMIT], env=environment)
            if _git_head(source) != SOURCE_COMMIT or not _git_clean(source):
                _fail("Git checkout did not produce the clean pinned Chatterbox source")

    uv = shutil.which("uv")
    if not uv:
        _fail("uv is required on PATH for the hash-locked isolated runtime")
    UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            uv,
            "sync",
            "--locked",
            "--no-dev",
            "--project",
            str(staging),
            "--python",
            "3.11",
        ],
        env=environment,
    )
    _install_pinned_source(staging, uv, environment)
    ok, note = runtime_ready(staging)
    if not ok:
        _fail(f"staged voice runtime failed verification: {note}")
    os.replace(staging, RUNTIME_ROOT)
    print(f"[character-voice] Published isolated runtime at {RUNTIME_ROOT}")


def _model_file_url(item: ModelFile) -> str:
    """Return a public, immutable-revision URL for one declared model file."""

    repository = urllib.parse.quote(MODEL_REPO, safe="/")
    revision = urllib.parse.quote(MODEL_REVISION, safe="")
    path = urllib.parse.quote(item.path, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{path}?download=true"


def _verified_existing_file(path: Path, item: ModelFile) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == item.size
        and _sha256(path) == item.sha256
    )


def download_model_file(
    item: ModelFile,
    root: Path,
    *,
    report: Callable[[str], None] = print,
) -> Path:
    """Download one entire pinned file with system curl and publish atomically.

    Hugging Face's Xet bridge has been observed to ignore Range requests in this
    environment.  A legacy ``.partial`` is therefore never trusted for resume
    and is never overwritten by a new attempt.  It remains available until a
    complete fresh file passes both its exact byte-count and SHA-256 pin.
    """

    destination = _safe_child(root, item.path)
    legacy_partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after mkdir so a concurrently introduced parent link cannot be
    # followed by curl's output path.
    destination = _safe_child(root, item.path)
    if destination.is_file() and not destination.is_symlink():
        if _verified_existing_file(destination, item):
            report(f"[character-voice] Verified cached {item.path}")
            return destination
        _fail(f"staged model file is present but invalid: {destination}")
    if destination.exists():
        _fail(f"unsafe staged model path: {destination}")
    if legacy_partial.is_symlink() or (legacy_partial.exists() and not legacy_partial.is_file()):
        _fail(f"refusing unsafe legacy partial download: {legacy_partial}")

    if _verified_existing_file(legacy_partial, item):
        with legacy_partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(legacy_partial, destination)
        report(f"[character-voice] Verified and published completed partial {item.path}")
        return destination

    if not Path("/usr/bin/curl").is_file():
        _fail("system curl is required for the verified HTTPS model download")

    legacy_size = legacy_partial.stat().st_size if legacy_partial.is_file() else 0
    if legacy_size:
        report(
            f"[character-voice] Preserving existing incomplete {item.path}.partial "
            f"({legacy_size:,} bytes); the CDN cannot be resumed safely here."
        )
    report(
        f"[character-voice] Downloading complete pinned {item.path} "
        f"({item.size:,} bytes); curl will show transfer progress."
    )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.fresh-",
        suffix=".partial",
        dir=destination.parent,
    )
    os.close(temporary_fd)
    fresh = Path(temporary_name)
    os.chmod(fresh, 0o600)
    command = [
        "/usr/bin/curl",
        "--disable",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--tlsv1.2",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--header",
        "Accept-Encoding: identity",
        "--user-agent",
        "local-character-voice-installer/2.0",
        "--progress-bar",
        "--output",
        str(fresh),
        _model_file_url(item),
    ]
    try:
        try:
            subprocess.run(command, check=True, env=_scrubbed_environment())
        except (OSError, subprocess.CalledProcessError) as exc:
            raise InstallError(f"secure model download failed for {item.path}: {exc}") from exc
        try:
            actual_size = fresh.stat().st_size
            actual_hash = _sha256(fresh)
        except OSError as exc:
            raise InstallError(f"could not verify downloaded {item.path}: {exc}") from exc
        if actual_size != item.size or actual_hash != item.sha256:
            _fail(
                f"downloaded model file failed its pin: {item.path} "
                f"({actual_size:,}/{item.size:,} bytes, SHA-256 {actual_hash})"
            )
        with fresh.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(fresh, destination)
        if legacy_partial.exists():
            legacy_partial.unlink()
        report(f"[character-voice] Installed verified {item.path}")
        return destination
    finally:
        # A failed whole-file attempt is not resumable on the observed Xet
        # bridge.  Remove only this invocation's private file; never the older
        # user-visible partial that may still be useful for forensic recovery.
        fresh.unlink(missing_ok=True)


def _missing_model_bytes(root: Path) -> int:
    missing = 0
    for item in MODEL_FILES:
        path = _safe_child(root, item.path)
        if path.is_file() and not path.is_symlink() and path.stat().st_size == item.size:
            continue
        # Range resumes are deliberately not used: an incomplete legacy partial
        # cannot reduce the space needed for a complete fresh download.
        missing += item.size
    return missing


def _install_model() -> None:
    if MODEL_DIR.exists():
        errors = verify_model(full_hash=True)
        if errors:
            _fail(f"refusing to replace unexpected model directory at {MODEL_DIR}: {errors[0]}")
        print("[character-voice] Exact V3 model files are already installed.")
        return
    if MODEL_STAGING.is_symlink():
        _fail(f"refusing symbolic-link staging directory: {MODEL_STAGING}")
    MODEL_STAGING.mkdir(parents=True, exist_ok=True, mode=0o700)
    required = _missing_model_bytes(MODEL_STAGING) + RESERVE_BYTES
    free = shutil.disk_usage(MODEL_STAGING).free
    if free < required:
        _fail(
            f"need at least {required / 2**30:.1f} GiB free to finish safely; "
            f"only {free / 2**30:.1f} GiB is available"
        )
    for item in MODEL_FILES:
        download_model_file(item, MODEL_STAGING)
    errors = verify_model(MODEL_STAGING, full_hash=True)
    if errors:
        _fail(f"staged model failed final verification: {errors[0]}")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.replace(MODEL_STAGING, MODEL_DIR)
    print(f"[character-voice] Published exact V3 model files at {MODEL_DIR}")


def _write_receipt() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_version": SOURCE_VERSION,
        "source_archive_sha256": SOURCE_PIN.archive_sha256,
        "source_tree_sha256": SOURCE_PIN.tree_sha256,
        "source_mode": "verified-snapshot" if (SOURCE_DIR / PROVENANCE_NAME).is_file() else "git",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_bytes": MODEL_TOTAL_BYTES,
        "model_manifest_sha256": manifest_digest(),
        "runtime_template_sha256": template_digest(),
        "offline_inference_required": True,
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".install-receipt-", suffix=".json", dir=RUNTIME_ROOT)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, RECEIPT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def _print_status(status: dict[str, object]) -> None:
    print(json.dumps(status, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install revision-pinned Chatterbox Multilingual V3 for strictly local character voices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check-only", action="store_true", help="inspect local readiness without network or writes")
    parser.add_argument("--full-hash", action="store_true", help="SHA-256 all 3.21 GB of model files during checks")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-only", action="store_true", help="install only source and locked Python runtime")
    group.add_argument("--model-only", action="store_true", help="download and verify only the pinned public model files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.check_only:
            status = readiness(full_hash=args.full_hash)
            _print_status(status)
            return 0 if status["ready"] else 1
        if not args.model_only:
            _install_runtime()
        if not args.runtime_only:
            _install_model()
        if not args.runtime_only and not args.model_only:
            _write_receipt()
            status = readiness(full_hash=False)
            if not status["ready"]:
                _fail("post-install verification did not report ready")
        _print_status(readiness(full_hash=False))
        return 0
    except (InstallError, SourceInstallError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
