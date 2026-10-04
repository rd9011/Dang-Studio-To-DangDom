#!/usr/bin/env python3
"""Install the pinned CoreML Real-ESRGAN runtime and x4plus model.

The reduced upstream source comes from an immutable, hash-pinned HTTPS archive
and is installed from its committed ``uv.lock``. A checkout at the same commit
is the network fallback. The small model release ZIP is treated only as transport:
three allowlisted files are extracted safely and verified against the hashes
already enforced by ``ai_video_upscale.py`` before atomic publication.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Sequence

from ai_video_upscale import (
    EXPECTED_RUNTIME,
    EXPECTED_UPSTREAM_COMMIT,
    EXPECTED_UPSTREAM_SCRIPT_SHA256,
    MODEL_FILE_PINS,
    UPSTREAM_SOURCE_PIN,
    sha256_file,
)
from pinned_source import (
    PROVENANCE_NAME,
    SourceInstallError,
    stage_pinned_source,
    verify_source_tree,
)
from studio_paths import PATHS


SOURCE_URL = "https://github.com/hanxiao/real-esrgan-coreml.git"
MODEL_URL = (
    "https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/"
    "RealESRGAN_x4plus_522_fp16.zip"
)
RUNTIME_ROOT = PATHS.ai_upscaler_runtime
MODEL_RELATIVE = Path("weights/RealESRGAN_x4plus_522_fp16.mlpackage")
MODEL_ROOT = RUNTIME_ROOT / MODEL_RELATIVE
RECEIPT = RUNTIME_ROOT / "install-receipt.json"
CACHE_ZIP = PATHS.cache_root / "ai-upscaler/RealESRGAN_x4plus_522_fp16.zip"
UV_CACHE = PATHS.cache_root / "uv-ai-upscaler"


class InstallError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise InstallError(message)


def _run(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, env=env, text=True, capture_output=True, check=False
        )
    except OSError as exc:
        raise InstallError(f"cannot run {command[0]}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise InstallError(f"command failed ({' '.join(command[:4])}): {detail}")
    return result.stdout.strip()


def _verify_model(root: Path) -> tuple[bool, str]:
    expected = {pin.relative_path for pin in MODEL_FILE_PINS}
    if root.is_symlink() or not root.is_dir():
        return False, "CoreML package is absent"
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        return False, f"CoreML package topology differs (missing={sorted(expected-actual)}, extra={sorted(actual-expected)})"
    for pin in MODEL_FILE_PINS:
        path = root / pin.relative_path
        if path.is_symlink() or path.stat().st_size != pin.size:
            return False, f"CoreML model size differs: {pin.relative_path}"
        if sha256_file(path) != pin.sha256:
            return False, f"CoreML model hash differs: {pin.relative_path}"
    return True, "exact CoreML x4plus package verified"


def verify_install() -> tuple[bool, str]:
    try:
        if RUNTIME_ROOT.is_symlink() or not RUNTIME_ROOT.is_dir():
            return False, "upscaler runtime is absent"
        if (RUNTIME_ROOT / PROVENANCE_NAME).is_file() and not (RUNTIME_ROOT / ".git").exists():
            source_ok, source_note = verify_source_tree(
                RUNTIME_ROOT,
                UPSTREAM_SOURCE_PIN,
                allowed_runtime_entries=(".venv", "weights", RECEIPT.name),
            )
            if not source_ok:
                return False, source_note
            source_mode = "verified-snapshot"
        else:
            git = shutil.which("git")
            if not git:
                return False, "Git upscaler checkout cannot be verified without Git on PATH"
            head = _run((git, "-C", str(RUNTIME_ROOT), "rev-parse", "HEAD"))
            if head != EXPECTED_UPSTREAM_COMMIT:
                return False, "upscaler source revision differs"
            source_mode = "git"
        script = RUNTIME_ROOT / "upscale.py"
        if not script.is_file() or sha256_file(script) != EXPECTED_UPSTREAM_SCRIPT_SHA256:
            return False, "upstream upscale.py hash differs"
        model_ok, model_note = _verify_model(MODEL_ROOT)
        if not model_ok:
            return False, model_note
        python = RUNTIME_ROOT / ".venv/bin/python"
        if not python.is_file():
            return False, "isolated upscaler Python is absent"
        code = (
            "import importlib.metadata as m,json; "
            "print(json.dumps({n:m.version(n) for n in "
            + repr(tuple(EXPECTED_RUNTIME))
            + "}))"
        )
        versions = json.loads(_run((str(python), "-c", code)))
        if versions != EXPECTED_RUNTIME:
            return False, f"upscaler dependency versions differ: {versions}"
        if not RECEIPT.is_file() or RECEIPT.is_symlink():
            return False, "upscaler receipt is absent"
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if receipt.get("schema") != "dang-studio-ai-upscaler/1":
            return False, "upscaler receipt schema differs"
        if receipt.get("source_commit") != EXPECTED_UPSTREAM_COMMIT:
            return False, "upscaler receipt source pin differs"
    except (OSError, UnicodeError, json.JSONDecodeError, InstallError) as exc:
        return False, f"upscaler verification failed: {exc}"
    return True, f"pinned CoreML Real-ESRGAN x4plus runtime and model ({source_mode})"


def _download_zip() -> None:
    CACHE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_ZIP.with_suffix(".zip.part")
    print("FETCH pinned CoreML x4plus transport archive (~31 MB)", flush=True)
    curl = shutil.which("curl")
    if not curl:
        _fail("curl is required on PATH for the resumable model download")
    # The ZIP itself is transport, not a trust root.  Curl resumes its private
    # partial; extraction below accepts only the three embedded, hash-pinned
    # CoreML files and promotes nothing before all three verify.
    _run(
        (
            curl,
            "--fail",
            "--location",
            "--retry",
            "3",
            "--max-filesize",
            "50000000",
            "--continue-at",
            "-",
            "--output",
            str(temporary),
            MODEL_URL,
        )
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, CACHE_ZIP)


def _extract_verified_model(archive: Path, destination: Path) -> None:
    expected = {pin.relative_path: pin for pin in MODEL_FILE_PINS}
    try:
        with zipfile.ZipFile(archive) as source:
            members = [item for item in source.infolist() if not item.is_dir()]
            selected: dict[str, zipfile.ZipInfo] = {}
            for member in members:
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    _fail(f"unsafe CoreML ZIP member: {member.filename}")
                mode = member.external_attr >> 16
                if mode and (mode & 0o170000) not in {0, 0o100000}:
                    _fail(f"non-regular CoreML ZIP member: {member.filename}")
                for relative in expected:
                    suffix = Path("RealESRGAN_x4plus_522_fp16.mlpackage") / relative
                    if tuple(path.parts[-len(suffix.parts):]) == suffix.parts:
                        if relative in selected:
                            _fail(f"duplicate CoreML ZIP member for {relative}")
                        if member.file_size != expected[relative].size:
                            _fail(f"CoreML ZIP member size differs for {relative}")
                        selected[relative] = member
            if set(selected) != set(expected):
                _fail(f"CoreML ZIP topology differs; found {sorted(selected)}")
            destination.mkdir(parents=True)
            for relative, member in selected.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_handle, target.open("xb") as output:
                    shutil.copyfileobj(input_handle, output, 8 << 20)
                    output.flush()
                    os.fsync(output.fileno())
    except (OSError, zipfile.BadZipFile) as exc:
        raise InstallError(f"cannot extract CoreML model archive: {exc}") from exc
    ready, detail = _verify_model(destination)
    if not ready:
        _fail(detail)


def _write_receipt(root: Path) -> None:
    payload = {
        "schema": "dang-studio-ai-upscaler/1",
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": SOURCE_URL,
        "source_commit": EXPECTED_UPSTREAM_COMMIT,
        "source_archive_sha256": UPSTREAM_SOURCE_PIN.archive_sha256,
        "source_tree_sha256": UPSTREAM_SOURCE_PIN.tree_sha256,
        "source_mode": "verified-snapshot" if (root / PROVENANCE_NAME).is_file() else "git",
        "upstream_script_sha256": EXPECTED_UPSTREAM_SCRIPT_SHA256,
        "model_url": MODEL_URL,
        "model_files": [pin.__dict__ for pin in MODEL_FILE_PINS],
        "runtime_versions": EXPECTED_RUNTIME,
    }
    path = root / RECEIPT.name
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def install() -> None:
    ready, note = verify_install()
    if ready:
        print(f"READY AI upscaler: {note}", flush=True)
        return
    if RUNTIME_ROOT.exists():
        _fail(f"refusing to replace incomplete or unexpected upscaler runtime: {RUNTIME_ROOT}")
    uv = shutil.which("uv")
    if not uv:
        _fail("uv is required on PATH for the upscaler runtime")
    if not CACHE_ZIP.is_file():
        _download_zip()
    RUNTIME_ROOT.parent.mkdir(parents=True, exist_ok=True)
    UV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ai-upscaler-", dir=RUNTIME_ROOT.parent) as raw:
        staging = Path(raw) / "runtime"
        print(f"FETCH immutable Real-ESRGAN CoreML source archive {EXPECTED_UPSTREAM_COMMIT[:12]}", flush=True)
        try:
            stage_pinned_source(
                staging,
                UPSTREAM_SOURCE_PIN,
                cache_root=PATHS.cache_root / "source-archives",
            )
        except SourceInstallError as archive_error:
            git = shutil.which("git")
            if not git:
                raise
            print(f"Archive transport unavailable; using pinned Git fallback: {archive_error}", flush=True)
            if staging.exists():
                shutil.rmtree(staging)
            _run((git, "clone", "--no-checkout", SOURCE_URL, str(staging)))
            _run((git, "-C", str(staging), "checkout", "--detach", EXPECTED_UPSTREAM_COMMIT))
        if sha256_file(staging / "upscale.py") != EXPECTED_UPSTREAM_SCRIPT_SHA256:
            _fail("pinned source upscale.py failed its embedded hash")
        environment = {
            **os.environ,
            "UV_CACHE_DIR": str(UV_CACHE),
            "UV_PROJECT_ENVIRONMENT": str(staging / ".venv"),
            "PYTHONNOUSERSITE": "1",
        }
        print("INSTALL lockfile-exact CoreML upscaler runtime", flush=True)
        _run((uv, "sync", "--frozen", "--no-dev", "--python", "3.12", "--project", str(staging)), env=environment)
        package = staging / MODEL_RELATIVE
        if package.exists():
            shutil.rmtree(package)
        try:
            _extract_verified_model(CACHE_ZIP, package)
        except InstallError:
            # This is installer-owned transport cache, never user media.  A
            # corrupt complete ZIP cannot be resumed into validity.
            CACHE_ZIP.unlink(missing_ok=True)
            raise
        _write_receipt(staging)
        os.replace(staging, RUNTIME_ROOT)
    ready, note = verify_install()
    if not ready:
        _fail(f"new upscaler failed final verification: {note}")
    print(f"INSTALLED AI upscaler: {note}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    ready, note = verify_install()
    if args.plan:
        print(json.dumps({"ready": ready, "detail": note, "destination": str(RUNTIME_ROOT)}, indent=2))
        return 0
    if args.verify:
        print(json.dumps({"ready": ready, "detail": note}, indent=2))
        return 0 if ready else 1
    install()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InstallError, SourceInstallError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
