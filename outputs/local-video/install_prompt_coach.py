#!/usr/bin/env python3
"""Install the exact local Qwen model used by Prompt Coach.

The installer resumably downloads Ollama's exact publisher archive and verifies
the full transport SHA-256 before safe extraction. Model transport is delegated
to Ollama's content-addressed puller, then the resulting manifest and every
referenced blob are verified against embedded sizes and SHA-256 digests before
a receipt is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Sequence

from studio_paths import PATHS


MODEL_NAME = "qwen3:4b-instruct-2507-q8_0"
MODEL_MANIFEST_RELATIVE = Path(
    "manifests/registry.ollama.ai/library/qwen3/4b-instruct-2507-q8_0"
)
MODEL_MANIFEST_BYTES = 859
MODEL_MANIFEST_SHA256 = "aa7252f68dda4d25dfffa65b3760af6d2c3231a140c0060c78d444686d98a374"
MODEL_BLOBS = {
    "48d7fd0fa10098dd8b26f0147e17118b84607ac02a3ca4eacdd989e8b02be8fe": 485,
    "af6e43ab13611311226e6f809f6a39b1a87b6df613bbf79468059f92ce819c4a": 4_280_404_960,
    "eade0a07cac7712787bbce23d12f9306adb4781d873d1df6e16f7840fa37afec": 1_379,
    "d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12": 11_338,
    "0914c7781e001948488d937994217538375b4fd8c1466c5e7a625221abd3ea7a": 119,
}
OLLAMA_VERSION = "0.34.4"
OLLAMA_EXECUTABLE_SHA256 = "bba8b79eac84ab092b9568978c1116e6c950eda588961b14d8608ac683a2af0b"
OLLAMA_ARCHIVE_URL = (
    "https://github.com/ollama/ollama/releases/download/v0.34.4/Ollama-darwin.zip"
)
OLLAMA_ARCHIVE_BYTES = 198_725_086
OLLAMA_ARCHIVE_SHA256 = "f7ed834269e98929d9ed63d884c982287ec09d7ab6ee89587f90b9697a003ff1"
OLLAMA_RESOURCE_PREFIX = ("Ollama.app", "Contents", "Resources")
RUNTIME_ROOT = PATHS.ollama_runtime
MODEL_ROOT = PATHS.ollama_models_root
RECEIPT = MODEL_ROOT / "dang-studio-prompt-coach-receipt.json"
RUNTIME_ARCHIVE = PATHS.cache_root / "prompt-coach/Ollama-darwin-v0.34.4.zip"


class InstallError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise InstallError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_runtime_archive() -> None:
    RUNTIME_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = RUNTIME_ARCHIVE.with_suffix(".zip.part")
    curl = shutil.which("curl")
    if not curl:
        _fail("macOS curl is required for the resumable Ollama runtime download")
    print(f"FETCH pinned Ollama {OLLAMA_VERSION} runtime archive (~199 MB)", flush=True)
    result = subprocess.run(
        (
            curl,
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            OLLAMA_ARCHIVE_URL,
        ),
        check=False,
    )
    if result.returncode:
        _fail(f"Ollama runtime download exited with status {result.returncode}")
    if partial.stat().st_size != OLLAMA_ARCHIVE_BYTES:
        _fail(
            f"Ollama runtime archive size differs: {partial.stat().st_size} "
            f"!= {OLLAMA_ARCHIVE_BYTES}"
        )
    if _sha256(partial) != OLLAMA_ARCHIVE_SHA256:
        _fail("Ollama runtime archive failed its publisher-pinned SHA-256")
    os.replace(partial, RUNTIME_ARCHIVE)


def _extract_runtime_archive(archive: Path, destination: Path) -> None:
    """Extract only Ollama's required Resources subtree from the pinned ZIP."""

    try:
        with zipfile.ZipFile(archive) as bundle:
            selected: list[tuple[zipfile.ZipInfo, Path]] = []
            for member in bundle.infolist():
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    _fail(f"unsafe Ollama ZIP member: {member.filename}")
                if tuple(path.parts[: len(OLLAMA_RESOURCE_PREFIX)]) != OLLAMA_RESOURCE_PREFIX:
                    continue
                relative = Path(*path.parts[len(OLLAMA_RESOURCE_PREFIX):])
                if not relative.parts or member.is_dir():
                    continue
                mode = member.external_attr >> 16
                kind = mode & 0o170000
                if kind not in {0, 0o100000}:
                    _fail(f"non-regular Ollama ZIP member: {member.filename}")
                selected.append((member, relative))
            if not selected or not any(relative == Path("ollama") for _, relative in selected):
                _fail("Ollama ZIP does not contain the expected Resources subtree")
            destination.mkdir(parents=True)
            for member, relative in selected:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as input_handle, target.open("xb") as output:
                    shutil.copyfileobj(input_handle, output, 8 << 20)
                    output.flush()
                    os.fsync(output.fileno())
                mode = member.external_attr >> 16
                target.chmod((mode & 0o777) or (0o755 if relative.name in {"ollama", "llama-server", "llama-quantize"} else 0o644))
    except (OSError, zipfile.BadZipFile) as exc:
        raise InstallError(f"cannot extract pinned Ollama runtime: {exc}") from exc


def verify_runtime() -> tuple[bool, str]:
    executable = RUNTIME_ROOT / "ollama"
    try:
        if RUNTIME_ROOT.is_symlink() or not executable.is_file() or not os.access(executable, os.X_OK):
            return False, "pinned local Ollama runtime is absent"
        if _sha256(executable) != OLLAMA_EXECUTABLE_SHA256:
            return False, "Ollama executable hash differs"
        result = subprocess.run(
            (str(executable), "--version"), text=True, capture_output=True, timeout=10, check=False
        )
        output = (result.stdout + result.stderr).lower()
        if OLLAMA_VERSION not in output:
            return False, f"Ollama version is not {OLLAMA_VERSION}"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"cannot verify Ollama runtime: {exc}"
    return True, f"pinned Ollama {OLLAMA_VERSION} runtime"


def verify_model() -> tuple[bool, str]:
    manifest_path = MODEL_ROOT / MODEL_MANIFEST_RELATIVE
    try:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return False, "Qwen model manifest is absent"
        if manifest_path.stat().st_size != MODEL_MANIFEST_BYTES or _sha256(manifest_path) != MODEL_MANIFEST_SHA256:
            return False, "Qwen model manifest differs from its immutable pin"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = {manifest["config"]["digest"], *(layer["digest"] for layer in manifest["layers"])}
        expected = {f"sha256:{digest}" for digest in MODEL_BLOBS}
        if declared != expected:
            return False, "Qwen manifest references an unexpected blob set"
        for digest, expected_bytes in MODEL_BLOBS.items():
            path = MODEL_ROOT / "blobs" / f"sha256-{digest}"
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
                return False, f"Qwen blob is missing or has wrong size: {digest[:12]}"
            if _sha256(path) != digest:
                return False, f"Qwen blob hash differs: {digest[:12]}"
        if not RECEIPT.is_file() or RECEIPT.is_symlink():
            return False, "Prompt Coach receipt is absent"
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if receipt.get("schema") != "dang-studio-prompt-coach/1":
            return False, "Prompt Coach receipt schema differs"
    except (OSError, UnicodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"cannot verify Prompt Coach model: {exc}"
    return True, "exact Qwen3 4B Instruct Q8 model"


def _install_runtime() -> None:
    ready, _ = verify_runtime()
    if ready:
        return
    if RUNTIME_ROOT.exists():
        _fail(f"refusing to replace incomplete or unexpected Ollama runtime: {RUNTIME_ROOT}")
    RUNTIME_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = RUNTIME_ROOT.with_name(f".{RUNTIME_ROOT.name}.{os.getpid()}.staging")
    if staging.exists():
        _fail(f"stale Ollama staging path requires inspection: {staging}")
    try:
        if RUNTIME_ARCHIVE.is_file():
            if (
                RUNTIME_ARCHIVE.stat().st_size != OLLAMA_ARCHIVE_BYTES
                or _sha256(RUNTIME_ARCHIVE) != OLLAMA_ARCHIVE_SHA256
            ):
                _fail(f"cached Ollama runtime archive is incomplete or unexpected: {RUNTIME_ARCHIVE}")
        else:
            _download_runtime_archive()
        _extract_runtime_archive(RUNTIME_ARCHIVE, staging)
        if _sha256(staging / "ollama") != OLLAMA_EXECUTABLE_SHA256:
            _fail("downloaded Ollama executable failed its immutable hash")
        os.replace(staging, RUNTIME_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _fail(f"private Ollama service exited during startup ({process.returncode})")
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    _fail("private Ollama service did not become ready")


def _write_receipt() -> None:
    payload = {
        "schema": "dang-studio-prompt-coach/1",
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ollama_version": OLLAMA_VERSION,
        "ollama_sha256": OLLAMA_EXECUTABLE_SHA256,
        "model": MODEL_NAME,
        "manifest_bytes": MODEL_MANIFEST_BYTES,
        "manifest_sha256": MODEL_MANIFEST_SHA256,
        "blobs": MODEL_BLOBS,
    }
    temporary = RECEIPT.with_name(f".{RECEIPT.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, RECEIPT)
    finally:
        temporary.unlink(missing_ok=True)


def install() -> None:
    _install_runtime()
    model_ready, note = verify_model()
    if model_ready:
        print(f"READY Prompt Coach: {note}", flush=True)
        return
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    host = f"127.0.0.1:{port}"
    url = f"http://{host}"
    environment = {
        **os.environ,
        "OLLAMA_HOST": host,
        "OLLAMA_MODELS": str(MODEL_ROOT),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_KEEP_ALIVE": "0",
    }
    executable = RUNTIME_ROOT / "ollama"
    service = subprocess.Popen(
        (str(executable), "serve"),
        cwd=RUNTIME_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_ready(url, service)
        print(f"FETCH {MODEL_NAME} with resumable content-addressed Ollama transport", flush=True)
        pull = subprocess.run(
            (str(executable), "pull", MODEL_NAME),
            cwd=RUNTIME_ROOT,
            env=environment,
            check=False,
        )
        if pull.returncode:
            _fail(f"Ollama model pull exited with status {pull.returncode}")
    finally:
        service.terminate()
        try:
            service.wait(timeout=10)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait(timeout=5)
    # Write the receipt only after the immutable model identity verifies.  The
    # first verification naturally reports the missing receipt, so validate
    # the manifest and blobs explicitly here.
    manifest = MODEL_ROOT / MODEL_MANIFEST_RELATIVE
    if not manifest.is_file() or manifest.stat().st_size != MODEL_MANIFEST_BYTES or _sha256(manifest) != MODEL_MANIFEST_SHA256:
        _fail("downloaded Qwen manifest differs from the immutable approved model")
    for digest, size in MODEL_BLOBS.items():
        blob = MODEL_ROOT / "blobs" / f"sha256-{digest}"
        if not blob.is_file() or blob.stat().st_size != size or _sha256(blob) != digest:
            _fail(f"downloaded Qwen blob failed verification: {digest[:12]}")
    _write_receipt()
    ready, note = verify_model()
    if not ready:
        _fail(f"new Prompt Coach model failed verification: {note}")
    print(f"INSTALLED Prompt Coach: {note}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    runtime_ready, runtime_note = verify_runtime()
    model_ready, model_note = verify_model()
    if args.plan:
        print(json.dumps({"ready": runtime_ready and model_ready, "runtime": runtime_note, "model": model_note}, indent=2))
        return 0
    if args.verify:
        print(json.dumps({"ready": runtime_ready and model_ready, "runtime": runtime_note, "model": model_note}, indent=2))
        return 0 if runtime_ready and model_ready else 1
    install()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InstallError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
