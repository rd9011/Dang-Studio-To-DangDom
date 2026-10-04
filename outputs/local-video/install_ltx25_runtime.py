#!/usr/bin/env python3
"""Install the revision-pinned LTX-2.5 MLX runtime under ``DANG_STUDIO_HOME``.

Model weights remain the responsibility of the existing Phosphene installer.
This installer checks out the public pinned source revision, applies the two
hash-verified low-memory backports, and creates a lockfile-exact isolated Python
environment.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Sequence

from studio_paths import PATHS


SOURCE_URL = "https://github.com/mrbizarro/ltx-2-mlx.git"
SOURCE_TAG = "v0.14.19+ltx25.7"
SOURCE_REVISION = "bf419b6f76e7753993eb7c3df3b15e1451310409"
UPSTREAM_URL = "https://github.com/dgrauet/ltx-2-mlx.git"
BACKPORT_COMMITS = (
    "add0142ece318bb09ad4bf295214724859fb241b",
    "99c9c7f124d3a56b5bfd63ebb8e96cf76cc7cd0a",
)
PATCH_PATHS = (
    "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py",
    "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/retake.py",
)
PATCH_SHA256 = {
    PATCH_PATHS[0]: "07e7de7e27cb9f003dd82bfb9d1b8e8702be9e241b28e96552a3228a76977a7c",
    PATCH_PATHS[1]: "1870e51a1b1de94996d75c63d44d27598bf7ea0d9f2b4cf6e98b2b04adde43d3",
}
RUNTIME_ROOT = PATHS.ltx_runtime
RECEIPT_PATH = RUNTIME_ROOT / "install-receipt.json"
UV_CACHE = PATHS.cache_root / "uv-ltx-runtime"


class RuntimeInstallError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise RuntimeInstallError(message)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RuntimeInstallError(f"cannot run {command[0]}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise RuntimeInstallError(
            f"command failed ({' '.join(command[:4])}): {detail or result.returncode}"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return _run(("git", "-C", str(root), *args)).stdout.decode().strip()


def verify_runtime(root: Path | None = None) -> tuple[bool, str]:
    root = RUNTIME_ROOT if root is None else root
    try:
        if root.is_symlink() or not root.is_dir():
            return False, "runtime directory is absent"
        if not (root / ".git").is_dir():
            return False, "runtime is not a public Git source checkout"
        if _git_output(root, "rev-parse", "HEAD") != SOURCE_REVISION:
            return False, "runtime source revision does not match the immutable pin"
        remote_tokens = _git_output(root, "remote", "-v").split()
        if SOURCE_URL not in remote_tokens:
            return False, "runtime source provenance is not the pinned repository"
        status = set(_git_output(root, "diff", "--name-only").splitlines())
        if status != set(PATCH_PATHS):
            return False, "runtime changes do not match the two-file low-memory backport"
        for relative, expected in PATCH_SHA256.items():
            path = root / relative
            if not path.is_file() or _sha256(path) != expected:
                return False, f"runtime backport hash mismatch: {relative}"
        cli = root / ".venv/bin/ltx-2-mlx"
        python = root / ".venv/bin/python"
        if not cli.is_file() or not python.is_file():
            return False, "isolated Python environment is incomplete"
        receipt_path = root / "install-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            return False, "runtime install receipt is absent"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = {
            "schema": "dang-studio-ltx-runtime/1",
            "source_revision": SOURCE_REVISION,
            "source_tag": SOURCE_TAG,
            "backport_commits": list(BACKPORT_COMMITS),
            "source_mode": "git",
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            return False, "runtime install receipt does not match the immutable pins"
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeInstallError) as exc:
        return False, f"runtime verification failed: {exc}"
    return True, f"{SOURCE_TAG} with verified Git source and low-memory edit backport"


def _atomic_receipt() -> None:
    payload = {
        "schema": "dang-studio-ltx-runtime/1",
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": SOURCE_URL,
        "source_tag": SOURCE_TAG,
        "source_revision": SOURCE_REVISION,
        "backport_source": UPSTREAM_URL,
        "backport_commits": list(BACKPORT_COMMITS),
        "patch_sha256": PATCH_SHA256,
        "source_mode": "git",
    }
    temporary = RECEIPT_PATH.with_name(f".{RECEIPT_PATH.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, RECEIPT_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def install_runtime() -> None:
    ready, note = verify_runtime()
    if ready:
        print(f"READY LTX runtime: {note}", flush=True)
        return
    if RUNTIME_ROOT.exists():
        _fail(
            f"refusing to replace an incomplete or unexpected runtime at {RUNTIME_ROOT}; "
            "move it aside before retrying"
        )
    uv = shutil.which("uv")
    git = shutil.which("git")
    if not uv:
        _fail("uv is required but was not found in PATH")
    if not git:
        _fail("Git is required to fetch the pinned public LTX source but was not found in PATH")
    RUNTIME_ROOT.parent.mkdir(parents=True, exist_ok=True)
    UV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ltx-runtime-", dir=RUNTIME_ROOT.parent) as raw:
        staging = Path(raw) / "source"
        print(f"FETCH pinned runtime {SOURCE_TAG}", flush=True)
        _run((git, "clone", "--no-checkout", SOURCE_URL, str(staging)))
        _run((git, "-C", str(staging), "checkout", "--detach", SOURCE_REVISION))
        tag_revision = _git_output(staging, "rev-list", "-n", "1", SOURCE_TAG)
        if tag_revision != SOURCE_REVISION:
            _fail(f"pinned tag resolved to unexpected revision {tag_revision}")
        for commit in BACKPORT_COMMITS:
            print(f"FETCH verified runtime backport {commit[:8]}", flush=True)
            _run((git, "-C", str(staging), "fetch", "--no-tags", UPSTREAM_URL, commit))
            patch = _run(
                (git, "-C", str(staging), "show", "--format=", "--binary", commit, "--", *PATCH_PATHS)
            ).stdout
            _run((git, "-C", str(staging), "apply", "--whitespace=nowarn", "-"), input_bytes=patch)
        for relative, expected in PATCH_SHA256.items():
            actual = _sha256(staging / relative)
            if actual != expected:
                _fail(f"low-memory backport verification failed for {relative}: {actual}")
        environment = {
            **os.environ,
            "UV_CACHE_DIR": str(UV_CACHE),
            "UV_PROJECT_ENVIRONMENT": str(staging / ".venv"),
            "PYTHONNOUSERSITE": "1",
        }
        print("INSTALL lockfile-exact isolated LTX Python runtime", flush=True)
        _run(
            (uv, "sync", "--frozen", "--no-dev", "--python", "3.11", "--project", str(staging)),
            env=environment,
        )
        os.replace(staging, RUNTIME_ROOT)
    _atomic_receipt()
    ready, note = verify_runtime()
    if not ready:
        _fail(f"new runtime failed final verification: {note}")
    print(f"INSTALLED LTX runtime: {note}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ready, note = verify_runtime()
    if args.plan:
        print(json.dumps({"ready": ready, "detail": note, "destination": str(RUNTIME_ROOT)}, indent=2))
        return 0
    if args.verify:
        print(json.dumps({"ready": ready, "detail": note}, indent=2))
        return 0 if ready else 1
    install_runtime()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeInstallError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
