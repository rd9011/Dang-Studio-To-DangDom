#!/usr/bin/env python3
"""Install the optional FLUX.2 Klein Q4 frame composer in an isolated runtime.

The LTX environment is deliberately never imported or modified. Runtime code
comes from a Git checkout locked to an immutable commit and is installed from
its committed ``uv.lock``. Model files are downloaded at one immutable Hugging
Face revision, then every byte count and SHA-256 is checked before publication.
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
from typing import Iterable, NoReturn, Sequence

from studio_paths import PATHS


RUNTIME_ROOT = PATHS.frame_composer_runtime
SOURCE_DIR = RUNTIME_ROOT / "mlx-gen"
VENV_DIR = RUNTIME_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
MLXGEN = VENV_DIR / "bin" / "mlxgen"
MODEL_DIR = PATHS.models_root / "flux2-klein-4b-q4"
MODEL_STAGING_DIR = PATHS.models_root / ".flux2-klein-4b-q4-staging"
RECEIPT_PATH = RUNTIME_ROOT / "install-receipt.json"
UV_CACHE_DIR = PATHS.cache_root / "uv-frame-composer"

RUNTIME_REPO = "https://github.com/lpalbou/mlx-gen.git"
RUNTIME_TAG = "v0.38.0"
RUNTIME_COMMIT = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"
RUNTIME_VERSION = "0.38.0"
MODEL_REPO = "AbstractFramework/flux.2-klein-4b-4bit"
MODEL_REVISION = "4540c0084d0bcfdd29b19a2b0f62c77bef18f5bd"
RESERVE_BYTES = 5 * 2**30


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str


MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(".gitattributes", 1_580, "caf3ca02d5e883f643e939cbed54f94237f94c171797f0dfea5e25c7e33315b3"),
    ModelFile("LICENSE.md", 9_584, "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe"),
    ModelFile("README.md", 1_817, "0fb81dc299c40e496529a103710f3cd4dc2baab20e5a7a50c087dc2f91d4153f"),
    ModelFile("text_encoder/0.safetensors", 2_135_435_091, "ce0d934f24652ca44c43bb75f29337dbe7d9073a0eac8eeccf2d735330e3e541"),
    ModelFile("text_encoder/1.safetensors", 127_582_109, "8aeafe3b9e0d6d91d9b5f29bc2c5bcfc65d0b6af1643a83c82e89bc5d0b4913a"),
    ModelFile("text_encoder/model.safetensors.index.json", 51_332, "473013582eda6d5e53c303a266e3c06319584d6d872349356676c61976a76341"),
    ModelFile("tokenizer/chat_template.jinja", 4_168, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
    ModelFile("tokenizer/tokenizer.json", 11_422_650, "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
    ModelFile("tokenizer/tokenizer_config.json", 703, "12d954e6dd43cefa344345292406442871ea7f3237065d2f068f6a0402f92bf6"),
    ModelFile("transformer/0.safetensors", 2_145_323_701, "e41477939bd671e513b2e0cbf296ab315eaf6552654483ede0ac049b4537bca7"),
    ModelFile("transformer/1.safetensors", 34_727_730, "0bcdef7560025d155128152e1135aa7db53c1c8f352bc209d02ffa11d710d8b4"),
    ModelFile("transformer/model.safetensors.index.json", 26_908, "39b4b575326064b262f294c63c5db6ceb27c05e5d6131002f4829c83a1ace96e"),
    ModelFile("vae/0.safetensors", 165_107_912, "3bcd0e35c46efb728378e6a8a07f2bfb2aabda1d992f138cd098ffb670236bd5"),
    ModelFile("vae/model.safetensors.index.json", 17_547, "a7849ede655020cc52cde4aca036a9612596a24fbb68dde763f3c9ef9c43fed2"),
)
MODEL_TOTAL_BYTES = sum(item.size for item in MODEL_FILES)


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


def manifest_digest() -> str:
    value = [item.__dict__ for item in MODEL_FILES]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(command[:3]) + (" …" if len(command) > 3 else "")
    print(f"[frame-composer] Running: {rendered}", flush=True)
    try:
        subprocess.run(list(command), cwd=cwd, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InstallError(f"command failed: {rendered} ({exc})") from exc


def _git_head(source: Path = SOURCE_DIR) -> str | None:
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


def _git_clean(source: Path = SOURCE_DIR) -> bool:
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


def runtime_ready() -> tuple[bool, str]:
    if _git_head() != RUNTIME_COMMIT:
        return False, "pinned MLX-Gen source is missing or at the wrong commit"
    if not _git_clean():
        return False, "pinned MLX-Gen source has tracked or untracked local changes"
    source_note = f"Git source at {RUNTIME_COMMIT[:12]}"
    if not VENV_PYTHON.is_file() or not MLXGEN.is_file():
        return False, "isolated frame-composer virtual environment is incomplete"
    try:
        result = subprocess.run(
            [
                str(VENV_PYTHON),
                "-c",
                "import importlib.metadata as m; print(m.version('mlx-gen'))",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except (OSError, subprocess.CalledProcessError):
        return False, "the isolated MLX-Gen package cannot be imported"
    if result.stdout.strip() != RUNTIME_VERSION:
        return False, f"MLX-Gen version is {result.stdout.strip() or 'unknown'}, expected {RUNTIME_VERSION}"
    return True, f"MLX-Gen {RUNTIME_VERSION}; {source_note}"


def _safe_model_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        _fail(f"unsafe model manifest path: {relative}")
    if root.is_symlink():
        _fail(f"unsafe symbolic-link model root: {root}")
    candidate = root / relative_path
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
        _fail(f"unsafe model manifest path: {relative}")
    cursor = root
    for part in relative_path.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"unsafe symbolic-link model directory in path: {relative}")
    return candidate


def _expected_model_directories() -> set[str]:
    expected: set[str] = set()
    for item in MODEL_FILES:
        for parent in Path(item.path).parents:
            if parent != Path("."):
                expected.add(parent.as_posix())
    return expected


def _unexpected_model_entries(root: Path) -> list[str]:
    expected_files = {item.path for item in MODEL_FILES}
    expected_directories = _expected_model_directories()
    unexpected: list[str] = []
    try:
        entries = list(root.rglob("*"))
    except OSError as exc:
        return [f"cannot enumerate model tree: {exc}"]
    for path in entries:
        relative = path.relative_to(root).as_posix()
        try:
            if path.is_symlink():
                unexpected.append(f"symbolic link: {relative}")
            elif path.is_dir():
                if relative not in expected_directories:
                    unexpected.append(f"unexpected directory: {relative}")
            elif path.is_file():
                if relative not in expected_files:
                    unexpected.append(f"unexpected file: {relative}")
            else:
                unexpected.append(f"non-regular entry: {relative}")
        except OSError as exc:
            unexpected.append(f"unreadable entry: {relative} ({exc})")
    return sorted(unexpected)


def verify_model(root: Path = MODEL_DIR, *, full_hash: bool = True) -> list[str]:
    errors: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return [f"model directory is missing or unsafe: {root}"]
    for item in MODEL_FILES:
        try:
            path = _safe_model_path(root, item.path)
        except InstallError as exc:
            errors.append(str(exc))
            continue
        try:
            if path.is_symlink() or not path.is_file():
                errors.append(f"missing or symlinked model file: {item.path}")
                continue
            actual = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot inspect {item.path}: {exc}")
            continue
        if actual != item.size:
            errors.append(f"wrong size for {item.path}: {actual:,} != {item.size:,}")
            continue
        if full_hash:
            try:
                digest = _sha256(path)
            except OSError as exc:
                errors.append(f"cannot hash {item.path}: {exc}")
                continue
            if digest != item.sha256:
                errors.append(f"SHA-256 mismatch for {item.path}")
    errors.extend(_unexpected_model_entries(root))
    return errors


def model_ready() -> tuple[bool, str]:
    errors = verify_model(full_hash=False)
    if errors:
        return False, errors[0]
    return True, f"FLUX.2 Klein 4B Q4 ({MODEL_TOTAL_BYTES:,} verified-size bytes)"


def readiness() -> dict[str, object]:
    runtime_ok, runtime_note = runtime_ready()
    model_ok, model_note = model_ready()
    receipt_ok = False
    if RECEIPT_PATH.is_file() and not RECEIPT_PATH.is_symlink():
        try:
            receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
            receipt_ok = (
                receipt.get("runtime_commit") == RUNTIME_COMMIT
                and receipt.get("model_revision") == MODEL_REVISION
                and receipt.get("model_manifest_sha256") == manifest_digest()
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            receipt_ok = False
    ready = runtime_ok and model_ok and receipt_ok
    note = (
        "Pinned isolated runtime, exact Q4 model files, and installation receipt are present."
        if ready
        else "; ".join(
            part
            for ok, part in ((runtime_ok, runtime_note), (model_ok, model_note), (receipt_ok, "installation receipt is missing or stale"))
            if not ok
        )
    )
    return {"ready": ready, "runtime_ready": runtime_ok, "model_ready": model_ok, "receipt_ready": receipt_ok, "note": note}


def _install_runtime() -> None:
    if RUNTIME_ROOT.is_symlink():
        _fail(f"refusing unsafe symbolic-link runtime root: {RUNTIME_ROOT}")
    RUNTIME_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_DIR.is_symlink():
        _fail(f"refusing unsafe symbolic-link runtime source: {SOURCE_DIR}")
    if SOURCE_DIR.exists():
        if not SOURCE_DIR.is_dir():
            _fail(f"refusing non-directory runtime source at {SOURCE_DIR}")
        if _git_head() != RUNTIME_COMMIT:
            _fail(
                f"refusing to replace unexpected runtime source at {SOURCE_DIR}; move it aside and retry"
            )
        if not _git_clean():
            _fail(
                f"refusing to execute dependency setup from a dirty runtime source at {SOURCE_DIR}"
            )
    else:
        with tempfile.TemporaryDirectory(prefix=".mlx-gen-source-", dir=RUNTIME_ROOT.parent) as temporary:
            staging = Path(temporary) / "mlx-gen"
            git = shutil.which("git")
            if not git:
                _fail("Git is required to fetch the pinned MLX-Gen source")
            _run([git, "clone", "--filter=blob:none", "--no-checkout", RUNTIME_REPO, str(staging)])
            _run([git, "-C", str(staging), "fetch", "--depth", "1", "origin", RUNTIME_COMMIT])
            _run([git, "-C", str(staging), "sparse-checkout", "init", "--cone"])
            _run([git, "-C", str(staging), "sparse-checkout", "set", "src"])
            _run([git, "-C", str(staging), "checkout", "--detach", RUNTIME_COMMIT])
            if _git_head(staging) != RUNTIME_COMMIT:
                _fail("Git checkout did not resolve to the pinned MLX-Gen commit")
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            os.replace(staging, SOURCE_DIR)

    if SOURCE_DIR.is_symlink():
        _fail("pinned runtime source failed the safety gate before dependency setup")
    if _git_head() != RUNTIME_COMMIT or not _git_clean():
        _fail("pinned runtime source failed the clean-checkout gate before dependency setup")

    for label, directory in (
        ("virtual environment", VENV_DIR),
        ("uv cache", UV_CACHE_DIR),
    ):
        if directory.is_symlink():
            _fail(f"refusing unsafe symbolic-link {label}: {directory}")
        if directory.exists() and not directory.is_dir():
            _fail(f"refusing non-directory {label}: {directory}")

    uv = shutil.which("uv")
    if not uv:
        _fail("uv is required on PATH for the lockfile-exact runtime install")
    environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(VENV_DIR),
        "UV_CACHE_DIR": str(UV_CACHE_DIR),
        "UV_NO_PROGRESS": "1",
        "UV_HTTP_TIMEOUT": "600",
        "UV_CONCURRENT_DOWNLOADS": "2",
        "PYTHONNOUSERSITE": "1",
    }
    _run(
        [uv, "sync", "--frozen", "--no-dev", "--python", "3.12", "--project", str(SOURCE_DIR)],
        cwd=SOURCE_DIR,
        env=environment,
    )
    ok, note = runtime_ready()
    if not ok:
        _fail(note)


def _model_file_url(item: ModelFile) -> str:
    repository = urllib.parse.quote(MODEL_REPO, safe="/")
    revision = urllib.parse.quote(MODEL_REVISION, safe="")
    path = urllib.parse.quote(item.path, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{path}?download=true"


def _download_model_file(item: ModelFile) -> None:
    target = _safe_model_path(MODEL_STAGING_DIR, item.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        _fail(f"refusing symbolic-link staged model file: {item.path}")
    if target.is_file() and target.stat().st_size == item.size and _sha256(target) == item.sha256:
        print(f"[frame-composer] Verified cached model file: {item.path}", flush=True)
        return
    if target.exists() and not target.is_file():
        _fail(f"refusing non-file staged model target: {item.path}")

    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        _fail(f"refusing unsafe staged partial: {partial}")
    if partial.exists():
        partial.unlink()
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
        "--silent",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--header",
        "Accept-Encoding: identity",
        "--output",
        str(partial),
        _model_file_url(item),
    ]
    print(f"[frame-composer] Downloading pinned model file: {item.path}", flush=True)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InstallError(f"secure model download failed for {item.path}: {exc}") from exc
    try:
        size = partial.stat().st_size
        digest = _sha256(partial)
    except OSError as exc:
        raise InstallError(f"could not verify downloaded model file {item.path}: {exc}") from exc
    if size != item.size or digest != item.sha256:
        partial.unlink(missing_ok=True)
        _fail(
            f"downloaded model file failed its pin: {item.path} "
            f"({size:,} bytes, SHA-256 {digest})"
        )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, target)
    print(f"[frame-composer] Installed verified model file: {item.path}", flush=True)


def _install_model() -> None:
    ok, _ = model_ready()
    if ok:
        print("[frame-composer] Exact model files are already present.")
        return
    if MODEL_DIR.exists():
        _fail(f"refusing to overwrite incomplete or unexpected model directory: {MODEL_DIR}")
    ready, note = runtime_ready()
    if not ready:
        _fail(f"install the isolated runtime first: {note}")
    if MODEL_STAGING_DIR.is_symlink():
        _fail(f"refusing unsafe symbolic-link staging directory: {MODEL_STAGING_DIR}")
    MODEL_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODEL_STAGING_DIR.parent).free
    if free < MODEL_TOTAL_BYTES + RESERVE_BYTES:
        _fail(
            f"need at least {(MODEL_TOTAL_BYTES + RESERVE_BYTES) / 2**30:.1f} GiB free before model download; "
            f"only {free / 2**30:.1f} GiB is available"
        )
    if not Path("/usr/bin/curl").is_file():
        _fail("system curl is required for the verified HTTPS model download")
    for item in MODEL_FILES:
        _download_model_file(item)
    metadata = MODEL_STAGING_DIR / ".cache"
    if metadata.exists():
        if metadata.is_symlink() or not metadata.is_dir():
            _fail(f"refusing unsafe download metadata path: {metadata}")
        shutil.rmtree(metadata)
    errors = verify_model(MODEL_STAGING_DIR, full_hash=True)
    if errors:
        _fail("downloaded model failed verification: " + errors[0])
    unexpected = _unexpected_model_entries(MODEL_STAGING_DIR)
    if unexpected:
        _fail("staged model tree is not exact: " + unexpected[0])
    for item in MODEL_FILES:
        path = _safe_model_path(MODEL_STAGING_DIR, item.path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    os.replace(MODEL_STAGING_DIR, MODEL_DIR)


def _write_receipt() -> None:
    runtime_ok, runtime_note = runtime_ready()
    errors = verify_model(full_hash=True)
    if not runtime_ok or errors:
        _fail(runtime_note if not runtime_ok else errors[0])
    receipt = {
        "schema": 1,
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "runtime_repo": RUNTIME_REPO,
        "runtime_tag": RUNTIME_TAG,
        "runtime_commit": RUNTIME_COMMIT,
        "runtime_version": RUNTIME_VERSION,
        "runtime_source_mode": "git",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": manifest_digest(),
        "model_total_bytes": MODEL_TOTAL_BYTES,
    }
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT_PATH.with_name(f".{RECEIPT_PATH.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, RECEIPT_PATH)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="show pins, space, and readiness without changing files")
    action.add_argument("--install", action="store_true", help="install runtime and model, then write a verified receipt")
    action.add_argument("--verify", action="store_true", help="fully hash the installed model and verify the runtime")
    parser.add_argument("--runtime-only", action="store_true", help="with --install, stop after isolated runtime setup")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Runtime: {RUNTIME_REPO}@{RUNTIME_COMMIT} ({RUNTIME_TAG})")
    print(f"Model:   {MODEL_REPO}@{MODEL_REVISION}")
    print(f"Files:   {len(MODEL_FILES)} exact files, {MODEL_TOTAL_BYTES:,} bytes")
    print(f"Runtime destination: {RUNTIME_ROOT}")
    print(f"Model destination:   {MODEL_DIR}")
    if args.plan:
        print(json.dumps(readiness(), indent=2))
        return 0
    if args.verify:
        status = readiness()
        errors = verify_model(full_hash=True)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if not status["runtime_ready"] or not status["receipt_ready"]:
            print(f"ERROR: {status['note']}", file=sys.stderr)
            return 1
        print("Frame Composer verification passed.")
        return 0
    _install_runtime()
    if args.runtime_only:
        print("Isolated runtime installed; the LTX virtual environment was not touched.")
        return 0
    _install_model()
    _write_receipt()
    print("Frame Composer installation and full verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
