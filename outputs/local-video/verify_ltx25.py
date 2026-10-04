#!/usr/bin/env python3
"""Verify the pinned Phosphene LTX-2.5 Q4 runtime and split model packs.

The normal path is deliberately quick: it checks the immutable install
receipts, every declared byte size, all small-file hashes, the pinned source
checkout, ffmpeg, and a real MLX Metal operation. ``--full-hash`` additionally
streams the large safetensors files through SHA-256 (about 25.6 GiB total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from import_ddalcu_gemma_fallback import (
    FALLBACK_MODEL as DDALCU_FALLBACK_GEMMA_MODEL,
    FallbackImportError,
    verify_fallback_install,
)
from studio_paths import PATHS


HERE = PATHS.app_root
WORKSPACE = PATHS.workspace  # Legacy compatibility; new code should use PATHS.
RUNTIME = PATHS.ltx_runtime
VENV_PYTHON = RUNTIME / ".venv" / "bin" / "python"
CLI = RUNTIME / ".venv" / "bin" / "ltx-2-mlx"
MODELS_ROOT = PATHS.models_root
MODEL = MODELS_ROOT / "ltx-2.5-mlx-q4"
GEMMA_MODEL = MODELS_ROOT / "gemma4-12b-ltx25-q4"

SOURCE_REPOSITORY = "mrbizarro/ltx-2-mlx"
SOURCE_URL = "https://github.com/mrbizarro/ltx-2-mlx.git"
SOURCE_TAG = "v0.14.19+ltx25.7"
SOURCE_REVISION = "bf419b6f76e7753993eb7c3df3b15e1451310409"
# Minimal backport of upstream dgrauet commits add0142 + 99c9c7f.  It fixes
# Retake/Extend's mandatory frame-rate forwarding and exposes block streaming
# so the Q8 dev transformer can run on a 16 GB Mac.  Any byte drift remains a
# hard failure; ordinary generation may also run from the pristine tag.
LOW_RAM_EDIT_PATCHES = {
    "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py": (
        "07e7de7e27cb9f003dd82bfb9d1b8e8702be9e241b28e96552a3228a76977a7c"
    ),
    "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/retake.py": (
        "1870e51a1b1de94996d75c63d44d27598bf7ea0d9f2b4cf6e98b2b04adde43d3"
    ),
}
RELEASE_REPOSITORY = "mrbizarro/Phosphene"
RELEASE_TAG = "weights-ltx25-v1"

# Backward-compatible names imported by the edit wrapper and older callers.
MODEL_REPO = f"{RELEASE_REPOSITORY} GitHub release"
MODEL_REVISION = RELEASE_TAG

CURRENT_LICENSE_SIZE = 34_545
CURRENT_LICENSE_SHA256 = "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654"
RECEIPT_NAME = ".phosphene-install-receipt.json"
SMALL_HASH_LIMIT = 64 << 20


@dataclass(frozen=True)
class Artifact:
    size: int
    sha256: str


@dataclass(frozen=True)
class Pack:
    key: str
    path: Path
    manifest_asset: str
    manifest_sha256: str
    files: dict[str, Artifact]

    @property
    def bytes(self) -> int:
        return sum(item.size for item in self.files.values())


# Exact published manifests, except for the license file. The publisher
# replaced that shared release asset after publishing the manifests; the local
# installer independently pins and records the current license bytes below.
BASE_FILES: dict[str, Artifact] = {
    "LICENSE-LTX-2.x-Community-License.md": Artifact(CURRENT_LICENSE_SIZE, CURRENT_LICENSE_SHA256),
    "NOTICE.md": Artifact(4_884, "354c701af72ffc11f09bd6d17b8ac59e7021aad3731691e54a8a88daa82bf04e"),
    "audio_vae.safetensors": Artifact(106_509_004, "481723e829eb184a0a28693b94ff8471a261c0bd183ff61c20032f963367f6f3"),
    "config.json": Artifact(2_557, "533a4b8d5ae82218a280976ad202a7a2a6fdffc033e43cc8cd24d83c229a646a"),
    "connector.safetensors": Artifact(6_344_495_432, "120b43898339895d0342dfc979080b23df55d60ac7179f487f76f895856e6dec"),
    "duration_head.safetensors": Artifact(3_808_330, "17188b898d7194e1e8687c6aa03cdc5daf5dca25a11cc24557d93961e5351800"),
    "embedded_config.json": Artifact(7_589, "78ac0af8c8207407f07024d7b9d4642a796c4226b61ca847b86b24a150d5c886"),
    "manifest.json": Artifact(2_164, "90bae3d285cb4af4a998d92c859f869a59f2b95515ec6082ef574967e839370c"),
    "phosphene_quant_manifest.json": Artifact(3_628, "66a3f6103f34a74456b79c7716203c4c5e06914caae1b11a2d98749f5973d22e"),
    "quantize_config.json": Artifact(101, "4263d314fc42edc54f62bab8fd50419a024c3459f223b761aab589515ce3614a"),
    "spatial_upscaler_x2_v1_1.safetensors": Artifact(995_745_088, "0883210010de110994241be1bea97d705b8c6f598f3da51f662b6a15c321dfb0"),
    "spatial_upscaler_x2_v1_1_config.json": Artifact(275, "c0646d97bf1bc3fa6e66ec7d17f16ef31b717e64bf86474b7dc3993b6c42346a"),
    "split_model.json": Artifact(305, "57eeeec0731655335ee6cdac9ba2b847513ee934991629a012cd9e6bced7d882"),
    "temporal_upscaler_x2_v1_0.safetensors": Artifact(261_945_600, "84beac95274aace69f576343c89e6815e7d2b45028780a234ad0b4c682e455cc"),
    "temporal_upscaler_x2_v1_0_config.json": Artifact(273, "f890af9e95cde921d6e2eb6c334742a81c53c25c3fdc817b4b363622e2c08f3f"),
    "transformer-distilled.safetensors": Artifact(11_320_074_467, "7c8c4f71a4cf88c683a7be47983bc98749883fbc9d702750487e0943c8e48102"),
    "vae_decoder.safetensors": Artifact(814_349_496, "be257d967de1959e459bf38965c3724d2e6460c1e285711a8e405c3e6d78c75e"),
    "vae_encoder.safetensors": Artifact(637_885_234, "11f6297ea96731e1263b634e61a3847abbcca3e499c1b5995ca2c7484a2baf0e"),
    "vocoder.safetensors": Artifact(258_313_296, "fd10ffff704ec6393220e3e1962dee75ea4b91b2075b072ebbfe2d1b2b2f7b99"),
}

GEMMA_FILES: dict[str, Artifact] = {
    "LICENSE-LTX-2.x-Community-License.md": Artifact(CURRENT_LICENSE_SIZE, CURRENT_LICENSE_SHA256),
    "NOTICE.md": Artifact(4_884, "354c701af72ffc11f09bd6d17b8ac59e7021aad3731691e54a8a88daa82bf04e"),
    "chat_template.jinja": Artifact(18_683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"),
    "config.json": Artifact(4_439, "b1bf058616210d00b126236812396fc990282b377cdffad6d72d8db358039c16"),
    "generation_config.json": Artifact(255, "c70f87dc2995fc43406c0bcfb41b69c6d31c2d0c033fa09e536ffabc091ae24c"),
    "model-00001-of-00001.safetensors": Artifact(6_699_142_782, "a3e8b1629e3cd3974b9c03fd337315c02edbd517fe58300b8e409e2959547704"),
    "model.safetensors.index.json": Artifact(108_762, "a49c77f68883ed9076ec2c8807a667425ad18382395960eb9cd35d4646780d0c"),
    "phosphene_quant_manifest.json": Artifact(1_077, "72524ed82b106356af5ef60b1a14bce313537c11866bf65d697580b7b9e0a68e"),
    "processor_config.json": Artifact(1_382, "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9"),
    "tokenizer.json": Artifact(32_169_626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
    "tokenizer_config.json": Artifact(3_736, "794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8"),
}

BASE_PACK = Pack(
    key="q4_25",
    path=MODEL,
    manifest_asset="q4_25__phosphene_release_manifest.json",
    manifest_sha256="0013bfab7524d61d7c0e3d95733e6453cbab9dde944eb5a9100db8bf531ab424",
    files=BASE_FILES,
)
GEMMA_PACK = Pack(
    key="gemma4_25",
    path=GEMMA_MODEL,
    manifest_asset="gemma4_25__phosphene_release_manifest.json",
    manifest_sha256="b995e479a8e3d2c08af35572fccf9412fad8b3c4f6ef3f2934d733f6596472d5",
    files=GEMMA_FILES,
)
PACKS = (BASE_PACK, GEMMA_PACK)

# Legacy name used by UI readiness code; it intentionally covers the base pack
# only. New code should iterate PACKS so Gemma cannot be accidentally skipped.
MODEL_FILES = BASE_FILES
MODEL_BYTES = BASE_PACK.bytes + GEMMA_PACK.bytes


class VerificationError(RuntimeError):
    """A preflight check failed with an actionable user-facing message."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None, description: str) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise VerificationError(f"Could not {description}: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise VerificationError(f"Could not {description}: {detail or f'exit {completed.returncode}'}")
    return completed.stdout.strip()


def _verify_source(report: Callable[[str], None]) -> None:
    if not RUNTIME.is_dir():
        raise VerificationError(f"LTX runtime is missing: {RUNTIME}")
    if not (RUNTIME / ".git").is_dir():
        raise VerificationError(
            f"LTX runtime is not a public Git source checkout: {RUNTIME}. "
            "Reinstall it with install_ltx25_runtime.py."
        )
    revision = _run(["git", "rev-parse", "HEAD"], cwd=RUNTIME, description="read the LTX source revision")
    if revision != SOURCE_REVISION:
        raise VerificationError(
            f"LTX source revision mismatch: {revision} != {SOURCE_REVISION}. "
            f"Reinstall {SOURCE_REPOSITORY} tag {SOURCE_TAG}."
        )
    tag_revision = _run(
        ["git", "rev-list", "-n", "1", SOURCE_TAG],
        cwd=RUNTIME,
        description=f"resolve pinned source tag {SOURCE_TAG}",
    )
    if tag_revision != SOURCE_REVISION:
        raise VerificationError(f"Pinned tag {SOURCE_TAG} resolves to {tag_revision}, not {SOURCE_REVISION}.")
    remotes = _run(["git", "remote", "-v"], cwd=RUNTIME, description="inspect the LTX source remotes")
    if SOURCE_URL not in remotes.split():
        raise VerificationError(
            f"Pinned source remote is missing: {SOURCE_URL}. Refusing a matching commit with unknown provenance."
        )
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=RUNTIME,
        description="inspect the LTX source checkout",
    )
    source_state = _verify_low_ram_patch_state(status=status)
    if not VENV_PYTHON.is_file() or not CLI.is_file():
        raise VerificationError(f"The LTX virtual environment is incomplete. Expected {VENV_PYTHON} and {CLI}.")
    report(
        f"OK source {SOURCE_REPOSITORY}@{SOURCE_TAG} "
        f"({SOURCE_REVISION}, pinned remote, {source_state})"
    )


def _verify_low_ram_patch_state(*, status: str | None = None, required: bool = False) -> str:
    """Accept only a pristine checkout or the exact two-file edit backport."""
    if status is None:
        status = _run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=RUNTIME,
            description="inspect the LTX low-RAM edit patch",
        )
    if not status:
        if required:
            raise VerificationError(
                "The verified Retake/Extend low-RAM backport is missing from the pinned runtime."
            )
        return "clean"

    staged = _run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=RUNTIME,
        description="inspect staged LTX source changes",
    )
    untracked = _run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=RUNTIME,
        description="inspect untracked LTX source files",
    )
    modified = _run(
        ["git", "diff", "--name-only"],
        cwd=RUNTIME,
        description="inspect modified LTX source files",
    )
    expected = set(LOW_RAM_EDIT_PATCHES)
    actual = set(modified.splitlines()) if modified else set()
    if staged or untracked or actual != expected:
        sample = "\n    ".join(status.splitlines()[:10])
        raise VerificationError(
            "The pinned LTX source checkout has changes outside the verified low-RAM edit backport:\n"
            f"    {sample}"
        )
    for relative, expected_hash in LOW_RAM_EDIT_PATCHES.items():
        path = RUNTIME / relative
        if not path.is_file():
            raise VerificationError(f"Verified low-RAM patch file is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise VerificationError(
                f"Low-RAM patch SHA-256 mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
    return "verified add0142+99c9c7f low-RAM edit backport"


def verify_low_ram_edit_patch(report: Callable[[str], None] = print) -> None:
    """Require the exact Retake/Extend backport before an advanced edit render."""
    state = _verify_low_ram_patch_state(required=True)
    report(f"OK runtime {state}")


def _receipt_expected(pack: Pack) -> dict[str, object]:
    return {
        "schema": "local-video-phosphene-install-receipt/1",
        "release_repository": RELEASE_REPOSITORY,
        "release_tag": RELEASE_TAG,
        "repo_key": pack.key,
        "pack": pack.path.name,
        "manifest_asset": pack.manifest_asset,
        "manifest_sha256": pack.manifest_sha256,
        "files": len(pack.files),
        "bytes": pack.bytes,
    }


def _verify_receipt(pack: Pack, report: Callable[[str], None]) -> None:
    receipt_path = pack.path / RECEIPT_NAME
    if not receipt_path.is_file():
        raise VerificationError(
            f"Missing atomic install receipt: {receipt_path}. Rerun install_phosphene_ltx25_release.py "
            "so the complete pack is verified before use."
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid install receipt {receipt_path}: {exc}") from exc
    expected = _receipt_expected(pack)
    mismatches = [
        f"{field}={receipt.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if receipt.get(field) != value
    ]
    license_exception = receipt.get("license_exception")
    if not isinstance(license_exception, dict) or license_exception.get("installed_sha256") != CURRENT_LICENSE_SHA256:
        mismatches.append("license_exception does not pin the installed current license")
    if mismatches:
        raise VerificationError(f"Install receipt mismatch for {pack.path.name}: " + "; ".join(mismatches))
    report(f"OK {pack.key} receipt ({pack.manifest_sha256})")


def _verify_pack(pack: Pack, *, full_hash: bool, report: Callable[[str], None]) -> None:
    if not pack.path.is_dir():
        raise VerificationError(
            f"Model pack is missing: {pack.path}\n"
            f"Install {pack.key} from {RELEASE_REPOSITORY} release {RELEASE_TAG}."
        )
    _verify_receipt(pack, report)
    failures: list[str] = []
    for relative, artifact in pack.files.items():
        path = pack.path / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != artifact.size:
            failures.append(f"wrong size {relative}: {actual_size:,} != {artifact.size:,} bytes")
    partials = sorted(
        path.relative_to(pack.path).as_posix()
        for path in pack.path.rglob("*")
        if path.is_file()
        and (
            path.name.endswith((".part", ".partial", ".incomplete", ".download", ".assembling"))
            or path.name.startswith(".tmp")
        )
    )
    if partials:
        failures.append(f"incomplete download files remain: {', '.join(partials[:8])}")
    if failures:
        raise VerificationError(
            f"Pinned {pack.key} manifest check failed:\n  - "
            + "\n  - ".join(failures)
            + f"\nExpected {len(pack.files)} files / {pack.bytes:,} bytes from {RELEASE_TAG}."
        )

    # Small metadata hashes are cheap enough to enforce on every launch. Large
    # tensors are size+receipt checked normally and streamed only on --full-hash.
    for relative, artifact in pack.files.items():
        if artifact.size > SMALL_HASH_LIMIT and not full_hash:
            continue
        actual = sha256(pack.path / relative)
        if actual != artifact.sha256:
            raise VerificationError(f"SHA-256 mismatch for {pack.key}/{relative}: {actual} != {artifact.sha256}")
        if full_hash and artifact.size > SMALL_HASH_LIMIT:
            report(f"OK SHA-256 {pack.key}/{relative}")
    report(
        f"OK {pack.key} exact {len(pack.files)}-file manifest "
        f"({pack.bytes / 2**30:.3f} GiB; {'all hashes' if full_hash else 'small hashes + receipt'} verified)"
    )


def _verify_model(
    *,
    full_hash: bool,
    report: Callable[[str], None],
    gemma_model: Path = GEMMA_MODEL,
) -> None:
    selected_gemma = Path(gemma_model).resolve()
    if selected_gemma == GEMMA_MODEL.resolve():
        gemma_identity = "official Phosphene Gemma 4 q4"
    elif selected_gemma == DDALCU_FALLBACK_GEMMA_MODEL.resolve():
        gemma_identity = "explicit revision-pinned ddalcu compatibility fallback"
    else:
        raise VerificationError(
            f"Untrusted Gemma directory {selected_gemma}. Select either the official pack "
            f"({GEMMA_MODEL}) or the explicit validated fallback ({DDALCU_FALLBACK_GEMMA_MODEL})."
        )

    _verify_pack(BASE_PACK, full_hash=full_hash, report=report)
    if selected_gemma == GEMMA_MODEL.resolve():
        _verify_pack(GEMMA_PACK, full_hash=full_hash, report=report)
    else:
        try:
            verify_fallback_install(selected_gemma, full_hash=full_hash, report=report)
        except FallbackImportError as exc:
            raise VerificationError(str(exc)) from exc

    base_config = json.loads((MODEL / "embedded_config.json").read_text(encoding="utf-8"))
    if base_config.get("model_version") != "2.5.0":
        raise VerificationError("Base pack embedded_config.json is not LTX-2.5.0.")
    quantize = json.loads((MODEL / "quantize_config.json").read_text(encoding="utf-8"))
    quantization = quantize.get("quantization", {})
    if quantization != {"bits": 4, "group_size": 64, "only_transformer_blocks": True}:
        raise VerificationError(f"Unexpected base q4 quantization metadata: {quantization!r}")
    gemma_config = json.loads((selected_gemma / "config.json").read_text(encoding="utf-8"))
    gemma_quant = gemma_config.get("quantization", {})
    if gemma_config.get("model_type") != "gemma4_unified" or gemma_config.get("gemma_version") != "gemma4-12b-ltx-v1":
        raise VerificationError("Text pack config is not the LTX-2.5 Gemma 4 tower.")
    if gemma_quant != {"group_size": 64, "bits": 4}:
        raise VerificationError(f"Unexpected Gemma q4 metadata: {gemma_quant!r}")
    report(f"OK LTX-2.5 distilled q4 + {gemma_identity}")


def _verify_tools(*, require_gpu: bool, report: Callable[[str], None]) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise VerificationError(
            f"This launcher requires Apple Silicon macOS; detected {platform.system()} {platform.machine()}."
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VerificationError("ffmpeg is not on PATH. Install ffmpeg before rendering.")
    report(f"OK ffmpeg {ffmpeg}")

    version = _run(
        [str(VENV_PYTHON), "-c", "from importlib.metadata import version; print(version('ltx-pipelines-mlx'))"],
        description="inspect the installed LTX package",
    )
    if version != "0.14.19+ltx25.7":
        raise VerificationError(
            f"Installed ltx-pipelines-mlx version {version!r} != pinned 0.14.19+ltx25.7."
        )
    report(f"OK ltx-pipelines-mlx {version} ({SOURCE_TAG})")

    if not require_gpu:
        report("SKIP Metal execution test (--skip-gpu)")
        return
    sandbox_block = _metal_sandbox_block_reason()
    if sandbox_block:
        raise VerificationError(sandbox_block)
    probe = _run(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "import mlx.core as mx; "
                "x=mx.array([1.0,2.0]); y=x*x; mx.eval(y); "
                "assert y.tolist()==[1.0,4.0]; "
                "print(f'{mx.__version__} device={mx.default_device()} tiny_op={y.tolist()}')"
            ),
        ],
        description=(
            "run a Metal operation. Close memory-heavy apps and run this launcher from the normal macOS "
            "desktop session (Metal is unavailable in some sandboxes/headless sessions)"
        ),
    )
    if "gpu" not in probe.lower():
        raise VerificationError(f"MLX did not select the GPU: {probe}")
    report(f"OK MLX Metal {probe}")


def _metal_sandbox_block_reason() -> str | None:
    """Detect a restricted macOS seatbelt before MLX can abort the process.

    MLX currently assumes Metal device discovery returns at least one device.
    In a restricted Codex shell, opening the required IOKit user client is
    denied and importing :mod:`mlx.core` aborts in native code, which also
    triggers macOS's "Python quit unexpectedly" dialog. Query the same
    permission first so command-line wrappers fail with an actionable Python
    error instead of spawning the crashing Metal probe.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        checker = ctypes.CDLL("/usr/lib/libsandbox.dylib").sandbox_check
        checker.restype = ctypes.c_int
        denied = int(checker(os.getpid(), b"iokit-open-user-client", 0))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if denied == 0:
        return None
    return (
        "macOS blocked Apple GPU/Metal access for this Python process. Run GPU checks and "
        "renders from outputs/local-video/launch_ui.command or a normal Terminal; restricted "
        "Codex shells must use verify_ltx25.py --skip-gpu and must not import mlx.core."
    )


def verify_install(
    *,
    full_hash: bool = False,
    require_gpu: bool = True,
    gemma_model: Path = GEMMA_MODEL,
    report: Callable[[str], None] = print,
) -> None:
    """Raise :class:`VerificationError` unless the installation is render-ready."""
    _verify_source(report)
    _verify_model(full_hash=full_hash, report=report, gemma_model=gemma_model)
    _verify_tools(require_gpu=require_gpu, report=report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the pinned local LTX-2.5 q4 MLX installation.")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="SHA-256 all large weights (reads about 25.6 GiB; receipts, sizes, and small hashes are always checked)",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="skip the live Metal test (for diagnostics only; a render still requires Metal)",
    )
    parser.add_argument(
        "--gemma-pack",
        choices=("official", "ddalcu-fallback"),
        default="official",
        help="official is preferred; the alternate is accepted only with its distinct validated receipt",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gemma_model = GEMMA_MODEL if args.gemma_pack == "official" else DDALCU_FALLBACK_GEMMA_MODEL
    try:
        verify_install(
            full_hash=args.full_hash,
            require_gpu=not args.skip_gpu,
            gemma_model=gemma_model,
        )
    except (VerificationError, json.JSONDecodeError, OSError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print("INSTALLATION VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
