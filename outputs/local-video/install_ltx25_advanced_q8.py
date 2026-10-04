#!/usr/bin/env python3
"""Install and verify selective LTX-2.5 Q8 overlays.

The overlay deliberately downloads only the two Q8 transformers required by
the advanced pipelines plus three tiny Q8-specific metadata files.  Runtime
sidecars that are byte-identical to the already installed Q4 pack are verified
against the pinned Q8 release manifest and exposed through exact relative
symlinks.  The optional 8.9 GB distilled LoRA is intentionally excluded:
forced low-RAM, default-strength two-stage generation swaps to the pre-fused
Q8 distilled transformer instead.

All release manifests are immutable byte/size/SHA-256 pins.  Weight shards use
the resumable, atomic, hash-verifying downloader in
``install_phosphene_ltx25_release.py``.  A custom receipt is published only
after every selected file and shared source has verified.

The default ``full`` scope retains that two-transformer behavior.  The
explicit ``distilled-control`` scope is narrower: it consults only the pinned
Q8 manifest, installs the distilled transformer plus its three metadata files,
and publishes a distinct receipt suitable for Ingredients/Union/Motion.  It
never selects the HQ/dev transformer or any HQ-manifest weight.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, NoReturn, Sequence

import install_phosphene_ltx25_release as release


DEFAULT_MODELS_ROOT = release.DEFAULT_MODELS_ROOT
Q4_DIRECTORY = "ltx-2.5-mlx-q4"
OVERLAY_DIRECTORY = "ltx-2.5-mlx-q8"
RECEIPT_NAME = ".ltx25-q8-overlay-receipt.json"
RECEIPT_SCHEMA = "local-video-ltx25-q8-overlay-receipt/1"
CONTROL_RECEIPT_NAME = ".ltx25-q8-distilled-control-receipt.json"
CONTROL_RECEIPT_SCHEMA = "local-video-ltx25-q8-distilled-control-receipt/1"
FULL_SCOPE = "full"
CONTROL_SCOPE = "distilled-control"
DEFAULT_RESERVE_BYTES = 6 * 2**30

# Immutable trust roots for the two official Phosphene GitHub release
# manifests.  Pinning the manifest bytes prevents a mutable tag or replaced
# release asset from silently redefining any selected weight or shard.
Q8_PIN = release.PackPin(
    key="q8_25",
    label="LTX 2.5 Q8 distilled pack",
    directory=OVERLAY_DIRECTORY,
    manifest_asset="q8_25__phosphene_release_manifest.json",
    manifest_bytes=10_558,
    manifest_sha256="bb93c860e4713d900c9e9408517de4986b22951768b0598e1fbd2833a131399d",
)
HQ_PIN = release.PackPin(
    key="hq_25",
    label="LTX 2.5 Q8 HQ add-on",
    directory=OVERLAY_DIRECTORY,
    manifest_asset="hq_25__phosphene_release_manifest.json",
    manifest_bytes=5_365,
    manifest_sha256="7de6ebce960c2836fbd69c725fc199f067d2b44f7fcc7a660f5e301f5bfbaf89",
)
PINS = (Q8_PIN, HQ_PIN)

Q8_TRANSFORMER = "transformer-distilled.safetensors"
HQ_TRANSFORMER = "transformer-dev.safetensors"
HQ_DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450.safetensors"
Q8_METADATA = (
    "phosphene_quant_manifest.json",
    "quantize_config.json",
    "split_model.json",
)
SHARED_SIDECARS = (
    release.LICENSE_NAME,
    "NOTICE.md",
    "audio_vae.safetensors",
    "config.json",
    "connector.safetensors",
    "duration_head.safetensors",
    "embedded_config.json",
    "manifest.json",
    "spatial_upscaler_x2_v1_1.safetensors",
    "spatial_upscaler_x2_v1_1_config.json",
    "temporal_upscaler_x2_v1_0.safetensors",
    "temporal_upscaler_x2_v1_0_config.json",
    "vae_decoder.safetensors",
    "vae_encoder.safetensors",
    "vocoder.safetensors",
)
EXPECTED_Q8_FILES = frozenset((*SHARED_SIDECARS, *Q8_METADATA, Q8_TRANSFORMER))
EXPECTED_HQ_FILES = frozenset(
    (release.LICENSE_NAME, "NOTICE.md", HQ_DISTILLED_LORA, HQ_TRANSFORMER)
)


class OverlayError(release.InstallerError):
    """The advanced Q8 overlay could not be created or verified safely."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    kind: str
    ready: bool
    detail: str


@dataclass(frozen=True)
class EmbeddedFilePin:
    """Immutable offline identity for one selected or shared overlay file."""

    bytes: int
    sha256: str


# These are copied from the two byte-pinned official manifests above.  They
# make render-time verification wholly offline; validate_overlay_manifests()
# also cross-checks every value whenever the installer fetches the manifests.
PHYSICAL_FILE_PINS: Mapping[str, EmbeddedFilePin] = MappingProxyType(
    {
        "phosphene_quant_manifest.json": EmbeddedFilePin(
            4_550, "b27e4d864aa5180994f0a9e293262fa08548105f6871c1bd600043f9bfd826c7"
        ),
        "quantize_config.json": EmbeddedFilePin(
            101, "e7c7e255c25fd32ecd71364aaa452447d648abdd8c1247b5a181d70b789d837a"
        ),
        "split_model.json": EmbeddedFilePin(
            363, "c83a8892b861a8e2bbcf37103926509ef34dc04b15f8f170c0f04c4962a2f8aa"
        ),
        Q8_TRANSFORMER: EmbeddedFilePin(
            20_595_260_317,
            "36a346c121235d3309450104df8c2de650bce76759b53abc3e9136a08284cdde",
        ),
        HQ_TRANSFORMER: EmbeddedFilePin(
            20_595_256_595,
            "90b7e01dd0411c17476996c76024276bc3daa06ce233b6cd4b2f7703dc64e45b",
        ),
    }
)
CONTROL_PHYSICAL_FILE_PINS: Mapping[str, EmbeddedFilePin] = MappingProxyType(
    {name: PHYSICAL_FILE_PINS[name] for name in (*Q8_METADATA, Q8_TRANSFORMER)}
)
SHARED_FILE_PINS: Mapping[str, EmbeddedFilePin] = MappingProxyType(
    {
        release.LICENSE_NAME: EmbeddedFilePin(
            34_545, "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654"
        ),
        "NOTICE.md": EmbeddedFilePin(
            4_884, "354c701af72ffc11f09bd6d17b8ac59e7021aad3731691e54a8a88daa82bf04e"
        ),
        "audio_vae.safetensors": EmbeddedFilePin(
            106_509_004,
            "481723e829eb184a0a28693b94ff8471a261c0bd183ff61c20032f963367f6f3",
        ),
        "config.json": EmbeddedFilePin(
            2_557, "533a4b8d5ae82218a280976ad202a7a2a6fdffc033e43cc8cd24d83c229a646a"
        ),
        "connector.safetensors": EmbeddedFilePin(
            6_344_495_432,
            "120b43898339895d0342dfc979080b23df55d60ac7179f487f76f895856e6dec",
        ),
        "duration_head.safetensors": EmbeddedFilePin(
            3_808_330,
            "17188b898d7194e1e8687c6aa03cdc5daf5dca25a11cc24557d93961e5351800",
        ),
        "embedded_config.json": EmbeddedFilePin(
            7_589, "78ac0af8c8207407f07024d7b9d4642a796c4226b61ca847b86b24a150d5c886"
        ),
        "manifest.json": EmbeddedFilePin(
            2_164, "90bae3d285cb4af4a998d92c859f869a59f2b95515ec6082ef574967e839370c"
        ),
        "spatial_upscaler_x2_v1_1.safetensors": EmbeddedFilePin(
            995_745_088,
            "0883210010de110994241be1bea97d705b8c6f598f3da51f662b6a15c321dfb0",
        ),
        "spatial_upscaler_x2_v1_1_config.json": EmbeddedFilePin(
            275, "c0646d97bf1bc3fa6e66ec7d17f16ef31b717e64bf86474b7dc3993b6c42346a"
        ),
        "temporal_upscaler_x2_v1_0.safetensors": EmbeddedFilePin(
            261_945_600,
            "84beac95274aace69f576343c89e6815e7d2b45028780a234ad0b4c682e455cc",
        ),
        "temporal_upscaler_x2_v1_0_config.json": EmbeddedFilePin(
            273, "f890af9e95cde921d6e2eb6c334742a81c53c25c3fdc817b4b363622e2c08f3f"
        ),
        "vae_decoder.safetensors": EmbeddedFilePin(
            814_349_496,
            "be257d967de1959e459bf38965c3724d2e6460c1e285711a8e405c3e6d78c75e",
        ),
        "vae_encoder.safetensors": EmbeddedFilePin(
            637_885_234,
            "11f6297ea96731e1263b634e61a3847abbcca3e499c1b5995ca2c7484a2baf0e",
        ),
        "vocoder.safetensors": EmbeddedFilePin(
            258_313_296,
            "fd10ffff704ec6393220e3e1962dee75ea4b91b2075b072ebbfe2d1b2b2f7b99",
        ),
    }
)
EXCLUDED_LORA_PIN = EmbeddedFilePin(
    8_899_889_568,
    "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5",
)
LORA_EXCLUSION_REASON = (
    "not required by forced low-RAM/default-strength A2V; "
    "stage two swaps to the pre-fused Q8 distilled transformer"
)


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> NoReturn:
    raise OverlayError(message)


def _files(manifest: Mapping[str, object]) -> dict[str, dict]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise release.ManifestError("manifest files are not an object")
    if not all(isinstance(name, str) and isinstance(spec, dict) for name, spec in files.items()):
        raise release.ManifestError("manifest contains a malformed file entry")
    return files  # type: ignore[return-value]


def _require_embedded_pin(
    manifest_key: str,
    name: str,
    spec: Mapping[str, object],
    pin: EmbeddedFilePin,
) -> None:
    if spec.get("bytes") != pin.bytes or spec.get("sha256") != pin.sha256:
        raise release.ManifestError(
            f"{manifest_key}/{name}: selected spec disagrees with the embedded offline pin"
        )


def validate_overlay_manifests(
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
) -> tuple[dict, dict]:
    """Strictly validate the two pinned manifests and the selective topology."""
    q8 = release.validate_manifest(q8_manifest, Q8_PIN)
    hq = release.validate_manifest(hq_manifest, HQ_PIN)
    q8_files = _files(q8)
    hq_files = _files(hq)
    if frozenset(q8_files) != EXPECTED_Q8_FILES:
        missing = sorted(EXPECTED_Q8_FILES - frozenset(q8_files))
        extra = sorted(frozenset(q8_files) - EXPECTED_Q8_FILES)
        raise release.ManifestError(
            f"q8_25: unexpected file topology (missing={missing}, extra={extra})"
        )
    if frozenset(hq_files) != EXPECTED_HQ_FILES:
        missing = sorted(EXPECTED_HQ_FILES - frozenset(hq_files))
        extra = sorted(frozenset(hq_files) - EXPECTED_HQ_FILES)
        raise release.ManifestError(
            f"hq_25: unexpected file topology (missing={missing}, extra={extra})"
        )
    for key, manifest in ((Q8_PIN.key, q8), (HQ_PIN.key, hq)):
        quantizer = manifest.get("quantizer")
        if not isinstance(quantizer, dict):
            raise release.ManifestError(f"{key}: missing quantizer identity")
        if quantizer.get("bits") != 8 or quantizer.get("group_size") != 64:
            raise release.ManifestError(f"{key}: expected the pinned Q8/group-size-64 recipe")
    for legal_name in (release.LICENSE_NAME, "NOTICE.md"):
        if q8_files[legal_name] != hq_files[legal_name]:
            raise release.ManifestError(
                f"Q8 and HQ manifests disagree about shared legal file {legal_name}"
            )
    for name in (*Q8_METADATA, Q8_TRANSFORMER):
        _require_embedded_pin(Q8_PIN.key, name, q8_files[name], PHYSICAL_FILE_PINS[name])
    _require_embedded_pin(
        HQ_PIN.key, HQ_TRANSFORMER, hq_files[HQ_TRANSFORMER], PHYSICAL_FILE_PINS[HQ_TRANSFORMER]
    )
    _require_embedded_pin(
        HQ_PIN.key, HQ_DISTILLED_LORA, hq_files[HQ_DISTILLED_LORA], EXCLUDED_LORA_PIN
    )
    for name in SHARED_SIDECARS:
        if name == release.LICENSE_NAME:
            continue
        _require_embedded_pin(Q8_PIN.key, name, q8_files[name], SHARED_FILE_PINS[name])
    return q8, hq


def validate_control_manifest(q8_manifest: Mapping[str, object]) -> dict:
    """Validate the one manifest trusted by the distilled-control scope."""
    q8 = release.validate_manifest(q8_manifest, Q8_PIN)
    q8_files = _files(q8)
    if frozenset(q8_files) != EXPECTED_Q8_FILES:
        missing = sorted(EXPECTED_Q8_FILES - frozenset(q8_files))
        extra = sorted(frozenset(q8_files) - EXPECTED_Q8_FILES)
        raise release.ManifestError(
            f"q8_25: unexpected file topology (missing={missing}, extra={extra})"
        )
    quantizer = q8.get("quantizer")
    if not isinstance(quantizer, dict):
        raise release.ManifestError("q8_25: missing quantizer identity")
    if quantizer.get("bits") != 8 or quantizer.get("group_size") != 64:
        raise release.ManifestError("q8_25: expected the pinned Q8/group-size-64 recipe")
    for name in (*Q8_METADATA, Q8_TRANSFORMER):
        _require_embedded_pin(Q8_PIN.key, name, q8_files[name], PHYSICAL_FILE_PINS[name])
    for name in SHARED_SIDECARS:
        if name == release.LICENSE_NAME:
            continue
        _require_embedded_pin(Q8_PIN.key, name, q8_files[name], SHARED_FILE_PINS[name])
    return q8


def physical_specs(
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
) -> tuple[tuple[release.PackPin, str, dict], ...]:
    """Files intentionally stored as physical bytes in the overlay."""
    q8_files = _files(q8_manifest)
    hq_files = _files(hq_manifest)
    return (
        *((Q8_PIN, name, q8_files[name]) for name in Q8_METADATA),
        (Q8_PIN, Q8_TRANSFORMER, q8_files[Q8_TRANSFORMER]),
        (HQ_PIN, HQ_TRANSFORMER, hq_files[HQ_TRANSFORMER]),
    )


def control_physical_specs(
    q8_manifest: Mapping[str, object],
) -> tuple[tuple[release.PackPin, str, dict], ...]:
    """Physical files selected by the control-only scope (never HQ/dev)."""
    q8_files = _files(q8_manifest)
    return (
        *((Q8_PIN, name, q8_files[name]) for name in Q8_METADATA),
        (Q8_PIN, Q8_TRANSFORMER, q8_files[Q8_TRANSFORMER]),
    )


def shared_specs(q8_manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Specs used to verify each Q4 source behind a shared relative link."""
    q8_files = _files(q8_manifest)
    specs: dict[str, Mapping[str, object]] = {
        name: q8_files[name] for name in SHARED_SIDECARS
    }
    # The publisher replaced this legal-text release asset after publishing the
    # manifests.  Reuse the separately pinned current license already installed
    # in Q4, matching the exception in the base installer.
    specs[release.LICENSE_NAME] = release.CURRENT_LICENSE_SPEC
    return specs


def selected_physical_bytes(
    q8_manifest: Mapping[str, object], hq_manifest: Mapping[str, object]
) -> int:
    return sum(int(spec["bytes"]) for _, _, spec in physical_specs(q8_manifest, hq_manifest))


def selected_control_physical_bytes(q8_manifest: Mapping[str, object]) -> int:
    return sum(int(spec["bytes"]) for _, _, spec in control_physical_specs(q8_manifest))


def _expected_link_text(name: str) -> str:
    return str(Path("..") / Q4_DIRECTORY / name)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _check_pack_directory(path: Path, *, label: str, required: bool) -> None:
    if path.is_symlink():
        fail(f"refusing symlinked {label}: {path}")
    if not path.exists():
        if required:
            fail(f"missing {label}: {path}")
        return
    if not path.is_dir():
        fail(f"refusing non-directory {label}: {path}")


def verify_shared_source(
    models_root: Path,
    name: str,
    spec: Mapping[str, object],
) -> tuple[bool, str]:
    source_dir = models_root / Q4_DIRECTORY
    if source_dir.is_symlink():
        return False, f"unsafe symlinked Q4 source directory: {source_dir}"
    if not source_dir.is_dir():
        return False, f"missing Q4 source directory: {source_dir}"
    return release.verify_local_file(source_dir / name, spec)


def verify_shared_link(
    models_root: Path,
    name: str,
    spec: Mapping[str, object],
) -> tuple[bool, str]:
    """Verify source bytes and the exact relative link without trusting resolve()."""
    source_ready, source_detail = verify_shared_source(models_root, name, spec)
    if not source_ready:
        return False, f"Q4 source invalid: {source_detail}"
    link_ready, link_detail = _verify_link_shape(models_root, name)
    if not link_ready:
        return False, link_detail
    return True, f"{link_detail}; source size and SHA-256 verified"


def _verify_link_shape(models_root: Path, name: str) -> tuple[bool, str]:
    """Verify only the destination link after its source was already hashed."""
    overlay_dir = models_root / OVERLAY_DIRECTORY
    if overlay_dir.is_symlink():
        return False, f"unsafe symlinked overlay directory: {overlay_dir}"
    target = overlay_dir / name
    if not target.is_symlink():
        if _lexists(target):
            return False, "expected a relative symlink, found another file type"
        return False, "relative symlink missing (Q4 source is verified)"
    try:
        link_text = os.readlink(target)
    except OSError as exc:
        return False, f"cannot read symlink: {exc}"
    expected = _expected_link_text(name)
    if link_text != expected or os.path.isabs(link_text):
        return False, f"unsafe link target {link_text!r}; expected {expected!r}"
    # The exact link spelling above is the primary boundary.  This second check
    # catches a moved/replaced directory between link creation and verification.
    try:
        resolved = target.resolve(strict=True)
        expected_resolved = (models_root / Q4_DIRECTORY / name).resolve(strict=True)
    except OSError as exc:
        return False, f"link target cannot be resolved: {exc}"
    if resolved != expected_resolved:
        return False, "link resolves outside the verified Q4 source"
    return True, f"relative link -> {expected}"


def install_shared_link(
    models_root: Path,
    name: str,
    spec: Mapping[str, object],
    *,
    source_preverified: bool = False,
) -> bool:
    """Create one exact relative link; never overwrite an unexpected entry."""
    if not source_preverified:
        source_ready, source_detail = verify_shared_source(models_root, name, spec)
        if not source_ready:
            fail(f"cannot link {name}: Q4 source is not verified ({source_detail})")
    overlay_dir = models_root / OVERLAY_DIRECTORY
    _check_pack_directory(overlay_dir, label="Q8 overlay directory", required=False)
    overlay_dir.mkdir(mode=0o755, parents=False, exist_ok=True)
    target = overlay_dir / name
    ready, _ = _verify_link_shape(models_root, name)
    if ready:
        return False
    if _lexists(target):
        fail(f"refusing to replace unexpected overlay entry: {target}")
    try:
        os.symlink(_expected_link_text(name), target)
    except FileExistsError as exc:
        raise OverlayError(f"overlay entry appeared concurrently: {target}") from exc
    release._fsync_directory(overlay_dir)
    ready, detail = _verify_link_shape(models_root, name)
    if not ready:
        try:
            target.unlink()
        except OSError:
            pass
        fail(f"new shared link failed verification: {name} ({detail})")
    return True


def verify_receipt(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
) -> tuple[bool, str]:
    expected = receipt_payload(q8_manifest, hq_manifest, installed_utc=None)
    return _verify_receipt_against(models_root, expected)


def verify_control_receipt(
    models_root: Path,
    q8_manifest: Mapping[str, object],
) -> tuple[bool, str]:
    expected = control_receipt_payload(q8_manifest, installed_utc=None)
    return _verify_receipt_against(
        models_root, expected, receipt_name=CONTROL_RECEIPT_NAME
    )


def _verify_receipt_against(
    models_root: Path,
    expected_payload: Mapping[str, object],
    *,
    receipt_name: str = RECEIPT_NAME,
) -> tuple[bool, str]:
    overlay_dir = models_root / OVERLAY_DIRECTORY
    if overlay_dir.is_symlink():
        return False, "unsafe symlinked overlay directory"
    if overlay_dir.exists() and not overlay_dir.is_dir():
        return False, "overlay path is not a directory"
    path = overlay_dir / receipt_name
    if path.is_symlink():
        return False, "unsafe symlinked receipt"
    if not path.is_file():
        return False, "missing"
    try:
        if path.stat().st_size > 256 << 10:
            return False, "receipt exceeds safety limit"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON ({exc})"
    if not isinstance(payload, dict):
        return False, "receipt root is not an object"
    timestamp = payload.pop("installed_utc", None)
    if not isinstance(timestamp, str):
        return False, "missing installed_utc"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return False, "invalid installed_utc"
    if parsed.tzinfo is None:
        return False, "installed_utc has no timezone"
    expected = dict(expected_payload)
    expected.pop("installed_utc", None)
    if payload != expected:
        return False, "content does not match the pinned overlay selection"
    return True, "pinned manifest identities and overlay selection verified"


def verify_receipt_offline(models_root: Path) -> tuple[bool, str]:
    """Verify the overlay receipt using embedded pins and no network state."""
    return _verify_receipt_against(models_root, embedded_receipt_payload())


def verify_control_receipt_offline(models_root: Path) -> tuple[bool, str]:
    """Verify the distinct distilled-control receipt without network state."""
    return _verify_receipt_against(
        models_root,
        embedded_control_receipt_payload(),
        receipt_name=CONTROL_RECEIPT_NAME,
    )


def receipt_payload(
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
    *,
    installed_utc: str | None = None,
) -> dict[str, object]:
    physical: dict[str, object] = {}
    for pin, name, spec in physical_specs(q8_manifest, hq_manifest):
        physical[name] = {
            "manifest": pin.key,
            "bytes": int(spec["bytes"]),
            "sha256": str(spec["sha256"]),
        }
    links = {
        name: {
            "target": _expected_link_text(name),
            "bytes": int(spec["bytes"]),
            "sha256": str(spec["sha256"]),
        }
        for name, spec in shared_specs(q8_manifest).items()
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "installed_utc": installed_utc or datetime.now(timezone.utc).isoformat(),
        "release_repository": release.RELEASE_REPOSITORY,
        "release_tag": release.RELEASE_TAG,
        "overlay_directory": OVERLAY_DIRECTORY,
        "base_directory": Q4_DIRECTORY,
        "manifests": {
            Q8_PIN.key: {
                "asset": Q8_PIN.manifest_asset,
                "bytes": Q8_PIN.manifest_bytes,
                "sha256": Q8_PIN.manifest_sha256,
            },
            HQ_PIN.key: {
                "asset": HQ_PIN.manifest_asset,
                "bytes": HQ_PIN.manifest_bytes,
                "sha256": HQ_PIN.manifest_sha256,
            },
        },
        "physical_files": physical,
        "shared_relative_links": links,
        "physical_bytes": selected_physical_bytes(q8_manifest, hq_manifest),
        "excluded_optional_files": {
            HQ_DISTILLED_LORA: {
                "bytes": int(_files(hq_manifest)[HQ_DISTILLED_LORA]["bytes"]),
                "sha256": str(_files(hq_manifest)[HQ_DISTILLED_LORA]["sha256"]),
                "reason": LORA_EXCLUSION_REASON,
            }
        },
        "license_exception": {
            "reason": "release license asset was replaced after manifest publication",
            "stale_manifest_sha256": release.STALE_LICENSE_SPEC["sha256"],
            "installed_sha256": release.CURRENT_LICENSE_PIN,
        },
    }


def control_receipt_payload(
    q8_manifest: Mapping[str, object],
    *,
    installed_utc: str | None = None,
) -> dict[str, object]:
    """Build the truthful receipt for the distilled control-only selection."""
    q8 = validate_control_manifest(q8_manifest)
    physical = {
        name: {
            "manifest": Q8_PIN.key,
            "bytes": int(spec["bytes"]),
            "sha256": str(spec["sha256"]),
        }
        for _, name, spec in control_physical_specs(q8)
    }
    links = {
        name: {
            "target": _expected_link_text(name),
            "bytes": int(spec["bytes"]),
            "sha256": str(spec["sha256"]),
        }
        for name, spec in shared_specs(q8).items()
    }
    return {
        "schema": CONTROL_RECEIPT_SCHEMA,
        "scope": CONTROL_SCOPE,
        "installed_utc": installed_utc or datetime.now(timezone.utc).isoformat(),
        "release_repository": release.RELEASE_REPOSITORY,
        "release_tag": release.RELEASE_TAG,
        "overlay_directory": OVERLAY_DIRECTORY,
        "base_directory": Q4_DIRECTORY,
        "manifests": {
            Q8_PIN.key: {
                "asset": Q8_PIN.manifest_asset,
                "bytes": Q8_PIN.manifest_bytes,
                "sha256": Q8_PIN.manifest_sha256,
            }
        },
        "physical_files": physical,
        "shared_relative_links": links,
        "physical_bytes": selected_control_physical_bytes(q8),
        "enabled_workflows": ["ingredients", "union", "motion"],
        "not_selected_files": {
            HQ_TRANSFORMER: {
                "bytes": PHYSICAL_FILE_PINS[HQ_TRANSFORMER].bytes,
                "sha256": PHYSICAL_FILE_PINS[HQ_TRANSFORMER].sha256,
                "reason": "dev/HQ transformer is outside the distilled-control scope",
            },
            HQ_DISTILLED_LORA: {
                "bytes": EXCLUDED_LORA_PIN.bytes,
                "sha256": EXCLUDED_LORA_PIN.sha256,
                "reason": "HQ add-on is outside the distilled-control scope",
            },
        },
        "unsupported_workflows": ["retake", "extend", "audio-to-video"],
        "license_exception": {
            "reason": "release license asset was replaced after manifest publication",
            "stale_manifest_sha256": release.STALE_LICENSE_SPEC["sha256"],
            "installed_sha256": release.CURRENT_LICENSE_PIN,
        },
    }


def embedded_receipt_payload(*, installed_utc: str | None = None) -> dict[str, object]:
    """Return the expected receipt solely from immutable in-module pins."""
    physical_sources = {
        **{name: Q8_PIN.key for name in Q8_METADATA},
        Q8_TRANSFORMER: Q8_PIN.key,
        HQ_TRANSFORMER: HQ_PIN.key,
    }
    physical = {
        name: {
            "manifest": physical_sources[name],
            "bytes": pin.bytes,
            "sha256": pin.sha256,
        }
        for name, pin in PHYSICAL_FILE_PINS.items()
    }
    links = {
        name: {
            "target": _expected_link_text(name),
            "bytes": pin.bytes,
            "sha256": pin.sha256,
        }
        for name, pin in SHARED_FILE_PINS.items()
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "installed_utc": installed_utc or datetime.now(timezone.utc).isoformat(),
        "release_repository": release.RELEASE_REPOSITORY,
        "release_tag": release.RELEASE_TAG,
        "overlay_directory": OVERLAY_DIRECTORY,
        "base_directory": Q4_DIRECTORY,
        "manifests": {
            Q8_PIN.key: {
                "asset": Q8_PIN.manifest_asset,
                "bytes": Q8_PIN.manifest_bytes,
                "sha256": Q8_PIN.manifest_sha256,
            },
            HQ_PIN.key: {
                "asset": HQ_PIN.manifest_asset,
                "bytes": HQ_PIN.manifest_bytes,
                "sha256": HQ_PIN.manifest_sha256,
            },
        },
        "physical_files": physical,
        "shared_relative_links": links,
        "physical_bytes": sum(pin.bytes for pin in PHYSICAL_FILE_PINS.values()),
        "excluded_optional_files": {
            HQ_DISTILLED_LORA: {
                "bytes": EXCLUDED_LORA_PIN.bytes,
                "sha256": EXCLUDED_LORA_PIN.sha256,
                "reason": LORA_EXCLUSION_REASON,
            }
        },
        "license_exception": {
            "reason": "release license asset was replaced after manifest publication",
            "stale_manifest_sha256": release.STALE_LICENSE_SPEC["sha256"],
            "installed_sha256": release.CURRENT_LICENSE_PIN,
        },
    }


def embedded_control_receipt_payload(
    *, installed_utc: str | None = None
) -> dict[str, object]:
    """Expected distilled-control receipt derived only from immutable pins."""
    physical = {
        name: {
            "manifest": Q8_PIN.key,
            "bytes": pin.bytes,
            "sha256": pin.sha256,
        }
        for name, pin in CONTROL_PHYSICAL_FILE_PINS.items()
    }
    links = {
        name: {
            "target": _expected_link_text(name),
            "bytes": pin.bytes,
            "sha256": pin.sha256,
        }
        for name, pin in SHARED_FILE_PINS.items()
    }
    return {
        "schema": CONTROL_RECEIPT_SCHEMA,
        "scope": CONTROL_SCOPE,
        "installed_utc": installed_utc or datetime.now(timezone.utc).isoformat(),
        "release_repository": release.RELEASE_REPOSITORY,
        "release_tag": release.RELEASE_TAG,
        "overlay_directory": OVERLAY_DIRECTORY,
        "base_directory": Q4_DIRECTORY,
        "manifests": {
            Q8_PIN.key: {
                "asset": Q8_PIN.manifest_asset,
                "bytes": Q8_PIN.manifest_bytes,
                "sha256": Q8_PIN.manifest_sha256,
            }
        },
        "physical_files": physical,
        "shared_relative_links": links,
        "physical_bytes": sum(pin.bytes for pin in CONTROL_PHYSICAL_FILE_PINS.values()),
        "enabled_workflows": ["ingredients", "union", "motion"],
        "not_selected_files": {
            HQ_TRANSFORMER: {
                "bytes": PHYSICAL_FILE_PINS[HQ_TRANSFORMER].bytes,
                "sha256": PHYSICAL_FILE_PINS[HQ_TRANSFORMER].sha256,
                "reason": "dev/HQ transformer is outside the distilled-control scope",
            },
            HQ_DISTILLED_LORA: {
                "bytes": EXCLUDED_LORA_PIN.bytes,
                "sha256": EXCLUDED_LORA_PIN.sha256,
                "reason": "HQ add-on is outside the distilled-control scope",
            },
        },
        "unsupported_workflows": ["retake", "extend", "audio-to-video"],
        "license_exception": {
            "reason": "release license asset was replaced after manifest publication",
            "stale_manifest_sha256": release.STALE_LICENSE_SPEC["sha256"],
            "installed_sha256": release.CURRENT_LICENSE_PIN,
        },
    }


def _sha256_regular(path: Path) -> tuple[str | None, str | None]:
    """Hash one non-symlink regular file while holding its descriptor."""
    import hashlib

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        return None, str(exc)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None, "not a regular file"
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(release.CHUNK_BYTES), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            return None, "file changed while hashing"
        return digest.hexdigest(), None
    finally:
        os.close(descriptor)


def _validate_safetensors_structure(
    path: Path, expected_bytes: int
) -> tuple[bool, str]:
    """Validate a safetensors index without reading its multi-GB data region."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        return False, f"cannot open safely ({exc})"
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return False, "not a regular file"
        if info.st_size != expected_bytes:
            return False, f"size mismatch ({info.st_size} != {expected_bytes})"
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return False, "missing 8-byte safetensors header length"
            header_bytes = int.from_bytes(raw_length, "little")
            if not 2 <= header_bytes <= 64 << 20:
                return False, f"unsafe safetensors header length {header_bytes}"
            raw_header = handle.read(header_bytes)
            if len(raw_header) != header_bytes:
                return False, "truncated safetensors JSON header"
        try:
            header = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return False, f"invalid safetensors JSON header ({exc})"
        if not isinstance(header, dict):
            return False, "safetensors header is not an object"
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if not tensors:
            return False, "safetensors header contains no tensors"
        data_bytes = expected_bytes - 8 - header_bytes
        if data_bytes < 0:
            return False, "safetensors header exceeds file size"
        spans: list[tuple[int, int, str]] = []
        for name, tensor in tensors.items():
            if not isinstance(name, str) or not isinstance(tensor, dict):
                return False, "invalid safetensors tensor entry"
            dtype = tensor.get("dtype")
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            valid_shape = isinstance(shape, list) and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in shape
            )
            valid_offsets = (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            )
            if (
                not isinstance(dtype, str)
                or not dtype
                or not valid_shape
                or not valid_offsets
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes
            ):
                return False, f"invalid safetensors tensor metadata for {name!r}"
            spans.append((offsets[0], offsets[1], name))
        cursor = 0
        for start, end, name in sorted(spans):
            if start != cursor:
                return False, f"non-contiguous safetensors data before {name!r}"
            cursor = end
        if cursor != data_bytes:
            return False, "safetensors data region is not fully indexed"
        after = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            return False, "file changed while validating safetensors structure"
        return True, "exact size and safetensors structure verified"
    finally:
        os.close(descriptor)


def _verify_embedded_file(
    path: Path,
    pin: EmbeddedFilePin,
    *,
    hash_required: bool,
) -> tuple[bool, str]:
    if path.is_symlink():
        return False, "unsafe symlink where a regular source file is required"
    if not path.is_file():
        return False, "missing or not a regular file"
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat ({exc})"
    if actual_size != pin.bytes:
        return False, f"size mismatch ({actual_size} != {pin.bytes})"
    detail = "exact size verified"
    if path.name.endswith(".safetensors"):
        ready, detail = _validate_safetensors_structure(path, pin.bytes)
        if not ready:
            return False, detail
    if hash_required:
        digest, error = _sha256_regular(path)
        if error:
            return False, f"could not hash safely ({error})"
        if digest != pin.sha256:
            return False, f"SHA-256 mismatch ({digest} != {pin.sha256})"
        detail += "; SHA-256 verified"
    return True, detail


def verify_installed_overlay(
    models_root: Path = DEFAULT_MODELS_ROOT,
    *,
    full_hash: bool = False,
) -> tuple[CheckResult, ...]:
    """Offline render gate for the installed selective Q8 overlay.

    Quick mode verifies the immutable receipt, exact physical sizes, every
    safetensors header/index, hashes all small metadata/legal files, and checks
    that every shared sidecar is the exact relative link to the Q4 pack.
    ``full_hash=True`` additionally hashes both 20.6 GB transformers.  The
    function performs no network calls and raises :class:`OverlayError` on the
    first failed invariant; successful checks are returned for diagnostics.
    """
    if not isinstance(full_hash, bool):
        fail("full_hash must be a boolean")
    root = Path(models_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        fail(f"unsafe or missing models root: {root}")
    q4_dir = root / Q4_DIRECTORY
    overlay_dir = root / OVERLAY_DIRECTORY
    _check_pack_directory(q4_dir, label="Q4 source directory", required=True)
    _check_pack_directory(overlay_dir, label="Q8 overlay directory", required=True)

    results: list[CheckResult] = []
    receipt_ready, receipt_detail = verify_receipt_offline(root)
    if not receipt_ready:
        fail(f"offline overlay receipt verification failed: {receipt_detail}")
    results.append(CheckResult(RECEIPT_NAME, "receipt", True, receipt_detail))

    for name, pin in PHYSICAL_FILE_PINS.items():
        is_transformer = name in {Q8_TRANSFORMER, HQ_TRANSFORMER}
        ready, detail = _verify_embedded_file(
            overlay_dir / name,
            pin,
            hash_required=(not is_transformer or full_hash),
        )
        if not ready:
            fail(f"offline physical-file verification failed for {name}: {detail}")
        results.append(CheckResult(name, "physical", True, detail))

    for name, pin in SHARED_FILE_PINS.items():
        link_ready, link_detail = _verify_link_shape(root, name)
        if not link_ready:
            fail(f"offline shared-link verification failed for {name}: {link_detail}")
        source = q4_dir / name
        ready, detail = _verify_embedded_file(
            source,
            pin,
            hash_required=not name.endswith(".safetensors"),
        )
        if not ready:
            fail(f"offline Q4 source verification failed for {name}: {detail}")
        results.append(
            CheckResult(name, "shared-link", True, f"{link_detail}; {detail}")
        )
    return tuple(results)


def verify_installed_control_overlay(
    models_root: Path = DEFAULT_MODELS_ROOT,
    *,
    full_hash: bool = False,
) -> tuple[CheckResult, ...]:
    """Offline gate for Ingredients/Union/Motion without requiring HQ/dev.

    A distilled-control receipt is preferred.  A valid legacy/full receipt is
    also accepted so an existing complete overlay remains usable.  In either
    case only the four Q8 physical files and shared Q4 links are required.
    """
    if not isinstance(full_hash, bool):
        fail("full_hash must be a boolean")
    root = Path(models_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        fail(f"unsafe or missing models root: {root}")
    q4_dir = root / Q4_DIRECTORY
    overlay_dir = root / OVERLAY_DIRECTORY
    _check_pack_directory(q4_dir, label="Q4 source directory", required=True)
    _check_pack_directory(overlay_dir, label="Q8 overlay directory", required=True)

    control_receipt = overlay_dir / CONTROL_RECEIPT_NAME
    if _lexists(control_receipt):
        receipt_ready, receipt_detail = verify_control_receipt_offline(root)
        receipt_name = CONTROL_RECEIPT_NAME
    else:
        receipt_ready, receipt_detail = verify_receipt_offline(root)
        receipt_name = RECEIPT_NAME
    if not receipt_ready:
        fail(f"offline control receipt verification failed: {receipt_detail}")
    results: list[CheckResult] = [
        CheckResult(receipt_name, "receipt", True, receipt_detail)
    ]

    for name, pin in CONTROL_PHYSICAL_FILE_PINS.items():
        ready, detail = _verify_embedded_file(
            overlay_dir / name,
            pin,
            hash_required=(name != Q8_TRANSFORMER or full_hash),
        )
        if not ready:
            fail(f"offline control-file verification failed for {name}: {detail}")
        results.append(CheckResult(name, "physical", True, detail))

    for name, pin in SHARED_FILE_PINS.items():
        link_ready, link_detail = _verify_link_shape(root, name)
        if not link_ready:
            fail(f"offline shared-link verification failed for {name}: {link_detail}")
        ready, detail = _verify_embedded_file(
            q4_dir / name,
            pin,
            hash_required=not name.endswith(".safetensors"),
        )
        if not ready:
            fail(f"offline Q4 source verification failed for {name}: {detail}")
        results.append(
            CheckResult(name, "shared-link", True, f"{link_detail}; {detail}")
        )
    return tuple(results)


def overlay_status(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
) -> tuple[list[CheckResult], int]:
    results: list[CheckResult] = []
    missing_physical = 0
    overlay_dir = models_root / OVERLAY_DIRECTORY
    unsafe_overlay = overlay_dir.is_symlink() or (
        overlay_dir.exists() and not overlay_dir.is_dir()
    )
    for _, name, spec in physical_specs(q8_manifest, hq_manifest):
        if unsafe_overlay:
            ready, detail = False, "unsafe overlay directory"
        else:
            ready, detail = release.verify_local_file(overlay_dir / name, spec)
        results.append(CheckResult(name, "physical", ready, detail))
        if not ready:
            missing_physical += int(spec["bytes"])
    for name, spec in shared_specs(q8_manifest).items():
        ready, detail = verify_shared_link(models_root, name, spec)
        results.append(CheckResult(name, "shared-link", ready, detail))
    ready, detail = verify_receipt(models_root, q8_manifest, hq_manifest)
    results.append(CheckResult(RECEIPT_NAME, "receipt", ready, detail))
    return results, missing_physical


def control_overlay_status(
    models_root: Path,
    q8_manifest: Mapping[str, object],
) -> tuple[list[CheckResult], int]:
    """Return status for only the distilled-control selection."""
    results: list[CheckResult] = []
    missing_physical = 0
    overlay_dir = models_root / OVERLAY_DIRECTORY
    unsafe_overlay = overlay_dir.is_symlink() or (
        overlay_dir.exists() and not overlay_dir.is_dir()
    )
    for _, name, spec in control_physical_specs(q8_manifest):
        if unsafe_overlay:
            ready, detail = False, "unsafe overlay directory"
        else:
            ready, detail = release.verify_local_file(overlay_dir / name, spec)
        results.append(CheckResult(name, "physical", ready, detail))
        if not ready:
            missing_physical += int(spec["bytes"])
    for name, spec in shared_specs(q8_manifest).items():
        ready, detail = verify_shared_link(models_root, name, spec)
        results.append(CheckResult(name, "shared-link", ready, detail))
    ready, detail = verify_control_receipt(models_root, q8_manifest)
    results.append(CheckResult(CONTROL_RECEIPT_NAME, "receipt", ready, detail))
    return results, missing_physical


def verified_staged_shard_bytes(
    models_root: Path,
    pin: release.PackPin,
    name: str,
    spec: Mapping[str, object],
) -> int:
    """Count complete imported shards already occupying the models volume.

    Credit is deliberately narrow: the private path must be a non-writable
    regular file of the exact pinned size, its sidecar must bind it to this
    manifest and shard, and the importer must have journaled the SHA-256 it
    verified twice.  install_file still hashes the shard again before use.
    """
    credited = 0
    shards = spec["shards"]
    assert isinstance(shards, list)
    for shard in shards:
        assert isinstance(shard, dict)
        shard_path, metadata_path = release._stage_paths(
            models_root,
            pin,
            name,
            str(shard["asset"]),
        )
        try:
            shard_info = shard_path.lstat()
            metadata_info = metadata_path.lstat()
        except OSError:
            continue
        if (
            stat.S_ISLNK(shard_info.st_mode)
            or not stat.S_ISREG(shard_info.st_mode)
            or stat.S_ISLNK(metadata_info.st_mode)
            or not stat.S_ISREG(metadata_info.st_mode)
        ):
            continue
        expected_bytes = int(shard["bytes"])
        expected_sha256 = str(shard["sha256"])
        if shard_info.st_size != expected_bytes:
            continue
        # Imported clones are mode 0400. A writable file is no longer the
        # immutable object covered by the importer's verification journal.
        if stat.S_IMODE(shard_info.st_mode) & 0o222:
            continue
        metadata = release._read_json(metadata_path)
        identity = release._shard_identity(pin, pin.manifest_sha256, shard)
        if not release._resume_metadata_matches(metadata, identity):
            continue
        if metadata.get(release.VERIFIED_SHARD_SHA256_FIELD) != expected_sha256:
            continue
        credited += expected_bytes
    return credited


def verified_selected_staged_bytes(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
) -> int:
    """Return disk bytes already held by exact selected overlay shards."""
    overlay_dir = models_root / OVERLAY_DIRECTORY
    credited = 0
    for pin, name, spec in physical_specs(q8_manifest, hq_manifest):
        ready, _ = release.verify_local_file(overlay_dir / name, spec)
        if ready:
            continue
        credited += verified_staged_shard_bytes(models_root, pin, name, spec)
    return credited


def verified_control_staged_bytes(
    models_root: Path,
    q8_manifest: Mapping[str, object],
) -> int:
    """Return staged bytes credited to the distilled-control selection."""
    overlay_dir = models_root / OVERLAY_DIRECTORY
    credited = 0
    for pin, name, spec in control_physical_specs(q8_manifest):
        ready, _ = release.verify_local_file(overlay_dir / name, spec)
        if ready:
            continue
        credited += verified_staged_shard_bytes(models_root, pin, name, spec)
    return credited


def estimate_required_space(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
    *,
    reserve_bytes: int,
    range_chunk_bytes: int,
    range_workers: int,
) -> int:
    overlay_dir = models_root / OVERLAY_DIRECTORY
    if overlay_dir.is_symlink() or (overlay_dir.exists() and not overlay_dir.is_dir()):
        fail(f"unsafe Q8 overlay directory: {overlay_dir}")
    missing = 0
    largest_shard = 0
    for pin, name, spec in physical_specs(q8_manifest, hq_manifest):
        ready, _ = release.verify_local_file(overlay_dir / name, spec)
        if ready:
            continue
        staged = verified_staged_shard_bytes(models_root, pin, name, spec)
        missing += max(0, int(spec["bytes"]) - staged)
        shards = spec["shards"]
        assert isinstance(shards, list)
        largest_shard = max(largest_shard, *(int(shard["bytes"]) for shard in shards))
    range_transient = min(range_chunk_bytes, largest_shard) if range_chunk_bytes else 0
    return missing + largest_shard + range_workers * range_transient + reserve_bytes


def estimate_control_required_space(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    *,
    reserve_bytes: int,
    range_chunk_bytes: int,
    range_workers: int,
) -> int:
    """Conservative space requirement for Q8 distilled-control only."""
    overlay_dir = models_root / OVERLAY_DIRECTORY
    if overlay_dir.is_symlink() or (overlay_dir.exists() and not overlay_dir.is_dir()):
        fail(f"unsafe Q8 overlay directory: {overlay_dir}")
    missing = 0
    largest_shard = 0
    for pin, name, spec in control_physical_specs(q8_manifest):
        ready, _ = release.verify_local_file(overlay_dir / name, spec)
        if ready:
            continue
        staged = verified_staged_shard_bytes(models_root, pin, name, spec)
        missing += max(0, int(spec["bytes"]) - staged)
        shards = spec["shards"]
        assert isinstance(shards, list)
        largest_shard = max(largest_shard, *(int(shard["bytes"]) for shard in shards))
    range_transient = min(range_chunk_bytes, largest_shard) if range_chunk_bytes else 0
    return missing + largest_shard + range_workers * range_transient + reserve_bytes


@contextlib.contextmanager
def _overlay_lock(models_root: Path):
    lock_path = models_root / ".ltx25-q8-overlay.install.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise OverlayError(f"cannot open overlay install lock {lock_path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            fail(f"refusing non-regular overlay install lock: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OverlayError("another advanced Q8 overlay operation is running") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def install_overlay(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    hq_manifest: Mapping[str, object],
    manifest_digests: Mapping[str, str],
    *,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int,
    range_workers: int,
    report: Callable[[str], None] = log,
) -> None:
    """Install the exact selected files, links, and verified receipt."""
    q8_manifest, hq_manifest = validate_overlay_manifests(q8_manifest, hq_manifest)
    specs = shared_specs(q8_manifest)
    # Validate every shared source before the first weight request.  A partial
    # or incompatible Q4 base therefore cannot trigger a 41 GB transfer.
    for name, spec in specs.items():
        ready, detail = verify_shared_source(models_root, name, spec)
        if not ready:
            fail(f"Q4 prerequisite failed for {name}: {detail}")

    with _overlay_lock(models_root):
        overlay_dir = models_root / OVERLAY_DIRECTORY
        _check_pack_directory(overlay_dir, label="Q8 overlay directory", required=False)
        overlay_dir.mkdir(mode=0o755, parents=False, exist_ok=True)
        for name, spec in specs.items():
            changed = install_shared_link(
                models_root, name, spec, source_preverified=True
            )
            report(
                f"[link] {name}: "
                f"{'created and verified' if changed else 'already verified'}"
            )

        # Metadata comes first and is tiny.  Transformers remain sequential so
        # at most one 1.9 GB release shard is transient at a time by default.
        for pin, name, spec in physical_specs(q8_manifest, hq_manifest):
            release.install_file(
                pin,
                manifest_digests[pin.key],
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

        payload = receipt_payload(q8_manifest, hq_manifest)
        release._atomic_json(overlay_dir / RECEIPT_NAME, payload)
        ready, detail = verify_receipt(models_root, q8_manifest, hq_manifest)
        if not ready:
            fail(f"new overlay receipt failed verification: {detail}")
        report(
            f"[receipt] {RECEIPT_NAME}: written only after all selected files and links verified"
        )


def install_control_overlay(
    models_root: Path,
    q8_manifest: Mapping[str, object],
    manifest_digest: str,
    *,
    opener: object,
    attempts: int,
    timeout: int,
    range_chunk_bytes: int,
    range_workers: int,
    report: Callable[[str], None] = log,
) -> None:
    """Install only Q8 distilled-control bytes, links, and its receipt."""
    q8 = validate_control_manifest(q8_manifest)
    specs = shared_specs(q8)
    for name, spec in specs.items():
        ready, detail = verify_shared_source(models_root, name, spec)
        if not ready:
            fail(f"Q4 prerequisite failed for {name}: {detail}")

    with _overlay_lock(models_root):
        overlay_dir = models_root / OVERLAY_DIRECTORY
        _check_pack_directory(overlay_dir, label="Q8 overlay directory", required=False)
        overlay_dir.mkdir(mode=0o755, parents=False, exist_ok=True)
        for name, spec in specs.items():
            changed = install_shared_link(
                models_root, name, spec, source_preverified=True
            )
            report(
                f"[link] {name}: "
                f"{'created and verified' if changed else 'already verified'}"
            )

        for pin, name, spec in control_physical_specs(q8):
            release.install_file(
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

        payload = control_receipt_payload(q8)
        release._atomic_json(overlay_dir / CONTROL_RECEIPT_NAME, payload)
        ready, detail = verify_control_receipt(models_root, q8)
        if not ready:
            fail(f"new distilled-control receipt failed verification: {detail}")
        report(
            f"[receipt] {CONTROL_RECEIPT_NAME}: distilled-control scope verified; "
            "HQ/dev was not selected"
        )


def fetch_manifests(*, opener: object, timeout: int) -> tuple[dict, dict]:
    manifests: dict[str, dict] = {}
    for pin in PINS:
        manifest, raw = release.fetch_pinned_manifest(pin, opener=opener, timeout=timeout)
        digest = release.sha256_bytes(raw)
        if digest != pin.manifest_sha256:
            raise release.ManifestError(f"{pin.key}: internal manifest digest mismatch")
        manifests[pin.key] = manifest
        log(f"[manifest] {pin.key}: {len(raw)} bytes, SHA-256 {digest} verified")
    return validate_overlay_manifests(manifests[Q8_PIN.key], manifests[HQ_PIN.key])


def fetch_control_manifest(*, opener: object, timeout: int) -> dict:
    """Fetch and validate only the immutable Q8 manifest."""
    manifest, raw = release.fetch_pinned_manifest(Q8_PIN, opener=opener, timeout=timeout)
    digest = release.sha256_bytes(raw)
    if digest != Q8_PIN.manifest_sha256:
        raise release.ManifestError("q8_25: internal manifest digest mismatch")
    log(f"[manifest] {Q8_PIN.key}: {len(raw)} bytes, SHA-256 {digest} verified")
    return validate_control_manifest(manifest)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the minimal verified LTX-2.5 Q8 advanced-feature overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="verify both pinned manifests and tiny CDN probes; download no weights",
    )
    mode.add_argument(
        "--plan",
        action="store_true",
        help="verify Q4 prerequisites and calculate disk needs; download no weights",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="hash-check the installed overlay, links, and receipt; download no weights",
    )
    mode.add_argument(
        "--install",
        action="store_true",
        help="install the exact selective overlay",
    )
    parser.add_argument(
        "--scope",
        choices=(FULL_SCOPE, CONTROL_SCOPE),
        default=FULL_SCOPE,
        help=(
            "full keeps the existing dev+distilled overlay; distilled-control selects "
            "only Ingredients/Union/Motion prerequisites and never HQ/dev weights"
        ),
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--range-chunk-mib",
        type=int,
        default=0,
        help="use exact resumable HTTP ranges of this size; 0 uses normal shard streaming",
    )
    parser.add_argument(
        "--range-workers",
        type=int,
        default=1,
        help="parallel exact-range requests within the current shard",
    )
    parser.add_argument("--ipv6", action="store_true", help="force curl's IPv6 route")
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=DEFAULT_RESERVE_BYTES / 2**30,
        help="free disk space to leave untouched",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm acceptance of the LTX-2.x Community License (install only)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts <= 0 or args.reserve_gib < 0:
        parser.error("--timeout and --attempts must be positive; --reserve-gib cannot be negative")
    if not 0 <= args.range_chunk_mib <= 1024:
        parser.error("--range-chunk-mib must be 0 or between 1 and 1024")
    if not 1 <= args.range_workers <= 16:
        parser.error("--range-workers must be between 1 and 16")
    if args.range_workers > 1 and args.range_chunk_mib == 0:
        parser.error("--range-workers above 1 requires --range-chunk-mib")
    if args.install and not args.accept_license:
        parser.error("--install requires --accept-license after reviewing the license")
    return args


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    models_root = args.models_root.expanduser().resolve()
    range_chunk_bytes = args.range_chunk_mib * 2**20
    opener = release.build_opener(force_ipv6=args.ipv6)
    try:
        if args.scope == CONTROL_SCOPE:
            q8_manifest = fetch_control_manifest(opener=opener, timeout=args.timeout)
            if args.preflight:
                result = release.probe_small_asset(
                    Q8_PIN, q8_manifest, opener=opener, timeout=args.timeout
                )
                log(f"[preflight] {Q8_PIN.key}: {result}")
                log(
                    "[preflight] PASS - Q8 manifest/CDN verified; HQ manifest and "
                    "all weight shards were not requested"
                )
                return 0

            if args.plan or args.verify:
                results, missing = control_overlay_status(models_root, q8_manifest)
                failures = 0
                for result in results:
                    marker = "OK" if result.ready else "NEEDED"
                    log(f"[{marker}] {result.kind}/{result.name}: {result.detail}")
                    failures += int(args.verify and not result.ready)
                if args.plan:
                    required = estimate_control_required_space(
                        models_root,
                        q8_manifest,
                        reserve_bytes=int(args.reserve_gib * 2**30),
                        range_chunk_bytes=range_chunk_bytes,
                        range_workers=args.range_workers,
                    )
                    free = shutil.disk_usage(_nearest_existing(models_root)).free
                    log(
                        "[plan] distilled-control physical bytes: "
                        f"{selected_control_physical_bytes(q8_manifest):,}"
                    )
                    log(f"[plan] missing physical bytes: {missing:,}")
                    log(
                        "[plan] verified selected bytes already in private staging: "
                        f"{verified_control_staged_bytes(models_root, q8_manifest):,}"
                    )
                    log(f"[plan] conservative free-space requirement: {required:,}")
                    log(f"[plan] currently free: {free:,}")
                    log(
                        f"[plan] excluded by scope: {HQ_TRANSFORMER} and "
                        f"{HQ_DISTILLED_LORA}; HQ manifest was not fetched"
                    )
                    return 0
                if failures:
                    log(
                        f"[verify] FAIL - {failures} distilled-control entry/entries "
                        "missing or invalid"
                    )
                    return 1
                log(
                    "[verify] PASS - distilled-control files, safe links, and distinct "
                    "receipt match the pinned Q8 manifest"
                )
                return 0

            models_root.mkdir(parents=True, exist_ok=True)
            _check_pack_directory(models_root, label="models root", required=True)
            _check_pack_directory(
                models_root / Q4_DIRECTORY,
                label="verified Q4 prerequisite",
                required=True,
            )
            required = estimate_control_required_space(
                models_root,
                q8_manifest,
                reserve_bytes=int(args.reserve_gib * 2**30),
                range_chunk_bytes=range_chunk_bytes,
                range_workers=args.range_workers,
            )
            free = shutil.disk_usage(models_root).free
            if free < required:
                fail(
                    f"insufficient free space: need conservative {required:,} bytes, "
                    f"have {free:,} bytes"
                )
            log(
                f"[install] distilled-control disk preflight passed: {free:,} bytes free, "
                f"{required:,} bytes conservative requirement"
            )
            install_control_overlay(
                models_root,
                q8_manifest,
                Q8_PIN.manifest_sha256,
                opener=opener,
                attempts=args.attempts,
                timeout=args.timeout,
                range_chunk_bytes=range_chunk_bytes,
                range_workers=args.range_workers,
            )
            log(
                "[install] PASS - distilled-control overlay complete; HQ/dev and HQ "
                "distilled LoRA were not requested"
            )
            return 0

        q8_manifest, hq_manifest = fetch_manifests(opener=opener, timeout=args.timeout)
        if args.preflight:
            for pin, manifest in ((Q8_PIN, q8_manifest), (HQ_PIN, hq_manifest)):
                result = release.probe_small_asset(
                    pin, manifest, opener=opener, timeout=args.timeout
                )
                log(f"[preflight] {pin.key}: {result}")
            log("[preflight] PASS - manifests/CDN verified; no weight shard downloaded")
            return 0

        if args.plan or args.verify:
            results, missing = overlay_status(models_root, q8_manifest, hq_manifest)
            failures = 0
            for result in results:
                marker = "OK" if result.ready else "NEEDED"
                log(f"[{marker}] {result.kind}/{result.name}: {result.detail}")
                failures += int(args.verify and not result.ready)
            if args.plan:
                required = estimate_required_space(
                    models_root,
                    q8_manifest,
                    hq_manifest,
                    reserve_bytes=int(args.reserve_gib * 2**30),
                    range_chunk_bytes=range_chunk_bytes,
                    range_workers=args.range_workers,
                )
                free = shutil.disk_usage(_nearest_existing(models_root)).free
                log(f"[plan] selected physical bytes: {selected_physical_bytes(q8_manifest, hq_manifest):,}")
                log(f"[plan] missing physical bytes: {missing:,}")
                log(
                    "[plan] verified selected bytes already in private staging: "
                    f"{verified_selected_staged_bytes(models_root, q8_manifest, hq_manifest):,}"
                )
                log(f"[plan] conservative free-space requirement: {required:,}")
                log(f"[plan] currently free: {free:,}")
                log(
                    f"[plan] excluded by design: {HQ_DISTILLED_LORA} "
                    f"({int(_files(hq_manifest)[HQ_DISTILLED_LORA]['bytes']):,} bytes)"
                )
                return 0
            if failures:
                log(f"[verify] FAIL - {failures} selected entry/entries missing or invalid")
                return 1
            log("[verify] PASS - physical files, safe links, and receipt match pinned manifests")
            return 0

        models_root.mkdir(parents=True, exist_ok=True)
        _check_pack_directory(models_root, label="models root", required=True)
        _check_pack_directory(
            models_root / Q4_DIRECTORY, label="verified Q4 prerequisite", required=True
        )
        required = estimate_required_space(
            models_root,
            q8_manifest,
            hq_manifest,
            reserve_bytes=int(args.reserve_gib * 2**30),
            range_chunk_bytes=range_chunk_bytes,
            range_workers=args.range_workers,
        )
        free = shutil.disk_usage(models_root).free
        if free < required:
            fail(
                f"insufficient free space: need conservative {required:,} bytes, "
                f"have {free:,} bytes"
            )
        log(
            f"[install] disk preflight passed: {free:,} bytes free, "
            f"{required:,} bytes conservative requirement"
        )
        install_overlay(
            models_root,
            q8_manifest,
            hq_manifest,
            {Q8_PIN.key: Q8_PIN.manifest_sha256, HQ_PIN.key: HQ_PIN.manifest_sha256},
            opener=opener,
            attempts=args.attempts,
            timeout=args.timeout,
            range_chunk_bytes=range_chunk_bytes,
            range_workers=args.range_workers,
        )
        log(
            "[install] PASS - selective Q8 overlay complete; the optional distilled LoRA "
            "was not downloaded"
        )
        return 0
    except KeyboardInterrupt:
        log("[install] interrupted; verified shard progress is preserved for resume")
        return 130
    except release.InstallerError as exc:
        log(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
