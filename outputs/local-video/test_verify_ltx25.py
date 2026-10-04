from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import install_phosphene_ltx25_release as installer
import verify_ltx25 as subject


class VerifyLtx25Tests(unittest.TestCase):
    def test_metal_sandbox_guard_detects_denial_without_importing_mlx(self) -> None:
        checker = Mock(return_value=1)
        library = Mock(sandbox_check=checker)
        with (
            patch.object(subject.sys, "platform", "darwin"),
            patch("ctypes.CDLL", return_value=library),
        ):
            reason = subject._metal_sandbox_block_reason()

        self.assertIn("blocked Apple GPU/Metal access", reason)
        self.assertIn("--skip-gpu", reason)
        checker.assert_called_once_with(os.getpid(), b"iokit-open-user-client", 0)

    def test_gpu_verification_stops_before_mlx_probe_when_sandboxed(self) -> None:
        runner = Mock(return_value="0.14.19+ltx25.7")
        with (
            patch.object(subject.platform, "system", return_value="Darwin"),
            patch.object(subject.platform, "machine", return_value="arm64"),
            patch.object(subject.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"),
            patch.object(subject, "_run", runner),
            patch.object(subject, "_metal_sandbox_block_reason", return_value="Metal denied"),
        ):
            with self.assertRaisesRegex(subject.VerificationError, "Metal denied"):
                subject._verify_tools(require_gpu=True, report=lambda _message: None)

        self.assertEqual(runner.call_count, 1)

    def test_runtime_release_and_split_pack_pins_are_exact(self) -> None:
        self.assertEqual(subject.SOURCE_REPOSITORY, "mrbizarro/ltx-2-mlx")
        self.assertEqual(subject.SOURCE_URL, "https://github.com/mrbizarro/ltx-2-mlx.git")
        self.assertEqual(subject.SOURCE_TAG, "v0.14.19+ltx25.7")
        self.assertEqual(subject.SOURCE_REVISION, "bf419b6f76e7753993eb7c3df3b15e1451310409")
        self.assertEqual(subject.RELEASE_REPOSITORY, installer.RELEASE_REPOSITORY)
        self.assertEqual(subject.RELEASE_TAG, installer.RELEASE_TAG)
        self.assertEqual(subject.MODEL.name, "ltx-2.5-mlx-q4")
        self.assertEqual(subject.GEMMA_MODEL.name, "gemma4-12b-ltx25-q4")
        self.assertEqual(
            subject.DDALCU_FALLBACK_GEMMA_MODEL.name,
            "gemma4-12b-ltx25-q4-ddalcu-fallback",
        )
        self.assertEqual(subject.BASE_PACK.manifest_sha256, installer.PACKS["q4_25"].manifest_sha256)
        self.assertEqual(subject.GEMMA_PACK.manifest_sha256, installer.PACKS["gemma4_25"].manifest_sha256)

    def test_embedded_manifests_cover_both_complete_packs(self) -> None:
        self.assertEqual(len(subject.BASE_FILES), 19)
        self.assertEqual(len(subject.GEMMA_FILES), 11)
        self.assertEqual(subject.BASE_PACK.bytes, 20_743_182_268)
        self.assertEqual(subject.GEMMA_PACK.bytes, 6_731_490_171)
        self.assertEqual(subject.MODEL_BYTES, 27_474_672_439)
        self.assertEqual(
            subject.BASE_FILES["transformer-distilled.safetensors"],
            subject.Artifact(
                11_320_074_467,
                "7c8c4f71a4cf88c683a7be47983bc98749883fbc9d702750487e0943c8e48102",
            ),
        )
        self.assertEqual(
            subject.GEMMA_FILES["model-00001-of-00001.safetensors"],
            subject.Artifact(
                6_699_142_782,
                "a3e8b1629e3cd3974b9c03fd337315c02edbd517fe58300b8e409e2959547704",
            ),
        )
        for pack in subject.PACKS:
            self.assertIn("LICENSE-LTX-2.x-Community-License.md", pack.files)
            self.assertTrue(all(item.size > 0 and len(item.sha256) == 64 for item in pack.files.values()))

    def test_receipt_contract_rejects_any_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            pack = subject.Pack(
                key="tiny",
                path=path,
                manifest_asset="tiny.json",
                manifest_sha256="a" * 64,
                files={"tiny.bin": subject.Artifact(1, "b" * 64)},
            )
            receipt = subject._receipt_expected(pack)
            receipt["license_exception"] = {
                "installed_sha256": subject.CURRENT_LICENSE_SHA256,
            }
            receipt_path = path / subject.RECEIPT_NAME
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            subject._verify_receipt(pack, lambda _message: None)

            receipt["release_tag"] = "moved-tag"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(subject.VerificationError, "receipt mismatch"):
                subject._verify_receipt(pack, lambda _message: None)

    def test_unrecognized_gemma_directory_is_rejected_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(subject.VerificationError, "Untrusted Gemma directory"):
                subject._verify_model(
                    full_hash=False,
                    report=lambda _message: None,
                    gemma_model=Path(temporary) / "arbitrary-gemma",
                )

    def test_low_ram_edit_patch_is_exactly_allowlisted_and_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "packages/runtime/cli.py": b"verified cli patch\n",
                "packages/runtime/retake.py": b"verified retake patch\n",
            }
            pins = {}
            for relative, payload in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                pins[relative] = subject.sha256(path)
            modified = "\n".join(files)
            with (
                patch.object(subject, "RUNTIME", root),
                patch.object(subject, "LOW_RAM_EDIT_PATCHES", pins),
                patch.object(subject, "_run", side_effect=["", "", modified]),
            ):
                state = subject._verify_low_ram_patch_state(
                    status=" M packages/runtime/cli.py\n M packages/runtime/retake.py",
                    required=True,
                )
            self.assertIn("verified", state)

        with self.assertRaisesRegex(subject.VerificationError, "backport is missing"):
            subject._verify_low_ram_patch_state(status="", required=True)


if __name__ == "__main__":
    unittest.main()
