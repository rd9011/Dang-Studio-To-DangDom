from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install_ai_upscaler as subject


class AIUpscalerInstallerTests(unittest.TestCase):
    def test_source_archive_and_reduced_tree_are_immutable(self) -> None:
        pin = subject.UPSTREAM_SOURCE_PIN
        self.assertEqual(pin.revision, subject.EXPECTED_UPSTREAM_COMMIT)
        self.assertEqual(pin.archive_bytes, 1_038_973)
        self.assertEqual(
            pin.archive_sha256,
            "e76a9e768d77d76d6e4f9fe8fcab7e1b5ffd554a756ce19c8d8b76e231426039",
        )
        self.assertEqual(
            pin.tree_sha256,
            "32ee96c2fe35439d614d02a72cb38bbedfcad06f5bbde8c58f5273d629eff6c0",
        )
        # This upstream has no standalone LICENSE file; its README's License
        # section is therefore part of the exact reduced source snapshot.
        self.assertIn("README.md", pin.includes)

    def test_install_uses_archive_and_uv_from_path_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = root / "runtimes/ai-video-upscaler"
            source_fixture = root / "fixtures/upscale.py"
            source_fixture.parent.mkdir(parents=True)
            source_fixture.write_text(
                "# Minimal source-only Real-ESRGAN installer fixture.\n"
                "def upscale_image_coreml(*_args, **_kwargs):\n"
                "    raise AssertionError('installer must not import source')\n",
                encoding="utf-8",
            )
            source_hash = hashlib.sha256(source_fixture.read_bytes()).hexdigest()
            cache_zip = root / "cache/model.zip"
            cache_zip.parent.mkdir(parents=True)
            cache_zip.write_bytes(b"transport fixture")
            commands: list[list[str]] = []
            staged_pins = []

            def fake_stage(destination, pin, *_args, **_kwargs):
                staged_pins.append(pin)
                destination.mkdir(parents=True)
                shutil.copyfile(source_fixture, destination / "upscale.py")
                (destination / subject.PROVENANCE_NAME).write_text("{}", encoding="utf-8")
                return "archive"

            def fake_extract(_archive, destination):
                destination.mkdir(parents=True)

            def fake_receipt(root_path):
                (root_path / subject.RECEIPT.name).write_text("{}", encoding="utf-8")

            def fake_run(command, **_kwargs):
                commands.append(list(command))
                return ""

            with (
                mock.patch.object(subject, "RUNTIME_ROOT", runtime),
                mock.patch.object(subject, "RECEIPT", runtime / "install-receipt.json"),
                mock.patch.object(subject, "CACHE_ZIP", cache_zip),
                mock.patch.object(subject, "UV_CACHE", root / "cache/uv"),
                mock.patch.object(subject, "EXPECTED_UPSTREAM_SCRIPT_SHA256", source_hash),
                mock.patch.object(
                    subject,
                    "verify_install",
                    side_effect=[(False, "absent"), (True, "ready")],
                ),
                mock.patch.object(subject, "stage_pinned_source", side_effect=fake_stage),
                mock.patch.object(subject.shutil, "which", return_value="/usr/local/bin/uv"),
                mock.patch.object(subject, "_extract_verified_model", side_effect=fake_extract),
                mock.patch.object(subject, "_write_receipt", side_effect=fake_receipt),
                mock.patch.object(subject, "_run", side_effect=fake_run),
            ):
                subject.install()

            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][:2], ["/usr/local/bin/uv", "sync"])
            self.assertIn("--frozen", commands[0])
            self.assertIn("--no-dev", commands[0])
            self.assertEqual(commands[0][commands[0].index("--python") + 1], "3.12")
            self.assertFalse(any("git" in command[0] for command in commands))
            self.assertEqual(staged_pins, [subject.UPSTREAM_SOURCE_PIN])
            self.assertTrue((runtime / "upscale.py").is_file())
            self.assertEqual(
                (runtime / "upscale.py").read_bytes(), source_fixture.read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
