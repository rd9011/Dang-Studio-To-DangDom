from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install_character_voice as subject


class CharacterVoiceInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exact_source_model_and_dependency_pins(self) -> None:
        self.assertEqual(subject.SOURCE_COMMIT, "65b18437192794391a0308a8f705b1e33e633948")
        self.assertEqual(
            subject.SOURCE_PIN.archive_sha256,
            "c7c0db5207ef7ba8f660b870b9c84e4796fb16de9f39867d12b27e1eef3144ac",
        )
        self.assertEqual(
            subject.SOURCE_PIN.tree_sha256,
            "55d1d2ec850ec23b1a0e97804118c9df8991c67ff9e2dfeea7ccdba43c684143",
        )
        self.assertIn("LICENSE", subject.SOURCE_PIN.includes)
        self.assertEqual(subject.MODEL_REVISION, "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
        self.assertEqual(subject.MODEL_TOTAL_BYTES, 3_208_844_550)
        self.assertEqual(subject.EXPECTED_PACKAGES["chatterbox-tts"], "0.1.7")
        self.assertEqual(subject.EXPECTED_PACKAGES["setuptools"], "84.0.0")
        self.assertEqual(subject.EXPECTED_PACKAGES["torch"], "2.6.0")
        self.assertEqual(subject.EXPECTED_PACKAGES["transformers"], "5.2.0")
        self.assertEqual(subject.EXPECTED_PACKAGES["resemble-perth"], "1.0.1")
        files = {item.path: item for item in subject.MODEL_FILES}
        self.assertEqual(set(files), {
            "t3_mtl23ls_v3.safetensors",
            "s3gen.pt",
            "ve.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
        })
        self.assertEqual(
            files["t3_mtl23ls_v3.safetensors"].sha256,
            "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953",
        )

    def test_runtime_template_is_hash_locked(self) -> None:
        lock = subject.TEMPLATE_ROOT / "uv.lock"
        project = subject.TEMPLATE_ROOT / "pyproject.toml"
        self.assertTrue(lock.is_file())
        self.assertTrue(project.is_file())
        lock_text = lock.read_text(encoding="utf-8")
        self.assertIn('name = "torch"', lock_text)
        self.assertIn('version = "2.6.0"', lock_text)
        self.assertIn("sha256:", lock_text)
        self.assertEqual(len(subject.template_digest()), 64)

    def test_cli_has_no_token_or_insecure_transport_option(self) -> None:
        parser = subject._build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertFalse({"token", "hf_token", "access_token", "insecure_tls"} & destinations)
        help_text = parser.format_help().lower()
        self.assertNotIn("--token", help_text)
        self.assertNotIn("--insecure", help_text)

    def test_verify_model_checks_size_hash_and_symlink(self) -> None:
        blob = b"verified-model"
        item = subject.ModelFile("model.bin", len(blob), hashlib.sha256(blob).hexdigest())
        model = self.root / "model"
        model.mkdir()
        (model / "model.bin").write_bytes(blob)
        with mock.patch.object(subject, "MODEL_FILES", (item,)):
            self.assertEqual(subject.verify_model(model, full_hash=True), [])
            (model / "model.bin").write_bytes(b"wrong")
            self.assertRegex(subject.verify_model(model, full_hash=True)[0], "wrong size")
            (model / "model.bin").unlink()
            (model / "target").write_bytes(blob)
            (model / "model.bin").symlink_to(model / "target")
            self.assertRegex(subject.verify_model(model, full_hash=True)[0], "symlinked")

    def test_whole_file_curl_is_pinned_public_secure_and_hash_verified(self) -> None:
        blob = b"abcdef"
        item = subject.ModelFile("tiny.bin", len(blob), hashlib.sha256(blob).hexdigest())
        root = self.root / "stage"
        root.mkdir()
        old_partial = root / "tiny.bin.partial"
        old_partial.write_bytes(b"abc")
        commands: list[list[str]] = []
        environments: list[dict[str, str]] = []

        def fake_run(command, *, check, env):
            self.assertTrue(check)
            command = list(command)
            commands.append(command)
            environments.append(env)
            Path(command[command.index("--output") + 1]).write_bytes(blob)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(subject.subprocess, "run", side_effect=fake_run), mock.patch.dict(
            subject.os.environ,
            {"HF_TOKEN": "must-not-leak", "HUGGING_FACE_HUB_TOKEN": "also-secret"},
            clear=False,
        ):
            result = subject.download_model_file(item, root, report=lambda _message: None)
        self.assertEqual(result.read_bytes(), blob)
        self.assertFalse(old_partial.exists())
        command = commands[0]
        self.assertEqual(command[:2], ["/usr/bin/curl", "--disable"])
        self.assertIn("=https", command)
        self.assertIn("--tlsv1.2", command)
        self.assertIn("--retry-all-errors", command)
        self.assertIn("--progress-bar", command)
        self.assertNotIn("--silent", command)
        self.assertNotIn("--insecure", command)
        self.assertNotIn("-k", command)
        self.assertNotIn("Range:", " ".join(command))
        self.assertNotIn("Authorization:", " ".join(command))
        self.assertTrue(command[-1].startswith("https://huggingface.co/"))
        self.assertIn(subject.MODEL_REVISION, command[-1])
        self.assertNotIn("HF_TOKEN", environments[0])
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", environments[0])

    def test_failed_fresh_download_preserves_old_partial_and_cleans_private_attempt(self) -> None:
        blob = b"abcdef"
        item = subject.ModelFile("tiny.bin", len(blob), hashlib.sha256(blob).hexdigest())
        root = self.root / "stage"
        root.mkdir()
        partial = root / "tiny.bin.partial"
        partial.write_bytes(b"abc")

        def failed_run(command, **_kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"new-but-incomplete")
            raise subprocess.CalledProcessError(18, command)

        with mock.patch.object(subject.subprocess, "run", side_effect=failed_run):
            with self.assertRaisesRegex(subject.InstallError, "secure model download failed"):
                subject.download_model_file(item, root, report=lambda _message: None)
        self.assertEqual(partial.read_bytes(), b"abc")
        self.assertEqual(list(root.glob(".tiny.bin.fresh-*.partial")), [])
        self.assertFalse((root / "tiny.bin").exists())

    def test_hash_failure_preserves_old_partial_and_does_not_publish(self) -> None:
        blob = b"abcdef"
        item = subject.ModelFile("tiny.bin", len(blob), hashlib.sha256(blob).hexdigest())
        root = self.root / "stage"
        root.mkdir()
        partial = root / "tiny.bin.partial"
        partial.write_bytes(b"abc")

        def corrupt_run(command, **_kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"ghijkl")
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(subject.subprocess, "run", side_effect=corrupt_run):
            with self.assertRaisesRegex(subject.InstallError, "failed its pin"):
                subject.download_model_file(item, root, report=lambda _message: None)
        self.assertEqual(partial.read_bytes(), b"abc")
        self.assertFalse((root / "tiny.bin").exists())

    def test_completed_legacy_partial_is_verified_and_published_without_network(self) -> None:
        blob = b"abcdef"
        item = subject.ModelFile("tiny.bin", len(blob), hashlib.sha256(blob).hexdigest())
        root = self.root / "stage"
        root.mkdir()
        (root / "tiny.bin.partial").write_bytes(blob)
        with mock.patch.object(subject.subprocess, "run") as run:
            result = subject.download_model_file(item, root, report=lambda _message: None)
        run.assert_not_called()
        self.assertEqual(result.read_bytes(), blob)
        self.assertFalse((root / "tiny.bin.partial").exists())

    def test_incomplete_partial_does_not_reduce_required_fresh_space(self) -> None:
        blob = b"abcdef"
        item = subject.ModelFile("tiny.bin", len(blob), hashlib.sha256(blob).hexdigest())
        root = self.root / "stage"
        root.mkdir()
        (root / "tiny.bin.partial").write_bytes(b"abc")
        with mock.patch.object(subject, "MODEL_FILES", (item,)):
            self.assertEqual(subject._missing_model_bytes(root), len(blob))

    def test_model_path_rejects_symlinked_parent(self) -> None:
        root = self.root / "stage"
        outside = self.root / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "nested").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(subject.InstallError, "unsafe manifest path"):
            subject._safe_child(root, "nested/model.bin")

    def test_scrubbed_environment_removes_credentials(self) -> None:
        with mock.patch.dict(
            subject.os.environ,
            {"HF_TOKEN": "secret", "GITHUB_TOKEN": "secret-2", "SAFE_VALUE": "yes"},
            clear=False,
        ):
            environment = subject._scrubbed_environment()
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("GITHUB_TOKEN", environment)
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(environment["SAFE_VALUE"], "yes")

    def test_runtime_install_uses_archive_and_uv_from_path_without_git(self) -> None:
        runtime = self.root / "runtime"
        staging = self.root / "runtime-staging"
        uv_cache = self.root / "uv-cache"
        commands: list[list[str]] = []

        def fake_stage(destination, *_args, **_kwargs):
            destination.mkdir(parents=True)
            (destination / subject.PROVENANCE_NAME).write_text("{}", encoding="utf-8")
            return "archive"

        def fake_run(command, **_kwargs):
            commands.append(list(command))

        with (
            mock.patch.object(subject, "RUNTIME_ROOT", runtime),
            mock.patch.object(subject, "RUNTIME_STAGING", staging),
            mock.patch.object(subject, "UV_CACHE_DIR", uv_cache),
            mock.patch.object(subject, "stage_pinned_source", side_effect=fake_stage),
            mock.patch.object(subject.shutil, "which", return_value="/usr/local/bin/uv"),
            mock.patch.object(subject, "_run", side_effect=fake_run),
            mock.patch.object(subject, "_install_pinned_source") as install_source,
            mock.patch.object(subject, "runtime_ready", return_value=(True, "ready")),
        ):
            subject._install_runtime()

        self.assertEqual(commands[0][0:2], ["/usr/local/bin/uv", "sync"])
        self.assertFalse(any("git" in command[0] for command in commands))
        install_source.assert_called_once()
        self.assertTrue((runtime / "source" / subject.PROVENANCE_NAME).is_file())

    def test_runtime_accepts_uv_interpreter_symlink_but_not_venv_directory_symlink(self) -> None:
        runtime = self.root / "runtime"
        source = runtime / "source"
        bin_dir = runtime / ".venv" / "bin"
        home = self.root / "cpython" / "bin"
        source.mkdir(parents=True)
        bin_dir.mkdir(parents=True)
        home.mkdir(parents=True)
        for name in ("pyproject.toml", "uv.lock"):
            (runtime / name).write_bytes((subject.TEMPLATE_ROOT / name).read_bytes())
        interpreter = home / "python3.11"
        interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
        interpreter.chmod(0o755)
        (bin_dir / "python").symlink_to(interpreter)
        (runtime / ".venv" / "pyvenv.cfg").write_text(
            f"home = {home}\nimplementation = CPython\nversion_info = 3.11.13\n",
            encoding="utf-8",
        )

        with mock.patch.object(subject, "_git_head", return_value=subject.SOURCE_COMMIT), mock.patch.object(
            subject, "_git_clean", return_value=True
        ), mock.patch.object(
            subject, "_runtime_versions", return_value=(dict(subject.EXPECTED_PACKAGES), "")
        ), mock.patch.object(subject, "_runtime_import_probe", return_value=""):
            ready, _note = subject.runtime_ready(runtime)
        self.assertTrue(ready)

        (runtime / ".venv").rename(runtime / ".venv-real")
        (runtime / ".venv").symlink_to(runtime / ".venv-real", target_is_directory=True)
        with mock.patch.object(subject, "_git_head", return_value=subject.SOURCE_COMMIT), mock.patch.object(
            subject, "_git_clean", return_value=True
        ):
            ready, note = subject.runtime_ready(runtime)
        self.assertFalse(ready)
        self.assertIn("incomplete", note)

    def test_runtime_rejects_interpreter_symlink_that_disagrees_with_pyvenv_config(self) -> None:
        runtime = self.root / "runtime"
        source = runtime / "source"
        bin_dir = runtime / ".venv" / "bin"
        real_home = self.root / "real-home"
        declared_home = self.root / "declared-home"
        source.mkdir(parents=True)
        bin_dir.mkdir(parents=True)
        real_home.mkdir()
        declared_home.mkdir()
        for name in ("pyproject.toml", "uv.lock"):
            (runtime / name).write_bytes((subject.TEMPLATE_ROOT / name).read_bytes())
        interpreter = real_home / "python3.11"
        interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
        interpreter.chmod(0o755)
        declared = declared_home / "python3.11"
        declared.write_text("#!/bin/sh\n", encoding="utf-8")
        declared.chmod(0o755)
        (bin_dir / "python").symlink_to(interpreter)
        (runtime / ".venv" / "pyvenv.cfg").write_text(
            f"home = {declared_home}\nimplementation = CPython\nversion_info = 3.11.13\n",
            encoding="utf-8",
        )
        with mock.patch.object(subject, "_git_head", return_value=subject.SOURCE_COMMIT), mock.patch.object(
            subject, "_git_clean", return_value=True
        ), mock.patch.object(subject, "_runtime_versions") as versions, mock.patch.object(
            subject, "_runtime_import_probe"
        ) as import_probe:
            ready, note = subject.runtime_ready(runtime)
        self.assertFalse(ready)
        self.assertIn("does not match", note)
        versions.assert_not_called()
        import_probe.assert_not_called()

    def test_check_only_parser_does_not_install(self) -> None:
        with mock.patch.object(subject, "readiness", return_value={"ready": False}), mock.patch.object(
            subject, "_install_runtime"
        ) as runtime, mock.patch.object(subject, "_install_model") as model, mock.patch.object(
            subject, "_print_status"
        ):
            result = subject.main(["--check-only"])
        self.assertEqual(result, 1)
        runtime.assert_not_called()
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
