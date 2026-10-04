from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import install_ltx25_runtime as runtime
import verify_ltx25 as verification


HERE = Path(__file__).resolve().parent
CHECKOUT_RUNTIME = HERE.parents[1] / "work/local-video/ltx-2-mlx"


def _copy_backport_files(root: Path) -> None:
    for relative in runtime.PATCH_PATHS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CHECKOUT_RUNTIME / relative, destination)


def _make_venv(root: Path) -> None:
    bin_dir = root / ".venv/bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").write_bytes(b"python")
    (bin_dir / "ltx-2-mlx").write_bytes(b"cli")


def _write_receipt(root: Path) -> None:
    receipt = {
        "schema": "dang-studio-ltx-runtime/1",
        "source_revision": runtime.SOURCE_REVISION,
        "source_tag": runtime.SOURCE_TAG,
        "backport_commits": list(runtime.BACKPORT_COMMITS),
        "source_mode": "git",
    }
    (root / "install-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def _verified_git_output(_root: Path, *args: str) -> str:
    if args == ("rev-parse", "HEAD"):
        return runtime.SOURCE_REVISION
    if args == ("remote", "-v"):
        return f"origin\t{runtime.SOURCE_URL} (fetch)\norigin\t{runtime.SOURCE_URL} (push)"
    if args == ("diff", "--name-only"):
        return "\n".join(runtime.PATCH_PATHS)
    if args == ("rev-list", "-n", "1", runtime.SOURCE_TAG):
        return runtime.SOURCE_REVISION
    raise AssertionError(f"unexpected Git query: {args}")


class PublicGitRuntimeTests(unittest.TestCase):
    def test_verify_runtime_accepts_exact_pinned_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            _copy_backport_files(root)
            _make_venv(root)
            _write_receipt(root)

            with mock.patch.object(runtime, "_git_output", side_effect=_verified_git_output):
                ready, note = runtime.verify_runtime(root)

            self.assertTrue(ready, note)
            self.assertIn("verified Git source", note)
            self.assertIn("low-memory edit backport", note)

    def test_patch_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            _copy_backport_files(root)
            _make_venv(root)
            _write_receipt(root)
            (root / runtime.PATCH_PATHS[0]).write_bytes(b"tampered")

            with mock.patch.object(runtime, "_git_output", side_effect=_verified_git_output):
                ready, note = runtime.verify_runtime(root)

            self.assertFalse(ready)
            self.assertIn("hash mismatch", note)

    def test_install_fetches_public_git_source_and_uses_uv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            destination = temporary / "data/runtimes/ltx-2-mlx"
            receipt = destination / "install-receipt.json"
            uv_cache = temporary / "data/cache/uv-ltx-runtime"
            commands: list[tuple[str, ...]] = []

            def fake_run(command, **_kwargs):
                argv = tuple(str(item) for item in command)
                commands.append(argv)
                if argv[:3] == ("/usr/bin/git", "clone", "--no-checkout"):
                    checkout = Path(argv[-1])
                    (checkout / ".git").mkdir(parents=True)
                    _copy_backport_files(checkout)
                elif argv[0] == "/tools/uv":
                    project = Path(argv[argv.index("--project") + 1])
                    _make_venv(project)
                return subprocess.CompletedProcess(argv, 0, b"patch", b"")

            def fake_which(name: str) -> str | None:
                return {"git": "/usr/bin/git", "uv": "/tools/uv"}.get(name)

            with (
                mock.patch.object(runtime, "RUNTIME_ROOT", destination),
                mock.patch.object(runtime, "RECEIPT_PATH", receipt),
                mock.patch.object(runtime, "UV_CACHE", uv_cache),
                mock.patch.object(runtime.shutil, "which", side_effect=fake_which),
                mock.patch.object(runtime, "_run", side_effect=fake_run),
                mock.patch.object(runtime, "_git_output", side_effect=_verified_git_output),
            ):
                runtime.install_runtime()

            clone_commands = [command for command in commands if command[0:2] == ("/usr/bin/git", "clone")]
            self.assertEqual(len(clone_commands), 1)
            self.assertEqual(clone_commands[0][-2], runtime.SOURCE_URL)
            self.assertTrue(any(command[0:2] == ("/tools/uv", "sync") for command in commands))
            self.assertTrue((destination / ".git").is_dir())
            installed_receipt = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(installed_receipt["source_mode"], "git")

    def test_model_verifier_accepts_exact_public_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            _copy_backport_files(root)
            _make_venv(root)

            def fake_run(command, **_kwargs):
                args = tuple(command[1:])
                if args == ("rev-parse", "HEAD"):
                    return runtime.SOURCE_REVISION
                if args == ("rev-list", "-n", "1", runtime.SOURCE_TAG):
                    return runtime.SOURCE_REVISION
                if args == ("remote", "-v"):
                    return f"origin\t{runtime.SOURCE_URL} (fetch)"
                if args == ("status", "--porcelain", "--untracked-files=all"):
                    return "\n".join(f" M {path}" for path in runtime.PATCH_PATHS)
                if args == ("diff", "--cached", "--name-only"):
                    return ""
                if args == ("ls-files", "--others", "--exclude-standard"):
                    return ""
                if args == ("diff", "--name-only"):
                    return "\n".join(runtime.PATCH_PATHS)
                raise AssertionError(f"unexpected verification command: {command}")

            reports: list[str] = []
            with (
                mock.patch.object(verification, "RUNTIME", root),
                mock.patch.object(verification, "VENV_PYTHON", root / ".venv/bin/python"),
                mock.patch.object(verification, "CLI", root / ".venv/bin/ltx-2-mlx"),
                mock.patch.object(verification, "_run", side_effect=fake_run),
            ):
                verification._verify_source(reports.append)

            self.assertIn("pinned remote", reports[0])
            self.assertIn("low-RAM edit backport", reports[0])


if __name__ == "__main__":
    unittest.main()
