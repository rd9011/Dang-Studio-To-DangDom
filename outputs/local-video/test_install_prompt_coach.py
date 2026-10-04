from __future__ import annotations

import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import install_prompt_coach as subject


class PromptCoachInstallerTests(unittest.TestCase):
    def test_extracts_only_regular_ollama_resources(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "ollama.zip"
            executable = zipfile.ZipInfo("Ollama.app/Contents/Resources/ollama")
            executable.external_attr = (stat.S_IFREG | 0o755) << 16
            library = zipfile.ZipInfo("Ollama.app/Contents/Resources/lib/example.dylib")
            library.external_attr = (stat.S_IFREG | 0o644) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(executable, b"ollama")
                bundle.writestr(library, b"library")
                bundle.writestr("Ollama.app/Contents/MacOS/Ollama", b"not selected")
            destination = root / "runtime"
            subject._extract_runtime_archive(archive, destination)
            self.assertEqual((destination / "ollama").read_bytes(), b"ollama")
            self.assertEqual((destination / "lib/example.dylib").read_bytes(), b"library")
            self.assertTrue((destination / "ollama").stat().st_mode & stat.S_IXUSR)

    def test_rejects_archive_traversal_even_outside_selected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "ollama.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape", b"bad")
                bundle.writestr("Ollama.app/Contents/Resources/ollama", b"ollama")
            with self.assertRaises(subject.InstallError):
                subject._extract_runtime_archive(archive, root / "runtime")
            self.assertFalse((root / "escape").exists())

    def test_rejects_symbolic_link_members(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "ollama.zip"
            symlink = zipfile.ZipInfo("Ollama.app/Contents/Resources/ollama")
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(symlink, b"target")
            with self.assertRaises(subject.InstallError):
                subject._extract_runtime_archive(archive, root / "runtime")


if __name__ == "__main__":
    unittest.main()
