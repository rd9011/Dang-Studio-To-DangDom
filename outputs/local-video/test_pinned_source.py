from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pinned_source as subject


class PinnedSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _source(self) -> Path:
        source = self.root / "expected"
        (source / "src/package").mkdir(parents=True)
        (source / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
        (source / "LICENSE").write_text("fixture licence\n", encoding="utf-8")
        (source / "src/package/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        return source

    def _archive(self, source: Path, *, symlink: bool = False) -> Path:
        archive = self.root / ("bad.tar.gz" if symlink else "source.tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source).as_posix()
                handle.add(path, arcname=f"fixture-deadbeef/{relative}", recursive=False)
            ignored = b"not part of the reduced runtime"
            info = tarfile.TarInfo("fixture-deadbeef/docs/ignored.txt")
            info.size = len(ignored)
            handle.addfile(info, io.BytesIO(ignored))
            if symlink:
                link = tarfile.TarInfo("fixture-deadbeef/docs/unsafe")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                handle.addfile(link)
        return archive

    def _pin(self, source: Path, archive: Path) -> subject.SourcePin:
        return subject.SourcePin(
            name="fixture",
            repository="https://github.com/example/fixture.git",
            revision="deadbeef",
            archive_url="https://codeload.github.com/example/fixture/tar.gz/deadbeef",
            archive_sha256=subject.sha256_file(archive),
            archive_bytes=archive.stat().st_size,
            archive_root="fixture-deadbeef",
            tree_sha256=subject.source_tree_digest(source),
            includes=("pyproject.toml", "LICENSE", "src/"),
        )

    def test_safe_extraction_reduces_tree_and_verifies_provenance(self) -> None:
        source = self._source()
        archive = self._archive(source)
        pin = self._pin(source, archive)
        destination = self.root / "installed"
        subject.extract_verified_archive(archive, destination, pin)
        self.assertTrue((destination / "LICENSE").is_file())
        self.assertFalse((destination / "docs").exists())
        ready, note = subject.verify_source_tree(destination, pin)
        self.assertTrue(ready, note)
        self.assertIn("Git-free", note)

    def test_hash_drift_and_tree_mutation_are_rejected(self) -> None:
        source = self._source()
        archive = self._archive(source)
        pin = self._pin(source, archive)
        bad_pin = subject.SourcePin(**{**pin.__dict__, "archive_sha256": "0" * 64})
        with self.assertRaisesRegex(subject.SourceInstallError, "byte/hash pin"):
            subject.extract_verified_archive(archive, self.root / "bad", bad_pin)
        destination = self.root / "installed"
        subject.extract_verified_archive(archive, destination, pin)
        (destination / "src/package/__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
        ready, note = subject.verify_source_tree(destination, pin)
        self.assertFalse(ready)
        self.assertIn("tree SHA-256 differs", note)

    def test_archive_rejects_symlink_even_outside_reduced_subset(self) -> None:
        source = self._source()
        archive = self._archive(source, symlink=True)
        pin = self._pin(source, archive)
        with self.assertRaisesRegex(subject.SourceInstallError, "member type"):
            subject.extract_verified_archive(archive, self.root / "installed", pin)

    def test_stage_pinned_source_uses_verified_archive_and_preserves_license(self) -> None:
        source = self._source()
        archive = self._archive(source)
        pin = self._pin(source, archive)
        destination = self.root / "installed"
        with mock.patch.object(subject, "download_verified_archive", return_value=archive):
            mode = subject.stage_pinned_source(
                destination,
                pin,
                cache_root=self.root / "cache",
            )
        self.assertEqual(mode, "archive")
        self.assertEqual((destination / "LICENSE").read_text(encoding="utf-8"), "fixture licence\n")
        self.assertTrue(subject.verify_source_tree(destination, pin)[0])

    def test_archive_downloader_resolves_curl_from_path_and_scrubs_tokens(self) -> None:
        source = self._source()
        archive = self._archive(source)
        pin = self._pin(source, archive)
        cache = self.root / "cache"

        def fake_run(command, *, env, check):
            self.assertFalse(check)
            self.assertEqual(command[0], "/usr/bin/curl")
            self.assertNotIn("GITHUB_TOKEN", env)
            self.assertNotIn("HF_TOKEN", env)
            Path(command[command.index("--output") + 1]).write_bytes(archive.read_bytes())
            return subject.subprocess.CompletedProcess(command, 0)

        with mock.patch.object(subject.shutil, "which", return_value="/usr/bin/curl"), mock.patch.object(
            subject.subprocess, "run", side_effect=fake_run
        ), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "secret", "HF_TOKEN": "secret"}):
            downloaded = subject.download_verified_archive(pin, cache)
        self.assertEqual(downloaded.read_bytes(), archive.read_bytes())


if __name__ == "__main__":
    unittest.main()
