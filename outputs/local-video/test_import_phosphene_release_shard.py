"""GPU-free tests for collision-safe external Phosphene shard imports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import import_phosphene_release_shard as subject
import install_phosphene_ltx25_release as release


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_pin(key: str) -> release.PackPin:
    return release.PackPin(
        key=key,
        label=f"synthetic {key}",
        directory=f"{key}-model",
        manifest_asset=f"{key}__manifest.json",
        manifest_bytes=1,
        manifest_sha256="0" * 64,
    )


def make_manifest(pin: release.PackPin, asset: str, payload: bytes) -> dict:
    return {
        "schema": "phosphene-release-manifest/1",
        "repo_key": pin.key,
        "pack": pin.directory,
        "release": {
            "release_repo": release.RELEASE_REPOSITORY,
            "tag": release.RELEASE_TAG,
            "asset_prefix": f"{pin.key}__",
        },
        "files": {
            release.LICENSE_NAME: json.loads(json.dumps(release.STALE_LICENSE_SPEC)),
            "model.safetensors": {
                "bytes": len(payload),
                "sha256": digest(payload),
                "shards": [
                    {
                        "asset": asset,
                        "bytes": len(payload),
                        "sha256": digest(payload),
                    }
                ],
            },
        },
    }


class NoNetworkOpener:
    def open(self, *_args, **_kwargs):
        raise AssertionError("verified staged shard unexpectedly used the network")


class ShardImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()
        self.models = self.root / "models"
        self.q4_payload = b"tiny q4 shard payload"
        self.gemma_payload = b"tiny gemma shard payload"
        self.q4_asset = "q4_25__transformer.safetensors.part004"
        self.gemma_asset = "gemma4_25__model.safetensors.part000"
        self.q4_pin = make_pin("q4_25")
        self.gemma_pin = make_pin("gemma4_25")
        self.pins = {"q4_25": self.q4_pin, "gemma4_25": self.gemma_pin}
        self.raw = {"q4_25": b"q4 manifest", "gemma4_25": b"gemma manifest"}
        self.manifests = {
            "q4_25": make_manifest(self.q4_pin, self.q4_asset, self.q4_payload),
            "gemma4_25": make_manifest(
                self.gemma_pin,
                self.gemma_asset,
                self.gemma_payload,
            ),
        }
        self.patches = [
            mock.patch.object(release, "PACKS", self.pins),
            mock.patch.object(release, "PACK_ORDER", tuple(self.pins)),
            mock.patch.object(subject.advanced, "PINS", ()),
            mock.patch.object(
                release,
                "fetch_pinned_manifest",
                side_effect=lambda pin, **_kwargs: (
                    self.manifests[pin.key],
                    self.raw[pin.key],
                ),
            ),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_source(self, asset: str, payload: bytes) -> Path:
        path = self.downloads / asset
        path.write_bytes(payload)
        return path

    def stage_paths(self, pin: release.PackPin, asset: str) -> tuple[Path, Path]:
        return release._stage_paths(
            self.models.resolve(),
            pin,
            "model.safetensors",
            asset,
        )

    def seed_q4_partial(
        self,
        payload: bytes,
        *,
        metadata: dict | None = None,
    ) -> tuple[Path, Path, dict]:
        staged, sidecar_path = self.stage_paths(self.q4_pin, self.q4_asset)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)
        identity = release._shard_identity(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
        )
        release._atomic_json(sidecar_path, metadata if metadata is not None else identity)
        return staged, sidecar_path, identity

    def test_q4_import_uses_distinct_cow_inode_and_is_immediately_reusable(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        source_inode = source.stat().st_ino
        result = subject.import_shard(
            source,
            pack_key="q4_25",
            models_root=self.models,
            opener=object(),
            report=lambda _message: None,
        )

        staged, metadata = self.stage_paths(self.q4_pin, self.q4_asset)
        self.assertFalse(source.exists())
        self.assertEqual(result.staged_path, staged)
        self.assertEqual(staged.read_bytes(), self.q4_payload)
        self.assertNotEqual(staged.stat().st_ino, source_inode)
        self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o400)
        expected_identity = release._shard_identity(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
        )
        self.assertEqual(
            json.loads(metadata.read_text()),
            {
                **expected_identity,
                release.VERIFIED_SHARD_SHA256_FIELD: digest(self.q4_payload),
            },
        )

        release.download_shard(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
            staged,
            metadata,
            opener=NoNetworkOpener(),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )

    def test_cow_candidate_is_private_before_its_full_hash(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        original_hash = subject._sha256_descriptor
        observed_modes: list[int] = []

        def inspect_mode(descriptor: int) -> str:
            observed_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
            return original_hash(descriptor)

        with mock.patch.object(subject, "_sha256_descriptor", side_effect=inspect_mode):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertEqual(len(observed_modes), 2)
        self.assertEqual(observed_modes[1], 0o400)

    def test_auto_detects_gemma_from_exact_manifest_lookup(self) -> None:
        source = self.write_source(self.gemma_asset, self.gemma_payload)
        result = subject.import_shard(
            source,
            models_root=self.models,
            opener=object(),
            report=lambda _message: None,
        )
        self.assertEqual(result.pack, "gemma4_25")
        staged, _metadata = self.stage_paths(self.gemma_pin, self.gemma_asset)
        self.assertTrue(staged.is_file())

    def test_q8_import_is_registered_and_consumable_by_installer(self) -> None:
        payload = b"tiny q8 transformer shard"
        asset = "q8_25__transformer-distilled.safetensors.part000"
        pin = make_pin("q8_25")
        manifest = make_manifest(pin, asset, payload)
        manifest["files"][subject.advanced.Q8_TRANSFORMER] = manifest["files"].pop(
            "model.safetensors"
        )
        raw = b"q8 manifest"
        source = self.write_source(asset, payload)

        with (
            mock.patch.object(subject.advanced, "PINS", (pin,)),
            mock.patch.object(
                release,
                "fetch_pinned_manifest",
                return_value=(manifest, raw),
            ),
        ):
            result = subject.import_shard(
                source,
                pack_key="q8_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )

        staged, metadata = release._stage_paths(
            self.models.resolve(),
            pin,
            subject.advanced.Q8_TRANSFORMER,
            asset,
        )
        self.assertEqual(result.staged_path, staged)
        self.assertFalse(source.exists())
        self.assertEqual(
            release._read_json(metadata)[release.VERIFIED_SHARD_SHA256_FIELD],
            digest(payload),
        )

        changed = release.install_file(
            pin,
            digest(raw),
            subject.advanced.Q8_TRANSFORMER,
            manifest["files"][subject.advanced.Q8_TRANSFORMER],
            models_root=self.models.resolve(),
            opener=NoNetworkOpener(),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        installed = (
            self.models.resolve() / pin.directory / subject.advanced.Q8_TRANSFORMER
        )
        self.assertTrue(changed)
        self.assertEqual(installed.read_bytes(), payload)

    def test_hq_import_surface_excludes_the_optional_distilled_lora(self) -> None:
        transformer = b"dev"
        lora = b"excluded"
        manifest = make_manifest(
            subject.advanced.HQ_PIN,
            "hq_25__transformer-dev.safetensors.part000",
            transformer,
        )
        manifest["files"][subject.advanced.HQ_TRANSFORMER] = manifest["files"].pop(
            "model.safetensors"
        )
        manifest["files"][subject.advanced.HQ_DISTILLED_LORA] = {
            "bytes": len(lora),
            "sha256": digest(lora),
            "shards": [
                {
                    "asset": "hq_25__ltx-2.5-22b-distilled-lora-450.safetensors.part000",
                    "bytes": len(lora),
                    "sha256": digest(lora),
                }
            ],
        }
        selected = subject._importable_files(subject.advanced.HQ_PIN, manifest)
        self.assertEqual(set(selected), {subject.advanced.HQ_TRANSFORMER})

    def test_normal_installer_consumes_import_without_a_network_request(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        subject.import_shard(
            source,
            pack_key="q4_25",
            models_root=self.models,
            opener=object(),
            report=lambda _message: None,
        )
        spec = self.manifests["q4_25"]["files"]["model.safetensors"]
        changed = release.install_file(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            "model.safetensors",
            spec,
            models_root=self.models.resolve(),
            opener=NoNetworkOpener(),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        installed = self.models.resolve() / self.q4_pin.directory / "model.safetensors"
        self.assertTrue(changed)
        self.assertEqual(installed.read_bytes(), self.q4_payload)

    def test_explicit_replace_partial_atomically_installs_verified_full_shard(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        source_inode = source.stat().st_ino
        prefix = self.q4_payload[:7]
        staged, metadata, identity = self.seed_q4_partial(prefix)
        old_inode = staged.stat().st_ino
        sidecar = {
            **identity,
            "validator_kind": "etag",
            "validator": '"synthetic-stable"',
        }
        release._atomic_json(metadata, sidecar)

        result = subject.import_shard(
            source,
            pack_key="q4_25",
            models_root=self.models,
            opener=object(),
            replace_partial=True,
            report=lambda _message: None,
        )

        self.assertEqual(result.staged_path, staged)
        self.assertFalse(source.exists())
        self.assertEqual(staged.read_bytes(), self.q4_payload)
        self.assertNotEqual(staged.stat().st_ino, source_inode)
        self.assertNotEqual(staged.stat().st_ino, old_inode)
        self.assertEqual(json.loads(metadata.read_text()), sidecar)
        self.assertFalse(list(staged.parent.glob(".*.external-replacement-*")))
        release.download_shard(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
            staged,
            metadata,
            opener=NoNetworkOpener(),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )

    def test_trusted_partial_still_requires_explicit_replace_option(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        staged, metadata, _identity = self.seed_q4_partial(self.q4_payload[:5])
        prior = staged.read_bytes()
        sidecar = metadata.read_bytes()
        with self.assertRaisesRegex(subject.ShardImportError, "replace-partial"):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertEqual(staged.read_bytes(), prior)
        self.assertEqual(metadata.read_bytes(), sidecar)
        self.assertEqual(source.read_bytes(), self.q4_payload)

    def test_replace_partial_rejects_full_corrupt_unbound_and_symlink_states(self) -> None:
        cases = ("full", "corrupt", "missing-sidecar", "untrusted", "symlink")
        for case in cases:
            with self.subTest(case=case):
                case_root = self.root / f"replace-{case}"
                models = case_root / "models"
                downloads = case_root / "downloads"
                downloads.mkdir(parents=True)
                source = downloads / self.q4_asset
                source.write_bytes(self.q4_payload)
                staged, metadata = release._stage_paths(
                    models.resolve(),
                    self.q4_pin,
                    "model.safetensors",
                    self.q4_asset,
                )
                staged.parent.mkdir(parents=True)
                identity = release._shard_identity(
                    self.q4_pin,
                    digest(self.raw["q4_25"]),
                    self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
                )
                if case == "full":
                    staged.write_bytes(self.q4_payload)
                    release._atomic_json(metadata, identity)
                    expected = "full-size"
                elif case == "corrupt":
                    staged.write_bytes(b"z" * 7)
                    release._atomic_json(metadata, identity)
                    expected = "not an exact prefix"
                elif case == "missing-sidecar":
                    staged.write_bytes(self.q4_payload[:7])
                    expected = "no sidecar"
                elif case == "untrusted":
                    staged.write_bytes(self.q4_payload[:7])
                    release._atomic_json(metadata, {**identity, "sha256": "0" * 64})
                    expected = "untrusted"
                else:
                    real_partial = case_root / "real-partial"
                    real_partial.write_bytes(self.q4_payload[:7])
                    staged.symlink_to(real_partial)
                    release._atomic_json(metadata, identity)
                    expected = "symlinked"
                with self.assertRaisesRegex(subject.ShardImportError, expected):
                    subject.import_shard(
                        source,
                        pack_key="q4_25",
                        models_root=models,
                        opener=object(),
                        replace_partial=True,
                        report=lambda _message: None,
                    )
                self.assertEqual(source.read_bytes(), self.q4_payload)

    def test_replace_failure_rolls_back_candidate_and_preserves_old_partial(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        staged, metadata, _identity = self.seed_q4_partial(self.q4_payload[:8])
        old_inode = staged.stat().st_ino
        old_sidecar = metadata.read_bytes()
        with mock.patch.object(subject.os, "replace", side_effect=OSError("synthetic replace failure")):
            with self.assertRaisesRegex(OSError, "synthetic replace failure"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    replace_partial=True,
                    report=lambda _message: None,
                )
        self.assertEqual(source.read_bytes(), self.q4_payload)
        self.assertEqual(staged.read_bytes(), self.q4_payload[:8])
        self.assertEqual(staged.stat().st_ino, old_inode)
        self.assertEqual(metadata.read_bytes(), old_sidecar)
        self.assertFalse(list(staged.parent.glob(".*.external-replacement-*")))

    def test_unlink_failure_after_replace_preserves_both_complete_cow_files(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        source_inode = source.stat().st_ino
        staged, metadata, _identity = self.seed_q4_partial(self.q4_payload[:8])
        with mock.patch.object(subject, "_unlink_source", side_effect=OSError("busy")):
            with self.assertRaisesRegex(subject.ShardImportError, "committed to staging"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    replace_partial=True,
                    report=lambda _message: None,
                )
        self.assertEqual(source.read_bytes(), self.q4_payload)
        self.assertEqual(staged.read_bytes(), self.q4_payload)
        self.assertEqual(source.stat().st_ino, source_inode)
        self.assertNotEqual(staged.stat().st_ino, source_inode)
        self.assertTrue(metadata.is_file())
        self.assertFalse(list(staged.parent.glob(".*.external-replacement-*")))

    def test_source_writer_after_clone_cannot_mutate_committed_shard(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)

        def corrupt_then_unlink(path: Path) -> None:
            path.write_bytes(b"z" * len(self.q4_payload))
            path.unlink()

        with mock.patch.object(subject, "_unlink_source", side_effect=corrupt_then_unlink):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        staged, _metadata = self.stage_paths(self.q4_pin, self.q4_asset)
        self.assertEqual(staged.read_bytes(), self.q4_payload)
        self.assertEqual(digest(staged.read_bytes()), digest(self.q4_payload))

    def test_disappeared_source_preserves_verified_clone_on_precommit_failure(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        staged, metadata, _identity = self.seed_q4_partial(self.q4_payload[:8])
        original_clone = subject._clone_candidate

        def clone_then_remove_source(*args, **kwargs):
            candidate = original_clone(*args, **kwargs)
            source.unlink()
            return candidate

        with (
            mock.patch.object(subject, "_clone_candidate", side_effect=clone_then_remove_source),
            mock.patch.object(subject.os, "replace", side_effect=OSError("no commit")),
        ):
            with self.assertRaisesRegex(subject.ShardImportError, "recovery copy was preserved"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    replace_partial=True,
                    report=lambda _message: None,
                )
        self.assertFalse(source.exists())
        self.assertEqual(staged.read_bytes(), self.q4_payload[:8])
        self.assertTrue(metadata.is_file())
        recovery = list(staged.parent.glob(".*.external-replacement-*"))
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0].read_bytes(), self.q4_payload)

    def test_weak_resume_validator_is_not_trusted_for_replacement(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        staged, _metadata, identity = self.seed_q4_partial(
            self.q4_payload[:8],
            metadata={
                **release._shard_identity(
                    self.q4_pin,
                    digest(self.raw["q4_25"]),
                    self.manifests["q4_25"]["files"]["model.safetensors"]["shards"][0],
                ),
                "validator_kind": "etag",
                "validator": 'W/"weak"',
            },
        )
        with self.assertRaisesRegex(subject.ShardImportError, "unexpected fields"):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                replace_partial=True,
                report=lambda _message: None,
            )
        self.assertEqual(staged.read_bytes(), self.q4_payload[:8])
        self.assertEqual(source.read_bytes(), self.q4_payload)

    def test_wrong_pack_unknown_and_browser_partial_names_are_rejected(self) -> None:
        wrong_pack = self.write_source(self.q4_asset, self.q4_payload)
        with self.assertRaisesRegex(subject.ShardImportError, "not an exact pinned"):
            subject.import_shard(
                wrong_pack,
                pack_key="gemma4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertEqual(wrong_pack.read_bytes(), self.q4_payload)

        unknown = self.write_source("q4_25__renamed.bin", b"data")
        with self.assertRaisesRegex(subject.ShardImportError, "not an exact pinned"):
            subject.import_shard(
                unknown,
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )

        partial = self.write_source(self.q4_asset + ".crdownload", self.q4_payload)
        with self.assertRaisesRegex(subject.ShardImportError, "browser partial"):
            subject.import_shard(
                partial,
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )

    def test_stale_license_asset_is_not_importable(self) -> None:
        source = self.write_source(release.LICENSE_NAME, b"not accepted here")
        with self.assertRaisesRegex(subject.ShardImportError, "not an exact pinned"):
            subject.import_shard(
                source,
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertTrue(source.exists())

    def test_requires_absolute_regular_non_symlink_source(self) -> None:
        with self.assertRaisesRegex(subject.ShardImportError, "absolute"):
            subject.import_shard(
                Path(self.q4_asset),
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )

        directory = self.downloads / self.q4_asset
        directory.mkdir()
        with self.assertRaisesRegex(subject.ShardImportError, "non-regular"):
            subject.import_shard(
                directory,
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        directory.rmdir()

        real = self.downloads / "real"
        real.write_bytes(self.q4_payload)
        link = self.downloads / self.q4_asset
        link.symlink_to(real)
        with self.assertRaisesRegex(subject.ShardImportError, "symlinked"):
            subject.import_shard(
                link,
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_bytes(), self.q4_payload)

    def test_wrong_size_and_hash_leave_source_unmodified(self) -> None:
        short = self.write_source(self.q4_asset, b"short")
        original_mode = stat.S_IMODE(short.stat().st_mode)
        with self.assertRaisesRegex(subject.ShardImportError, "wrong size"):
            subject.import_shard(
                short,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertEqual(short.read_bytes(), b"short")

        short.unlink()
        wrong_hash = self.write_source(self.q4_asset, b"x" * len(self.q4_payload))
        with self.assertRaisesRegex(subject.ShardImportError, "SHA-256 mismatch"):
            subject.import_shard(
                wrong_hash,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertEqual(wrong_hash.read_bytes(), b"x" * len(self.q4_payload))
        self.assertEqual(stat.S_IMODE(wrong_hash.stat().st_mode), original_mode)
        staged, metadata = self.stage_paths(self.q4_pin, self.q4_asset)
        self.assertFalse(staged.exists())
        self.assertFalse(metadata.exists())

    def test_existing_shard_metadata_or_final_file_is_never_overwritten(self) -> None:
        cases = ("shard", "metadata", "final")
        for case in cases:
            with self.subTest(case=case):
                case_root = self.root / case
                models = case_root / "models"
                source_dir = case_root / "downloads"
                source_dir.mkdir(parents=True)
                source = source_dir / self.q4_asset
                source.write_bytes(self.q4_payload)
                staged, metadata = release._stage_paths(
                    models,
                    self.q4_pin,
                    "model.safetensors",
                    self.q4_asset,
                )
                sentinel = b"do not replace"
                if case == "shard":
                    staged.parent.mkdir(parents=True)
                    staged.write_bytes(sentinel)
                    collision = staged
                elif case == "metadata":
                    metadata.parent.mkdir(parents=True)
                    metadata.write_bytes(sentinel)
                    collision = metadata
                else:
                    collision = models / self.q4_pin.directory / "model.safetensors"
                    collision.parent.mkdir(parents=True)
                    collision.write_bytes(sentinel)
                with self.assertRaisesRegex(subject.ShardImportError, "overwrite"):
                    subject.import_shard(
                        source,
                        pack_key="q4_25",
                        models_root=models,
                        opener=object(),
                        report=lambda _message: None,
                    )
                self.assertEqual(collision.read_bytes(), sentinel)
                self.assertEqual(source.read_bytes(), self.q4_payload)

    def test_completed_assembly_position_rejects_duplicate_shard(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        spec = self.manifests["q4_25"]["files"]["model.safetensors"]
        assembly, metadata = release._stage_paths(
            self.models,
            self.q4_pin,
            "model.safetensors",
        )
        assembly.parent.mkdir(parents=True)
        assembly.write_bytes(self.q4_payload)
        identity = release._assembly_identity(
            self.q4_pin,
            digest(self.raw["q4_25"]),
            "model.safetensors",
            spec,
        )
        release._atomic_json(
            metadata,
            {**identity, "shards_done": 1, "assembled_bytes": len(self.q4_payload)},
        )
        with self.assertRaisesRegex(subject.ShardImportError, "already incorporated"):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                replace_partial=True,
                report=lambda _message: None,
            )
        self.assertTrue(source.exists())

    def test_cli_parses_explicit_replace_partial_option(self) -> None:
        source = self.downloads / self.q4_asset
        args = subject._parse_args(
            [str(source), "--pack", "q4_25", "--replace-partial"]
        )
        self.assertTrue(args.replace_partial)
        self.assertEqual(args.pack, "q4_25")

    def test_cross_filesystem_refusal_and_unlink_failure_cleanup(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        with mock.patch.object(subject, "_same_filesystem", return_value=False):
            with self.assertRaisesRegex(subject.ShardImportError, "different filesystems"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    report=lambda _message: None,
                )
        self.assertTrue(source.exists())

        with mock.patch.object(subject, "_unlink_source", side_effect=OSError("busy")):
            with self.assertRaisesRegex(subject.ShardImportError, "committed to staging"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    report=lambda _message: None,
                )
        staged, metadata = self.stage_paths(self.q4_pin, self.q4_asset)
        self.assertTrue(source.exists())
        self.assertEqual(staged.read_bytes(), self.q4_payload)
        self.assertTrue(metadata.is_file())
        self.assertNotEqual(staged.stat().st_ino, source.stat().st_ino)

    def test_source_replacement_after_open_is_detected(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        displaced = self.downloads / "original.inode"
        original = subject._sha256_descriptor

        def replace_path(descriptor: int) -> str:
            observed = original(descriptor)
            source.rename(displaced)
            source.write_bytes(self.q4_payload)
            return observed

        with mock.patch.object(subject, "_sha256_descriptor", side_effect=replace_path):
            with self.assertRaisesRegex(subject.ShardImportError, "source changed"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    report=lambda _message: None,
                )
        self.assertEqual(source.read_bytes(), self.q4_payload)
        self.assertEqual(displaced.read_bytes(), self.q4_payload)
        staged, metadata = self.stage_paths(self.q4_pin, self.q4_asset)
        self.assertFalse(staged.exists())
        self.assertFalse(metadata.exists())

    def test_shared_installer_lock_refuses_concurrent_import(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        with release._file_staging_lock(
            self.models,
            self.q4_pin,
            "model.safetensors",
        ):
            with self.assertRaisesRegex(release.InstallerError, "owns staging"):
                subject.import_shard(
                    source,
                    pack_key="q4_25",
                    models_root=self.models,
                    opener=object(),
                    report=lambda _message: None,
                )
        self.assertTrue(source.exists())

    def test_symlinked_staging_root_is_rejected(self) -> None:
        source = self.write_source(self.q4_asset, self.q4_payload)
        self.models.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.models / ".phosphene-release-staging").symlink_to(outside)
        with self.assertRaisesRegex(subject.ShardImportError, "unsafe directory"):
            subject.import_shard(
                source,
                pack_key="q4_25",
                models_root=self.models,
                opener=object(),
                report=lambda _message: None,
            )
        self.assertTrue(source.exists())
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
