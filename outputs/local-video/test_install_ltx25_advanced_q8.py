"""Offline tests for the selective, verified LTX-2.5 Q8 overlay."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install_ltx25_advanced_q8 as subject
import install_phosphene_ltx25_release as release


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec(name: str, *, size: int | None = None) -> dict:
    body = name.encode("utf-8")
    declared = len(body) if size is None else size
    return {
        "bytes": declared,
        "sha256": digest(body) if size is None else hashlib.sha256(name.encode()).hexdigest(),
        "shards": [
            {
                "asset": f"asset-{name}",
                "bytes": declared,
                "sha256": digest(body) if size is None else hashlib.sha256(name.encode()).hexdigest(),
            }
        ],
    }


def manifest_for(pin: release.PackPin, names: set[str]) -> dict:
    files: dict[str, dict] = {}
    for name in sorted(names):
        if name == release.LICENSE_NAME:
            files[name] = copy.deepcopy(release.STALE_LICENSE_SPEC)
        else:
            files[name] = spec(name)
    return {
        "schema": "phosphene-release-manifest/1",
        "repo_key": pin.key,
        "pack": pin.directory,
        "release": {
            "release_repo": release.RELEASE_REPOSITORY,
            "tag": release.RELEASE_TAG,
            "asset_prefix": f"{pin.key}__",
        },
        "quantizer": {"bits": 8, "group_size": 64},
        "files": files,
    }


def pinned_spec(name: str, pin: subject.EmbeddedFilePin, *, asset_prefix: str) -> dict:
    return {
        "bytes": pin.bytes,
        "sha256": pin.sha256,
        "shards": [
            {
                "asset": f"{asset_prefix}__{name}",
                "bytes": pin.bytes,
                "sha256": pin.sha256,
            }
        ],
    }


def fixture_manifests() -> tuple[dict, dict]:
    q8 = manifest_for(subject.Q8_PIN, set(subject.EXPECTED_Q8_FILES))
    hq = manifest_for(subject.HQ_PIN, set(subject.EXPECTED_HQ_FILES))
    for name, pin in subject.PHYSICAL_FILE_PINS.items():
        manifest = hq if name == subject.HQ_TRANSFORMER else q8
        prefix = "hq_25" if manifest is hq else "q8_25"
        manifest["files"][name] = pinned_spec(name, pin, asset_prefix=prefix)
    for name, pin in subject.SHARED_FILE_PINS.items():
        if name == release.LICENSE_NAME:
            continue
        q8["files"][name] = pinned_spec(name, pin, asset_prefix="q8_25")
    hq["files"][subject.HQ_DISTILLED_LORA] = pinned_spec(
        subject.HQ_DISTILLED_LORA,
        subject.EXCLUDED_LORA_PIN,
        asset_prefix="hq_25",
    )
    # The official manifests share the exact legal entries.
    hq["files"]["NOTICE.md"] = copy.deepcopy(q8["files"]["NOTICE.md"])
    return q8, hq


class AdvancedQ8OverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"
        self.models.mkdir()

    def test_official_manifest_pins_and_selective_names(self) -> None:
        self.assertEqual(subject.Q8_PIN.manifest_bytes, 10_558)
        self.assertEqual(
            subject.Q8_PIN.manifest_sha256,
            "bb93c860e4713d900c9e9408517de4986b22951768b0598e1fbd2833a131399d",
        )
        self.assertEqual(subject.HQ_PIN.manifest_bytes, 5_365)
        self.assertEqual(
            subject.HQ_PIN.manifest_sha256,
            "7de6ebce960c2836fbd69c725fc199f067d2b44f7fcc7a660f5e301f5bfbaf89",
        )
        q8, hq = fixture_manifests()
        q8, hq = subject.validate_overlay_manifests(q8, hq)
        selected = [(pin.key, name) for pin, name, _ in subject.physical_specs(q8, hq)]
        self.assertEqual(
            selected,
            [
                ("q8_25", "phosphene_quant_manifest.json"),
                ("q8_25", "quantize_config.json"),
                ("q8_25", "split_model.json"),
                ("q8_25", subject.Q8_TRANSFORMER),
                ("hq_25", subject.HQ_TRANSFORMER),
            ],
        )
        self.assertNotIn(subject.HQ_DISTILLED_LORA, [name for _, name in selected])

    def test_distilled_control_scope_selects_only_q8_files_and_distinct_receipt(self) -> None:
        q8, _hq = fixture_manifests()
        q8 = subject.validate_control_manifest(q8)
        selected = [(pin.key, name) for pin, name, _ in subject.control_physical_specs(q8)]
        self.assertEqual(
            selected,
            [
                ("q8_25", "phosphene_quant_manifest.json"),
                ("q8_25", "quantize_config.json"),
                ("q8_25", "split_model.json"),
                ("q8_25", subject.Q8_TRANSFORMER),
            ],
        )
        payload = subject.control_receipt_payload(
            q8, installed_utc="2026-09-24T10:00:00+00:00"
        )
        self.assertEqual(payload["schema"], subject.CONTROL_RECEIPT_SCHEMA)
        self.assertEqual(payload["scope"], subject.CONTROL_SCOPE)
        self.assertEqual(set(payload["manifests"]), {subject.Q8_PIN.key})
        self.assertNotIn(subject.HQ_TRANSFORMER, payload["physical_files"])
        self.assertIn(subject.HQ_TRANSFORMER, payload["not_selected_files"])
        self.assertNotEqual(subject.CONTROL_RECEIPT_NAME, subject.RECEIPT_NAME)

    def test_control_manifest_fetch_never_requests_hq_manifest(self) -> None:
        q8, _hq = fixture_manifests()
        with (
            mock.patch.object(
                release, "fetch_pinned_manifest", return_value=(q8, b"fixture")
            ) as fetch,
            mock.patch.object(
                release, "sha256_bytes", return_value=subject.Q8_PIN.manifest_sha256
            ),
        ):
            fetched = subject.fetch_control_manifest(opener=object(), timeout=1)
        self.assertEqual(fetched["repo_key"], subject.Q8_PIN.key)
        fetch.assert_called_once_with(subject.Q8_PIN, opener=mock.ANY, timeout=1)

    def test_scope_cli_is_explicit_and_full_remains_default(self) -> None:
        default = subject._parse_args(["--plan"])
        control = subject._parse_args(["--plan", "--scope", subject.CONTROL_SCOPE])
        self.assertEqual(default.scope, subject.FULL_SCOPE)
        self.assertEqual(control.scope, subject.CONTROL_SCOPE)

    def test_manifest_validation_rejects_topology_or_non_q8_recipe(self) -> None:
        q8, hq = fixture_manifests()
        q8["files"]["surprise.bin"] = spec("surprise.bin")
        with self.assertRaisesRegex(release.ManifestError, "unexpected file topology"):
            subject.validate_overlay_manifests(q8, hq)

        q8, hq = fixture_manifests()
        hq["quantizer"]["bits"] = 4
        with self.assertRaisesRegex(release.ManifestError, "Q8/group-size-64"):
            subject.validate_overlay_manifests(q8, hq)

        q8, hq = fixture_manifests()
        hq["files"]["NOTICE.md"] = spec("different-notice")
        with self.assertRaisesRegex(release.ManifestError, "disagree"):
            subject.validate_overlay_manifests(q8, hq)

    def test_shared_link_is_relative_exact_and_source_hash_verified(self) -> None:
        source_dir = self.models / subject.Q4_DIRECTORY
        source_dir.mkdir()
        body = b"shared-sidecar"
        file_spec = {
            "bytes": len(body),
            "sha256": digest(body),
            "shards": [],
        }
        (source_dir / "config.json").write_bytes(body)

        changed = subject.install_shared_link(self.models, "config.json", file_spec)
        self.assertTrue(changed)
        link = self.models / subject.OVERLAY_DIRECTORY / "config.json"
        self.assertTrue(link.is_symlink())
        self.assertEqual(
            os.readlink(link),
            f"../{subject.Q4_DIRECTORY}/config.json",
        )
        ready, detail = subject.verify_shared_link(self.models, "config.json", file_spec)
        self.assertTrue(ready, detail)

        link.unlink()
        os.symlink(str(source_dir / "config.json"), link)
        ready, detail = subject.verify_shared_link(self.models, "config.json", file_spec)
        self.assertFalse(ready)
        self.assertIn("unsafe link target", detail)

    def test_shared_link_refuses_bad_source_and_existing_entry(self) -> None:
        source_dir = self.models / subject.Q4_DIRECTORY
        source_dir.mkdir()
        path = source_dir / "config.json"
        path.write_bytes(b"wrong")
        wanted = {"bytes": 5, "sha256": digest(b"right"), "shards": []}
        with self.assertRaisesRegex(subject.OverlayError, "source is not verified"):
            subject.install_shared_link(self.models, "config.json", wanted)

        path.write_bytes(b"right")
        overlay = self.models / subject.OVERLAY_DIRECTORY
        overlay.mkdir()
        (overlay / "config.json").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaisesRegex(subject.OverlayError, "refusing to replace"):
            subject.install_shared_link(self.models, "config.json", wanted)

    def test_receipt_round_trip_is_strict_and_records_lora_exclusion(self) -> None:
        q8, hq = fixture_manifests()
        subject.validate_overlay_manifests(q8, hq)
        overlay = self.models / subject.OVERLAY_DIRECTORY
        overlay.mkdir()
        payload = subject.receipt_payload(
            q8, hq, installed_utc="2026-09-24T10:00:00+00:00"
        )
        release._atomic_json(overlay / subject.RECEIPT_NAME, payload)
        ready, detail = subject.verify_receipt(self.models, q8, hq)
        self.assertTrue(ready, detail)
        offline_ready, offline_detail = subject.verify_receipt_offline(self.models)
        self.assertTrue(offline_ready, offline_detail)
        excluded = payload["excluded_optional_files"]
        self.assertEqual(set(excluded), {subject.HQ_DISTILLED_LORA})
        self.assertIn("pre-fused Q8 distilled", excluded[subject.HQ_DISTILLED_LORA]["reason"])

        payload["manifests"]["q8_25"]["sha256"] = "0" * 64
        release._atomic_json(overlay / subject.RECEIPT_NAME, payload)
        ready, detail = subject.verify_receipt(self.models, q8, hq)
        self.assertFalse(ready)
        self.assertIn("does not match", detail)

    def test_safetensors_structure_check_reads_header_not_payload(self) -> None:
        path = self.root / "tiny.safetensors"
        header = json.dumps(
            {"tensor": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}},
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"data")
        ready, detail = subject._validate_safetensors_structure(path, path.stat().st_size)
        self.assertTrue(ready, detail)

        path.write_bytes(len(header).to_bytes(8, "little") + header + b"bad")
        ready, _ = subject._validate_safetensors_structure(path, path.stat().st_size + 1)
        self.assertFalse(ready)

    def test_public_offline_gate_uses_quick_and_full_hash_policies(self) -> None:
        q4 = self.models / subject.Q4_DIRECTORY
        overlay = self.models / subject.OVERLAY_DIRECTORY
        q4.mkdir()
        overlay.mkdir()
        for name in subject.PHYSICAL_FILE_PINS:
            (overlay / name).write_bytes(b"placeholder")
        for name in subject.SHARED_FILE_PINS:
            (q4 / name).write_bytes(b"placeholder")
            os.symlink(f"../{subject.Q4_DIRECTORY}/{name}", overlay / name)
        release._atomic_json(
            overlay / subject.RECEIPT_NAME,
            subject.embedded_receipt_payload(installed_utc="2026-09-24T10:00:00+00:00"),
        )

        calls: list[tuple[str, bool]] = []

        def fake_file_check(path, pin, *, hash_required):
            calls.append((path.name, hash_required))
            return True, "fixture verified"

        with mock.patch.object(subject, "_verify_embedded_file", side_effect=fake_file_check):
            results = subject.verify_installed_overlay(self.models, full_hash=False)
        self.assertEqual(len(results), 1 + len(subject.PHYSICAL_FILE_PINS) + len(subject.SHARED_FILE_PINS))
        quick = dict(calls)
        self.assertFalse(quick[subject.Q8_TRANSFORMER])
        self.assertFalse(quick[subject.HQ_TRANSFORMER])
        self.assertTrue(quick["quantize_config.json"])
        self.assertFalse(quick["connector.safetensors"])
        self.assertTrue(quick["config.json"])

        calls.clear()
        with mock.patch.object(subject, "_verify_embedded_file", side_effect=fake_file_check):
            subject.verify_installed_overlay(self.models, full_hash=True)
        full = dict(calls)
        self.assertTrue(full[subject.Q8_TRANSFORMER])
        self.assertTrue(full[subject.HQ_TRANSFORMER])

    def test_control_offline_gate_never_requires_dev_transformer(self) -> None:
        q4 = self.models / subject.Q4_DIRECTORY
        overlay = self.models / subject.OVERLAY_DIRECTORY
        q4.mkdir()
        overlay.mkdir()
        for name in subject.CONTROL_PHYSICAL_FILE_PINS:
            (overlay / name).write_bytes(b"placeholder")
        for name in subject.SHARED_FILE_PINS:
            (q4 / name).write_bytes(b"placeholder")
            os.symlink(f"../{subject.Q4_DIRECTORY}/{name}", overlay / name)
        release._atomic_json(
            overlay / subject.CONTROL_RECEIPT_NAME,
            subject.embedded_control_receipt_payload(
                installed_utc="2026-09-24T10:00:00+00:00"
            ),
        )

        calls: list[tuple[str, bool]] = []

        def fake_file_check(path, pin, *, hash_required):
            calls.append((path.name, hash_required))
            return True, "fixture verified"

        with mock.patch.object(subject, "_verify_embedded_file", side_effect=fake_file_check):
            results = subject.verify_installed_control_overlay(self.models, full_hash=False)
        checked = dict(calls)
        self.assertNotIn(subject.HQ_TRANSFORMER, checked)
        self.assertFalse(checked[subject.Q8_TRANSFORMER])
        self.assertEqual(
            len(results),
            1 + len(subject.CONTROL_PHYSICAL_FILE_PINS) + len(subject.SHARED_FILE_PINS),
        )

    def test_overlay_lock_closes_its_descriptor_exactly_once(self) -> None:
        real_close = os.close
        with mock.patch.object(subject.os, "close", wraps=real_close) as close:
            with subject._overlay_lock(self.models):
                pass
        self.assertEqual(close.call_count, 1)

    def test_install_overlay_requests_only_five_selected_files(self) -> None:
        q8, hq = fixture_manifests()
        calls: list[tuple[str, str]] = []

        def fake_install(pin, manifest_digest, name, file_spec, **kwargs):
            calls.append((pin.key, name))
            return True

        with (
            mock.patch.object(
                subject, "verify_shared_source", return_value=(True, "verified")
            ),
            mock.patch.object(subject, "install_shared_link", return_value=False),
            mock.patch.object(release, "install_file", side_effect=fake_install),
        ):
            subject.install_overlay(
                self.models,
                q8,
                hq,
                {"q8_25": "a" * 64, "hq_25": "b" * 64},
                opener=object(),
                attempts=1,
                timeout=1,
                range_chunk_bytes=0,
                range_workers=1,
                report=lambda _message: None,
            )

        self.assertEqual(
            calls,
            [
                ("q8_25", "phosphene_quant_manifest.json"),
                ("q8_25", "quantize_config.json"),
                ("q8_25", "split_model.json"),
                ("q8_25", subject.Q8_TRANSFORMER),
                ("hq_25", subject.HQ_TRANSFORMER),
            ],
        )
        self.assertNotIn(subject.HQ_DISTILLED_LORA, [name for _, name in calls])
        self.assertTrue((self.models / subject.OVERLAY_DIRECTORY / subject.RECEIPT_NAME).is_file())

    def test_install_control_overlay_requests_only_four_q8_files(self) -> None:
        q8, _hq = fixture_manifests()
        calls: list[tuple[str, str]] = []

        def fake_install(pin, manifest_digest, name, file_spec, **kwargs):
            calls.append((pin.key, name))
            return True

        with (
            mock.patch.object(
                subject, "verify_shared_source", return_value=(True, "verified")
            ),
            mock.patch.object(subject, "install_shared_link", return_value=False),
            mock.patch.object(release, "install_file", side_effect=fake_install),
        ):
            subject.install_control_overlay(
                self.models,
                q8,
                subject.Q8_PIN.manifest_sha256,
                opener=object(),
                attempts=1,
                timeout=1,
                range_chunk_bytes=0,
                range_workers=1,
                report=lambda _message: None,
            )

        self.assertEqual(
            calls,
            [
                ("q8_25", "phosphene_quant_manifest.json"),
                ("q8_25", "quantize_config.json"),
                ("q8_25", "split_model.json"),
                ("q8_25", subject.Q8_TRANSFORMER),
            ],
        )
        self.assertFalse(any(name == subject.HQ_TRANSFORMER for _, name in calls))
        overlay = self.models / subject.OVERLAY_DIRECTORY
        self.assertTrue((overlay / subject.CONTROL_RECEIPT_NAME).is_file())
        self.assertFalse((overlay / subject.RECEIPT_NAME).exists())

    def test_space_estimate_charges_selected_files_one_shard_and_reserve(self) -> None:
        q8, hq = fixture_manifests()
        physical = subject.physical_specs(q8, hq)
        missing = sum(int(entry[2]["bytes"]) for entry in physical)
        largest = max(int(entry[2]["shards"][0]["bytes"]) for entry in physical)
        reserve = 123
        required = subject.estimate_required_space(
            self.models,
            q8,
            hq,
            reserve_bytes=reserve,
            range_chunk_bytes=0,
            range_workers=1,
        )
        self.assertEqual(required, missing + largest + reserve)

    def test_space_estimate_credits_only_journaled_read_only_staged_shards(self) -> None:
        q8, hq = fixture_manifests()
        payload = b"manually imported q8 shard"
        shard = {
            "asset": "q8_25__transformer-distilled.safetensors.part000",
            "bytes": len(payload),
            "sha256": digest(payload),
        }
        q8["files"][subject.Q8_TRANSFORMER] = {
            "bytes": len(payload),
            "sha256": digest(payload),
            "shards": [shard],
        }
        staged, metadata = release._stage_paths(
            self.models,
            subject.Q8_PIN,
            subject.Q8_TRANSFORMER,
            shard["asset"],
        )
        staged.parent.mkdir(parents=True)
        staged.write_bytes(payload)
        staged.chmod(0o400)
        identity = release._shard_identity(
            subject.Q8_PIN,
            subject.Q8_PIN.manifest_sha256,
            shard,
        )
        release._atomic_json(
            metadata,
            {
                **identity,
                release.VERIFIED_SHARD_SHA256_FIELD: digest(payload),
            },
        )

        with_credit = subject.estimate_required_space(
            self.models,
            q8,
            hq,
            reserve_bytes=0,
            range_chunk_bytes=0,
            range_workers=1,
        )
        metadata_value = release._read_json(metadata)
        assert metadata_value is not None
        metadata_value.pop(release.VERIFIED_SHARD_SHA256_FIELD)
        release._atomic_json(metadata, metadata_value)
        no_journal = subject.estimate_required_space(
            self.models,
            q8,
            hq,
            reserve_bytes=0,
            range_chunk_bytes=0,
            range_workers=1,
        )
        self.assertEqual(no_journal - with_credit, len(payload))

        release._atomic_json(
            metadata,
            {
                **identity,
                release.VERIFIED_SHARD_SHA256_FIELD: digest(payload),
            },
        )
        staged.chmod(0o600)
        writable = subject.estimate_required_space(
            self.models,
            q8,
            hq,
            reserve_bytes=0,
            range_chunk_bytes=0,
            range_workers=1,
        )
        self.assertEqual(writable, no_journal)

    def test_unsafe_overlay_directory_is_never_followed(self) -> None:
        q8, hq = fixture_manifests()
        outside = self.root / "outside"
        outside.mkdir()
        os.symlink(outside, self.models / subject.OVERLAY_DIRECTORY)
        with self.assertRaisesRegex(subject.OverlayError, "unsafe Q8 overlay"):
            subject.estimate_required_space(
                self.models,
                q8,
                hq,
                reserve_bytes=0,
                range_chunk_bytes=0,
                range_workers=1,
            )
        ready, detail = subject.verify_receipt(self.models, q8, hq)
        self.assertFalse(ready)
        self.assertIn("symlinked overlay", detail)

    def test_install_requires_explicit_license_acceptance(self) -> None:
        with self.assertRaises(SystemExit):
            subject._parse_args(["--install"])
        args = subject._parse_args(["--install", "--accept-license"])
        self.assertTrue(args.install)


if __name__ == "__main__":
    unittest.main()
