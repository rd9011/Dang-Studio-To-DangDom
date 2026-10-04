"""Offline integrity, resume, and atomic-promotion tests for the GitHub installer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import install_phosphene_ltx25_release as subject


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "https://release-assets.githubusercontent.com/object",
    ):
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = subject._CaseInsensitiveHeaders(
            {key.lower(): value for key, value in (headers or {}).items()}
        )
        self._url = url

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._body.close()


class FakeOpener:
    def __init__(self, *responses, before_open=None):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []
        self.before_open = before_open

    def open(self, request: urllib.request.Request, timeout: int):
        self.requests.append(request)
        if self.before_open:
            self.before_open(request)
        if not self.responses:
            raise AssertionError("unexpected network request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class RangeMapOpener:
    """Thread-safe fake that serves responses by their exact Range header."""

    def __init__(
        self,
        body: bytes,
        *,
        delays: dict[tuple[int, int], float] | None = None,
        validators: dict[tuple[int, int], str] | None = None,
        failures: set[tuple[int, int]] | None = None,
    ):
        self.body = body
        self.delays = delays or {}
        self.validators = validators or {}
        self.failures = failures or set()
        self.requests: list[urllib.request.Request] = []
        self.completed: list[tuple[int, int]] = []
        self.lock = threading.Lock()

    def open(self, request: urllib.request.Request, timeout: int):
        range_header = request.get_header("Range")
        if not range_header or not range_header.startswith("bytes="):
            raise AssertionError("expected an exact Range header")
        start_text, end_text = range_header[6:].split("-", 1)
        key = (int(start_text), int(end_text))
        with self.lock:
            self.requests.append(request)
        time.sleep(self.delays.get(key, 0.0))
        if key in self.failures:
            raise urllib.error.URLError(f"synthetic failure for {key}")
        start, end = key
        body = self.body[start : end + 1]
        with self.lock:
            self.completed.append(key)
        return range_response(
            body,
            start=start,
            end=end,
            total=len(self.body),
            validator=self.validators.get(key, '"stable"'),
        )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_pin(raw: bytes = b"") -> subject.PackPin:
    return subject.PackPin(
        key="test",
        label="test pack",
        directory="test-pack",
        manifest_asset="test__manifest.json",
        manifest_bytes=len(raw),
        manifest_sha256=digest(raw),
    )


def make_spec(*shards: bytes, file_sha: str | None = None) -> dict:
    body = b"".join(shards)
    return {
        "bytes": len(body),
        "sha256": file_sha or digest(body),
        "shards": [
            {
                "asset": f"test__blob.part{index:03d}",
                "bytes": len(shard),
                "sha256": digest(shard),
            }
            for index, shard in enumerate(shards)
        ],
    }


def make_manifest(pin: subject.PackPin, spec: dict | None = None) -> dict:
    return {
        "schema": "phosphene-release-manifest/1",
        "repo_key": pin.key,
        "pack": pin.directory,
        "release": {
            "release_repo": subject.RELEASE_REPOSITORY,
            "tag": subject.RELEASE_TAG,
            "asset_prefix": f"{pin.key}__",
        },
        "files": {
            subject.LICENSE_NAME: json.loads(json.dumps(subject.STALE_LICENSE_SPEC)),
            "blob.bin": spec or make_spec(b"abc"),
        },
    }


def range_response(
    body: bytes,
    *,
    start: int,
    end: int,
    total: int,
    validator: str = '"stable"',
    content_length: int | None = None,
) -> FakeResponse:
    headers = {
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(len(body) if content_length is None else content_length),
        "ETag": validator,
    }
    return FakeResponse(body, status=206, headers=headers)


class InstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_official_manifest_pins_and_destinations(self) -> None:
        q4 = subject.PACKS["q4_25"]
        self.assertEqual(q4.directory, "ltx-2.5-mlx-q4")
        self.assertEqual(q4.manifest_bytes, 9_430)
        self.assertEqual(
            q4.manifest_sha256,
            "0013bfab7524d61d7c0e3d95733e6453cbab9dde944eb5a9100db8bf531ab424",
        )
        gemma = subject.PACKS["gemma4_25"]
        self.assertEqual(gemma.directory, "gemma4-12b-ltx25-q4")
        self.assertEqual(gemma.manifest_bytes, 5_387)
        self.assertEqual(
            gemma.manifest_sha256,
            "b995e479a8e3d2c08af35572fccf9412fad8b3c4f6ef3f2934d733f6596472d5",
        )

    def test_license_exception_is_exact_and_weight_files_are_unchanged(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin)
        validated = subject.validate_manifest(manifest, pin)
        self.assertNotIn(subject.LICENSE_NAME, subject.pack_files(validated))
        self.assertIn("blob.bin", subject.pack_files(validated))
        self.assertEqual(subject.CURRENT_LICENSE_SPEC["bytes"], 34_545)
        self.assertEqual(
            subject.CURRENT_LICENSE_PIN,
            "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654",
        )

        changed = make_manifest(pin)
        changed["files"][subject.LICENSE_NAME]["sha256"] = "0" * 64
        with self.assertRaisesRegex(subject.ManifestError, "exact known stale entry"):
            subject.validate_manifest(changed, pin)

    def test_manifest_rejects_traversal_and_inconsistent_shard_sizes(self) -> None:
        pin = make_pin()
        traversal = make_manifest(pin)
        traversal["files"]["../escape"] = traversal["files"].pop("blob.bin")
        with self.assertRaisesRegex(subject.ManifestError, "unsafe file name"):
            subject.validate_manifest(traversal, pin)

        inconsistent = make_manifest(pin)
        inconsistent["files"]["blob.bin"]["bytes"] += 1
        with self.assertRaisesRegex(subject.ManifestError, "shard bytes"):
            subject.validate_manifest(inconsistent, pin)

    def test_pinned_manifest_bytes_are_checked_before_json_is_trusted(self) -> None:
        provisional = make_pin()
        raw = json.dumps(make_manifest(provisional), separators=(",", ":")).encode()
        pin = make_pin(raw)
        # Rebuild because the pin's length/hash do not participate in manifest JSON.
        raw = json.dumps(make_manifest(pin), separators=(",", ":")).encode()
        pin = make_pin(raw)
        opener = FakeOpener(FakeResponse(raw, headers={"Content-Length": str(len(raw))}))
        manifest, received = subject.fetch_pinned_manifest(pin, opener=opener, timeout=1)
        self.assertEqual(received, raw)
        self.assertEqual(manifest["repo_key"], "test")

        tampered = raw[:-1] + (b" " if raw[-1:] != b" " else b"\n")
        with self.assertRaisesRegex(subject.ManifestError, "SHA-256 mismatch"):
            subject.fetch_pinned_manifest(
                pin,
                opener=FakeOpener(FakeResponse(tampered)),
                timeout=1,
            )

    def test_current_license_fetch_has_independent_size_and_hash_pin(self) -> None:
        body = b"license"
        original_spec = subject.CURRENT_LICENSE_SPEC
        original_pin = subject.CURRENT_LICENSE_PIN
        try:
            subject.CURRENT_LICENSE_SPEC = make_spec(body)
            subject.CURRENT_LICENSE_PIN = digest(body)
            result = subject.fetch_current_license(
                opener=FakeOpener(FakeResponse(body)), timeout=1
            )
            self.assertEqual(result, body)
            with self.assertRaisesRegex(subject.VerificationError, "SHA-256 mismatch"):
                subject.fetch_current_license(
                    opener=FakeOpener(FakeResponse(b"licensf")), timeout=1
                )
        finally:
            subject.CURRENT_LICENSE_SPEC = original_spec
            subject.CURRENT_LICENSE_PIN = original_pin

    def test_preflight_requests_one_byte_and_never_authorization(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin, make_spec(b"abc"))
        opener = FakeOpener(
            FakeResponse(
                b"a",
                status=206,
                headers={"Content-Range": "bytes 0-0/3", "Content-Length": "1"},
            )
        )
        detail = subject.probe_small_asset(pin, manifest, opener=opener, timeout=1)
        self.assertIn("HTTP 206", detail)
        request = opener.requests[0]
        self.assertEqual(request.get_header("Range"), "bytes=0-0")
        self.assertIsNone(request.get_header("Authorization"))

    def test_https_redirect_handler_rejects_plaintext(self) -> None:
        handler = subject.HttpsOnlyRedirectHandler()
        request = urllib.request.Request("https://github.com/start")
        with self.assertRaisesRegex(urllib.error.HTTPError, "non-HTTPS"):
            handler.redirect_request(
                request, None, 302, "Found", {}, "http://example.test/object"
            )

    def test_resume_requires_exact_206_range_and_appends(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"abc")
        identity = subject._shard_identity(pin, "1" * 64, shard)
        subject._atomic_json(
            metadata,
            {**identity, "validator_kind": "etag", "validator": '"same"'},
        )
        opener = FakeOpener(
            FakeResponse(
                b"def",
                status=206,
                headers={
                    "Content-Range": "bytes 3-5/6",
                    "Content-Length": "3",
                    "ETag": '"same"',
                },
            )
        )
        subject.download_shard(
            pin,
            "1" * 64,
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"abcdef")
        self.assertEqual(opener.requests[0].get_header("Range"), "bytes=3-")
        self.assertEqual(opener.requests[0].get_header("If-range"), '"same"')
        self.assertIsNone(opener.requests[0].get_header("Authorization"))

    def test_ignored_range_restarts_private_partial_instead_of_appending(self) -> None:
        pin = make_pin()
        shard = make_spec(b"uvwxyz")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"abc")
        identity = subject._shard_identity(pin, "2" * 64, shard)
        subject._atomic_json(metadata, identity)
        subject.download_shard(
            pin,
            "2" * 64,
            shard,
            partial,
            metadata,
            opener=FakeOpener(
                FakeResponse(b"uvwxyz", status=200, headers={"Content-Length": "6"})
            ),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"uvwxyz")

    def test_bad_resume_range_never_changes_existing_partial(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"abc")
        subject._atomic_json(metadata, subject._shard_identity(pin, "3" * 64, shard))
        with self.assertRaisesRegex(subject.DownloadError, "unsafe resume Content-Range"):
            subject.download_shard(
                pin,
                "3" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    FakeResponse(
                        b"def",
                        status=206,
                        headers={"Content-Range": "bytes 2-5/6", "Content-Length": "3"},
                    )
                ),
                attempts=1,
                timeout=1,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), b"abc")

    def test_fixed_range_mode_uses_exact_chunks_and_stable_if_range(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        opener = FakeOpener(
            range_response(b"ab", start=0, end=1, total=6),
            range_response(b"cd", start=2, end=3, total=6),
            range_response(b"ef", start=4, end=5, total=6),
        )
        subject.download_shard(
            pin,
            "a" * 64,
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=1,
            range_chunk_bytes=2,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"abcdef")
        self.assertEqual(
            [request.get_header("Range") for request in opener.requests],
            ["bytes=0-1", "bytes=2-3", "bytes=4-5"],
        )
        self.assertIsNone(opener.requests[0].get_header("If-range"))
        self.assertEqual(opener.requests[1].get_header("If-range"), '"stable"')
        self.assertEqual(opener.requests[2].get_header("If-range"), '"stable"')
        self.assertTrue(
            all(request.get_header("Authorization") is None for request in opener.requests)
        )
        self.assertFalse(list(self.root.glob(".phosphene-range-*.chunk")))

    def test_fixed_range_mode_resumes_from_arbitrary_partial_size(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"abc")
        subject._atomic_json(
            metadata,
            {
                **subject._shard_identity(pin, "b" * 64, shard),
                "validator_kind": "etag",
                "validator": '"stable"',
            },
        )
        opener = FakeOpener(
            range_response(b"de", start=3, end=4, total=6),
            range_response(b"f", start=5, end=5, total=6),
        )
        subject.download_shard(
            pin,
            "b" * 64,
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=1,
            range_chunk_bytes=2,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"abcdef")
        self.assertEqual(
            [request.get_header("Range") for request in opener.requests],
            ["bytes=3-4", "bytes=5-5"],
        )
        self.assertEqual(opener.requests[0].get_header("If-range"), '"stable"')

    def test_parallel_ranges_finish_out_of_order_but_append_in_order(self) -> None:
        pin = make_pin()
        body = b"abcdefgh"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        opener = RangeMapOpener(
            body,
            delays={(0, 1): 0.06, (2, 3): 0.01, (4, 5): 0.04, (6, 7): 0.02},
        )
        append_order: list[int] = []
        original_append = subject._append_range_chunk

        def recording_append(source, download_path, *, offset, chunk_bytes):
            append_order.append(offset)
            return original_append(
                source,
                download_path,
                offset=offset,
                chunk_bytes=chunk_bytes,
            )

        with mock.patch.object(subject, "_append_range_chunk", side_effect=recording_append):
            subject.download_shard(
                pin,
                "f" * 64,
                shard,
                partial,
                metadata,
                opener=opener,
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                range_workers=4,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), body)
        self.assertNotEqual(opener.completed, sorted(opener.completed))
        self.assertEqual(append_order, [0, 2, 4, 6])
        self.assertFalse(list(self.root.glob(".phosphene-range-*.chunk")))

    def test_parallel_ranges_resume_at_arbitrary_offset(self) -> None:
        pin = make_pin()
        body = b"abcdefghi"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"abc")
        subject._atomic_json(
            metadata,
            {
                **subject._shard_identity(pin, "0" * 64, shard),
                "validator_kind": "etag",
                "validator": '"stable"',
            },
        )
        opener = RangeMapOpener(body)
        subject.download_shard(
            pin,
            "0" * 64,
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=1,
            range_chunk_bytes=2,
            range_workers=3,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), body)
        requested = [request.get_header("Range") for request in opener.requests]
        self.assertEqual(set(requested), {"bytes=3-4", "bytes=5-6", "bytes=7-8"})
        self.assertTrue(
            all(request.get_header("If-range") == '"stable"' for request in opener.requests)
        )

    def test_parallel_validator_mismatch_rejects_entire_batch_before_append(self) -> None:
        pin = make_pin()
        body = b"abcdef"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        opener = RangeMapOpener(body, validators={(2, 3): '"changed"'})
        with self.assertRaisesRegex(subject.DownloadError, "within parallel batch"):
            subject.download_shard(
                pin,
                "1" * 64,
                shard,
                partial,
                metadata,
                opener=opener,
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                range_workers=3,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(list(self.root.glob(".phosphene-range-*.chunk")))

    def test_parallel_failed_chunk_cancels_batch_without_append_or_temp_leak(self) -> None:
        pin = make_pin()
        body = b"abcdef"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        opener = RangeMapOpener(
            body,
            delays={(0, 1): 0.03, (4, 5): 0.03},
            failures={(2, 3)},
        )
        with self.assertRaisesRegex(subject.DownloadError, "range 2-3 failed"):
            subject.download_shard(
                pin,
                "2" * 64,
                shard,
                partial,
                metadata,
                opener=opener,
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                range_workers=3,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(list(self.root.glob(".phosphene-range-*.chunk")))

    def test_parallel_keyboard_interrupt_cleans_batch_without_append(self) -> None:
        pin = make_pin()
        body = b"abcdef"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"

        class InterruptingOpener(RangeMapOpener):
            def open(self, request: urllib.request.Request, timeout: int):
                if request.get_header("Range") == "bytes=2-3":
                    raise KeyboardInterrupt()
                return super().open(request, timeout)

        with self.assertRaises(KeyboardInterrupt):
            subject.download_shard(
                pin,
                "4" * 64,
                shard,
                partial,
                metadata,
                opener=InterruptingOpener(body, delays={(0, 1): 0.02, (4, 5): 0.02}),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                range_workers=3,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(list(self.root.glob(".phosphene-range-*.chunk")))

    def test_parallel_missing_chunk_is_rejected_before_append(self) -> None:
        pin = make_pin()
        body = b"abcd"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        with mock.patch.object(subject, "_fetch_range_batch", return_value=[]):
            with self.assertRaisesRegex(subject.DownloadError, "did not return every"):
                subject.download_shard(
                    pin,
                    "3" * 64,
                    shard,
                    partial,
                    metadata,
                    opener=RangeMapOpener(body),
                    attempts=1,
                    timeout=1,
                    range_chunk_bytes=2,
                    range_workers=2,
                    report=lambda _message: None,
                )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_fixed_range_mode_rejects_wrong_content_range_without_append(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"ab")
        subject._atomic_json(
            metadata,
            {
                **subject._shard_identity(pin, "c" * 64, shard),
                "validator_kind": "etag",
                "validator": '"stable"',
            },
        )
        with self.assertRaisesRegex(subject.DownloadError, "unsafe range Content-Range"):
            subject.download_shard(
                pin,
                "c" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    range_response(b"cd", start=1, end=2, total=6)
                ),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), b"ab")

    def test_fixed_range_mode_rejects_changed_validator_without_append(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        partial.write_bytes(b"ab")
        subject._atomic_json(
            metadata,
            {
                **subject._shard_identity(pin, "d" * 64, shard),
                "validator_kind": "etag",
                "validator": '"first"',
            },
        )
        with self.assertRaisesRegex(subject.DownloadError, "validator changed"):
            subject.download_shard(
                pin,
                "d" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    range_response(
                        b"cd", start=2, end=3, total=6, validator='"second"'
                    )
                ),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), b"ab")
        self.assertEqual(
            subject._read_json(metadata),
            {
                **subject._shard_identity(pin, "d" * 64, shard),
                "validator_kind": "etag",
                "validator": '"first"',
            },
        )

    def test_fixed_range_mode_rejects_short_chunk_body_without_append(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        with self.assertRaisesRegex(subject.DownloadError, "range body ended"):
            subject.download_shard(
                pin,
                "e" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    range_response(
                        b"a", start=0, end=1, total=6, content_length=2
                    )
                ),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_fixed_range_mode_rejects_malformed_content_length_without_append(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        response = range_response(b"ab", start=0, end=1, total=6)
        response.headers["content-length"] = "two"
        with self.assertRaisesRegex(subject.DownloadError, "malformed Content-Length"):
            subject.download_shard(
                pin,
                "f" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(response),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_fixed_range_mode_rejects_ignored_range_without_append(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        with self.assertRaisesRegex(subject.DownloadError, "HTTP 200 instead of 206"):
            subject.download_shard(
                pin,
                "1" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    FakeResponse(
                        b"ab",
                        status=200,
                        headers={"Content-Length": "2", "ETag": '"stable"'},
                    )
                ),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_fixed_range_mode_still_requires_whole_shard_hash(self) -> None:
        pin = make_pin()
        shard = make_spec(b"abcdef")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        with self.assertRaisesRegex(subject.VerificationError, "shard SHA-256 mismatch"):
            subject.download_shard(
                pin,
                "2" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    range_response(b"ab", start=0, end=1, total=6),
                    range_response(b"XX", start=2, end=3, total=6),
                    range_response(b"ef", start=4, end=5, total=6),
                ),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_curl_fixed_range_transport_has_exact_header_and_no_credentials(self) -> None:
        pin = make_pin()
        body = b"abcd"
        shard = make_spec(body)["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        commands: list[list[str]] = []
        lock = threading.Lock()

        def fake_run(command, *, stdout, check):
            self.assertFalse(check)
            with lock:
                commands.append(command)
            range_header = next(
                item for item in command if item.startswith("Range: bytes=")
            )
            start_text, end_text = range_header.removeprefix("Range: bytes=").split("-")
            start, end = int(start_text), int(end_text)
            chunk = body[start : end + 1]
            stdout.write(chunk)
            headers_path = Path(command[command.index("--dump-header") + 1])
            headers_path.write_bytes(
                b"HTTP/1.1 302 Found\r\nLocation: https://release-assets.example/object\r\n\r\n"
                + (
                    f"HTTP/2 206\r\nContent-Range: bytes {start}-{end}/{len(body)}\r\n"
                    f"Content-Length: {len(chunk)}\r\nETag: \"stable\"\r\n\r\n"
                ).encode()
            )
            return subject.subprocess.CompletedProcess(command, 0)

        with mock.patch.object(subject.subprocess, "run", side_effect=fake_run):
            subject.download_shard(
                pin,
                "3" * 64,
                shard,
                partial,
                metadata,
                opener=subject.CurlOpener("/usr/bin/curl", force_ipv6=True),
                attempts=1,
                timeout=1,
                range_chunk_bytes=2,
                range_workers=2,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), body)
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            {
                item
                for command in commands
                for item in command
                if item.startswith("Range: bytes=")
            },
            {"Range: bytes=0-1", "Range: bytes=2-3"},
        )
        for command in commands:
            self.assertEqual(command[:2], ["/usr/bin/curl", "--disable"])
            self.assertIn("--ipv6", command)
            self.assertEqual(command[command.index("--max-time") + 1], "1")
            self.assertNotIn("Authorization:", " ".join(command))
            self.assertNotIn("--insecure", command)
            self.assertNotIn("-k", command)

    def test_hash_mismatch_discards_only_private_shard(self) -> None:
        pin = make_pin()
        shard = make_spec(b"good")["shards"][0]
        partial = self.root / "asset.partial"
        metadata = self.root / "asset.partial.json"
        with self.assertRaisesRegex(subject.DownloadError, "SHA-256 mismatch"):
            subject.download_shard(
                pin,
                "4" * 64,
                shard,
                partial,
                metadata,
                opener=FakeOpener(
                    FakeResponse(b"evil", headers={"Content-Length": "4"})
                ),
                attempts=1,
                timeout=1,
                report=lambda _message: None,
            )
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())

    def test_recovery_hashes_complete_shards_and_truncates_torn_tail(self) -> None:
        pin = make_pin()
        spec = make_spec(b"abc", b"def")
        assembly = self.root / "blob.assembling"
        metadata = self.root / "blob.assembling.json"
        assembly.write_bytes(b"abcde")
        identity = subject._assembly_identity(pin, "5" * 64, "blob.bin", spec)
        subject._atomic_json(metadata, identity)
        completed = subject.recover_assembly(
            assembly, metadata, identity, spec["shards"]
        )
        self.assertEqual(completed, 1)
        self.assertEqual(assembly.read_bytes(), b"abc")

    def test_invalid_destination_is_preserved_until_full_file_verifies(self) -> None:
        pin = make_pin()
        spec = make_spec(b"abc", b"def")
        destination = self.root / pin.directory / "blob.bin"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"OLD")

        def still_old(_request) -> None:
            self.assertEqual(destination.read_bytes(), b"OLD")

        opener = FakeOpener(
            FakeResponse(b"abc", headers={"Content-Length": "3"}),
            FakeResponse(b"def", headers={"Content-Length": "3"}),
            before_open=still_old,
        )
        changed = subject.install_file(
            pin,
            "6" * 64,
            "blob.bin",
            spec,
            models_root=self.root,
            opener=opener,
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        self.assertTrue(changed)
        self.assertEqual(destination.read_bytes(), b"abcdef")

    def test_full_file_mismatch_does_not_overwrite_existing_destination(self) -> None:
        pin = make_pin()
        spec = make_spec(b"abc", b"def", file_sha="0" * 64)
        destination = self.root / pin.directory / "blob.bin"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"OLD")
        with self.assertRaisesRegex(subject.VerificationError, "assembled SHA-256"):
            subject.install_file(
                pin,
                "7" * 64,
                "blob.bin",
                spec,
                models_root=self.root,
                opener=FakeOpener(
                    FakeResponse(b"abc", headers={"Content-Length": "3"}),
                    FakeResponse(b"def", headers={"Content-Length": "3"}),
                ),
                attempts=1,
                timeout=1,
                report=lambda _message: None,
            )
        self.assertEqual(destination.read_bytes(), b"OLD")

    def test_verified_destination_skips_network(self) -> None:
        pin = make_pin()
        spec = make_spec(b"complete")
        destination = self.root / pin.directory / "blob.bin"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"complete")
        changed = subject.install_file(
            pin,
            "8" * 64,
            "blob.bin",
            spec,
            models_root=self.root,
            opener=FakeOpener(),
            attempts=1,
            timeout=1,
            report=lambda _message: None,
        )
        self.assertFalse(changed)

    def test_parallel_pack_install_has_bounded_unique_file_owners(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin)
        for index in range(7):
            manifest["files"][f"extra-{index}.bin"] = make_spec(bytes([index + 1]))
        destination = self.root / pin.directory
        destination.mkdir(parents=True)
        active = 0
        maximum_active = 0
        seen: list[str] = []
        lock = threading.Lock()

        def fake_install_file(_pin, _manifest_digest, name, _spec, **_kwargs):
            nonlocal active, maximum_active
            with lock:
                self.assertNotIn(name, seen)
                seen.append(name)
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.015)
            with lock:
                active -= 1
            return True

        with mock.patch.object(subject, "install_file", side_effect=fake_install_file):
            subject.install_pack(
                pin,
                manifest,
                "9" * 64,
                models_root=self.root,
                opener=FakeOpener(),
                attempts=1,
                timeout=1,
                jobs=3,
                report=lambda _message: None,
            )
        expected = set(subject.pack_files(manifest)) | {subject.LICENSE_NAME}
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), len(expected))
        self.assertGreaterEqual(maximum_active, 2)
        self.assertLessEqual(maximum_active, 3)

    def test_metadata_only_install_skips_weights_and_writes_no_receipt(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin)
        manifest["files"]["model.safetensors"] = make_spec(b"weights")
        destination = self.root / pin.directory
        destination.mkdir(parents=True)
        seen: list[str] = []

        def fake_install_file(_pin, _manifest_digest, name, _spec, **_kwargs):
            seen.append(name)
            return True

        with mock.patch.object(subject, "install_file", side_effect=fake_install_file):
            subject.install_pack(
                pin,
                manifest,
                "a" * 64,
                models_root=self.root,
                opener=FakeOpener(),
                attempts=1,
                timeout=1,
                jobs=1,
                metadata_only=True,
                report=lambda _message: None,
            )

        self.assertEqual(set(seen), {"blob.bin", subject.LICENSE_NAME})
        self.assertNotIn("model.safetensors", seen)
        self.assertFalse(
            (destination / ".phosphene-install-receipt.json").exists()
        )

    def test_metadata_only_space_plan_excludes_safetensors(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin)
        manifest["files"]["model.safetensors"] = make_spec(b"large-weight-file")
        manifests = {pin.key: manifest}
        full = subject.estimate_required_space(
            [pin], manifests, models_root=self.root, reserve_bytes=0, jobs=1
        )
        metadata = subject.estimate_required_space(
            [pin],
            manifests,
            models_root=self.root,
            reserve_bytes=0,
            jobs=1,
            metadata_only=True,
        )
        self.assertEqual(full - metadata, len(b"large-weight-file"))

    def test_parallel_space_plan_charges_one_transient_shard_per_worker(self) -> None:
        pin = make_pin()
        manifest = make_manifest(pin, make_spec(b"abc"))
        manifests = {pin.key: manifest}
        one = subject.estimate_required_space(
            [pin], manifests, models_root=self.root, reserve_bytes=10, jobs=1
        )
        four = subject.estimate_required_space(
            [pin], manifests, models_root=self.root, reserve_bytes=10, jobs=4
        )
        # Largest synthetic shard is 34,545-byte current license.
        self.assertEqual(four - one, 3 * int(subject.CURRENT_LICENSE_SPEC["bytes"]))

        ranged = subject.estimate_required_space(
            [pin],
            manifests,
            models_root=self.root,
            reserve_bytes=10,
            jobs=4,
            range_chunk_bytes=64,
        )
        self.assertEqual(ranged - four, 4 * 64)

        serial_ranges = subject.estimate_required_space(
            [pin],
            manifests,
            models_root=self.root,
            reserve_bytes=10,
            jobs=1,
            range_chunk_bytes=64,
            range_workers=1,
        )
        parallel_ranges = subject.estimate_required_space(
            [pin],
            manifests,
            models_root=self.root,
            reserve_bytes=10,
            jobs=1,
            range_chunk_bytes=64,
            range_workers=8,
        )
        self.assertEqual(parallel_ranges - serial_ranges, 7 * 64)

    def test_range_worker_cli_defaults_and_bounds(self) -> None:
        defaults = subject._parse_args(["--verify"])
        self.assertEqual(defaults.range_workers, 1)
        valid = subject._parse_args(
            [
                "--plan",
                "--jobs",
                "1",
                "--range-chunk-mib",
                "64",
                "--range-workers",
                "8",
            ]
        )
        self.assertEqual(valid.range_workers, 8)

        invalid_argv = [
            ["--plan", "--range-workers", "0"],
            ["--plan", "--range-workers", "17", "--range-chunk-mib", "64"],
            ["--plan", "--range-workers", "2"],
            [
                "--plan",
                "--jobs",
                "4",
                "--range-chunk-mib",
                "64",
                "--range-workers",
                "8",
            ],
        ]
        for argv in invalid_argv:
            with self.subTest(argv=argv), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    subject._parse_args(argv)

    def test_metadata_only_cli_requires_install_and_license_acceptance(self) -> None:
        valid = subject._parse_args(
            ["--install", "--accept-license", "--metadata-only"]
        )
        self.assertTrue(valid.install)
        self.assertTrue(valid.metadata_only)

        invalid_argv = [
            ["--verify", "--metadata-only"],
            ["--plan", "--metadata-only"],
            ["--install", "--metadata-only"],
        ]
        for argv in invalid_argv:
            with self.subTest(argv=argv), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    subject._parse_args(argv)

    @unittest.skipUnless(
        os.environ.get("PHOSPHENE_LIVE_TEST") == "1",
        "set PHOSPHENE_LIVE_TEST=1 for the 101-byte GitHub integration test",
    )
    def test_live_system_curl_fresh_and_resumed_tiny_asset(self) -> None:
        pin = subject.PACKS["q4_25"]
        opener = subject.build_opener()
        manifest, raw = subject.fetch_pinned_manifest(pin, opener=opener, timeout=30)
        shard = manifest["files"]["quantize_config.json"]["shards"][0]
        partial = self.root / "tiny.partial"
        metadata = self.root / "tiny.partial.json"
        subject.download_shard(
            pin,
            digest(raw),
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=30,
            report=lambda _message: None,
        )
        complete = partial.read_bytes()
        self.assertEqual(len(complete), 101)
        partial.write_bytes(complete[:37])
        subject.download_shard(
            pin,
            digest(raw),
            shard,
            partial,
            metadata,
            opener=opener,
            attempts=1,
            timeout=30,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), complete)


if __name__ == "__main__":
    unittest.main()
