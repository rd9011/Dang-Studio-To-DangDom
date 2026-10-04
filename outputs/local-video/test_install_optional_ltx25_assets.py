"""Mocked-network tests for install_optional_ltx25_assets.py."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import install_optional_ltx25_assets as subject


def _safetensors(data: bytes = b"abcdef") -> bytes:
    header = json.dumps(
        {"tensor": {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + data


class FakeResponse:
    def __init__(self, body: bytes, *, status: int, headers: dict[str, str]):
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: int):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class OptionalAssetInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _asset(
        self,
        key: str = "test",
        *,
        blob: bytes | None = None,
        sha256: str | None | object = object(),
        modelscope_repo: str | None = None,
    ) -> tuple[subject.Asset, bytes]:
        content = _safetensors() if blob is None else blob
        if not isinstance(sha256, (str, type(None))):
            digest: str | None = hashlib.sha256(content).hexdigest()
        else:
            digest = sha256
        return (
            subject.Asset(
                key=key,
                label=f"{key} asset",
                repo="owner/repo",
                revision="a" * 40,
                filename=f"{key}.safetensors",
                destination=self.root / "models" / f"{key}.safetensors",
                size=len(content),
                sha256=digest,
                modelscope_repo=modelscope_repo,
            ),
            content,
        )

    def test_manifest_exact_pins_and_wrapper_destinations(self) -> None:
        self.assertEqual(set(subject.ASSETS), {"union", "motion", "ingredients"})
        self.assertNotIn("a2v", subject.BUNDLES)
        self.assertNotIn("hq", subject.BUNDLES)

        union = subject.ASSETS["union"]
        self.assertEqual(union.revision, "b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff")
        self.assertEqual(union.filename, "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
        self.assertEqual(union.size, 654_465_352)
        self.assertEqual(union.sha256, "a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18")

        motion = subject.ASSETS["motion"]
        self.assertEqual(motion.revision, "572bb9c9a1ba3d8e8724cce69783ffc2422386db")
        self.assertEqual(motion.filename, "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors")
        self.assertEqual(motion.size, 327_309_314)
        self.assertEqual(motion.sha256, "e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca")
        self.assertEqual(
            motion.modelscope_repo,
            "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        )

        ingredients = subject.ASSETS["ingredients"]
        self.assertEqual(ingredients.revision, "12040e4091ac2008d3906a594e31a7fb1ab9d546")
        self.assertEqual(ingredients.filename, "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors")
        self.assertEqual(ingredients.size, 1_308_787_472)
        self.assertEqual(
            ingredients.sha256,
            "ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95",
        )
        self.assertEqual(
            ingredients.modelscope_repo,
            "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        )
        self.assertIsNone(union.modelscope_repo)
        for key in ("union", "motion", "ingredients"):
            self.assertEqual(subject.ASSETS[key].destination.parent, subject.CONTROL_ROOT)

    def test_bundle_expansion_is_sequential_and_deduplicated(self) -> None:
        assets = subject.selected_assets(["union", "motion", "union"])
        self.assertEqual(
            [asset.key for asset in assets],
            ["union", "motion"],
        )

    def test_cli_defaults_to_safe_auto_source_policy(self) -> None:
        args = subject._parse_args(["--bundle", "motion", "--plan"])
        self.assertEqual(args.source, subject.AUTO)

    def test_modelscope_urls_are_exact_https_official_paths(self) -> None:
        motion = subject.ASSETS["motion"]
        ingredients = subject.ASSETS["ingredients"]
        self.assertEqual(
            subject._download_url(motion, subject.MODELSCOPE),
            "https://www.modelscope.cn/models/Lightricks/"
            "LTX-2.3-22b-IC-LoRA-Motion-Track-Control/resolve/master/"
            "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        )
        self.assertEqual(
            subject._download_url(ingredients, subject.MODELSCOPE),
            "https://www.modelscope.cn/models/Lightricks/"
            "LTX-2.5-22b-IC-LoRA-Ingredients/resolve/master/"
            "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        )
        self.assertTrue(subject._download_url(motion, subject.MODELSCOPE).startswith("https://"))

    def test_modelscope_is_allowed_only_with_a_pinned_hash_and_declared_repo(self) -> None:
        pinned, _ = self._asset(modelscope_repo="Lightricks/test")
        self.assertEqual(
            subject._source_candidates(pinned, subject.AUTO),
            (subject.HUGGINGFACE, subject.MODELSCOPE),
        )
        unhashed, _ = self._asset(sha256=None, modelscope_repo="Lightricks/test")
        with self.assertRaisesRegex(subject.InstallerError, "pinned size and SHA-256"):
            subject._source_candidates(unhashed, subject.MODELSCOPE)
        with self.assertRaisesRegex(subject.InstallerError, "no ModelScope fallback"):
            subject._source_candidates(subject.ASSETS["union"], subject.MODELSCOPE)

    def test_source_partials_and_resume_metadata_are_isolated(self) -> None:
        asset, _ = self._asset(modelscope_repo="Lightricks/test")
        hf_partial, hf_metadata = subject._partial_paths(asset, subject.HUGGINGFACE)
        ms_partial, ms_metadata = subject._partial_paths(asset, subject.MODELSCOPE)
        self.assertNotEqual(hf_partial, ms_partial)
        self.assertNotEqual(hf_metadata, ms_metadata)
        ms_metadata.parent.mkdir(parents=True)
        subject._write_resume_metadata(ms_metadata, asset, '"modelscope-etag"', subject.MODELSCOPE)
        self.assertIsNone(subject._read_resume_metadata(ms_metadata, asset, subject.HUGGINGFACE))
        self.assertEqual(
            subject._read_resume_metadata(ms_metadata, asset, subject.MODELSCOPE)["source"],
            subject.MODELSCOPE,
        )

    def test_modelscope_request_never_receives_hugging_face_token(self) -> None:
        asset, blob = self._asset(modelscope_repo="Lightricks/test")
        opener = FakeOpener(
            FakeResponse(
                blob,
                status=200,
                headers={"Content-Length": str(len(blob)), "ETag": '"mirror-etag"'},
            )
        )
        subject.download_asset(
            asset,
            self.root / "modelscope.part",
            self.root / "modelscope.part.json",
            token="hf-secret-token",
            insecure_tls=False,
            source=subject.MODELSCOPE,
            opener=opener,
            report=lambda _message: None,
        )
        request = opener.requests[0]
        self.assertEqual(request.full_url, subject._download_url(asset, subject.MODELSCOPE))
        self.assertIsNone(request.get_header("Authorization"))

    def test_auto_fallback_keeps_sources_separate_until_verified_publish(self) -> None:
        asset, blob = self._asset(modelscope_repo="Lightricks/test")
        receipts = self.root / "receipts.json"
        calls: list[tuple[str, Path, str | None]] = []
        hf_partial, _ = subject._partial_paths(asset, subject.HUGGINGFACE)

        def downloader(_asset, partial, _metadata, **kwargs):
            source = kwargs["source"]
            calls.append((source, partial, kwargs["token"]))
            if source == subject.HUGGINGFACE:
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(b"x" * len(blob))
                return
            self.assertTrue(hf_partial.is_file(), "fallback must not overwrite the HF staging file")
            self.assertEqual(hf_partial.read_bytes(), b"x" * len(blob))
            partial.write_bytes(blob)

        subject.install_asset(
            asset,
            replace_invalid=False,
            insecure_tls=False,
            token="hf-secret-token",
            source=subject.AUTO,
            receipts_path=receipts,
            downloader=downloader,
            report=lambda _message: None,
        )
        self.assertEqual(
            calls,
            [
                (subject.HUGGINGFACE, subject._partial_paths(asset, subject.HUGGINGFACE)[0], "hf-secret-token"),
                (subject.MODELSCOPE, subject._partial_paths(asset, subject.MODELSCOPE)[0], None),
            ],
        )
        self.assertEqual(asset.destination.read_bytes(), blob)
        for candidate in (subject.HUGGINGFACE, subject.MODELSCOPE):
            partial, metadata = subject._partial_paths(asset, candidate)
            self.assertFalse(partial.exists())
            self.assertFalse(metadata.exists())

    def test_saved_token_discovery_prefers_environment_without_printing(self) -> None:
        token_dir = self.root / "hf"
        token_dir.mkdir()
        (token_dir / "token").write_text("file-secret\n", encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stdout):
            token = subject.discover_hf_token(
                environ={"HF_TOKEN": "env-secret", "HF_HOME": str(token_dir)},
                home=self.root,
            )
        self.assertEqual(token, "env-secret")
        self.assertEqual(stdout.getvalue(), "")

    def test_cross_origin_redirect_drops_authorization(self) -> None:
        handler = subject.SafeRedirectHandler()
        request = urllib.request.Request(
            "https://huggingface.co/owner/repo/resolve/rev/file",
            headers={"Authorization": "Bearer super-secret", "Range": "bytes=3-"},
        )
        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cdn-lfs.example.net/signed-object",
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(redirected.get_header("Range"), "bytes=3-")

    def test_same_origin_redirect_keeps_authorization(self) -> None:
        handler = subject.SafeRedirectHandler()
        request = urllib.request.Request(
            "https://huggingface.co/one",
            headers={"Authorization": "Bearer super-secret"},
        )
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://huggingface.co/two")
        self.assertEqual(redirected.get_header("Authorization"), "Bearer super-secret")

    def test_relative_https_redirect_is_resolved_without_losing_authorization(self) -> None:
        handler = subject.SafeRedirectHandler()
        request = urllib.request.Request(
            "https://huggingface.co/owner/repo/resolve/rev/file",
            headers={"Authorization": "Bearer super-secret"},
        )
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "/login-check")
        self.assertEqual(redirected.full_url, "https://huggingface.co/login-check")
        self.assertEqual(redirected.get_header("Authorization"), "Bearer super-secret")

    def test_resume_appends_only_for_exact_validator_backed_206(self) -> None:
        asset, _ = self._asset(blob=b"abcdef")
        partial = self.root / "partial"
        metadata = self.root / "partial.json"
        partial.write_bytes(b"abc")
        subject._write_resume_metadata(metadata, asset, '"etag-1"')
        opener = FakeOpener(
            FakeResponse(
                b"def",
                status=206,
                headers={
                    "Content-Range": "bytes 3-5/6",
                    "Content-Length": "3",
                    "ETag": '"etag-1"',
                },
            )
        )
        subject.download_asset(
            asset,
            partial,
            metadata,
            token="secret-token",
            insecure_tls=False,
            opener=opener,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"abcdef")
        request = opener.requests[0]
        self.assertEqual(request.get_header("Range"), "bytes=3-")
        self.assertEqual(request.get_header("If-range"), '"etag-1"')
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")

    def test_unsafe_partial_response_is_rejected_without_touching_bytes(self) -> None:
        asset, _ = self._asset(blob=b"abcdef")
        partial = self.root / "partial"
        metadata = self.root / "partial.json"
        partial.write_bytes(b"abc")
        subject._write_resume_metadata(metadata, asset, '"etag-1"')
        opener = FakeOpener(
            FakeResponse(
                b"def",
                status=206,
                headers={
                    "Content-Range": "bytes 2-5/6",
                    "Content-Length": "3",
                    "ETag": '"etag-1"',
                },
            )
        )
        with self.assertRaisesRegex(subject.DownloadError, "unsafe resume response"):
            subject.download_asset(
                asset,
                partial,
                metadata,
                token=None,
                insecure_tls=False,
                opener=opener,
                report=lambda _message: None,
            )
        self.assertEqual(partial.read_bytes(), b"abc")

    def test_range_fallback_200_restarts_instead_of_appending(self) -> None:
        asset, _ = self._asset(blob=b"uvwxyz")
        partial = self.root / "partial"
        metadata = self.root / "partial.json"
        partial.write_bytes(b"abc")
        subject._write_resume_metadata(metadata, asset, '"old"')
        opener = FakeOpener(
            FakeResponse(
                b"uvwxyz",
                status=200,
                headers={"Content-Length": "6", "ETag": '"new"'},
            )
        )
        subject.download_asset(
            asset,
            partial,
            metadata,
            token=None,
            insecure_tls=False,
            opener=opener,
            report=lambda _message: None,
        )
        self.assertEqual(partial.read_bytes(), b"uvwxyz")

    def test_restart_truncates_old_bytes_before_publishing_new_validator(self) -> None:
        asset, _ = self._asset(blob=b"uvwxyz")
        partial = self.root / "partial"
        metadata = self.root / "partial.json"
        partial.write_bytes(b"abc")
        subject._write_resume_metadata(metadata, asset, '"old"')
        opener = FakeOpener(
            FakeResponse(b"uvwxyz", status=200, headers={"Content-Length": "6", "ETag": '"new"'})
        )
        original = subject._write_resume_metadata
        observed: list[bytes] = []

        def checked_write(path, selected_asset, validator, source=subject.HUGGINGFACE):
            observed.append(partial.read_bytes())
            return original(path, selected_asset, validator, source)

        with mock.patch.object(subject, "_write_resume_metadata", side_effect=checked_write):
            subject.download_asset(
                asset,
                partial,
                metadata,
                token=None,
                insecure_tls=False,
                opener=opener,
                report=lambda _message: None,
            )
        self.assertEqual(observed, [b""])
        self.assertEqual(partial.read_bytes(), b"uvwxyz")

    def test_weak_etag_does_not_create_resumable_metadata(self) -> None:
        asset, _ = self._asset(blob=b"uvwxyz")
        partial = self.root / "partial"
        metadata = self.root / "partial.json"
        opener = FakeOpener(
            FakeResponse(b"uvwxyz", status=200, headers={"Content-Length": "6", "ETag": 'W/"weak"'})
        )
        subject.download_asset(
            asset,
            partial,
            metadata,
            token=None,
            insecure_tls=False,
            opener=opener,
            report=lambda _message: None,
        )
        self.assertFalse(metadata.exists())

    def test_partial_without_validator_is_never_resumed(self) -> None:
        asset, _ = self._asset(blob=b"uvwxyz")
        partial = self.root / "partial"
        metadata = self.root / "missing.json"
        partial.write_bytes(b"abc")
        opener = FakeOpener(
            FakeResponse(b"uvwxyz", status=200, headers={"Content-Length": "6", "ETag": '"new"'})
        )
        subject.download_asset(
            asset,
            partial,
            metadata,
            token=None,
            insecure_tls=False,
            opener=opener,
            report=lambda _message: None,
        )
        self.assertIsNone(opener.requests[0].get_header("Range"))
        self.assertEqual(partial.read_bytes(), b"uvwxyz")

    def test_gated_403_has_actionable_terms_and_login_message(self) -> None:
        asset = subject.ASSETS["ingredients"]
        error = urllib.error.HTTPError("https://huggingface.co", 403, "Forbidden", {}, None)
        opener = FakeOpener(error)
        with self.assertRaisesRegex(subject.DownloadError, "Accept the repository terms") as raised:
            subject.download_asset(
                asset,
                self.root / "partial",
                self.root / "partial.json",
                token=None,
                insecure_tls=False,
                opener=opener,
                report=lambda _message: None,
            )
        message = str(raised.exception)
        self.assertIn(asset.terms_url, message)
        self.assertIn("hf auth login", message)

    def test_insecure_hashed_ingredients_warning_retains_digest_guarantee(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            subject._warn_insecure([subject.ASSETS["ingredients"]])
        warning = stderr.getvalue()
        self.assertIn("intercept both", warning)
        self.assertIn("pinned size and SHA-256", warning)
        self.assertNotIn("not cryptographically authenticated", warning)

    def test_atomic_replacement_preserves_old_file_until_new_file_verifies(self) -> None:
        asset, blob = self._asset()
        asset.destination.parent.mkdir(parents=True)
        asset.destination.write_bytes(b"old-file")
        receipts = self.root / "receipts.json"

        def bad_downloader(_asset, partial, _metadata, **_kwargs):
            partial.write_bytes(b"x" * len(blob))

        with self.assertRaises(subject.VerificationError):
            subject.install_asset(
                asset,
                replace_invalid=True,
                insecure_tls=False,
                token=None,
                receipts_path=receipts,
                downloader=bad_downloader,
                report=lambda _message: None,
            )
        self.assertEqual(asset.destination.read_bytes(), b"old-file")

        def good_downloader(_asset, partial, _metadata, **_kwargs):
            partial.write_bytes(blob)

        subject.install_asset(
            asset,
            replace_invalid=True,
            insecure_tls=False,
            token=None,
            receipts_path=receipts,
            downloader=good_downloader,
            report=lambda _message: None,
        )
        self.assertEqual(asset.destination.read_bytes(), blob)
        receipt = json.loads(receipts.read_text(encoding="utf-8"))
        self.assertEqual(receipt["assets"][asset.key]["sha256"], hashlib.sha256(blob).hexdigest())

    def test_invalid_existing_file_requires_explicit_replace(self) -> None:
        asset, _ = self._asset()
        asset.destination.parent.mkdir(parents=True)
        asset.destination.write_bytes(b"invalid")
        with self.assertRaisesRegex(subject.InstallerError, "--replace-invalid"):
            subject.install_asset(
                asset,
                replace_invalid=False,
                insecure_tls=False,
                token=None,
                receipts_path=self.root / "receipt.json",
                downloader=lambda *_args, **_kwargs: self.fail("must not download"),
                report=lambda _message: None,
            )

    def test_ingredients_receipt_detects_later_same_size_corruption(self) -> None:
        first = _safetensors(b"abcdef")
        second = _safetensors(b"ghijkl")
        self.assertEqual(len(first), len(second))
        asset, _ = self._asset(key="ingredients-test", blob=first, sha256=None)
        asset.destination.parent.mkdir(parents=True)
        asset.destination.write_bytes(first)
        receipts_path = self.root / "receipt.json"
        digest = subject.verify_asset(asset, asset.destination)
        subject._record_receipt(asset, digest, path=receipts_path)
        asset.destination.write_bytes(second)
        receipts = subject._load_receipts(receipts_path)
        with self.assertRaisesRegex(subject.VerificationError, "SHA-256 mismatch"):
            subject.verify_asset(
                asset,
                asset.destination,
                receipt_sha256=subject._receipt_digest(asset, receipts),
            )

    def test_existing_unhashed_asset_gets_a_local_receipt_without_downloading(self) -> None:
        blob = _safetensors(b"abcdef")
        asset, _ = self._asset(key="ingredients-test", blob=blob, sha256=None)
        asset.destination.parent.mkdir(parents=True)
        asset.destination.write_bytes(blob)
        receipts = self.root / "receipt.json"
        subject.install_asset(
            asset,
            replace_invalid=False,
            insecure_tls=False,
            token=None,
            receipts_path=receipts,
            downloader=lambda *_args, **_kwargs: self.fail("must not download"),
            report=lambda _message: None,
        )
        payload = json.loads(receipts.read_text(encoding="utf-8"))
        self.assertEqual(payload["assets"][asset.key]["sha256"], hashlib.sha256(blob).hexdigest())

    def test_unresumable_partial_requires_full_transfer_but_reuses_its_disk(self) -> None:
        asset, _ = self._asset(blob=b"uvwxyz")
        partial, _metadata = subject._partial_paths(asset)
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"abc")
        state = subject.inspect_assets([asset], receipts_path=self.root / "none.json")[0]
        self.assertEqual(state.transfer_bytes, asset.size)
        self.assertEqual(state.disk_bytes, asset.size - 3)
        self.assertIn("will restart", state.detail)

    def test_plan_mode_reports_space_without_creating_targets(self) -> None:
        asset, _ = self._asset()
        states = subject.inspect_assets([asset], receipts_path=self.root / "none.json")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            transfer = subject.print_plan(states, bundles=["test"], free_bytes=asset.size + subject.RESERVE_BYTES)
        self.assertEqual(transfer, asset.size)
        self.assertIn("Disk preflight: READY", stdout.getvalue())
        self.assertFalse(asset.destination.parent.exists())

    def test_main_plan_is_read_only(self) -> None:
        asset, _ = self._asset()
        stdout = io.StringIO()
        disk = mock.Mock(free=asset.size + subject.RESERVE_BYTES)
        with (
            mock.patch.object(subject, "selected_assets", return_value=(asset,)),
            mock.patch.object(subject.shutil, "disk_usage", return_value=disk),
            redirect_stdout(stdout),
        ):
            result = subject.main(["--bundle", "controls", "--plan"])
        self.assertEqual(result, 0)
        self.assertIn("PLAN ONLY", stdout.getvalue())
        self.assertFalse(asset.destination.parent.exists())


if __name__ == "__main__":
    unittest.main()
