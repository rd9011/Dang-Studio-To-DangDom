import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pinned_source import PROVENANCE_NAME, PROVENANCE_SCHEMA, source_tree_digest


MODULE_PATH = Path(__file__).with_name("ai_video_upscale.py")
SPEC = importlib.util.spec_from_file_location("ai_video_upscale", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DimensionsTests(unittest.TestCase):
    def test_portrait_uses_exact_short_edge_and_even_long_edge(self):
        self.assertEqual(MODULE.calculate_target_dimensions(448, 768, 1080), (1080, 1852))

    def test_landscape_preserves_ratio_on_even_grid(self):
        self.assertEqual(MODULE.calculate_target_dimensions(512, 320, 1080), (1728, 1080))
        self.assertEqual(MODULE.calculate_target_dimensions(384, 256, 1080), (1620, 1080))

    def test_rejects_non_upscale_and_unapproved_target(self):
        with self.assertRaises(ValueError):
            MODULE.calculate_target_dimensions(1920, 1080, 1080)
        with self.assertRaises(ValueError):
            MODULE.calculate_target_dimensions(448, 768, 720)


class ProbeTests(unittest.TestCase):
    def test_parses_exact_rational_frame_rate_count_duration_and_audio(self):
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 448,
                    "height": 768,
                    "avg_frame_rate": "24000/1001",
                    "r_frame_rate": "24/1",
                    "nb_frames": "73",
                    "nb_read_frames": "73",
                    "duration": "3.044708",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "3.044708"},
        }
        info = MODULE.parse_probe_payload(payload)
        self.assertEqual(info.fps, "24000/1001")
        self.assertEqual(info.frame_count, 73)
        self.assertEqual((info.width, info.height), (448, 768))
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 3.044708)

    def test_falls_back_to_r_frame_rate_and_format_duration(self):
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 512,
                    "height": 320,
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "24/1",
                    "nb_read_frames": "72",
                    "duration": "N/A",
                }
            ],
            "format": {"duration": "3.0"},
        }
        info = MODULE.parse_probe_payload(payload)
        self.assertEqual(info.fps, "24/1")
        self.assertEqual(info.duration, 3.0)
        self.assertFalse(info.has_audio)


class CommandTests(unittest.TestCase):
    def test_probe_requests_counted_frames(self):
        command = MODULE.build_probe_command("/tools/ffprobe", Path("source.mp4"))
        self.assertEqual(command[0], "/tools/ffprobe")
        self.assertIn("-count_frames", command)
        self.assertEqual(command[-1], "source.mp4")

    def test_decoder_streams_raw_rgb_without_shell_syntax(self):
        command = MODULE.build_decoder_command("ffmpeg", Path("a b.mp4"))
        self.assertEqual(command[-1], "pipe:1")
        self.assertIn("rgb24", command)
        self.assertIn("passthrough", command)
        self.assertNotIn("-y", command)
        self.assertIn("a b.mp4", command)

    def test_encoder_preserves_rational_fps_and_copies_optional_audio(self):
        command = MODULE.build_encoder_command(
            "ffmpeg",
            Path("source.mp4"),
            Path("new.mp4"),
            1080,
            1852,
            "24000/1001",
            73,
        )
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-framerate") + 1], "24000/1001")
        self.assertEqual(command[command.index("-r") + 1], "24000/1001")
        self.assertEqual(command[command.index("-video_size") + 1], "1080x1852")
        self.assertIn("1:a?", command)
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(command[command.index("-crf") + 1], "18")
        self.assertEqual(command[command.index("-pix_fmt", 10) + 1], "yuv420p")
        self.assertIn("+faststart", command)
        self.assertEqual(command[-1], "new.mp4")


class StreamingHelpersTests(unittest.TestCase):
    def test_read_exact_frame_handles_partial_stream_reads(self):
        class PartialReader(io.BytesIO):
            def read(self, amount=-1):
                return super().read(min(amount, 2))

        stream = PartialReader(b"abcdef")
        self.assertEqual(MODULE.read_exact_frame(stream, 6), b"abcdef")
        self.assertIsNone(MODULE.read_exact_frame(stream, 6))

    def test_read_exact_frame_rejects_truncated_frame(self):
        with self.assertRaises(MODULE.UpscaleError):
            MODULE.read_exact_frame(io.BytesIO(b"abc"), 6)

    def test_write_all_handles_short_writes(self):
        class ShortWriter:
            def __init__(self):
                self.data = bytearray()

            def write(self, payload):
                portion = bytes(payload[:2])
                self.data.extend(portion)
                return len(portion)

        writer = ShortWriter()
        MODULE.write_all(writer, b"abcdef")
        self.assertEqual(bytes(writer.data), b"abcdef")

    def test_atomic_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / ".partial.mp4"
            output = root / "output.mp4"
            partial.write_bytes(b"new")
            output.write_bytes(b"old")
            with self.assertRaises(MODULE.UpscaleError):
                MODULE.publish_without_overwrite(partial, output)
            self.assertEqual(output.read_bytes(), b"old")
            self.assertEqual(partial.read_bytes(), b"new")

    def test_atomic_publish_links_then_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / ".partial.mp4"
            output = root / "output.mp4"
            partial.write_bytes(b"complete")
            MODULE.publish_without_overwrite(partial, output)
            self.assertFalse(partial.exists())
            self.assertEqual(output.read_bytes(), b"complete")


class InstalledPreflightTests(unittest.TestCase):
    def test_pinned_install_preflight_is_ready_without_importing_coreml(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            upscaler = root / "runtime"
            upscaler.mkdir()
            upstream_script = upscaler / "upscale.py"
            upstream_script.write_text(
                "# Minimal source-only preflight fixture.\n"
                "def get_mlpackage_path(*_args):\n"
                "    raise AssertionError('preflight must not import this module')\n",
                encoding="utf-8",
            )

            fixture_commit = "f" * 40
            source_pin = MODULE.SourcePin(
                name="real-esrgan-coreml-test",
                repository="https://example.invalid/real-esrgan-coreml-test.git",
                revision=fixture_commit,
                archive_url="https://example.invalid/source.tar.gz",
                archive_sha256="a" * 64,
                archive_bytes=1,
                archive_root="source-test",
                tree_sha256=source_tree_digest(upscaler),
                includes=("upscale.py",),
            )
            provenance = {
                "schema": PROVENANCE_SCHEMA,
                "name": source_pin.name,
                "repository": source_pin.repository,
                "revision": source_pin.revision,
                "archive_url": source_pin.archive_url,
                "archive_sha256": source_pin.archive_sha256,
                "archive_bytes": source_pin.archive_bytes,
                "tree_sha256": source_pin.tree_sha256,
                "includes": list(source_pin.includes),
            }
            (upscaler / PROVENANCE_NAME).write_text(
                json.dumps(provenance), encoding="utf-8"
            )

            venv = upscaler / ".venv"
            fixture_python = venv / "bin/python"
            fixture_python.parent.mkdir(parents=True)
            fixture_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fixture_python.chmod(0o755)

            model = upscaler / "weights/Test.mlpackage"
            manifest = model / "Manifest.json"
            model_file = model / "Data/com.apple.CoreML/model.mlmodel"
            weight_file = model / "Data/com.apple.CoreML/weights/weight.bin"
            manifest.parent.mkdir(parents=True)
            model_file.parent.mkdir(parents=True)
            weight_file.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "fileFormatVersion": "1.0.0",
                        "rootModelIdentifier": "root",
                        "itemInfoEntries": {
                            "root": {"path": "com.apple.CoreML/model.mlmodel"},
                            "weights": {"path": "com.apple.CoreML/weights"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            model_file.write_bytes(b"model fixture")
            weight_file.write_bytes(b"weight fixture")
            model_pins = tuple(
                MODULE.FilePin(
                    path.relative_to(model).as_posix(),
                    path.stat().st_size,
                    MODULE.sha256_file(path),
                )
                for path in (manifest, model_file, weight_file)
            )

            tools = root / "tools"
            tools.mkdir()
            ffmpeg = tools / "ffmpeg"
            ffprobe = tools / "ffprobe"
            for executable in (ffmpeg, ffprobe):
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)

            previous_coreml = sys.modules.pop("coremltools", None)
            output = io.StringIO()
            try:
                with (
                    mock.patch.object(MODULE, "UPSCALER_ROOT", upscaler),
                    mock.patch.object(MODULE, "UPSTREAM_SCRIPT", upstream_script),
                    mock.patch.object(MODULE, "UPSCALER_VENV", venv),
                    mock.patch.object(MODULE, "MODEL_PACKAGE", model),
                    mock.patch.object(MODULE, "UPSTREAM_SOURCE_PIN", source_pin),
                    mock.patch.object(MODULE, "EXPECTED_UPSTREAM_COMMIT", fixture_commit),
                    mock.patch.object(
                        MODULE,
                        "EXPECTED_UPSTREAM_SCRIPT_SHA256",
                        MODULE.sha256_file(upstream_script),
                    ),
                    mock.patch.object(MODULE, "MODEL_FILE_PINS", model_pins),
                    mock.patch.object(MODULE, "EXPECTED_RUNTIME", {"fixture-runtime": "1.0"}),
                    mock.patch.object(MODULE.platform, "system", return_value="Darwin"),
                    mock.patch.object(MODULE.platform, "machine", return_value="arm64"),
                    mock.patch.object(MODULE.sys, "prefix", str(venv)),
                    mock.patch.object(
                        MODULE.importlib.metadata, "version", return_value="1.0"
                    ) as version,
                    contextlib.redirect_stdout(output),
                ):
                    return_code = MODULE.main(
                        [
                            "--preflight",
                            "--ffmpeg",
                            str(ffmpeg),
                            "--ffprobe",
                            str(ffprobe),
                        ]
                    )
                self.assertEqual(return_code, 0)
                self.assertNotIn("coremltools", sys.modules)
                version.assert_called_once_with("fixture-runtime")
            finally:
                sys.modules.pop("coremltools", None)
                if previous_coreml is not None:
                    sys.modules["coremltools"] = previous_coreml

        events = [
            json.loads(line) for line in output.getvalue().splitlines() if line.strip()
        ]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["phase"], "preflight")
        self.assertTrue(event["ready"])
        self.assertFalse(event["coreml_imported"])
        self.assertEqual(event["compute_unit"], "CPU_AND_GPU")
        self.assertEqual(event["upstream_commit"], fixture_commit)


if __name__ == "__main__":
    unittest.main()
