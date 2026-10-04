"""Unit and GPU-free integration tests for edit_ltx25.py."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path

import edit_ltx25 as edit


class EditLtx25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _file(self, name: str, content: bytes = b"not-empty") -> Path:
        path = self.root / name
        path.write_bytes(content)
        return path

    @staticmethod
    def _video_info(path: Path, *, frames: int = 49, fps: float = 24.0) -> edit.MediaInfo:
        return edit.MediaInfo(
            path=path,
            duration=frames / fps,
            has_video=True,
            has_audio=True,
            width=768,
            height=448,
            frame_rate=fps,
            num_frames=frames,
        )

    def test_long_prompt_common_validation_preserves_exact_text(self) -> None:
        prompt = "  START " + ("continue the motion " * 2_800) + "EDIT-END  "
        self.assertGreater(len(prompt), 50_000)
        args = edit.parse_args(["retake", prompt, "--video", str(self.root / "source.mp4")])

        edit._validate_common(args)

        self.assertEqual(args.prompt, prompt)

    def test_a2v_plan_is_local_low_ram_and_two_stage(self) -> None:
        audio = self._file("music.wav")
        args = edit.parse_args(
            [
                "a2v",
                "rhythmic abstract motion",
                "--audio",
                str(audio),
                "--profile",
                "quality",
                "--frames",
                "97",
                "--dry-run",
                "--output",
                "test/a2v.mp4",
            ]
        )
        plan = edit.build_plan(
            args,
            media_probe=lambda path: edit.MediaInfo(path, 8.0, False, True),
            now=datetime(2026, 1, 2, 3, 4, 5),
            pid=99,
        )
        command = list(plan.command)
        self.assertEqual(command[1], "a2v")
        self.assertIn("--low-ram", command)
        self.assertEqual(command[command.index("--model") + 1], str(edit.HQ_MODEL))
        self.assertEqual(command[command.index("--gemma") + 1], str(edit.GEMMA_MODEL))
        self.assertEqual(command[command.index("--frames") + 1], "97")
        self.assertEqual(command[command.index("--height") + 1], "448")
        self.assertEqual(command[command.index("--width") + 1], "768")
        self.assertTrue(edit._is_within(plan.output, edit.OUTPUT_ROOT))
        self.assertIn(".partial.mp4", str(plan.temporary_output))

    def test_a2v_rejects_audio_shorter_than_requested_clip(self) -> None:
        audio = self._file("music.wav")
        args = edit.parse_args(
            ["a2v", "motion", "--audio", str(audio), "--frames", "97", "--dry-run"]
        )
        with self.assertRaisesRegex(edit.UserInputError, "need 4.042s"):
            edit.build_plan(args, media_probe=lambda path: edit.MediaInfo(path, 2.0, False, True))

    def test_a2v_clip3s_profile_reuses_73_frame_low_cost_settings(self) -> None:
        audio = self._file("music.wav")
        args = edit.parse_args(
            ["a2v", "motion", "--audio", str(audio), "--profile", "clip3s", "--dry-run"]
        )
        plan = edit.build_plan(
            args,
            media_probe=lambda path: edit.MediaInfo(path, 4.0, False, True),
            pid=98,
        )
        command = list(plan.command)
        self.assertEqual(command[command.index("--frames") + 1], "73")
        self.assertEqual(command[command.index("--stage1-steps") + 1], "30")
        self.assertEqual(command[command.index("--stg-scale") + 1], "1.0")

    def test_a2v_vertical_aspect_uses_backend_derived_safe_dimensions(self) -> None:
        audio = self._file("music.wav")
        args = edit.parse_args(
            [
                "a2v",
                "vertical performance",
                "--audio",
                str(audio),
                "--profile",
                "quality",
                "--aspect-ratio",
                "9:16",
                "--dry-run",
            ]
        )
        plan = edit.build_plan(
            args,
            media_probe=lambda path: edit.MediaInfo(path, 4.0, False, True),
            pid=97,
        )
        command = list(plan.command)
        self.assertEqual(command[command.index("--width") + 1], "448")
        self.assertEqual(command[command.index("--height") + 1], "768")
        self.assertTrue(any("requested 9:16 aspect" in line for line in plan.summary))

    def test_a2v_rejects_non_temporal_grid_frame_count(self) -> None:
        audio = self._file("music.wav")
        args = edit.parse_args(
            ["a2v", "motion", "--audio", str(audio), "--frames", "96", "--dry-run"]
        )
        with self.assertRaisesRegex(edit.UserInputError, "8k\+1"):
            edit.build_plan(args, media_probe=lambda path: edit.MediaInfo(path, 10.0, False, True))

    def test_retake_seconds_are_converted_to_exclusive_latent_range(self) -> None:
        video = self._file("source.mp4")
        args = edit.parse_args(
            [
                "retake",
                "replace the middle action",
                "--video",
                str(video),
                "--start-seconds",
                "0.5",
                "--end-seconds",
                "1.0",
                "--dry-run",
            ]
        )
        plan = edit.build_plan(args, media_probe=self._video_info, pid=100)
        command = list(plan.command)
        self.assertIn("--low-ram", command)
        self.assertEqual(command[command.index("--start") + 1], "1")
        self.assertEqual(command[command.index("--end") + 1], "4")

    def test_extend_defaults_to_one_three_second_pass(self) -> None:
        video = self._file("source.mp4")
        args = edit.parse_args(
            ["extend", "continue the movement", "--video", str(video), "--dry-run"]
        )
        plan = edit.build_plan(args, media_probe=self._video_info, pid=101)
        command = list(plan.command)
        self.assertIn("--low-ram", command)
        self.assertEqual(command[command.index("--extend-frames") + 1], "9")
        self.assertTrue(any("result 121 frames" in line for line in plan.summary))
        self.assertTrue(any("~3.00 s" in line for line in plan.summary))

    def test_extend_explicit_amount_overrides_three_second_default(self) -> None:
        video = self._file("source.mp4")
        args = edit.parse_args(
            [
                "extend",
                "continue the movement",
                "--video",
                str(video),
                "--extend-frames",
                "2",
                "--dry-run",
            ]
        )
        plan = edit.build_plan(args, media_probe=self._video_info, pid=102)
        command = list(plan.command)
        self.assertEqual(command[command.index("--extend-frames") + 1], "2")
        self.assertTrue(any("result 65 frames" in line for line in plan.summary))

    def test_extend_explicit_target_frames_is_preserved(self) -> None:
        video = self._file("source.mp4")
        args = edit.parse_args(
            [
                "extend",
                "continue the movement",
                "--video",
                str(video),
                "--target-frames",
                "73",
                "--dry-run",
            ]
        )
        plan = edit.build_plan(args, media_probe=self._video_info, pid=103)
        command = list(plan.command)
        self.assertEqual(command[command.index("--extend-frames") + 1], "3")
        self.assertTrue(any("result 73 frames" in line for line in plan.summary))

    def test_output_escape_is_rejected(self) -> None:
        video = self._file("source.mp4")
        args = edit.parse_args(
            [
                "extend",
                "continue",
                "--video",
                str(video),
                "--extend-frames",
                "1",
                "--output",
                "../../escape.mp4",
                "--dry-run",
            ]
        )
        with self.assertRaisesRegex(edit.UserInputError, "must stay inside"):
            edit.build_plan(args, media_probe=self._video_info)

    def test_input_extension_is_validated_case_insensitively(self) -> None:
        video = self._file("source.MP4")
        args = edit.parse_args(
            [
                "extend",
                "continue",
                "--video",
                str(video),
                "--extend-frames",
                "1",
                "--dry-run",
            ]
        )
        edit.build_plan(args, media_probe=self._video_info)

        bad = self._file("source.txt")
        bad_args = edit.parse_args(
            [
                "extend",
                "continue",
                "--video",
                str(bad),
                "--extend-frames",
                "1",
                "--dry-run",
            ]
        )
        with self.assertRaisesRegex(edit.UserInputError, "unsupported extension"):
            edit.build_plan(bad_args, media_probe=self._video_info)

    def test_a2v_asset_preflight_names_exact_overlay_files(self) -> None:
        missing = edit.missing_mode_assets("a2v", model=self.root)
        self.assertEqual(
            missing,
            (
                str(self.root / ".ltx25-q8-overlay-receipt.json"),
                str(self.root / "transformer-dev.safetensors"),
                str(self.root / "transformer-distilled.safetensors"),
            ),
        )
        self._file(".ltx25-q8-overlay-receipt.json", b"{}")
        dev = self._file("transformer-dev.safetensors")
        os.truncate(dev, edit.HQ_DEV_SIZE)
        distilled = self._file("transformer-distilled.safetensors")
        os.truncate(distilled, edit.HQ_DISTILLED_SIZE)
        self.assertEqual(edit.missing_mode_assets("a2v", model=self.root), ())
        distilled.unlink()
        self.assertEqual(edit.missing_mode_assets("retake", model=self.root), ())

    def test_offline_environment_is_explicit_and_does_not_mutate_process(self) -> None:
        before = os.environ.get("HF_HUB_OFFLINE")
        env = edit.offline_environment()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(
            env["PYTHONPATH"],
            str(edit.RUNTIME / "packages" / "ltx-pipelines-mlx" / "src"),
        )
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), before)

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe not installed")
    def test_real_audio_dry_run_does_not_need_gpu_or_complete_model(self) -> None:
        audio = self.root / "tone.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16_000 * 2)
        result = edit.main(
            [
                "a2v",
                "simple motion",
                "--audio",
                str(audio),
                "--profile",
                "draft",
                "--dry-run",
                "--output",
                "tests/dry-run.mp4",
            ]
        )
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
