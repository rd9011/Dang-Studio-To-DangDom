from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_ltx25 as launcher


class GenerateLtx25FeatureTests(unittest.TestCase):
    def parse(self, *arguments: str):
        with patch.object(sys, "argv", ["generate_ltx25.py", *arguments]):
            return launcher._parse_args()

    def parse_error(self, *arguments: str) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.parse(*arguments)
        self.assertEqual(raised.exception.code, 2)

    def validate(self, args):
        return launcher._validate_args(args)

    def build(self, args):
        profile, schedule_preset, conditionings = self.validate(args)
        return launcher._build_command(
            args,
            profile=profile,
            schedule_preset=schedule_preset,
            output=Path("/tmp/ltx25-wrapper-test.mp4"),
            conditionings=conditionings,
        )

    def test_long_prompt_survives_wrapper_validation_and_command_exactly(self):
        prompt = "  START " + ("cinematic motion " * 3_200) + "WRAPPER-END  "
        self.assertGreater(len(prompt), 50_000)
        args = self.parse(prompt)
        command = self.build(args)

        self.assertEqual(args.prompt, prompt)
        self.assertEqual(command[command.index("--prompt") + 1], prompt)
        logged = launcher.format_command_for_log(command)
        self.assertNotIn(prompt, logged)
        self.assertIn(f"{len(prompt):,} chars", logged)
        self.assertIn("sha256:", logged)

    def test_scene_lengths_attach_to_preceding_scene(self):
        args = self.parse(
            "global",
            "--scene",
            " opening ",
            "--scene-length",
            "2",
            "--scene",
            "middle",
            "--scene",
            "closing",
            "--scene-length",
            "2",
        )

        command = self.build(args)

        self.assertEqual(
            args.scenes,
            [
                launcher.SceneBeat("opening", 2),
                launcher.SceneBeat("middle", None),
                launcher.SceneBeat("closing", 2),
            ],
        )
        self.assertEqual(args.resolved_scene_lengths, [2, 3, 2])
        self.assertEqual(
            command[-9:],
            [
                "--segment",
                "opening",
                "2",
                "--segment",
                "middle",
                "3",
                "--segment",
                "closing",
                "2",
            ],
        )

    def test_default_profiles_emit_distilled_split_pack_contract(self):
        for profile_name, profile in launcher.PROFILES.items():
            with self.subTest(profile=profile_name):
                args = self.parse("global", "--profile", profile_name)
                command = self.build(args)
                self.assertNotIn("--segment", command)
                self.assertNotIn("--num-generated-keyframes", command)
                self.assertNotIn("--one-stage", command)
                self.assertNotIn("--cfg-scale", command)
                self.assertNotIn("--stg-scale", command)
                self.assertNotIn("--steps", command)
                self.assertIn("--distilled", command)
                self.assertIn("--low-ram", command)
                self.assertEqual(command[command.index("--model") + 1], str(launcher.MODEL))
                self.assertEqual(command[command.index("--gemma") + 1], str(launcher.GEMMA_MODEL))
                self.assertEqual(command[command.index("--frames") + 1], str(profile.frames))
                self.assertEqual(
                    command[command.index("--schedule-preset") + 1], profile.schedule_preset
                )

    def test_alternate_gemma_requires_explicit_selector(self):
        official = self.build(self.parse("global"))
        fallback = self.build(self.parse("global", "--gemma-pack", "ddalcu-fallback"))
        self.assertEqual(official[official.index("--gemma") + 1], str(launcher.GEMMA_MODEL))
        self.assertEqual(
            fallback[fallback.index("--gemma") + 1],
            str(launcher.DDALCU_FALLBACK_GEMMA_MODEL),
        )
        self.assertNotEqual(official[official.index("--gemma") + 1], fallback[fallback.index("--gemma") + 1])

    def test_explicit_zero_generated_keyframes_is_an_exact_no_op(self):
        implicit = self.build(self.parse("global"))
        explicit = self.build(self.parse("global", "--generated-keyframes", "0"))
        self.assertEqual(explicit, implicit)

    def test_schedule_override_and_legacy_steps_map_to_distilled_presets(self):
        fast = self.build(self.parse("global", "--schedule-preset", "fast"))
        legacy_fast = self.build(self.parse("global", "--steps", "5"))
        graded = self.build(self.parse("global", "--steps", "8"))
        self.assertEqual(fast[fast.index("--schedule-preset") + 1], "fast")
        self.assertEqual(legacy_fast[legacy_fast.index("--schedule-preset") + 1], "fast")
        self.assertEqual(graded[graded.index("--schedule-preset") + 1], "default")
        with self.assertRaisesRegex(SystemExit, "select different distilled schedules"):
            self.validate(self.parse("global", "--steps", "5", "--schedule-preset", "default"))

    def test_cfg_and_stg_are_rejected_for_distilled_inference(self):
        with self.assertRaisesRegex(SystemExit, "distilled checkpoint"):
            self.validate(self.parse("global", "--cfg-scale", "3"))
        with self.assertRaisesRegex(SystemExit, "distilled checkpoint"):
            self.validate(self.parse("global", "--stg-scale", "1"))

    def test_scene_length_must_follow_one_scene_and_cannot_repeat(self):
        self.parse_error("global", "--scene-length", "1")
        self.parse_error("global", "--scene", "one", "--scene-length", "1", "--scene-length", "1")
        self.parse_error("global", "--scene", "one", "--scene-length", "0")

    def test_generated_keyframe_choices_are_strict(self):
        self.parse_error("global", "--generated-keyframes", "3")
        self.parse_error("global", "--generated-keyframes", "-1")
        for count in (1, 2):
            with self.subTest(count=count):
                with self.assertRaisesRegex(SystemExit, "not exposed by the pinned distilled runtime"):
                    self.validate(self.parse("global", "--generated-keyframes", str(count)))

    def test_empty_scene_is_rejected(self):
        args = self.parse("global", "--scene", "   ")
        with self.assertRaisesRegex(SystemExit, "--scene #1 cannot be empty"):
            self.validate(args)

    def test_auto_scene_lengths_are_balanced_and_fully_explicit_upstream(self):
        expected = {
            "draft": [2, 2, 1],
            "balanced": [2, 2, 2],
            "quality": [3, 2, 2],
            "max": [2, 1, 1],
            "clip3s": [4, 3, 3],
        }
        for profile_name, lengths in expected.items():
            with self.subTest(profile=profile_name):
                args = self.parse(
                    "global",
                    "--profile",
                    profile_name,
                    "--scene",
                    "one",
                    "--scene",
                    "two",
                    "--scene",
                    "three",
                )
                command = self.build(args)
                self.assertEqual(args.resolved_scene_lengths, lengths)
                segment_positions = [i for i, value in enumerate(command) if value == "--segment"]
                self.assertEqual([int(command[i + 2]) for i in segment_positions], lengths)

    def test_clip3s_profile_is_conservative_and_ltx_frame_aligned(self):
        profile = launcher.PROFILES["clip3s"]
        self.assertEqual(
            (profile.width, profile.height, profile.frames, profile.schedule_preset),
            (512, 256, 73, "default"),
        )
        self.assertEqual((profile.frames - 1) % 8, 0)
        self.assertAlmostEqual(profile.frames / 24, 3.0416666667)

        args = self.parse("global", "--profile", "clip3s")
        command = self.build(args)
        self.assertEqual(command[command.index("--frames") + 1], "73")
        self.assertEqual(command[command.index("--width") + 1], "512")
        self.assertEqual(command[command.index("--height") + 1], "256")
        self.assertEqual(command[command.index("--schedule-preset") + 1], "default")

    def test_duration_control_maps_one_to_ten_seconds_to_exact_ltx_grid(self):
        for seconds in range(1, 11):
            with self.subTest(seconds=seconds):
                args = self.parse("global", "--profile", "quality", "--duration-seconds", str(seconds))
                profile, *_ = self.validate(args)
                command = self.build(args)
                expected_frames = 24 * seconds + 1
                self.assertEqual(profile.frames, expected_frames)
                self.assertEqual((profile.frames - 1) % 8, 0)
                self.assertEqual(command[command.index("--frames") + 1], str(expected_frames))
                # Duration changes length only; the selected quality/resolution
                # profile still controls the spatial and denoising settings.
                self.assertEqual((profile.width, profile.height), (768, 448))

    def test_duration_control_bounds_are_strict(self):
        self.parse_error("global", "--duration-seconds", "0")
        self.parse_error("global", "--duration-seconds", "11")
        self.parse_error("global", "--duration-seconds", "2.5")

    def test_aspect_presets_preserve_profile_budget_on_safe_spatial_grid(self):
        for profile_name, base in launcher.PROFILES.items():
            for aspect in ("16:9", "9:16", "1:1", "4:3", "3:4"):
                with self.subTest(profile=profile_name, aspect=aspect):
                    width, height = launcher.resolve_aspect_dimensions(
                        base.width,
                        base.height,
                        aspect,
                    )
                    self.assertEqual(width % 64, 0)
                    self.assertEqual(height % 64, 0)
                    self.assertGreaterEqual(width * height, base.width * base.height * 0.8)
                    self.assertLessEqual(width * height, base.width * base.height * 1.2)
                    if aspect in {"16:9", "4:3"}:
                        self.assertGreater(width, height)
                    elif aspect in {"9:16", "3:4"}:
                        self.assertLess(width, height)
                    else:
                        self.assertEqual(width, height)

    def test_explicit_vertical_aspect_reorients_quality_canvas(self):
        args = self.parse("global", "--profile", "quality", "--aspect-ratio", "9:16")
        profile, *_ = self.validate(args)
        command = self.build(args)
        self.assertEqual((profile.width, profile.height), (448, 768))
        self.assertEqual(command[command.index("--width") + 1], "448")
        self.assertEqual(command[command.index("--height") + 1], "768")

    def test_profile_aspect_is_the_backward_compatible_default(self):
        args = self.parse("global", "--profile", "quality")
        profile, *_ = self.validate(args)
        self.assertEqual(args.aspect_ratio, "profile")
        self.assertEqual((profile.width, profile.height), (768, 448))

    def test_long_clip_warning_describes_optional_extend_without_stale_unavailable_claim(self):
        self.assertIn("Q8 Extend lane", launcher.LONG_CLIP_WARNING)
        self.assertIn("installed and ready", launcher.LONG_CLIP_WARNING)
        self.assertNotIn("remains unavailable", launcher.LONG_CLIP_WARNING)

    def test_prompt_relay_uses_duration_override_latent_timeline(self):
        args = self.parse(
            "global",
            "--duration-seconds",
            "10",
            "--scene",
            "opening",
            "--scene",
            "middle",
            "--scene",
            "closing",
        )
        profile, *_ = self.validate(args)
        self.assertEqual(profile.frames, 241)
        self.assertEqual(args.resolved_scene_lengths, [11, 10, 10])

    def test_scene_that_cannot_receive_one_latent_frame_is_rejected(self):
        args = self.parse(
            "global",
            "--profile",
            "max",
            "--scene",
            "one",
            "--scene",
            "two",
            "--scene",
            "three",
            "--scene",
            "four",
            "--scene",
            "five",
        )
        with self.assertRaisesRegex(SystemExit, "--scene #5 resolves to zero latent frames"):
            self.validate(args)

    def test_explicit_scene_length_that_would_be_clipped_is_rejected(self):
        args = self.parse(
            "global",
            "--scene",
            "one",
            "--scene-length",
            "4",
            "--scene",
            "two",
            "--scene-length",
            "4",
        )
        with self.assertRaisesRegex(SystemExit, "scene lengths require 8 latent frames"):
            self.validate(args)

    def test_programmatic_legacy_namespace_needs_no_new_attributes(self):
        args = self.parse("global", "--profile", "draft")
        del args.scenes
        del args.generated_keyframes
        command = self.build(args)
        self.assertNotIn("--segment", command)
        self.assertNotIn("--num-generated-keyframes", command)


if __name__ == "__main__":
    unittest.main()
