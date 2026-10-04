from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import condition_ltx25 as subject


def _write_safetensors_header(path: Path, tensors: dict[str, list[int]]) -> None:
    """Write a tiny, structurally sufficient safetensors fixture."""
    offset = 0
    header: dict[str, object] = {"__metadata__": {"fixture": "true"}}
    for key, shape in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        byte_count = elements * 2
        header[key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + byte_count]}
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))


class ConditionLTX25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output_root = self.root / "outputs"
        self.output_patch = mock.patch.object(subject, "OUTPUT_ROOT", self.output_root)
        self.output_patch.start()
        self.image_a = self.root / "a.png"
        self.image_b = self.root / "b.jpg"
        Image.new("RGB", (80, 120), (220, 30, 40)).save(self.image_a)
        Image.new("RGB", (160, 90), (20, 100, 230)).save(self.image_b)
        self.video = self.root / "guide.mp4"
        self.video.write_bytes(b"test fixture path only")

    def tearDown(self) -> None:
        self.output_patch.stop()
        self.temporary.cleanup()

    def test_long_control_prompt_survives_plan_and_command_exactly(self) -> None:
        prompt = "  START " + ("preserve guided motion " * 2_500) + "CONTROL-END  "
        self.assertGreater(len(prompt), 50_000)
        args = subject.parse_args(
            ["union", prompt, "--control-type", "canny", "--guide-video", str(self.video)]
        )

        plan = subject.build_plan(args)

        self.assertEqual(plan.prompt, prompt)
        self.assertEqual(plan.command[plan.command.index("--prompt") + 1], prompt)

    def test_ingredients_defaults_to_duration_compliant_safe_profile_and_mixes_anchors(self) -> None:
        args = subject.parse_args(
            [
                "ingredients",
                "The pilot enters the ship",
                "--reference",
                str(self.image_a),
                "pilot",
                "front-facing pilot in an orange suit",
                "--reference",
                str(self.image_b),
                "ship",
                "blue compact spacecraft",
                "--start-image",
                str(self.image_a),
                "--end-image",
                str(self.image_b),
            ]
        )
        plan = subject.build_plan(args)
        self.assertEqual(plan.profile.frames, 121)
        self.assertEqual((plan.profile.width, plan.profile.height), (384, 224))
        self.assertIn("Reference sheet:", plan.prompt)
        self.assertIn("Generated video: The pilot enters the ship", plan.prompt)
        self.assertEqual([item.frame for item in plan.image_conditionings], [0, 120])
        self.assertEqual(plan.command.count("--image"), 2)
        self.assertIn("--single-stage", plan.command)
        self.assertIn("--low-ram", plan.command)
        self.assertEqual(plan.command[plan.command.index("--model") + 1], str(subject.HQ_MODEL))
        self.assertEqual(plan.command[plan.command.index("--gemma") + 1], str(subject.GEMMA_MODEL))
        self.assertEqual(plan.command[plan.command.index("--seed") + 1], "1234")

    def test_seed_is_forwarded_to_ic_lora(self) -> None:
        args = subject.parse_args(
            [
                "ingredients",
                "walk",
                "--seed",
                "987654",
                "--reference",
                str(self.image_a),
                "person",
                "front view",
            ]
        )
        plan = subject.build_plan(args)
        self.assertEqual(plan.command[plan.command.index("--seed") + 1], "987654")

    def test_short_ingredients_is_refused_instead_of_silently_truncating_reference(self) -> None:
        args = subject.parse_args(
            [
                "ingredients",
                "walk",
                "--profile",
                "control3s",
                "--reference",
                str(self.image_a),
                "person",
                "red jacket",
            ]
        )
        with self.assertRaisesRegex(subject.UserInputError, "at least 121"):
            subject.build_plan(args)

    def test_union_source_auto_preprocessing_is_honestly_canny_only(self) -> None:
        args = subject.parse_args(
            ["union", "follow the pose", "--control-type", "depth", "--source-video", str(self.video)]
        )
        with self.assertRaisesRegex(subject.UserInputError, "Canny only"):
            subject.build_plan(args)

        prepared = subject.parse_args(
            ["union", "follow the pose", "--control-type", "depth", "--guide-video", str(self.video)]
        )
        plan = subject.build_plan(prepared)
        self.assertEqual(plan.profile.frames, 73)
        self.assertEqual(plan.source, self.video.resolve())

    def test_union_and_motion_duration_override_reaches_ten_seconds_on_ltx_grid(self) -> None:
        for seconds in range(1, 11):
            with self.subTest(seconds=seconds):
                args = subject.parse_args(
                    [
                        "union",
                        "follow structure",
                        "--guide-video",
                        str(self.video),
                        "--duration-seconds",
                        str(seconds),
                    ]
                )
                plan = subject.build_plan(args)
                self.assertEqual(plan.profile.frames, 24 * seconds + 1)
                self.assertEqual((plan.profile.frames - 1) % 8, 0)
                self.assertEqual(plan.command[plan.command.index("--frames") + 1], str(24 * seconds + 1))

    def test_ingredients_duration_remains_locked_to_required_bucket(self) -> None:
        bad = subject.parse_args(
            [
                "ingredients",
                "walk",
                "--duration-seconds",
                "3",
                "--reference",
                str(self.image_a),
                "person",
                "front view",
            ]
        )
        with self.assertRaisesRegex(subject.UserInputError, "fixed at 121 frames"):
            subject.build_plan(bad)

        good = subject.parse_args(
            [
                "ingredients",
                "walk",
                "--duration-seconds",
                "5",
                "--reference",
                str(self.image_a),
                "person",
                "front view",
            ]
        )
        self.assertEqual(subject.build_plan(good).profile.frames, 121)

    def test_ingredients_aspect_presets_are_experimental_and_keep_121_frame_lock(self) -> None:
        expected_dimensions = {
            "16:9": (384, 224),
            "9:16": (224, 384),
            "1:1": (288, 288),
            "4:3": (352, 256),
            "3:4": (256, 352),
        }
        for aspect_ratio, dimensions in expected_dimensions.items():
            with self.subTest(aspect_ratio=aspect_ratio):
                args = subject.parse_args(
                    [
                        "ingredients",
                        "walk",
                        "--aspect-ratio",
                        aspect_ratio,
                        "--reference",
                        str(self.image_a),
                        "person",
                        "front view",
                    ]
                )
                plan = subject.build_plan(args)
                self.assertEqual((plan.profile.width, plan.profile.height), dimensions)
                self.assertEqual(plan.profile.frames, 121)
                self.assertIn("EXPERIMENTAL Ingredients", plan.profile.description)
                self.assertEqual(plan.command[plan.command.index("--frames") + 1], "121")

        default = subject.parse_args(
            [
                "ingredients",
                "walk",
                "--reference",
                str(self.image_a),
                "person",
                "front view",
            ]
        )
        default_plan = subject.build_plan(default)
        self.assertEqual((default_plan.profile.width, default_plan.profile.height), (384, 224))
        self.assertNotIn("EXPERIMENTAL", default_plan.profile.description)

    def test_ingredients_experimental_aspect_prints_backend_warning(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(subject.sys, "stderr", stderr):
            result = subject.main(
                [
                    "ingredients",
                    "walk",
                    "--aspect-ratio",
                    "9:16",
                    "--reference",
                    str(self.image_a),
                    "person",
                    "front view",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("EXPERIMENTAL INGREDIENTS ASPECT WARNING", stderr.getvalue())
        self.assertIn("Profile default", stderr.getvalue())

    def test_union_and_motion_aspect_presets_are_backend_derived(self) -> None:
        union = subject.parse_args(
            [
                "union",
                "follow structure",
                "--guide-video",
                str(self.video),
                "--profile",
                "control3s-plus",
                "--aspect-ratio",
                "9:16",
            ]
        )
        union_plan = subject.build_plan(union)
        self.assertEqual((union_plan.profile.width, union_plan.profile.height), (288, 512))
        self.assertEqual(union_plan.command[union_plan.command.index("--width") + 1], "288")
        self.assertEqual(union_plan.command[union_plan.command.index("--height") + 1], "512")

        tracks = self.root / "aspect-tracks.json"
        tracks.write_text(json.dumps([[{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.7}]]))
        motion = subject.parse_args(
            [
                "motion",
                "move vertically",
                "--start-image",
                str(self.image_a),
                "--tracks-json",
                str(tracks),
                "--aspect-ratio",
                "1:1",
            ]
        )
        motion_plan = subject.build_plan(motion)
        self.assertEqual((motion_plan.profile.width, motion_plan.profile.height), (320, 320))

        exact_landscape = subject.parse_args(
            [
                "union",
                "follow structure",
                "--guide-video",
                str(self.video),
                "--aspect-ratio",
                "4:3",
            ]
        )
        exact_portrait = subject.parse_args(
            [
                "motion",
                "move vertically",
                "--start-image",
                str(self.image_a),
                "--tracks-json",
                str(tracks),
                "--aspect-ratio",
                "3:4",
            ]
        )
        landscape_plan = subject.build_plan(exact_landscape)
        portrait_plan = subject.build_plan(exact_portrait)
        self.assertEqual(
            (landscape_plan.profile.width, landscape_plan.profile.height),
            (384, 288),
        )
        self.assertEqual(
            (portrait_plan.profile.width, portrait_plan.profile.height),
            (288, 384),
        )

    def test_motion_experimental_resolution_is_640_by_384_and_three_seconds_only(self) -> None:
        tracks = self.root / "experimental-tracks.json"
        tracks.write_text(json.dumps([[{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.7}]]))
        accepted = subject.parse_args(
            [
                "motion",
                "move through the tracked arc",
                "--profile",
                "motion3s-experimental",
                "--duration-seconds",
                "3",
                "--start-image",
                str(self.image_a),
                "--tracks-json",
                str(tracks),
            ]
        )
        accepted_plan = subject.build_plan(accepted)
        self.assertEqual((accepted_plan.profile.width, accepted_plan.profile.height), (640, 384))
        self.assertEqual(accepted_plan.profile.frames, 73)

        too_long = subject.parse_args(
            [
                "motion",
                "move through the tracked arc",
                "--profile",
                "motion3s-experimental",
                "--duration-seconds",
                "4",
                "--start-image",
                str(self.image_a),
                "--tracks-json",
                str(tracks),
            ]
        )
        with self.assertRaisesRegex(subject.UserInputError, "restricted to 3 seconds"):
            subject.build_plan(too_long)

        invalid_union = subject.parse_args(
            [
                "union",
                "follow the guide",
                "--profile",
                "motion3s-experimental",
                "--guide-video",
                str(self.video),
            ]
        )
        with self.assertRaisesRegex(subject.UserInputError, "Union requires"):
            subject.build_plan(invalid_union)

    def test_motion_normalized_tracks_and_opening_image_compile_to_command(self) -> None:
        tracks = self.root / "tracks.json"
        tracks.write_text(json.dumps([[{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.7}]]))
        args = subject.parse_args(
            [
                "motion",
                "the ball arcs right",
                "--start-image",
                str(self.image_a),
                "--tracks-json",
                str(tracks),
                "--start-strength",
                "0.8",
            ]
        )
        plan = subject.build_plan(args)
        self.assertEqual(plan.profile.frames, 73)
        self.assertEqual(plan.image_conditionings[0].strength, 0.8)
        self.assertEqual(len(plan.tracks), 1)
        image_index = plan.command.index("--image")
        self.assertEqual(plan.command[image_index + 2 : image_index + 4], ("0", "0.8"))

    def test_invalid_track_coordinate_is_rejected(self) -> None:
        tracks = self.root / "tracks.json"
        tracks.write_text(json.dumps([[{"x": 0, "y": 0}, {"x": 1.2, "y": 1}]]))
        args = subject.parse_args(
            ["motion", "move", "--start-image", str(self.image_a), "--tracks-json", str(tracks)]
        )
        with self.assertRaisesRegex(subject.UserInputError, "normalized"):
            subject.build_plan(args)

    def test_reference_sheet_preserves_canvas_size_and_multiple_panels(self) -> None:
        target = self.root / "sheet.png"
        refs = (
            subject.Reference(self.image_a, "a", "portrait"),
            subject.Reference(self.image_b, "b", "ship"),
        )
        subject.create_reference_sheet(refs, target, width=512, height=288)
        with Image.open(target) as result:
            self.assertEqual(result.size, (512, 288))
            self.assertNotEqual(result.getpixel((100, 144)), result.getpixel((400, 144)))

    def test_motion_interpolation_preserves_endpoints_and_age_colours(self) -> None:
        samples = subject.interpolate_track(((0.1, 0.2), (0.4, 0.9), (0.8, 0.3)), 73)
        self.assertEqual(len(samples), 73)
        self.assertAlmostEqual(samples[0][0], 0.1)
        self.assertAlmostEqual(samples[-1][0], 0.8)
        self.assertEqual(subject._age_colour(0.0), (255, 0, 0))
        self.assertEqual(subject._age_colour(1.0), (0, 0, 255))

    @unittest.skipUnless(subject.shutil.which("ffmpeg") and subject.shutil.which("ffprobe"), "ffmpeg unavailable")
    def test_motion_guide_encoder_writes_requested_frame_count(self) -> None:
        destination = self.root / "motion.mp4"
        profile = subject.Profile(64, 64, 9, 8, "test")
        subject.create_motion_guide(
            (((0.1, 0.1), (0.9, 0.9)),),
            destination,
            profile=profile,
            frame_directory=self.root / "frames",
        )
        result = subprocess.run(
            [
                subject.shutil.which("ffprobe") or "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "9")

    def test_output_escape_and_duplicate_anchor_frame_are_rejected(self) -> None:
        args = subject.parse_args(
            ["union", "test", "--guide-video", str(self.video), "--output", str(self.root / "escape.mp4")]
        )
        with self.assertRaisesRegex(subject.UserInputError, "must stay inside"):
            subject.build_plan(args)

        duplicate = subject.parse_args(
            [
                "union",
                "test",
                "--guide-video",
                str(self.video),
                "--start-image",
                str(self.image_a),
                "--anchor",
                str(self.image_b),
                "0",
                "0.5",
            ]
        )
        with self.assertRaisesRegex(subject.UserInputError, "unique"):
            subject.build_plan(duplicate)

    def test_asset_manifest_uses_current_official_revisions(self) -> None:
        self.assertEqual(subject.ASSETS["union"].revision, "b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff")
        self.assertEqual(subject.ASSETS["motion"].revision, "572bb9c9a1ba3d8e8724cce69783ffc2422386db")
        self.assertEqual(subject.ASSETS["ingredients"].size, 1_308_787_472)

    def test_missing_assets_accepts_distinct_distilled_control_receipt(self) -> None:
        model = self.root / "q8"
        controls = self.root / "controls"
        model.mkdir()
        controls.mkdir()
        control_receipt = model / "control-receipt.json"
        full_receipt = model / "full-receipt.json"
        transformer = model / "transformer-distilled.safetensors"
        adapter = controls / "ingredients.safetensors"
        control_receipt.write_text("{}", encoding="utf-8")
        transformer.write_bytes(b"transformer")
        adapter.write_bytes(b"adapter")
        fixture_asset = subject.Asset(
            filename=adapter.name,
            size=adapter.stat().st_size,
            sha256=None,
            repo="fixture/repo",
            revision="fixture",
        )
        with (
            mock.patch.object(subject, "CONTROL_RECEIPT", control_receipt),
            mock.patch.object(subject, "HQ_RECEIPT", full_receipt),
            mock.patch.object(subject, "DISTILLED_TRANSFORMER", transformer),
            mock.patch.object(subject, "DISTILLED_SIZE", transformer.stat().st_size),
            mock.patch.object(subject, "CONTROL_ROOT", controls),
            mock.patch.dict(subject.ASSETS, {"ingredients": fixture_asset}),
        ):
            self.assertEqual(subject.missing_mode_assets("ingredients"), ())

    def test_preflight_invokes_distilled_control_offline_verifier(self) -> None:
        with (
            mock.patch.object(subject, "verify_install") as verify_base,
            mock.patch.object(subject, "verify_installed_control_overlay") as verify_control,
            mock.patch.object(subject, "verify_control_assets") as verify_adapter,
            mock.patch.object(subject.shutil, "which", return_value="/usr/bin/tool"),
        ):
            result = subject.main(
                [
                    "ingredients",
                    "walk forward",
                    "--reference",
                    str(self.image_a),
                    "person",
                    "front view",
                    "--preflight-only",
                ]
            )
        self.assertEqual(result, 0)
        verify_base.assert_called_once_with(full_hash=False, require_gpu=True)
        verify_control.assert_called_once_with(full_hash=False)
        verify_adapter.assert_called_once()

    def test_lora_key_preflight_matches_runtime_prefix_and_comfy_remaps(self) -> None:
        adapter = self.root / "compatible.safetensors"
        _write_safetensors_header(
            adapter,
            {
                "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight": [4, 16],
                "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_B.weight": [32, 4],
                "transformer_blocks.3.ff.net.0.proj.lora_A.weight": [8, 32],
                "transformer_blocks.3.ff.net.0.proj.lora_B.weight": [64, 8],
                "unrelated.tensor": [1],
            },
        )

        coverage = subject.validate_lora_key_compatibility(adapter, label="test adapter")

        self.assertEqual(coverage.blocks, (0, 3))
        self.assertEqual(coverage.pairs, 2)
        self.assertEqual(coverage.recognized_tensors, 4)
        self.assertEqual(
            subject._normalize_lora_key(
                "diffusion_model.transformer_blocks.3.ff.net.0.proj.lora_A.weight"
            ),
            "transformer_blocks.3.ff.proj_in.lora_A.weight",
        )

    def test_lora_key_preflight_rejects_unpaired_or_rank_mismatched_targets(self) -> None:
        unpaired = self.root / "unpaired.safetensors"
        _write_safetensors_header(
            unpaired,
            {"diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": [4, 16]},
        )
        with self.assertRaisesRegex(subject.PreflightError, "unpaired transformer-block"):
            subject.validate_lora_key_compatibility(unpaired, label="test adapter")

        mismatched = self.root / "rank-mismatch.safetensors"
        _write_safetensors_header(
            mismatched,
            {
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": [4, 16],
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": [16, 8],
            },
        )
        with self.assertRaisesRegex(subject.PreflightError, "incompatible A/B ranks"):
            subject.validate_lora_key_compatibility(mismatched, label="test adapter")

    def test_ingredients_and_motion_asset_preflight_reject_runtime_incompatible_key_prefix(self) -> None:
        transformer = self.root / "transformer.safetensors"
        _write_safetensors_header(transformer, {"transformer.weight": [1]})
        adapter = self.root / "wrong-prefix.safetensors"
        _write_safetensors_header(
            adapter,
            {
                "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": [4, 16],
                "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight": [16, 4],
            },
        )
        plan = subject.RenderPlan(
            mode="ingredients",
            profile=subject.PROFILES["ingredients5s"],
            prompt="fixture",
            output=self.output_root / "fixture.mp4",
            temporary_output=self.output_root / ".fixture.mp4.part",
            adapter=adapter,
            guide_placeholder=self.root / "guide.mp4",
            command=(),
        )

        with (
            mock.patch.object(subject, "DISTILLED_TRANSFORMER", transformer),
            mock.patch.object(subject, "DISTILLED_SIZE", transformer.stat().st_size),
        ):
            for mode in ("ingredients", "motion"):
                with self.subTest(mode=mode):
                    asset = subject.Asset(
                        filename=adapter.name,
                        size=adapter.stat().st_size,
                        sha256=None,
                        repo="fixture/repo",
                        revision="fixture",
                    )
                    with mock.patch.dict(subject.ASSETS, {mode: asset}):
                        with self.assertRaisesRegex(subject.PreflightError, "no LoRA tensors compatible"):
                            subject.verify_control_assets(
                                subject.replace(plan, mode=mode),
                                full_hash=False,
                                report=lambda _message: None,
                            )


if __name__ == "__main__":
    unittest.main()
