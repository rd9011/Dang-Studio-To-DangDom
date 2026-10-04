from __future__ import annotations

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compose_ltx_frame as composer
import install_frame_composer as installer
import ltx_ui as ui


PNG_1X1_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FrameComposerTests(unittest.TestCase):
    def test_long_prompt_is_not_truncated_by_frame_composer(self):
        prompt = "Use @maya. " + ("cinematic bathroom detail. " * 2_100) + "FRAME-END-SENTINEL"
        self.assertGreater(len(prompt), 50_000)
        compiled = composer.compile_tagged_prompt(
            prompt,
            [composer.Reference("@maya", Path("/tmp/maya.png"))],
        )
        self.assertIn("FRAME-END-SENTINEL", compiled)
        self.assertGreater(len(compiled), 50_000)

    def test_manifest_is_complete_and_uses_immutable_pins(self):
        self.assertEqual(installer.RUNTIME_COMMIT, "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3")
        self.assertEqual(installer.MODEL_REVISION, "4540c0084d0bcfdd29b19a2b0f62c77bef18f5bd")
        self.assertEqual(len(installer.MODEL_FILES), 14)
        self.assertEqual(installer.MODEL_TOTAL_BYTES, 4_619_712_832)
        self.assertEqual(len(installer.manifest_digest()), 64)
        self.assertEqual(len({item.path for item in installer.MODEL_FILES}), 14)
        for item in installer.MODEL_FILES:
            self.assertGreater(item.size, 0)
            self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")

    def test_runtime_install_rejects_dirty_existing_source_before_uv(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary) / "runtime"
            source = runtime_root / "mlx-gen"
            venv = runtime_root / ".venv"
            uv_cache = Path(temporary) / "uv-cache"
            source.mkdir(parents=True)
            with (
                patch.object(installer, "RUNTIME_ROOT", runtime_root),
                patch.object(installer, "SOURCE_DIR", source),
                patch.object(installer, "VENV_DIR", venv),
                patch.object(installer, "UV_CACHE_DIR", uv_cache),
                patch.object(installer, "_git_head", return_value=installer.RUNTIME_COMMIT),
                patch.object(installer, "_git_clean", return_value=False),
                patch.object(installer, "_run") as run,
            ):
                with self.assertRaisesRegex(installer.InstallError, "dirty runtime source"):
                    installer._install_runtime()
            run.assert_not_called()

    def test_runtime_install_uses_pinned_git_checkout_and_uv_from_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary) / "runtime"
            source = runtime_root / "mlx-gen"
            venv = runtime_root / ".venv"
            uv_cache = Path(temporary) / "uv-cache"
            commands: list[list[str]] = []
            environments: list[dict[str, str] | None] = []

            def fake_run(command, **kwargs):
                command = list(command)
                commands.append(command)
                environments.append(kwargs.get("env"))
                if "clone" in command:
                    Path(command[-1]).mkdir(parents=True)

            with (
                patch.object(installer, "RUNTIME_ROOT", runtime_root),
                patch.object(installer, "SOURCE_DIR", source),
                patch.object(installer, "VENV_DIR", venv),
                patch.object(installer, "UV_CACHE_DIR", uv_cache),
                patch.object(
                    installer.shutil,
                    "which",
                    side_effect=lambda name: f"/usr/local/bin/{name}",
                ),
                patch.object(installer, "_git_head", return_value=installer.RUNTIME_COMMIT),
                patch.object(installer, "_git_clean", return_value=True),
                patch.object(installer, "_run", side_effect=fake_run),
                patch.object(installer, "runtime_ready", return_value=(True, "ready")),
            ):
                installer._install_runtime()

            self.assertEqual(len(commands), 6)
            self.assertEqual(commands[0][0:2], ["/usr/local/bin/git", "clone"])
            self.assertEqual(commands[-1][:2], ["/usr/local/bin/uv", "sync"])
            self.assertTrue(all(command[0] == "/usr/local/bin/git" for command in commands[:-1]))
            uv_environment = environments[-1]
            assert uv_environment is not None
            self.assertEqual(uv_environment["UV_HTTP_TIMEOUT"], "600")
            self.assertEqual(uv_environment["UV_CONCURRENT_DOWNLOADS"], "2")
            self.assertTrue(source.is_dir())

    def test_runtime_install_rejects_symlinked_virtual_environment_before_uv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / "runtime"
            source = runtime_root / "mlx-gen"
            source.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            venv = runtime_root / ".venv"
            venv.symlink_to(outside, target_is_directory=True)
            with (
                patch.object(installer, "RUNTIME_ROOT", runtime_root),
                patch.object(installer, "SOURCE_DIR", source),
                patch.object(installer, "VENV_DIR", venv),
                patch.object(installer, "UV_CACHE_DIR", root / "uv-cache"),
                patch.object(installer, "_git_head", return_value=installer.RUNTIME_COMMIT),
                patch.object(installer, "_git_clean", return_value=True),
                patch.object(installer, "_run") as run,
            ):
                with self.assertRaisesRegex(installer.InstallError, "symbolic-link virtual environment"):
                    installer._install_runtime()
            run.assert_not_called()

    def test_model_manifest_path_rejects_symlinked_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "model"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "transformer").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(installer.InstallError, "unsafe model manifest path"):
                installer._safe_model_path(root, "transformer/0.safetensors")

    def test_exact_model_tree_rejects_unexpected_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unexpected-empty").mkdir()
            self.assertEqual(
                installer._unexpected_model_entries(root),
                ["unexpected directory: unexpected-empty"],
            )

    def test_model_downloader_uses_pinned_https_curl_and_full_hash(self):
        payload = b"pinned model bytes"
        item = installer.ModelFile(
            "transformer/test.safetensors",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                command = list(command)
                commands.append(command)
                Path(command[command.index("--output") + 1]).write_bytes(payload)
                return installer.subprocess.CompletedProcess(command, 0)

            with (
                patch.object(installer, "MODEL_STAGING_DIR", staging),
                patch.object(installer.subprocess, "run", side_effect=fake_run),
            ):
                installer._download_model_file(item)

            target = staging / item.path
            self.assertEqual(target.read_bytes(), payload)
            command = commands[0]
            self.assertEqual(command[:2], ["/usr/bin/curl", "--disable"])
            self.assertIn("=https", command)
            self.assertNotIn("--insecure", command)
            self.assertNotIn("-k", command)
            self.assertNotIn("Authorization:", " ".join(command))
            self.assertIn(installer.MODEL_REVISION, command[-1])
            self.assertTrue(command[-1].startswith("https://huggingface.co/"))

    def test_tag_compilation_is_deterministic_and_positional(self):
        references = [
            composer.Reference("@maya", Path("/tmp/maya.png")),
            composer.Reference("@jacket", Path("/tmp/jacket.png")),
        ]
        compiled = composer.compile_tagged_prompt(
            "Keep @maya's face; put @jacket on @maya.", references
        )
        self.assertIn('- image 1: internal alias "maya__view_1"; reference group "maya"; view 1 of 1', compiled)
        self.assertIn('- image 2: internal alias "jacket__view_1"; reference group "jacket"; view 1 of 1', compiled)
        self.assertIn('image 1 (reference group "maya")\'s face', compiled)
        self.assertIn('put image 2 (reference group "jacket") on image 1', compiled)
        self.assertNotIn("@maya", compiled)
        self.assertNotIn("@jacket", compiled)

    def test_tag_compilation_groups_repeated_tags_without_creating_extra_subjects(self):
        references = [
            composer.Reference("@maya", Path("/tmp/maya-front.png")),
            composer.Reference("@maya", Path("/tmp/maya-profile.png")),
            composer.Reference("@jacket", Path("/tmp/jacket.png")),
        ]
        compiled = composer.compile_tagged_prompt(
            "Keep @maya's identity and dress @maya in @jacket.", references
        )
        self.assertIn('image 1: internal alias "maya__view_1"', compiled)
        self.assertIn('image 2: internal alias "maya__view_2"', compiled)
        self.assertIn('reference group "maya"; view 2 of 2', compiled)
        self.assertIn(
            'reference group "maya" (same ingredient shown in image 1 and image 2)',
            compiled,
        )
        self.assertIn("complementary views of one ingredient, not separate copies", compiled)
        self.assertNotIn("@maya", compiled)

    def test_tag_compilation_rejects_unknown_and_missing_tags_but_allows_group_views(self):
        maya = composer.Reference("@maya", Path("/tmp/maya.png"))
        jacket = composer.Reference("@jacket", Path("/tmp/jacket.png"))
        with self.assertRaisesRegex(composer.ComposerError, "unregistered"):
            composer.compile_tagged_prompt("Use @maya with @unknown.", [maya])
        with self.assertRaisesRegex(composer.ComposerError, "mention every active"):
            composer.compile_tagged_prompt("Use @maya.", [maya, jacket])
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "one.png"
            image.write_bytes(b"png")
            references = composer.validate_references((("@maya", str(image)), ("maya", str(image))))
            self.assertEqual([reference.tag for reference in references], ["@maya", "@maya"])

    def test_build_command_preserves_reference_order_and_exact_ltx_dimensions(self):
        references = [
            composer.Reference("@maya", Path("/tmp/maya.png")),
            composer.Reference("@jacket", Path("/tmp/jacket.png")),
        ]
        command = composer.build_command(
            "compiled prompt; not a shell command",
            references,
            profile="quality",
            seed=77,
            output=Path("/tmp/output.png"),
        )
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "--image"],
            ["/tmp/maya.png", "/tmp/jacket.png"],
        )
        self.assertEqual(command[command.index("--width") + 1], "768")
        self.assertEqual(command[command.index("--height") + 1], "448")
        self.assertEqual(command[command.index("--canvas-policy") + 1], "exact-resize")
        self.assertEqual(command[command.index("--guidance") + 1], "1.0")
        self.assertIn("--low-ram", command)
        self.assertNotIn("--vae-tiling", command)
        self.assertEqual(command.count("compiled prompt; not a shell command"), 1)

    def test_build_command_matches_target_video_aspect_and_topology_grid(self):
        references = [composer.Reference("@maya", Path("/tmp/maya.png"))]
        portrait = composer.build_command(
            "compiled portrait",
            references,
            profile="quality",
            target_mode="generate",
            aspect_ratio="9:16",
            seed=7,
            output=Path("/tmp/portrait.png"),
        )
        self.assertEqual(portrait[portrait.index("--width") + 1], "448")
        self.assertEqual(portrait[portrait.index("--height") + 1], "768")

        single_stage = composer.build_command(
            "compiled control",
            references,
            profile="control3s",
            target_mode="motion",
            aspect_ratio="4:3",
            seed=8,
            output=Path("/tmp/control.png"),
        )
        self.assertEqual(single_stage[single_stage.index("--width") + 1], "384")
        self.assertEqual(single_stage[single_stage.index("--height") + 1], "288")
        ingredients = composer.build_command(
            "compiled ingredients portrait",
            references,
            profile="ingredients5s",
            target_mode="ingredients",
            aspect_ratio="9:16",
            seed=9,
            output=Path("/tmp/ingredients.png"),
        )
        self.assertEqual(ingredients[ingredients.index("--width") + 1], "224")
        self.assertEqual(ingredients[ingredients.index("--height") + 1], "384")

    def test_ingredients_aspects_keep_profile_default_and_use_safe_pixel_budget(self):
        self.assertEqual(
            composer.resolve_target_dimensions(
                "ingredients5s",
                target_mode="ingredients",
                aspect_ratio="profile",
            ),
            (384, 224),
        )
        self.assertEqual(
            composer.resolve_target_dimensions(
                "trained5s",
                target_mode="ingredients",
                aspect_ratio="profile",
            ),
            (768, 448),
        )
        expected_by_profile = {
            "ingredients5s": {
                "16:9": (384, 224),
                "9:16": (224, 384),
                "1:1": (288, 288),
                "4:3": (352, 256),
                "3:4": (256, 352),
            },
            "trained5s": {
                "16:9": (800, 448),
                "9:16": (448, 800),
                "1:1": (576, 576),
                "4:3": (672, 512),
                "3:4": (512, 672),
            },
        }
        for profile, expected in expected_by_profile.items():
            for aspect_ratio, dimensions in expected.items():
                with self.subTest(profile=profile, aspect_ratio=aspect_ratio):
                    self.assertEqual(
                        composer.resolve_target_dimensions(
                            profile,
                            target_mode="ingredients",
                            aspect_ratio=aspect_ratio,
                        ),
                        dimensions,
                    )

    def test_output_path_is_confined_to_generated_frame_root(self):
        with patch.object(composer, "OUTPUT_ROOT", Path("/tmp/frame-composer-test-root")):
            self.assertEqual(
                composer.safe_output_path("frame.png"),
                Path("/tmp/frame-composer-test-root/frame.png").resolve(),
            )
            for value in ("../escape.png", "/tmp/escape.png", "frame.jpg", "nested/frame.png"):
                with self.subTest(value=value):
                    with self.assertRaises(composer.ComposerError):
                        composer.safe_output_path(value)

    def test_ui_request_validation_keeps_four_photo_cap_and_groups_repeated_tags(self):
        payload = {
            "prompt": "Keep @maya identical and use the outfit from @jacket.",
            "profile": "clip3s",
            "seed": 42,
            "references": [
                {"tag": "MAYA", "image": PNG_1X1_DATA_URL},
                {"tag": "@maya", "image": PNG_1X1_DATA_URL},
                {"tag": "@jacket", "image": PNG_1X1_DATA_URL},
            ],
        }
        spec = ui._validate_frame_composer_request(payload)
        self.assertEqual(spec["profile"], "clip3s")
        self.assertEqual(spec["target_mode"], "generate")
        self.assertEqual(spec["aspect_ratio"], "profile")
        self.assertEqual((spec["width"], spec["height"]), (512, 256))
        self.assertEqual([item["tag"] for item in spec["references"]], ["@maya", "@maya", "@jacket"])
        self.assertEqual(spec["ingredient_count"], 2)
        self.assertEqual(spec["references"][0]["image"][0], "png")

        too_many = dict(payload)
        too_many["references"] = [
            {"tag": f"@ref{index}", "image": PNG_1X1_DATA_URL} for index in range(5)
        ]
        with self.assertRaisesRegex(ui.APIError, "1–4"):
            ui._validate_frame_composer_request(too_many)
        unknown = dict(payload, prompt="Use @maya and @other.")
        with self.assertRaisesRegex(ui.APIError, "unknown tag"):
            ui._validate_frame_composer_request(unknown)

        portrait = dict(payload, target_mode="generate", profile="quality", aspect_ratio="9:16")
        portrait_spec = ui._validate_frame_composer_request(portrait)
        self.assertEqual((portrait_spec["width"], portrait_spec["height"]), (448, 768))

        control = dict(payload, target_mode="union", profile="control3s", aspect_ratio="4:3")
        control_spec = ui._validate_frame_composer_request(control)
        self.assertEqual((control_spec["width"], control_spec["height"]), (384, 288))

        portrait_ingredients = dict(
            payload,
            target_mode="ingredients",
            profile="ingredients5s",
            aspect_ratio="9:16",
        )
        portrait_ingredients_spec = ui._validate_frame_composer_request(portrait_ingredients)
        self.assertEqual(
            (portrait_ingredients_spec["width"], portrait_ingredients_spec["height"]),
            (224, 384),
        )

        legacy_ingredients = dict(payload, profile="ingredients5s")
        legacy_spec = ui._validate_frame_composer_request(legacy_ingredients)
        self.assertEqual(legacy_spec["target_mode"], "ingredients")
        self.assertEqual((legacy_spec["width"], legacy_spec["height"]), (384, 224))

    def test_ui_exposes_assignment_controls_and_keeps_control_readiness_independent(self):
        for marker in (
            'id="frameComposerPrompt"',
            'id="assignComposedStart"',
            'id="assignComposedEnd"',
            'id="assignComposedInterior"',
            "0 / 4 active photos",
            "+ Add ingredient",
            "one or more reference views",
            "library may still hold eight references",
        ):
            self.assertIn(marker, ui.PAGE)
        gemma = {"auto": {"ready": True, "selected": "official", "note": "ready"}}
        with (
            patch.object(ui, "_cheap_phosphene_pack_issues", return_value=0),
            patch.object(ui, "_control_mode_assets", return_value=(2, "Q8 control assets needed")),
        ):
            missing = ui._workflow_readiness(gemma)
        with (
            patch.object(ui, "_cheap_phosphene_pack_issues", return_value=0),
            patch.object(ui, "_control_mode_assets", return_value=(0, "Q8 control assets ready")),
        ):
            present = ui._workflow_readiness(gemma)
        for mode in ui.CONDITION_MODES:
            self.assertTrue(missing[mode]["blocked"])
            self.assertIn("Q8 control assets needed", missing[mode]["note"])
            self.assertFalse(present[mode]["blocked"])
            self.assertIn("Q8 control assets ready", present[mode]["note"])

    def test_frame_state_stages_private_inputs_without_starting_gpu_in_unit_test(self):
        state = ui.FrameComposerState()
        spec = ui._validate_frame_composer_request(
            {
                "prompt": "Use @maya as the sole subject.",
                "profile": "draft",
                "seed": 9,
                "references": [{"tag": "@maya", "image": PNG_1X1_DATA_URL}],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(ui, "FRAME_GENERATED_DIR", root),
                patch.object(ui, "_frame_composer_readiness", return_value={"ready": True, "note": "ready"}),
                patch.object(ui.threading, "Thread") as thread,
            ):
                queued = state.start(spec)
                self.assertEqual(queued["job"]["status"], "starting")
                self.assertEqual(queued["job"]["reference_count"], 1)
                self.assertEqual(queued["job"]["ingredient_count"], 1)
                assert state._job is not None
                staged = state._job["references"][0]["path"]
                self.assertEqual(staged.parent, root.resolve())
                self.assertEqual(staged.stat().st_mode & 0o777, 0o600)
                self.assertEqual(
                    staged.read_bytes(),
                    base64.b64decode(PNG_1X1_DATA_URL.split(",", 1)[1]),
                )
                thread.return_value.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
