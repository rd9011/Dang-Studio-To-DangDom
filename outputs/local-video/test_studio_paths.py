from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from studio_paths import PathConfigurationError, resolve_studio_paths


HERE = Path(__file__).resolve().parent


class StudioPathsTest(unittest.TestCase):
    def test_unset_environment_preserves_checkout_layout(self) -> None:
        module_dir = Path("/private/tmp/project/outputs/local-video")
        paths = resolve_studio_paths({}, module_dir=module_dir)

        self.assertFalse(paths.custom_data_root)
        self.assertEqual(paths.app_root, module_dir)
        self.assertEqual(paths.data_root, Path("/private/tmp/project/work/local-video"))
        self.assertEqual(paths.models_root, paths.data_root / "models")
        self.assertEqual(paths.ltx_runtime, paths.data_root / "ltx-2-mlx")
        self.assertEqual(paths.generated_root, module_dir / "generated")
        self.assertEqual(paths.edit_root, module_dir / "renders")
        self.assertEqual(paths.frames_root, module_dir / "generated-frames")
        self.assertEqual(paths.voices_root, paths.data_root / "character-voices")

    def test_custom_data_root_uses_self_contained_layout(self) -> None:
        paths = resolve_studio_paths(
            {"DANG_STUDIO_HOME": "/Users/example/DangStudioData"},
            module_dir=Path("/private/tmp/project/outputs/local-video"),
        )

        home = Path("/Users/example/DangStudioData")
        self.assertTrue(paths.custom_data_root)
        self.assertEqual(paths.app_root, Path("/private/tmp/project/outputs/local-video"))
        self.assertEqual(paths.data_root, home)
        self.assertEqual(paths.models_root, home / "models")
        self.assertEqual(paths.ltx_runtime, home / "runtimes/ltx-2-mlx")
        self.assertEqual(paths.character_voice_runtime, home / "runtimes/character-voice-runtime")
        self.assertEqual(paths.ai_upscaler_runtime, home / "runtimes/ai-video-upscaler")
        self.assertEqual(paths.generated_root, home / "outputs/generated")
        self.assertEqual(paths.characters_root, home / "characters")
        self.assertEqual(paths.voices_root, home / "voices")
        self.assertEqual(paths.ollama_models_root, home / "models/ollama")

    def test_relative_data_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(PathConfigurationError, "DANG_STUDIO_HOME must be an absolute path"):
            resolve_studio_paths(
                {"DANG_STUDIO_HOME": "relative/data"},
                module_dir=Path("/private/tmp/project/outputs/local-video"),
            )

    def test_runtime_modules_share_custom_data_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home).resolve()
            env = os.environ.copy()
            env.update(
                {
                    "DANG_STUDIO_HOME": str(home),
                    "PYTHONPATH": str(HERE),
                }
            )
            code = """
import json
import ai_video_upscale as upscale
import character_voice as voice
import compose_ltx_frame as frame
import condition_ltx25 as condition
import edit_ltx25 as edit
import install_character_voice as voice_install
import install_frame_composer as frame_install
import prompt_coach as coach
import verify_ltx25 as verify
print(json.dumps({
    'model': str(verify.MODEL),
    'runtime': str(verify.RUNTIME),
    'edit': str(edit.OUTPUT_ROOT),
    'condition': str(condition.OUTPUT_ROOT),
    'voice': str(voice.VOICE_ROOT),
    'voice_output': str(voice.OUTPUT_ROOT),
    'frame_output': str(frame.OUTPUT_ROOT),
    'voice_runtime': str(voice_install.RUNTIME_ROOT),
    'frame_runtime': str(frame_install.RUNTIME_ROOT),
    'upscaler': str(upscale.UPSCALER_ROOT),
    'ollama': str(coach._PINNED_OLLAMA_EXECUTABLE),
}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=HERE,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            actual = json.loads(completed.stdout)
            expected = {
                "model": str(home / "models/ltx-2.5-mlx-q4"),
                "runtime": str(home / "runtimes/ltx-2-mlx"),
                "edit": str(home / "outputs/renders"),
                "condition": str(home / "outputs/generated"),
                "voice": str(home / "voices"),
                "voice_output": str(home / "outputs/generated/voices"),
                "frame_output": str(home / "outputs/generated-frames"),
                "voice_runtime": str(home / "runtimes/character-voice-runtime"),
                "frame_runtime": str(home / "runtimes/frame-composer"),
                "upscaler": str(home / "runtimes/ai-video-upscaler"),
                "ollama": str(home / "runtimes/runtime/ollama-v0.34.4/ollama"),
            }
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
