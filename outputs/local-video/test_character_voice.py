from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_voice as subject


def write_wave(path: Path, *, seconds: float = 3.2, sample_rate: int = 24_000, amplitude: int = 4_000) -> None:
    frames = int(seconds * sample_rate)
    payload = bytearray()
    for index in range(frames):
        value = int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
        payload.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


class CharacterVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.voice_root = self.root / "voices"
        self.reference_root = self.voice_root / "references"
        self.registry = self.voice_root / "registry.json"
        self.output_root = self.root / "outputs"
        patches = (
            mock.patch.object(subject, "HERE", self.root),
            mock.patch.object(subject, "VOICE_ROOT", self.voice_root),
            mock.patch.object(subject, "REFERENCE_ROOT", self.reference_root),
            mock.patch.object(subject, "REGISTRY_PATH", self.registry),
            mock.patch.object(subject, "OUTPUT_ROOT", self.output_root),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.reference = self.root / "authorized.wav"
        write_wave(self.reference)

    def _register(self, name: str = "aria-voice") -> dict:
        return subject.register_preset(
            name,
            self.reference,
            "en",
            consent_confirmed=True,
            voice_description="warm and calm",
        )

    def test_language_contract_is_exactly_23(self) -> None:
        self.assertEqual(len(subject.SUPPORTED_LANGUAGES), 23)
        self.assertEqual(
            set(subject.SUPPORTED_LANGUAGES),
            {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"},
        )

    def test_registration_requires_explicit_consent(self) -> None:
        with self.assertRaisesRegex(subject.UserInputError, "consent-confirmed"):
            subject.register_preset("aria", self.reference, "en", consent_confirmed=False)
        self.assertFalse(self.registry.exists())

    def test_register_copies_hashes_and_assigns_stable_preset(self) -> None:
        entry = self._register()
        expected = hashlib.sha256(self.reference.read_bytes()).hexdigest()
        self.assertEqual(entry["sha256"], expected)
        canonical = self.voice_root / entry["reference"]
        self.assertTrue(canonical.is_file())
        self.assertEqual(hashlib.sha256(canonical.read_bytes()).hexdigest(), expected)
        assignment = subject.assign_character("aria", "aria-voice")
        self.assertEqual(assignment, {"character": "aria", "preset": "aria-voice"})
        character, preset, resolved, reference = subject.resolve_preset(character="aria", preset=None)
        self.assertEqual((character, preset), ("aria", "aria-voice"))
        self.assertEqual(resolved["sha256"], expected)
        self.assertEqual(reference, canonical.resolve())

    def test_changed_canonical_reference_is_rejected(self) -> None:
        entry = self._register()
        canonical = self.voice_root / entry["reference"]
        canonical.write_bytes(canonical.read_bytes() + b"tamper")
        with self.assertRaisesRegex(subject.PreflightError, "changed on disk"):
            subject.resolve_preset(character=None, preset="aria-voice")

    def test_identity_and_description_have_separate_roles(self) -> None:
        calm = subject.resolve_delivery("deep older narrator, calm and measured")
        dramatic = subject.resolve_delivery("young bright voice, excited and dramatic")
        neutral = subject.resolve_delivery("deep raspy older speaker")
        self.assertEqual(calm.style, "calm")
        self.assertEqual(dramatic.style, "dramatic")
        self.assertEqual(neutral.style, "neutral")
        self.assertIn("identity comes exclusively from the canonical reference", calm.description_effect)
        explicit = subject.resolve_delivery("calm", style="neutral", exaggeration=0.6, cfg_weight=0.1)
        self.assertEqual(explicit.style, "neutral")
        self.assertEqual((explicit.exaggeration, explicit.cfg_weight), (0.6, 0.1))

    def test_text_language_and_reference_audio_validation(self) -> None:
        self.assertEqual(subject.validate_text(" Hello. "), "Hello.")
        with self.assertRaises(subject.UserInputError):
            subject.validate_text("")
        with self.assertRaises(subject.UserInputError):
            subject.validate_text("x" * (subject.MAX_TEXT_CHARACTERS + 1))
        with self.assertRaises(subject.UserInputError):
            subject.validate_language("xx")
        short = self.root / "short.wav"
        write_wave(short, seconds=1.0)
        with self.assertRaisesRegex(subject.UserInputError, "reference duration"):
            subject.inspect_wave(short, reference=True)

    def test_output_is_confined_and_symlinks_are_rejected(self) -> None:
        output = subject._safe_output(Path("scene/aria.wav"), default_stem="aria")
        self.assertEqual(output, (self.output_root / "scene" / "aria.wav").resolve())
        with self.assertRaisesRegex(subject.UserInputError, "confined"):
            subject._safe_output(Path("../../escape.wav"), default_stem="aria")
        outside = self.root / "outside"
        outside.mkdir()
        link = self.output_root / "linked"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(subject.PreflightError, "symbolic link"):
            subject._safe_output(Path("linked/escape.wav"), default_stem="aria")
        target = self.output_root / "target.wav"
        target.write_bytes(b"existing")
        final_link = self.output_root / "final.wav"
        final_link.symlink_to(target)
        with self.assertRaisesRegex(subject.PreflightError, "symbolic link"):
            subject._safe_output(Path("final.wav"), default_stem="aria")

    def test_synthesis_uses_offline_isolated_job_and_atomically_publishes(self) -> None:
        self._register()
        subject.assign_character("aria", "aria-voice")
        fake_python = self.root / "python"
        fake_runtime = self.root / "runtime.py"
        fake_python.write_text("placeholder", encoding="utf-8")
        fake_runtime.write_text("placeholder", encoding="utf-8")
        observed: dict[str, object] = {}

        def runner(command, environment):
            observed["command"] = list(command)
            observed["environment"] = environment
            job_path = Path(command[command.index("--job") + 1])
            result_path = Path(command[command.index("--result") + 1])
            job = json.loads(job_path.read_text(encoding="utf-8"))
            observed["job"] = job
            output = Path(job["output"])
            write_wave(output, seconds=0.25)
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "output": str(output),
                        "offline": True,
                        "device": "mps",
                        "watermarked": True,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        ready = {
            "ready": True,
            "runtime_note": "ok",
            "model_note": "ok",
            "receipt_note": "ok",
        }
        with mock.patch.object(subject.installer, "readiness", return_value=ready), mock.patch.object(
            subject, "VENV_PYTHON", fake_python
        ), mock.patch.object(subject, "RUNTIME_SCRIPT", fake_runtime), mock.patch.dict(
            os.environ, {"HF_TOKEN": "must-not-leak", "HTTPS_PROXY": "http://proxy.invalid"}, clear=False
        ):
            result = subject.synthesize(
                "The storm is close.",
                "en",
                character="aria",
                output=Path("scene/aria-001.wav"),
                voice_description="calm but urgent",
                runner=runner,
            )

        destination = self.output_root / "scene" / "aria-001.wav"
        manifest = self.output_root / "scene" / "aria-001.voice.json"
        self.assertTrue(destination.is_file())
        self.assertTrue(manifest.is_file())
        self.assertEqual(result["audio_path"], str(destination.resolve()))
        self.assertEqual(result["character"], "aria")
        self.assertEqual(result["preset"], "aria-voice")
        self.assertTrue(result["offline"])
        self.assertTrue(result["watermarked"])
        self.assertNotIn("text", result)
        self.assertEqual(result["text_sha256"], hashlib.sha256(b"The storm is close.").hexdigest())
        environment = observed["environment"]
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        job = observed["job"]
        self.assertEqual(job["language"], "en")
        self.assertEqual(job["text"], "The storm is close.")

    def test_preflight_failure_does_not_run_inference_or_publish(self) -> None:
        self._register()
        ready = {
            "ready": False,
            "runtime_note": "missing runtime",
            "model_note": "missing model",
            "receipt_note": "missing receipt",
        }
        runner = mock.Mock()
        with mock.patch.object(subject.installer, "readiness", return_value=ready):
            with self.assertRaisesRegex(subject.PreflightError, "not ready"):
                subject.synthesize("Hello.", "en", preset="aria-voice", output=Path("hello.wav"), runner=runner)
        runner.assert_not_called()
        self.assertFalse((self.output_root / "hello.wav").exists())

    def test_cli_never_accepts_token(self) -> None:
        parser = subject._parser()
        destinations: set[str] = set()
        for action in parser._actions:
            destinations.add(action.dest)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for child in choices.values():
                    destinations.update(item.dest for item in child._actions)
        self.assertFalse({"token", "hf_token", "access_token"} & destinations)


if __name__ == "__main__":
    unittest.main()
