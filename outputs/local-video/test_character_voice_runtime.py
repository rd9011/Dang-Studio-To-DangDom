from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_voice_runtime as subject


class CharacterVoiceRuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.references = self.root / "references"
        self.references.mkdir()
        self.reference = self.references / "voice.wav"
        self.reference.write_bytes(b"reference")
        self.job_dir = self.root / "job"
        self.job_dir.mkdir()
        self.output = self.job_dir / "audio.wav"

    def _payload(self) -> dict:
        return {
            "schema_version": 1,
            "text": "Local speech only.",
            "language": "en",
            "reference": str(self.reference),
            "output": str(self.output),
            "seed": 1234,
            "device": "auto",
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }

    def test_private_job_validation_confines_reference_and_output(self) -> None:
        with mock.patch.object(subject, "REFERENCE_ROOT", self.references):
            job = subject.validate_job(self._payload(), job_dir=self.job_dir)
        self.assertEqual(job["language"], "en")
        self.assertEqual(job["reference"], self.reference.resolve())
        bad = self._payload()
        bad["output"] = str(self.root / "outside.wav")
        with mock.patch.object(subject, "REFERENCE_ROOT", self.references):
            with self.assertRaisesRegex(subject.RuntimeJobError, "private job directory"):
                subject.validate_job(bad, job_dir=self.job_dir)

    def test_all_23_language_ids_match_public_wrapper(self) -> None:
        self.assertEqual(len(subject.SUPPORTED_LANGUAGES), 23)
        self.assertIn("hi", subject.SUPPORTED_LANGUAGES)
        self.assertIn("zh", subject.SUPPORTED_LANGUAGES)
        payload = self._payload()
        payload["language"] = "xx"
        with mock.patch.object(subject, "REFERENCE_ROOT", self.references):
            with self.assertRaisesRegex(subject.RuntimeJobError, "unsupported language"):
                subject.validate_job(payload, job_dir=self.job_dir)

    def test_local_cangjie_resolver_never_falls_back_to_hub(self) -> None:
        model = self.root / "model"
        model.mkdir()
        cangjie = model / "Cangjie5_TC.json"
        cangjie.write_text("[]", encoding="utf-8")

        class TokenizerModule:
            hf_hub_download = None

        module = TokenizerModule()
        with mock.patch.object(subject, "MODEL_DIR", model):
            subject._install_local_cangjie_resolver(module)
        self.assertEqual(
            module.hf_hub_download(repo_id="ResembleAI/chatterbox", filename="Cangjie5_TC.json"),
            str(cangjie),
        )
        with self.assertRaises(subject.OfflineViolation):
            module.hf_hub_download(repo_id="someone/else", filename="anything")

    def test_offline_guard_blocks_sockets_in_fresh_process(self) -> None:
        script = (
            "import json,os,socket,sys; "
            f"sys.path.insert(0,{str(subject.HERE)!r}); "
            "import character_voice_runtime as r; r.enforce_offline_environment(); "
            "blocked=False\n"
            "try:\n socket.create_connection(('example.com',443))\n"
            "except r.OfflineViolation:\n blocked=True\n"
            "print(json.dumps({'blocked':blocked,'offline':os.environ.get('HF_HUB_OFFLINE'),"
            "'token':os.environ.get('HF_TOKEN')}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**dict(subject.os.environ), "HF_TOKEN": "secret"},
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["offline"], "1")
        self.assertIsNone(result["token"])

    def test_device_selection_does_not_silently_ignore_explicit_mps(self) -> None:
        class MPS:
            @staticmethod
            def is_available() -> bool:
                return False

        class Backends:
            mps = MPS()

        class FakeTorch:
            backends = Backends()

        with self.assertRaisesRegex(subject.RuntimeJobError, "unavailable"):
            subject._choose_device(FakeTorch(), "mps")
        self.assertEqual(subject._choose_device(FakeTorch(), "auto"), "cpu")


if __name__ == "__main__":
    unittest.main()
