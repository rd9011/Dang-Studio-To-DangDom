#!/usr/bin/env python3
"""Private isolated-process runner for local Chatterbox Multilingual V3.

The public wrapper writes a validated job into a private staging directory and
invokes this file with the pinned Python environment.  Network access is blocked
before importing the model stack, and Chatterbox is loaded only through
``from_local``.  This is not a user-facing CLI contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, NoReturn, Sequence

from studio_paths import PATHS


HERE = PATHS.app_root
WORKSPACE = PATHS.workspace
LOCAL_VIDEO_ROOT = PATHS.data_root
RUNTIME_ROOT = PATHS.character_voice_runtime
SOURCE_ROOT = RUNTIME_ROOT / "source" / "src"
MODEL_DIR = PATHS.models_root / "chatterbox-multilingual-v3"
VOICE_ROOT = PATHS.voices_root
REFERENCE_ROOT = VOICE_ROOT / "references"
T3_MODEL = "t3_mtl23ls_v3.safetensors"
MAX_TEXT_CHARACTERS = 1_600

SUPPORTED_LANGUAGES = frozenset(
    ("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")
)


class RuntimeJobError(ValueError):
    """The private job payload violates the wrapper/runtime contract."""


class OfflineViolation(RuntimeError):
    """A dependency attempted network access during local inference."""


def _fail(message: str) -> NoReturn:
    raise RuntimeJobError(message)


def enforce_offline_environment() -> None:
    """Set library offline switches and reject socket connections at runtime."""
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        os.environ.pop(key, None)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
            raise OfflineViolation(f"network access is disabled during inference ({event})")

    sys.addaudithook(audit)

    def blocked_connection(*args: object, **kwargs: object) -> NoReturn:
        raise OfflineViolation("network access is disabled during inference")

    socket.create_connection = blocked_connection  # type: ignore[assignment]


def _safe_existing_file(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{field} must be absolute")
    resolved_root = root.resolve(strict=False)
    resolved = path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        _fail(f"{field} is outside the managed voice reference directory")
    if path.is_symlink() or not resolved.is_file():
        _fail(f"{field} must be a regular non-symlink file")
    return resolved


def _safe_output(job_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        _fail("output must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        _fail("output must be absolute")
    root = job_dir.resolve(strict=True)
    resolved = path.resolve(strict=False)
    if root not in resolved.parents or resolved.suffix.lower() != ".wav":
        _fail("output must be a WAV inside the private job directory")
    if path.exists() or path.is_symlink():
        _fail("private output already exists")
    return resolved


def _finite_number(value: object, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _fail(f"{field} must be between {minimum} and {maximum}")
    return number


def validate_job(payload: object, *, job_dir: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        _fail("unsupported runtime job schema")
    text = payload.get("text")
    if not isinstance(text, str):
        _fail("text must be a string")
    text = text.strip()
    if not text or len(text) > MAX_TEXT_CHARACTERS or "\x00" in text:
        _fail(f"text must contain 1 to {MAX_TEXT_CHARACTERS} characters and no NUL")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        _fail("text contains unsupported control characters")
    language = payload.get("language")
    if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
        _fail("unsupported language")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        _fail("seed must be an integer from 0 to 4294967295")
    device = payload.get("device", "auto")
    if device not in {"auto", "mps", "cpu"}:
        _fail("device must be auto, mps, or cpu")
    return {
        "text": text,
        "language": language,
        "reference": _safe_existing_file(REFERENCE_ROOT, payload.get("reference"), field="reference"),
        "output": _safe_output(job_dir, payload.get("output")),
        "seed": seed,
        "device": device,
        "exaggeration": _finite_number(payload.get("exaggeration"), field="exaggeration", minimum=0.0, maximum=1.5),
        "cfg_weight": _finite_number(payload.get("cfg_weight"), field="cfg_weight", minimum=0.0, maximum=1.0),
        "temperature": _finite_number(payload.get("temperature", 0.8), field="temperature", minimum=0.05, maximum=2.0),
        "repetition_penalty": _finite_number(
            payload.get("repetition_penalty", 1.2),
            field="repetition_penalty",
            minimum=1.0,
            maximum=2.0,
        ),
    }


def _choose_device(torch: Any, requested: str) -> str:
    available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise RuntimeJobError("MPS was explicitly requested but is unavailable")
    if requested == "cpu":
        return "cpu"
    return "mps" if available else "cpu"


def _install_local_cangjie_resolver(tokenizer_module: Any) -> None:
    cangjie = MODEL_DIR / "Cangjie5_TC.json"

    def local_hf_hub_download(*, repo_id: str, filename: str, **kwargs: object) -> str:
        if repo_id == "ResembleAI/chatterbox" and filename == "Cangjie5_TC.json" and cangjie.is_file():
            return str(cangjie)
        raise OfflineViolation(f"unavailable local Hub asset requested: {repo_id}/{filename}")

    tokenizer_module.hf_hub_download = local_hf_hub_download


def _write_pcm16_wav(path: Path, samples: Any, sample_rate: int) -> tuple[int, float, float]:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeJobError("model returned an empty or non-mono waveform")
    if not np.isfinite(audio).all():
        raise RuntimeJobError("model returned non-finite audio samples")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise RuntimeJobError("model returned silent audio")
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).round().astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return int(audio.size), float(audio.size / sample_rate), peak


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    enforce_offline_environment()
    if not SOURCE_ROOT.is_dir() or SOURCE_ROOT.is_symlink():
        raise RuntimeJobError("pinned Chatterbox source is unavailable")
    sys.path.insert(0, str(SOURCE_ROOT))

    import torch
    import chatterbox.models.tokenizers.tokenizer as tokenizer_module
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    _install_local_cangjie_resolver(tokenizer_module)
    device = _choose_device(torch, job["device"])
    torch.manual_seed(job["seed"])
    model = ChatterboxMultilingualTTS.from_local(MODEL_DIR, device=device, t3_model=T3_MODEL)
    waveform = model.generate(
        job["text"],
        language_id=job["language"],
        audio_prompt_path=str(job["reference"]),
        exaggeration=job["exaggeration"],
        cfg_weight=job["cfg_weight"],
        temperature=job["temperature"],
        repetition_penalty=job["repetition_penalty"],
    )
    frames, duration, peak = _write_pcm16_wav(job["output"], waveform.detach().cpu().numpy(), int(model.sr))
    return {
        "schema_version": 1,
        "output": str(job["output"]),
        "sample_rate": int(model.sr),
        "channels": 1,
        "sample_width_bytes": 2,
        "frames": frames,
        "duration_seconds": duration,
        "peak": peak,
        "device": device,
        "watermarked": True,
        "offline": True,
        "model": "Chatterbox Multilingual V3",
    }


def _atomic_json(path: Path, payload: object) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=".result-", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        job_path = args.job.resolve(strict=True)
        result_path = args.result.resolve(strict=False)
        if job_path.is_symlink() or not job_path.is_file() or job_path.stat().st_size > 64 * 1024:
            _fail("unsafe private job file")
        job_dir = job_path.parent.resolve(strict=True)
        if job_dir not in result_path.parents or result_path.suffix.lower() != ".json" or result_path.exists():
            _fail("result must be a new JSON file inside the private job directory")
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        job = validate_job(payload, job_dir=job_dir)
        _atomic_json(result_path, execute_job(job))
        return 0
    except (RuntimeJobError, OfflineViolation, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
