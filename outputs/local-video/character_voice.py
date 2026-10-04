#!/usr/bin/env python3
"""Secure local character-voice registry and Chatterbox V3 wrapper.

Voice identity is always anchored by a named preset that contains one explicit,
canonical, consented WAV reference.  A character is assigned to a preset, so
separate dialogue cues cannot silently inherit another character's voice.

Free-form voice descriptions are metadata plus a deliberately small set of
coarse delivery hints.  Chatterbox Multilingual V3 does *not* synthesize a new,
persistent speaker identity from prose; the canonical reference creates the
identity and must remain the same across cues.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence

import install_character_voice as installer
from studio_paths import PATHS


HERE = PATHS.app_root
WORKSPACE = PATHS.workspace
LOCAL_VIDEO_ROOT = PATHS.data_root
RUNTIME_ROOT = PATHS.character_voice_runtime
VENV_PYTHON = RUNTIME_ROOT / ".venv" / "bin" / "python"
RUNTIME_SCRIPT = HERE / "character_voice_runtime.py"
VOICE_ROOT = PATHS.voices_root
REFERENCE_ROOT = VOICE_ROOT / "references"
REGISTRY_PATH = VOICE_ROOT / "registry.json"
OUTPUT_ROOT = PATHS.voice_outputs_root.resolve()

MAX_TEXT_CHARACTERS = 1_600
MAX_DESCRIPTION_CHARACTERS = 500
MAX_REFERENCE_BYTES = 50 * 2**20
MAX_OUTPUT_BYTES = 512 * 2**20
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 30.0
ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?\Z")

SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

STYLE_SETTINGS: dict[str, tuple[float, float]] = {
    "neutral": (0.50, 0.50),
    "calm": (0.35, 0.50),
    "dramatic": (0.75, 0.35),
}
CALM_HINTS = frozenset(("calm", "gentle", "soft", "quiet", "measured", "warm", "soothing"))
DRAMATIC_HINTS = frozenset(("dramatic", "expressive", "energetic", "excited", "intense", "urgent", "angry"))


class UserInputError(ValueError):
    """The request is invalid before model loading."""


class PreflightError(RuntimeError):
    """The pinned local runtime or model is not ready."""


@dataclass(frozen=True)
class WaveInfo:
    channels: int
    sample_width: int
    sample_rate: int
    frames: int
    duration_seconds: float
    rms: float
    size: int


@dataclass(frozen=True)
class DeliverySettings:
    style: str
    exaggeration: float
    cfg_weight: float
    description_effect: str


def _fail(message: str) -> NoReturn:
    raise UserInputError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if not ID_PATTERN.fullmatch(normalized):
        _fail(f"{field} must be 1-64 lowercase letters, digits, hyphens, or underscores")
    return normalized


def validate_language(value: str) -> str:
    language = value.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        _fail(f"unsupported language {value!r}; choose one of {', '.join(SUPPORTED_LANGUAGES)}")
    return language


def validate_text(value: str) -> str:
    text = value.strip()
    if not text:
        _fail("speech text cannot be empty")
    if len(text) > MAX_TEXT_CHARACTERS:
        _fail(
            f"speech text is {len(text)} characters; split narration into cues of at most "
            f"{MAX_TEXT_CHARACTERS} characters to limit hallucinated continuation"
        )
    if "\x00" in text or any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        _fail("speech text contains unsupported control characters")
    return text


def validate_description(value: str | None) -> str:
    if value is None:
        return ""
    description = value.strip()
    if len(description) > MAX_DESCRIPTION_CHARACTERS:
        _fail(f"voice description cannot exceed {MAX_DESCRIPTION_CHARACTERS} characters")
    if "\x00" in description or any(ord(char) < 32 and char not in "\n\r\t" for char in description):
        _fail("voice description contains unsupported control characters")
    return description


def _pcm16_rms(handle: wave.Wave_read, frames: int) -> float:
    remaining = frames
    total_squares = 0
    total_samples = 0
    while remaining:
        block_frames = min(remaining, 65_536)
        raw = handle.readframes(block_frames)
        remaining -= block_frames
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        total_squares += sum(sample * sample for sample in samples)
        total_samples += len(samples)
    return math.sqrt(total_squares / total_samples) if total_samples else 0.0


def inspect_wave(path: Path, *, reference: bool) -> WaveInfo:
    original = path.expanduser()
    if original.is_symlink():
        _fail(f"symbolic-link audio is not accepted: {original}")
    try:
        resolved = original.resolve(strict=True)
    except OSError as exc:
        _fail(f"audio file does not exist: {original} ({exc})")
    if not resolved.is_file() or resolved.suffix.lower() != ".wav":
        _fail("audio must be a regular .wav file")
    size = resolved.stat().st_size
    limit = MAX_REFERENCE_BYTES if reference else MAX_OUTPUT_BYTES
    if size <= 44 or size > limit:
        _fail(f"WAV size must be between 45 and {limit:,} bytes")
    try:
        with wave.open(str(resolved), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
            if compression != "NONE":
                _fail("only uncompressed PCM WAV audio is accepted")
            if sample_width != 2:
                _fail("WAV audio must use signed 16-bit PCM samples")
            if channels not in ((1, 2) if reference else (1,)):
                _fail("reference WAV must be mono/stereo; generated WAV must be mono")
            if not 16_000 <= sample_rate <= 96_000:
                _fail("WAV sample rate must be between 16 kHz and 96 kHz")
            duration = frames / sample_rate if sample_rate else 0.0
            if reference and not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
                _fail(
                    f"reference duration must be {MIN_REFERENCE_SECONDS:g}-{MAX_REFERENCE_SECONDS:g} seconds; "
                    f"got {duration:.2f}"
                )
            if not reference and not 0.05 <= duration <= 300.0:
                _fail(f"generated duration is unsafe or implausible: {duration:.2f} seconds")
            rms = _pcm16_rms(handle, frames)
    except (wave.Error, EOFError) as exc:
        _fail(f"invalid PCM WAV file: {exc}")
    if rms < 32.0:
        _fail("WAV is silent or too quiet to use safely")
    return WaveInfo(channels, sample_width, sample_rate, frames, duration, rms, size)


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": 1, "presets": {}, "characters": {}}


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return _empty_registry()
    if REGISTRY_PATH.is_symlink() or not REGISTRY_PATH.is_file() or REGISTRY_PATH.stat().st_size > 1 << 20:
        raise PreflightError(f"voice registry is unsafe: {REGISTRY_PATH}")
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"voice registry is unreadable: {exc}") from exc
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != 1
        or not isinstance(registry.get("presets"), dict)
        or not isinstance(registry.get("characters"), dict)
    ):
        raise PreflightError("voice registry has an unsupported schema")
    return registry


def _atomic_json(path: Path, payload: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or path.is_symlink():
        raise PreflightError(f"refusing symbolic-link JSON path: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def register_preset(
    name: str,
    reference: Path,
    reference_language: str,
    *,
    consent_confirmed: bool,
    voice_description: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    preset = _identifier(name, field="preset")
    language = validate_language(reference_language)
    description = validate_description(voice_description)
    if not consent_confirmed:
        _fail("--consent-confirmed is required; use only your own voice or a clip you are authorized to clone")
    info = inspect_wave(reference, reference=True)
    source = reference.expanduser().resolve(strict=True)
    digest = _sha256(source)
    registry = load_registry()
    existing = registry["presets"].get(preset)
    if existing:
        if isinstance(existing, dict) and existing.get("sha256") == digest:
            return existing
        if not force:
            _fail(f"preset {preset!r} already exists with a different canonical voice; use --force deliberately")

    if VOICE_ROOT.is_symlink() or REFERENCE_ROOT.is_symlink():
        raise PreflightError("managed voice directory cannot be a symbolic link")
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    canonical_name = f"{preset}--{digest[:16]}.wav"
    canonical = REFERENCE_ROOT / canonical_name
    if canonical.exists():
        if canonical.is_symlink() or not canonical.is_file() or _sha256(canonical) != digest:
            raise PreflightError(f"canonical reference path is occupied by unexpected content: {canonical}")
    else:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{preset}-", suffix=".wav", dir=REFERENCE_ROOT)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as destination, source.open("rb") as origin:
                shutil.copyfileobj(origin, destination, length=1 << 20)
                destination.flush()
                os.fsync(destination.fileno())
            if _sha256(temporary) != digest:
                raise PreflightError("canonical reference copy failed its SHA-256 check")
            os.chmod(temporary, 0o600)
            os.replace(temporary, canonical)
        finally:
            if temporary.exists():
                temporary.unlink()

    entry = {
        "name": preset,
        "reference": canonical.relative_to(VOICE_ROOT).as_posix(),
        "sha256": digest,
        "reference_language": language,
        "duration_seconds": round(info.duration_seconds, 6),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "voice_description": description,
        "description_role": "audition metadata and coarse delivery hints only; not speaker identity",
        "consent_confirmed": True,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["presets"][preset] = entry
    _atomic_json(REGISTRY_PATH, registry)
    return entry


def assign_character(character: str, preset: str, *, force: bool = False) -> dict[str, str]:
    character_id = _identifier(character, field="character")
    preset_id = _identifier(preset, field="preset")
    registry = load_registry()
    if preset_id not in registry["presets"]:
        _fail(f"unknown voice preset: {preset_id}")
    existing = registry["characters"].get(character_id)
    if existing and existing != preset_id and not force:
        _fail(f"character {character_id!r} is already assigned to {existing!r}; use --force deliberately")
    registry["characters"][character_id] = preset_id
    _atomic_json(REGISTRY_PATH, registry)
    return {"character": character_id, "preset": preset_id}


def resolve_preset(*, character: str | None, preset: str | None) -> tuple[str | None, str, dict[str, Any], Path]:
    if bool(character) == bool(preset):
        _fail("choose exactly one of character or preset")
    registry = load_registry()
    character_id: str | None = None
    if character:
        character_id = _identifier(character, field="character")
        preset_id = registry["characters"].get(character_id)
        if not isinstance(preset_id, str):
            _fail(f"character {character_id!r} has no assigned voice preset")
    else:
        preset_id = _identifier(str(preset), field="preset")
    entry = registry["presets"].get(preset_id)
    if not isinstance(entry, dict):
        _fail(f"unknown voice preset: {preset_id}")
    if VOICE_ROOT.is_symlink() or REFERENCE_ROOT.is_symlink():
        raise PreflightError("managed voice directory cannot be a symbolic link")
    relative = entry.get("reference")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise PreflightError(f"preset {preset_id!r} has an unsafe reference path")
    unresolved_reference = VOICE_ROOT / relative
    if unresolved_reference.is_symlink():
        raise PreflightError(f"preset {preset_id!r} canonical reference is unsafe")
    reference = unresolved_reference.resolve(strict=True)
    root = REFERENCE_ROOT.resolve(strict=True)
    if root not in reference.parents or not reference.is_file():
        raise PreflightError(f"preset {preset_id!r} canonical reference is unsafe")
    if _sha256(reference) != entry.get("sha256"):
        raise PreflightError(f"preset {preset_id!r} canonical reference changed on disk")
    inspect_wave(reference, reference=True)
    return character_id, preset_id, entry, reference


def resolve_delivery(
    description: str | None,
    *,
    style: str = "auto",
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
) -> DeliverySettings:
    prose = validate_description(description)
    if style not in {"auto", *STYLE_SETTINGS}:
        _fail("style must be auto, neutral, calm, or dramatic")
    resolved_style = style
    if style == "auto":
        words = set(re.findall(r"[a-z]+", prose.lower()))
        if words & DRAMATIC_HINTS:
            resolved_style = "dramatic"
        elif words & CALM_HINTS:
            resolved_style = "calm"
        else:
            resolved_style = "neutral"
    base_exaggeration, base_cfg = STYLE_SETTINGS[resolved_style]
    if exaggeration is not None:
        if not math.isfinite(exaggeration) or not 0.0 <= exaggeration <= 1.5:
            _fail("exaggeration must be between 0 and 1.5")
        base_exaggeration = float(exaggeration)
    if cfg_weight is not None:
        if not math.isfinite(cfg_weight) or not 0.0 <= cfg_weight <= 1.0:
            _fail("cfg-weight must be between 0 and 1")
        base_cfg = float(cfg_weight)
    return DeliverySettings(
        style=resolved_style,
        exaggeration=base_exaggeration,
        cfg_weight=base_cfg,
        description_effect=(
            "coarse delivery knobs only; speaker identity comes exclusively from the canonical reference"
        ),
    )


def _safe_output(value: Path | None, *, default_stem: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ROOT.is_symlink():
        raise PreflightError(f"output root cannot be a symbolic link: {OUTPUT_ROOT}")
    resolved_root = OUTPUT_ROOT.resolve(strict=True)
    resolved_here = HERE.resolve(strict=True)
    if resolved_here not in resolved_root.parents:
        raise PreflightError(f"output root escapes the deliverable directory through a symbolic link: {OUTPUT_ROOT}")
    if value is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        value = Path(f"{default_stem}-{timestamp}.wav")
    if value.is_absolute():
        if value.is_symlink():
            raise PreflightError(f"output cannot be a symbolic link: {value}")
        candidate = value.resolve(strict=False)
    else:
        raw_candidate = resolved_root / value
        if raw_candidate.is_symlink():
            raise PreflightError(f"output cannot be a symbolic link: {raw_candidate}")
        current = resolved_root
        for part in value.parts[:-1]:
            if part in {"", ".", ".."}:
                continue
            current = current / part
            if current.exists() and current.is_symlink():
                raise PreflightError(f"output parent cannot be a symbolic link: {current}")
        candidate = raw_candidate.resolve(strict=False)
    if candidate.suffix.lower() != ".wav" or resolved_root not in candidate.parents:
        _fail(f"output must be a .wav file confined to {resolved_root}")
    current = resolved_root
    for part in candidate.relative_to(resolved_root).parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PreflightError(f"output parent cannot be a symbolic link: {current}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _offline_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(key, None)
    environment.update(
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
    return environment


def _default_runner(command: Sequence[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def synthesize(
    text: str,
    language: str,
    *,
    character: str | None = None,
    preset: str | None = None,
    output: Path | None = None,
    voice_description: str | None = None,
    style: str = "auto",
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    seed: int = 1234,
    device: str = "auto",
    force: bool = False,
    full_hash: bool = False,
    preflight_only: bool = False,
    runner: Callable[[Sequence[str], dict[str, str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    speech = validate_text(text)
    language_id = validate_language(language)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        _fail("seed must be an integer from 0 to 4294967295")
    if device not in {"auto", "mps", "cpu"}:
        _fail("device must be auto, mps, or cpu")
    character_id, preset_id, preset_entry, reference = resolve_preset(character=character, preset=preset)
    description = validate_description(voice_description) or str(preset_entry.get("voice_description", ""))
    delivery = resolve_delivery(
        description,
        style=style,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    destination = _safe_output(output, default_stem=character_id or preset_id)
    manifest_path = destination.with_suffix(".voice.json")
    if not force and (destination.exists() or manifest_path.exists()):
        _fail(f"output already exists; choose another name or use --force: {destination}")
    if destination.is_symlink() or manifest_path.is_symlink():
        raise PreflightError("refusing to replace a symbolic-link output")

    status = installer.readiness(full_hash=full_hash)
    plan = {
        "schema_version": 1,
        "ready": bool(status["ready"]),
        "character": character_id,
        "preset": preset_id,
        "language": language_id,
        "reference_sha256": preset_entry["sha256"],
        "output": str(destination),
        "style": delivery.style,
        "exaggeration": delivery.exaggeration,
        "cfg_weight": delivery.cfg_weight,
        "description_effect": delivery.description_effect,
        "text_sha256": hashlib.sha256(speech.encode("utf-8")).hexdigest(),
        "text_characters": len(speech),
        "seed": seed,
        "device_requested": device,
        "offline": True,
    }
    if preflight_only:
        if not status["ready"]:
            raise PreflightError(
                f"voice backend is not ready: {status['runtime_note']}; {status['model_note']}; "
                f"{status['receipt_note']}"
            )
        return plan
    if not status["ready"]:
        raise PreflightError(
            "voice backend is not ready; run install_character_voice.py first: "
            f"{status['runtime_note']}; {status['model_note']}; {status['receipt_note']}"
        )
    if not VENV_PYTHON.is_file() or not RUNTIME_SCRIPT.is_file():
        raise PreflightError("isolated runtime entry points are missing")

    staging = Path(tempfile.mkdtemp(prefix=".voice-job-", dir=destination.parent))
    os.chmod(staging, 0o700)
    try:
        private_output = staging / "audio.wav"
        job_path = staging / "job.json"
        result_path = staging / "result.json"
        job = {
            "schema_version": 1,
            "text": speech,
            "language": language_id,
            "reference": str(reference),
            "output": str(private_output),
            "seed": seed,
            "device": device,
            "exaggeration": delivery.exaggeration,
            "cfg_weight": delivery.cfg_weight,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
        _atomic_json(job_path, job)
        command = [
            str(VENV_PYTHON),
            "-I",
            str(RUNTIME_SCRIPT),
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ]
        completed = (runner or _default_runner)(command, _offline_subprocess_environment())
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown isolated-runtime failure").strip()[-3000:]
            raise PreflightError(f"voice inference failed: {detail}")
        if not result_path.is_file() or result_path.is_symlink() or result_path.stat().st_size > 64 * 1024:
            raise PreflightError("isolated runtime did not return a safe result manifest")
        try:
            runtime_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreflightError(f"isolated runtime result is unreadable: {exc}") from exc
        if runtime_result.get("output") != str(private_output) or runtime_result.get("offline") is not True:
            raise PreflightError("isolated runtime result violates the offline output contract")
        audio_info = inspect_wave(private_output, reference=False)
        audio_sha = _sha256(private_output)
        result = {
            **plan,
            "ready": True,
            "audio_path": str(destination),
            "manifest_path": str(manifest_path),
            "audio_sha256": audio_sha,
            "sample_rate": audio_info.sample_rate,
            "channels": audio_info.channels,
            "duration_seconds": round(audio_info.duration_seconds, 6),
            "device_used": runtime_result.get("device"),
            "watermarked": runtime_result.get("watermarked") is True,
            "source_commit": installer.SOURCE_COMMIT,
            "model_revision": installer.MODEL_REVISION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        staged_manifest = staging / "audio.voice.json"
        _atomic_json(staged_manifest, result)
        os.replace(staged_manifest, manifest_path)
        os.replace(private_output, destination)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _add_synthesis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text", help="one character's spoken cue; keep different characters in separate calls")
    parser.add_argument("--language", required=True, choices=tuple(SUPPORTED_LANGUAGES))
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--character")
    identity.add_argument("--preset")
    parser.add_argument("--output", type=Path, help="relative WAV path under generated/voices")
    parser.add_argument(
        "--voice-description",
        help="audition note/coarse delivery hint only; never creates or replaces persistent speaker identity",
    )
    parser.add_argument("--style", choices=("auto", "neutral", "calm", "dramatic"), default="auto")
    parser.add_argument("--exaggeration", type=float)
    parser.add_argument("--cfg-weight", type=float)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full-hash", action="store_true", help="hash all 3.21 GB of model files first")
    parser.add_argument("--preflight-only", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage stable local character voices and render one isolated dialogue cue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("languages", help="list all 23 supported language IDs")
    status = sub.add_parser("status", help="inspect pinned runtime/model readiness")
    status.add_argument("--full-hash", action="store_true")
    sub.add_parser("list", help="list named presets and character assignments")

    register = sub.add_parser("register-preset", help="copy one authorized canonical WAV into a named preset")
    register.add_argument("name")
    register.add_argument("--reference", type=Path, required=True)
    register.add_argument("--reference-language", required=True, choices=tuple(SUPPORTED_LANGUAGES))
    register.add_argument("--voice-description")
    register.add_argument("--consent-confirmed", action="store_true")
    register.add_argument("--force", action="store_true")

    assign = sub.add_parser("assign", help="bind a character ID to one stable named voice preset")
    assign.add_argument("character")
    assign.add_argument("preset")
    assign.add_argument("--force", action="store_true")

    synthesize_parser = sub.add_parser("synthesize", help="render one character cue to a confined atomic WAV")
    _add_synthesis_args(synthesize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "languages":
            _json_print(SUPPORTED_LANGUAGES)
        elif args.command == "status":
            status = installer.readiness(full_hash=args.full_hash)
            _json_print(status)
            return 0 if status["ready"] else 1
        elif args.command == "list":
            _json_print(load_registry())
        elif args.command == "register-preset":
            _json_print(
                register_preset(
                    args.name,
                    args.reference,
                    args.reference_language,
                    consent_confirmed=args.consent_confirmed,
                    voice_description=args.voice_description,
                    force=args.force,
                )
            )
        elif args.command == "assign":
            _json_print(assign_character(args.character, args.preset, force=args.force))
        elif args.command == "synthesize":
            _json_print(
                synthesize(
                    args.text,
                    args.language,
                    character=args.character,
                    preset=args.preset,
                    output=args.output,
                    voice_description=args.voice_description,
                    style=args.style,
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight,
                    seed=args.seed,
                    device=args.device,
                    force=args.force,
                    full_hash=args.full_hash,
                    preflight_only=args.preflight_only,
                )
            )
        return 0
    except (UserInputError, PreflightError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
