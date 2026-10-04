#!/usr/bin/env python3
"""Stream a video through the pinned CoreML Real-ESRGAN x4plus model.

The program deliberately keeps only one decoded frame (and its restored result)
in memory.  ffmpeg supplies raw RGB frames and consumes the resized RGB result,
so no temporary PNG sequence is created.  Progress is emitted as JSON Lines on
stdout; diagnostic output is confined to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple

from pinned_source import PROVENANCE_NAME, SourcePin, verify_source_tree
from studio_paths import PATHS


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = PATHS.workspace
UPSCALER_ROOT = PATHS.ai_upscaler_runtime
UPSTREAM_SCRIPT = UPSCALER_ROOT / "upscale.py"
UPSCALER_VENV = UPSCALER_ROOT / ".venv"
MODEL_PACKAGE = (
    UPSCALER_ROOT
    / "weights"
    / "RealESRGAN_x4plus_522_fp16.mlpackage"
)

EXPECTED_UPSTREAM_COMMIT = "15ab3bf80a577cecb93bb2b7ed48711ba7d894d5"
UPSTREAM_SOURCE_PIN = SourcePin(
    name="real-esrgan-coreml",
    repository="https://github.com/hanxiao/real-esrgan-coreml.git",
    revision=EXPECTED_UPSTREAM_COMMIT,
    archive_url=(
        "https://codeload.github.com/hanxiao/real-esrgan-coreml/tar.gz/"
        "15ab3bf80a577cecb93bb2b7ed48711ba7d894d5"
    ),
    archive_sha256="e76a9e768d77d76d6e4f9fe8fcab7e1b5ffd554a756ce19c8d8b76e231426039",
    archive_bytes=1_038_973,
    archive_root="real-esrgan-coreml-15ab3bf80a577cecb93bb2b7ed48711ba7d894d5",
    tree_sha256="32ee96c2fe35439d614d02a72cb38bbedfcad06f5bbde8c58f5273d629eff6c0",
    includes=("pyproject.toml", "uv.lock", "README.md", "upscale.py"),
)
EXPECTED_UPSTREAM_SCRIPT_SHA256 = (
    "33a26e6ae7373aed190080b7f645b0ffcf656e9484cb88b290aa7aec37df934f"
)
EXPECTED_RUNTIME = {
    "coremltools": "9.0",
    "numpy": "2.4.3",
    "Pillow": "12.1.1",
}
TARGET_SHORT_EDGES = (1080,)
MODEL_NAME = "x4plus"
MODEL_SIZE = 522
MODEL_SCALE = 4
TILE_SIZE = 512
TILE_OVERLAP = 32
PRE_PAD = 10


@dataclass(frozen=True)
class FilePin:
    relative_path: str
    size: int
    sha256: str


MODEL_FILE_PINS = (
    FilePin(
        "Manifest.json",
        617,
        "0b870fbf547de382aa0a7b5555713e12e4eb67178fa53e1b6183346fc97877bd",
    ),
    FilePin(
        "Data/com.apple.CoreML/model.mlmodel",
        645668,
        "382faabef987f5f5220e3781524c2b7d8099349e629a203a9113a22b0994b8d3",
    ),
    FilePin(
        "Data/com.apple.CoreML/weights/weight.bin",
        33440896,
        "b0771b2a3874414cc1bc409300447a3398f8096fd92515b27f348d1137c6db3e",
    ),
)


class UpscaleError(RuntimeError):
    """A user-actionable upscaling failure."""


class InstallValidationError(UpscaleError):
    """The pinned runtime or model installation is not intact."""


class UpscaleCancelled(UpscaleError):
    """The operation was interrupted by SIGINT or SIGTERM."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: str
    frame_count: int
    duration: float
    has_audio: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _valid_positive_fraction(value: Any) -> Optional[str]:
    if not isinstance(value, str) or value in ("", "0/0", "N/A"):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if parsed <= 0:
        return None
    # Keep ffprobe's rational spelling instead of converting it through a float.
    return value


def parse_probe_payload(payload: Mapping[str, Any]) -> VideoInfo:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise UpscaleError("ffprobe returned no stream list")
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise UpscaleError("input contains no video stream")

    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UpscaleError("ffprobe returned invalid video dimensions") from exc
    if width <= 0 or height <= 0:
        raise UpscaleError("video dimensions must be positive")

    fps = _valid_positive_fraction(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _valid_positive_fraction(video_stream.get("r_frame_rate"))
    if fps is None:
        raise UpscaleError("ffprobe returned no usable source frame rate")

    frame_count = 0
    for key in ("nb_read_frames", "nb_frames"):
        raw_count = video_stream.get(key)
        if raw_count not in (None, "", "N/A"):
            try:
                frame_count = int(raw_count)
            except (TypeError, ValueError):
                frame_count = 0
            if frame_count > 0:
                break
    if frame_count <= 0:
        raise UpscaleError("ffprobe could not determine the exact source frame count")

    raw_duration = video_stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        format_payload = payload.get("format")
        if isinstance(format_payload, Mapping):
            raw_duration = format_payload.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = float(Fraction(frame_count, 1) / Fraction(fps))
    if duration <= 0:
        duration = float(Fraction(frame_count, 1) / Fraction(fps))

    has_audio = any(
        isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
        for stream in streams
    )
    return VideoInfo(width, height, fps, frame_count, duration, has_audio)


def calculate_target_dimensions(width: int, height: int, short_edge: int) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if short_edge not in TARGET_SHORT_EDGES:
        raise ValueError("target short edge must currently be 1080")
    if min(width, height) >= short_edge:
        raise ValueError("source already has a short edge of 1080 pixels or greater")

    if width <= height:
        target_width = short_edge
        exact_height = height * short_edge / width
        target_height = max(2, int(round(exact_height / 2.0)) * 2)
    else:
        target_height = short_edge
        exact_width = width * short_edge / height
        target_width = max(2, int(round(exact_width / 2.0)) * 2)
    return target_width, target_height


def build_probe_command(ffprobe: str, input_path: Path) -> List[str]:
    return [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration:format=duration"
        ),
        "-of",
        "json",
        str(input_path),
    ]


def build_decoder_command(ffmpeg: str, input_path: Path) -> List[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]


def build_encoder_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: str,
    frame_count: int,
) -> List[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        fps,
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-frames:v",
        str(frame_count),
        "-r",
        fps,
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise InstallValidationError(f"executable is unavailable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise InstallValidationError(f"executable is unavailable on PATH: {value}")
    return resolved


def _check_executable(binary: str) -> None:
    result = subprocess.run(
        [binary, "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise InstallValidationError(f"{binary} -version failed: {detail}")


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise InstallValidationError(
            "could not verify the pinned upscaler checkout: " + result.stderr.strip()
        )
    return result.stdout.strip()


def validate_install(
    ffmpeg_value: str,
    ffprobe_value: str,
    require_current_runtime: bool = True,
) -> Dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise InstallValidationError("the CoreML upscaler requires Apple Silicon macOS")
    if not UPSCALER_ROOT.is_dir():
        raise InstallValidationError(f"upscaler checkout is missing: {UPSCALER_ROOT}")
    if (UPSCALER_ROOT / PROVENANCE_NAME).is_file() and not (UPSCALER_ROOT / ".git").exists():
        source_ready, source_note = verify_source_tree(
            UPSCALER_ROOT,
            UPSTREAM_SOURCE_PIN,
            allowed_runtime_entries=(".venv", "weights", "install-receipt.json"),
        )
        if not source_ready:
            raise InstallValidationError(source_note)
    elif _git_head(UPSCALER_ROOT) != EXPECTED_UPSTREAM_COMMIT:
        raise InstallValidationError("upscaler checkout does not match the pinned commit")
    if not UPSTREAM_SCRIPT.is_file():
        raise InstallValidationError(f"pinned upstream module is missing: {UPSTREAM_SCRIPT}")
    if sha256_file(UPSTREAM_SCRIPT) != EXPECTED_UPSTREAM_SCRIPT_SHA256:
        raise InstallValidationError("pinned upstream upscale.py failed its SHA-256 check")

    expected_paths = {pin.relative_path for pin in MODEL_FILE_PINS}
    if not MODEL_PACKAGE.is_dir():
        raise InstallValidationError(f"CoreML model package is missing: {MODEL_PACKAGE}")
    actual_paths = {
        path.relative_to(MODEL_PACKAGE).as_posix()
        for path in MODEL_PACKAGE.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise InstallValidationError(
            f"CoreML model manifest differs (missing={missing}, extra={extra})"
        )
    for pin in MODEL_FILE_PINS:
        path = MODEL_PACKAGE / pin.relative_path
        if path.stat().st_size != pin.size:
            raise InstallValidationError(f"CoreML model size check failed: {pin.relative_path}")
        if sha256_file(path) != pin.sha256:
            raise InstallValidationError(f"CoreML model hash check failed: {pin.relative_path}")

    try:
        manifest = json.loads((MODEL_PACKAGE / "Manifest.json").read_text("utf-8"))
        root_id = manifest["rootModelIdentifier"]
        entries = manifest["itemInfoEntries"]
        root_entry = entries[root_id]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise InstallValidationError("CoreML Manifest.json is invalid") from exc
    if manifest.get("fileFormatVersion") != "1.0.0":
        raise InstallValidationError("unexpected CoreML package format version")
    if root_entry.get("path") != "com.apple.CoreML/model.mlmodel":
        raise InstallValidationError("CoreML package root model path is invalid")
    entry_paths = {
        entry.get("path") for entry in entries.values() if isinstance(entry, Mapping)
    }
    if "com.apple.CoreML/weights" not in entry_paths:
        raise InstallValidationError("CoreML package weights entry is missing")

    if not UPSCALER_VENV.is_dir() or not (UPSCALER_VENV / "bin" / "python").exists():
        raise InstallValidationError(f"dedicated upscaler runtime is missing: {UPSCALER_VENV}")
    if require_current_runtime:
        try:
            in_expected_venv = Path(sys.prefix).samefile(UPSCALER_VENV)
        except (FileNotFoundError, OSError):
            in_expected_venv = False
        if not in_expected_venv:
            raise InstallValidationError(
                f"run this wrapper with {UPSCALER_VENV / 'bin' / 'python'}"
            )
        for distribution, expected_version in EXPECTED_RUNTIME.items():
            try:
                actual_version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise InstallValidationError(
                    f"runtime dependency is missing: {distribution}"
                ) from exc
            if actual_version != expected_version:
                raise InstallValidationError(
                    f"{distribution} {actual_version} does not match pinned {expected_version}"
                )

    ffmpeg = _resolve_executable(ffmpeg_value)
    ffprobe = _resolve_executable(ffprobe_value)
    _check_executable(ffmpeg)
    _check_executable(ffprobe)
    return {
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "upstream_script_sha256": EXPECTED_UPSTREAM_SCRIPT_SHA256,
        "model": str(MODEL_PACKAGE),
        "model_files": len(MODEL_FILE_PINS),
        "model_bytes": sum(pin.size for pin in MODEL_FILE_PINS),
        "model_manifest": [
            {
                "path": pin.relative_path,
                "bytes": pin.size,
                "sha256": pin.sha256,
            }
            for pin in MODEL_FILE_PINS
        ],
        "runtime": str(UPSCALER_VENV),
        "runtime_versions": dict(EXPECTED_RUNTIME),
        "compute_unit": "CPU_AND_GPU",
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
    }


def probe_video(ffprobe: str, input_path: Path) -> VideoInfo:
    result = subprocess.run(
        build_probe_command(ffprobe, input_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UpscaleError("ffprobe failed: " + result.stderr.strip())
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UpscaleError("ffprobe returned invalid JSON") from exc
    return parse_probe_payload(payload)


def load_pinned_upscaler() -> Tuple[ModuleType, Any, str]:
    """Load the verified upstream module and model without its download helper."""
    spec = importlib.util.spec_from_file_location("dang_pinned_realesrgan", UPSTREAM_SCRIPT)
    if spec is None or spec.loader is None:
        raise InstallValidationError("could not load pinned upstream upscale.py")
    module = importlib.util.module_from_spec(spec)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
    except Exception as exc:
        raise InstallValidationError(f"could not import pinned upscale.py: {exc}") from exc

    expected_path = module.get_mlpackage_path(MODEL_NAME, MODEL_SIZE, True)
    if expected_path.resolve() != MODEL_PACKAGE.resolve():
        raise InstallValidationError("upstream model path no longer matches the pinned package")

    # Do not call load_coreml_model/ensure_model: those helpers can download.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = module.ct.models.MLModel(
                str(MODEL_PACKAGE),
                compute_units=module.ct.ComputeUnit.CPU_AND_GPU,
            )
            model_spec = model.get_spec()
    except Exception as exc:
        raise InstallValidationError(f"could not load pinned CoreML model: {exc}") from exc
    outputs = model_spec.description.output
    if len(outputs) != 1 or not outputs[0].name:
        raise InstallValidationError("CoreML model exposes an unexpected output signature")
    return module, model, outputs[0].name


def restore_frame(
    module: ModuleType,
    model: Any,
    output_key: str,
    rgb_bytes: bytes,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> bytes:
    # Heavy runtime dependencies remain lazy so --preflight never imports CoreML.
    import numpy as np
    from PIL import Image

    expected = source_width * source_height * 3
    if len(rgb_bytes) != expected:
        raise UpscaleError(f"decoded frame has {len(rgb_bytes)} bytes; expected {expected}")
    source = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
        (source_height, source_width, 3)
    )
    source_float = source.astype(np.float32) / 255.0
    try:
        with contextlib.redirect_stdout(sys.stderr):
            restored = module.upscale_image_coreml(
                model,
                output_key,
                source_float,
                MODEL_SIZE,
                scale=MODEL_SCALE,
                pre_pad=PRE_PAD,
                tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP,
                use_batch=False,
            )
    except Exception as exc:
        raise UpscaleError(f"CoreML frame restoration failed: {exc}") from exc
    restored_uint8 = np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(restored_uint8, "RGB")
    if image.size != (target_width, target_height):
        image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS,
        )
    return image.tobytes()


def read_exact_frame(stream: BinaryIO, frame_bytes: int) -> Optional[bytes]:
    chunks: List[bytes] = []
    remaining = frame_bytes
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            received = frame_bytes - remaining
            raise UpscaleError(
                f"decoder ended inside a raw frame ({received}/{frame_bytes} bytes)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_all(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = stream.write(view)
        if written is None:
            # Buffered streams are allowed to report success as None.
            return
        if written <= 0:
            raise BrokenPipeError("encoder pipe stopped accepting data")
        view = view[written:]


class ProcessSupervisor:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []
        self.cancelled_signal: Optional[int] = None
        self._previous_handlers: Dict[int, Any] = {}

    def add(self, process: subprocess.Popen) -> None:
        self.processes.append(process)

    def terminate_all(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.terminate()
        deadline = time.monotonic() + 3.0
        for process in self.processes:
            if process.poll() is None:
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, OSError):
                        process.kill()
        for process in self.processes:
            if process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.cancelled_signal = signum
        self.terminate_all()
        raise UpscaleCancelled(f"interrupted by signal {signum}")

    def __enter__(self) -> "ProcessSupervisor":
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.terminate_all()
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)


def _read_process_log(log: BinaryIO, limit: int = 12000) -> str:
    log.seek(0)
    data = log.read(limit + 1)
    if len(data) > limit:
        data = data[:limit] + b"\n[diagnostic output truncated]"
    return data.decode("utf-8", "replace").strip()


def emit_progress(
    phase: str,
    completed: int,
    total: int,
    **details: Any,
) -> None:
    progress = 1.0 if total == 0 else max(0.0, min(1.0, completed / total))
    default_messages = {
        "preflight": "Pinned AI upscaler is ready",
        "loading": "Loading the pinned CoreML Real-ESRGAN model",
        "restoring": "Restoring video frames with Real-ESRGAN",
        "encoding": "Encoding the restored video and copying source audio",
        "complete": "AI-upscaled video is ready",
        "cancelled": "AI upscaling was cancelled",
        "error": "AI upscaling failed",
    }
    message = details.pop("message", default_messages.get(phase, phase))
    event: Dict[str, Any] = {
        "phase": phase,
        "completed": completed,
        "total": total,
        "progress": round(progress, 6),
        "message": message,
    }
    event.update(details)
    print(json.dumps(event, sort_keys=True, separators=(",", ":")), flush=True)


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.partial{output_path.suffix}"
    )


def publish_without_overwrite(partial_path: Path, output_path: Path) -> None:
    """Atomically publish via a hard link; fail if output appeared meanwhile."""
    try:
        os.link(partial_path, output_path)
    except FileExistsError as exc:
        raise UpscaleError(f"refusing to overwrite existing output: {output_path}") from exc
    partial_path.unlink()


def validate_io_paths(input_path: Path, output_path: Path) -> Tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise UpscaleError(f"input video is unavailable: {input_path}")
    if input_path == output_path:
        raise UpscaleError("input and output paths must differ")
    if output_path.exists():
        raise UpscaleError(f"refusing to overwrite existing output: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        raise UpscaleError("output must use the .mp4 extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_path, output_path


def upscale_video(
    input_path: Path,
    output_path: Path,
    target_short_edge: int,
    ffmpeg_value: str,
    ffprobe_value: str,
) -> None:
    input_path, output_path = validate_io_paths(input_path, output_path)
    installation = validate_install(ffmpeg_value, ffprobe_value, True)
    ffmpeg = installation["ffmpeg"]
    ffprobe = installation["ffprobe"]
    info = probe_video(ffprobe, input_path)
    target_width, target_height = calculate_target_dimensions(
        info.width, info.height, target_short_edge
    )

    emit_progress(
        "loading",
        0,
        1,
        source_width=info.width,
        source_height=info.height,
        target_width=target_width,
        target_height=target_height,
        fps=info.fps,
        frame_count=info.frame_count,
    )
    module, model, output_key = load_pinned_upscaler()
    emit_progress("loading", 1, 1, compute_unit="CPU_AND_GPU", model=MODEL_NAME)

    partial_path = _partial_output_path(output_path)
    decoder_log = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
    encoder_log = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
    frame_size = info.width * info.height * 3
    completed = 0

    try:
        with ProcessSupervisor() as supervisor:
            decoder = subprocess.Popen(
                build_decoder_command(ffmpeg, input_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=decoder_log,
                shell=False,
            )
            supervisor.add(decoder)
            encoder = subprocess.Popen(
                build_encoder_command(
                    ffmpeg,
                    input_path,
                    partial_path,
                    target_width,
                    target_height,
                    info.fps,
                    info.frame_count,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=encoder_log,
                shell=False,
            )
            supervisor.add(encoder)
            if decoder.stdout is None or encoder.stdin is None:
                raise UpscaleError("failed to create ffmpeg streaming pipes")

            emit_progress("restoring", 0, info.frame_count)
            while True:
                frame = read_exact_frame(decoder.stdout, frame_size)
                if frame is None:
                    break
                if completed >= info.frame_count:
                    raise UpscaleError(
                        "decoder produced more frames than the verified source frame count"
                    )
                restored = restore_frame(
                    module,
                    model,
                    output_key,
                    frame,
                    info.width,
                    info.height,
                    target_width,
                    target_height,
                )
                try:
                    write_all(encoder.stdin, restored)
                    encoder.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise UpscaleError("ffmpeg encoder closed its input early") from exc
                completed += 1
                emit_progress("restoring", completed, info.frame_count)

            decoder.stdout.close()
            decoder_returncode = decoder.wait()
            if completed != info.frame_count:
                raise UpscaleError(
                    f"decoded {completed} frames; ffprobe verified {info.frame_count}"
                )
            if decoder_returncode != 0:
                raise UpscaleError("ffmpeg decoder failed: " + _read_process_log(decoder_log))

            emit_progress("encoding", 0, 1)
            encoder.stdin.close()
            encoder_returncode = encoder.wait()
            if encoder_returncode != 0:
                raise UpscaleError("ffmpeg encoder failed: " + _read_process_log(encoder_log))
            if not partial_path.is_file() or partial_path.stat().st_size == 0:
                raise UpscaleError("ffmpeg did not create a usable output video")
            emit_progress("encoding", 1, 1)

        publish_without_overwrite(partial_path, output_path)
        emit_progress(
            "complete",
            1,
            1,
            output=str(output_path),
            source_width=info.width,
            source_height=info.height,
            target_width=target_width,
            target_height=target_height,
            fps=info.fps,
            frame_count=completed,
            duration=info.duration,
            audio_copied=info.has_audio,
            bytes=output_path.stat().st_size,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial_path.unlink()
        decoder_log.close()
        encoder_log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streaming CoreML Real-ESRGAN video restoration and 1080p upscale"
    )
    parser.add_argument("--input", type=Path, help="source video")
    parser.add_argument("--output", type=Path, help="new MP4 path (must not already exist)")
    parser.add_argument(
        "--target-short-edge",
        type=int,
        choices=TARGET_SHORT_EDGES,
        default=1080,
        help="exact target short edge (currently 1080 only)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable or path")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable or path")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="verify the pinned checkout, model, runtime and media tools, then exit",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            details = validate_install(args.ffmpeg, args.ffprobe, True)
            emit_progress(
                "preflight",
                1,
                1,
                ready=True,
                coreml_imported="coremltools" in sys.modules,
                **details,
            )
            return 0
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --preflight is used")
        upscale_video(
            args.input,
            args.output,
            args.target_short_edge,
            args.ffmpeg,
            args.ffprobe,
        )
        return 0
    except UpscaleCancelled as exc:
        emit_progress("cancelled", 0, 1, error=str(exc))
        return 130
    except (UpscaleError, OSError, subprocess.SubprocessError) as exc:
        emit_progress("error", 0, 1, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
