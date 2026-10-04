#!/usr/bin/env python3
"""Safe local wrappers for LTX-2.5 retake, extend, and audio-to-video.

The upstream MLX CLI exposes these beta pipelines directly. This wrapper adds
the local installation's safety contract: local-only inputs and model paths,
16 GB-friendly limits, low-RAM streaming where upstream supports it, offline
environment flags, confined/atomic outputs, mode-specific asset checks, and a
GPU-free dry run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Callable, NoReturn, Sequence

from studio_paths import PATHS

from generate_ltx25 import (
    ASPECT_RATIO_PRESETS,
    MIN_FREE_GIB,
    PROFILES,
    Profile,
    format_command_for_log,
    resolve_aspect_dimensions,
    validate_render_prompt,
)
from install_ltx25_advanced_q8 import OverlayError, verify_installed_overlay
from verify_ltx25 import (
    CLI,
    GEMMA_MODEL,
    MODEL,
    RUNTIME,
    VerificationError,
    verify_install,
    verify_low_ram_edit_patch,
)


FRAME_RATE = 24.0
TEMPORAL_COMPRESSION = 8
MAX_FRAMES = 241  # 10.04 s at 24 fps; bounds attention memory on a 16 GB Mac.
DEFAULT_EXTEND_LATENT_FRAMES = 9  # Adds 72 pixel frames: exactly 3.0 s at 24 fps.
MAX_SOURCE_PIXELS = 768 * 448
MAX_EXTEND_LATENT_FRAMES = 24  # Eight seconds at 24 fps.
OUTPUT_ROOT = PATHS.edit_root.resolve()
HQ_MODEL = PATHS.models_root / "ltx-2.5-mlx-q8"
HQ_RECEIPT = HQ_MODEL / ".ltx25-q8-overlay-receipt.json"
HQ_DEV_TRANSFORMER = HQ_MODEL / "transformer-dev.safetensors"
HQ_DISTILLED_TRANSFORMER = HQ_MODEL / "transformer-distilled.safetensors"

HQ_DEV_SIZE = 20_595_256_595
HQ_DISTILLED_SIZE = 20_595_260_317

DEV_ONLY_MODE_NOTE = (
    "This workflow uses the separate Q8 advanced lane. The fast Q4 generator remains unchanged; "
    "Retake and Extend require its dev transformer, while Audio-to-Video also uses the pre-fused "
    "Q8 distilled transformer for low-RAM stage 2."
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".aif",
    ".aiff",
    ".caf",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class UserInputError(ValueError):
    """A request is invalid before model loading."""


class PreflightError(RuntimeError):
    """The local runtime cannot safely execute a validated request."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    num_frames: int | None = None


@dataclass(frozen=True)
class RenderPlan:
    mode: str
    command: tuple[str, ...]
    output: Path
    temporary_output: Path
    inputs: tuple[Path, ...]
    summary: tuple[str, ...]
    missing_assets: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise UserInputError(message)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt", help="description of the intended edited/generated shot")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "destination MP4; relative paths are placed under outputs/local-video/renders, "
            "and absolute paths must remain inside that directory"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="deterministic seed (0 to 4294967295)")
    parser.add_argument(
        "--steps",
        type=int,
        help="denoising steps (8 to 60; defaults to 30 for edits or the selected A2V profile)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        help="CFG guidance scale (1 to 7; defaults to 3 for edits or the selected A2V profile)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        help="STG guidance scale (0 to 2; defaults to 1 for edits or the selected A2V profile)",
    )
    parser.add_argument("--force", action="store_true", help="atomically replace the exact output if it exists")
    parser.add_argument("--quiet", action="store_true", help="suppress upstream progress output")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="SHA-256 all base model weights during execution/preflight (reads about 25.9 GiB)",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the command without checking the model, Metal, or writing files",
    )
    run_mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="run complete model, asset, disk, output, ffprobe, and Metal checks without rendering",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local/offline LTX-2.5 edit and input-audio modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    retake = subparsers.add_parser(
        "retake",
        help="regenerate an interval of a source video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(retake)
    retake.add_argument("--video", type=Path, required=True, help="local source video")
    retake.add_argument(
        "--start-frame",
        "--start",
        dest="start_frame",
        type=int,
        help="first latent frame to regenerate (inclusive; one latent frame is about 8 pixel frames)",
    )
    retake.add_argument(
        "--end-frame",
        "--end",
        dest="end_frame",
        type=int,
        help="last latent frame to regenerate (exclusive)",
    )
    retake.add_argument("--start-seconds", type=float, help="time-based alternative to --start-frame")
    retake.add_argument("--end-seconds", type=float, help="time-based alternative to --end-frame")
    retake.add_argument(
        "--preserve-audio",
        action="store_true",
        help="keep source audio unchanged instead of regenerating audio in the retaken interval",
    )

    extend = subparsers.add_parser(
        "extend",
        help="add generated frames before or after a source video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(extend)
    extend.add_argument("--video", type=Path, required=True, help="local source video")
    amount = extend.add_mutually_exclusive_group()
    amount.add_argument(
        "--extend-frames",
        type=int,
        help="latent frames to add (each produces about 8 pixel frames)",
    )
    amount.add_argument("--extend-seconds", type=float, help="seconds to add; rounded up to the latent grid")
    amount.add_argument(
        "--target-frames",
        type=int,
        help=(
            "explicit target total pixel-frame count (must be 8k+1); when no extension amount "
            f"is given, the default adds {DEFAULT_EXTEND_LATENT_FRAMES} latent / 72 pixel frames "
            "(3.0 s at 24 fps)"
        ),
    )
    extend.add_argument("--direction", choices=("before", "after"), default="after")

    a2v = subparsers.add_parser(
        "a2v",
        aliases=["audio-to-video"],
        help="generate video motion from a local audio track and prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(a2v)
    a2v.add_argument("--audio", type=Path, required=True, help="local audio input")
    a2v.add_argument("--audio-start", type=float, default=0.0, help="start offset in the audio, in seconds")
    a2v.add_argument("--profile", choices=tuple(PROFILES), default="quality")
    a2v.add_argument(
        "--aspect-ratio",
        choices=tuple(ASPECT_RATIO_PRESETS),
        default="profile",
        help="output shape on the nearest safe two-stage LTX grid at the profile's pixel budget",
    )
    a2v.add_argument(
        "--frames",
        type=int,
        help="override profile length; must be 8k+1 and no more than 241 (about 10 seconds)",
    )
    a2v.add_argument("--stage2-steps", type=int, default=3, help="full-resolution refinement steps (1 to 8)")
    a2v.add_argument(
        "--start-image",
        "--image",
        dest="start_image",
        type=Path,
        help="optional first-frame visual reference",
    )
    a2v.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="first-frame conditioning strength (0 to 1)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; kept public so callers can test/integrate the backend."""
    args = _build_parser().parse_args(argv)
    if args.mode == "audio-to-video":
        args.mode = "a2v"
    return args


def _finite(value: float, *, option: str, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        _fail(f"{option} must be a finite number between {minimum:g} and {maximum:g}")
    return value


def _validate_common(args: argparse.Namespace) -> None:
    try:
        validate_render_prompt(args.prompt)
    except ValueError as exc:
        _fail(str(exc))
    if not 0 <= args.seed <= 0xFFFFFFFF:
        _fail("--seed must be between 0 and 4294967295")
    if args.steps is not None and not 8 <= args.steps <= 60:
        _fail("--steps must be between 8 and 60")
    if args.cfg_scale is not None:
        _finite(args.cfg_scale, option="--cfg-scale", minimum=1.0, maximum=7.0)
    if args.stg_scale is not None:
        _finite(args.stg_scale, option="--stg-scale", minimum=0.0, maximum=2.0)


def _local_file(value: Path, *, option: str, extensions: set[str]) -> Path:
    try:
        path = value.expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"{option} cannot be resolved: {value} ({exc})")
    if not path.is_file():
        _fail(f"{option} is not a local file: {path}")
    suffix = path.suffix.lower()
    if suffix not in extensions:
        choices = ", ".join(sorted(extensions))
        _fail(f"unsupported extension for {option}: {suffix or '(none)'} (allowed: {choices})")
    try:
        if path.stat().st_size <= 0:
            _fail(f"{option} is empty: {path}")
    except OSError as exc:
        _fail(f"{option} cannot be inspected: {path} ({exc})")
    return path


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _resolve_output(args: argparse.Namespace, *, now: datetime | None = None) -> Path:
    if args.output is None:
        timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        requested = OUTPUT_ROOT / f"ltx25-{args.mode}-seed{args.seed}-{timestamp}.mp4"
    else:
        requested = args.output.expanduser()
        if not requested.is_absolute():
            # Accept both `name.mp4` and the existing launcher's familiar
            # `renders/name.mp4` spelling without creating renders/renders.
            parts = requested.parts
            requested = (
                PATHS.outputs_root / requested
                if parts and parts[0] == "renders"
                else OUTPUT_ROOT / requested
            )
    output = requested.resolve()
    if not _is_within(output, OUTPUT_ROOT):
        _fail(f"--output must stay inside {OUTPUT_ROOT}: {output}")
    if output.suffix.lower() != ".mp4":
        _fail(f"--output must end in .mp4: {output}")
    if output.exists():
        if output.is_dir():
            _fail(f"output path is a directory: {output}")
        if not args.force:
            _fail(f"output already exists: {output} (pass --force to atomically replace it)")
    return output


def _ratio(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def probe_media(path: Path) -> MediaInfo:
    """Inspect a media file with ffprobe and return normalized metadata."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PreflightError("ffprobe is not on PATH; install ffmpeg/ffprobe before using edit modes")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError as exc:
        raise PreflightError(f"could not run ffprobe for {path}: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise UserInputError(f"ffprobe could not read {path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc

    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    format_data = payload.get("format") or {}

    duration_value = (video or audio or {}).get("duration") or format_data.get("duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise UserInputError(f"media has no positive finite duration: {path}")

    if video is None:
        return MediaInfo(path=path, duration=duration, has_video=False, has_audio=audio is not None)

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    if width <= 0 or height <= 0 or fps is None:
        raise UserInputError(f"video metadata is incomplete (size/fps): {path}")
    raw_frames = video.get("nb_frames")
    try:
        num_frames = int(raw_frames)
    except (TypeError, ValueError):
        num_frames = max(1, int(math.floor(duration * fps + 1e-6)))
    return MediaInfo(
        path=path,
        duration=duration,
        has_video=True,
        has_audio=audio is not None,
        width=width,
        height=height,
        frame_rate=fps,
        num_frames=num_frames,
    )


def _validate_source_video(info: MediaInfo) -> tuple[int, int]:
    if not info.has_video or info.width is None or info.height is None:
        _fail(f"--video has no video stream: {info.path}")
    if info.frame_rate is None or info.num_frames is None:
        _fail(f"--video has incomplete frame metadata: {info.path}")
    if not 1.0 <= info.frame_rate <= 60.0:
        _fail(f"source frame rate must be between 1 and 60 fps; got {info.frame_rate:g}")
    if info.width % 32 or info.height % 32:
        _fail(
            f"source dimensions must both be divisible by 32 for the convolutional VAE; "
            f"got {info.width}x{info.height}"
        )
    if info.width * info.height > MAX_SOURCE_PIXELS:
        _fail(
            f"source {info.width}x{info.height} exceeds this 16 GB wrapper's {MAX_SOURCE_PIXELS:,}-pixel "
            "limit (768x448); resize/crop it first"
        )
    if info.num_frames < 9:
        _fail(f"source must contain at least 9 frames; ffprobe found {info.num_frames}")
    if info.num_frames > MAX_FRAMES:
        _fail(
            f"source has {info.num_frames} frames; this 16 GB wrapper permits at most {MAX_FRAMES} "
            "(about 10 seconds at 24 fps); trim it first"
        )
    compatible_frames = 1 + max(1, (info.num_frames - 1) // TEMPORAL_COMPRESSION) * TEMPORAL_COMPRESSION
    latent_frames = math.ceil(compatible_frames / TEMPORAL_COMPRESSION)
    return compatible_frames, latent_frames


def _resolve_retake_range(args: argparse.Namespace, info: MediaInfo, latent_frames: int) -> tuple[int, int]:
    frame_pair = args.start_frame is not None or args.end_frame is not None
    seconds_pair = args.start_seconds is not None or args.end_seconds is not None
    if frame_pair and seconds_pair:
        _fail("choose either latent --start-frame/--end-frame or --start-seconds/--end-seconds, not both")
    if frame_pair:
        if args.start_frame is None or args.end_frame is None:
            _fail("--start-frame and --end-frame must be supplied together")
        start, end = args.start_frame, args.end_frame
    elif seconds_pair:
        if args.start_seconds is None or args.end_seconds is None:
            _fail("--start-seconds and --end-seconds must be supplied together")
        assert info.frame_rate is not None
        start_s = _finite(args.start_seconds, option="--start-seconds", minimum=0.0, maximum=info.duration)
        end_s = _finite(args.end_seconds, option="--end-seconds", minimum=0.0, maximum=info.duration)
        if start_s >= end_s:
            _fail("--start-seconds must be smaller than --end-seconds")
        start = math.floor(start_s * info.frame_rate / TEMPORAL_COMPRESSION)
        # The extra endpoint latent covers the causal VAE block containing the
        # last requested pixel frame. The upstream end index is exclusive.
        end = math.ceil(end_s * info.frame_rate / TEMPORAL_COMPRESSION) + 1
        end = min(end, latent_frames)
    else:
        _fail("retake requires a complete frame pair or seconds pair")
    if not 0 <= start < end <= latent_frames:
        _fail(
            f"retake range must satisfy 0 <= start < end <= {latent_frames} latent frames; "
            f"got {start}:{end}"
        )
    return start, end


def _resolve_extend_frames(args: argparse.Namespace, info: MediaInfo, compatible_frames: int) -> int:
    if args.extend_frames is not None:
        extend_frames = args.extend_frames
    elif args.extend_seconds is not None:
        assert info.frame_rate is not None
        seconds = _finite(args.extend_seconds, option="--extend-seconds", minimum=0.01, maximum=8.0)
        extend_frames = math.ceil(seconds * info.frame_rate / TEMPORAL_COMPRESSION)
    elif args.target_frames is not None:
        target_frames = args.target_frames
        if not 9 <= target_frames <= MAX_FRAMES:
            _fail(f"--target-frames must be between 9 and {MAX_FRAMES}")
        if (target_frames - 1) % TEMPORAL_COMPRESSION:
            _fail("--target-frames must satisfy frames = 8k+1 (for example 49, 73, 97, or 241)")
        if target_frames <= compatible_frames:
            _fail(
                f"--target-frames ({target_frames}) must exceed the VAE-compatible source length "
                f"({compatible_frames}); use --extend-frames/--extend-seconds or a larger --target-frames"
            )
        extend_frames = (target_frames - compatible_frames) // TEMPORAL_COMPRESSION
    else:
        extend_frames = DEFAULT_EXTEND_LATENT_FRAMES
    if not 1 <= extend_frames <= MAX_EXTEND_LATENT_FRAMES:
        _fail(f"--extend-frames must be between 1 and {MAX_EXTEND_LATENT_FRAMES}")
    total_frames = compatible_frames + extend_frames * TEMPORAL_COMPRESSION
    if total_frames > MAX_FRAMES:
        _fail(
            f"extension would produce about {total_frames} pixel frames; the 16 GB limit is {MAX_FRAMES}. "
            "Trim the source or request a shorter extension."
        )
    return extend_frames


def _a2v_dimensions(profile: Profile, aspect_ratio: str = "profile") -> tuple[int, int]:
    # A2V is two-stage, so full dimensions must be multiples of 64. This is
    # the same floor performed by upstream's snap_output_dimensions().
    width, height = resolve_aspect_dimensions(profile.width, profile.height, aspect_ratio)
    return (width // 64) * 64, (height // 64) * 64


def _temporary_output(output: Path, *, pid: int | None = None) -> Path:
    return output.with_name(f".{output.stem}.ltx25-edit-{pid or os.getpid()}.partial.mp4")


def _has_exact_asset(path: Path, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size
    except OSError:
        return False


def missing_mode_assets(mode: str, *, model: Path = MODEL) -> tuple[str, ...]:
    """Return the exact HQ surface absent from the distilled-only base install.

    ``model`` remains injectable for old callers/tests; the production path
    uses :data:`HQ_MODEL`, where Phosphene installs the Q8 pack and HQ add-on.
    """
    target = HQ_MODEL if model == MODEL else model
    missing: list[str] = []
    receipt = target / HQ_RECEIPT.name
    if not receipt.is_file():
        missing.append(str(receipt))
    if not _has_exact_asset(target / HQ_DEV_TRANSFORMER.name, HQ_DEV_SIZE):
        missing.append(str(target / "transformer-dev.safetensors"))
    if mode == "a2v" and not _has_exact_asset(
        target / HQ_DISTILLED_TRANSFORMER.name,
        HQ_DISTILLED_SIZE,
    ):
        missing.append(str(target / "transformer-distilled.safetensors"))
    return tuple(missing)


def build_plan(
    args: argparse.Namespace,
    *,
    media_probe: Callable[[Path], MediaInfo] = probe_media,
    now: datetime | None = None,
    pid: int | None = None,
) -> RenderPlan:
    """Validate a request and build the exact upstream command without loading MLX."""
    _validate_common(args)
    output = _resolve_output(args, now=now)
    partial = _temporary_output(output, pid=pid)
    if partial.exists():
        _fail(f"temporary render path already exists: {partial}")

    base = [
        str(CLI),
        args.mode,
        "--model",
        str(HQ_MODEL),
        "--gemma",
        str(GEMMA_MODEL),
        "--prompt",
        args.prompt,
        "--output",
        str(partial),
        "--seed",
        str(args.seed),
    ]
    inputs: list[Path] = []
    summary: list[str] = []

    if args.mode in {"retake", "extend"}:
        base.append("--low-ram")
        steps = 30 if args.steps is None else args.steps
        cfg_scale = 3.0 if args.cfg_scale is None else args.cfg_scale
        stg_scale = 1.0 if args.stg_scale is None else args.stg_scale
        video = _local_file(args.video, option="--video", extensions=VIDEO_EXTENSIONS)
        info = media_probe(video)
        compatible_frames, latent_frames = _validate_source_video(info)
        inputs.append(video)
        base.extend(["--video", str(video)])
        if info.num_frames != compatible_frames:
            summary.append(
                f"causal VAE uses the first {compatible_frames} of {info.num_frames} source frames (8k+1 grid)"
            )
        if args.mode == "retake":
            start, end = _resolve_retake_range(args, info, latent_frames)
            base.extend(["--start", str(start), "--end", str(end)])
            if args.preserve_audio:
                base.append("--no-regen-audio")
            summary.append(f"retake latent interval {start}:{end} of {latent_frames}")
        else:
            extend_frames = _resolve_extend_frames(args, info, compatible_frames)
            base.extend(["--extend-frames", str(extend_frames), "--direction", args.direction])
            assert info.frame_rate is not None
            added_seconds = extend_frames * TEMPORAL_COMPRESSION / info.frame_rate
            result_frames = compatible_frames + extend_frames * TEMPORAL_COMPRESSION
            result_seconds = result_frames / info.frame_rate
            summary.append(
                f"extend {args.direction} by {extend_frames} latent frames (~{added_seconds:.2f} s); "
                f"result {result_frames} frames (~{result_seconds:.2f} s at source fps)"
            )
        base.extend(
            [
                "--steps",
                str(steps),
                "--cfg-scale",
                str(cfg_scale),
                "--stg-scale",
                str(stg_scale),
            ]
        )
        summary.append(
            f"source {info.width}x{info.height}, {info.num_frames} frames at {info.frame_rate:.3f} fps"
        )

    elif args.mode == "a2v":
        audio = _local_file(args.audio, option="--audio", extensions=AUDIO_EXTENSIONS)
        audio_info = media_probe(audio)
        if not audio_info.has_audio:
            _fail(f"--audio has no audio stream: {audio}")
        audio_start = _finite(args.audio_start, option="--audio-start", minimum=0.0, maximum=86_400.0)
        if audio_start >= audio_info.duration:
            _fail(f"--audio-start ({audio_start:g}s) is at or beyond the audio duration ({audio_info.duration:.3f}s)")

        profile = PROFILES[args.profile]
        steps = 30 if args.steps is None else args.steps
        cfg_scale = 3.0 if args.cfg_scale is None else args.cfg_scale
        stg_scale = 1.0 if args.stg_scale is None else args.stg_scale
        frames = profile.frames if args.frames is None else args.frames
        if not 9 <= frames <= MAX_FRAMES:
            _fail(f"--frames must be between 9 and {MAX_FRAMES}")
        if (frames - 1) % TEMPORAL_COMPRESSION:
            _fail("--frames must satisfy frames = 8k+1 (for example 49, 97, or 241)")
        if not 1 <= args.stage2_steps <= 8:
            _fail("--stage2-steps must be between 1 and 8")
        width, height = _a2v_dimensions(profile, args.aspect_ratio)
        required_audio = frames / FRAME_RATE
        available_audio = audio_info.duration - audio_start
        if available_audio + 0.05 < required_audio:
            _fail(
                f"audio after --audio-start is {available_audio:.3f}s, but {frames} frames at 24 fps need "
                f"{required_audio:.3f}s; use fewer frames or a longer input"
            )
        inputs.append(audio)
        base.extend(
            [
                "--low-ram",
                "--audio",
                str(audio),
                "--audio-start",
                str(audio_start),
                "--height",
                str(height),
                "--width",
                str(width),
                "--frames",
                str(frames),
                "--frame-rate",
                str(int(FRAME_RATE)),
                "--stage1-steps",
                str(steps),
                "--stage2-steps",
                str(args.stage2_steps),
                "--cfg-scale",
                str(cfg_scale),
                "--stg-scale",
                str(stg_scale),
            ]
        )
        if args.start_image is not None:
            image = _local_file(args.start_image, option="--start-image", extensions=IMAGE_EXTENSIONS)
            strength = _finite(args.image_strength, option="--image-strength", minimum=0.0, maximum=1.0)
            inputs.append(image)
            base.extend(["--image", str(image), "0", str(strength)])
            summary.append(f"first-frame reference {image.name} at strength {strength:g}")
        elif args.image_strength != 1.0:
            _fail("--image-strength requires --start-image")
        summary.append(f"audio-to-video {width}x{height}, {frames} frames @ 24 fps ({required_audio:.2f} s)")
        if args.aspect_ratio != "profile":
            summary.append(f"requested {args.aspect_ratio} aspect on the nearest safe LTX grid")
        if (profile.width, profile.height) != (width, height):
            summary.append(
                f"two-stage grid snaps profile {profile.width}x{profile.height} to {width}x{height}"
            )
    else:  # Defensive guard for programmatic callers bypassing argparse.
        _fail(f"unsupported mode: {args.mode!r}")

    if args.quiet:
        base.append("--quiet")
    for input_path in inputs:
        if input_path == output or input_path == partial:
            _fail(f"an input file cannot also be the output: {input_path}")

    return RenderPlan(
        mode=args.mode,
        command=tuple(base),
        output=output,
        temporary_output=partial,
        inputs=tuple(inputs),
        summary=tuple(summary),
        missing_assets=missing_mode_assets(args.mode),
    )


def offline_environment() -> dict[str, str]:
    """Return the environment used for every render (never mutates os.environ)."""
    cache_root = PATHS.cache_root / "ltx25"
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            # The pinned runtime carries a small, hash-verified backport that
            # adds low-RAM streaming to Retake/Extend. Prefer that source tree
            # over the immutable wheel installed in the venv.
            "PYTHONPATH": str(RUNTIME / "packages" / "ltx-pipelines-mlx" / "src"),
            "LTX2_GEMMA_MAX_LENGTH": "1024",
            "LTX2_VAE_DECODE_BUDGET_GB": "7",
        }
    )
    return env


def _asset_error(plan: RenderPlan) -> PreflightError:
    formatted = "\n  - ".join(plan.missing_assets)
    return PreflightError(
        f"{plan.mode} cannot run with the current distilled-only Q4 pack; missing HQ asset(s):\n"
        f"  - {formatted}\n"
        f"{DEV_ONLY_MODE_NOTE} "
        "Run the pinned Q8 overlay installer first; this launcher never downloads model files."
    )


def _prepare_output(plan: RenderPlan) -> None:
    try:
        plan.output.parent.mkdir(parents=True, exist_ok=True)
        probe = plan.output.parent / f".ltx25-edit-write-test-{os.getpid()}"
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("write-test\n")
        probe.unlink()
    except OSError as exc:
        raise PreflightError(f"output directory is not writable: {plan.output.parent} ({exc})") from exc


def _preflight(plan: RenderPlan, args: argparse.Namespace) -> None:
    if plan.missing_assets:
        raise _asset_error(plan)
    try:
        verify_install(full_hash=args.full_hash, require_gpu=True)
        verify_low_ram_edit_patch()
        verify_installed_overlay(full_hash=args.full_hash)
    except (VerificationError, OverlayError, json.JSONDecodeError, OSError) as exc:
        raise PreflightError(str(exc)) from exc
    _prepare_output(plan)
    free_gib = shutil.disk_usage(plan.output.parent).free / 2**30
    if free_gib < MIN_FREE_GIB:
        raise PreflightError(
            f"only {free_gib:.1f} GiB is free; keep at least {MIN_FREE_GIB:.0f} GiB for macOS swap, "
            "media intermediates, and the output"
        )
    if free_gib < 10:
        print(f"WARNING: only {free_gib:.1f} GiB is free; close other apps before rendering.", file=sys.stderr)


def _print_plan(plan: RenderPlan, *, dry_run: bool) -> None:
    print(f"Mode: {plan.mode}")
    for line in plan.summary:
        print(f"  {line}")
    print("  low-RAM block streaming: forced on where the upstream mode supports it")
    print("  network/model downloads: forced off")
    print(f"Output: {plan.output}")
    print("Command: " + format_command_for_log(plan.command))
    if dry_run and plan.missing_assets:
        print("Asset status (not fatal during --dry-run):", file=sys.stderr)
        for path in plan.missing_assets:
            print(f"  MISSING {path}", file=sys.stderr)


def _discard_partial(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        print(f"WARNING: could not remove incomplete temporary output {path}: {exc}", file=sys.stderr)


def execute(plan: RenderPlan) -> int:
    """Run a fully preflighted plan and atomically promote a verified MP4."""
    try:
        completed = subprocess.run(
            list(plan.command),
            cwd=RUNTIME,
            env=offline_environment(),
            check=False,
        )
    except OSError as exc:
        _discard_partial(plan.temporary_output)
        print(f"RENDER FAILED: could not start the pinned runtime: {exc}", file=sys.stderr)
        return 3
    if completed.returncode:
        _discard_partial(plan.temporary_output)
        print(
            f"RENDER FAILED (exit {completed.returncode}). The incomplete temporary MP4 was removed; "
            "no existing output was replaced.",
            file=sys.stderr,
        )
        return completed.returncode
    if not plan.temporary_output.is_file() or plan.temporary_output.stat().st_size <= 0:
        _discard_partial(plan.temporary_output)
        print("RENDER FAILED: runtime returned success without a non-empty MP4.", file=sys.stderr)
        return 3
    try:
        rendered = probe_media(plan.temporary_output)
    except (UserInputError, PreflightError) as exc:
        _discard_partial(plan.temporary_output)
        print(f"RENDER FAILED: generated MP4 did not pass ffprobe validation: {exc}", file=sys.stderr)
        return 3
    if not rendered.has_video:
        _discard_partial(plan.temporary_output)
        print("RENDER FAILED: generated MP4 has no video stream.", file=sys.stderr)
        return 3
    try:
        os.replace(plan.temporary_output, plan.output)
    except OSError as exc:
        _discard_partial(plan.temporary_output)
        print(f"RENDER FAILED: could not atomically promote output: {exc}", file=sys.stderr)
        return 3
    print(f"VIDEO READY: {plan.output} ({plan.output.stat().st_size / 2**20:.1f} MiB)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        plan = build_plan(args)
        _print_plan(plan, dry_run=args.dry_run)
        if args.dry_run:
            print("DRY RUN COMPLETE; model, Metal, disk, and output-directory checks were not run.")
            return 0
        _preflight(plan, args)
        if args.preflight_only:
            print("PRE-FLIGHT VERIFIED; render not started (--preflight-only).")
            return 0
        return execute(plan)
    except UserInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PreflightError as exc:
        print(f"PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
