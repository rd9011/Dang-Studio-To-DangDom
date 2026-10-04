#!/usr/bin/env python3
"""Generate local LTX-2.5 q4 video + audio on a 16 GB Apple Silicon Mac."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from studio_paths import PATHS

from verify_ltx25 import (
    CLI,
    DDALCU_FALLBACK_GEMMA_MODEL,
    GEMMA_MODEL,
    HERE,
    MODEL,
    RUNTIME,
    VerificationError,
    verify_install,
)


@dataclass(frozen=True)
class Profile:
    width: int
    height: int
    frames: int
    schedule_preset: str
    description: str


@dataclass(frozen=True)
class ConditioningImage:
    path: Path
    frame: int
    strength: float


@dataclass(frozen=True)
class SceneBeat:
    """One wrapper-facing Prompt Relay beat.

    ``length`` uses the upstream runtime's latent-frame unit.  A value of
    ``None`` asks the runtime to distribute the remaining timeline.
    """

    text: str
    length: int | None = None


class _SceneAction(argparse.Action):
    """Append one fixed-arity scene prompt without upstream's nargs ambiguity."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: str,
        option_string: str | None = None,
    ) -> None:
        scenes = list(getattr(namespace, self.dest, None) or [])
        scenes.append(SceneBeat(text=value))
        setattr(namespace, self.dest, scenes)


class _SceneLengthAction(argparse.Action):
    """Attach a length to the most recently declared ``--scene``."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: int,
        option_string: str | None = None,
    ) -> None:
        scenes = list(getattr(namespace, "scenes", None) or [])
        if not scenes:
            parser.error(f"{option_string} must follow the --scene it sizes")
        latest = scenes[-1]
        if latest.length is not None:
            parser.error(f"{option_string} was supplied more than once for the latest --scene")
        if value <= 0:
            parser.error(f"{option_string} must be a positive latent-frame count")
        scenes[-1] = SceneBeat(text=latest.text, length=value)
        setattr(namespace, "scenes", scenes)
        setattr(namespace, self.dest, value)


# All profiles use the distilled q4 transformer, separate Gemma-4 q4 tower,
# two-stage 2x spatial upscaler, low-RAM block streaming, native audio, and
# 24 fps. LTX-2.5 distilled has fixed trained schedules rather than CFG/STG:
# ``default`` is the graded 8+2 path and ``fast`` is a 5+2 draft path.
PROFILES: dict[str, Profile] = {
    "draft": Profile(512, 256, 33, "fast", "fast 5+2 schedule; 1.4 s composition check"),
    "balanced": Profile(640, 384, 41, "default", "graded 8+2 schedule; balanced 1.7 s preview"),
    "quality": Profile(768, 448, 49, "default", "graded 8+2 schedule; recommended 2.0 s final on 16 GB"),
    "max": Profile(768, 448, 25, "default", "graded 8+2 schedule; shortest high-detail final attempt"),
    "clip3s": Profile(512, 256, 73, "default", "graded 8+2 schedule; conservative 3.04 s iteration"),
}

# ``profile`` preserves the launcher's original dimensions exactly.  Explicit
# aspect presets are resolved against the selected profile's pixel budget and
# snapped to the 64-pixel full-resolution grid required by the two-stage path.
# At these intentionally modest local resolutions the resulting canvas can be
# a few percent away from the nominal ratio (for example 768x448 for 16:9), but
# it remains model-safe and avoids silently increasing memory use.
ASPECT_RATIO_PRESETS: dict[str, tuple[int, int] | None] = {
    "profile": None,
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
}
SPATIAL_ALIGNMENT = 64


def resolve_aspect_dimensions(
    width: int,
    height: int,
    aspect_ratio: str = "profile",
    *,
    alignment: int = SPATIAL_ALIGNMENT,
) -> tuple[int, int]:
    """Return the nearest safe grid for a nominal aspect at the same pixel budget."""
    preset = ASPECT_RATIO_PRESETS.get(aspect_ratio)
    if aspect_ratio not in ASPECT_RATIO_PRESETS:
        choices = ", ".join(ASPECT_RATIO_PRESETS)
        raise ValueError(f"aspect ratio must be one of: {choices}")
    if preset is None:
        return width, height
    if width <= 0 or height <= 0 or alignment <= 0:
        raise ValueError("source dimensions and alignment must be positive")

    target_pixels = width * height
    ratio = preset[0] / preset[1]

    ideal_width = math.sqrt(target_pixels * ratio)
    ideal_height = math.sqrt(target_pixels / ratio)

    def nearby_units(value: float) -> range:
        center = max(1, int(math.floor(value / alignment + 0.5)))
        return range(max(1, center - 3), center + 4)

    candidates: list[tuple[float, float, int, int]] = []
    for width_units in nearby_units(ideal_width):
        for height_units in nearby_units(ideal_height):
            candidate_width = width_units * alignment
            candidate_height = height_units * alignment
            multiplier = (candidate_width * candidate_height) / target_pixels
            if not 0.8 <= multiplier <= 1.2:
                continue
            area_error = abs(math.log(multiplier))
            ratio_error = abs(math.log((candidate_width / candidate_height) / ratio))
            # Prefer the requested composition, while still keeping compute
            # near the selected profile.  This makes low-resolution 16:9
            # choose 448x256 instead of preserving the profile's 2:1 canvas.
            candidates.append(
                (area_error + 2 * ratio_error, area_error, candidate_width, candidate_height)
            )
    if not candidates:  # pragma: no cover - guards future extreme presets
        raise ValueError(
            f"aspect ratio {aspect_ratio} cannot preserve the profile pixel budget on the {alignment}-pixel grid"
        )
    _, _, resolved_width, resolved_height = min(candidates)
    return resolved_width, resolved_height

MIN_FREE_GIB = 6.0
MAX_CONDITIONING_IMAGES = 4
MAX_DURATION_SECONDS = 10
MAX_RENDER_PROMPT_CHARS = 110_000
MAX_RENDER_PROMPT_BYTES = 288 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LONG_CLIP_WARNING = (
    "LONG-CLIP WARNING: more than 5 seconds sharply increases attention memory and the chance of "
    "identity/motion drift on a 16 GB Mac. Prefer separate 3-second takes, then continue them with "
    "the optional Q8 Extend lane when it is installed and ready. This direct attempt remains enabled "
    "as requested."
)


def validate_render_prompt(prompt: str, *, label: str = "the prompt") -> str:
    """Reject unsafe process-sized input without changing a single character."""
    if not prompt.strip():
        raise ValueError(f"{label} cannot be empty")
    if "\x00" in prompt:
        raise ValueError(f"{label} contains an unsupported NUL character")
    if len(prompt) > MAX_RENDER_PROMPT_CHARS or len(prompt.encode("utf-8")) > MAX_RENDER_PROMPT_BYTES:
        raise ValueError(
            f"{label} exceeds the local process safety ceiling "
            f"({MAX_RENDER_PROMPT_CHARS:,} characters / {MAX_RENDER_PROMPT_BYTES // 1024} KiB UTF-8); "
            "it was rejected without truncation"
        )
    return prompt


def format_command_for_log(command: list[str] | tuple[str, ...]) -> str:
    """Render a copyable command summary without flooding logs with prompt text."""
    display = list(command)
    try:
        index = display.index("--prompt") + 1
        prompt = display[index]
    except (ValueError, IndexError):
        return shlex.join(display)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    display[index] = (
        f"<prompt preserved: {len(prompt):,} chars, {len(prompt.encode('utf-8')):,} bytes, "
        f"sha256:{digest}>"
    )
    return shlex.join(display)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fully local LTX-2.5 q4 video with native audio. "
            "Uses the distilled two-stage path and conservative 16 GB defaults."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", help="detailed description of one coherent shot")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quality")
    parser.add_argument(
        "--aspect-ratio",
        choices=tuple(ASPECT_RATIO_PRESETS),
        default="profile",
        help=(
            "output shape; profile keeps the original profile canvas, while named ratios use the nearest "
            "64-pixel LTX grid at approximately the same pixel budget"
        ),
    )
    parser.add_argument(
        "--gemma-pack",
        choices=("official", "ddalcu-fallback"),
        default="official",
        help=(
            "text encoder pack; official is preferred. ddalcu-fallback is accepted only after "
            "the separate revision-pinned compatibility importer succeeds"
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        choices=range(1, MAX_DURATION_SECONDS + 1),
        metavar="SECONDS",
        help=(
            "override the profile length with an integer 1 to 10 seconds; maps to the nearest exact "
            "LTX grid as 24*SECONDS+1 frames (for example, 3 -> 73 / 3.04 s)"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="deterministic seed (0 to 4294967295)")
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 path; a relative path is resolved inside the Studio output directory",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="backward-compatible alias for --start-image (frame 0)",
    )
    parser.add_argument("--start-image", type=Path, help="hard first-frame reference image (frame 0)")
    parser.add_argument(
        "--start-strength",
        type=float,
        help="conditioning strength for --image/--start-image, from 0 to 1",
    )
    parser.add_argument(
        "--end-image",
        type=Path,
        help="soft final-frame reference image (resolved to the selected profile's last frame)",
    )
    parser.add_argument(
        "--end-strength",
        type=float,
        help="conditioning strength for --end-image, from 0 to 1",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        nargs=3,
        default=[],
        metavar=("PATH", "FRAME", "STRENGTH"),
        help="repeatable soft keyframe reference; FRAME is 0-based and STRENGTH is 0 to 1",
    )
    parser.add_argument(
        "--scene",
        action=_SceneAction,
        dest="scenes",
        metavar="TEXT",
        help=(
            "repeatable Prompt Relay scene beat, in timeline order; the global prompt still applies "
            "to every frame"
        ),
    )
    parser.add_argument(
        "--scene-length",
        action=_SceneLengthAction,
        type=int,
        metavar="LATENT_FRAMES",
        help=(
            "optional positive latent-frame length for the immediately preceding --scene; omit it "
            "to auto-distribute that beat"
        ),
    )
    parser.add_argument(
        "--generated-keyframes",
        type=int,
        choices=(0, 1, 2),
        default=0,
        metavar="{0,1,2}",
        help=(
            "compatibility option; this pinned distilled runtime supports 0 only. "
            "Use timed --anchor images for motion structure"
        ),
    )
    parser.add_argument("--force", action="store_true", help="replace the exact --output file if it exists")
    parser.add_argument(
        "--schedule-preset",
        choices=("default", "fast"),
        help="override the profile schedule: default is the graded 8+2 path; fast is 5+2",
    )
    parser.add_argument(
        "--steps",
        type=int,
        choices=(5, 8),
        help="backward-compatible schedule selector: 5 maps to fast, 8 maps to default",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        help="unsupported compatibility option; distilled inference has no CFG",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        help="unsupported compatibility option; distilled inference has no STG",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="reserved compatibility flag; unsupported by this pinned distilled CLI",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress upstream progress output")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="SHA-256 all model weights during preflight (reads about 25.6 GiB)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify installation and print the resolved render command without generating",
    )
    return parser.parse_args()


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def _image_path(value: str | Path, *, option: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        _fail(f"{option} does not exist or is not a file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        _fail(f"unsupported image extension for {option}: {path.suffix or '(none)'}")
    return path


def _strength(value: float, *, option: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _fail(f"{option} must be a finite number between 0 and 1")
    return value


def _conditioning_images(args: argparse.Namespace, profile: Profile) -> list[ConditioningImage]:
    """Normalize all image aliases to upstream's repeatable image contract."""
    legacy_image = getattr(args, "image", None)
    start_image = getattr(args, "start_image", None)
    start_strength_arg = getattr(args, "start_strength", None)
    end_image = getattr(args, "end_image", None)
    end_strength_arg = getattr(args, "end_strength", None)
    raw_anchors = getattr(args, "anchor", None) or []

    if legacy_image is not None and start_image is not None:
        _fail("--image and --start-image both target frame 0; choose one alias")
    selected_start = start_image if start_image is not None else legacy_image
    if start_strength_arg is not None and selected_start is None:
        _fail("--start-strength requires --image or --start-image")
    if end_strength_arg is not None and end_image is None:
        _fail("--end-strength requires --end-image")

    conditionings: list[ConditioningImage] = []
    if selected_start is not None:
        option = "--start-image" if start_image is not None else "--image"
        strength = 1.0 if start_strength_arg is None else _strength(start_strength_arg, option="--start-strength")
        conditionings.append(ConditioningImage(_image_path(selected_start, option=option), 0, strength))

    for index, raw_anchor in enumerate(raw_anchors, start=1):
        if len(raw_anchor) != 3:
            _fail(f"--anchor #{index} must contain PATH FRAME STRENGTH")
        raw_path, raw_frame, raw_strength = raw_anchor
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError):
            _fail(f"--anchor #{index} FRAME must be an integer, got {raw_frame!r}")
        try:
            strength_value = float(raw_strength)
        except (TypeError, ValueError):
            _fail(f"--anchor #{index} STRENGTH must be a number, got {raw_strength!r}")
        if not 0 <= frame < profile.frames:
            _fail(f"--anchor #{index} FRAME must be between 0 and {profile.frames - 1} for {args.profile}")
        conditionings.append(
            ConditioningImage(
                _image_path(raw_path, option=f"--anchor #{index}"),
                frame,
                _strength(strength_value, option=f"--anchor #{index} STRENGTH"),
            )
        )

    if end_image is not None:
        strength = 1.0 if end_strength_arg is None else _strength(end_strength_arg, option="--end-strength")
        conditionings.append(
            ConditioningImage(
                _image_path(end_image, option="--end-image"),
                profile.frames - 1,
                strength,
            )
        )

    if len(conditionings) > MAX_CONDITIONING_IMAGES:
        _fail(
            f"at most {MAX_CONDITIONING_IMAGES} conditioning images are allowed on this 16 GB wrapper; "
            f"received {len(conditionings)}"
        )

    by_frame: dict[int, ConditioningImage] = {}
    for conditioning in conditionings:
        if conditioning.frame == 0 and 0 in by_frame:
            _fail("at most one conditioning image may replace frame 0")
        if conditioning.frame in by_frame:
            first = by_frame[conditioning.frame]
            _fail(
                f"multiple conditioning images target frame {conditioning.frame}: "
                f"{first.path.name!r} and {conditioning.path.name!r}; frame indices must be unique"
            )
        by_frame[conditioning.frame] = conditioning
    return sorted(conditionings, key=lambda item: item.frame)


def _distributed_scene_lengths(scenes: list[SceneBeat], latent_frames: int) -> list[int]:
    """Resolve auto beats exactly, avoiding upstream's ceil-and-clamp edge case."""
    specified = [scene.length for scene in scenes]
    auto_count = sum(length is None for length in specified)
    if not auto_count:
        return [int(length) for length in specified]

    pinned_total = sum(length for length in specified if length is not None)
    remaining = latent_frames - pinned_total
    if remaining <= 0:
        auto_lengths = [0] * auto_count
    else:
        quotient, extra = divmod(remaining, auto_count)
        auto_lengths = [quotient + (index < extra) for index in range(auto_count)]

    auto_iter = iter(auto_lengths)
    return [next(auto_iter) if length is None else int(length) for length in specified]


def _validate_scenes(args: argparse.Namespace, profile: Profile) -> list[SceneBeat]:
    scenes = list(getattr(args, "scenes", None) or [])
    if not scenes:
        args.scenes = []
        args.resolved_scene_lengths = []
        return []

    normalized: list[SceneBeat] = []
    for index, scene in enumerate(scenes, start=1):
        text = scene.text.strip()
        if not text:
            _fail(f"--scene #{index} cannot be empty")
        if scene.length is not None and (
            not isinstance(scene.length, int) or isinstance(scene.length, bool) or scene.length <= 0
        ):
            _fail(f"--scene-length for --scene #{index} must be a positive integer")
        normalized.append(SceneBeat(text=text, length=scene.length))

    combined_prompt = " ".join([args.prompt, *(scene.text for scene in normalized)])
    try:
        validate_render_prompt(combined_prompt, label="the global prompt plus --scene text")
    except ValueError as exc:
        _fail(str(exc))

    # LTX's convolutional video VAE maps F pixel frames to ceil(F / 8)
    # latent frames.  Prompt Relay lengths are expressed on that latent axis.
    latent_frames = (profile.frames + 7) // 8
    effective = _distributed_scene_lengths(normalized, latent_frames)
    if sum(effective) > latent_frames:
        _fail(
            f"--scene lengths require {sum(effective)} latent frames, but profile {args.profile} "
            f"has only {latent_frames}"
        )
    for index, (scene, actual) in enumerate(zip(normalized, effective), start=1):
        if actual <= 0:
            _fail(
                f"--scene #{index} resolves to zero latent frames for profile {args.profile} "
                f"({latent_frames} available); shorten earlier lengths or remove a scene"
            )

    args.scenes = normalized
    args.resolved_scene_lengths = effective
    return normalized


def _validate_args(args: argparse.Namespace) -> tuple[Profile, str, list[ConditioningImage]]:
    try:
        validate_render_prompt(args.prompt)
    except ValueError as exc:
        _fail(str(exc))

    if not 0 <= args.seed <= 0xFFFFFFFF:
        _fail("--seed must be between 0 and 4294967295")

    profile = PROFILES[args.profile]
    if args.aspect_ratio != "profile":
        width, height = resolve_aspect_dimensions(profile.width, profile.height, args.aspect_ratio)
        profile = replace(
            profile,
            width=width,
            height=height,
            description=f"{profile.description}; {args.aspect_ratio} canvas on the safe LTX grid",
        )
    duration_seconds = getattr(args, "duration_seconds", None)
    if duration_seconds is not None:
        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or not 1 <= duration_seconds <= MAX_DURATION_SECONDS
        ):
            _fail(f"--duration-seconds must be an integer from 1 to {MAX_DURATION_SECONDS}")
        profile = replace(
            profile,
            frames=24 * duration_seconds + 1,
            description=(
                f"{profile.description}; duration override {duration_seconds}s "
                f"({24 * duration_seconds + 1} frames / {duration_seconds + 1 / 24:.2f}s encoded)"
            ),
        )
    schedule_preset = getattr(args, "schedule_preset", None) or profile.schedule_preset
    legacy_steps = getattr(args, "steps", None)
    if legacy_steps is not None:
        mapped = {5: "fast", 8: "default"}.get(legacy_steps)
        if mapped is None:
            _fail("--steps supports only 5 (fast 5+2) or 8 (graded default 8+2) on distilled LTX-2.5")
        explicit_preset = getattr(args, "schedule_preset", None)
        if explicit_preset is not None and explicit_preset != mapped:
            _fail("--steps and --schedule-preset select different distilled schedules")
        schedule_preset = mapped
    if getattr(args, "cfg_scale", None) is not None:
        _fail("--cfg-scale is unavailable on the distilled checkpoint; use --schedule-preset instead")
    if getattr(args, "stg_scale", None) is not None:
        _fail("--stg-scale is unavailable on the distilled checkpoint; use --schedule-preset instead")
    generated_keyframes = getattr(args, "generated_keyframes", 0)
    if (
        not isinstance(generated_keyframes, int)
        or isinstance(generated_keyframes, bool)
        or generated_keyframes not in (0, 1, 2)
    ):
        _fail("--generated-keyframes must be one of 0, 1, or 2")
    if generated_keyframes:
        _fail(
            "generated keyframe slots are not exposed by the pinned distilled runtime; "
            "use --start-image, --anchor, and --end-image instead"
        )
    if getattr(args, "no_audio", False):
        _fail("--no-audio is unavailable in this pinned distilled CLI; native audio is always rendered")

    _validate_scenes(args, profile)
    conditionings = _conditioning_images(args, profile)
    return profile, schedule_preset, conditionings


def _resolve_output(args: argparse.Namespace) -> Path:
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = PATHS.edit_root / f"ltx25-{args.profile}-seed{args.seed}-{timestamp}.mp4"
    else:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = PATHS.outputs_root / output
    output = output.resolve()
    if output.suffix.lower() != ".mp4":
        _fail(f"--output must end in .mp4: {output}")
    if output.exists():
        if output.is_dir():
            _fail(f"output path is a directory: {output}")
        if not args.force:
            _fail(f"output already exists: {output} (pass --force to replace this exact file)")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _offline_environment() -> dict[str, str]:
    cache_root = PATHS.cache_root / "ltx25"
    cache_root.mkdir(parents=True, exist_ok=True)
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
            # Keep prompt encoding bounded; 1024 is the runtime/model default.
            "LTX2_GEMMA_MAX_LENGTH": "1024",
            # Force conservative automatic VAE tiling on this 16 GB machine.
            # The untiled 768x448x49 conv decode is estimated around 11.8 GiB.
            "LTX2_VAE_DECODE_BUDGET_GB": "7",
        }
    )
    return env


def _build_command(
    args: argparse.Namespace,
    *,
    profile: Profile,
    schedule_preset: str,
    output: Path,
    conditionings: list[ConditioningImage],
) -> list[str]:
    gemma_model = (
        DDALCU_FALLBACK_GEMMA_MODEL
        if getattr(args, "gemma_pack", "official") == "ddalcu-fallback"
        else GEMMA_MODEL
    )
    command = [
        str(CLI),
        "generate",
        "--model",
        str(MODEL),
        "--gemma",
        str(gemma_model),
        "--distilled",
        "--low-ram",
        "--prompt",
        args.prompt,
        "--output",
        str(output),
        "--height",
        str(profile.height),
        "--width",
        str(profile.width),
        "--frames",
        str(profile.frames),
        "--frame-rate",
        "24",
        "--seed",
        str(args.seed),
        "--schedule-preset",
        schedule_preset,
    ]
    for conditioning in conditionings:
        command.extend(
            ["--image", str(conditioning.path), str(conditioning.frame), str(conditioning.strength)]
        )
    scenes = getattr(args, "scenes", None) or []
    resolved_scene_lengths = getattr(args, "resolved_scene_lengths", [])
    for scene, length in zip(scenes, resolved_scene_lengths):
        command.extend(["--segment", scene.text, str(length)])
    if args.quiet:
        command.append("--quiet")
    return command


def main() -> int:
    args = _parse_args()
    profile, schedule_preset, conditionings = _validate_args(args)
    output = _resolve_output(args)

    gemma_model = (
        DDALCU_FALLBACK_GEMMA_MODEL if args.gemma_pack == "ddalcu-fallback" else GEMMA_MODEL
    )
    try:
        verify_install(full_hash=args.full_hash, require_gpu=True, gemma_model=gemma_model)
    except (VerificationError, json.JSONDecodeError, OSError) as exc:
        print(f"PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 2

    free_gib = shutil.disk_usage(PATHS.data_root).free / 2**30
    if free_gib < MIN_FREE_GIB:
        print(
            f"PRE-FLIGHT FAILED: only {free_gib:.1f} GiB is free; keep at least {MIN_FREE_GIB:.0f} GiB "
            "for macOS swap, decode intermediates, and the output video.",
            file=sys.stderr,
        )
        return 2
    if free_gib < 10:
        print(f"WARNING: only {free_gib:.1f} GiB is free; close apps and avoid other large downloads.", file=sys.stderr)

    # Render beside the destination and atomically promote only a successful,
    # non-empty MP4.  With --force this preserves the previous result if the
    # long render fails or macOS kills it for memory pressure.
    render_output = output.with_name(f".{output.stem}.ltx25-{os.getpid()}.partial.mp4")
    if render_output.exists():
        _fail(f"temporary render path already exists: {render_output}")
    command = _build_command(
        args,
        profile=profile,
        schedule_preset=schedule_preset,
        output=render_output,
        conditionings=conditionings,
    )
    print(
        f"LTX-2.5 q4 distilled {args.profile}: {profile.width}x{profile.height}, "
        f"{profile.frames} frames @ 24 fps, schedule={schedule_preset} "
        f"({'5+2' if schedule_preset == 'fast' else '8+2'}), seed={args.seed}"
    )
    print(f"Profile intent: {profile.description}")
    if args.gemma_pack == "ddalcu-fallback":
        print(
            "FALLBACK NOTICE: using the separately validated ddalcu tensor serialization; "
            "the official Phosphene Gemma pack remains preferred.",
            file=sys.stderr,
        )
    duration_seconds = getattr(args, "duration_seconds", None)
    if duration_seconds is not None and duration_seconds > 5:
        print(LONG_CLIP_WARNING, file=sys.stderr)
    nonzero_anchor_count = sum(item.frame != 0 for item in conditionings)
    if conditionings:
        summary = ", ".join(
            f"frame {item.frame}={item.path.name} (strength {item.strength:g})" for item in conditionings
        )
        print(f"Conditioning: {summary}")
        if nonzero_anchor_count:
            print(
                f"Memory note: {nonzero_anchor_count} nonzero anchor(s) add keyframe tokens; "
                "keep other apps closed on this 16 GB Mac.",
                file=sys.stderr,
            )
    scenes = getattr(args, "scenes", None) or []
    if scenes:
        resolved = getattr(args, "resolved_scene_lengths", [])
        summary = ", ".join(
            f"{index}:{length} latent frame{'s' if length != 1 else ''}"
            for index, length in enumerate(resolved, start=1)
        )
        print(f"Prompt Relay: {len(scenes)} scene beat(s) ({summary})")
    print(f"Output: {output}")
    print("Command: " + format_command_for_log(command))
    if args.preflight_only:
        print("PRE-FLIGHT VERIFIED; render not started (--preflight-only).")
        return 0

    try:
        completed = subprocess.run(command, cwd=RUNTIME, env=_offline_environment(), check=False)
    except OSError as exc:
        print(f"RENDER FAILED: could not start the pinned runtime: {exc}", file=sys.stderr)
        return 3
    if completed.returncode:
        if render_output.exists():
            render_output.unlink()
        print(
            f"RENDER FAILED (exit {completed.returncode}). No successful output was claimed. "
            "The incomplete temporary MP4 was removed. Close memory-heavy apps and retry the draft "
            "profile if macOS killed the process.",
            file=sys.stderr,
        )
        return completed.returncode
    if not render_output.is_file() or render_output.stat().st_size == 0:
        if render_output.exists():
            render_output.unlink()
        print(
            f"RENDER FAILED: runtime returned success but did not create a non-empty file at {render_output}",
            file=sys.stderr,
        )
        return 3
    os.replace(render_output, output)
    print(f"VIDEO READY: {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
