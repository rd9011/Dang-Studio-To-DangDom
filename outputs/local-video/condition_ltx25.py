#!/usr/bin/env python3
"""Safe local LTX-2.5 IC-LoRA workflows for Ingredients and controls.

This wrapper deliberately keeps preprocessing local and dependency-light:

* ``ingredients`` composes several still references into one reference sheet
  and loops it into the static guide video expected by the official 2.5
  Ingredients adapter.
* ``union`` accepts a prepared depth/canny/pose guide, or derives a Canny-like
  edge guide from a local source video with ffmpeg.
* ``motion`` converts normalized spline control points into the sparse coloured
  trail video used by the official Motion Track adapter.

All three modes use the separate q8 distilled transformer, single-stage IC-LoRA,
``--low-ram``, offline execution, confined atomic outputs, and conservative
three-second defaults for a 16 GB Apple Silicon Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence

from studio_paths import PATHS

from PIL import Image, ImageDraw, ImageOps

from generate_ltx25 import (
    ASPECT_RATIO_PRESETS,
    format_command_for_log,
    resolve_aspect_dimensions,
    validate_render_prompt,
)
from install_ltx25_advanced_q8 import (
    CONTROL_RECEIPT_NAME,
    OverlayError,
    verify_installed_control_overlay,
)
from verify_ltx25 import CLI, GEMMA_MODEL, RUNTIME, VerificationError, verify_install


FRAME_RATE = 24.0
OUTPUT_ROOT = PATHS.generated_root.resolve()
CONTROL_ROOT = PATHS.models_root / "LTX-2.5-control-adapters"
HQ_MODEL = PATHS.models_root / "ltx-2.5-mlx-q8"
HQ_RECEIPT = HQ_MODEL / ".ltx25-q8-overlay-receipt.json"
CONTROL_RECEIPT = HQ_MODEL / CONTROL_RECEIPT_NAME
DISTILLED_TRANSFORMER = HQ_MODEL / "transformer-distilled.safetensors"

DISTILLED_SIZE = 20_595_260_317
DISTILLED_SHA256 = "36a346c121235d3309450104df8c2de650bce76759b53abc3e9136a08284cdde"

# Keep this sequence in lockstep with LTXV_LORA_COMFY_RENAMING_MAP in the
# pinned MLX runtime.  Importing that module here would import mlx.core, which
# would defeat this pre-GPU/pre-MLX compatibility check.
LORA_KEY_REPLACEMENTS = (
    ("diffusion_model.", ""),
    (".to_out.0.", ".to_out."),
    (".ff.net.0.proj.", ".ff.proj_in."),
    (".ff.net.2.", ".ff.proj_out."),
    (".linear_1.", ".linear1."),
    (".linear_2.", ".linear2."),
    ("audio_ff.net.0.proj.", "audio_ff.proj_in."),
    ("audio_ff.net.2.", "audio_ff.proj_out."),
)
LORA_BLOCK_PREFIX = "transformer_blocks."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


@dataclass(frozen=True)
class Asset:
    filename: str
    size: int
    sha256: str | None
    repo: str
    revision: str


ASSETS: dict[str, Asset] = {
    "ingredients": Asset(
        filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        size=1_308_787_472,
        sha256="ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95",
        repo="Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        revision="12040e4091ac2008d3906a594e31a7fb1ab9d546",
    ),
    "union": Asset(
        filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        size=654_465_352,
        sha256="a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18",
        repo="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        revision="b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff",
    ),
    "motion": Asset(
        filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        size=327_309_314,
        sha256="e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca",
        repo="Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        revision="572bb9c9a1ba3d8e8724cce69783ffc2422386db",
    ),
}


@dataclass(frozen=True)
class Profile:
    width: int
    height: int
    frames: int
    steps: int
    description: str


PROFILES: dict[str, Profile] = {
    "control3s": Profile(384, 256, 73, 8, "lowest-risk 3.04 s Union/Motion iteration"),
    "control3s-plus": Profile(512, 320, 73, 8, "higher-detail 3.04 s Union/Motion iteration"),
    "motion3s-experimental": Profile(
        640,
        384,
        73,
        8,
        "EXPERIMENTAL 3.04 s Motion Track canvas; high memory risk on 16 GB",
    ),
    "ingredients5s": Profile(
        384,
        224,
        121,
        8,
        "5.04 s Ingredients duration at reduced 12:7 resolution for 16 GB",
    ),
    "trained5s": Profile(
        768,
        448,
        121,
        8,
        "official Ingredients training bucket; high memory and experimental on 16 GB",
    ),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
MAX_REFERENCES = 8
MAX_TRACKS = 8
MAX_TRACK_POINTS = 16
MAX_DURATION_SECONDS = 10


class UserInputError(ValueError):
    """A request is invalid before loading any model."""


class PreflightError(RuntimeError):
    """The pinned local assets or runtime are not ready."""


@dataclass(frozen=True)
class Reference:
    path: Path
    label: str
    description: str


@dataclass(frozen=True)
class ImageConditioning:
    path: Path
    frame: int
    strength: float


@dataclass(frozen=True)
class RenderPlan:
    mode: str
    profile: Profile
    prompt: str
    output: Path
    temporary_output: Path
    adapter: Path
    guide_placeholder: Path
    command: tuple[str, ...]
    references: tuple[Reference, ...] = ()
    source: Path | None = None
    start_image: Path | None = None
    tracks: tuple[tuple[tuple[float, float], ...], ...] = ()
    image_conditionings: tuple[ImageConditioning, ...] = ()


@dataclass(frozen=True)
class LoraKeyCoverage:
    """GPU-free summary of LoRA pairs the streaming runtime can consume."""

    blocks: tuple[int, ...]
    pairs: int
    recognized_tensors: int


def _fail(message: str) -> NoReturn:
    raise UserInputError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully local LTX-2.5 q8 Ingredients, Union Control, and Motion Track workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    ingredients = sub.add_parser("ingredients", help="generate from a multi-image reference sheet")
    _add_common_args(ingredients, default_profile="ingredients5s")
    ingredients.add_argument(
        "--reference",
        action="append",
        nargs=3,
        required=True,
        metavar=("IMAGE", "LABEL", "DESCRIPTION"),
        help="reference panel; repeat for several views/characters/props/locations",
    )

    union = sub.add_parser("union", help="structure control from canny/depth/pose guide video")
    _add_common_args(union, default_profile="control3s")
    union.add_argument("--control-type", choices=("canny", "depth", "pose", "prepared"), default="canny")
    control_input = union.add_mutually_exclusive_group(required=True)
    control_input.add_argument(
        "--source-video",
        type=Path,
        help="ordinary source footage; built-in preprocessing is currently available for Canny only",
    )
    control_input.add_argument(
        "--guide-video",
        type=Path,
        help="already annotated depth/canny/pose guide video",
    )

    motion = sub.add_parser("motion", help="animate a still along locally drawn sparse motion tracks")
    _add_common_args(motion, default_profile="control3s")
    motion.add_argument(
        "--tracks-json",
        type=Path,
        required=True,
        help="JSON array of tracks; each track is an array of normalized {x,y} control points",
    )
    motion.add_argument(
        "--image-strength",
        type=float,
        default=None,
        help="backward-compatible alias for --start-strength (default: 1.0)",
    )
    return parser


def _add_common_args(parser: argparse.ArgumentParser, *, default_profile: str) -> None:
    parser.add_argument("prompt", help="description of the generated action and shot")
    parser.add_argument("--profile", choices=tuple(PROFILES), default=default_profile)
    parser.add_argument(
        "--aspect-ratio",
        choices=tuple(ASPECT_RATIO_PRESETS),
        default="profile",
        help=(
            "output shape on the nearest safe LTX grid; Profile default is recommended for "
            "Ingredients and its other aspect presets are experimental"
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        choices=range(1, MAX_DURATION_SECONDS + 1),
        metavar="SECONDS",
        help=(
            "Union/Motion length override from 1 to 10 seconds (24*SECONDS+1 frames); "
            "Ingredients is fixed to its required 121-frame / 5.04-second bucket"
        ),
    )
    parser.add_argument("--output", type=Path, help="MP4 name/path confined to outputs/local-video/generated")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-strength", type=float, default=1.0)
    parser.add_argument("--start-image", type=Path, help="optional opening-frame anchor (required by motion)")
    parser.add_argument("--start-strength", type=float, help="opening-frame anchor strength (0 to 1)")
    parser.add_argument("--end-image", type=Path, help="optional final-frame anchor")
    parser.add_argument("--end-strength", type=float, help="final-frame anchor strength (0 to 1)")
    parser.add_argument(
        "--anchor",
        action="append",
        nargs=3,
        default=[],
        metavar=("IMAGE", "FRAME", "STRENGTH"),
        help="repeatable interior pixel-frame anchor; combines with any one IC-LoRA mode",
    )
    parser.add_argument("--force", action="store_true", help="replace the exact output atomically")
    parser.add_argument("--keep-guide", action="store_true", help="save the generated sheet/guide beside the video")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--full-hash", action="store_true", help="SHA-256 base and optional control weights")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--dry-run", action="store_true", help="validate and print a command without asset/GPU checks")
    run_mode.add_argument("--preflight-only", action="store_true", help="check assets, tools, disk, and Metal; do not render")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _finite(value: float, *, option: str, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        _fail(f"{option} must be a finite number between {minimum:g} and {maximum:g}")
    return value


def _local_file(value: Path | str, *, option: str, extensions: set[str]) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        _fail(f"{option} cannot be resolved: {value} ({exc})")
    if not path.is_file():
        _fail(f"{option} is not a local file: {path}")
    if path.suffix.lower() not in extensions:
        _fail(f"{option} has unsupported extension {path.suffix or '(none)'}")
    return path


def _clean_text(value: str, *, option: str, maximum: int) -> str:
    text = value.strip()
    if not text:
        _fail(f"{option} cannot be empty")
    if len(text) > maximum or "\x00" in text:
        _fail(f"{option} is too long or contains a NUL character")
    return text


def _resolve_output(value: Path | None, *, mode: str, seed: int, force: bool) -> tuple[Path, Path]:
    output_root = OUTPUT_ROOT.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if value is None:
        value = Path(f"ltx25-{mode}-{seed}.mp4")
    candidate = value if value.is_absolute() else output_root / value
    candidate = candidate.expanduser().resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError:
        _fail(f"--output must stay inside {output_root}")
    if candidate.suffix.lower() != ".mp4":
        _fail("--output must end in .mp4")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists() and not force:
        _fail(f"output already exists (use --force to replace exactly this file): {candidate}")
    temporary = candidate.with_name(f".{candidate.stem}.ltx25-control-{os.getpid()}.partial.mp4")
    if temporary.exists():
        _fail(f"temporary output already exists: {temporary}")
    return candidate, temporary


def _load_tracks(path: Path) -> tuple[tuple[tuple[float, float], ...], ...]:
    source = _local_file(path, option="--tracks-json", extensions={".json"})
    if source.stat().st_size > 1_000_000:
        _fail("--tracks-json is unexpectedly large (maximum 1 MB)")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"--tracks-json is not valid UTF-8 JSON: {exc}")
    if not isinstance(raw, list) or not raw:
        _fail("--tracks-json must contain at least one track")
    if len(raw) > MAX_TRACKS:
        _fail(f"--tracks-json supports at most {MAX_TRACKS} tracks")
    tracks: list[tuple[tuple[float, float], ...]] = []
    for track_index, track in enumerate(raw, start=1):
        if not isinstance(track, list) or not 2 <= len(track) <= MAX_TRACK_POINTS:
            _fail(f"track {track_index} must contain 2 to {MAX_TRACK_POINTS} control points")
        points: list[tuple[float, float]] = []
        for point_index, point in enumerate(track, start=1):
            if not isinstance(point, dict) or set(point) != {"x", "y"}:
                _fail(f"track {track_index} point {point_index} must contain only x and y")
            try:
                x, y = float(point["x"]), float(point["y"])
            except (TypeError, ValueError):
                _fail(f"track {track_index} point {point_index} coordinates must be numbers")
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                _fail(f"track {track_index} point {point_index} coordinates must be normalized from 0 to 1")
            points.append((x, y))
        tracks.append(tuple(points))
    return tuple(tracks)


def _ingredients_prompt(prompt: str, references: Sequence[Reference]) -> str:
    panels = []
    for index, reference in enumerate(references, start=1):
        detail = reference.description or reference.label
        panels.append(f"Panel {index} shows {reference.label}: {detail}")
    return "Reference sheet: " + "; ".join(panels) + "\n\nGenerated video: " + prompt


def _plan_image_conditionings(args: argparse.Namespace, profile: Profile) -> tuple[ImageConditioning, ...]:
    conditionings: list[ImageConditioning] = []
    start_image: Path | None = None
    if args.start_image is not None:
        start_image = _local_file(args.start_image, option="--start-image", extensions=IMAGE_EXTENSIONS)
        motion_alias = getattr(args, "image_strength", None)
        if args.start_strength is not None and motion_alias is not None and args.start_strength != motion_alias:
            _fail("use either --start-strength or --image-strength for motion, not conflicting values")
        start_strength = args.start_strength if args.start_strength is not None else motion_alias
        conditionings.append(
            ImageConditioning(
                start_image,
                0,
                _finite(1.0 if start_strength is None else start_strength, option="--start-strength", minimum=0, maximum=1),
            )
        )
    elif args.start_strength is not None:
        _fail("--start-strength requires --start-image")

    for index, raw in enumerate(args.anchor, start=1):
        path = _local_file(raw[0], option=f"--anchor {index}", extensions=IMAGE_EXTENSIONS)
        try:
            frame = int(raw[1])
            strength = float(raw[2])
        except ValueError:
            _fail(f"--anchor {index} FRAME must be an integer and STRENGTH must be numeric")
        if not 0 <= frame < profile.frames:
            _fail(f"--anchor {index} frame must be between 0 and {profile.frames - 1}")
        conditionings.append(
            ImageConditioning(path, frame, _finite(strength, option=f"--anchor {index} strength", minimum=0, maximum=1))
        )

    if args.end_image is not None:
        end_image = _local_file(args.end_image, option="--end-image", extensions=IMAGE_EXTENSIONS)
        end_strength = _finite(
            1.0 if args.end_strength is None else args.end_strength,
            option="--end-strength",
            minimum=0,
            maximum=1,
        )
        conditionings.append(ImageConditioning(end_image, profile.frames - 1, end_strength))
    elif args.end_strength is not None:
        _fail("--end-strength requires --end-image")

    if len(conditionings) > 4:
        _fail("use at most four start/interior/end image anchors with an IC-LoRA mode")
    frames = [item.frame for item in conditionings]
    if len(frames) != len(set(frames)):
        _fail("start, interior, and end image anchors must use unique frame indices")
    return tuple(conditionings)


def build_plan(args: argparse.Namespace) -> RenderPlan:
    try:
        prompt = validate_render_prompt(args.prompt)
    except ValueError as exc:
        _fail(str(exc))
    if not 0 <= args.seed <= 0xFFFFFFFF:
        _fail("--seed must be between 0 and 4294967295")
    _finite(args.lora_strength, option="--lora-strength", minimum=0, maximum=2)
    _finite(args.conditioning_strength, option="--conditioning-strength", minimum=0, maximum=1)
    profile = PROFILES[args.profile]
    if args.aspect_ratio != "profile":
        # All three workflows force the upstream single-stage IC-LoRA
        # topology, whose full-resolution VAE grid is 32 pixels. Reusing the
        # 64-pixel two-stage grid can materially miss a requested ratio (for
        # example, control3s 4:3 would otherwise remain 384x256 / 3:2).
        width, height = resolve_aspect_dimensions(
            profile.width,
            profile.height,
            args.aspect_ratio,
            alignment=32,
        )
        if args.mode == "ingredients":
            aspect_description = (
                f"EXPERIMENTAL Ingredients {args.aspect_ratio} canvas on the safe LTX grid; "
                "the profile-default landscape canvas remains recommended because other "
                "aspects may reduce identity and composition consistency"
            )
        else:
            aspect_description = f"{args.aspect_ratio} canvas on the safe LTX grid"
        profile = replace(
            profile,
            width=width,
            height=height,
            description=f"{profile.description}; {aspect_description}",
        )
    duration_seconds = getattr(args, "duration_seconds", None)
    if duration_seconds is not None:
        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or not 1 <= duration_seconds <= MAX_DURATION_SECONDS
        ):
            _fail(f"--duration-seconds must be an integer from 1 to {MAX_DURATION_SECONDS}")
        if args.mode == "ingredients" and duration_seconds != 5:
            _fail("Ingredients duration is fixed at 121 frames / 5.04 seconds; select 5 seconds")
        if args.mode != "ingredients":
            profile = replace(
                profile,
                frames=24 * duration_seconds + 1,
                description=(
                    f"{profile.description}; duration override {duration_seconds}s "
                    f"({24 * duration_seconds + 1} frames / {duration_seconds + 1 / 24:.2f}s encoded)"
                ),
            )
    if args.mode == "ingredients" and args.profile not in {"ingredients5s", "trained5s"}:
        _fail(
            "the official LTX-2.5 Ingredients adapter requires at least 121 output/reference frames; "
            "use --profile ingredients5s (lower resolution) or trained5s (official 768x448 bucket)"
        )
    if args.mode == "union" and args.profile not in {"control3s", "control3s-plus"}:
        _fail("Union requires --profile control3s or control3s-plus")
    if args.mode == "motion" and args.profile not in {
        "control3s",
        "control3s-plus",
        "motion3s-experimental",
    }:
        _fail(
            "Motion Track requires --profile control3s, control3s-plus, or "
            "motion3s-experimental"
        )
    if args.profile == "motion3s-experimental" and duration_seconds not in {None, 3}:
        _fail("motion3s-experimental is restricted to 3 seconds on this 16 GB Mac")
    output, temporary = _resolve_output(args.output, mode=args.mode, seed=args.seed, force=args.force)
    adapter_asset = ASSETS[args.mode]
    adapter = CONTROL_ROOT / adapter_asset.filename
    guide = Path(f"<generated-{args.mode}-guide.mp4>")
    references: tuple[Reference, ...] = ()
    source: Path | None = None
    start_image: Path | None = None
    tracks: tuple[tuple[tuple[float, float], ...], ...] = ()

    effective_prompt = prompt
    if args.mode == "ingredients":
        if not 1 <= len(args.reference) <= MAX_REFERENCES:
            _fail(f"ingredients requires 1 to {MAX_REFERENCES} --reference entries")
        refs: list[Reference] = []
        for index, raw in enumerate(args.reference, start=1):
            image_path = _local_file(raw[0], option=f"--reference {index}", extensions=IMAGE_EXTENSIONS)
            label = _clean_text(raw[1], option=f"reference {index} label", maximum=100)
            description = _clean_text(raw[2], option=f"reference {index} description", maximum=500)
            refs.append(Reference(image_path, label, description))
        references = tuple(refs)
        effective_prompt = _ingredients_prompt(prompt, references)
    elif args.mode == "union":
        if args.source_video is not None:
            if args.control_type != "canny":
                _fail(
                    "built-in source preprocessing currently supports Canny only; "
                    "for depth or pose select --guide-video with an already annotated clip"
                )
            source = _local_file(args.source_video, option="--source-video", extensions=VIDEO_EXTENSIONS)
        else:
            source = _local_file(args.guide_video, option="--guide-video", extensions=VIDEO_EXTENSIONS)
    elif args.mode == "motion":
        if args.start_image is None:
            _fail("motion requires --start-image")
        tracks = _load_tracks(args.tracks_json)
    else:  # pragma: no cover - argparse owns this invariant
        _fail(f"unsupported mode: {args.mode}")
    try:
        validate_render_prompt(effective_prompt, label="the prompt plus Ingredients reference context")
    except ValueError as exc:
        _fail(str(exc))

    image_conditionings = _plan_image_conditionings(args, profile)
    if image_conditionings:
        start_image = next((item.path for item in image_conditionings if item.frame == 0), None)
    image_args = [
        value
        for item in image_conditionings
        for value in ("--image", str(item.path), str(item.frame), f"{item.strength:g}")
    ]

    command = (
        str(CLI),
        "ic-lora",
        "--model",
        str(HQ_MODEL),
        "--gemma",
        str(GEMMA_MODEL),
        "--prompt",
        effective_prompt,
        "--output",
        str(temporary),
        "--lora",
        str(adapter),
        f"{args.lora_strength:g}",
        "--video-conditioning",
        str(guide),
        f"{args.conditioning_strength:g}",
        "--height",
        str(profile.height),
        "--width",
        str(profile.width),
        "--frames",
        str(profile.frames),
        "--frame-rate",
        f"{FRAME_RATE:g}",
        "--seed",
        str(args.seed),
        "--stage1-steps",
        str(profile.steps),
        "--single-stage",
        "--low-ram",
        *image_args,
        *(["--quiet"] if args.quiet else []),
    )
    return RenderPlan(
        mode=args.mode,
        profile=profile,
        prompt=effective_prompt,
        output=output,
        temporary_output=temporary,
        adapter=adapter,
        guide_placeholder=guide,
        command=command,
        references=references,
        source=source,
        start_image=start_image,
        tracks=tracks,
        image_conditionings=image_conditionings,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_safetensors(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("missing 8-byte header length")
            header_size = int.from_bytes(prefix, "little")
            if not 2 <= header_size <= 64 << 20:
                raise ValueError(f"unsafe header size {header_size}")
            encoded_header = handle.read(header_size)
            if len(encoded_header) != header_size:
                raise ValueError("truncated safetensors header")
            header = json.loads(encoded_header)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PreflightError(f"{label} is not a valid safetensors file: {path} ({exc})") from exc
    if not isinstance(header, dict):
        raise PreflightError(f"{label} safetensors header is not an object: {path}")
    tensor_keys = [key for key in header if key != "__metadata__"]
    if not tensor_keys:
        raise PreflightError(f"{label} contains no tensors: {path}")
    return header


def _normalize_lora_key(raw_key: str) -> str:
    """Apply exactly the key-only remaps used by the pinned MLX runtime."""
    key = raw_key
    for content, replacement in LORA_KEY_REPLACEMENTS:
        key = key.replace(content, replacement)
    return key


def _lora_tensor_shape(entry: Any, *, key: str, label: str, path: Path) -> tuple[int, int]:
    if not isinstance(entry, dict):
        raise PreflightError(f"{label} has an invalid safetensors entry for {key!r}: {path}")
    shape = entry.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
    ):
        raise PreflightError(f"{label} LoRA tensor {key!r} must have a positive rank-2 shape: {path}")
    return shape[0], shape[1]


def _validate_lora_header_compatibility(
    header: dict[str, Any], *, label: str, path: Path
) -> LoraKeyCoverage:
    """Match header keys exactly as ``BlockLoraSource`` does, without MLX.

    A usable adapter needs at least one complete A/B pair under a numbered
    ``transformer_blocks`` path.  Every recognized half-pair must have its
    mate, and the matrix ranks must agree, otherwise the runtime would either
    silently skip it or fail only after loading the model.
    """
    slots: dict[tuple[int, str], dict[str, tuple[str, Any]]] = {}
    recognized_tensors = 0
    for raw_key, entry in header.items():
        if raw_key == "__metadata__" or not isinstance(raw_key, str):
            continue
        model_key = _normalize_lora_key(raw_key)
        if not model_key.startswith(LORA_BLOCK_PREFIX):
            continue
        rest = model_key[len(LORA_BLOCK_PREFIX) :]
        index_text, separator, parameter_path = rest.partition(".")
        if not separator or not parameter_path:
            continue
        try:
            block_index = int(index_text)
        except ValueError:
            continue
        if block_index < 0:
            continue
        if parameter_path.endswith(LORA_A_SUFFIX):
            parameter = parameter_path[: -len(LORA_A_SUFFIX)]
            slot = "a"
        elif parameter_path.endswith(LORA_B_SUFFIX):
            parameter = parameter_path[: -len(LORA_B_SUFFIX)]
            slot = "b"
        else:
            continue
        if not parameter:
            continue
        recognized_tensors += 1
        slots.setdefault((block_index, parameter), {})[slot] = (raw_key, entry)

    if not slots:
        raise PreflightError(
            f"{label} has no LoRA tensors compatible with the MLX streaming key map: {path}. "
            f"Expected paired {LORA_BLOCK_PREFIX}<index>.<parameter>{LORA_A_SUFFIX} / lora_B keys "
            "(the diffusion_model. ComfyUI prefix is supported)."
        )

    incomplete = [pair for pair, pair_slots in slots.items() if set(pair_slots) != {"a", "b"}]
    if incomplete:
        example_block, example_parameter = incomplete[0]
        present = "/".join(sorted(slots[incomplete[0]]))
        raise PreflightError(
            f"{label} has {len(incomplete)} unpaired transformer-block LoRA target(s): {path}. "
            f"Example block {example_block} target {example_parameter!r} contains only {present.upper()}."
        )

    blocks: set[int] = set()
    for (block_index, parameter), pair_slots in slots.items():
        a_key, a_entry = pair_slots["a"]
        b_key, b_entry = pair_slots["b"]
        a_rank, _ = _lora_tensor_shape(a_entry, key=a_key, label=label, path=path)
        _, b_rank = _lora_tensor_shape(b_entry, key=b_key, label=label, path=path)
        if a_rank != b_rank:
            raise PreflightError(
                f"{label} has incompatible A/B ranks for block {block_index} target {parameter!r}: "
                f"{a_rank} != {b_rank} ({path})"
            )
        blocks.add(block_index)

    return LoraKeyCoverage(
        blocks=tuple(sorted(blocks)),
        pairs=len(slots),
        recognized_tensors=recognized_tensors,
    )


def validate_lora_key_compatibility(path: Path, *, label: str) -> LoraKeyCoverage:
    """Validate an adapter's safetensors header without importing or loading MLX."""
    header = _validate_safetensors(path, label=label)
    return _validate_lora_header_compatibility(header, label=label, path=path)


def missing_mode_assets(mode: str) -> tuple[str, ...]:
    """Return cheap, exact-size readiness failures for one IC-LoRA mode."""
    if mode not in ASSETS:
        return (f"unsupported control mode: {mode}",)
    missing: list[str] = []
    receipt_ready = any(
        path.is_file() and not path.is_symlink()
        for path in (CONTROL_RECEIPT, HQ_RECEIPT)
    )
    if not receipt_ready:
        missing.append(f"{CONTROL_RECEIPT} (or legacy full receipt {HQ_RECEIPT})")
    for path, size in (
        (DISTILLED_TRANSFORMER, DISTILLED_SIZE),
        (CONTROL_ROOT / ASSETS[mode].filename, ASSETS[mode].size),
    ):
        try:
            ready = path.is_file() and path.stat().st_size == size
        except OSError:
            ready = False
        if not ready:
            missing.append(str(path))
    return tuple(missing)


def verify_control_assets(plan: RenderPlan, *, full_hash: bool, report: Callable[[str], None] = print) -> None:
    expected = ASSETS[plan.mode]
    checks = [
        (DISTILLED_TRANSFORMER, DISTILLED_SIZE, DISTILLED_SHA256, "q8 distilled transformer"),
        (plan.adapter, expected.size, expected.sha256, f"{plan.mode} IC-LoRA"),
    ]
    for path, size, digest, label in checks:
        if not path.is_file():
            if path == DISTILLED_TRANSFORMER:
                raise PreflightError(
                    f"Missing {label}: {path}\nInstall or verify the pinned selective Q8 advanced overlay."
                )
            raise PreflightError(
                f"Missing {label}: {path}\nSource: {expected.repo} at revision {expected.revision}."
            )
        actual_size = path.stat().st_size
        if actual_size != size:
            raise PreflightError(f"Wrong size for {label}: {actual_size:,} != {size:,} bytes ({path})")
        if path == plan.adapter:
            coverage = validate_lora_key_compatibility(path, label=label)
            report(
                f"OK {label} key map ({coverage.pairs} A/B pairs across "
                f"{len(coverage.blocks)} transformer blocks)"
            )
        else:
            _validate_safetensors(path, label=label)
        report(f"OK {label} ({actual_size / 2**30:.3f} GiB)")
        if full_hash and digest:
            actual_digest = _sha256(path)
            if actual_digest != digest:
                raise PreflightError(f"SHA-256 mismatch for {label}: {actual_digest} != {digest}")
            report(f"OK SHA-256 {label}")


def _run_tool(command: Sequence[str], *, description: str) -> None:
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise PreflightError(f"Could not {description}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip()
        raise PreflightError(f"Could not {description}: {detail[-2000:] or f'exit {result.returncode}'}")


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (14, 15, 18))
    fitted = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (size[0] - 8, size[1] - 8))
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    panel.paste(fitted, (left, top))
    return panel


def create_reference_sheet(references: Sequence[Reference], destination: Path, *, width: int, height: int) -> None:
    count = len(references)
    columns, rows = ((1, 1) if count == 1 else (2, 1) if count == 2 else (2, 2) if count <= 4 else (3, 2) if count <= 6 else (4, 2))
    sheet = Image.new("RGB", (width, height), (8, 9, 12))
    x_edges = [round(i * width / columns) for i in range(columns + 1)]
    y_edges = [round(i * height / rows) for i in range(rows + 1)]
    for index, reference in enumerate(references):
        row, column = divmod(index, columns)
        cell = (x_edges[column + 1] - x_edges[column], y_edges[row + 1] - y_edges[row])
        try:
            with Image.open(reference.path) as image:
                panel = _fit_panel(image, cell)
        except (OSError, ValueError) as exc:
            raise UserInputError(f"Could not decode reference image {reference.path}: {exc}") from exc
        sheet.paste(panel, (x_edges[column], y_edges[row]))
    draw = ImageDraw.Draw(sheet)
    for x in x_edges[1:-1]:
        draw.line((x, 0, x, height), fill=(48, 50, 58), width=2)
    for y in y_edges[1:-1]:
        draw.line((0, y, width, y), fill=(48, 50, 58), width=2)
    sheet.save(destination, format="PNG", optimize=True)


def _encode_still_video(image: Path, destination: Path, *, frames: int, width: int, height: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreflightError("ffmpeg is required to create an Ingredients guide video")
    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "24",
            "-i",
            str(image),
            "-vf",
            f"scale={width}:{height}:flags=lanczos",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(destination),
        ],
        description="encode the static Ingredients reference video",
    )


def _normalize_control_video(
    source: Path,
    destination: Path,
    *,
    profile: Profile,
    canny: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreflightError("ffmpeg is required to prepare the Union Control guide")
    filters = [
        f"fps={FRAME_RATE:g}",
        f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={profile.width}:{profile.height}",
    ]
    if canny:
        filters.append("edgedetect=low=0.08:high=0.25")
        filters.append("format=gray")
    filters.extend(("tpad=stop_mode=clone:stop_duration=10", f"trim=end_frame={profile.frames}"))
    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(destination),
        ],
        description="normalize the Union Control guide video",
    )


def _catmull_rom(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], t: float) -> tuple[float, float]:
    t2, t3 = t * t, t * t * t
    values = []
    for axis in range(2):
        values.append(
            0.5
            * (
                2 * p1[axis]
                + (-p0[axis] + p2[axis]) * t
                + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t2
                + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * t3
            )
        )
    return values[0], values[1]


def interpolate_track(points: Sequence[tuple[float, float]], samples: int) -> tuple[tuple[float, float], ...]:
    if len(points) == 2:
        a, b = points
        return tuple(
            (a[0] + (b[0] - a[0]) * i / (samples - 1), a[1] + (b[1] - a[1]) * i / (samples - 1))
            for i in range(samples)
        )
    padded = (points[0], *points, points[-1])
    segments = len(padded) - 3
    result = []
    for index in range(samples):
        global_t = index / (samples - 1) * segments
        segment = min(int(global_t), segments - 1)
        result.append(_catmull_rom(padded[segment], padded[segment + 1], padded[segment + 2], padded[segment + 3], global_t - segment))
    return tuple(result)


def _age_colour(ratio: float) -> tuple[int, int, int]:
    # Official guide renderer computes RGB blue->green->yellow->red, then swaps
    # RGB to BGR to match the IC-LoRA's training data.
    if ratio <= 1 / 3:
        local = ratio * 3
        rgb = (0.0, local, 1.0 - local)
    elif ratio <= 2 / 3:
        local = (ratio - 1 / 3) * 3
        rgb = (local, 1.0, 0.0)
    else:
        local = (ratio - 2 / 3) * 3
        rgb = (1.0, 1.0 - local, 0.0)
    return tuple(round(channel * 255) for channel in reversed(rgb))


def create_motion_guide(
    tracks: Sequence[Sequence[tuple[float, float]]],
    destination: Path,
    *,
    profile: Profile,
    frame_directory: Path,
) -> None:
    short_side = 1080
    if profile.height <= profile.width:
        render_height = short_side
        render_width = round(profile.width * short_side / profile.height)
    else:
        render_width = short_side
        render_height = round(profile.height * short_side / profile.width)
    sampled = [interpolate_track(track, profile.frames) for track in tracks]
    frame_directory.mkdir(parents=True, exist_ok=True)
    for frame_index in range(profile.frames):
        canvas = Image.new("RGB", (render_width, render_height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        first = max(0, frame_index - 50)
        for track in sampled:
            for point_index in range(first, frame_index + 1):
                age = frame_index - point_index
                ratio = 1.0 - age / 50.0
                radius = 2.0 + 6.0 * ratio
                x = min(1.0, max(0.0, track[point_index][0])) * render_width
                y = min(1.0, max(0.0, track[point_index][1])) * render_height
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_age_colour(ratio))
        frame = canvas.resize((profile.width, profile.height), resample=Image.Resampling.BILINEAR)
        frame.save(frame_directory / f"{frame_index:05d}.png", format="PNG", optimize=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreflightError("ffmpeg is required to encode the Motion Track guide")
    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "24",
            "-i",
            str(frame_directory / "%05d.png"),
            "-frames:v",
            str(profile.frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(destination),
        ],
        description="encode the Motion Track guide video",
    )


def _offline_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "LTX2_GEMMA_MAX_LENGTH": "512",
        }
    )
    return env


def _replace_guide(command: Sequence[str], placeholder: Path, actual: Path) -> list[str]:
    return [str(actual) if item == str(placeholder) else item for item in command]


def _prepare_guide(plan: RenderPlan, args: argparse.Namespace, temp: Path) -> tuple[Path, Path | None]:
    guide = temp / "control-guide.mp4"
    preview: Path | None = None
    if plan.mode == "ingredients":
        preview = temp / "reference-sheet.png"
        create_reference_sheet(plan.references, preview, width=plan.profile.width, height=plan.profile.height)
        # The official 2.5 adapter requires the static reference video to be at
        # least 121 frames, even when the requested draft itself is shorter.
        _encode_still_video(preview, guide, frames=max(121, plan.profile.frames), width=plan.profile.width, height=plan.profile.height)
    elif plan.mode == "union":
        assert plan.source is not None
        canny = args.source_video is not None and args.control_type == "canny"
        _normalize_control_video(plan.source, guide, profile=plan.profile, canny=canny)
    elif plan.mode == "motion":
        create_motion_guide(plan.tracks, guide, profile=plan.profile, frame_directory=temp / "track-frames")
    else:  # pragma: no cover
        raise AssertionError(plan.mode)
    if not guide.is_file() or guide.stat().st_size == 0:
        raise PreflightError("guide preprocessing returned no usable video")
    return guide, preview


def _keep_guides(plan: RenderPlan, guide: Path, preview: Path | None) -> None:
    guide_target = plan.output.with_name(f"{plan.output.stem}-control-guide.mp4")
    shutil.copy2(guide, guide_target)
    if preview:
        shutil.copy2(preview, plan.output.with_name(f"{plan.output.stem}-reference-sheet.png"))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        plan = build_plan(args)
    except UserInputError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"LTX-2.5 q8 {plan.mode}: {plan.profile.width}x{plan.profile.height}, "
        f"{plan.profile.frames} frames @ 24 fps ({plan.profile.frames / FRAME_RATE:.2f}s), "
        f"seed={args.seed}"
    )
    print(f"Profile intent: {plan.profile.description}")
    if args.mode == "ingredients" and args.aspect_ratio != "profile":
        print(
            "EXPERIMENTAL INGREDIENTS ASPECT WARNING: the selected aspect is outside the "
            "recommended profile-default landscape canvas. Character identity, reference "
            "placement, and composition may be less consistent; use Profile default when "
            "reliability matters.",
            file=sys.stderr,
        )
    if args.profile == "trained5s":
        print("MEMORY WARNING: the 768x448x121 training bucket can exceed a 16 GB Mac's safe headroom.", file=sys.stderr)
    if args.profile == "motion3s-experimental":
        print(
            "EXPERIMENTAL MOTION RESOLUTION WARNING: 640x384 has substantially higher attention, "
            "swap, runtime, and Python/Metal termination risk than the preferred 512x320 profile. "
            "Keep this canary to 3 seconds and one simple track.",
            file=sys.stderr,
        )
    if getattr(args, "duration_seconds", None) is not None and args.duration_seconds > 5:
        print(
            "LONG-CONTROL WARNING: more than 5 seconds sharply increases reference-attention memory and "
            "continuity drift on 16 GB. Prefer a 3-second control render followed by Extend.",
            file=sys.stderr,
        )
    print("Command template: " + format_command_for_log(plan.command))
    if args.dry_run:
        print("DRY RUN OK; no files were written and no model/GPU checks were run.")
        return 0

    try:
        verify_install(full_hash=args.full_hash, require_gpu=True)
        verify_installed_control_overlay(full_hash=args.full_hash)
        verify_control_assets(plan, full_hash=args.full_hash)
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise PreflightError("ffmpeg and ffprobe must both be on PATH")
    except (VerificationError, OverlayError, PreflightError, OSError, json.JSONDecodeError) as exc:
        print(f"PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("PRE-FLIGHT VERIFIED; render not started (--preflight-only).")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".ltx25-{plan.mode}-", dir=OUTPUT_ROOT) as raw_temp:
            temp = Path(raw_temp)
            guide, preview = _prepare_guide(plan, args, temp)
            command = _replace_guide(plan.command, plan.guide_placeholder, guide)
            print("Render command: " + format_command_for_log(command))
            try:
                completed = subprocess.run(command, cwd=RUNTIME, env=_offline_environment(), check=False)
            except OSError as exc:
                raise PreflightError(f"Could not start the pinned MLX runtime: {exc}") from exc
            if completed.returncode:
                raise PreflightError(f"MLX render exited with status {completed.returncode}")
            if not plan.temporary_output.is_file() or plan.temporary_output.stat().st_size == 0:
                raise PreflightError("MLX returned success but created no non-empty temporary MP4")
            if args.keep_guide:
                _keep_guides(plan, guide, preview)
            os.replace(plan.temporary_output, plan.output)
    except (PreflightError, UserInputError, OSError) as exc:
        if plan.temporary_output.exists():
            plan.temporary_output.unlink()
        print(f"RENDER FAILED: {exc}. No successful output was claimed.", file=sys.stderr)
        return 3
    print(f"VIDEO READY: {plan.output} ({plan.output.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
