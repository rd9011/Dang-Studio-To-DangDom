#!/usr/bin/env python3
"""Compose an LTX-sized still with FLUX.2 Klein and up to four tagged photos."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence

import install_frame_composer as installation
from generate_ltx25 import ASPECT_RATIO_PRESETS, resolve_aspect_dimensions
from studio_paths import PATHS


HERE = PATHS.app_root
OUTPUT_ROOT = PATHS.frames_root
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_BYTES = 256 * 1024
TAG_RE = re.compile(r"^@[a-z][a-z0-9_]{0,31}$")
TAG_SCAN_RE = re.compile(r"(?<![A-Za-z0-9_])@[a-z][a-z0-9_]{0,31}(?![A-Za-z0-9_])")
PROFILE_DIMENSIONS: dict[str, tuple[int, int]] = {
    "draft": (512, 256),
    "balanced": (640, 384),
    "quality": (768, 448),
    "max": (768, 448),
    "clip3s": (512, 256),
    "ingredients5s": (384, 224),
    "trained5s": (768, 448),
    "control3s": (384, 256),
    "control3s-plus": (512, 320),
    "motion3s-experimental": (640, 384),
}
TARGET_MODES = ("generate", "ingredients", "union", "motion")
TARGET_PROFILES = {
    "generate": {"draft", "balanced", "quality", "max", "clip3s"},
    "ingredients": {"ingredients5s", "trained5s"},
    "union": {"control3s", "control3s-plus"},
    "motion": {"control3s", "control3s-plus", "motion3s-experimental"},
}
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class ComposerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reference:
    tag: str
    path: Path


def _fail(message: str) -> NoReturn:
    raise ComposerError(message)


def normalize_tag(value: str) -> str:
    tag = value.strip().lower()
    if not tag.startswith("@"):
        tag = "@" + tag
    if not TAG_RE.fullmatch(tag):
        _fail("reference tags must look like @maya and use lowercase letters, numbers, or underscores")
    return tag


def validate_references(values: Sequence[Sequence[str]]) -> list[Reference]:
    if not 1 <= len(values) <= 4:
        _fail("Frame Composer requires 1–4 reference images; every photo consumes one slot")
    references: list[Reference] = []
    for raw in values:
        if len(raw) != 2:
            _fail("each --reference needs TAG and IMAGE")
        tag = normalize_tag(raw[0])
        source = Path(raw[1]).expanduser()
        try:
            if source.is_symlink():
                _fail(f"reference {tag} must not be a symbolic link")
            path = source.resolve(strict=True)
            stat = path.stat()
        except OSError as exc:
            raise ComposerError(f"cannot read reference {tag}: {exc}") from exc
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            _fail(f"reference {tag} must be a PNG, JPEG, or WebP file")
        if not 1 <= stat.st_size <= MAX_REFERENCE_BYTES:
            _fail(f"reference {tag} must be between 1 byte and 8 MB")
        references.append(Reference(tag, path))
    return references


def reference_groups(references: Sequence[Reference]) -> dict[str, list[tuple[int, Reference]]]:
    """Return stable public-tag groups with one-based positions in input order."""

    groups: dict[str, list[tuple[int, Reference]]] = {}
    for position, reference in enumerate(references, start=1):
        groups.setdefault(reference.tag, []).append((position, reference))
    return groups


def compile_tagged_prompt(prompt: str, references: Sequence[Reference]) -> str:
    value = prompt
    if not value.strip():
        _fail("prompt cannot be empty")
    if "\x00" in value:
        _fail("prompt contains an unsupported NUL character")
    if len(value) > MAX_PROMPT_CHARS or len(value.encode("utf-8")) > MAX_PROMPT_BYTES:
        _fail(
            "prompt exceeds the local process safety ceiling "
            f"({MAX_PROMPT_CHARS:,} characters / {MAX_PROMPT_BYTES // 1024} KiB UTF-8); "
            "it was rejected without truncation"
        )
    groups = reference_groups(references)
    mentioned = set(TAG_SCAN_RE.findall(value))
    unknown = sorted(mentioned - groups.keys())
    if unknown:
        _fail("prompt uses unregistered reference tag(s): " + ", ".join(unknown))
    missing = [tag for tag in groups if tag not in mentioned]
    if missing:
        _fail("mention every active ingredient tag in the prompt: " + ", ".join(missing))

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(tag) for tag in groups)
        + r")(?![A-Za-z0-9_])"
    )

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        members = groups[tag]
        label = tag[1:]
        if len(members) == 1:
            position = members[0][0]
            return f'image {position} (reference group "{label}")'
        positions = " and ".join(f"image {position}" for position, _reference in members)
        return f'reference group "{label}" (same ingredient shown in {positions})'

    compiled = pattern.sub(replace, value)
    mapping_lines: list[str] = []
    for tag, members in groups.items():
        label = tag[1:]
        total = len(members)
        for view_index, (position, _reference) in enumerate(members, start=1):
            alias = f"{label}__view_{view_index}"
            mapping_lines.append(
                f'- image {position}: internal alias "{alias}"; '
                f'reference group "{label}"; view {view_index} of {total}'
            )
    mapping = "\n".join(mapping_lines)
    return (
        "REFERENCE ORDER (fixed; do not exchange identities or attributes):\n"
        + mapping
        + "\n\nGROUPING RULES (strict): Images in the same reference group are "
        "complementary views of one ingredient, not separate copies. Use all views as "
        "identity evidence and produce only the number of subjects requested. Internal "
        "aliases identify views, never distinct subjects."
        + "\n\nCOMPOSITION REQUEST:\n"
        + compiled
    )


def safe_output_path(value: str) -> Path:
    if not value:
        _fail("--output is required")
    raw = Path(value).expanduser()
    if raw.suffix.lower() != ".png":
        _fail("Frame Composer output must use a .png extension")
    if OUTPUT_ROOT.is_symlink():
        _fail(f"refusing unsafe symbolic-link output directory: {OUTPUT_ROOT}")
    root = OUTPUT_ROOT.resolve(strict=False)
    path = raw.resolve(strict=False) if raw.is_absolute() else (root / raw).resolve(strict=False)
    if path == root or root not in path.parents:
        _fail(f"output must stay inside {root}")
    if path.parent != root:
        _fail("nested Frame Composer output directories are not supported")
    if path.is_symlink():
        _fail("output must not be a symbolic link")
    return path


def resolve_target_dimensions(
    profile: str,
    *,
    target_mode: str | None = None,
    aspect_ratio: str = "profile",
) -> tuple[int, int]:
    """Resolve the exact LTX canvas that this still is intended to anchor."""
    try:
        width, height = PROFILE_DIMENSIONS[profile]
    except KeyError:
        _fail("unknown LTX dimension profile")
    if target_mode is None:
        if profile in TARGET_PROFILES["ingredients"]:
            target_mode = "ingredients"
        elif profile in TARGET_PROFILES["union"]:
            target_mode = "union"
        else:
            target_mode = "generate"
    if target_mode not in TARGET_MODES:
        _fail("unknown LTX target mode")
    if profile not in TARGET_PROFILES[target_mode]:
        _fail(f"LTX profile {profile!r} is not valid for target mode {target_mode!r}")
    if aspect_ratio not in ASPECT_RATIO_PRESETS:
        _fail("unknown LTX aspect ratio")
    # Every IC-LoRA workflow uses the upstream single-stage topology and its
    # 32-pixel full-resolution VAE grid.  Keep ``profile`` byte-for-byte
    # compatible with the trained Ingredients buckets, but let an explicitly
    # selected experimental preset reuse the profile's pixel budget on that
    # safe grid.  This must mirror condition_ltx25 so a composed start/end
    # frame has exactly the dimensions of the video it will anchor.
    alignment = 32 if target_mode in {"ingredients", "union", "motion"} else 64
    try:
        return resolve_aspect_dimensions(width, height, aspect_ratio, alignment=alignment)
    except ValueError as exc:
        raise ComposerError(str(exc)) from exc


def build_command(
    prompt: str,
    references: Sequence[Reference],
    *,
    profile: str,
    seed: int,
    output: Path,
    target_mode: str | None = None,
    aspect_ratio: str = "profile",
) -> list[str]:
    width, height = resolve_target_dimensions(
        profile,
        target_mode=target_mode,
        aspect_ratio=aspect_ratio,
    )
    command = [
        str(installation.MLXGEN),
        "generate",
        "--model",
        str(installation.MODEL_DIR.resolve()),
    ]
    for reference in references:
        command.extend(["--image", str(reference.path)])
    command.extend(
        [
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--canvas-policy",
            "exact-resize",
            "--steps",
            "4",
            "--guidance",
            "1.0",
            "--seed",
            str(seed),
            "--low-ram",
            "--output",
            str(output),
        ]
    )
    return command


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        action="append",
        nargs=2,
        metavar=("TAG", "IMAGE"),
        required=True,
        help="tagged photo; repeat TAG for complementary views of the same ingredient (four photos total)",
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_DIMENSIONS), default="quality")
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default=None,
        help="LTX topology the still will anchor; controls the safe spatial grid",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=tuple(ASPECT_RATIO_PRESETS),
        default="profile",
        help="match the target video's resolved aspect-ratio canvas",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help=f"PNG destination inside {OUTPUT_ROOT}")
    parser.add_argument("--replace", action="store_true", help="replace an existing output atomically")
    parser.add_argument("prompt", help="composition prompt; mention each reference by its @tag")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0 <= args.seed <= 0xFFFFFFFF:
        _fail("seed must be from 0 to 4,294,967,295")
    status = installation.readiness()
    if not status["ready"]:
        _fail("Frame Composer is not ready: " + str(status["note"]))
    references = validate_references(args.reference)
    compiled = compile_tagged_prompt(args.prompt, references)
    output = safe_output_path(args.output)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.replace:
        _fail(f"output already exists (use --replace to replace it atomically): {output.name}")
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.partial.png")
    width, height = resolve_target_dimensions(
        args.profile,
        target_mode=args.target_mode,
        aspect_ratio=args.aspect_ratio,
    )
    command = build_command(
        compiled,
        references,
        profile=args.profile,
        seed=args.seed,
        output=temporary,
        target_mode=args.target_mode,
        aspect_ratio=args.aspect_ratio,
    )
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    group_count = len(reference_groups(references))
    print(
        f"FRAME COMPOSER PREFLIGHT VERIFIED: {len(references)} photo(s) in "
        f"{group_count} ingredient group(s), {width}x{height}, seed {args.seed}",
        flush=True,
    )
    try:
        subprocess.run(command, cwd=HERE, env=environment, check=True)
        dimensions = _png_dimensions(temporary)
        if dimensions != (width, height):
            _fail(f"generator output dimensions are {dimensions}, expected {(width, height)}")
        output_size = temporary.stat().st_size
        if not 32 <= output_size <= MAX_REFERENCE_BYTES:
            _fail("generator output must be a valid PNG no larger than 8 MB for LTX anchor use")
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComposerError(f"Frame Composer failed: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    print(f"FRAME READY: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComposerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
