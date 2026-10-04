#!/usr/bin/env python3
"""Central filesystem layout for the localhost Dang Studio application.

By default, executable code lives in ``outputs/local-video`` and large mutable
assets live in the repository's ignored ``work/local-video`` directory.  Set
``DANG_STUDIO_HOME`` to an absolute path when a different local data directory
is preferable.  Keeping this decision here gives every runtime helper the same
source and data roots without hard-coded paths from a developer machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class PathConfigurationError(ValueError):
    """A configured filesystem location is unsafe or ambiguous."""


@dataclass(frozen=True)
class StudioPaths:
    app_root: Path
    workspace: Path
    data_root: Path
    models_root: Path
    runtimes_root: Path
    ltx_runtime: Path
    frame_composer_runtime: Path
    character_voice_runtime: Path
    ai_upscaler_runtime: Path
    ollama_runtime: Path
    ollama_models_root: Path
    cache_root: Path
    outputs_root: Path
    generated_root: Path
    edit_root: Path
    frames_root: Path
    voice_outputs_root: Path
    characters_root: Path
    voices_root: Path
    custom_data_root: bool


def _configured_path(
    environ: Mapping[str, str], name: str, default: Path
) -> tuple[Path, bool]:
    raw = environ.get(name, "").strip()
    if not raw:
        return default.resolve(strict=False), False
    value = Path(raw).expanduser()
    if not value.is_absolute():
        raise PathConfigurationError(f"{name} must be an absolute path")
    return value.resolve(strict=False), True


def resolve_studio_paths(
    environ: Mapping[str, str] | None = None,
    *,
    module_dir: Path | None = None,
) -> StudioPaths:
    """Resolve the source checkout and its default or configured data layout.

    ``module_dir`` exists for focused tests; production callers use the folder
    containing this module.  No directory is created during resolution.
    """

    env = os.environ if environ is None else environ
    source_root = (module_dir or Path(__file__).resolve().parent).resolve(strict=False)
    if len(source_root.parents) < 2:
        raise PathConfigurationError("Studio source root is too shallow to infer its workspace")
    legacy_workspace = source_root.parents[1]
    legacy_data_root = legacy_workspace / "work" / "local-video"

    app_root = source_root
    data_root, custom_data_root = _configured_path(env, "DANG_STUDIO_HOME", legacy_data_root)

    if custom_data_root:
        models_root = data_root / "models"
        runtimes_root = data_root / "runtimes"
        outputs_root = data_root / "outputs"
        cache_root = data_root / "cache"
        characters_root = data_root / "characters"
        voices_root = data_root / "voices"
    else:
        # Exact compatibility with the existing source-checkout installation.
        models_root = data_root / "models"
        runtimes_root = data_root
        outputs_root = app_root
        cache_root = data_root / "cache"
        characters_root = data_root / "characters"
        voices_root = data_root / "character-voices"

    generated_root = outputs_root / "generated"
    return StudioPaths(
        app_root=app_root,
        workspace=legacy_workspace,
        data_root=data_root,
        models_root=models_root,
        runtimes_root=runtimes_root,
        ltx_runtime=runtimes_root / "ltx-2-mlx",
        frame_composer_runtime=runtimes_root / "frame-composer",
        character_voice_runtime=runtimes_root / "character-voice-runtime",
        ai_upscaler_runtime=runtimes_root / "ai-video-upscaler",
        ollama_runtime=runtimes_root / "runtime" / "ollama-v0.34.4",
        ollama_models_root=(
            models_root / "ollama"
            if custom_data_root
            else Path.home() / ".ollama" / "models"
        ),
        cache_root=cache_root,
        outputs_root=outputs_root,
        generated_root=generated_root,
        edit_root=outputs_root / "renders",
        frames_root=outputs_root / "generated-frames",
        voice_outputs_root=generated_root / "voices",
        characters_root=characters_root,
        voices_root=voices_root,
        custom_data_root=custom_data_root,
    )


PATHS = resolve_studio_paths()
