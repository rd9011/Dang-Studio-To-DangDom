#!/usr/bin/env python3
"""Small, dependency-free localhost UI for the local LTX-2 MLX launcher."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import prompt_coach as prompt_coach_backend
from studio_paths import PATHS


# Direct Python launches do not always inherit Homebrew's shell PATH.
_homebrew_bin = Path("/opt/homebrew/bin")
if (_homebrew_bin / "ffmpeg").is_file():
    os.environ["PATH"] = os.pathsep.join(
        [str(_homebrew_bin)] + [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry != str(_homebrew_bin)]
    )

HERE = PATHS.app_root
WORKSPACE = PATHS.workspace
GENERATOR_SCRIPT = HERE / "generate_ltx25.py"
EDIT_SCRIPT = HERE / "edit_ltx25.py"
CONDITION_SCRIPT = HERE / "condition_ltx25.py"
FRAME_COMPOSER_SCRIPT = HERE / "compose_ltx_frame.py"
CHARACTER_VOICE_SCRIPT = HERE / "character_voice.py"
LTX_VENV_PYTHON = PATHS.ltx_runtime / ".venv" / "bin" / "python"
GENERATOR_PYTHON = LTX_VENV_PYTHON if LTX_VENV_PYTHON.is_file() else Path(sys.executable)
FRAME_COMPOSER_PYTHON = PATHS.frame_composer_runtime / ".venv" / "bin" / "python"
AI_UPSCALER_ROOT = PATHS.ai_upscaler_runtime
AI_UPSCALER_PYTHON = AI_UPSCALER_ROOT / ".venv" / "bin" / "python"
AI_UPSCALER_SCRIPT = HERE / "ai_video_upscale.py"
AI_UPSCALER_MODEL = AI_UPSCALER_ROOT / "weights" / "RealESRGAN_x4plus_522_fp16.mlpackage"
AI_UPSCALER_MODEL_WEIGHT = AI_UPSCALER_MODEL / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
GENERATED_DIR = PATHS.generated_root
EDIT_OUTPUT_DIR = PATHS.edit_root
FRAME_GENERATED_DIR = PATHS.frames_root
VOICE_GENERATED_DIR = PATHS.voice_outputs_root
CHARACTERS_DIR = PATHS.characters_root

HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PROFILES = {"draft", "balanced", "quality", "max", "clip3s"}
PROFILE_FRAMES = {"draft": 33, "balanced": 41, "quality": 49, "max": 25, "clip3s": 73}
ASPECT_RATIOS = {"profile", "16:9", "9:16", "1:1", "4:3", "3:4"}
NAMED_ASPECT_RATIO_VALUES = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
}
WORKFLOW_MODES = {"generate", "retake", "extend", "a2v", "ingredients", "union", "motion"}
# The browser never clips prompt text.  These deliberately high ceilings only
# protect the macOS argv boundary used by the local renderer; over-limit input
# is rejected explicitly rather than sliced.  50k-character prompts (including
# 4-byte Unicode) remain comfortably inside the supported envelope.
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_BYTES = 256 * 1024
MAX_FINAL_PROMPT_CHARS = 110_000
MAX_FINAL_PROMPT_BYTES = 288 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 24 * 1024 * 1024
MAX_ACTIVE_IMAGES = 4
MAX_ANCHOR_GROUPS = 4
MAX_IMAGE_PIXELS = 40_000_000
MAX_BODY_BYTES = 128 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_REFERENCE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_REQUEST_UPLOAD_BYTES = 96 * 1024 * 1024
MAX_LOG_CHARS = 240_000
MAX_MEDIA_NAME = 160
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
MEDIA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,155}\.mp4$")
FRAME_MEDIA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,155}\.png$")
VOICE_MEDIA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,155}\.wav$")
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CHARACTER_FIELDS = (
    ("character_name", "Character name", 80, 320),
    ("character_appearance", "Fixed appearance", 400, 1_600),
    ("character_wardrobe", "Fixed wardrobe", 300, 1_200),
    ("character_continuity", "Defining traits and continuity rules", 400, 1_600),
    ("character_avoid", "Avoid / do not introduce", 300, 1_200),
)
CHARACTER_PROFILE_SCHEMA = "ltx-studio-character-profile/1"
CHARACTER_PROFILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CHARACTER_REFERENCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_CHARACTER_PROFILES = 64
MAX_CHARACTER_REFERENCES = 4
MAX_CHARACTER_REFERENCE_BYTES = MAX_IMAGE_BYTES
MAX_TOTAL_CHARACTER_REFERENCE_BYTES = MAX_TOTAL_IMAGE_BYTES
CHARACTER_STORE_LOCK = threading.Lock()
FRAME_DESCRIPTION_MAX_CHARS = 500
FRAME_DESCRIPTION_MAX_BYTES = 2_000
ANCHOR_NAME_MAX_CHARS = 80
ANCHOR_NAME_MAX_BYTES = 320
ANCHOR_DESCRIPTION_MAX_CHARS = 500
ANCHOR_DESCRIPTION_MAX_BYTES = 2_000
SCENE_TEXT_MAX_CHARS = 1_000
SCENE_TEXT_MAX_BYTES = 4_000
GENERATED_KEYFRAME_CHOICES = {0}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aif", ".aiff", ".caf"}
MAX_INGREDIENT_REFERENCES = 8
MAX_TRACKS = 8
MAX_TRACK_POINTS = 16
INGREDIENT_TAG_RE = re.compile(r"^@[a-z][a-z0-9_]{0,31}$")
INGREDIENT_PROFILES = {"ingredients5s", "trained5s"}
CONTROL_PROFILES = {"control3s", "control3s-plus"}
MOTION_PROFILES = {*CONTROL_PROFILES, "motion3s-experimental"}
EDIT_MODES = {"retake", "extend", "a2v"}
CONDITION_MODES = {"ingredients", "union", "motion"}
GEMMA_PACK_CHOICES = {"auto", "official", "ddalcu-fallback"}
# Only primary renderer outputs may be restored as the current source after a
# restart.  The deliberately strict shape excludes fast/AI upscale artifacts,
# voiced derivatives, hidden partials, and arbitrary MP4s copied into the
# managed directory.
RESTORABLE_OUTPUT_RE = re.compile(
    r"^ltx25-(?P<label>draft|balanced|quality|max|clip3s|retake|extend|a2v|ingredients|union|motion)-"
    r"(?P<stamp>[0-9]{8}-[0-9]{6})-(?P<token>[0-9a-f]{8})\.mp4$"
)
COMPUTE_ADMISSION_LOCK = threading.Lock()
PROMPT_COACH_ADMISSION_LOCK = threading.Lock()
_PROMPT_COACH_ACTIVE = False
FRAME_COMPOSER_PROFILES = {
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
FRAME_COMPOSER_MODE_PROFILES = {
    "generate": PROFILES,
    "ingredients": INGREDIENT_PROFILES,
    "union": CONTROL_PROFILES,
    "motion": MOTION_PROFILES,
}
MAX_FRAME_COMPOSER_REFERENCES = 4
MAX_VOICE_CUES = 12
MAX_VOICE_REFERENCE_BYTES = 50 * 1024 * 1024
MAX_VOICE_TEXT_CHARS = 1_600
MAX_VOICE_DESCRIPTION_CHARS = 500
FAST_UPSCALE_SHORT_EDGE = 720
AI_UPSCALE_SHORT_EDGE = 1080
UPSCALE_PROFILES = {"fast720", "quality-ai"}
UPSCALE_DURATION_TOLERANCE_SECONDS = 0.20


def _prompt_coach_is_active() -> bool:
    with PROMPT_COACH_ADMISSION_LOCK:
        return _PROMPT_COACH_ACTIVE


def _claim_prompt_coach() -> bool:
    global _PROMPT_COACH_ACTIVE
    with PROMPT_COACH_ADMISSION_LOCK:
        if _PROMPT_COACH_ACTIVE:
            return False
        _PROMPT_COACH_ACTIVE = True
        return True


def _release_prompt_coach() -> None:
    global _PROMPT_COACH_ACTIVE
    with PROMPT_COACH_ADMISSION_LOCK:
        _PROMPT_COACH_ACTIVE = False


def _reject_if_prompt_coach_active() -> None:
    if _prompt_coach_is_active():
        raise APIError(
            HTTPStatus.CONFLICT,
            "Prompt Coach is finishing a local AI reply. Wait a moment before starting media compute.",
        )


def _metal_sandbox_block_reason() -> str | None:
    """Return an actionable message when macOS denies Metal user clients.

    MLX aborts the Python process when it cannot create an Apple GPU device, so
    this inexpensive sandbox query must run before any MLX module is imported.
    A normal Finder/Terminal launch returns zero (allowed). Codex's restricted
    seatbelt returns non-zero and should fail cleanly instead of producing a
    macOS "Python quit unexpectedly" dialog.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        checker = ctypes.CDLL("/usr/lib/libsandbox.dylib").sandbox_check
        checker.restype = ctypes.c_int
        denied = int(checker(os.getpid(), b"iokit-open-user-client", 0))
    except (AttributeError, OSError, TypeError, ValueError):
        # Do not block supported desktop launches merely because the private
        # diagnostic API is unavailable on a future macOS release.
        return None
    if denied == 0:
        return None
    return (
        "macOS blocked Apple GPU/Metal access for this studio process. Close it and reopen "
        "outputs/local-video/launch_ui.command from Finder or a normal Terminal. GPU jobs cannot "
        "run inside the Codex sandbox; CPU-only verification there must use --skip-gpu."
    )
MAX_VOICE_START_SECONDS = 86_400.0
VOICE_CHARACTER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dang Studio To DangDom — Local LTX‑2.5</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08090d;
      --panel: rgba(20, 22, 30, .82);
      --panel-2: #101219;
      --line: rgba(255, 255, 255, .1);
      --text: #f6f5f2;
      --muted: #a2a5b0;
      --accent: #c9ff5c;
      --accent-2: #78e7ff;
      --danger: #ff6f7d;
      --radius: 20px;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 5%, rgba(120, 231, 255, .13), transparent 28rem),
        radial-gradient(circle at 93% 12%, rgba(201, 255, 92, .11), transparent 25rem),
        var(--bg);
      font: 15px/1.5 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .035;
      background-image: linear-gradient(rgba(255,255,255,.6) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255,255,255,.6) 1px, transparent 1px);
      background-size: 34px 34px;
    }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 44px 0 64px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; margin-bottom: 28px; }
    .eyebrow { color: var(--accent); font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .18em; text-transform: uppercase; }
    h1 { margin: 7px 0 4px; font-size: clamp(32px, 6vw, 63px); line-height: .95; letter-spacing: -.055em; font-weight: 760; }
    .subtitle { color: var(--muted); margin: 10px 0 0; max-width: 620px; }
    .header-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 8px; }
    .help-button { padding: 9px 12px; color: var(--text); background: #171a23; border: 1px solid var(--line); }
    .local-badge { white-space: nowrap; color: var(--accent-2); border: 1px solid rgba(120,231,255,.25); padding: 8px 12px; border-radius: 999px; background: rgba(120,231,255,.07); font-size: 12px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.03fr) minmax(360px, .97fr); gap: 18px; align-items: start; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: 0 24px 80px rgba(0,0,0,.28); backdrop-filter: blur(18px); }
    .composer { padding: 24px; }
    .mode-tabs { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 18px; }
    .mode-tab { padding: 10px 11px; color: var(--muted); background: #0b0d12; border: 1px solid var(--line); }
    .mode-tab[aria-selected="true"] { color: #101407; background: var(--accent); border-color: var(--accent); }
    .mode-tab[aria-disabled="true"] { opacity: .58; border-style: dashed; }
    .badge { display: inline-block; margin-left: 5px; padding: 2px 5px; border-radius: 999px; color: #ffdca0; background: rgba(255,183,77,.14); font-size: 8px; letter-spacing: .06em; text-transform: uppercase; vertical-align: 1px; }
    .mode-tab[aria-selected="true"] .badge { color: #332407; background: rgba(20,20,10,.14); }
    .readiness { margin: -8px 0 17px; min-height: 18px; color: var(--muted); font-size: 11px; }
    .readiness.ready { color: var(--accent); }
    .readiness.needs { color: #ffd18a; }
    label, legend { display: block; margin-bottom: 8px; color: #d9dae0; font-size: 13px; font-weight: 650; }
    textarea, input[type="number"], input[type="text"], select {
      width: 100%; color: var(--text); background: #0b0d12; border: 1px solid var(--line); border-radius: 13px; outline: none;
      transition: border-color .16s, box-shadow .16s;
    }
    textarea { min-height: 142px; padding: 15px 16px; resize: vertical; font: inherit; }
    input[type="number"], input[type="text"], select { height: 44px; padding: 0 12px; font: inherit; }
    input[type="number"] { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    input[type="checkbox"] { accent-color: var(--accent); }
    textarea:focus, input:focus, select:focus { border-color: rgba(201,255,92,.65); box-shadow: 0 0 0 3px rgba(201,255,92,.09); }
    .counter { text-align: right; color: #747783; margin: 5px 2px 15px; font-size: 11px; }
    fieldset { padding: 0; border: 0; margin: 0 0 18px; }
    .profiles { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .profile { position: relative; }
    .profile input { position: absolute; opacity: 0; pointer-events: none; }
    .profile span { display: block; min-height: 74px; padding: 12px; border: 1px solid var(--line); border-radius: 13px; color: var(--muted); background: #0b0d12; cursor: pointer; transition: .16s ease; }
    .profile strong { display: block; color: var(--text); font-size: 13px; margin-bottom: 3px; }
    .profile small { font-size: 11px; }
    .profile input:checked + span { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); background: rgba(201,255,92,.06); }
    .row { display: grid; grid-template-columns: 1fr 1.4fr; gap: 12px; margin-bottom: 20px; }
    .mode-panel { margin-bottom: 18px; padding: 14px; border: 1px solid var(--line); border-radius: 14px; background: rgba(11,13,18,.72); }
    .mode-panel h2 { margin: 0 0 5px; font-size: 15px; }
    .mode-panel > p { margin: 0 0 12px; color: var(--muted); font-size: 11px; }
    .field-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 11px; }
    .field-grid .wide { grid-column: 1 / -1; }
    .field-grid label { margin-bottom: 5px; font-size: 11px; }
    .duration-box { padding: 11px 12px; border: 1px solid var(--line); border-radius: 12px; background: #0b0d12; }
    .duration-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
    .duration-value { color: var(--accent-2); font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .memory-warning { margin-top: 8px; padding: 9px 10px; color: #ffd9a8; background: rgba(255,164,72,.09); border: 1px solid rgba(255,164,72,.28); border-radius: 9px; font-size: 11px; }
    .locked-duration { padding: 10px 11px; color: var(--accent-2); border: 1px solid rgba(120,231,255,.25); border-radius: 10px; background: rgba(120,231,255,.06); font-size: 11px; }
    .check-row { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 12px; }
    .check-row input { width: auto; }
    .reference-list { display: grid; gap: 9px; margin: 10px 0; }
    .reference-card { padding: 11px; border: 1px solid var(--line); border-radius: 11px; background: #090b0f; }
    .reference-head { display:flex; justify-content:space-between; gap:8px; align-items:center; }
    .reference-head strong { font-size:12px; }
    .reference-card textarea { min-height: 64px; padding: 9px 10px; font-size: 12px; }
    .json-editor { min-height: 150px; font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .frame-composer-result { margin-top: 12px; padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: #080a0e; }
    .frame-composer-result img { display: block; width: 100%; max-height: 300px; object-fit: contain; border-radius: 9px; background: #050608; }
    .frame-composer-actions { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
    .frame-composer-status { min-height: 18px; margin: 8px 0; color: var(--muted); font-size: 11px; }
    .frame-composer-status.ready { color: var(--accent); }
    .frame-composer-status.needs { color: #ffd18a; }
    .seed-wrap { display: grid; grid-template-columns: 1fr 44px; gap: 7px; }
    button, .download {
      border: 0; border-radius: 12px; font: 700 13px/1 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; cursor: pointer; text-decoration: none; transition: transform .12s, opacity .12s, border-color .12s;
    }
    button:active { transform: translateY(1px); }
    button:disabled { cursor: not-allowed; opacity: .45; }
    .icon-button { color: var(--muted); background: #151821; border: 1px solid var(--line); font-size: 17px; }
    .upload { height: 44px; position: relative; display: flex; align-items: center; gap: 11px; padding: 0 12px; border: 1px dashed rgba(255,255,255,.2); border-radius: 12px; background: #0b0d12; cursor: pointer; overflow: hidden; }
    .upload:hover { border-color: rgba(120,231,255,.5); }
    .upload input { position: absolute; opacity: 0; inset: 0; cursor: pointer; width: 100%; }
    .thumb { width: 27px; height: 27px; object-fit: cover; border-radius: 6px; background: #1b1e28; display: none; }
    .upload-copy { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font-size: 12px; }
    .control-panel { margin: 0 0 20px; border: 1px solid var(--line); border-radius: 14px; background: rgba(11,13,18,.7); overflow: hidden; }
    .control-panel summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 14px; cursor: pointer; color: #e8e9ed; font-weight: 700; }
    .control-panel summary::marker { color: var(--accent-2); }
    .control-panel summary small { color: #777b87; font-size: 10px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
    .panel-body { padding: 2px 14px 14px; border-top: 1px solid var(--line); }
    .frame-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; }
    .frame-card, .anchor-card, .scene-card { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: #0b0d12; }
    .frame-card h3, .anchor-card h3, .scene-card h3 { margin: 0 0 10px; font-size: 12px; }
    .frame-card textarea, .anchor-card textarea, .scene-card textarea { min-height: 66px; padding: 9px 10px; font-size: 12px; }
    .compact-grid { display: grid; grid-template-columns: 1fr 92px; gap: 9px; align-items: end; margin-top: 9px; }
    .compact-grid label { margin-bottom: 5px; font-size: 11px; }
    .previews { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; min-height: 0; }
    .previews img { width: 48px; height: 48px; object-fit: cover; border-radius: 7px; border: 1px solid var(--line); }
    .anchor-tools { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin: 12px 0; }
    .slot-counter { color: var(--accent-2); font: 700 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .slot-counter.over { color: var(--danger); }
    .small-button { padding: 9px 11px; color: var(--text); background: #171a23; border: 1px solid var(--line); }
    .remove-anchor, .remove-scene { color: #ffd9dd; background: transparent; padding: 4px 7px; border: 1px solid rgba(255,111,125,.28); }
    .anchor-head, .scene-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .anchor-list, .scene-list { display: grid; gap: 10px; }
    .motion-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(170px, .55fr); gap: 11px; align-items: end; margin: 11px 0 13px; }
    .technical-note { margin: 11px 0; padding: 10px 11px; color: #aeb2bd; background: rgba(120,231,255,.055); border-left: 3px solid rgba(120,231,255,.55); border-radius: 5px 10px 10px 5px; font-size: 11px; }
    .technical-note strong { color: var(--accent-2); }
    .character-bible { margin: 0 0 20px; border: 1px solid var(--line); border-radius: 14px; background: rgba(11,13,18,.7); overflow: hidden; }
    .character-bible summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 14px; cursor: pointer; color: #e8e9ed; font-weight: 700; }
    .character-bible summary::marker { color: var(--accent-2); }
    .character-bible summary small { color: #777b87; font-size: 10px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
    .character-body { padding: 2px 14px 14px; border-top: 1px solid var(--line); }
    .identity-note { margin: 12px 0 14px; padding: 10px 11px; color: #aeb2bd; background: rgba(120,231,255,.055); border-left: 3px solid rgba(120,231,255,.55); border-radius: 5px 10px 10px 5px; font-size: 11px; }
    .identity-note strong { color: var(--accent-2); }
    .character-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; }
    .character-grid .wide { grid-column: 1 / -1; }
    .character-grid label { margin-bottom: 6px; font-size: 11px; }
    .character-grid textarea { min-height: 70px; padding: 10px 11px; font-size: 12px; }
    .character-profile-row { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 9px; align-items: end; margin: 12px 0 9px; }
    .character-profile-actions { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
    .character-profile-status { min-height: 18px; margin: 8px 0 10px; color: var(--muted); font-size: 11px; }
    .character-profile-status.ready { color: var(--accent); }
    .character-profile-status.needs { color: #ffd18a; }
    .voice-status { min-height: 18px; margin: 8px 0 12px; color: var(--muted); font-size: 11px; }
    .voice-status.ready { color: var(--accent); }
    .voice-status.needs { color: #ffd18a; }
    .voice-cue-list { display: grid; gap: 9px; margin: 10px 0 12px; }
    .voice-result-list { display: grid; gap: 7px; margin-top: 10px; }
    .voice-result { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; padding: 9px 10px; border: 1px solid var(--line); border-radius: 10px; background: #080a0e; color: var(--muted); font-size: 11px; }
    .voice-result audio { width: min(100%, 280px); height: 32px; }
    .voice-job-progress { margin: 8px 0; color: var(--muted); font-size: 11px; }
    .actions { display: flex; gap: 10px; }
    .generate { flex: 1; height: 49px; background: var(--accent); color: #101407; font-size: 14px; }
    .cancel { height: 49px; padding: 0 18px; color: #ffd9dd; background: rgba(255,111,125,.1); border: 1px solid rgba(255,111,125,.3); }
    .notice { min-height: 19px; margin: 11px 2px 0; color: var(--danger); font-size: 12px; }
    .output { overflow: hidden; }
    .statusbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 17px 18px; border-bottom: 1px solid var(--line); }
    .progress-wrap { padding: 11px 18px 13px; border-bottom: 1px solid var(--line); background: rgba(8,9,13,.52); }
    .progress-copy { display:flex; justify-content:space-between; gap:12px; margin-bottom:7px; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.08em; }
    progress { width: 100%; height: 7px; display:block; accent-color: var(--accent-2); }
    .status-label { display: flex; align-items: center; gap: 9px; font-weight: 650; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #666a75; box-shadow: 0 0 0 4px rgba(102,106,117,.1); }
    .dot.running, .dot.starting, .dot.cancelling { background: var(--accent-2); box-shadow: 0 0 0 4px rgba(120,231,255,.12); animation: pulse 1.4s infinite; }
    .dot.succeeded { background: var(--accent); box-shadow: 0 0 0 4px rgba(201,255,92,.12); }
    .dot.failed, .dot.cancelled { background: var(--danger); box-shadow: 0 0 0 4px rgba(255,111,125,.12); }
    @keyframes pulse { 50% { opacity: .4; } }
    .meta { color: var(--muted); font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .stage { min-height: 290px; background: #07080b; display: grid; place-items: center; position: relative; }
    .empty { padding: 32px; text-align: center; color: #6f727c; }
    .empty-mark { font-size: 48px; opacity: .45; margin-bottom: 8px; }
    video { display: none; width: 100%; max-height: 460px; background: #000; }
    .result-actions { display: none; position: absolute; right: 13px; bottom: 13px; }
    .download { padding: 9px 12px; color: #0c1006; background: var(--accent); }
    .capture-panel, .upscale-panel { padding: 14px 18px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: rgba(8,9,13,.52); }
    .upscale-panel { border-bottom: 0; }
    .upscale-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }
    .upscale-option { padding: 12px; border: 1px solid var(--line); border-radius: 11px; background: #080a0e; }
    .upscale-option.recommended { border-color: rgba(201,255,92,.42); background: rgba(201,255,92,.035); }
    .upscale-option-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 4px; }
    .upscale-badge { padding: 3px 6px; border-radius: 999px; color: #111507; background: var(--accent); font-size: 8px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .upscale-actions { display: flex; flex-wrap: wrap; gap: 7px; align-items: center; margin-top: 9px; }
    .upscale-status { margin-top: 8px; color: var(--muted); font-size: 11px; }
    .upscale-progress { margin-top: 10px; }
    .capture-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 9px; }
    .capture-copy { color: var(--muted); font-size: 11px; }
    .capture-result { display: grid; grid-template-columns: 112px minmax(0,1fr); gap: 12px; align-items: center; margin-top: 11px; padding: 10px; border: 1px solid var(--line); border-radius: 11px; background: #080a0e; }
    .capture-result img { display: block; width: 112px; height: 72px; object-fit: contain; border-radius: 7px; background: #050608; }
    .capture-result-actions { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 8px; }
    .log-head { padding: 15px 18px 8px; color: var(--muted); font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .13em; text-transform: uppercase; }
    pre { height: 180px; overflow: auto; margin: 0; padding: 5px 18px 18px; white-space: pre-wrap; word-break: break-word; color: #aeb4c1; font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .privacy { margin: 15px 3px 0; color: #6f727c; font-size: 11px; }
    .help-panel { margin: 0 0 24px; border: 1px solid var(--line); border-radius: 16px; background: rgba(11,13,18,.76); overflow: hidden; }
    .help-panel > summary { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 14px 16px; cursor: pointer; font-weight: 700; }
    .help-panel > summary::marker { color: var(--accent); }
    .help-panel > summary small { color: var(--muted); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
    .help-body { padding: 15px 16px 17px; border-top: 1px solid var(--line); }
    .help-lead { margin: 0 0 12px; color: #c9cbd2; font-size: 12px; }
    .help-advisor { margin-bottom: 12px; padding: 12px; border: 1px solid rgba(201,255,92,.25); border-radius: 12px; background: rgba(201,255,92,.045); }
    .help-advisor strong { display: block; margin-bottom: 4px; }
    .help-advisor p { margin: 0; color: var(--muted); font-size: 11px; }
    .help-advisor-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .setup-advice { margin-top: 10px; padding: 10px 11px; border-radius: 10px; background: #080a0e; color: var(--muted); font-size: 11px; }
    .setup-advice strong { color: var(--accent); }
    .setup-advice ul { margin: 7px 0 0; padding-left: 18px; }
    .setup-advice li + li { margin-top: 5px; }
    .help-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 9px; }
    .help-topic { border: 1px solid var(--line); border-radius: 12px; background: #090b0f; overflow: hidden; }
    .help-topic > summary { padding: 11px 12px; cursor: pointer; color: #e8e9ed; font-size: 12px; font-weight: 700; }
    .help-copy { padding: 0 12px 12px; color: var(--muted); font-size: 11px; }
    .help-copy p { margin: 4px 0 9px; }
    .help-copy ul { margin: 2px 0 0; padding-left: 17px; }
    .help-copy li + li { margin-top: 6px; }
    .prompt-template { margin: 8px 0 0; padding: 10px; white-space: pre-wrap; color: #cfd3dd; background: #06070a; border: 1px solid var(--line); border-radius: 9px; font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .prompt-coach-launcher { position: fixed; right: 18px; bottom: 18px; z-index: 40; display: flex; align-items: center; gap: 8px; padding: 13px 15px; color: #101407; background: var(--accent); box-shadow: 0 14px 42px rgba(0,0,0,.5); }
    .prompt-coach-launcher span { padding: 3px 6px; color: #dff8ff; background: #1b3b43; border-radius: 999px; font-size: 8px; letter-spacing: .08em; text-transform: uppercase; }
    .prompt-coach { position: fixed; right: 18px; bottom: calc(74px + env(safe-area-inset-bottom)); z-index: 41; display: flex; flex-direction: column; width: min(430px, calc(100vw - 24px)); max-height: min(760px, calc(100dvh - 94px - env(safe-area-inset-bottom))); overflow: hidden; border: 1px solid rgba(201,255,92,.27); border-radius: 18px; background: #11141c; box-shadow: 0 28px 90px rgba(0,0,0,.66); }
    .prompt-coach-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding: 14px 15px; border-bottom: 1px solid var(--line); background: #151923; }
    .prompt-coach-head strong { display: block; }
    .prompt-coach-head small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; }
    .coach-provider { display: inline-flex; align-items: center; gap: 5px; margin-top: 7px; padding: 3px 7px; border: 1px solid rgba(120,231,255,.24); border-radius: 999px; color: var(--accent-2); background: rgba(120,231,255,.07); font-size: 9px; }
    .coach-provider.guided { color: #ffd18a; border-color: rgba(255,209,138,.25); background: rgba(255,209,138,.07); }
    .prompt-coach-close { width: 44px; height: 44px; flex: 0 0 44px; padding: 0; color: var(--muted); background: #0b0d12; border: 1px solid var(--line); font-size: 20px; }
    .prompt-coach-scroll { min-height: 0; overflow: auto; padding: 13px 14px 15px; }
    .coach-message { margin: 0 0 11px; padding: 10px 11px; border-radius: 12px 12px 12px 3px; color: #d9dbe2; background: #1a1e28; font-size: 11px; }
    .coach-message strong { color: var(--accent-2); }
    .coach-conversation { display: grid; gap: 8px; max-height: 220px; overflow: auto; margin-bottom: 13px; padding: 2px; }
    .coach-chat { max-width: 92%; padding: 10px 11px; border-radius: 13px; white-space: pre-wrap; font-size: 11px; line-height: 1.48; }
    .coach-chat.assistant { justify-self: start; border-radius: 13px 13px 13px 3px; color: #d9dbe2; background: #1a1e28; }
    .coach-chat.user { justify-self: end; border-radius: 13px 13px 3px 13px; color: #101407; background: var(--accent); }
    .coach-chat-meta { display: block; margin-bottom: 4px; color: #9296a2; font-size: 8px; letter-spacing: .06em; text-transform: uppercase; }
    .coach-chat.user .coach-chat-meta { color: rgba(16,20,7,.65); }
    .coach-disclaimer { margin-top: 5px; color: #9296a2; font-size: 10px; }
    .prompt-coach textarea { min-height: 96px; padding: 11px 12px; font-size: 12px; }
    .coach-actions { display: flex; flex-wrap: wrap; gap: 7px; margin: 9px 0 11px; }
    .coach-action-help { flex-basis: 100%; margin: 1px 0 0; color: #9296a2; font-size: 9px; line-height: 1.45; }
    .coach-primary { padding: 10px 12px; color: #101407; background: var(--accent); }
    .coach-result { display: grid; gap: 9px; }
    .coach-audit { border: 1px solid var(--line); border-radius: 12px; background: #090b0f; overflow: hidden; }
    .coach-audit > summary { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 11px; cursor: pointer; color: #d9dbe2; font-size: 10px; font-weight: 700; }
    .coach-audit > summary::after { content: "Optional detail"; color: #9296a2; font-size: 8px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
    .coach-audit[open] > summary { border-bottom: 1px solid var(--line); }
    .coach-audit[open] > summary::after { content: "Expanded"; }
    .coach-audit-body { display: grid; gap: 9px; padding: 9px; }
    .coach-score { display: grid; grid-template-columns: 72px minmax(0,1fr); gap: 11px; align-items: center; padding: 11px; border: 1px solid var(--line); border-radius: 12px; background: #090b0f; }
    .coach-score-number { display: grid; place-items: center; width: 64px; height: 64px; border-radius: 50%; color: var(--accent); background: conic-gradient(var(--accent) var(--coach-score, 0%), #252a35 0); font: 800 18px ui-monospace, SFMono-Regular, Menlo, monospace; position: relative; }
    .coach-score-number::before { content: ""; position: absolute; inset: 5px; border-radius: 50%; background: #090b0f; }
    .coach-score-number span { position: relative; }
    .coach-score-copy strong { display: block; margin-bottom: 3px; }
    .coach-score-copy p { margin: 0; color: var(--muted); font-size: 10px; }
    .coach-block { padding: 10px 11px; border: 1px solid var(--line); border-radius: 11px; background: #090b0f; }
    .coach-block h3 { margin: 0 0 6px; color: #eef0f4; font-size: 11px; }
    .coach-block ul, .coach-block ol { margin: 0; padding-left: 18px; color: var(--muted); font-size: 10px; }
    .coach-block li + li { margin-top: 5px; }
    .coach-rewrite { min-height: 146px !important; color: #d6d9e1; font: 10px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace !important; }
    .coach-use { width: 100%; padding: 11px 12px; color: #101407; background: var(--accent-2); }
    .coach-followup { margin-top: 13px; padding-top: 12px; border-top: 1px solid var(--line); }
    .coach-followup textarea { min-height: 66px; }
    .coach-followup-row { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-top: 8px; }
    .coach-suggestion-heading { display: grid; gap: 2px; margin: 12px 0 0; color: #d9dbe2; font-size: 10px; }
    .coach-suggestion-heading span { color: #9296a2; font-size: 9px; }
    .coach-suggestions { display: flex; flex-wrap: wrap; gap: 6px; margin: 9px 0 0; }
    .coach-suggestion { padding: 6px 8px; color: var(--muted); background: #0b0d12; border: 1px solid var(--line); font-size: 9px; text-align: left; }
    .coach-provider-note { margin: 7px 0 0; color: #9296a2; font-size: 9px; }
    .coach-provider-action { margin: 7px 0 0; padding: 8px 9px; border: 1px solid rgba(255,209,138,.22); border-radius: 9px; color: #ffd18a; background: rgba(255,209,138,.06); font-size: 9px; }
    .tutorial-panel { position: fixed; left: 18px; bottom: 18px; z-index: 45; display: flex; flex-direction: column; width: min(460px, calc(100vw - 36px)); max-height: min(790px, calc(100dvh - 36px)); overflow: hidden; border: 1px solid rgba(120,231,255,.32); border-radius: 18px; background: #11141c; box-shadow: 0 28px 90px rgba(0,0,0,.7); }
    .tutorial-panel[hidden] { display: none; }
    .tutorial-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding: 14px 15px; border-bottom: 1px solid var(--line); background: #151923; }
    .tutorial-head strong { display: block; }
    .tutorial-head small { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; }
    .tutorial-close { width: 44px; height: 44px; flex: 0 0 44px; padding: 0; color: var(--muted); background: #0b0d12; border: 1px solid var(--line); font-size: 20px; }
    .tutorial-scroll { min-height: 0; overflow: auto; padding: 14px 15px 16px; }
    .tutorial-safety { margin: 0 0 12px; padding: 9px 10px; color: #dceeff; background: rgba(120,231,255,.07); border: 1px solid rgba(120,231,255,.22); border-radius: 10px; font-size: 10px; }
    .tutorial-safety strong { color: var(--accent-2); }
    .tutorial-intro { margin: 0 0 12px; color: var(--muted); font-size: 11px; line-height: 1.5; }
    .tutorial-lesson-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .tutorial-lesson { min-height: 112px; padding: 11px; color: var(--text); background: #0b0d12; border: 1px solid var(--line); text-align: left; }
    .tutorial-lesson:hover { border-color: rgba(120,231,255,.4); }
    .tutorial-lesson strong { display: block; margin: 5px 0; font-size: 12px; line-height: 1.25; }
    .tutorial-lesson small { display: block; color: var(--muted); font-size: 9px; line-height: 1.4; }
    .tutorial-lesson-status { display: inline-flex; align-items: center; padding: 3px 6px; border-radius: 999px; color: var(--accent); background: rgba(201,255,92,.08); border: 1px solid rgba(201,255,92,.2); font-size: 8px; letter-spacing: .05em; text-transform: uppercase; }
    .tutorial-lesson-status.mixed { color: #ffd18a; background: rgba(255,209,138,.07); border-color: rgba(255,209,138,.22); }
    .tutorial-lesson-status.blocked { color: #ffb5bd; background: rgba(255,111,125,.07); border-color: rgba(255,111,125,.22); }
    .tutorial-menu-actions { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-top: 12px; }
    .tutorial-completed { color: var(--accent); font-size: 10px; }
    .tutorial-progress-copy { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: .07em; }
    .tutorial-progress { width: 100%; height: 6px; margin: 7px 0 11px; accent-color: var(--accent-2); }
    .tutorial-status { margin-bottom: 11px; padding: 8px 9px; color: var(--muted); background: #090b0f; border: 1px solid var(--line); border-radius: 9px; font-size: 9px; line-height: 1.45; }
    .tutorial-step h3 { margin: 0 0 7px; font-size: 16px; }
    .tutorial-step-body { margin: 0; color: #cfd2da; font-size: 11px; line-height: 1.55; white-space: pre-wrap; }
    .tutorial-points { margin: 10px 0 0; padding-left: 18px; color: var(--muted); font-size: 10px; line-height: 1.48; }
    .tutorial-points li + li { margin-top: 5px; }
    .tutorial-tools { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
    .tutorial-primary { padding: 10px 12px; color: #101407; background: var(--accent-2); }
    .tutorial-practice { padding: 10px 12px; color: #101407; background: var(--accent); }
    .tutorial-practice-note { min-height: 17px; margin: 8px 0 0; color: #ffd18a; font-size: 9px; line-height: 1.4; }
    .tutorial-quiz { display: grid; gap: 7px; margin-top: 12px; }
    .tutorial-quiz button { padding: 9px 10px; color: var(--text); background: #0b0d12; border: 1px solid var(--line); text-align: left; font-size: 10px; line-height: 1.35; }
    .tutorial-quiz button.correct { color: var(--accent); border-color: rgba(201,255,92,.42); background: rgba(201,255,92,.06); }
    .tutorial-quiz button.incorrect { color: #ffb5bd; border-color: rgba(255,111,125,.4); background: rgba(255,111,125,.06); }
    .tutorial-quiz-feedback { min-height: 31px; margin-top: 8px; color: var(--muted); font-size: 10px; line-height: 1.45; }
    .tutorial-footer { position: sticky; bottom: -16px; display: flex; justify-content: space-between; gap: 8px; margin: 14px -15px -16px; padding: 11px 15px max(11px, env(safe-area-inset-bottom)); border-top: 1px solid var(--line); background: rgba(17,20,28,.98); }
    .tutorial-footer .small-button { min-height: 42px; }
    .tutorial-resume { position: fixed; left: max(18px, env(safe-area-inset-left)); bottom: max(18px, env(safe-area-inset-bottom)); z-index: 44; padding: 12px 14px; color: #101407; background: var(--accent-2); box-shadow: 0 12px 36px rgba(0,0,0,.55); }
    .tutorial-resume[hidden] { display: none; }
    .tutorial-highlight { position: relative; z-index: 3; outline: 3px solid var(--accent-2) !important; outline-offset: 5px; scroll-margin: 100px; animation: tutorial-pulse 1.15s ease-in-out 2; }
    @keyframes tutorial-pulse { 50% { outline-color: var(--accent); box-shadow: 0 0 0 10px rgba(120,231,255,.1); } }
    body.tutorial-active #generate { cursor: not-allowed; }
    @media (max-width: 820px) { main { padding-top: 26px; } header { align-items: flex-start; flex-direction: column; } .header-actions { justify-content: flex-start; } .grid, .help-grid, .upscale-grid { grid-template-columns: 1fr; } .row { grid-template-columns: 1fr; } }
    @media (max-width: 470px) { .profiles, .character-grid, .frame-grid, .motion-grid, .field-grid, .help-grid, .tutorial-lesson-grid { grid-template-columns: 1fr; } .character-grid .wide, .field-grid .wide { grid-column: auto; } .profile span { min-height: 0; } .composer { padding: 18px; } .prompt-coach-launcher { right: max(12px, env(safe-area-inset-right)); bottom: max(12px, env(safe-area-inset-bottom)); } .prompt-coach { right: max(12px, env(safe-area-inset-right)); bottom: calc(68px + env(safe-area-inset-bottom)); max-height: calc(100dvh - 80px - env(safe-area-inset-bottom)); } .tutorial-panel { left: max(12px, env(safe-area-inset-left)); bottom: max(12px, env(safe-area-inset-bottom)); width: calc(100vw - max(24px, env(safe-area-inset-left) + env(safe-area-inset-right))); max-height: calc(100dvh - 24px - env(safe-area-inset-top) - env(safe-area-inset-bottom)); } }
    @media (prefers-reduced-motion: reduce) { .tutorial-highlight { animation: none; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">LTX-2 · Q4 fast lane · Q8 advanced lane</div>
        <h1>Dang Studio To DangDom</h1>
        <p class="subtitle">Turn a prompt and optional timed reference frames into a video on this Mac. One render runs at a time.</p>
      </div>
      <div class="header-actions"><button id="openTutorial" class="help-button" type="button" aria-controls="tutorialPanel" aria-expanded="false">Learn Studio</button><button id="openHelp" class="help-button" type="button">Help &amp; setup advisor</button><div class="local-badge">● 100% localhost</div></div>
    </header>
    <details id="helpPanel" class="help-panel" open>
      <summary><span>LTX‑2.5 help, prompt guide &amp; setup advisor</span><small>Choose the right workflow</small></summary>
      <div class="help-body">
        <p class="help-lead"><strong>Good first run:</strong> use Generate, 3 seconds, and Draft or Conservative while testing. Keep a successful seed and change one thing at a time. Use Quality for the final take. One compute job runs at a time.</p>
        <div class="help-advisor">
          <strong>Recommend settings from your current prompt</strong>
          <p>Type the idea in “Describe the shot” below, then run this local rule-based advisor. It reads only the text already in your browser and never uploads it.</p>
          <div class="help-advisor-actions"><button id="startTutorialFromHelp" class="small-button" type="button">Start interactive tutorial</button><button id="recommendSetup" class="small-button" type="button">Recommend setup</button><button id="copyPromptTemplate" class="small-button" type="button">Copy agent prompt template</button></div>
          <div id="setupAdvice" class="setup-advice" aria-live="polite">No prompt analysed yet.</div>
        </div>
        <div class="help-grid">
          <details class="help-topic" open><summary>What to ask an AI agent for</summary><div class="help-copy"><p>Ask for an <strong>LTX‑2.5 video prompt</strong>, not a “Gemma4 prompt.” Gemma 4 is the internal text encoder; LTX‑2.5 is the video model and workflow you are directing.</p><div id="agentPromptTemplate" class="prompt-template">Write a production-ready LTX-2.5 prompt for a 3-second 16:9 video. Use one continuous shot. Include: fixed subject identity and wardrobe; action in chronological order; camera framing and movement; setting, lighting, atmosphere and physical motion; native sound or dialogue; continuity rules; and a short “avoid” clause. Keep it feasible for one short clip. If I provide @character or @object tags, preserve them exactly and keep each dialogue line assigned to its named speaker. Also recommend the most suitable LTX Studio workflow, quality profile, aspect ratio, duration, and reference-image controls.</div></div></details>
          <details class="help-topic" open><summary>Prompt structure that LTX‑2.5 follows best</summary><div class="help-copy"><ul><li><strong>Identity:</strong> who or what must remain unchanged—face, hair, age, clothing, object colours and environment. When a Start image already establishes appearance, focus the prompt on what changes.</li><li><strong>Chronology:</strong> describe the action in order using “begins,” “then,” and “ends.” One continuous shot is the safest short test; LTX‑2.5 also supports two to four explicitly named shots when each transition, new framing, recurring identity and audio continuity are stated.</li><li><strong>Camera:</strong> framing, lens feel, movement and focus behaviour. Avoid asking for several incompatible camera moves inside one shot.</li><li><strong>Look:</strong> lighting, time of day, weather, texture and realism or art style.</li><li><strong>Audio:</strong> ambience and speech. Use Character Voices for dependable speaker identity; native generated dialogue can swap voices.</li><li><strong>Avoid:</strong> unintended cuts, duplicate people, extra limbs, critical exact text, identity changes, or unwanted camera motion. For deliberate multi-shot work, name every cut instead.</li><li><strong>Length:</strong> Studio preserves pasted prompt text instead of clipping it. The LTX/Gemma text encoder still has a finite token context, so concise prompts remain easier for the model to follow.</li></ul></div></details>
          <details class="help-topic"><summary>Generate, quality, duration &amp; aspect</summary><div class="help-copy"><ul><li><strong>Draft:</strong> quickest composition test. <strong>Balanced:</strong> practical preview. <strong>Quality:</strong> preferred final render. <strong>Max:</strong> shorter detail-focused take. <strong>Conservative:</strong> safest 3-second test.</li><li>Use 3 seconds for iteration and complex motion. Clips beyond 5 seconds sharply increase memory pressure and drift; the 10-second control is a ceiling, not a target.</li><li>Use 16:9 for landscape/YouTube, 9:16 for phone/Reels/Shorts, 1:1 for square posts, 4:3 for classic landscape, and 3:4 for portrait.</li><li>Named ratios select the nearest safe model grid; their exact pixels vary by quality profile.</li></ul></div></details>
          <details class="help-topic"><summary>Characters, Ingredients &amp; image references</summary><div class="help-copy"><ul><li><strong>Character Bible</strong> is restart-safe: it stores fixed text and up to four managed reference views locally. Use clear, consistent front and three-quarter images.</li><li><strong>Frame Composer</strong> uses up to four active photos to create one still. Mention every ingredient tag once, then assign the still as Start, End or Interior.</li><li><strong>Start/End/Interior anchors</strong> guide particular times. They are not true global identity embeddings. On this Mac, Start + one Interior + End is the practical complex-shot setup.</li><li><strong>Ingredients</strong> is the true multi-reference identity lane. It needs the Q8 distilled transformer and adapter, uses a fixed 5.04-second bucket, and cannot be combined with Motion Track in the same render. Its trained landscape aspect is recommended; every other ratio is an out-of-training experiment and automatically opens Prompt Coach guidance.</li></ul></div></details>
          <details class="help-topic"><summary>Continue, repair or guide motion</summary><div class="help-copy"><ul><li><strong>Capture current frame</strong> makes a visual hand-off: pause on the last good frame and use it as the next Generate clip’s Start. It does not preserve hidden velocity, latent state or audio.</li><li><strong>Extend</strong> is model-aware continuation before or after a clip and inherits its dimensions. Start with +3 seconds. It requires the Q8/HQ lane.</li><li><strong>Retake</strong> replaces a selected interval. Preserve source audio when the sound must survive. It requires the Q8/HQ lane.</li><li><strong>Scene Beats</strong> divide one continuous generated clip into chronological actions; they are not separate edits or cuts.</li></ul></div></details>
          <details class="help-topic"><summary>Audio, dialogue &amp; voices</summary><div class="help-copy"><ul><li><strong>Audio-to-Video</strong> follows a local audio track and optional start image; it requires the Q8 advanced lane.</li><li>Prompt tags help associate dialogue but do not guarantee native speaker separation.</li><li><strong>Character Voices</strong> makes one isolated local TTS call per explicit @character using an authorized reference voice, supports 23 languages, and can mux timed cues into a separate MP4.</li><li>Use sentence-sized cues, explicit start times and one assigned preset per character. The mux keeps the original video unchanged.</li></ul></div></details>
          <details class="help-topic"><summary>Union Control, storyboard &amp; Motion Track</summary><div class="help-copy"><ul><li><strong>Union Control</strong> follows structure from a control video: Canny edges can be derived locally; depth, pose or other guides must be prepared first.</li><li>A comic strip or static storyboard does not automatically become complete motion. Use its panels as Start, Interior and End anchors, or turn it into a timed guide video before Union Control.</li><li><strong>Motion Track</strong> needs a Start image and normalized paths to steer selected movement. It requires its adapter plus Q8 and is a separate render from Ingredients.</li><li>The preferred 512×320 Motion profile is the reliable quality ceiling for this Mac. The 640×384 profile is a three-second experiment with substantially higher attention work and automatically opens Prompt Coach guidance.</li><li>Use Union for whole-frame structure; use Motion Track for trajectories; use anchors for a few key compositions.</li></ul></div></details>
          <details class="help-topic"><summary>Results, capture &amp; proper upscaling</summary><div class="help-copy"><ul><li><strong>AI Quality Upscale · 1080p</strong> is the recommended final pass. Real‑ESRGAN x4plus reconstructs edges and texture on the Apple GPU, then writes a separate aspect-preserving MP4 with the source frame rate and audio.</li><li><strong>Fast 720p</strong> uses Lanczos and mild sharpening. It is much faster and more faithful, but does not reconstruct missing detail.</li><li>AI upscaling cannot repair anatomy, identity, motion or composition. Inspect hair, wings, droplets and other fine texture because independent-frame enhancement can introduce shimmer.</li><li>Keep faces and important hands in close or medium framing when using 384×224 or 384×256 control workflows.</li><li><strong>Download MP4</strong> uses your browser's download location. Every upscale keeps the original render.</li></ul></div></details>
          <details class="help-topic"><summary>Current availability on this Mac</summary><div class="help-copy"><ul id="currentAvailabilityList"><li>Checking the installed workflow assets…</li></ul></div></details>
          <details class="help-topic"><summary>16 GB Mac best practices</summary><div class="help-copy"><ul><li>Plug in power, close GPU-heavy applications, and begin with a 3-second render.</li><li>Use one main action and one deliberate camera move per clip. Complex choreography, crowds and tiny faces are more likely to hallucinate.</li><li>Reuse a successful seed while changing only one variable; a seed helps comparison but does not guarantee identical output after other settings change.</li><li>A percentage appears only during reported denoising steps. Model loading, prompt encoding, decoding and muxing can remain indeterminate.</li><li>Prompts, references, voices and outputs stay on this Mac; the studio has no analytics or cloud generation calls.</li></ul></div></details>
        </div>
      </div>
    </details>
    <div class="grid">
      <section class="card composer">
        <form id="form">
          <div class="mode-tabs" role="tablist" aria-label="Workflow mode">
            <button id="modeTabGenerate" class="mode-tab" type="button" data-select-mode="generate" aria-selected="true">Generate</button>
            <button id="modeTabRetake" class="mode-tab" type="button" data-select-mode="retake" aria-selected="false">Retake <span class="badge">Beta · Q8</span></button>
            <button id="modeTabExtend" class="mode-tab" type="button" data-select-mode="extend" aria-selected="false">Extend <span class="badge">Beta · Q8</span></button>
            <button id="modeTabA2v" class="mode-tab" type="button" data-select-mode="a2v" aria-selected="false">Audio-to-Video <span class="badge">Beta · Q8</span></button>
            <button id="modeTabIngredients" class="mode-tab" type="button" data-select-mode="ingredients" aria-selected="false">Ingredients <span class="badge">Q8</span></button>
            <button id="modeTabUnion" class="mode-tab" type="button" data-select-mode="union" aria-selected="false">Union Control <span class="badge">Q8</span></button>
            <button id="modeTabMotion" class="mode-tab" type="button" data-select-mode="motion" aria-selected="false">Motion Track <span class="badge">Q8</span></button>
          </div>
          <div id="modeReadiness" class="readiness">Checking local workflow readiness…</div>

          <label id="promptLabel" for="prompt">Describe the shot</label>
          <textarea id="prompt" required placeholder="A slow cinematic push through a rain-soaked neon alley, reflections rippling in puddles, subtle handheld movement…"></textarea>
          <div class="counter"><span id="count">0</span> characters · full text preserved</div>

          <fieldset id="generateSettingsPanel" data-modes="generate">
            <legend>Quality profile</legend>
            <div class="profiles">
              <label class="profile"><input type="radio" name="profile" value="draft"><span><strong>Draft</strong><small>Fast 5+2 schedule · 512×256</small></span></label>
              <label class="profile"><input type="radio" name="profile" value="balanced"><span><strong>Balanced</strong><small>Graded 8+2 · practical preview</small></span></label>
              <label class="profile"><input type="radio" name="profile" value="quality" checked><span><strong>Quality</strong><small>Graded 8+2 · best final detail</small></span></label>
              <label class="profile"><input type="radio" name="profile" value="max"><span><strong>Max detail</strong><small>Graded 8+2 · short high-detail take</small></span></label>
              <label class="profile"><input type="radio" name="profile" value="clip3s"><span><strong>Conservative</strong><small>Graded 8+2 · 512×256 · 3 seconds</small></span></label>
            </div>
            <div class="duration-box" style="margin-top:10px">
              <div class="duration-head"><label for="generateDuration">Duration</label><span id="generateDurationValue" class="duration-value"></span></div>
              <input id="generateDuration" type="range" min="1" max="10" step="1" value="3">
              <div id="generateDurationWarning" class="memory-warning" hidden><strong>16 GB warning:</strong> clips over 5 seconds sharply increase attention memory and identity drift. Prefer separate 3-second takes, then continue them with the Q8 Extend lane.</div>
            </div>
            <div class="field-grid" style="margin-top:10px">
              <div><label for="generateAspect">Aspect ratio</label><select id="generateAspect"><option value="profile" selected>Profile default · landscape</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · vertical</option><option value="1:1">1:1 · square</option><option value="4:3">4:3 · classic landscape</option><option value="3:4">3:4 · portrait</option></select></div>
              <p class="technical-note wide">Named ratios use the nearest safe 64-pixel LTX grid while preserving the selected quality profile's approximate pixel budget.</p>
              <div class="wide"><label for="gemmaPack">Text encoder provenance</label><select id="gemmaPack"><option value="auto" selected>Auto · official preferred, validated fallback if needed</option><option value="official">Official Phosphene only</option><option value="ddalcu-fallback">Validated ddalcu compatibility fallback</option></select><p id="gemmaPackNote" class="technical-note">Checking the locally verified Gemma packs…</p></div>
            </div>
          </fieldset>

          <section id="retakePanel" class="mode-panel" data-mode-panel="retake" hidden>
            <h2>Retake an interval</h2><p>Runs in the separate low-RAM Q8 advanced lane; the normal Q4 generator stays unchanged.</p>
            <div class="field-grid">
              <div class="wide"><label for="retakeVideo">Source video · max 64 MB</label><input id="retakeVideo" type="file" accept="video/*,.mkv,.avi"></div>
              <div><label for="retakeStart">Start · seconds</label><input id="retakeStart" type="number" min="0" max="86400" step="0.01" value="0"></div>
              <div><label for="retakeEnd">End · seconds</label><input id="retakeEnd" type="number" min="0.01" max="86400" step="0.01" value="1"></div>
              <label class="check-row wide"><input id="retakePreserveAudio" type="checkbox"> Preserve source audio instead of regenerating the interval’s audio</label>
            </div>
          </section>

          <section id="extendPanel" class="mode-panel" data-mode-panel="extend" hidden>
            <h2>Extend a clip</h2><p>Native latent continuation in the separate low-RAM Q8 advanced lane. Extend always inherits the source video's aspect ratio and dimensions. Three seconds is the recommended first pass.</p>
            <div class="field-grid">
              <div class="wide"><label for="extendVideo">Source video · max 64 MB</label><input id="extendVideo" type="file" accept="video/*,.mkv,.avi"></div>
              <div><label for="extendSeconds">Add duration</label><select id="extendSeconds"><option value="1">+1 second</option><option value="2">+2 seconds</option><option value="3" selected>+3 seconds · recommended</option><option value="4">+4 seconds</option><option value="5">+5 seconds</option><option value="6">+6 seconds</option><option value="7">+7 seconds</option><option value="8">+8 seconds · maximum</option></select></div>
              <div><label for="extendDirection">Direction</label><select id="extendDirection"><option value="after" selected>After source</option><option value="before">Before source</option></select></div>
            </div>
            <div class="memory-warning">Combined output is capped at 241 frames (~10.04s). The wrapper may reject an extension if the uploaded source plus added duration exceeds that cap.</div>
          </section>

          <section id="a2vPanel" class="mode-panel" data-mode-panel="a2v" hidden>
            <h2>Audio-to-Video</h2><p>Uses the separate low-RAM Q8 dev + distilled two-stage lane, driven by your local audio.</p>
            <div class="field-grid">
              <div class="wide"><label for="a2vAudio">Audio · max 64 MB</label><input id="a2vAudio" type="file" accept="audio/*,.caf,.aif,.aiff"></div>
              <div><label for="a2vProfile">Quality profile</label><select id="a2vProfile"><option value="draft">Draft</option><option value="balanced">Balanced</option><option value="quality">Quality</option><option value="max">Max detail</option><option value="clip3s" selected>Conservative</option></select></div>
              <div><label for="a2vAspect">Aspect ratio</label><select id="a2vAspect"><option value="profile" selected>Profile default · landscape</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · vertical</option><option value="1:1">1:1 · square</option><option value="4:3">4:3 · classic landscape</option><option value="3:4">3:4 · portrait</option></select></div>
              <div><label for="a2vAudioStart">Audio start offset · seconds</label><input id="a2vAudioStart" type="number" min="0" max="86400" step="0.01" value="0"></div>
              <div class="wide duration-box"><div class="duration-head"><label for="a2vDuration">Duration</label><span id="a2vDurationValue" class="duration-value"></span></div><input id="a2vDuration" type="range" min="1" max="10" step="1" value="3"><div id="a2vDurationWarning" class="memory-warning" hidden><strong>16 GB warning:</strong> over 5 seconds uses much more attention memory and can drift. Prefer 3 seconds and iterate.</div></div>
              <div><label for="a2vStartImage">Optional start image</label><input id="a2vStartImage" type="file" accept="image/png,image/jpeg,image/webp"></div>
              <div><label for="a2vImageStrength">Image strength</label><input id="a2vImageStrength" type="number" min="0" max="1" step="0.05" value="1"></div>
            </div>
          </section>

          <section id="ingredientsPanel" class="mode-panel" data-mode-panel="ingredients" hidden>
            <h2>Ingredients references</h2><p>Uses the official Ingredients adapter with the Q8 distilled lane so multi-reference identity control is not crushed by Q4 fusion. You may also mix Start, End, and Interior anchors.</p>
            <div class="locked-duration"><strong>Locked duration:</strong> 121 frames / 5.04 seconds. The trained landscape canvas is recommended; other aspect ratios are available only as clearly marked experiments because the Ingredients adapter was trained on one landscape bucket.</div>
            <div class="memory-warning"><strong>Separate access required:</strong> the official Ingredients adapter is gated independently from the base model and must already exist locally.</div>
            <div class="field-grid" style="margin-top:11px">
              <div><label for="ingredientsProfile">Profile</label><select id="ingredientsProfile"><option value="ingredients5s" selected>Preferred on 16 GB · 384×224</option><option value="trained5s" data-experimental="ingredients-memory">Official bucket · 768×448 · Experimental on 16 GB</option></select></div>
              <div><label for="ingredientsAspect">Aspect ratio</label><select id="ingredientsAspect"><option value="profile" selected>Trained landscape · Preferred</option><option value="16:9" data-experimental="ingredients-aspect">16:9 · Experimental</option><option value="9:16" data-experimental="ingredients-aspect">9:16 · Experimental portrait</option><option value="1:1" data-experimental="ingredients-aspect">1:1 · Experimental square</option><option value="4:3" data-experimental="ingredients-aspect">4:3 · Experimental landscape</option><option value="3:4" data-experimental="ingredients-aspect">3:4 · Experimental portrait</option></select></div>
              <div><label for="ingredientsLora">LoRA strength</label><input id="ingredientsLora" type="number" min="0" max="2" step="0.05" value="1"></div>
              <div><label for="ingredientsConditioning">Guide strength</label><input id="ingredientsConditioning" type="number" min="0" max="1" step="0.05" value="1"></div>
            </div>
            <div id="ingredientsAspectWarning" class="memory-warning" hidden><strong>Experimental canvas:</strong> Prompt Coach has opened with the exact risks and a safer comparison plan. Identity and reference adherence can be weaker outside the trained landscape bucket.</div>
            <div class="anchor-tools"><span id="ingredientCounter" class="slot-counter">0 / 8 references</span><button id="addIngredient" class="small-button" type="button">+ Add reference</button></div>
            <div id="ingredientList" class="reference-list"></div>
            <label for="ingredientDialogue">Speaker-tagged dialogue guidance <span style="color:#71747e;font-weight:400">(optional)</span></label>
            <textarea id="ingredientDialogue" maxlength="2000" placeholder="@maya: &quot;We leave now.&quot;&#10;@jon: &quot;I’m right behind you.&quot;"></textarea>
            <p class="technical-note"><strong>Voice limitation:</strong> tags can guide who says each line, but native generated audio may still swap voices. For reliable speaker separation, use separately recorded and labeled tracks in a later audio pass.</p>
          </section>

          <section id="unionPanel" class="mode-panel" data-mode-panel="union" hidden>
            <h2>Union Control</h2><p>Generate a Canny guide locally from ordinary footage, or upload an already prepared Canny/depth/pose guide.</p>
            <div class="field-grid">
              <div><label for="unionProfile">Profile</label><select id="unionProfile"><option value="control3s" selected>16 GB-friendly · 384×256</option><option value="control3s-plus">Higher detail · 512×320</option></select></div>
              <div><label for="unionAspect">Aspect ratio</label><select id="unionAspect"><option value="profile" selected>Profile default · landscape</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · vertical</option><option value="1:1">1:1 · square</option><option value="4:3">4:3 · classic landscape</option><option value="3:4">3:4 · portrait</option></select></div>
              <div><label for="unionRoute">Input kind</label><select id="unionRoute"><option value="source" selected>Ordinary footage → built-in Canny</option><option value="guide">Prepared guide upload</option></select></div>
              <div><label for="unionControlType">Guide type</label><select id="unionControlType"><option value="canny" selected>Canny</option><option value="depth">Prepared depth</option><option value="pose">Prepared pose</option><option value="prepared">Other prepared guide</option></select></div>
              <div><label for="unionVideo">Source / guide video · max 64 MB</label><input id="unionVideo" type="file" accept="video/*,.mkv,.avi"></div>
              <div><label for="unionLora">LoRA strength</label><input id="unionLora" type="number" min="0" max="2" step="0.05" value="1"></div>
              <div><label for="unionConditioning">Guide strength</label><input id="unionConditioning" type="number" min="0" max="1" step="0.05" value="1"></div>
              <div class="wide duration-box"><div class="duration-head"><label for="unionDuration">Duration</label><span id="unionDurationValue" class="duration-value"></span></div><input id="unionDuration" type="range" min="1" max="10" step="1" value="3"><div id="unionDurationWarning" class="memory-warning" hidden><strong>16 GB warning:</strong> control clips over 5 seconds sharply increase attention memory and identity drift. Prefer separate 3-second takes and stitch them.</div></div>
            </div>
          </section>

          <section id="motionPanel" class="mode-panel" data-mode-panel="motion" hidden>
            <h2>Motion Track</h2><p>Animate the required start image along 1–8 normalized motion paths. JSON is validated before launch.</p>
            <div class="field-grid">
              <div><label for="motionProfile">Profile</label><select id="motionProfile"><option value="control3s">Conservative · 384×256</option><option value="control3s-plus" selected>Preferred quality · 512×320</option><option value="motion3s-experimental" data-experimental="motion-resolution">Experimental · 640×384 · 3s only</option></select></div>
              <div><label for="motionAspect">Aspect ratio</label><select id="motionAspect"><option value="profile" selected>Profile default · landscape</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · vertical</option><option value="1:1">1:1 · square</option><option value="4:3">4:3 · classic landscape</option><option value="3:4">3:4 · portrait</option></select></div>
              <div><label for="motionLora">LoRA strength</label><input id="motionLora" type="number" min="0" max="2" step="0.05" value="1"></div>
              <div><label for="motionConditioning">Guide strength</label><input id="motionConditioning" type="number" min="0" max="1" step="0.05" value="1"></div>
              <div class="wide duration-box"><div class="duration-head"><label for="motionDuration">Duration</label><span id="motionDurationValue" class="duration-value"></span></div><input id="motionDuration" type="range" min="1" max="10" step="1" value="3"><div id="motionDurationWarning" class="memory-warning" hidden><strong>16 GB warning:</strong> motion-control clips over 5 seconds sharply increase attention memory and identity drift. Prefer separate 3-second takes and stitch them.</div></div>
              <div class="wide"><label for="motionTracks">Normalized path JSON</label><textarea id="motionTracks" class="json-editor" spellcheck="false">[[{"x":0.15,"y":0.5},{"x":0.5,"y":0.35},{"x":0.85,"y":0.5}]]</textarea></div>
            </div>
            <div id="motionProfileWarning" class="memory-warning" hidden><strong>Experimental native resolution:</strong> duration was returned to three seconds and Prompt Coach has opened with memory, runtime, and Python-exit risks. Use the preferred profile if reliability matters.</div>
          </section>

          <div class="row" style="grid-template-columns:minmax(0,280px)">
            <div>
              <label for="seed">Seed</label>
              <div class="seed-wrap">
                <input id="seed" type="number" min="0" max="4294967295" step="1" value="42" required>
                <button class="icon-button" id="randomSeed" type="button" title="Randomize seed" aria-label="Randomize seed">↻</button>
              </div>
            </div>
          </div>

          <details id="characterBiblePanel" class="character-bible" open>
            <summary><span>Character Bible <span style="color:#71747e;font-weight:400">(optional)</span></span><small>Identity controls</small></summary>
            <div class="character-body">
              <p class="identity-note"><strong>Restart-safe library:</strong> saved text and up to four reference images stay only on this Mac under <code>work/local-video/characters</code>. Loading a profile automatically includes its managed references in supported image-conditioned renders. Voice assignments remain separate.</p>
              <div class="character-profile-row">
                <div><label for="characterProfileSelect">Saved visual character</label><select id="characterProfileSelect"><option value="">No saved characters yet</option></select></div>
                <button id="loadCharacterProfile" class="small-button" type="button">Load</button>
              </div>
              <div class="character-profile-actions">
                <button id="saveCharacterProfile" class="small-button" type="button">Save as new</button>
                <button id="replaceCharacterProfile" class="small-button" type="button">Replace selected</button>
                <button id="deleteCharacterProfile" class="remove-anchor" type="button">Delete selected</button>
              </div>
              <div id="characterProfileStatus" class="character-profile-status">Visual character profiles are stored locally and are never uploaded externally.</div>
              <div class="character-grid">
                <div class="wide">
                  <label for="characterName">Character name</label>
                  <input id="characterName" type="text" maxlength="80" autocomplete="off" placeholder="Mara Voss">
                </div>
                <div>
                  <label for="characterAppearance">Fixed appearance</label>
                  <textarea id="characterAppearance" maxlength="400" placeholder="Age range, face, hair, eyes, build, distinguishing marks…"></textarea>
                </div>
                <div>
                  <label for="characterWardrobe">Fixed wardrobe</label>
                  <textarea id="characterWardrobe" maxlength="300" placeholder="Clothing, colors, accessories, materials…"></textarea>
                </div>
                <div>
                  <label for="characterContinuity">Defining traits / continuity rules</label>
                  <textarea id="characterContinuity" maxlength="400" placeholder="Mannerisms and details that must remain unchanged in every frame…"></textarea>
                </div>
                <div>
                  <label for="characterAvoid">Avoid / negative constraints</label>
                  <textarea id="characterAvoid" maxlength="300" placeholder="No wardrobe changes, extra accessories, altered hair, duplicate limbs…"></textarea>
                </div>
                <div class="wide">
                  <label>Character reference views · up to 4 images / 24 MB total</label>
                  <label class="upload" for="characterReferenceImages"><span id="characterReferenceCopy" class="upload-copy">PNG, JPEG, or WebP · managed locally when saved</span><input id="characterReferenceImages" type="file" multiple accept="image/png,image/jpeg,image/webp"></label>
                  <div id="characterReferencePreview" class="previews"></div>
                </div>
              </div>
            </div>
          </details>

          <details id="characterVoicesPanel" class="control-panel">
            <summary><span>Character Voices <span style="color:#71747e;font-weight:400">(optional)</span></span><small>Local · 23 languages</small></summary>
            <div class="panel-body">
              <p class="technical-note"><strong>Identity is reference-anchored:</strong> a clean, authorized WAV creates a reusable voice preset. A prose description changes only coarse delivery (calm, neutral, or dramatic); it cannot invent an exact persistent speaker identity.</p>
              <div id="voiceReadiness" class="voice-status">Checking the optional local voice backend…</div>

              <div class="reference-card">
                <div class="reference-head"><strong>1 · Register an authorized voice preset</strong></div>
                <div class="field-grid" style="margin-top:9px">
                  <div><label for="voicePresetName">Preset ID</label><input id="voicePresetName" type="text" maxlength="64" autocomplete="off" placeholder="maya-voice"></div>
                  <div><label for="voiceReferenceLanguage">Reference language</label><select id="voiceReferenceLanguage"></select></div>
                  <div class="wide"><label for="voiceReferenceFile">Canonical reference · PCM16 WAV · 3–30s · max 50 MB</label><input id="voiceReferenceFile" type="file" accept="audio/wav,.wav"></div>
                  <div class="wide"><label for="voicePresetDescription">Delivery note <span style="color:#71747e;font-weight:400">(optional, not identity)</span></label><input id="voicePresetDescription" type="text" maxlength="500" placeholder="warm, calm, measured delivery"></div>
                  <label class="check-row wide"><input id="voiceConsent" type="checkbox"> I confirm this is my voice or I have explicit permission to clone it.</label>
                  <label class="check-row wide"><input id="voiceReplacePreset" type="checkbox"> Deliberately replace an existing preset with this ID.</label>
                </div>
                <button id="registerVoicePreset" class="small-button" type="button" style="margin-top:10px">Register preset locally</button>
              </div>

              <div class="reference-card" style="margin-top:9px">
                <div class="reference-head"><strong>2 · Assign one preset to each @character</strong></div>
                <div class="field-grid" style="margin-top:9px">
                  <div><label for="voiceCharacter">Character tag</label><input id="voiceCharacter" type="text" maxlength="65" autocomplete="off" placeholder="@maya"></div>
                  <div><label for="voicePresetSelect">Voice preset</label><select id="voicePresetSelect"><option value="">Register a preset first</option></select></div>
                  <label class="check-row wide"><input id="voiceReplaceAssignment" type="checkbox"> Deliberately replace this character’s existing assignment.</label>
                </div>
                <button id="assignVoicePreset" class="small-button" type="button" style="margin-top:10px">Assign character voice</button>
                <div id="voiceAssignments" class="voice-job-progress">No character assignments yet.</div>
              </div>

              <div class="reference-card" style="margin-top:9px">
                <div class="reference-head"><strong>3 · Render isolated dialogue cues</strong><span id="voiceCueCounter" class="slot-counter">0 / 12 cues</span></div>
                <p class="technical-note"><strong>No speaker intermixing:</strong> each row is one separate TTS call tied to its explicit assigned @character. Dialogue is never inferred from the video prompt and two speakers are never sent in one call.</p>
                <div id="voiceCueList" class="voice-cue-list"></div>
                <div class="anchor-tools"><span>Sentence-sized cues reduce unwanted continuation.</span><button id="addVoiceCue" class="small-button" type="button">+ Add dialogue cue</button></div>
                <label class="check-row"><input id="voiceMuxLatest" type="checkbox"> After synthesis, replace the latest completed video’s native audio with these timed cues.</label>
                <p class="technical-note"><strong>Conservative mux:</strong> each cue is delayed by its start time, mixed into one dialogue bus, AAC-encoded, and atomically published as a new MP4. The original video is kept unchanged.</p>
                <div class="actions">
                  <button id="synthesizeVoices" class="small-button" type="button">Generate voice cues locally</button>
                  <button id="cancelVoices" class="remove-anchor" type="button" hidden>Cancel voices</button>
                </div>
                <div id="voiceNotice" class="notice" role="alert"></div>
                <div id="voiceJobProgress" class="voice-job-progress"></div>
                <div id="voiceResults" class="voice-result-list"></div>
              </div>
            </div>
          </details>

          <details id="frameComposerPanel" class="control-panel" data-modes="generate ingredients union motion">
            <summary><span>Frame Composer <span style="color:#71747e;font-weight:400">(optional)</span></span><small>FLUX.2 Klein Q4</small></summary>
            <div class="panel-body">
              <p class="technical-note"><strong>Identity-first still composition:</strong> create an ingredient such as <code>@maya</code> and add one or more reference views to it. Mention each ingredient tag once in the composition prompt, then assign the resulting still as a Start, End, or Interior anchor. FLUX accepts four active photos total across all ingredients; each additional view consumes a slot. The separate Ingredients library may still hold eight references.</p>
              <div id="frameComposerReadiness" class="frame-composer-status">Checking optional Frame Composer readiness…</div>
              <label for="frameComposerPrompt">Frame composition prompt</label>
              <textarea id="frameComposerPrompt" placeholder="Keep @maya's face and hair; dress @maya in the exact jacket from @jacket; cinematic medium shot, no duplicates, no text."></textarea>
              <div class="counter"><span id="frameComposerCount">0</span> characters · full text preserved</div>
              <div class="field-grid" style="margin-top:10px">
                <div><label for="frameComposerProfile">Matched LTX anchor profile</label><select id="frameComposerProfile" disabled><option value="draft">Draft / 3s · 512×256</option><option value="balanced">Balanced · 640×384</option><option value="quality" selected>Quality / Max · 768×448</option><option value="max">Max · 768×448</option><option value="clip3s">Conservative · 512×256</option><option value="ingredients5s">Ingredients-friendly · 384×224</option><option value="trained5s">Ingredients official · 768×448</option><option value="control3s">Control · 384×256</option><option value="control3s-plus">Control Plus · 512×320</option><option value="motion3s-experimental">Motion experimental · 640×384</option></select><p id="frameComposerCanvas" class="technical-note">Automatically matches the active workflow profile and aspect ratio.</p></div>
                <div><label for="frameComposerSeed">Seed</label><input id="frameComposerSeed" type="number" min="0" max="4294967295" step="1" value="42"></div>
              </div>
              <div class="anchor-tools"><span id="frameComposerCounter" class="slot-counter">0 / 4 active photos · 0 ingredients</span><button id="addFrameComposerReference" class="small-button" type="button">+ Add ingredient</button></div>
              <div id="frameComposerReferences" class="reference-list"></div>
              <div class="actions" style="margin-top:10px">
                <button id="composeFrame" class="small-button" type="button">Compose frame locally</button>
                <button id="cancelFrameComposer" class="remove-anchor" type="button" hidden>Cancel frame</button>
              </div>
              <div id="frameComposerNotice" class="notice" role="alert"></div>
              <div id="frameComposerResult" class="frame-composer-result" hidden>
                <img id="frameComposerImage" alt="Composed frame preview">
                <div class="frame-composer-actions">
                  <button id="assignComposedStart" class="small-button" type="button">Use as Start</button>
                  <button id="assignComposedEnd" class="small-button" type="button">Use as End</button>
                  <button id="assignComposedInterior" class="small-button" type="button">Add as Interior</button>
                  <a id="downloadComposedFrame" class="download" download>Download PNG</a>
                </div>
              </div>
            </div>
          </details>

          <details id="sceneBeatsPanel" class="control-panel" data-modes="generate">
            <summary><span>Scene Beats &amp; Motion <span style="color:#71747e;font-weight:400">(optional)</span></span><small>Prompt Relay</small></summary>
            <div class="panel-body">
              <p class="technical-note"><strong>Temporal guidance:</strong> each scene beat is a local action prompt for its part of this one continuous clip. Character Bible and the main shot prompt remain global. Optional lengths use latent frames; leave them blank for a balanced automatic split.</p>
              <p class="technical-note"><strong>Keyframe status:</strong> model-generated keyframe slots belong to the dev/HQ lane and are disabled for the installed distilled Q4 pack. Uploaded start, end, and interior timeline anchors below remain supported.</p>
              <div class="anchor-tools"><span id="sceneCounter" class="slot-counter">0 beats · 7 latent frames</span><button id="addScene" class="small-button" type="button">+ Add scene beat</button></div>
              <div id="sceneList" class="scene-list"></div>
            </div>
          </details>

          <details id="boundaryFramesPanel" class="control-panel" data-modes="generate ingredients union motion" open>
            <summary><span>Start &amp; End Frames <span style="color:#71747e;font-weight:400">(optional)</span></span><small>Boundary anchors</small></summary>
            <div class="panel-body">
              <p class="technical-note"><strong>Accurate behavior:</strong> an uploaded image anchors the matching boundary frame. A description only conditions the main video prompt; it does not synthesize a separate still image.</p>
              <div class="frame-grid">
                <div class="frame-card">
                  <h3>Start frame</h3>
                  <label class="upload" for="startImage"><span class="upload-copy" id="startCopy">PNG, JPEG, or WebP · max 8 MB</span><input id="startImage" type="file" accept="image/png,image/jpeg,image/webp"></label>
                  <div id="startPreview" class="previews"></div>
                  <div class="compact-grid">
                    <div><label for="startDescription">Description</label><textarea id="startDescription" maxlength="500" placeholder="Opening composition, pose, lighting…"></textarea></div>
                    <div><label for="startStrength">Image strength</label><input id="startStrength" type="number" min="0" max="1" step="0.05" value="1"></div>
                  </div>
                </div>
                <div class="frame-card">
                  <h3>End frame</h3>
                  <label class="upload" for="endImage"><span class="upload-copy" id="endCopy">PNG, JPEG, or WebP · max 8 MB</span><input id="endImage" type="file" accept="image/png,image/jpeg,image/webp"></label>
                  <div id="endPreview" class="previews"></div>
                  <div class="compact-grid">
                    <div><label for="endDescription">Description</label><textarea id="endDescription" maxlength="500" placeholder="Final composition, pose, lighting…"></textarea></div>
                    <div><label for="endStrength">Image strength</label><input id="endStrength" type="number" min="0" max="1" step="0.05" value="1"></div>
                  </div>
                </div>
              </div>
            </div>
          </details>

          <details id="timelineAnchorsPanel" class="control-panel" data-modes="generate ingredients union motion" open>
            <summary><span>Multi-reference Timeline Anchors <span style="color:#71747e;font-weight:400">(optional)</span></span><small>Interior anchors</small></summary>
            <div class="panel-body">
              <p class="technical-note"><strong>Not LTX Ingredients IC-LoRA:</strong> these session-only groups become evenly spaced timed frame anchors, not a global semantic or identity embedding. Images are center-cropped to the render aspect ratio. On this 16 GB Mac, start + one interior + end is the practical recommendation; four active images are complex and can use roughly twice the attention memory at Quality.</p>
              <div class="anchor-tools"><span id="slotCounter" class="slot-counter">0 / 4 active images</span><button id="addAnchor" class="small-button" type="button">+ Add anchor group</button></div>
              <div id="anchorList" class="anchor-list"></div>
            </div>
          </details>

          <div class="actions">
            <button id="generate" class="generate" type="submit">Generate video</button>
            <button id="cancel" class="cancel" type="button" hidden>Cancel</button>
          </div>
          <div id="notice" class="notice" role="alert"></div>
        </form>
        <p class="privacy">Prompts and images stay on this machine. This page contains no analytics, cloud APIs, or external assets.</p>
      </section>

      <section id="outputPanel" class="card output" aria-live="polite">
        <div class="statusbar">
          <div class="status-label"><span id="dot" class="dot idle"></span><span id="status">Ready</span></div>
          <div id="meta" class="meta">idle</div>
        </div>
        <div id="progressWrap" class="progress-wrap" hidden>
          <div class="progress-copy"><span id="progressStage">Queued</span><span id="progressValue">Working locally</span></div>
          <progress id="progressBar" max="100"></progress>
        </div>
        <div class="stage">
          <div id="empty" class="empty"><div class="empty-mark">◫</div>Your completed video will appear here.</div>
          <video id="video" controls playsinline preload="metadata"></video>
          <div id="resultActions" class="result-actions"><a id="download" class="download" download>Download MP4</a></div>
        </div>
        <div id="upscalePanel" class="upscale-panel">
          <div class="upscale-grid">
            <div class="upscale-option recommended">
              <div class="upscale-option-head"><strong>AI Quality Upscale · 1080p</strong><span class="upscale-badge">Recommended</span></div>
              <div class="capture-copy">Real‑ESRGAN reconstructs detail on the Apple GPU, then preserves the original aspect ratio, frame rate and audio. Best choice for a finished, shareable video.</div>
              <div class="upscale-actions"><button id="createAiUpscale" class="small-button" type="button" disabled>AI upscale to 1080p</button></div>
            </div>
            <div class="upscale-option">
              <div class="upscale-option-head"><strong>Fast 720p resize</strong></div>
              <div class="capture-copy">Lanczos resize with mild sharpening. Quick and faithful, but it only adds pixels—it does not reconstruct detail.</div>
              <div class="upscale-actions"><button id="createUpscale" class="small-button" type="button" disabled>Create fast 720p copy</button></div>
            </div>
          </div>
          <div class="upscale-actions">
            <button id="cancelUpscale" class="remove-anchor" type="button" hidden>Cancel upscale</button>
            <a id="downloadUpscale" class="download" download hidden>Download upscaled MP4</a>
          </div>
          <div id="upscaleProgress" class="upscale-progress" hidden>
            <div class="progress-copy"><span id="upscaleProgressStage">Preparing</span><span id="upscaleProgressValue">Working locally</span></div>
            <progress id="upscaleProgressBar" max="100"></progress>
          </div>
          <div id="upscaleStatus" class="upscale-status">Finish a video to enable local upscaling.</div>
          <div class="capture-copy" style="margin-top:6px"><strong>AI limitation:</strong> it can restore edges and texture, but cannot repair anatomy, identity, motion or composition. Because frames are enhanced independently, very fine texture may shimmer.</div>
        </div>
        <div id="capturePanel" class="capture-panel">
          <div class="capture-head">
            <div><strong>Continue from a displayed frame</strong><div class="capture-copy">Pauses the completed video and saves its exact current frame locally as a PNG. This prepares a separate image-to-video clip; it does not trim or stitch the rejected tail, carry its audio, or perform native Extend.</div></div>
            <button id="captureCurrentFrame" class="small-button" type="button" disabled>Capture current frame</button>
          </div>
          <div id="captureStatus" class="capture-copy" style="margin-top:8px">Finish a video, seek to the desired hand-off frame, then capture it.</div>
          <div id="captureResult" class="capture-result" hidden>
            <img id="capturePreview" alt="Captured video frame preview">
            <div><div id="captureTimestamp" class="meta"></div><div class="capture-copy">For the next prompt, describe only what happens after this frame. The captured image preserves the visual hand-off, but not hidden motion or audio state. Trim the rejected tail and stitch clips separately.</div><div class="capture-result-actions"><button id="useCapturedStart" class="small-button" type="button">Use as next clip’s start frame</button><a id="downloadCapturedFrame" class="download" download>Download PNG</a></div></div>
          </div>
        </div>
        <div class="log-head">Render log</div>
        <pre id="log">Waiting for a render…</pre>
      </section>
    </div>
  </main>
  <button id="promptCoachLauncher" class="prompt-coach-launcher" type="button" aria-controls="promptCoachPanel" aria-expanded="false">✦ Prompt Coach <span>Local</span></button>
  <aside id="promptCoachPanel" class="prompt-coach" role="dialog" aria-modal="true" aria-label="LTX-2.5 Prompt Coach" hidden>
    <div class="prompt-coach-head">
      <div><strong>LTX‑2.5 Prompt Coach</strong><small>Creative development · conservative review · exact walkthrough</small><span id="promptCoachProvider" class="coach-provider">Private local agent</span></div>
      <button id="closePromptCoach" class="prompt-coach-close" type="button" aria-label="Close Prompt Coach">×</button>
    </div>
    <div class="prompt-coach-scroll">
      <div id="promptCoachConversation" class="coach-conversation" aria-live="polite"><div class="coach-chat assistant"><span class="coach-chat-meta">Coach · LTX‑2.5 Production Skill v1.4 loaded</span>Use Ask Coach for any Studio question or follow-up; chat never rewrites your video prompt. Use Develop / review prompt only when you want me to create a revised production draft, and Score prompt only for the fixed readiness checklist. I reason over the selected mode, duration, aspect ratio, references and locally available features.<div class="coach-disclaimer">The percentage measures prompt completeness—not the probability of a successful render. Even 100% cannot guarantee anatomy, identity, motion or audio.</div></div></div>
      <label for="promptCoachInput">Prompt under review</label>
      <textarea id="promptCoachInput" placeholder="Paste a video idea here…"></textarea>
      <div class="counter"><span id="promptCoachCount">0</span> characters · full text preserved</div>
      <div class="coach-actions">
        <button id="promptCoachLoad" class="small-button" type="button">Load Studio prompt</button>
        <button id="promptCoachScoreButton" class="small-button" type="button" aria-describedby="promptCoachActionHelp">Score prompt</button>
        <button id="promptCoachAnalyze" class="coach-primary" type="button" aria-describedby="promptCoachActionHelp">Develop / review prompt</button>
        <p id="promptCoachActionHelp" class="coach-action-help"><strong>Score prompt</strong> runs the fixed readiness checklist without adding a chat turn or rewriting your words. <strong>Develop / review prompt</strong> authorizes Coach to produce a new draft. Questions belong in the separate Ask Coach box below and never rewrite this prompt.</p>
      </div>
      <div id="promptCoachNotice" class="notice" role="alert"></div>
      <section id="promptCoachResult" class="coach-result" hidden>
        <details id="promptCoachAudit" class="coach-audit" open>
          <summary id="promptCoachAuditSummary">Prompt audit &amp; production draft</summary>
          <div class="coach-audit-body">
            <div class="coach-score">
              <div id="promptCoachScore" class="coach-score-number"><span>0%</span></div>
              <div class="coach-score-copy"><strong id="promptCoachVerdict">Not analysed</strong><p>Readiness, not render-success confidence. Feature availability is evaluated separately.</p></div>
            </div>
            <div id="promptCoachStrengths" class="coach-block"></div>
            <div id="promptCoachFixes" class="coach-block"></div>
            <div id="promptCoachFeatures" class="coach-block"></div>
            <div id="promptCoachWalkthrough" class="coach-block"></div>
            <div id="promptCoachDraftBlock" class="coach-block"><h3>Coach’s production-ready draft</h3><textarea id="promptCoachRewrite" class="coach-rewrite" readonly></textarea><p id="promptCoachRewriteSafety" class="coach-provider-action" hidden></p></div>
            <button id="promptCoachUse" class="coach-use" type="button">Use prompt draft in Studio</button>
          </div>
        </details>
      </section>
      <div class="coach-followup">
        <label for="promptCoachMessage">Ask Coach anything or continue the conversation</label>
        <textarea id="promptCoachMessage" maxlength="2000" placeholder="Ask about any Studio feature, workflow, prompt decision or result…"></textarea>
        <div class="coach-followup-row"><span class="coach-disclaimer">Enter asks · Shift+Enter makes a new line</span><button id="promptCoachSend" class="coach-primary" type="button">Ask Coach</button></div>
        <p id="promptCoachQuestionsLabel" class="coach-suggestion-heading" hidden><strong>Optional questions from Coach</strong><span>Choose one, then type your answer. Nothing is sent until you press Send.</span></p>
        <div id="promptCoachSuggestions" class="coach-suggestions" role="group" aria-labelledby="promptCoachQuestionsLabel"></div>
        <p id="promptCoachProviderNote" class="coach-provider-note"></p>
        <p id="promptCoachProviderAction" class="coach-provider-action" hidden></p>
      </div>
    </div>
  </aside>
  <aside id="tutorialPanel" class="tutorial-panel" role="dialog" aria-labelledby="tutorialTitle" aria-describedby="tutorialSafety" hidden>
    <div class="tutorial-head">
      <div><strong id="tutorialTitle">Learn Dang Studio</strong><small>Interactive lessons over the real controls</small></div>
      <button id="closeTutorial" class="tutorial-close" type="button" aria-label="Close learning mode">×</button>
    </div>
    <div class="tutorial-scroll">
      <p id="tutorialSafety" class="tutorial-safety"><strong>Learning mode is safe:</strong> it cannot start video, image, voice, capture, or upscale jobs. You will always leave learning mode and press Generate yourself.</p>
      <section id="tutorialMenu">
        <p class="tutorial-intro">Choose what you want to accomplish. Each lesson uses the Studio’s live installed-feature status, points to the real controls, includes a short knowledge check, and takes about 2–4 minutes.</p>
        <div id="tutorialLessonGrid" class="tutorial-lesson-grid"></div>
        <div class="tutorial-menu-actions"><span id="tutorialCompletedCount" class="tutorial-completed"></span><button id="tutorialReset" class="small-button" type="button">Reset lesson progress</button></div>
      </section>
      <section id="tutorialLessonView" class="tutorial-step" hidden>
        <div class="tutorial-progress-copy"><span id="tutorialStepLabel" aria-live="polite"></span><span id="tutorialGoalLabel"></span></div>
        <progress id="tutorialProgress" class="tutorial-progress" min="0" max="1" value="0" aria-label="Lesson progress"></progress>
        <div id="tutorialAvailability" class="tutorial-status">Checking live workflow availability…</div>
        <h3 id="tutorialStepTitle" tabindex="-1"></h3>
        <p id="tutorialStepBody" class="tutorial-step-body"></p>
        <ul id="tutorialPoints" class="tutorial-points"></ul>
        <div id="tutorialQuiz" class="tutorial-quiz" hidden></div>
        <div id="tutorialQuizFeedback" class="tutorial-quiz-feedback" aria-live="polite"></div>
        <div class="tutorial-tools">
          <button id="tutorialShowControl" class="tutorial-primary" type="button">Show this control</button>
          <button id="tutorialApplyPractice" class="tutorial-practice" type="button">Apply safe practice setup</button>
          <button id="tutorialRestoreSetup" class="small-button" type="button" hidden>Restore previous setup</button>
          <button id="tutorialAskCoach" class="small-button" type="button">Continue with Prompt Coach</button>
        </div>
        <div id="tutorialPracticeNotice" class="tutorial-practice-note" aria-live="polite"></div>
        <div class="tutorial-footer">
          <button id="tutorialAllLessons" class="small-button" type="button">All lessons</button>
          <button id="tutorialBack" class="small-button" type="button">Back</button>
          <button id="tutorialNext" class="tutorial-primary" type="button">Next</button>
        </div>
      </section>
    </div>
  </aside>
  <button id="tutorialResume" class="tutorial-resume" type="button" aria-controls="tutorialPanel" hidden>Resume learning</button>
  <script>
    "use strict";
    const $ui = (id) => document.getElementById(id);
    const formUi = $ui("form"), promptUi = $ui("prompt"), noticeUi = $ui("notice");
    const submitUi = $ui("generate"), cancelUi = $ui("cancel");
    const statusUi = $ui("status"), dotUi = $ui("dot"), metaUi = $ui("meta"), logUi = $ui("log");
    const videoUi = $ui("video"), emptyUi = $ui("empty"), resultUi = $ui("resultActions"), downloadUi = $ui("download");
    const frameComposerNoticeUi = $ui("frameComposerNotice"), frameComposerResultUi = $ui("frameComposerResult");
    const composeFrameUi = $ui("composeFrame"), cancelFrameComposerUi = $ui("cancelFrameComposer");
    const voiceNoticeUi = $ui("voiceNotice"), voiceReadinessUi = $ui("voiceReadiness");
    const synthesizeVoicesUi = $ui("synthesizeVoices"), cancelVoicesUi = $ui("cancelVoices");
    const createUpscaleUi = $ui("createUpscale"), createAiUpscaleUi = $ui("createAiUpscale"), cancelUpscaleUi = $ui("cancelUpscale");
    const downloadUpscaleUi = $ui("downloadUpscale"), upscaleStatusUi = $ui("upscaleStatus");
    const upscaleProgressUi = $ui("upscaleProgress"), upscaleProgressBarUi = $ui("upscaleProgressBar");
    const upscaleProgressStageUi = $ui("upscaleProgressStage"), upscaleProgressValueUi = $ui("upscaleProgressValue");
    const helpPanelUi = $ui("helpPanel"), setupAdviceUi = $ui("setupAdvice");
    const promptCoachPanelUi = $ui("promptCoachPanel"), promptCoachInputUi = $ui("promptCoachInput"), promptCoachResultUi = $ui("promptCoachResult");
    const promptCoachMessageUi = $ui("promptCoachMessage"), promptCoachConversationUi = $ui("promptCoachConversation");
    const tutorialPanelUi = $ui("tutorialPanel"), tutorialMenuUi = $ui("tutorialMenu"), tutorialLessonViewUi = $ui("tutorialLessonView");
    const tutorialResumeUi = $ui("tutorialResume"), tutorialStorageKeyUi = "dangStudio.tutorial.v1";
    const previewUiUrls = new Map();
    const activeStates = new Set(["starting", "running", "cancelling"]);
    const stateLabels = { idle:"Ready", starting:"Starting…", running:"Rendering…", cancelling:"Cancelling…", succeeded:"Video ready", failed:"Render failed", cancelled:"Render cancelled" };
    const buttonLabels = { generate:"Generate video", retake:"Run retake", extend:"Extend video", a2v:"Generate from audio", ingredients:"Generate with Ingredients", union:"Generate with control", motion:"Generate motion" };
    const promptLabels = { generate:"Describe the shot", retake:"Describe the intended retake", extend:"Describe the extension", a2v:"Describe the audio-driven shot", ingredients:"Describe the generated shot", union:"Describe the controlled shot", motion:"Describe the animated shot" };
    const voiceLanguagesUi = {ar:"Arabic",da:"Danish",de:"German",el:"Greek",en:"English",es:"Spanish",fi:"Finnish",fr:"French",he:"Hebrew",hi:"Hindi",it:"Italian",ja:"Japanese",ko:"Korean",ms:"Malay",nl:"Dutch",no:"Norwegian",pl:"Polish",pt:"Portuguese",ru:"Russian",sv:"Swedish",sw:"Swahili",tr:"Turkish",zh:"Chinese"};
    const readinessByMode = {};
    const composedFrameAssignments = {start:null, end:null};
    let modeUi = "generate", pollUiBusy = false, renderUiBusy = false, videoUiUrl = null, anchorUiSerial = 0, sceneUiSerial = 0, ingredientUiSerial = 0;
    let frameComposerUiSerial = 0, frameComposerPollBusy = false, latestComposedFrameUrl = null, latestComposedFrameData = null;
    let voiceUiSerial = 0, voicePollBusy = false, voiceBackendReady = false;
    let upscalePollBusy = false, upscaleUiBusy = false, aiUpscaleReadyUi = false, currentMainJobIdUi = null;
    let activeCharacterProfileId = null, managedCharacterReferences = [], characterProfilesUi = [];
    let captureUiBusy = false, captureUiAvailable = false, latestCapturedFrameUrl = null, latestCapturedFrameData = null, latestCapturedFrameAspectUi = null;
    let latestPresentedFrameTimeUi = null, presentedFrameTrackerIdUi = null;
    let promptCoachHistoryUi = [], promptCoachBusyUi = false, promptCoachReturnFocusUi = null, promptCoachLastPromptUi = "", promptCoachRequestSerialUi = 0;
    let promptCoachTutorialContextUi = "";
    const experimentalCoachWarningsSeenUi = new Set();
    let tutorialSessionActiveUi = false, tutorialLessonIdUi = null, tutorialStepUi = 0, tutorialCompletedUi = new Set();
    let tutorialReturnFocusUi = null, tutorialSnapshotUi = null, tutorialAppliedUi = false, tutorialPendingPracticeUi = false;
    let tutorialHighlightUi = null, tutorialHighlightTimerUi = null, tutorialQuizAnsweredUi = false;

    function apiUi(path, options = {}) {
      const method = String(options.method || "GET").toUpperCase();
      const learningAllowedMutation = path === "/api/prompt-coach" || path === "/api/cancel" || path === "/api/upscale/cancel" || path === "/api/frame-composer/cancel" || path === "/api/voice/cancel";
      if (tutorialSessionActiveUi && method !== "GET" && !learningAllowedMutation) {
        return Promise.reject(new Error("Learning mode never starts or changes media jobs. Finish or close the lesson before using this action."));
      }
      return fetch(path, options).then(async (response) => {
        let data = {}; try { data = await response.json(); } catch (_) {}
        if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
        return data;
      });
    }
    const tutorialModeLabelsUi = {generate:"Generate", ingredients:"Ingredients", motion:"Motion Track", union:"Union Control", retake:"Retake", extend:"Extend", a2v:"Audio-to-Video"};
    const tutorialLessonsUi = [
      {
        id:"first-clip", title:"Create my first clip", summary:"Prompt, quality, duration, aspect, Coach review and the first safe test.", modes:["generate"],
        coachQuestion:"Teach me how to turn this into a safe first LTX-2.5 test. Explain each recommended control, but do not apply settings or render anything.",
        practice:{mode:"generate", profile:"draft", aspect:"16:9", duration:3, prompt:"One continuous 3-second 16:9 shot. A small red paper boat drifts across a rain puddle while a locked low camera watches. Warm streetlight reflects in the ripples; soft rain ambience. Preserve the boat's shape and colour. Avoid cuts, duplicates, text and sudden camera motion."},
        steps:[
          {title:"Start with Generate", body:"Generate creates a new shot from text and optional timed images. The advanced tabs solve narrower problems; they are not automatically better for a first clip.", points:["Use one short shot while learning.","Live readiness tells you whether the installed assets are eligible; submit-time preflight still decides whether a render can start."], targetId:"modeTabGenerate", mode:"generate"},
          {title:"Write a visible shot, not an aspiration", body:"Describe what the viewer can see and hear in chronological order. A useful short prompt names the fixed subject, one achievable action, one camera instruction, setting/light, sound, and a short avoid clause.", points:["Use begins → then → ends for several beats.","Do not spend tokens on words such as viral or masterpiece when a concrete visual detail would help more."], targetId:"prompt", mode:"generate"},
          {title:"Make the first attempt cheap", body:"Begin at 3 seconds with Draft or Conservative. Keep the seed fixed, solve composition and motion, then move to Quality for the final take.", points:["Longer than 5 seconds raises memory pressure and drift on this 16 GB Mac.","Change one variable per retry so you know what helped."], targetId:"generateSettingsPanel", mode:"generate"},
          {title:"Ask before spending compute", body:"Prompt Coach can review the exact prompt and current Studio setup. Its percentage is prompt completeness—not the probability that the render will be perfect.", points:["Coach separates the semantic best workflow from the currently runnable fallback.","It preserves your full prompt and stable @tags."], targetId:"promptCoachLauncher", mode:"generate"},
          {title:"You remain in control of rendering", body:"Learning mode never presses Generate. After leaving the lesson, review the prompt, profile, aspect, duration and required files. You decide when to start the job.", points:["One media compute job runs at a time.","Keep the first successful seed before changing anything."], targetId:"generate", mode:"generate"},
          {title:"Knowledge check", body:"Which setup is the safest way to learn whether a new composition works?", quiz:{options:["10 seconds at Quality with several camera moves","3 seconds at Draft or Conservative with one action","5 seconds at Max with a random seed on every retry"], correct:1, explanation:"A three-second low-cost pass isolates composition and motion. Keep the seed fixed and change one variable before spending time on Quality."}}
        ]
      },
      {
        id:"character-consistency", title:"Keep characters consistent", summary:"Character Bible, Frame Composer, true Ingredients and timed anchors.", modes:["generate","ingredients"],
        coachQuestion:"Teach me which identity layer this prompt needs: Character Bible, Frame Composer, Ingredients, or timed anchors. Explain the tradeoffs and do not render.",
        practice:{mode:"ingredients", fallbackMode:"generate", profile:"ingredients5s", aspect:"profile", duration:5, prompt:"@subject_a and @subject_b cross a rain-slick observation deck in one continuous shot. @subject_a checks the route while @subject_b notices a distant signal and points. Preserve both supplied identities, wardrobe, scale and screen position; cool dawn light, locked eye-level camera. Avoid character swapping, duplicate subjects, extra limbs and cuts."},
        steps:[
          {title:"Save the reusable identity description", body:"Character Bible stores a character's text and managed reference views locally across restarts. It is excellent for repeatable setup, but it is not itself the true multi-reference identity adapter.", points:["Keep appearance, wardrobe, continuity and avoid rules stable.","Voice assignments remain separate from visual identity."], targetId:"characterBiblePanel", mode:"generate"},
          {title:"Compose one identity-first still", body:"Frame Composer combines up to four active photos into one still. Assign that still as a Start, End or Interior anchor; it does not become a global video identity embedding.", points:["Mention each composition ingredient tag once.","Use the target video's matched canvas before generation."], targetId:"frameComposerPanel", mode:"generate"},
          {title:"Use Ingredients for true multi-reference identity", body:"Ingredients is the strongest lane when several referenced characters, objects or a location must remain distinguishable throughout one shot.", points:["Use one clean image and stable lowercase tag such as @subject_a per reference.","Its local workflow uses a fixed 5.04-second trained bucket and cannot share a render with Motion Track."], targetId:"ingredientsPanel", mode:"ingredients"},
          {title:"Use anchors for moments", body:"Start, End and Interior anchors guide particular compositions at particular times. They are useful for a visual bridge, but they do not provide a guaranteed full-clip identity lock.", points:["Start + one Interior + End is the practical complex-shot setup on this Mac.","When a Start image owns appearance, prompt what changes instead of redundantly describing every visible detail."], targetId:"boundaryFramesPanel", mode:"generate"},
          {title:"Knowledge check", body:"Three clean reference images must preserve two named characters throughout one shot. Which workflow is the semantic match?", quiz:{options:["Interior anchors only","Ingredients with stable @tags","Fast 720p after Generate"], correct:1, explanation:"Ingredients is the true multi-reference identity lane. Anchors control moments, while upscaling cannot repair identity drift."}}
        ]
      },
      {
        id:"motion-control", title:"Control motion or composition", summary:"Choose Motion Track, Union Control or a few timed anchors without conflicting guidance.", modes:["motion","union","generate"],
        coachQuestion:"Walk me through whether this idea needs Motion Track, Union Control, or Start/Interior/End anchors. Explain what owns movement and what the prompt should omit.",
        practice:{mode:"motion", fallbackMode:"generate", profile:"control3s", aspect:"16:9", duration:3, prompt:"A red paper airplane follows the supplied Motion Track through a sunlit studio while a locked wide camera watches. The airplane banks naturally with the curve; preserve its shape and scale. Let the path own the trajectory. Avoid cuts, duplicates and competing camera movement."},
        steps:[
          {title:"Establish the scene first", body:"Motion Track needs a Start image. The image owns the opening scene; the path owns trajectory; the prompt should describe subject, style and physical response without fighting either control.", points:["Begin with three or four clear path points.","Keep the path physically plausible for the subject."], targetId:"boundaryFramesPanel", mode:"motion"},
          {title:"Use Motion Track for an exact trajectory", body:"Choose Motion Track when a subject or point must travel along an explicit spatial path. Normalized JSON defines where it moves across the frame.", points:["Do not repeat a contradictory route in prose.","Ingredients and Motion Track cannot run in the same render."], targetId:"motionPanel", mode:"motion"},
          {title:"Use Union for timed whole-frame structure", body:"Union Control follows a Canny, depth or pose guide video. A static comic strip is not a timed motion guide; use its panels as anchors or first convert it into a suitable guide video.", points:["Do not contradict the guide's geometry or camera in the prompt.","Union requires its separate adapter and submit-time preflight."], targetId:"unionPanel", mode:"union"},
          {title:"Use anchors for a few exact poses", body:"When only a few compositions matter, Start, Interior and End anchors are simpler than a dense structural guide. They influence named moments rather than every frame.", points:["Prefer a small number of strong anchors.","More active images increase attention memory."], targetId:"timelineAnchorsPanel", mode:"generate"},
          {title:"Knowledge check", body:"A ball must follow an exact curved path across the frame. Which control should own that movement?", quiz:{options:["Motion Track","Ingredients","Fast 720p"], correct:0, explanation:"Motion Track owns exact spatial trajectories. Ingredients owns identity, and 720p changes delivery size only."}}
        ]
      },
      {
        id:"continue-repair", title:"Continue or repair a clip", summary:"Understand visual hand-off, native Extend and interval Retake.", modes:["generate","extend","retake"],
        coachQuestion:"Teach me whether this edit should use frame capture plus Generate, native Extend, or Retake. Explain continuity limits and current availability without rendering.",
        practice:{mode:"extend", fallbackMode:"generate", profile:"draft", aspect:"16:9", duration:3, prompt:"Continue immediately from the supplied captured Start frame. The subject completes the same gentle turn and settles into a still final pose while the camera remains locked. Preserve identity, scale, lighting and background geometry. Avoid a jump cut, duplicated subject or sudden motion change."},
        steps:[
          {title:"Capture creates a visual hand-off", body:"Pause a completed video on the last good frame and capture it. Use that PNG as the next Generate clip's Start frame.", points:["It does not preserve hidden velocity, latent state or audio.","Trim the rejected tail and stitch clips separately."], targetId:"capturePanel", mode:"generate"},
          {title:"Extend is native continuation", body:"Extend generates new material directly before or after an existing clip and inherits its dimensions. Describe only the new interval and boundary continuity.", points:["Begin with +3 seconds.","It requires the separate Q8/HQ editing lane."], targetId:"extendPanel", mode:"extend"},
          {title:"Retake repairs an interval", body:"Retake replaces a selected interval inside an existing clip. Describe only the replacement and how it should join both boundaries.", points:["Preserve source audio when it must survive unchanged.","It requires the separate Q8/HQ editing lane."], targetId:"retakePanel", mode:"retake"},
          {title:"Knowledge check", body:"The middle second of an existing video is wrong, but the beginning and end are good. What is the intended workflow?", quiz:{options:["Capture current frame","Retake","Ingredients"], correct:1, explanation:"Retake is designed to replace an interval while respecting both boundaries. Capture starts a separate visual continuation."}}
        ]
      },
      {
        id:"audio-dialogue", title:"Add speech or soundtrack", summary:"Audio-driven motion, @speaker attribution and dependable separate character voices.", modes:["a2v","generate"],
        coachQuestion:"Teach me whether this exact prompt needs Audio-to-Video, native generated dialogue, or the Character Voices pass. Explain who owns timing and how speaker separation should work. Do not render.",
        practice:{mode:"a2v", fallbackMode:"generate", profile:"clip3s", aspect:"16:9", duration:3, prompt:"A close portrait of @maya performing to the supplied audio on a softly lit stage. The audio owns timing and visible rhythm; the camera stays locked while her expression and subtle head movement follow the performance. Preserve @maya's identity and wardrobe. Avoid cuts, extra voices and background performers."},
        steps:[
          {title:"Let audio own timing", body:"Audio-to-Video is for movement, rhythm or performance that must follow a supplied audio segment. The audio remains unchanged; prompt the visible subject, scene, style and camera.", points:["Trim the source audio precisely.","Ordinary Generate should not be assumed to follow exact beats or phonemes."], targetId:"a2vPanel", mode:"a2v"},
          {title:"Separate dependable character voices", body:"Character Voices creates one isolated local TTS call per assigned @character and can mux timed cues into a separate MP4. It is the dependable layer when speakers must not intermix.", points:["Use only your own or explicitly authorized voice reference.","Keep cues sentence-sized and give each a start time."], targetId:"characterVoicesPanel", mode:"generate"},
          {title:"Use explicit speaker tags", body:"Assign every line to exactly one stable @speaker. Tags improve attribution, but native generated dialogue can still swap or blend voices.", points:["Visual identity and voice identity are separate systems.","Test the picture first, then create the controlled voice pass."], targetId:"prompt", mode:"generate"},
          {title:"Knowledge check", body:"A supplied song must determine the performer's movement timing. Which workflow is the semantic match?", quiz:{options:["Audio-to-Video","Character Bible","Retake"], correct:0, explanation:"Audio-to-Video lets the supplied audio own timing. Character Voices is a separate controlled speech pass."}}
        ]
      },
      {
        id:"finish-share", title:"Finish and upscale a video", summary:"Choose AI 1080p or fast 720p honestly, capture continuity and keep originals.", modes:["generate"],
        coachQuestion:"Teach me a practical final-pass and upscaling workflow for this prompt on a 16 GB Mac. Compare AI Quality Upscale with Fast 720p and explain what neither can repair.",
        practice:{mode:"generate", profile:"quality", aspect:"16:9", duration:3, prompt:"One continuous 3-second 16:9 cinematic shot. A ceramic cup releases a thin ribbon of steam beside a rain-streaked window while a slow subtle push-in ends on the cup. Soft overcast daylight, realistic reflections and quiet room tone. Preserve cup geometry. Avoid cuts, text, duplicate objects and sudden camera motion."},
        steps:[
          {title:"Move to Quality only after the shot works", body:"Draft and Conservative are iteration profiles. Quality is the preferred final render after composition, identity and motion are already behaving.", points:["Keep the successful seed.","Change one variable at a time between comparison renders."], targetId:"generateSettingsPanel", mode:"generate"},
          {title:"Choose the delivery shape before rendering", body:"Use 16:9 for landscape/YouTube, 9:16 for phone video, 1:1 for square, 4:3 for classic landscape and 3:4 for portrait. Named ratios choose the nearest safe LTX grid.", points:["Cropping later can remove important action.","Control workflows use their own lower-resolution profiles."], targetId:"generateAspect", mode:"generate"},
          {title:"Use AI 1080p for the final quality pass", body:"AI Quality Upscale uses local Core ML Real‑ESRGAN to reconstruct edges and texture, preserves audio and frame rate, and keeps the original.", points:["Recommended after you have accepted the motion and identities.","Inspect fine texture for shimmer; AI cannot repair anatomy, identity drift or composition."], targetId:"upscalePanel", mode:"generate"},
          {title:"Use Fast 720p when fidelity or speed matters", body:"Fast 720p writes a separate aspect-preserving MP4 with Lanczos resizing, mild sharpening and copied audio.", points:["It changes delivery size without inventing detail.","It usually finishes in seconds and keeps the original."], targetId:"upscalePanel", mode:"generate"},
          {title:"Keep successful hand-off frames", body:"Download good versions and capture a frame when you want the next clip to begin from the same visible composition. Stitch selected clips in an editor.", points:["Capture does not trim the source video.","The original render remains unchanged."], targetId:"capturePanel", mode:"generate"},
          {title:"Knowledge check", body:"When should you run AI Quality Upscale?", quiz:{options:["Before checking whether identities and motion are correct","After accepting the clip, to reconstruct final edge and texture detail","To repair extra limbs or a wrong composition"], correct:1, explanation:"Upscaling is a finishing pass. AI can enhance detail, but it cannot correct generation mistakes."}}
        ]
      }
    ];
    function tutorialLessonByIdUi(id) { return tutorialLessonsUi.find((lesson) => lesson.id === id) || null; }
    function loadTutorialProgressUi() {
      try {
        const stored = JSON.parse(localStorage.getItem(tutorialStorageKeyUi) || "{}");
        const lesson = tutorialLessonByIdUi(stored.lessonId);
        tutorialLessonIdUi = lesson ? lesson.id : null;
        tutorialStepUi = lesson && Number.isSafeInteger(stored.step) ? Math.max(0, Math.min(lesson.steps.length - 1, stored.step)) : 0;
        const allowed = new Set(tutorialLessonsUi.map((item) => item.id));
        tutorialCompletedUi = new Set(Array.isArray(stored.completed) ? stored.completed.filter((id) => allowed.has(id)) : []);
      } catch (_) { tutorialLessonIdUi = null; tutorialStepUi = 0; tutorialCompletedUi = new Set(); }
    }
    function saveTutorialProgressUi() {
      try { localStorage.setItem(tutorialStorageKeyUi, JSON.stringify({lessonId:tutorialLessonIdUi, step:tutorialStepUi, completed:[...tutorialCompletedUi]})); } catch (_) {}
    }
    function tutorialLessonStatusUi(lesson) {
      const known = lesson.modes.map((mode) => readinessByMode[mode]).filter(Boolean);
      if (!known.length) return {label:"Checking", className:"mixed"};
      const blocked = known.filter((item) => item.blocked).length;
      if (blocked === known.length) return {label:"Learn only", className:"blocked"};
      if (blocked) return {label:"Mixed availability", className:"mixed"};
      return {label:"Ready to practise", className:""};
    }
    function tutorialAvailabilityCopyUi(lesson) {
      const parts = lesson.modes.map((mode) => {
        const item = readinessByMode[mode];
        return `${tutorialModeLabelsUi[mode]}: ${!item ? "checking" : item.blocked ? "blocked / learn only" : "installed and eligible"}`;
      });
      const blocked = lesson.modes.map((mode) => ({mode, item:readinessByMode[mode]})).find(({item}) => item?.blocked);
      const reason = blocked?.item?.note ? ` ${String(blocked.item.note).slice(0, 240)}` : "";
      return `This Mac — ${parts.join(" · ")}.${reason} Installed eligibility is not a render guarantee; full receipt, source and Metal preflight runs on submit.`;
    }
    function renderTutorialMenuUi() {
      tutorialMenuUi.hidden = false; tutorialLessonViewUi.hidden = true; tutorialPendingPracticeUi = false;
      const grid = $ui("tutorialLessonGrid"); grid.replaceChildren();
      for (const lesson of tutorialLessonsUi) {
        const status = tutorialLessonStatusUi(lesson), button = document.createElement("button");
        button.type = "button"; button.className = "tutorial-lesson"; button.dataset.lessonId = lesson.id;
        const badge = document.createElement("span"); badge.className = `tutorial-lesson-status ${status.className}`.trim(); badge.textContent = status.label;
        const title = document.createElement("strong"); title.textContent = `${tutorialCompletedUi.has(lesson.id) ? "✓ " : ""}${lesson.title}`;
        const summary = document.createElement("small"); summary.textContent = lesson.summary;
        button.append(badge, title, summary); button.addEventListener("click", () => startTutorialLessonUi(lesson.id)); grid.appendChild(button);
      }
      $ui("tutorialCompletedCount").textContent = `${tutorialCompletedUi.size} of ${tutorialLessonsUi.length} lessons completed`;
      requestAnimationFrame(() => grid.querySelector("button")?.focus());
    }
    function renderTutorialStepUi() {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi);
      if (!lesson) { tutorialLessonIdUi = null; renderTutorialMenuUi(); return; }
      tutorialMenuUi.hidden = true; tutorialLessonViewUi.hidden = false;
      tutorialStepUi = Math.max(0, Math.min(lesson.steps.length - 1, tutorialStepUi));
      const step = lesson.steps[tutorialStepUi], progress = $ui("tutorialProgress");
      $ui("tutorialStepLabel").textContent = `Step ${tutorialStepUi + 1} of ${lesson.steps.length}`;
      $ui("tutorialGoalLabel").textContent = lesson.title;
      progress.max = lesson.steps.length; progress.value = tutorialStepUi + 1;
      $ui("tutorialAvailability").textContent = tutorialAvailabilityCopyUi(lesson);
      $ui("tutorialStepTitle").textContent = step.title; $ui("tutorialStepBody").textContent = step.body;
      const points = $ui("tutorialPoints"); points.replaceChildren(); points.hidden = !step.points?.length;
      for (const copy of step.points || []) { const item = document.createElement("li"); item.textContent = copy; points.appendChild(item); }
      const quiz = $ui("tutorialQuiz"); quiz.replaceChildren(); quiz.hidden = !step.quiz; tutorialQuizAnsweredUi = false;
      $ui("tutorialQuizFeedback").textContent = "";
      if (step.quiz) {
        step.quiz.options.forEach((option, index) => {
          const button = document.createElement("button"); button.type = "button"; button.textContent = option;
          button.addEventListener("click", () => answerTutorialQuizUi(index)); quiz.appendChild(button);
        });
      }
      const show = $ui("tutorialShowControl"); show.hidden = !step.targetId;
      const apply = $ui("tutorialApplyPractice"); apply.textContent = "Apply safe practice setup"; apply.hidden = !lesson.practice;
      $ui("tutorialRestoreSetup").hidden = !tutorialSnapshotUi;
      $ui("tutorialAskCoach").hidden = false; $ui("tutorialPracticeNotice").textContent = ""; tutorialPendingPracticeUi = false;
      $ui("tutorialBack").disabled = tutorialStepUi === 0; $ui("tutorialNext").disabled = Boolean(step.quiz); $ui("tutorialNext").textContent = tutorialStepUi === lesson.steps.length - 1 ? "Finish lesson" : "Next";
      saveTutorialProgressUi(); requestAnimationFrame(() => $ui("tutorialStepTitle").focus());
    }
    function renderTutorialUi() { if (tutorialLessonIdUi) renderTutorialStepUi(); else renderTutorialMenuUi(); }
    function startTutorialLessonUi(id) {
      if (!tutorialLessonByIdUi(id)) return;
      tutorialLessonIdUi = id; tutorialStepUi = 0; tutorialQuizAnsweredUi = false; saveTutorialProgressUi(); renderTutorialStepUi();
    }
    function tutorialCaptureSnapshotUi() {
      if (tutorialSnapshotUi) return;
      tutorialSnapshotUi = {
        mode:modeUi, prompt:promptUi.value,
        generateProfile:document.querySelector('input[name="profile"]:checked')?.value || "quality",
        generateAspect:$ui("generateAspect").value, generateDuration:$ui("generateDuration").value,
        a2vProfile:$ui("a2vProfile").value, a2vAspect:$ui("a2vAspect").value, a2vDuration:$ui("a2vDuration").value,
        ingredientsProfile:$ui("ingredientsProfile").value, ingredientsAspect:$ui("ingredientsAspect").value,
        motionProfile:$ui("motionProfile").value, motionAspect:$ui("motionAspect").value, motionDuration:$ui("motionDuration").value,
        details:Array.from(document.querySelectorAll("details")).map((element) => ({element, open:element.open}))
      };
    }
    function tutorialRestoreSnapshotUi() {
      const snapshot = tutorialSnapshotUi; if (!snapshot) return;
      promptUi.value = snapshot.prompt; promptUi.dispatchEvent(new Event("input", {bubbles:true}));
      const profile = document.querySelector(`input[name="profile"][value="${snapshot.generateProfile}"]`); if (profile) { profile.checked = true; profile.dispatchEvent(new Event("change", {bubbles:true})); }
      for (const [id, value] of Object.entries({generateAspect:snapshot.generateAspect, generateDuration:snapshot.generateDuration, a2vProfile:snapshot.a2vProfile, a2vAspect:snapshot.a2vAspect, a2vDuration:snapshot.a2vDuration, ingredientsProfile:snapshot.ingredientsProfile, ingredientsAspect:snapshot.ingredientsAspect, motionProfile:snapshot.motionProfile, motionAspect:snapshot.motionAspect, motionDuration:snapshot.motionDuration})) {
        const element = $ui(id); if (element) { element.value = value; element.dispatchEvent(new Event(element.type === "range" ? "input" : "change", {bubbles:true})); }
      }
      selectModeUi(snapshot.mode); for (const item of snapshot.details) item.element.open = item.open;
      tutorialSnapshotUi = null; tutorialAppliedUi = false; $ui("tutorialRestoreSetup").hidden = true; $ui("tutorialPracticeNotice").textContent = "Your pre-tutorial prompt, settings and selected workflow were restored.";
    }
    function tutorialSetSessionUi(active) {
      tutorialSessionActiveUi = active; document.body.classList.toggle("tutorial-active", active);
      if (active) { submitUi.disabled = true; submitUi.textContent = "Finish learning mode before rendering"; }
      else { submitUi.textContent = buttonLabels[modeUi]; displayReadinessUi(); }
    }
    function openTutorialUi() {
      if (!promptCoachPanelUi.hidden) closePromptCoachUi();
      if (!tutorialSessionActiveUi) { tutorialReturnFocusUi = document.activeElement; tutorialSetSessionUi(true); tutorialAppliedUi = false; }
      tutorialPanelUi.hidden = false; tutorialResumeUi.hidden = true; $ui("openTutorial").setAttribute("aria-expanded", "true"); renderTutorialUi();
    }
    function pauseTutorialUi() {
      if (!tutorialSessionActiveUi) return;
      tutorialPanelUi.hidden = true; tutorialResumeUi.hidden = false; $ui("openTutorial").setAttribute("aria-expanded", "false");
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi); tutorialResumeUi.textContent = lesson ? `Resume · ${lesson.title}` : "Resume learning";
    }
    function closeTutorialUi({restore = true} = {}) {
      if (restore && !tutorialAppliedUi) tutorialRestoreSnapshotUi();
      else if (tutorialAppliedUi) tutorialSnapshotUi = null;
      tutorialPanelUi.hidden = true; tutorialResumeUi.hidden = true; $ui("openTutorial").setAttribute("aria-expanded", "false"); clearTutorialHighlightUi(); tutorialSetSessionUi(false);
      if (tutorialReturnFocusUi?.focus) tutorialReturnFocusUi.focus(); tutorialReturnFocusUi = null;
    }
    function clearTutorialHighlightUi() {
      if (tutorialHighlightTimerUi) clearTimeout(tutorialHighlightTimerUi); tutorialHighlightTimerUi = null;
      if (tutorialHighlightUi) tutorialHighlightUi.classList.remove("tutorial-highlight"); tutorialHighlightUi = null;
    }
    function revealTutorialTargetUi() {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi), step = lesson?.steps[tutorialStepUi]; if (!step?.targetId) return;
      tutorialCaptureSnapshotUi(); if (step.mode) selectModeUi(step.mode);
      const target = $ui(step.targetId); if (!target) { $ui("tutorialPracticeNotice").textContent = "That control is not available in this build."; return; }
      let ancestor = target; while (ancestor) { if (ancestor.tagName === "DETAILS") ancestor.open = true; ancestor = ancestor.parentElement; }
      pauseTutorialUi(); requestAnimationFrame(() => {
        clearTutorialHighlightUi(); tutorialHighlightUi = target; target.classList.add("tutorial-highlight");
        const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"; target.scrollIntoView({behavior, block:"center"});
        if (target.matches("button,input,textarea,select,summary,a[href]")) target.focus({preventScroll:true});
        tutorialHighlightTimerUi = setTimeout(clearTutorialHighlightUi, 5200);
      });
    }
    function setTutorialPracticeFieldsUi(setup) {
      selectModeUi(setup.mode); promptUi.value = setup.prompt; promptUi.dispatchEvent(new Event("input", {bubbles:true}));
      if (setup.mode === "generate") {
        const profile = document.querySelector(`input[name="profile"][value="${setup.profile}"]`); if (profile) { profile.checked = true; profile.dispatchEvent(new Event("change", {bubbles:true})); }
        $ui("generateAspect").value = setup.aspect; $ui("generateAspect").dispatchEvent(new Event("change", {bubbles:true}));
        $ui("generateDuration").value = String(setup.duration); $ui("generateDuration").dispatchEvent(new Event("input", {bubbles:true}));
      } else if (setup.mode === "a2v") {
        $ui("a2vProfile").value = setup.profile; $ui("a2vAspect").value = setup.aspect; $ui("a2vDuration").value = String(setup.duration); $ui("a2vDuration").dispatchEvent(new Event("input", {bubbles:true}));
      } else if (setup.mode === "ingredients") {
        $ui("ingredientsProfile").value = setup.profile; $ui("ingredientsAspect").value = setup.aspect || "profile";
      } else if (setup.mode === "motion") {
        $ui("motionProfile").value = setup.profile; $ui("motionAspect").value = setup.aspect; $ui("motionDuration").value = String(setup.duration); $ui("motionDuration").dispatchEvent(new Event("input", {bubbles:true}));
      }
    }
    function applyTutorialPracticeUi() {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi); if (!lesson?.practice) return;
      if (promptUi.value.trim() && promptUi.value !== lesson.practice.prompt && !tutorialPendingPracticeUi) {
        tutorialPendingPracticeUi = true; $ui("tutorialApplyPractice").textContent = "Confirm replace with practice setup";
        $ui("tutorialPracticeNotice").textContent = "This replaces the visible prompt and a few settings only. Your current setup is kept in memory and can be restored; no job will start. Click again to confirm."; return;
      }
      tutorialCaptureSnapshotUi(); const primaryBlocked = Boolean(readinessByMode[lesson.practice.mode]?.blocked);
      const setup = primaryBlocked && lesson.practice.fallbackMode ? {...lesson.practice, mode:lesson.practice.fallbackMode, profile:"draft", aspect:"16:9", duration:3} : lesson.practice;
      setTutorialPracticeFieldsUi(setup); tutorialAppliedUi = true; tutorialPendingPracticeUi = false; $ui("tutorialRestoreSetup").hidden = false; $ui("tutorialApplyPractice").textContent = "Practice setup applied";
      $ui("tutorialPracticeNotice").textContent = primaryBlocked && setup.mode !== lesson.practice.mode ? `The intended ${tutorialModeLabelsUi[lesson.practice.mode]} lane is blocked, so Studio loaded its safe Generate learning fallback. Required reference media is never fabricated. No render started.` : "Practice prompt and safe local settings are loaded. Required images, audio, paths or source video are not fabricated. No render started; learning mode still blocks media actions.";
    }
    function answerTutorialQuizUi(index) {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi), quiz = lesson?.steps[tutorialStepUi]?.quiz; if (!quiz) return;
      const buttons = Array.from($ui("tutorialQuiz").querySelectorAll("button")), isCorrect = index === quiz.correct;
      if (!isCorrect) {
        buttons[index].disabled = true; buttons[index].classList.add("incorrect");
        $ui("tutorialQuizFeedback").textContent = `Not quite. Try again. ${quiz.explanation}`;
        return;
      }
      tutorialQuizAnsweredUi = true;
      buttons.forEach((button, itemIndex) => { button.disabled = true; if (itemIndex === quiz.correct) button.classList.add("correct"); });
      $ui("tutorialQuizFeedback").textContent = `Correct. ${quiz.explanation}`; $ui("tutorialNext").disabled = false;
    }
    function tutorialNextUi() {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi); if (!lesson) return;
      if (lesson.steps[tutorialStepUi]?.quiz && !tutorialQuizAnsweredUi) return;
      if (tutorialStepUi >= lesson.steps.length - 1) {
        tutorialCompletedUi.add(lesson.id); if (!tutorialAppliedUi) tutorialRestoreSnapshotUi(); tutorialLessonIdUi = null; tutorialStepUi = 0; saveTutorialProgressUi(); renderTutorialMenuUi(); return;
      }
      tutorialStepUi += 1; tutorialQuizAnsweredUi = false; renderTutorialStepUi();
    }
    function tutorialBackUi() { if (tutorialStepUi > 0) { tutorialStepUi -= 1; tutorialQuizAnsweredUi = false; renderTutorialStepUi(); } }
    function tutorialAllLessonsUi() { if (!tutorialAppliedUi) tutorialRestoreSnapshotUi(); tutorialLessonIdUi = null; tutorialStepUi = 0; saveTutorialProgressUi(); renderTutorialMenuUi(); }
    function tutorialCoachQuestionUi(lesson, coachPrompt) {
      const base = String(lesson?.coachQuestion || "Teach me this Dang Studio workflow.").trim();
      if (lesson?.id !== "audio-dialogue") return base;
      const tags = Array.from(new Set(String(coachPrompt || "").match(/@[A-Za-z][A-Za-z0-9_]*/g) || []));
      if (!tags.length) return `${base} No stable character tag is present, so explain whether I need to add one; do not invent a tag or speaker identity for me.`;
      const joined = tags.length === 1 ? tags[0] : `${tags.slice(0, -1).join(", ")} and ${tags.at(-1)}`;
      return `${base} The existing stable ${tags.length === 1 ? "tag is" : "tags are"} ${joined}; preserve ${tags.length === 1 ? "it" : "them"} exactly and identify which tagged character, if any, actually speaks.`;
    }
    function tutorialAskCoachUi() {
      const lesson = tutorialLessonByIdUi(tutorialLessonIdUi); if (!lesson) return;
      const coachPrompt = promptUi.value.trim() || lesson.practice?.prompt || "Teach me this Dang Studio workflow.";
      const coachQuestion = tutorialCoachQuestionUi(lesson, coachPrompt);
      const tutorialContext = `${lesson.id}\u0000${coachPrompt}\u0000${coachQuestion}`;
      const visibleCoachTurns = promptCoachConversationUi.querySelectorAll(".coach-chat").length;
      const hasPriorAnalysis = promptCoachHistoryUi.length > 0 || !promptCoachResultUi.hidden || visibleCoachTurns > 1;
      if (promptCoachInputUi.value !== coachPrompt || (hasPriorAnalysis && promptCoachTutorialContextUi !== tutorialContext)) {
        resetPromptCoachHistoryUi("The learning tutorial prepared a new question, so earlier Coach assumptions were cleared.");
      }
      promptCoachTutorialContextUi = tutorialContext;
      promptCoachInputUi.value = coachPrompt; promptCoachInputUi.dispatchEvent(new Event("input", {bubbles:true}));
      promptCoachLastPromptUi = coachPrompt; promptCoachMessageUi.value = coachQuestion;
      pauseTutorialUi(); openPromptCoachUi(); $ui("promptCoachNotice").textContent = "Tutorial question prepared. Review it, then press Send when you choose."; promptCoachMessageUi.focus();
    }
    function showSetupAdviceUi(items) {
      setupAdviceUi.replaceChildren();
      const heading = document.createElement("strong"); heading.textContent = "Suggested setup"; setupAdviceUi.appendChild(heading);
      const list = document.createElement("ul");
      for (const item of items) { const row = document.createElement("li"); row.textContent = item; list.appendChild(row); }
      setupAdviceUi.appendChild(list);
    }
    function availabilitySuffixUi(mode) {
      const item = readinessByMode[mode];
      return item?.blocked ? " This workflow is currently locked until its verified Q8 assets are installed." : " This workflow is available.";
    }
    function recommendSetupUi() {
      const raw = promptUi.value.trim(), text = raw.toLowerCase();
      if (!raw) {
        setupAdviceUi.textContent = "Enter your idea in “Describe the shot” first, then click Recommend setup.";
        promptUi.focus();
        return;
      }
      const suggestions = [];
      const durationMatch = text.match(/\b(10|[1-9])\s*(?:s|sec|secs|second|seconds)\b/);
      const duration = durationMatch ? Number(durationMatch[1]) : 3;
      const testing = /\b(test|draft|preview|rough|try|experiment)\b/.test(text);
      suggestions.push(`Quality: ${testing ? "Draft for this test; switch to Quality after composition works." : "Quality for the final take; use Draft first if camera or blocking is uncertain."}`);
      suggestions.push(`Duration: ${duration}s${duration > 5 ? " — high drift/memory risk; split it into short takes, then continue or stitch." : duration > 3 ? " — workable, but 3s is safer for complex movement or identity." : " — a good iteration length on this Mac."}`);
      let aspect = "Profile default landscape";
      if (/\b(9:16|vertical|reel|reels|shorts|tiktok|phone screen)\b/.test(text)) aspect = "9:16 vertical";
      else if (/\b(1:1|square|square post)\b/.test(text)) aspect = "1:1 square";
      else if (/\b(3:4|portrait photo|portrait post)\b/.test(text)) aspect = "3:4 portrait";
      else if (/\b(4:3|academy|classic television|classic frame)\b/.test(text)) aspect = "4:3 classic landscape";
      else if (/\b(16:9|widescreen|youtube|landscape|cinematic frame)\b/.test(text)) aspect = "16:9 landscape";
      suggestions.push(`Aspect ratio: ${aspect}.`);

      const tags = raw.match(/@[A-Za-z][A-Za-z0-9_]*/g) || [];
      const identity = tags.length || /\b(same character|character consistency|consistent face|identity|wardrobe|reference image|reference photo|same person)\b/.test(text);
      const dialogue = /["“”]|\b(dialogue|says|speaks|voiceover|voice-over|narrat|lip.?sync|talking)\b/.test(text);
      const continuation = /\b(continue|continuation|extend|next shot|next clip|after this|picks? up)\b/.test(text);
      const retake = /\b(retake|replace an interval|replace the|fix the (?:first|middle|last)|regenerate .*seconds?)\b/.test(text);
      const audioDriven = /\b(audio-driven|soundtrack|music video|to the beat|song|singing|dance to|lip.?sync)\b/.test(text);
      const structural = /\b(canny|depth guide|pose guide|control video|storyboard|comic strip|match the structure|same composition)\b/.test(text);
      const tracked = /\b(motion track|trajectory|follow a path|moves? from .+ to|camera path)\b/.test(text);
      const timedAnchors = /\b(exact opening|opening frame|start frame|exact ending|final frame|end frame|key composition)\b/.test(text);
      const multiBeat = /\b(then|afterward|finally|next,|at the end)\b/.test(text);

      if (retake) suggestions.push(`Workflow: Retake for the unwanted interval.${availabilitySuffixUi("retake")}`);
      else if (continuation) suggestions.push(`Workflow: capture the last good displayed frame for a visual hand-off now; use native Extend when available.${availabilitySuffixUi("extend")}`);
      else if (audioDriven) suggestions.push(`Workflow: Audio-to-Video for timing driven by music or speech.${availabilitySuffixUi("a2v")}`);
      else suggestions.push("Workflow: Generate for a new shot.");

      if (identity) {
        suggestions.push("Identity: load a saved Character Bible profile and use Frame Composer or clean Start/End anchors now.");
        if (tags.length > 1 || /\b(multiple characters|several characters|multi-reference)\b/.test(text)) suggestions.push(`True multi-reference identity: Ingredients is the strongest match.${availabilitySuffixUi("ingredients")}`);
      }
      if (dialogue) suggestions.push("Dialogue: keep every line attached to an explicit @character and use Character Voices for reliable speaker separation; native generated voices can swap.");
      if (structural) suggestions.push(`Structure: use Start/Interior/End panels for a static storyboard; use Union Control only after making a timed guide video.${availabilitySuffixUi("union")}`);
      if (tracked) suggestions.push(`Trajectory: Motion Track is appropriate and requires a Start image plus normalized paths.${availabilitySuffixUi("motion")}`);
      if (timedAnchors) suggestions.push("Composition: upload the matching Start or End frame; use one Interior anchor only where the middle composition is essential.");
      if (multiBeat) suggestions.push("Timing: add Scene Beats for the chronological actions, but keep them within one continuous shot without cuts.");
      if (raw.split(/\s+/).length < 12) suggestions.push("Prompt detail: add subject identity, chronological action, camera move, setting, light, sound and a short avoid clause.");
      suggestions.push("Low-resolution safeguard: keep important faces and hands in close or medium framing, especially in 384×224/256 control modes.");
      showSetupAdviceUi(suggestions);
    }
    function promptCoachAvailabilityUi(mode) {
      const item = readinessByMode[mode];
      if (!item) return "readiness is still being checked";
      return item.blocked ? "currently locked until the verified Q8 assets are installed" : "ready now";
    }
    function promptCoachSignalsUi(raw) {
      const text = raw.toLowerCase(), words = raw.trim().split(/\s+/).filter(Boolean);
      const tags = [...new Set(raw.match(/@[A-Za-z][A-Za-z0-9_]*/g) || [])];
      const durationMatch = text.match(/\b(10|[1-9])\s*(?:s|sec|secs|second|seconds)\b/);
      const duration = durationMatch ? Number(durationMatch[1]) : 3;
      let aspect = "profile-default landscape";
      if (/\b(9:16|vertical|reel|reels|shorts|tiktok|phone screen)\b/.test(text)) aspect = "9:16 vertical";
      else if (/\b(1:1|square|square post)\b/.test(text)) aspect = "1:1 square";
      else if (/\b(3:4|portrait photo|portrait post)\b/.test(text)) aspect = "3:4 portrait";
      else if (/\b(4:3|academy|classic television|classic frame)\b/.test(text)) aspect = "4:3 classic landscape";
      else if (/\b(16:9|widescreen|youtube|landscape|cinematic frame)\b/.test(text)) aspect = "16:9 landscape";
      const identityDetail = Boolean(tags.length) || /\b(wearing|wardrobe|hair|eyes|face|facial|age|build|scar|freckles|tattoo|consistent|unchanged|same character|same person|identity|colou?r(?:ed)? (?:shirt|jacket|dress|coat|suit))\b/.test(text);
      const subjectBasic = identityDetail || words.length >= 4;
      const action = /\b(walks?|runs?|turns?|looks?|raises?|lowers?|opens?|closes?|enters?|leaves?|speaks?|says?|performs?|performing|smiles?|cries?|drives?|flies?|falls?|jumps?|dances?|fights?|reaches?|holds?|moves?|crosses?|approaches?|pulls?|pushes?|sits?|stands?|begins?|ends?)\b/.test(text);
      const cameraTerms = text.match(/\b(close[- ]?up|medium shot|wide shot|full shot|macro|overhead|low angle|high angle|pov|tracking|dolly|push[- ]?in|pull[- ]?back|pan|tilt|crane|orbit|handheld|static camera|locked camera|camera (?:remains?|stays?) (?:locked|static|stationary|fixed)|rack focus|shallow depth|lens)\b/g) || [];
      const camera = cameraTerms.length > 0;
      const setting = /\b(interior|exterior|room|street|alley|forest|beach|desert|mountain|city|village|office|home|kitchen|bedroom|studio|stage|station|airport|space|underwater|rooftop|warehouse|garden|field|rain|snow|fog|day|night|sunset|sunrise|dawn|dusk)\b/.test(text);
      const look = /\b(light|lighting|lit|sunlight|moonlight|neon|shadow|cinematic|realistic|photoreal|anime|comic|illustration|film grain|colour|color|palette|atmosphere|moody|warm|cool|soft|hard light|volumetric)\b/.test(text);
      const chronological = /\b(begins?|first|then|while|afterward|afterwards|finally|ends?|at the end|throughout|one continuous shot|single continuous shot)\b/.test(text);
      const continuity = /\b(avoid|no cuts?|without cuts?|preserve|remain(?:s)?|unchanged|consistent|continuity|do not|must not|same (?:face|hair|clothes|wardrobe|character)|no (?:text|subtitles|duplicates?|extra limbs?))\b/.test(text);
      const dialogue = /["“”]|\b(dialogue|says|speaks|voiceover|voice-over|narrat|lip.?sync|talking)\b/.test(text);
      const continuation = /\b(continue|continuation|extend|next shot|next clip|after this|picks? up|last good frame)\b/.test(text);
      const retake = /\b(retake|replace an interval|replace the|fix the (?:first|middle|last)|regenerate .*seconds?|repair .*seconds?)\b/.test(text);
      const audioDriven = /\b(audio-driven|soundtrack|music video|to the beat|song|singing|dance to|lip.?sync|supplied audio|audio (?:owns|drives|determines) (?:the )?timing)\b/.test(text);
      const structural = /\b(canny|depth guide|pose guide|control video|storyboard|comic strip|match the structure|same composition)\b/.test(text);
      const tracked = /\b(motion track|trajectory|follow a path|moves? from .+ to|camera path|exact path)\b/.test(text);
      const timedAnchors = /\b(exact opening|opening frame|start frame|exact ending|final frame|end frame|key composition|interior anchor)\b/.test(text);
      const multiBeat = /\b(then|afterward|afterwards|finally|next,|at the end)\b/.test(text);
      const multiReference = tags.length > 1 || /\b(multi-reference|multiple reference|reference images|reference photos|several characters|multiple characters|character consistency)\b/.test(text);
      const multipleCuts = /\b(montage|multiple cuts?|several cuts?|cut to|smash cut|scene change|different locations?|transition to)\b/.test(text);
      const crowd = /\b(crowd|dozens|hundreds|packed audience|many people)\b/.test(text);
      const readableText = /\b(readable text|sign says|title card|exact logo|spells? out|subtitles?|on-screen text)\b/.test(text);
      const complexBody = /\b(fingers?|hands?|acrobat|fight|wrestl|complex dance|gymnast|juggle)\b/.test(text);
      return {raw, text, words, tags, duration, aspect, identityDetail, subjectBasic, action, cameraTerms, camera, setting, look, chronological, continuity, dialogue, continuation, retake, audioDriven, structural, tracked, timedAnchors, multiBeat, multiReference, multipleCuts, crowd, readableText, complexBody};
    }
    function promptCoachRewriteUi(signals) {
      let source = signals.raw.trim().replace(/\s+/g, " ");
      if (!/[.!?]$/.test(source)) source += ".";
      const parts = [];
      if (!/\b(one|single) continuous shot\b/i.test(source)) parts.push(`One continuous ${signals.duration}-second ${signals.aspect} shot.`);
      parts.push(source);
      if (!signals.identityDetail) parts.push("Subject continuity: [define fixed face, hair, wardrobe, colours, and any object details that must not change].");
      else parts.push("Preserve every stated identity, wardrobe, colour, and object detail in every frame.");
      if (!signals.action) parts.push("Action: [describe one physically achievable main action in chronological order].");
      if (!signals.camera) parts.push("Camera: [choose one framing and one deliberate camera movement, or specify a locked camera].");
      if (!(signals.setting && signals.look)) parts.push("Setting and light: [name the location, time of day, light source, and visual style].");
      if (!signals.chronological && signals.multiBeat) parts.push("Timing: rewrite the actions as begins → then → ends, without a scene cut.");
      if (signals.dialogue && !signals.tags.length) parts.push("Dialogue: assign every line to an explicit @character tag; generate isolated Character Voice cues when speaker separation matters.");
      parts.push("Avoid: cuts, duplicate subjects, extra limbs or fingers, identity or wardrobe drift, unreadable text, and unintended camera movement.");
      return parts.join(" ");
    }
    function promptCoachWorkflowUi(signals) {
      const quality = /\b(test|draft|preview|rough|try|experiment)\b/.test(signals.text) ? "Draft" : "Quality (use Draft first if composition is untested)";
      const duration = Math.min(10, Math.max(1, signals.duration));
      const features = [], steps = [];
      let primary = "Generate";
      if (signals.retake) primary = "Retake";
      else if (signals.continuation) primary = "Capture current frame + Generate now; native Extend when Q8 is ready";
      else if (signals.audioDriven) primary = "Audio-to-Video";
      else if (signals.tracked) primary = "Motion Track";
      else if (signals.structural && /\b(control video|canny|depth guide|pose guide)\b/.test(signals.text)) primary = "Union Control";
      else if (signals.multiReference) primary = "Ingredients for true multi-reference identity; Frame Composer + anchors as the ready-now path";
      features.push(`Best match: ${primary}.`);
      if (signals.retake) features.push(`Retake is ${promptCoachAvailabilityUi("retake")}; it replaces only a selected interval.`);
      if (signals.continuation) features.push(`Extend is ${promptCoachAvailabilityUi("extend")}; frame capture provides a visual hand-off but not latent motion or audio continuity.`);
      if (signals.audioDriven) features.push(`Audio-to-Video is ${promptCoachAvailabilityUi("a2v")}; use it when the soundtrack must drive timing.`);
      if (signals.multiReference) features.push(`Ingredients is ${promptCoachAvailabilityUi("ingredients")}; it is the strongest identity choice for several references.`);
      if (signals.structural) features.push(/\b(storyboard|comic strip)\b/.test(signals.text) ? `A static storyboard belongs in Start / Interior / End anchors; Union Control needs a timed guide video and is ${promptCoachAvailabilityUi("union")}.` : `Union Control is ${promptCoachAvailabilityUi("union")}; use it for whole-frame structure from a timed guide.`);
      if (signals.tracked) features.push(`Motion Track is ${promptCoachAvailabilityUi("motion")}; it needs a Start image and normalized path JSON.`);
      if (signals.dialogue) features.push("Character Voices is the dependable speaker-separation pass; native dialogue may swap voices.");
      if (signals.multiBeat) features.push("Scene Beats can order actions inside one shot; they do not create edited cuts.");
      if (signals.timedAnchors) features.push("Start & End Frames or one Interior anchor fit the requested exact compositions.");
      if (signals.complexBody) features.push("Keep hands/body action in medium or close framing and simplify the motion to reduce anatomy drift.");

      if (signals.retake && readinessByMode.retake && !readinessByMode.retake.blocked) {
        steps.push("Select Retake, upload the source MP4, and set the exact start and end seconds of the unwanted interval.");
        steps.push("Enable source-audio preservation when the original sound must remain, then describe only the replacement action.");
      } else if (signals.retake) {
        steps.push("Retake is unavailable while its Q8/HQ assets are missing; Generate cannot safely replace an interval inside an existing MP4.");
        steps.push("For now, export boundary frames around the unwanted interval, Generate a separate replacement shot from the first boundary, then trim and stitch it in an editor; preserve the source audio separately.");
      } else if (signals.continuation) {
        steps.push("Play the completed video, pause on the last good frame, and click “Capture current frame”.");
        steps.push("Click “Use as next clip’s start frame”; keep the same aspect ratio and seed, then describe only what happens next.");
        steps.push("Choose Generate and 3 seconds now. When Extend becomes ready, select Extend, upload the source, choose After source and start with +3 seconds.");
      } else if (signals.audioDriven && readinessByMode.a2v && !readinessByMode.a2v.blocked) {
        steps.push("Select Audio-to-Video, upload the local audio, and set the audio start offset and clip duration.");
        steps.push("Add an optional Start image for identity/composition, then keep the prompt focused on visible motion caused by the sound.");
      } else if (signals.audioDriven) {
        steps.push("Audio-to-Video is unavailable while its Q8/HQ assets are missing; ordinary Generate will not follow exact beats or phonemes.");
        steps.push("For now, Generate a silent 3-second visual test with the intended pacing, then align it to the audio in an editor. Use Audio-to-Video when its readiness line becomes available.");
      } else if (signals.tracked && readinessByMode.motion && !readinessByMode.motion.blocked) {
        steps.push("Select Motion Track, upload the required Start image, and choose the aspect ratio and 3-second duration.");
        steps.push("Enter one normalized path in the sample JSON format, then describe the subject that follows that path.");
      } else if (primary === "Union Control" && readinessByMode.union && !readinessByMode.union.blocked) {
        steps.push("Select Union Control and upload a timed guide video—not a static comic strip.");
        steps.push("Choose the matching control type, keep a 3-second first test, and describe appearance without fighting the guide’s structure.");
      } else if (signals.multiReference && readinessByMode.ingredients && !readinessByMode.ingredients.blocked) {
        steps.push("Select Ingredients, click “Add reference” for each image, and give each subject/object one stable @tag.");
        steps.push("Use every @tag exactly once in the prompt, choose the 16 GB-friendly profile first, then add Start / End anchors only when their compositions are essential.");
      } else {
        steps.push(`Select Generate. Set ${quality}, ${duration > 5 ? "3 seconds for the first take (split this longer idea)" : `${duration} second${duration === 1 ? "" : "s"}`}, and ${signals.aspect}.`);
        if (signals.multiReference) steps.push("For the ready-now identity path, expand Character Bible and save/load the recurring character; then expand Frame Composer, add up to four tagged photos, Compose frame, and assign it as Start.");
        else if (signals.identityDetail || signals.tags.length) steps.push("Expand Character Bible, load or save the subject’s fixed appearance and references, and keep the same tag/name in the prompt.");
        if (signals.structural && /\b(storyboard|comic strip)\b/.test(signals.text)) steps.push("Place storyboard panels in Start & End Frames and, only if essential, one Multi-reference Timeline Anchor; do not upload a static strip as motion.");
        else if (signals.timedAnchors) steps.push("Expand Start & End Frames and upload the exact boundary stills; add at most one Interior anchor for a critical middle composition.");
        if (signals.multiBeat) steps.push("Expand Scene Beats & Motion and add short chronological beats whose total remains one continuous shot.");
        steps.push("Paste the reviewed prompt, keep or record the seed, click Generate video, and change only one variable on the next try.");
      }
      if (signals.dialogue) steps.push("After the picture works, expand Character Voices, assign one authorized preset to each @character, create one cue per line, then mux the cues into a separate MP4.");
      if ((signals.tracked && promptCoachAvailabilityUi("motion").includes("locked")) || (primary === "Union Control" && promptCoachAvailabilityUi("union").includes("locked"))) steps.push("Until the Q8 control lane is ready, approximate this with Generate plus a Start image and simple camera/action wording; exact trajectory or structure control is not available in Q4.");
      return {features, steps};
    }
    function analysePromptCoachUi(raw) {
      const signals = promptCoachSignalsUi(raw), strengths = [], fixes = [], risks = [];
      let score = 0;
      if (signals.identityDetail) { score += 20; strengths.push("Subject identity or fixed visual details are explicit."); }
      else if (signals.subjectBasic) { score += 12; strengths.push("A subject/scene is present, but its fixed identity is still loose."); fixes.push("Add fixed face/object details, wardrobe, colours, and what must remain unchanged. (+8)"); }
      else fixes.push("Name the main subject and its fixed visual identity. (+20)");
      if (signals.action) { score += 20; strengths.push("A visible main action is described."); } else fixes.push("Add one physically achievable main action. (+20)");
      if (signals.camera) { score += 15; strengths.push("Camera framing or movement is specified."); } else fixes.push("Choose one framing and one camera move—or a locked camera. (+15)");
      if (signals.setting && signals.look) { score += 15; strengths.push("Setting and visual/lighting treatment are both defined."); }
      else if (signals.setting || signals.look) { score += 8; strengths.push("The setting or look is partly defined."); fixes.push("Add the missing location/time/lighting/style detail. (+7)"); }
      else fixes.push("Specify location, time of day, lighting, atmosphere, and style. (+15)");
      if (signals.chronological) { score += 15; strengths.push("Timing or chronological flow is explicit."); }
      else if (signals.action && !signals.multiBeat) { score += 9; fixes.push("State how the shot begins and ends, even for one action. (+6)"); }
      else fixes.push("Order the action as begins → then → ends within one shot. (+15)");
      if (signals.continuity) { score += 10; strengths.push("Continuity or avoid constraints are present."); } else fixes.push("Add continuity rules and a short avoid clause. (+10)");
      if (!signals.multipleCuts && !signals.crowd && !signals.readableText && signals.cameraTerms.length <= 2) score += 5;
      else fixes.push("Simplify the request to one feasible short shot. (+5 after risks are removed)");

      if (signals.multipleCuts) { score -= 10; risks.push("Multiple cuts/locations fight the short one-shot generation format. Split them into separate clips. (−10)"); }
      if (signals.cameraTerms.length > 2) { score -= 8; risks.push("Several camera instructions may conflict. Keep one framing plus one movement. (−8)"); }
      if (signals.crowd) { score -= 5; risks.push("Crowds increase duplicate-person and face drift risk; reduce the visible cast. (−5)"); }
      if (signals.readableText) { score -= 5; risks.push("Exact readable text/logos are unreliable in generated video; composite them afterward. (−5)"); }
      if (signals.duration > 5 && (signals.multiBeat || signals.complexBody || signals.multiReference)) { score -= 10; risks.push("This is a complex >5-second take; split it into 3-second clips and continue/stitch. (−10)"); }
      if (signals.dialogue && !signals.tags.length) { score -= 5; risks.push("Dialogue has no explicit @speaker tag, so voices may be intermixed. Add a stable tag such as @maya to every line. (−5)"); }
      if (signals.complexBody) risks.push("Detailed hands or complex body motion can still hallucinate even with a complete prompt.");
      score = Math.max(0, Math.min(100, score));
      if (!strengths.length) strengths.push("The core idea is present, but it needs production details before rendering.");
      if (!fixes.length && !risks.length) fixes.push("All scored prompt components are present. Render a 3-second test; 100% readiness still does not guarantee model output.");
      else fixes.push(...risks);
      const workflow = promptCoachWorkflowUi(signals);
      return {score, strengths, fixes, features:workflow.features, steps:workflow.steps, rewrite:promptCoachRewriteUi(signals)};
    }
    function promptCoachBlockUi(id, headingText, items, ordered = false) {
      const block = $ui(id); block.replaceChildren();
      const heading = document.createElement("h3"); heading.textContent = headingText; block.appendChild(heading);
      const list = document.createElement(ordered ? "ol" : "ul");
      for (const item of items) { const row = document.createElement("li"); row.textContent = item; list.appendChild(row); }
      block.appendChild(list);
    }
    function promptCoachContextUi() {
      let profile = "quality", aspect_ratio = "profile", duration_seconds = 3;
      if (modeUi === "generate") {
        profile = document.querySelector('input[name="profile"]:checked')?.value || "quality";
        aspect_ratio = $ui("generateAspect").value; duration_seconds = Number($ui("generateDuration").value);
      } else if (modeUi === "a2v") {
        profile = $ui("a2vProfile").value; aspect_ratio = $ui("a2vAspect").value; duration_seconds = Number($ui("a2vDuration").value);
      } else if (modeUi === "ingredients") {
        profile = $ui("ingredientsProfile").value; aspect_ratio = $ui("ingredientsAspect").value; duration_seconds = 5;
      } else if (modeUi === "union" || modeUi === "motion") {
        profile = $ui(`${modeUi}Profile`).value; aspect_ratio = $ui(`${modeUi}Aspect`).value; duration_seconds = Number($ui(`${modeUi}Duration`).value);
      } else if (modeUi === "extend") {
        duration_seconds = Number($ui("extendSeconds").value);
      } else if (modeUi === "retake") {
        duration_seconds = Math.max(1, Math.min(10, Math.round(Number($ui("retakeEnd").value) - Number($ui("retakeStart").value)) || 1));
      }
      let motionTracks = 0;
      try { const parsed = JSON.parse($ui("motionTracks").value); motionTracks = Array.isArray(parsed) ? parsed.length : 0; } catch (_) {}
      const references = {
        start_frame:Boolean($ui("startImage").files.length || composedFrameAssignments.start),
        end_frame:Boolean($ui("endImage").files.length || composedFrameAssignments.end),
        interior_anchors:Array.from(document.querySelectorAll(".anchor-card")).filter((card) => card.querySelector(".anchor-images")?.files.length || card._composedFrameData).length,
        character_references:managedCharacterReferences.length + $ui("characterReferenceImages").files.length,
        ingredient_references:Array.from(document.querySelectorAll(".ingredient-reference")).filter((card) => card.querySelector(".ingredient-image")?.files.length).length,
        control_video:Boolean($ui("unionVideo").files.length),
        audio:Boolean($ui("a2vAudio").files.length),
        motion_tracks:motionTracks,
        scene_beats:document.querySelectorAll(".scene-card").length,
      };
      const seedValue = Number($ui("seed").value);
      return {mode:modeUi, profile, aspect_ratio, duration_seconds, seed:Number.isSafeInteger(seedValue) && seedValue >= 0 && seedValue <= 4294967295 ? seedValue : 42, references};
    }
    function appendPromptCoachChatUi(role, content, label = "") {
      const bubble = document.createElement("div"); bubble.className = `coach-chat ${role}`;
      const meta = document.createElement("span"); meta.className = "coach-chat-meta"; meta.textContent = label || (role === "user" ? "You" : "Coach");
      bubble.appendChild(meta); bubble.appendChild(document.createTextNode(content)); promptCoachConversationUi.appendChild(bubble);
      promptCoachConversationUi.scrollTop = promptCoachConversationUi.scrollHeight;
    }
    function warnExperimentalSelectionUi(select) {
      const option = select.selectedOptions[0], kind = option?.dataset.experimental || "";
      if (select.id === "ingredientsAspect") $ui("ingredientsAspectWarning").hidden = kind !== "ingredients-aspect";
      if (select.id === "motionProfile") {
        const experimental = kind === "motion-resolution";
        $ui("motionProfileWarning").hidden = !experimental;
        if (experimental && $ui("motionDuration").value !== "3") {
          $ui("motionDuration").value = "3";
          $ui("motionDuration").dispatchEvent(new Event("input", {bubbles:true}));
        }
      }
      if (!kind) return;
      const key = `${select.id}:${option.value}`;
      if (experimentalCoachWarningsSeenUi.has(key)) return;
      experimentalCoachWarningsSeenUi.add(key);
      const messages = {
        "ingredients-aspect": `You selected ${option.textContent.trim()}. The Ingredients adapter was trained at one 768×448 landscape, 121-frame bucket; the preferred 384×224 profile preserves that geometry at lower memory. This alternate canvas is out of training distribution, so @tag identity, anatomy, panel placement and composition may weaken—especially in portrait. If you continue, render one five-second low-profile canary with a fixed seed and compare every referenced identity against the Preferred landscape result.`,
        "ingredients-memory": `You selected ${option.textContent.trim()}. This is the adapter's exact trained canvas and can improve native detail, but 768×448×121 is experimental on this 16 GB Mac because peak decode and reference-attention memory can trigger heavy swap or another Python/Metal termination. Use the 384×224 Preferred profile first; try this bucket only after that shot works, with other heavy apps closed.`,
        "motion-resolution": `You selected ${option.textContent.trim()}. The Motion adapter itself is not capped at 512×320, but 640×384 has about 1.5× as many pixels and roughly 2.25× the attention work of the Preferred 512×320 profile. This Q8 single-stage path has no IC-LoRA tiling on the 16 GB Mac, so runtime, swap and Python/Metal termination risk rise sharply. Studio has returned duration to three seconds. Start with one simple track and no unnecessary anchors; use Preferred plus upscaling when reliability matters.`
      };
      openPromptCoachUi({focusInput:false});
      appendPromptCoachChatUi("assistant", messages[kind] || "This setting is experimental on the current local workflow. Test a short canary before relying on it.", "Coach · Experimental guardrail");
      $ui("promptCoachNotice").textContent = "Local safety guidance only—no model call or render was started. Ask a follow-up if you want the agent to assess your exact prompt.";
    }
    function boundedPromptCoachHistoryUi(items) {
      const bounded = []; let used = 0;
      for (const item of items.slice(-10).reverse()) {
        const remaining = 8000 - used; if (remaining <= 0) break;
        const content = String(item.content || "").slice(0, Math.min(2000, remaining)); if (!content) continue;
        bounded.push({role:item.role, content}); used += content.length;
      }
      return bounded.reverse();
    }
    function promptCoachAssistantHistoryUi(result) {
      const answer = String(result.answer || "").trim(), rewrite = String(result.rewrite || "").trim();
      if (result.response_kind === "conversation" || result.response_kind === "workflow_guidance" || result.response_kind === "score_report") return answer.slice(0, 2000);
      if (!rewrite) return answer.slice(0, 2000);
      const marker = "\n\nCurrent production-ready draft:\n";
      const answerExcerpt = answer.length > 350 ? `${answer.slice(0, 349).trimEnd()}…` : answer;
      const remaining = Math.max(0, 2000 - answerExcerpt.length - marker.length);
      const draftExcerpt = rewrite.length > remaining && remaining > 1 ? `${rewrite.slice(0, remaining - 1).trimEnd()}…` : rewrite.slice(0, remaining);
      return `${answerExcerpt}${marker}${draftExcerpt}`;
    }
    function resetPromptCoachHistoryUi(message = "") {
      promptCoachHistoryUi = []; promptCoachLastPromptUi = ""; promptCoachTutorialContextUi = "";
      promptCoachConversationUi.replaceChildren(); promptCoachResultUi.hidden = true; $ui("promptCoachSuggestions").replaceChildren();
      $ui("promptCoachQuestionsLabel").hidden = true;
      if (message) appendPromptCoachChatUi("assistant", message, "Coach");
    }
    function renderPromptCoachSuggestionsUi(items) {
      const target = $ui("promptCoachSuggestions"), label = $ui("promptCoachQuestionsLabel"); target.replaceChildren();
      for (const item of (items || []).slice(0, 3)) {
        const question = String(item || "").trim(); if (!question) continue;
        const button = document.createElement("button"); button.type = "button"; button.className = "coach-suggestion"; button.textContent = question; button.setAttribute("aria-label", `Answer Coach question: ${question}`);
        button.addEventListener("click", () => {
          const scaffold = `Answer to Coach’s question:\n${question}\n\nMy answer: `;
          promptCoachMessageUi.value = scaffold; promptCoachMessageUi.focus(); promptCoachMessageUi.setSelectionRange(scaffold.length, scaffold.length);
        }); target.appendChild(button);
      }
      label.hidden = !target.childElementCount;
    }
    function renderPromptCoachResultUi(result) {
      const guidanceOnly = result.response_kind === "workflow_guidance" || result.response_kind === "conversation";
      const conversationOnly = result.response_kind === "conversation";
      const scoreOnly = result.response_kind === "score_report";
      const audit = $ui("promptCoachAudit"), auditSummary = $ui("promptCoachAuditSummary");
      audit.hidden = conversationOnly;
      audit.open = !guidanceOnly;
      if (scoreOnly) audit.open = true;
      auditSummary.textContent = scoreOnly ? "Prompt score & deterministic audit" : (guidanceOnly ? "Optional prompt audit & production draft" : "Prompt audit & production draft");
      const rewriteReview = scoreOnly ? {} : (result.rewrite_analysis || {}), draftScore = Number.isFinite(Number(rewriteReview.score)) ? Number(rewriteReview.score) : Number(result.score || 0);
      const score = $ui("promptCoachScore"); score.style.setProperty("--coach-score", `${draftScore}%`); score.querySelector("span").textContent = `${draftScore}%`;
      const sourceScoreCopy = Number.isFinite(Number(result.score)) ? ` · source idea ${result.score}%` : "";
      $ui("promptCoachVerdict").textContent = scoreOnly ? `${result.verdict || "Prompt readiness"} · source prompt` : `${rewriteReview.verdict || result.verdict} · finished draft readiness${sourceScoreCopy}`;
      const strengths = rewriteReview.strengths || result.strengths || [];
      const improvements = rewriteReview.improvements || result.improvements || [];
      const risks = rewriteReview.risks || result.risks || [];
      promptCoachBlockUi("promptCoachStrengths", scoreOnly ? "What the source prompt specifies well" : "Finished draft · what is specified well", strengths);
      promptCoachBlockUi("promptCoachFixes", scoreOnly ? "What would improve its readiness" : "Finished draft · remaining risks and improvements", [...improvements, ...risks]);
      const originalSettings = result.settings || {}, settings = rewriteReview.settings || originalSettings;
      const workflowChanged = rewriteReview.workflow && rewriteReview.workflow !== result.workflow;
      const workflow = rewriteReview.workflow || result.workflow || "generate";
      const semanticBest = rewriteReview.semantic_best || result.semantic_best;
      const workflowReason = rewriteReview.workflow_reason || result.workflow_reason || "best fit for the finished draft";
      const renderRisk = rewriteReview.render_risk || result.render_risk || "unknown";
      const recommendationCertainty = rewriteReview.recommendation_certainty || result.recommendation_certainty || "unknown";
      const workflowConflicts = rewriteReview.workflow_conflicts || result.workflow_conflicts || [];
      const availabilityNotes = rewriteReview.availability_notes || result.availability_notes || [];
      const features = [
        `Best available workflow for the finished draft: ${workflow.replaceAll("_", " ")}.`,
        ...((semanticBest && semanticBest !== workflow) ? [`Semantic best match: ${semanticBest.replaceAll("_", " ")}; the workflow above is the currently runnable fallback.`] : []),
        `Why: ${workflowReason}.`,
        `Finished-draft render risk: ${renderRisk} · recommendation certainty: ${recommendationCertainty}. These are qualitative production judgments, not success probabilities.`,
        `Settings for the draft below: ${settings.profile || "quality"} · ${settings.aspect_ratio || "profile"} · ${settings.duration_seconds || 3}s · seed ${settings.seed ?? 42}.`,
        ...workflowConflicts,
        ...(workflowChanged ? [`The rewritten draft now analyzes as ${rewriteReview.workflow}; review that workflow change before applying it.`] : []),
        ...availabilityNotes,
      ];
      promptCoachBlockUi("promptCoachFeatures", "Finished draft · feature and settings decision", features);
      promptCoachBlockUi("promptCoachWalkthrough", "Apply it in Dang Studio · finished draft", rewriteReview.walkthrough || result.walkthrough || [], true);
      $ui("promptCoachRewrite").value = result.rewrite || "";
      const rewriteSafety = $ui("promptCoachRewriteSafety"); rewriteSafety.textContent = result.rewrite_safety_note || ""; rewriteSafety.hidden = !result.rewrite_safety_note;
      $ui("promptCoachFeatures").hidden = scoreOnly || conversationOnly;
      $ui("promptCoachWalkthrough").hidden = scoreOnly || conversationOnly;
      $ui("promptCoachDraftBlock").hidden = scoreOnly || conversationOnly;
      $ui("promptCoachUse").hidden = scoreOnly || conversationOnly;
      const provider = $ui("promptCoachProvider"); provider.textContent = result.provider_label || "Private local agent"; provider.classList.toggle("guided", result.provider === "guided");
      $ui("promptCoachProviderNote").textContent = result.provider_note || (scoreOnly ? "Deterministic readiness checklist · no AI model, creative rewrite, or media job was started." : "");
      const action = $ui("promptCoachProviderAction"); action.textContent = result.provider_action || ""; action.hidden = !result.provider_action;
      renderPromptCoachSuggestionsUi(scoreOnly ? [] : result.follow_ups);
      promptCoachResultUi.hidden = false;
      if (guidanceOnly) {
        promptCoachConversationUi.scrollTop = promptCoachConversationUi.scrollHeight;
      } else {
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        promptCoachResultUi.scrollIntoView({behavior:reducedMotion ? "auto" : "smooth", block:"nearest"});
      }
    }
    async function askPromptCoachUi(message, visibleMessage = message, action = "chat") {
      const raw = promptCoachInputUi.value, notice = $ui("promptCoachNotice");
      if (action !== "chat" && !raw.trim()) { notice.textContent = "Paste a prompt or load the current Studio prompt first."; promptCoachInputUi.focus(); return; }
      if (!message.trim() || promptCoachBusyUi) return;
      if (promptCoachLastPromptUi && raw !== promptCoachLastPromptUi) resetPromptCoachHistoryUi("The prompt changed, so I’m treating this as a new review instead of carrying over assumptions from the earlier one.");
      promptCoachLastPromptUi = raw;
      promptCoachBusyUi = true; notice.textContent = "Coach is reasoning locally…";
      const requestSerial = ++promptCoachRequestSerialUi;
      const provider = $ui("promptCoachProvider"), scoreButton = $ui("promptCoachScoreButton"), analyseButton = $ui("promptCoachAnalyze"), sendButton = $ui("promptCoachSend");
      const loadButton = $ui("promptCoachLoad"), useButton = $ui("promptCoachUse");
      provider.textContent = "Thinking locally…"; provider.classList.remove("guided");
      promptCoachInputUi.disabled = true; loadButton.disabled = true; useButton.disabled = true;
      promptCoachPanelUi.setAttribute("aria-busy", "true");
      scoreButton.disabled = true; analyseButton.disabled = true; sendButton.disabled = true; analyseButton.textContent = "Thinking…"; sendButton.textContent = "Thinking…";
      appendPromptCoachChatUi("user", visibleMessage);
      try {
        const result = await apiUi("/api/prompt-coach", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action, message:message.trim(), prompt:raw, history:boundedPromptCoachHistoryUi(promptCoachHistoryUi), context:promptCoachContextUi()})});
        if (requestSerial !== promptCoachRequestSerialUi || promptCoachInputUi.value !== raw) {
          notice.textContent = "The prompt changed while Coach was replying, so the stale result was discarded. Ask again for the current prompt.";
          promptCoachLastPromptUi = "";
          return;
        }
        appendPromptCoachChatUi("assistant", result.answer, result.provider_label || "Coach");
        promptCoachHistoryUi.push({role:"user", content:message.trim()}, {role:"assistant", content:promptCoachAssistantHistoryUi(result)}); promptCoachHistoryUi = boundedPromptCoachHistoryUi(promptCoachHistoryUi);
        renderPromptCoachResultUi(result); notice.textContent = ""; promptCoachMessageUi.value = "";
      } catch (error) {
        notice.textContent = error.message;
        provider.textContent = "Coach unavailable"; provider.classList.add("guided");
        appendPromptCoachChatUi("assistant", `I couldn't complete that reply: ${error.message}`, "Coach");
      } finally {
        promptCoachBusyUi = false; promptCoachInputUi.disabled = false; loadButton.disabled = false; useButton.disabled = false;
        promptCoachPanelUi.removeAttribute("aria-busy");
        scoreButton.disabled = false; analyseButton.disabled = false; sendButton.disabled = false; scoreButton.textContent = "Score prompt"; analyseButton.textContent = "Develop / review prompt"; sendButton.textContent = "Ask Coach";
      }
    }
    async function scorePromptCoachUi() {
      const raw = promptCoachInputUi.value, notice = $ui("promptCoachNotice");
      if (!raw.trim()) { notice.textContent = "Paste a prompt or load the current Studio prompt first."; promptCoachInputUi.focus(); return; }
      if (promptCoachBusyUi) return;
      promptCoachBusyUi = true; notice.textContent = "Scoring the source prompt with the fixed readiness checklist…";
      const requestSerial = ++promptCoachRequestSerialUi;
      const provider = $ui("promptCoachProvider"), scoreButton = $ui("promptCoachScoreButton"), analyseButton = $ui("promptCoachAnalyze"), sendButton = $ui("promptCoachSend");
      const loadButton = $ui("promptCoachLoad"), useButton = $ui("promptCoachUse");
      provider.textContent = "Deterministic checklist…"; provider.classList.remove("guided");
      promptCoachPanelUi.setAttribute("aria-busy", "true");
      promptCoachInputUi.disabled = true; loadButton.disabled = true; useButton.disabled = true;
      scoreButton.disabled = true; analyseButton.disabled = true; sendButton.disabled = true; scoreButton.textContent = "Scoring…";
      try {
        const result = await apiUi("/api/prompt-coach", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action:"score", message:"Score this prompt.", prompt:raw, history:[], context:promptCoachContextUi()})});
        if (requestSerial !== promptCoachRequestSerialUi || promptCoachInputUi.value !== raw) {
          notice.textContent = "The prompt changed while it was being scored, so the stale score was discarded. Score the current prompt again.";
          return;
        }
        renderPromptCoachResultUi(result);
        notice.textContent = "Score complete. Your prompt was not rewritten and no chat turn, AI model, or media job was started.";
      } catch (error) {
        notice.textContent = `Scoring failed: ${error.message}`;
        provider.textContent = "Score unavailable"; provider.classList.add("guided");
      } finally {
        promptCoachBusyUi = false; promptCoachInputUi.disabled = false; loadButton.disabled = false; useButton.disabled = false;
        promptCoachPanelUi.removeAttribute("aria-busy");
        scoreButton.disabled = false; analyseButton.disabled = false; sendButton.disabled = false; scoreButton.textContent = "Score prompt"; analyseButton.textContent = "Develop / review prompt"; sendButton.textContent = "Ask Coach";
      }
    }
    function developPromptCoachUi() {
      askPromptCoachUi("If this is a rough or open-ended idea, develop it into one complete, coherent, concrete, production-ready shot by making sensible creative choices for action, chronology, camera, setting, lighting, style and ambience; do not leave bracketed placeholders. Preserve every named character and stated intent, and do not invent additional named characters, dialogue, lore or scene cuts unless requested. If the prompt is already detailed, review it conservatively and preserve its decisions. In either case, explain the best available workflow and exact settings for this Studio.", "Develop this idea, or conservatively review it if it is already detailed.", "develop");
    }
    function openPromptCoachUi({focusInput = true} = {}) {
      if (!tutorialPanelUi.hidden) pauseTutorialUi();
      promptCoachReturnFocusUi = document.activeElement;
      promptCoachPanelUi.hidden = false; $ui("promptCoachLauncher").setAttribute("aria-expanded", "true");
      if (!promptCoachInputUi.value.trim() && promptUi.value.trim()) { promptCoachInputUi.value = promptUi.value; promptCoachInputUi.dispatchEvent(new Event("input", {bubbles:true})); }
      if (focusInput) promptCoachInputUi.focus();
    }
    function closePromptCoachUi() { promptCoachPanelUi.hidden = true; $ui("promptCoachLauncher").setAttribute("aria-expanded", "false"); if (tutorialSessionActiveUi && !tutorialResumeUi.hidden) tutorialResumeUi.focus(); else if (promptCoachReturnFocusUi?.focus) promptCoachReturnFocusUi.focus(); }
    $ui("promptCoachLauncher").addEventListener("click", () => openPromptCoachUi());
    $ui("closePromptCoach").addEventListener("click", closePromptCoachUi);
    ["ingredientsProfile", "ingredientsAspect", "motionProfile"].forEach((id) => $ui(id).addEventListener("change", (event) => { if (event.isTrusted) warnExperimentalSelectionUi(event.currentTarget); }));
    $ui("promptCoachLoad").addEventListener("click", () => { const next = promptUi.value; if (next !== promptCoachInputUi.value) resetPromptCoachHistoryUi("Loaded a new Studio prompt. I’ve cleared the previous coach assumptions."); promptCoachInputUi.value = next; promptCoachInputUi.dispatchEvent(new Event("input", {bubbles:true})); promptCoachLastPromptUi = ""; promptCoachInputUi.focus(); $ui("promptCoachNotice").textContent = promptUi.value.trim() ? "Loaded the current Studio prompt." : "The Studio prompt is empty."; });
    $ui("promptCoachScoreButton").addEventListener("click", scorePromptCoachUi);
    $ui("promptCoachAnalyze").addEventListener("click", developPromptCoachUi);
    $ui("promptCoachSend").addEventListener("click", () => askPromptCoachUi(promptCoachMessageUi.value, promptCoachMessageUi.value, "chat"));
    promptCoachMessageUi.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); askPromptCoachUi(promptCoachMessageUi.value, promptCoachMessageUi.value, "chat"); } });
    $ui("promptCoachUse").addEventListener("click", () => {
      const appliedPrompt = $ui("promptCoachRewrite").value;
      promptUi.value = appliedPrompt; promptUi.dispatchEvent(new Event("input", {bubbles:true}));
      promptCoachInputUi.value = appliedPrompt; promptCoachInputUi.dispatchEvent(new Event("input", {bubbles:true})); promptCoachLastPromptUi = appliedPrompt;
      closePromptCoachUi(); promptUi.scrollIntoView({behavior:"smooth", block:"center"}); promptUi.focus();
      noticeUi.textContent = "Prompt Coach applied a complete draft to the prompt text. Review its creative assumptions before generating; Studio mode, profile, aspect ratio and duration are unchanged.";
    });
    $ui("openTutorial").addEventListener("click", openTutorialUi);
    $ui("startTutorialFromHelp").addEventListener("click", openTutorialUi);
    $ui("closeTutorial").addEventListener("click", () => closeTutorialUi());
    tutorialResumeUi.addEventListener("click", openTutorialUi);
    $ui("tutorialShowControl").addEventListener("click", revealTutorialTargetUi);
    $ui("tutorialApplyPractice").addEventListener("click", applyTutorialPracticeUi);
    $ui("tutorialRestoreSetup").addEventListener("click", tutorialRestoreSnapshotUi);
    $ui("tutorialAskCoach").addEventListener("click", tutorialAskCoachUi);
    $ui("tutorialNext").addEventListener("click", tutorialNextUi);
    $ui("tutorialBack").addEventListener("click", tutorialBackUi);
    $ui("tutorialAllLessons").addEventListener("click", tutorialAllLessonsUi);
    $ui("tutorialReset").addEventListener("click", () => { tutorialCompletedUi = new Set(); tutorialLessonIdUi = null; tutorialStepUi = 0; try { localStorage.removeItem(tutorialStorageKeyUi); } catch (_) {} renderTutorialMenuUi(); });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (!tutorialPanelUi.hidden) closeTutorialUi();
      else if (!promptCoachPanelUi.hidden) closePromptCoachUi();
    });
    $ui("openHelp").addEventListener("click", () => { helpPanelUi.open = true; helpPanelUi.scrollIntoView({behavior:"smooth", block:"start"}); });
    $ui("recommendSetup").addEventListener("click", recommendSetupUi);
    $ui("copyPromptTemplate").addEventListener("click", async () => {
      const button = $ui("copyPromptTemplate"), value = $ui("agentPromptTemplate").textContent.trim();
      try {
        await navigator.clipboard.writeText(value);
        button.textContent = "Copied";
      } catch (_) {
        const area = document.createElement("textarea"); area.value = value; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove(); button.textContent = "Copied";
      }
      setTimeout(() => { button.textContent = "Copy agent prompt template"; }, 1600);
    });
    function readNumberUi(id, minimum, maximum, label, integer = false) {
      const value = Number($ui(id).value);
      if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isSafeInteger(value))) {
        throw new Error(`${label} must be ${integer ? "a whole number" : "a number"} from ${minimum} to ${maximum}.`);
      }
      return value;
    }
    function fileDataUrlUi(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
        reader.readAsDataURL(file);
      });
    }
    function extensionUi(name) {
      const point = name.lastIndexOf("."); return point >= 0 ? name.slice(point).toLowerCase() : "";
    }
    function requiredFileUi(id, label, extensions, maxBytes) {
      const file = $ui(id).files[0] || null;
      if (!file) throw new Error(`${label} is required.`);
      if (!extensions.includes(extensionUi(file.name))) throw new Error(`${label} has an unsupported file extension.`);
      if (file.size < 1 || file.size > maxBytes) throw new Error(`${label} must be between 1 byte and ${Math.round(maxBytes / 1048576)} MB.`);
      return file;
    }
    function optionalImageUi(id, label) {
      const file = $ui(id).files[0] || null;
      if (!file) return null;
      if (![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)) || file.size < 1 || file.size > 8 * 1048576) {
        throw new Error(`${label} must be a PNG, JPEG, or WebP image no larger than 8 MB.`);
      }
      return file;
    }
    async function uploadUi(file) { return { name:file.name, data:await fileDataUrlUi(file) }; }

    const characterFieldIdsUi = {
      character_name:"characterName", character_appearance:"characterAppearance", character_wardrobe:"characterWardrobe",
      character_continuity:"characterContinuity", character_avoid:"characterAvoid"
    };
    function characterFieldsUi() {
      const payload = {};
      for (const [key, id] of Object.entries(characterFieldIdsUi)) payload[key] = $ui(id).value;
      return payload;
    }
    function setCharacterFieldsUi(profile) {
      for (const [key, id] of Object.entries(characterFieldIdsUi)) $ui(id).value = profile[key] || "";
    }
    function characterReferenceFilesUi() { return Array.from($ui("characterReferenceImages").files || []); }
    function renderCharacterReferencePreviewsUi() {
      const input = $ui("characterReferenceImages"), preview = $ui("characterReferencePreview");
      (previewUiUrls.get(input) || []).forEach((url) => URL.revokeObjectURL(url));
      previewUiUrls.delete(input); preview.replaceChildren();
      for (const reference of managedCharacterReferences) {
        const image = document.createElement("img"); image.src = reference.url; image.alt = `Saved character reference: ${reference.name}`; image.title = reference.name; preview.appendChild(image);
      }
      const urls = [];
      for (const file of characterReferenceFilesUi()) {
        const url = URL.createObjectURL(file), image = document.createElement("img"); urls.push(url); image.src = url; image.alt = `New character reference: ${file.name}`; image.title = file.name; preview.appendChild(image);
      }
      previewUiUrls.set(input, urls);
      const names = [...managedCharacterReferences.map((item) => item.name), ...characterReferenceFilesUi().map((file) => file.name)];
      $ui("characterReferenceCopy").textContent = names.length ? names.join(", ") : "PNG, JPEG, or WebP · managed locally when saved";
      updateAnchorUiSlots();
    }
    async function refreshCharacterProfilesUi(preferredId = null) {
      const data = await apiUi("/api/characters", {cache:"no-store"}), select = $ui("characterProfileSelect");
      characterProfilesUi = data.profiles || [];
      const selected = preferredId || select.value || activeCharacterProfileId || "";
      select.replaceChildren();
      const empty = document.createElement("option"); empty.value = ""; empty.textContent = characterProfilesUi.length ? "Select a saved character" : "No saved characters yet"; select.appendChild(empty);
      for (const profile of characterProfilesUi) {
        const option = document.createElement("option"); option.value = profile.id; option.textContent = `${profile.character_name} · ${profile.references.length} ref${profile.references.length === 1 ? "" : "s"}`; select.appendChild(option);
      }
      if (characterProfilesUi.some((profile) => profile.id === selected)) select.value = selected;
      const hasSelection = Boolean(select.value);
      $ui("loadCharacterProfile").disabled = !hasSelection;
      $ui("replaceCharacterProfile").disabled = !hasSelection;
      $ui("deleteCharacterProfile").disabled = !hasSelection;
      return data;
    }
    async function loadCharacterProfileUi(profileId) {
      if (!profileId) throw new Error("Select a saved character first.");
      const data = await apiUi(`/api/characters/${encodeURIComponent(profileId)}`, {cache:"no-store"}), profile = data.profile;
      activeCharacterProfileId = profile.id; managedCharacterReferences = profile.references || [];
      setCharacterFieldsUi(profile); $ui("characterReferenceImages").value = "";
      $ui("characterProfileSelect").value = profile.id; renderCharacterReferencePreviewsUi();
      $ui("characterProfileStatus").className = "character-profile-status ready";
      $ui("characterProfileStatus").textContent = `Loaded ${profile.character_name}. Its ${managedCharacterReferences.length} managed reference${managedCharacterReferences.length === 1 ? "" : "s"} will be submitted automatically.`;
    }
    async function saveCharacterProfileUi(replace) {
      const targetId = replace ? $ui("characterProfileSelect").value : null;
      if (replace && !targetId) throw new Error("Select the character to replace first.");
      const files = characterReferenceFilesUi(), payload = {...characterFieldsUi(), references:[], replace:Boolean(replace)};
      if (files.length) payload.references = await Promise.all(files.map(uploadUi));
      else if (replace) payload.copy_references_from = targetId;
      else if (activeCharacterProfileId && managedCharacterReferences.length) payload.copy_references_from = activeCharacterProfileId;
      const path = replace ? `/api/characters/${encodeURIComponent(targetId)}` : "/api/characters";
      const method = replace ? "PUT" : "POST";
      const data = await apiUi(path, {method, headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      await refreshCharacterProfilesUi(data.profile.id); await loadCharacterProfileUi(data.profile.id);
      $ui("characterProfileStatus").textContent = `${data.profile.character_name} was ${replace ? "replaced" : "saved"} locally with ${data.profile.references.length} managed reference${data.profile.references.length === 1 ? "" : "s"}.`;
    }
    $ui("characterProfileSelect").addEventListener("change", () => {
      const hasSelection = Boolean($ui("characterProfileSelect").value);
      $ui("loadCharacterProfile").disabled = !hasSelection; $ui("replaceCharacterProfile").disabled = !hasSelection; $ui("deleteCharacterProfile").disabled = !hasSelection;
      if (hasSelection && $ui("characterProfileSelect").value !== activeCharacterProfileId) {
        $ui("characterProfileStatus").className = "character-profile-status";
        $ui("characterProfileStatus").textContent = "Click Load to place this saved character and its managed references into the composer.";
      }
    });
    $ui("loadCharacterProfile").addEventListener("click", async () => {
      try { await loadCharacterProfileUi($ui("characterProfileSelect").value); }
      catch (error) { $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = error.message; }
    });
    $ui("saveCharacterProfile").addEventListener("click", async () => {
      try { await saveCharacterProfileUi(false); }
      catch (error) { $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = error.message; }
    });
    $ui("replaceCharacterProfile").addEventListener("click", async () => {
      const id = $ui("characterProfileSelect").value, summary = characterProfilesUi.find((item) => item.id === id);
      if (!id || !window.confirm(`Replace the saved visual profile for ${summary?.character_name || "this character"}?`)) return;
      try { await saveCharacterProfileUi(true); }
      catch (error) { $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = error.message; }
    });
    $ui("deleteCharacterProfile").addEventListener("click", async () => {
      const id = $ui("characterProfileSelect").value, summary = characterProfilesUi.find((item) => item.id === id);
      if (!id || !window.confirm(`Delete the local visual profile and managed images for ${summary?.character_name || "this character"}?`)) return;
      try {
        await apiUi(`/api/characters/${encodeURIComponent(id)}`, {method:"DELETE", headers:{"Content-Type":"application/json"}, body:JSON.stringify({confirm:true})});
        if (activeCharacterProfileId === id) { activeCharacterProfileId = null; managedCharacterReferences = []; setCharacterFieldsUi({}); $ui("characterReferenceImages").value = ""; renderCharacterReferencePreviewsUi(); }
        await refreshCharacterProfilesUi(); $ui("characterProfileStatus").className = "character-profile-status"; $ui("characterProfileStatus").textContent = "The selected local visual profile and its managed images were deleted.";
      } catch (error) { $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = error.message; }
    });
    $ui("characterReferenceImages").addEventListener("change", () => {
      const input = $ui("characterReferenceImages"), files = characterReferenceFilesUi(); let error = "";
      if (files.length > 4) error = "Use at most four new character reference images.";
      else if (files.some((file) => ![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)))) error = "Character references must be PNG, JPEG, or WebP images.";
      else if (files.some((file) => file.size < 1 || file.size > 8 * 1048576)) error = "Each character reference must be between 1 byte and 8 MB.";
      else if (files.reduce((sum, file) => sum + file.size, 0) > 24 * 1048576) error = "New character references may total at most 24 MB.";
      if (error) { input.value = ""; $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = error; }
      else { $ui("characterProfileStatus").className = "character-profile-status"; $ui("characterProfileStatus").textContent = files.length ? "New references are selected. Save the profile to retain private copies across restarts." : "No new character references selected."; }
      renderCharacterReferencePreviewsUi();
    });

    function canvasPngBlobUi(canvas) {
      return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("The browser could not encode this frame as PNG.")), "image/png"));
    }
    function stopPresentedFrameTrackingUi() {
      if (presentedFrameTrackerIdUi !== null && typeof videoUi.cancelVideoFrameCallback === "function") videoUi.cancelVideoFrameCallback(presentedFrameTrackerIdUi);
      presentedFrameTrackerIdUi = null; latestPresentedFrameTimeUi = null;
    }
    function startPresentedFrameTrackingUi() {
      stopPresentedFrameTrackingUi();
      if (typeof videoUi.requestVideoFrameCallback !== "function") return;
      const observe = (_now, metadata) => {
        presentedFrameTrackerIdUi = null;
        if (Number.isFinite(metadata.mediaTime)) latestPresentedFrameTimeUi = metadata.mediaTime;
        if (videoUiUrl) presentedFrameTrackerIdUi = videoUi.requestVideoFrameCallback(observe);
      };
      presentedFrameTrackerIdUi = videoUi.requestVideoFrameCallback(observe);
    }
    function waitForVideoSeekUi(video) {
      if (!video.seeking) return Promise.resolve();
      return new Promise((resolve, reject) => {
        let timer = null;
        const cleanup = () => { video.removeEventListener("seeked", onSeeked); video.removeEventListener("error", onError); if (timer !== null) clearTimeout(timer); };
        const onSeeked = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(new Error("The video could not finish seeking to that frame.")); };
        timer = setTimeout(() => { cleanup(); reject(new Error("The video did not finish seeking. Try that frame again.")); }, 5000);
        video.addEventListener("seeked", onSeeked, {once:true}); video.addEventListener("error", onError, {once:true});
      });
    }
    function displayedVideoTimestampUi(video) {
      const current = video.currentTime;
      return Number.isFinite(latestPresentedFrameTimeUi) && Math.abs(latestPresentedFrameTimeUi - current) <= 0.25 ? latestPresentedFrameTimeUi : current;
    }
    $ui("captureCurrentFrame").addEventListener("click", async () => {
      if (captureUiBusy) return;
      captureUiBusy = true; $ui("captureCurrentFrame").disabled = true; $ui("captureStatus").textContent = "Capturing the displayed frame locally…";
      try {
        if (!videoUiUrl || videoUi.readyState < 2 || !videoUi.videoWidth || !videoUi.videoHeight) throw new Error("Wait until the completed video is visible, then seek to the desired frame.");
        videoUi.pause();
        await waitForVideoSeekUi(videoUi);
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const timestamp = displayedVideoTimestampUi(videoUi), canvas = document.createElement("canvas");
        canvas.width = videoUi.videoWidth; canvas.height = videoUi.videoHeight;
        const context = canvas.getContext("2d", {alpha:false});
        if (!context) throw new Error("This browser cannot capture a video frame.");
        context.drawImage(videoUi, 0, 0, canvas.width, canvas.height);
        const blob = await canvasPngBlobUi(canvas);
        if (blob.size < 1 || blob.size > 8 * 1048576) throw new Error("The captured PNG exceeds the safe 8 MB image limit.");
        const dataUrl = await fileDataUrlUi(blob);
        const result = await apiUi("/api/frame-capture", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({image:dataUrl, source_media:videoUiUrl, timestamp_seconds:timestamp})});
        latestCapturedFrameUrl = result.frame.url; latestCapturedFrameData = dataUrl; latestCapturedFrameAspectUi = result.frame.suggested_aspect_ratio;
        $ui("capturePreview").src = result.frame.url; $ui("downloadCapturedFrame").href = result.frame.url;
        $ui("captureTimestamp").textContent = `Captured ${result.frame.width}×${result.frame.height} at ${result.frame.timestamp_seconds.toFixed(3)}s`;
        $ui("captureResult").hidden = false;
        $ui("captureStatus").textContent = "PNG saved locally. The completed video remains unchanged.";
      } catch (error) { $ui("captureStatus").textContent = error.message; }
      finally { captureUiBusy = false; $ui("captureCurrentFrame").disabled = !captureUiAvailable; }
    });
    $ui("useCapturedStart").addEventListener("click", () => {
      try {
        if (!latestCapturedFrameData || !latestCapturedFrameUrl) throw new Error("Capture a frame first.");
        selectModeUi("generate");
        const input = $ui("startImage"), preview = $ui("startPreview");
        const replacing = Boolean((input.files || []).length || composedFrameAssignments.start);
        if (sharedActiveFilesUi().length + sharedComposedCountUi() + sharedManagedCharacterCountUi() - Number(replacing) >= 4) throw new Error("All four LTX anchor-image slots are already active.");
        input.value = ""; clearImagePreviewUi(input, preview); composedFrameAssignments.start = latestCapturedFrameData;
        const image = document.createElement("img"); image.src = latestCapturedFrameUrl; image.alt = "Captured hand-off frame used as the next clip's start"; preview.appendChild(image);
        $ui("startCopy").textContent = "Captured hand-off frame · local PNG";
        $ui("startDescription").value = ""; $ui("startStrength").value = "1";
        const aspect = $ui("generateAspect");
        if (latestCapturedFrameAspectUi && Array.from(aspect.options).some((option) => option.value === latestCapturedFrameAspectUi)) aspect.value = latestCapturedFrameAspectUi;
        updateAnchorUiSlots(); promptUi.focus();
        $ui("captureStatus").textContent = `Prepared as Generate → Start frame at ${aspect.options[aspect.selectedIndex].text}. Start description was cleared and image strength reset to 1. Write what happens next; trim and stitch the separate clips afterward.`;
      } catch (error) { $ui("captureStatus").textContent = error.message; }
    });

    function voiceLanguageOptionsUi(selected = "en") {
      return Object.entries(voiceLanguagesUi).map(([id, name]) => `<option value="${id}"${id === selected ? " selected" : ""}>${name} · ${id}</option>`).join("");
    }
    $ui("voiceReferenceLanguage").innerHTML = voiceLanguageOptionsUi("en");
    function normalizeVoiceCharacterUi(raw) {
      let value = raw.trim().toLowerCase(); if (value.startsWith("@")) value = value.slice(1);
      if (!/^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$/.test(value)) throw new Error("Character tag must contain 1–64 lowercase letters, digits, hyphens, or underscores.");
      return value;
    }
    function updateVoiceCueCountUi() {
      const count = document.querySelectorAll(".voice-cue").length;
      $ui("voiceCueCounter").textContent = `${count} / 12 cues`;
      $ui("addVoiceCue").disabled = count >= 12;
    }
    function addVoiceCueUi() {
      if (document.querySelectorAll(".voice-cue").length >= 12) return;
      const id = ++voiceUiSerial, card = document.createElement("div"); card.className = "reference-card voice-cue";
      card.innerHTML = `<div class="reference-head"><strong>Cue ${id} · one speaker</strong><button class="remove-anchor" type="button">Remove</button></div><div class="field-grid" style="margin-top:9px"><div><label>Assigned @character</label><input class="voice-cue-character" type="text" maxlength="65" autocomplete="off" placeholder="@maya"></div><div><label>Language</label><select class="voice-cue-language">${voiceLanguageOptionsUi("en")}</select></div><div><label>Start time · seconds</label><input class="voice-cue-start" type="number" min="0" max="86400" step="0.01" value="0"></div><div><label>Seed</label><input class="voice-cue-seed" type="number" min="0" max="4294967295" step="1" value="1234"></div><div class="wide"><label>Spoken text</label><textarea class="voice-cue-text" maxlength="1600" placeholder="We should leave before the storm arrives."></textarea></div><div class="wide"><label>Delivery note <span style="color:#71747e;font-weight:400">(optional; not speaker identity)</span></label><input class="voice-cue-description" type="text" maxlength="500" placeholder="calm but increasingly urgent"></div></div>`;
      card.querySelector(".remove-anchor").addEventListener("click", () => { card.remove(); updateVoiceCueCountUi(); });
      $ui("voiceCueList").appendChild(card); updateVoiceCueCountUi();
    }
    $ui("addVoiceCue").addEventListener("click", addVoiceCueUi);
    function voiceCuesUi() {
      const cards = Array.from(document.querySelectorAll(".voice-cue"));
      if (!cards.length) throw new Error("Add at least one dialogue cue.");
      return cards.map((card, index) => {
        const character = normalizeVoiceCharacterUi(card.querySelector(".voice-cue-character").value);
        const text = card.querySelector(".voice-cue-text").value.trim();
        if (!text) throw new Error(`Cue ${index + 1} needs spoken text.`);
        const start_seconds = Number(card.querySelector(".voice-cue-start").value);
        if (!Number.isFinite(start_seconds) || start_seconds < 0 || start_seconds > 86400) throw new Error(`Cue ${index + 1} start time must be from 0 to 86400 seconds.`);
        const seed = Number(card.querySelector(".voice-cue-seed").value);
        if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295) throw new Error(`Cue ${index + 1} seed must be a whole number from 0 to 4294967295.`);
        return {character, text, language:card.querySelector(".voice-cue-language").value, start_seconds, voice_description:card.querySelector(".voice-cue-description").value, seed};
      });
    }
    $ui("registerVoicePreset").addEventListener("click", async () => {
      voiceNoticeUi.textContent = "";
      try {
        const file = requiredFileUi("voiceReferenceFile", "Canonical voice reference", [".wav"], 50 * 1048576);
        const consent = $ui("voiceConsent").checked;
        if (!consent) throw new Error("Confirm that you own or are authorized to clone this reference voice.");
        $ui("registerVoicePreset").disabled = true;
        await apiUi("/api/voice/presets", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({name:$ui("voicePresetName").value, reference:await fileDataUrlUi(file), reference_language:$ui("voiceReferenceLanguage").value, voice_description:$ui("voicePresetDescription").value, consent_confirmed:true, force:$ui("voiceReplacePreset").checked})});
        voiceNoticeUi.textContent = "Voice preset registered locally. Its canonical WAV—not prose—is the identity anchor.";
        $ui("voiceReferenceFile").value = ""; $ui("voiceConsent").checked = false; $ui("voiceReplacePreset").checked = false;
        await refreshVoiceUi();
      } catch (error) { voiceNoticeUi.textContent = error.message; }
      finally { $ui("registerVoicePreset").disabled = false; }
    });
    $ui("assignVoicePreset").addEventListener("click", async () => {
      voiceNoticeUi.textContent = "";
      try {
        const character = normalizeVoiceCharacterUi($ui("voiceCharacter").value), preset = $ui("voicePresetSelect").value;
        if (!preset) throw new Error("Register and select a voice preset first.");
        $ui("assignVoicePreset").disabled = true;
        await apiUi("/api/voice/assign", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({character, preset, force:$ui("voiceReplaceAssignment").checked})});
        $ui("voiceCharacter").value = `@${character}`; $ui("voiceReplaceAssignment").checked = false;
        voiceNoticeUi.textContent = `Assigned @${character} to ${preset}.`;
        await refreshVoiceUi();
      } catch (error) { voiceNoticeUi.textContent = error.message; }
      finally { $ui("assignVoicePreset").disabled = false; }
    });
    synthesizeVoicesUi.addEventListener("click", async () => {
      voiceNoticeUi.textContent = "";
      try {
        if (!voiceBackendReady) throw new Error("Install and verify the optional character-voice backend before synthesis.");
        synthesizeVoicesUi.disabled = true;
        await apiUi("/api/voice/synthesize", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({cues:voiceCuesUi(), mux_latest:$ui("voiceMuxLatest").checked})});
        await refreshVoiceUi();
      } catch (error) { voiceNoticeUi.textContent = error.message; synthesizeVoicesUi.disabled = !voiceBackendReady; }
    });
    cancelVoicesUi.addEventListener("click", async () => {
      try { await apiUi("/api/voice/cancel", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"}); await refreshVoiceUi(); }
      catch (error) { voiceNoticeUi.textContent = error.message; }
    });
    function renderVoiceResultsUi(job) {
      const target = $ui("voiceResults"); target.replaceChildren();
      for (const cue of (job?.cues || [])) {
        if (!cue.audio_url) continue;
        const row = document.createElement("div"); row.className = "voice-result";
        const label = document.createElement("span"); label.textContent = `@${cue.character} · ${cue.language} · ${cue.start_seconds.toFixed(2)}s · ${cue.duration_seconds.toFixed(2)}s`;
        const audio = document.createElement("audio"); audio.controls = true; audio.preload = "metadata"; audio.src = cue.audio_url;
        const link = document.createElement("a"); link.className = "download"; link.href = cue.audio_url; link.download = ""; link.textContent = "WAV";
        row.append(label, audio, link); target.appendChild(row);
      }
      if (job?.muxed_video_url) {
        const row = document.createElement("div"); row.className = "voice-result";
        const label = document.createElement("span"); label.textContent = "Timed dialogue mix · original video preserved";
        const link = document.createElement("a"); link.className = "download"; link.href = job.muxed_video_url; link.download = ""; link.textContent = "Download voiced MP4";
        row.append(label, link); target.appendChild(row);
      }
    }
    function renderVoiceUi(data) {
      const readiness = data.readiness || {}, job = data.job || null, registry = data.registry || {presets:[],characters:{}};
      const active = Boolean(job && activeStates.has(job.status)); voiceBackendReady = Boolean(readiness.ready);
      voiceReadinessUi.className = `voice-status ${readiness.ready ? "ready" : "needs"}`;
      voiceReadinessUi.textContent = `${readiness.ready ? "● Ready" : "● Setup needed"} — ${readiness.note || "Optional voice setup is incomplete."}`;
      const select = $ui("voicePresetSelect"), previous = select.value; select.replaceChildren();
      if (!(registry.presets || []).length) { const option = document.createElement("option"); option.value = ""; option.textContent = "Register a preset first"; select.appendChild(option); }
      for (const preset of (registry.presets || [])) { const option = document.createElement("option"); option.value = preset.name; option.textContent = `${preset.name} · ${preset.reference_language}`; select.appendChild(option); }
      if (Array.from(select.options).some((option) => option.value === previous)) select.value = previous;
      const assignments = Object.entries(registry.characters || {});
      $ui("voiceAssignments").textContent = assignments.length ? `Assignments: ${assignments.map(([character, preset]) => `@${character} → ${preset}`).join(" · ")}` : "No character assignments yet.";
      $ui("registerVoicePreset").disabled = active || !readiness.registry_ready;
      $ui("assignVoicePreset").disabled = active || !readiness.registry_ready || !(registry.presets || []).length;
      synthesizeVoicesUi.disabled = active || !readiness.ready; cancelVoicesUi.hidden = !active; cancelVoicesUi.disabled = job?.status === "cancelling";
      $ui("voiceMuxLatest").disabled = active || !readiness.mux_ready;
      if (!readiness.mux_ready) $ui("voiceMuxLatest").checked = false;
      if (job) {
        const progress = job.progress_percent == null ? "" : ` · ${job.progress_percent.toFixed(1)}%`;
        $ui("voiceJobProgress").textContent = `${job.stage || job.status}${progress}${job.current_cue ? ` · ${job.current_cue}` : ""}`;
        if (job.error) voiceNoticeUi.textContent = job.error;
        else if (job.status === "succeeded") voiceNoticeUi.textContent = job.muxed_video_url ? "Voice cues and a new voiced MP4 are ready." : "Voice cues are ready.";
        renderVoiceResultsUi(job);
      }
    }
    async function refreshVoiceUi() {
      if (voicePollBusy) return; voicePollBusy = true;
      try { renderVoiceUi(await apiUi("/api/voice/status", {cache:"no-store"})); }
      catch (error) { voiceReadinessUi.textContent = `Character Voice status unavailable: ${error.message}`; }
      finally { voicePollBusy = false; }
    }

    function bindDurationUi(prefix) {
      const input = $ui(`${prefix}Duration`), value = $ui(`${prefix}DurationValue`), warning = $ui(`${prefix}DurationWarning`);
      const update = () => {
        const seconds = Number(input.value), frames = 24 * seconds + 1;
        value.textContent = `${seconds}s → ${frames} frames · ~${(frames / 24).toFixed(2)}s encoded`;
        warning.hidden = seconds <= 5;
        if (prefix === "generate") updateSceneUiSlots();
      };
      input.addEventListener("input", update); update();
    }

    function clearImagePreviewUi(input, container) {
      (previewUiUrls.get(input) || []).forEach((url) => URL.revokeObjectURL(url));
      previewUiUrls.delete(input); container.replaceChildren();
    }
    function renderImagePreviewUi(input, container, copy) {
      clearImagePreviewUi(input, container);
      let files = Array.from(input.files || []);
      let error = "";
      if (files.some((file) => ![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)))) error = "Every reference must be a PNG, JPEG, or WebP image.";
      else if (files.some((file) => file.size < 1 || file.size > 8 * 1048576)) error = "Every reference image must be between 1 byte and 8 MB.";
      if (error) { input.value = ""; files = []; noticeUi.textContent = error; }
      const urls = [];
      files.forEach((file) => { const url = URL.createObjectURL(file), image = document.createElement("img"); urls.push(url); image.src = url; image.alt = "Selected reference preview"; container.appendChild(image); });
      previewUiUrls.set(input, urls);
      if (copy) copy.textContent = files.length ? files.map((file) => file.name).join(", ") : "PNG, JPEG, or WebP · max 8 MB";
      updateAnchorUiSlots();
    }
    function sharedImageInputsUi() { return [$ui("startImage"), $ui("endImage"), ...document.querySelectorAll(".anchor-images"), $ui("characterReferenceImages")]; }
    function sharedActiveFilesUi() {
      if (!["generate", "ingredients", "union", "motion"].includes(modeUi)) return [];
      return sharedImageInputsUi().flatMap((input) => Array.from(input.files || []));
    }
    function sharedComposedCountUi() {
      if (!["generate", "ingredients", "union", "motion"].includes(modeUi)) return 0;
      return Number(Boolean(composedFrameAssignments.start)) + Number(Boolean(composedFrameAssignments.end))
        + Array.from(document.querySelectorAll(".anchor-card")).filter((card) => Boolean(card._composedFrameData)).length;
    }
    function sharedManagedCharacterCountUi() {
      return ["generate", "ingredients", "union", "motion"].includes(modeUi) ? managedCharacterReferences.length : 0;
    }
    function updateAnchorUiSlots() {
      const count = sharedActiveFilesUi().length + sharedComposedCountUi() + sharedManagedCharacterCountUi();
      $ui("slotCounter").textContent = `${count} / 4 active images`;
      $ui("slotCounter").classList.toggle("over", count > 4);
      $ui("addAnchor").disabled = document.querySelectorAll(".anchor-card").length >= 4;
    }
    function addAnchorUi() {
      if (document.querySelectorAll(".anchor-card").length >= 4) return;
      const id = ++anchorUiSerial, card = document.createElement("div");
      card.className = "anchor-card";
      card.innerHTML = `<div class="anchor-head"><h3>Anchor group ${id}</h3><button class="remove-anchor" type="button">Remove</button></div><div class="compact-grid"><div><label>Group name</label><input class="anchor-name" type="text" maxlength="80" placeholder="Hero / vehicle / location"></div><div><label>Image strength</label><input class="anchor-strength" type="number" min="0" max="1" step="0.05" value="0.75"></div></div><label style="margin-top:9px">Reference views</label><label class="upload"><span class="upload-copy">Select one or more views</span><input class="anchor-images" type="file" multiple accept="image/png,image/jpeg,image/webp"></label><div class="previews"></div><label style="margin-top:9px">Description</label><textarea class="anchor-description" maxlength="500" placeholder="Appearance or composition at this moment…"></textarea>`;
      const input = card.querySelector(".anchor-images"), previews = card.querySelector(".previews"), copy = card.querySelector(".upload-copy");
      input.addEventListener("change", () => { card._composedFrameData = null; renderImagePreviewUi(input, previews, copy); });
      card.querySelector(".remove-anchor").addEventListener("click", () => { clearImagePreviewUi(input, previews); card._composedFrameData = null; card.remove(); updateAnchorUiSlots(); });
      $ui("anchorList").appendChild(card); updateAnchorUiSlots();
      return card;
    }
    $ui("startImage").addEventListener("change", () => { composedFrameAssignments.start = null; renderImagePreviewUi($ui("startImage"), $ui("startPreview"), $ui("startCopy")); });
    $ui("endImage").addEventListener("change", () => { composedFrameAssignments.end = null; renderImagePreviewUi($ui("endImage"), $ui("endPreview"), $ui("endCopy")); });
    $ui("addAnchor").addEventListener("click", addAnchorUi);

    function sceneCapacityUi() { return Math.ceil((24 * Number($ui("generateDuration").value) + 1) / 8); }
    function updateSceneUiSlots() {
      const count = document.querySelectorAll(".scene-card").length, capacity = sceneCapacityUi();
      $ui("sceneCounter").textContent = `${count} beat${count === 1 ? "" : "s"} · ${capacity} latent frames`;
      $ui("sceneCounter").classList.toggle("over", count > capacity);
      $ui("addScene").disabled = count >= capacity;
      document.querySelectorAll(".scene-length").forEach((input) => { input.max = String(capacity); });
    }
    function addSceneUi() {
      const capacity = sceneCapacityUi(); if (document.querySelectorAll(".scene-card").length >= capacity) return;
      const id = ++sceneUiSerial, card = document.createElement("div"); card.className = "scene-card";
      card.innerHTML = `<div class="scene-head"><h3>Scene beat ${id}</h3><button class="remove-scene" type="button">Remove</button></div><label>Local action or visual change</label><textarea class="scene-text" maxlength="1000" placeholder="The subject turns toward the window…"></textarea><div style="max-width:190px;margin-top:9px"><label>Length · latent frames</label><input class="scene-length" type="number" min="1" max="${capacity}" step="1" placeholder="Auto"></div>`;
      card.querySelector(".remove-scene").addEventListener("click", () => { card.remove(); updateSceneUiSlots(); });
      $ui("sceneList").appendChild(card); updateSceneUiSlots();
    }
    $ui("addScene").addEventListener("click", addSceneUi);

    function addIngredientUi() {
      if (document.querySelectorAll(".ingredient-reference").length >= 8) return;
      const id = ++ingredientUiSerial, card = document.createElement("div"); card.className = "reference-card ingredient-reference";
      card.innerHTML = `<div class="reference-head"><strong>Reference ${id}</strong><button class="remove-anchor" type="button">Remove</button></div><div class="field-grid" style="margin-top:9px"><div><label>Image</label><input class="ingredient-image" type="file" accept="image/png,image/jpeg,image/webp"></div><div><label>Stable prompt tag</label><input class="ingredient-tag" type="text" maxlength="33" placeholder="@maya"></div><div class="wide"><label>Display label</label><input class="ingredient-label" type="text" maxlength="100" placeholder="Maya / red car / rooftop"></div><div class="wide"><label>Description</label><textarea class="ingredient-description" maxlength="500" placeholder="Identity, materials, colors, or features to preserve…"></textarea></div></div>`;
      card.querySelector(".remove-anchor").addEventListener("click", () => { card.remove(); updateIngredientUiCount(); });
      $ui("ingredientList").appendChild(card); updateIngredientUiCount();
    }
    function updateIngredientUiCount() { const count = document.querySelectorAll(".ingredient-reference").length; $ui("ingredientCounter").textContent = `${count} / 8 references`; $ui("addIngredient").disabled = count >= 8; }
    $ui("addIngredient").addEventListener("click", addIngredientUi); addIngredientUi();

    function updateFrameComposerReferenceCountUi() {
      const cards = Array.from(document.querySelectorAll(".frame-composer-reference"));
      const count = cards.reduce((total, card) => total + card.querySelector(".frame-composer-image").files.length, 0);
      $ui("frameComposerCounter").textContent = `${count} / 4 active photos · ${cards.length} ingredient${cards.length === 1 ? "" : "s"}`;
      $ui("frameComposerCounter").classList.toggle("over", count > 4);
      $ui("addFrameComposerReference").disabled = cards.length >= 4 || count >= 4;
    }
    function addFrameComposerReferenceUi() {
      if (document.querySelectorAll(".frame-composer-reference").length >= 4) return;
      const id = ++frameComposerUiSerial, card = document.createElement("div");
      card.className = "reference-card frame-composer-reference";
      card.innerHTML = `<div class="reference-head"><strong>Ingredient ${id}</strong><button class="remove-anchor" type="button">Remove</button></div><div class="field-grid" style="margin-top:9px"><div><label>Stable prompt tag</label><input class="frame-composer-tag" type="text" maxlength="33" placeholder="@maya"></div><div><label>Reference views</label><label class="upload"><span class="upload-copy">Select one or more views</span><input class="frame-composer-image" type="file" multiple accept="image/png,image/jpeg,image/webp"></label><div class="previews"></div></div></div>`;
      const input = card.querySelector(".frame-composer-image"), previews = card.querySelector(".previews"), copy = card.querySelector(".upload-copy");
      input.addEventListener("change", () => {
        clearImagePreviewUi(input, previews);
        let files = Array.from(input.files || []), error = "";
        if (files.some((file) => ![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)))) error = "Every Frame Composer view must be a PNG, JPEG, or WebP image.";
        else if (files.some((file) => file.size < 1 || file.size > 8 * 1048576)) error = "Every Frame Composer view must be between 1 byte and 8 MB.";
        if (error) { input.value = ""; files = []; frameComposerNoticeUi.textContent = error; }
        const urls = [];
        files.forEach((file) => { const url = URL.createObjectURL(file), image = document.createElement("img"); urls.push(url); image.src = url; image.alt = "Selected ingredient view"; previews.appendChild(image); });
        previewUiUrls.set(input, urls);
        copy.textContent = files.length ? `${files.length} view${files.length === 1 ? "" : "s"} selected` : "Select one or more views";
        updateFrameComposerReferenceCountUi();
      });
      card.querySelector(".remove-anchor").addEventListener("click", () => { clearImagePreviewUi(input, previews); card.remove(); updateFrameComposerReferenceCountUi(); });
      $ui("frameComposerReferences").appendChild(card); updateFrameComposerReferenceCountUi();
    }
    $ui("addFrameComposerReference").addEventListener("click", addFrameComposerReferenceUi); addFrameComposerReferenceUi();

    function frameComposerTargetUi() {
      let targetMode = ["generate", "ingredients", "union", "motion"].includes(modeUi) ? modeUi : "generate";
      let profile, aspectRatio;
      if (targetMode === "generate") {
        profile = document.querySelector('input[name="profile"]:checked').value;
        aspectRatio = $ui("generateAspect").value;
      } else if (targetMode === "ingredients") {
        profile = $ui("ingredientsProfile").value;
        aspectRatio = $ui("ingredientsAspect").value;
      } else {
        profile = $ui(`${targetMode}Profile`).value;
        aspectRatio = $ui(`${targetMode}Aspect`).value;
      }
      $ui("frameComposerProfile").value = profile;
      $ui("frameComposerCanvas").textContent = `Matches ${targetMode} · ${profile} · ${aspectRatio === "profile" ? "profile aspect" : aspectRatio}.`;
      return {target_mode:targetMode, profile, aspect_ratio:aspectRatio};
    }
    document.querySelectorAll('input[name="profile"]').forEach((input) => input.addEventListener("change", frameComposerTargetUi));
    ["generateAspect", "ingredientsProfile", "ingredientsAspect", "unionProfile", "unionAspect", "motionProfile", "motionAspect"].forEach((id) => $ui(id).addEventListener("change", frameComposerTargetUi));

    async function frameComposerReferencesUi() {
      const references = [], tags = new Set();
      for (const [index, card] of Array.from(document.querySelectorAll(".frame-composer-reference")).entries()) {
        const files = Array.from(card.querySelector(".frame-composer-image").files || []);
        let tag = card.querySelector(".frame-composer-tag").value.trim().toLowerCase();
        if (!tag.startsWith("@")) tag = `@${tag}`;
        if (!files.length || !/^@[a-z][a-z0-9_]{0,31}$/.test(tag)) throw new Error(`Ingredient ${index + 1} needs at least one view and a tag such as @maya.`);
        if (tags.has(tag)) throw new Error(`Ingredient tag ${tag} is already used; add those views to the same ingredient.`);
        tags.add(tag); card.querySelector(".frame-composer-tag").value = tag;
        for (const file of files) {
          if (![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)) || file.size < 1 || file.size > 8 * 1048576) throw new Error(`Every view for ${tag} must be a PNG, JPEG, or WebP no larger than 8 MB.`);
          references.push({tag, image:await fileDataUrlUi(file)});
        }
      }
      if (!references.length || references.length > 4) throw new Error("Frame Composer requires 1–4 active photos total across all ingredients.");
      return references;
    }
    composeFrameUi.addEventListener("click", async () => {
      frameComposerNoticeUi.textContent = ""; composeFrameUi.disabled = true;
      try {
        const target = frameComposerTargetUi();
        const payload = {
          prompt:$ui("frameComposerPrompt").value,
          target_mode:target.target_mode,
          profile:target.profile,
          aspect_ratio:target.aspect_ratio,
          seed:readNumberUi("frameComposerSeed", 0, 4294967295, "Frame seed", true),
          references:await frameComposerReferencesUi(),
        };
        await apiUi("/api/frame-composer", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        latestComposedFrameUrl = null; latestComposedFrameData = null; frameComposerResultUi.hidden = true;
        await refreshFrameComposerUi();
      } catch (error) { frameComposerNoticeUi.textContent = error.message; composeFrameUi.disabled = false; }
    });
    cancelFrameComposerUi.addEventListener("click", async () => {
      try { await apiUi("/api/frame-composer/cancel", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"}); await refreshFrameComposerUi(); }
      catch (error) { frameComposerNoticeUi.textContent = error.message; }
    });
    async function ensureComposedFrameDataUi() {
      if (latestComposedFrameData) return latestComposedFrameData;
      if (!latestComposedFrameUrl) throw new Error("Compose a frame first.");
      const response = await fetch(latestComposedFrameUrl, {cache:"no-store"});
      if (!response.ok) throw new Error("Could not load the composed frame.");
      const blob = await response.blob();
      if (blob.size < 1 || blob.size > 8 * 1048576 || blob.type !== "image/png") throw new Error("Composed frame is not a safe PNG under 8 MB.");
      latestComposedFrameData = await fileDataUrlUi(blob); return latestComposedFrameData;
    }
    async function assignComposedBoundaryUi(prefix) {
      const data = await ensureComposedFrameDataUi(), input = $ui(`${prefix}Image`), preview = $ui(`${prefix}Preview`);
      const replacing = Boolean((input.files || []).length || composedFrameAssignments[prefix]);
      if (sharedActiveFilesUi().length + sharedComposedCountUi() + sharedManagedCharacterCountUi() - Number(replacing) >= 4) throw new Error("All four LTX anchor-image slots are already active.");
      input.value = ""; clearImagePreviewUi(input, preview); composedFrameAssignments[prefix] = data;
      const image = document.createElement("img"); image.src = latestComposedFrameUrl; image.alt = "Composed frame anchor"; preview.appendChild(image);
      $ui(`${prefix}Copy`).textContent = "Composed frame · local PNG";
      if (!$ui(`${prefix}Description`).value.trim()) $ui(`${prefix}Description`).value = $ui("frameComposerPrompt").value.trim();
      updateAnchorUiSlots(); frameComposerNoticeUi.textContent = `Assigned composed frame to ${prefix}.`;
    }
    $ui("assignComposedStart").addEventListener("click", async () => { try { await assignComposedBoundaryUi("start"); } catch (error) { frameComposerNoticeUi.textContent = error.message; } });
    $ui("assignComposedEnd").addEventListener("click", async () => { try { await assignComposedBoundaryUi("end"); } catch (error) { frameComposerNoticeUi.textContent = error.message; } });
    $ui("assignComposedInterior").addEventListener("click", async () => {
      try {
        if (sharedActiveFilesUi().length + sharedComposedCountUi() + sharedManagedCharacterCountUi() >= 4) throw new Error("All four LTX anchor-image slots are already active.");
        const data = await ensureComposedFrameDataUi(), card = addAnchorUi();
        if (!card) throw new Error("Remove an interior anchor group before adding another.");
        card._composedFrameData = data; card.querySelector(".anchor-name").value = "Composed frame";
        card.querySelector(".anchor-description").value = $ui("frameComposerPrompt").value.trim();
        const preview = card.querySelector(".previews"), image = document.createElement("img"); image.src = latestComposedFrameUrl; image.alt = "Composed interior anchor"; preview.appendChild(image);
        card.querySelector(".upload-copy").textContent = "Composed frame · local PNG";
        updateAnchorUiSlots(); frameComposerNoticeUi.textContent = "Added composed frame as an interior anchor.";
      } catch (error) { frameComposerNoticeUi.textContent = error.message; }
    });

    function renderFrameComposerUi(data) {
      const readiness = data.readiness || {}, job = data.job || null, active = Boolean(job && ["starting","running","cancelling"].includes(job.status));
      const readinessNode = $ui("frameComposerReadiness"); readinessNode.className = `frame-composer-status ${readiness.ready ? "ready" : "needs"}`;
      readinessNode.textContent = `${readiness.ready ? "● Ready" : "● Setup needed"} — ${readiness.note || "Optional isolated runtime and Q4 model are not installed."}`;
      composeFrameUi.disabled = active || !readiness.ready; cancelFrameComposerUi.hidden = !active;
      if (job) {
        const detail = job.error || job.log_tail || `${job.status}…`; frameComposerNoticeUi.textContent = detail;
        if (job.status === "succeeded" && job.output_url) {
          latestComposedFrameUrl = job.output_url; latestComposedFrameData = null;
          $ui("frameComposerImage").src = job.output_url; $ui("downloadComposedFrame").href = job.output_url; frameComposerResultUi.hidden = false;
        }
      }
    }
    async function refreshFrameComposerUi() {
      if (frameComposerPollBusy) return; frameComposerPollBusy = true;
      try { renderFrameComposerUi(await apiUi("/api/frame-composer/status", {cache:"no-store"})); }
      catch (error) { $ui("frameComposerReadiness").textContent = `Frame Composer status unavailable: ${error.message}`; }
      finally { frameComposerPollBusy = false; }
    }

    async function frameUi(prefix) {
      const file = optionalImageUi(`${prefix}Image`, `${prefix === "start" ? "Start" : "End"} image`);
      const image = file ? await fileDataUrlUi(file) : composedFrameAssignments[prefix];
      return { image:image || null, description:$ui(`${prefix}Description`).value, strength:readNumberUi(`${prefix}Strength`, 0, 1, `${prefix} strength`) };
    }
    async function anchorGroupsUi() {
      const groups = [];
      for (const card of document.querySelectorAll(".anchor-card")) {
        const name = card.querySelector(".anchor-name").value, description = card.querySelector(".anchor-description").value;
        const files = Array.from(card.querySelector(".anchor-images").files || []);
        const composed = card._composedFrameData || null;
        if (!name.trim() && !description.trim() && !files.length && !composed) continue;
        files.forEach((file) => { if (![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)) || file.size > 8 * 1048576 || file.size < 1) throw new Error("Every timeline anchor must be a PNG, JPEG, or WebP no larger than 8 MB."); });
        const images = await Promise.all(files.map(fileDataUrlUi)); if (composed) images.unshift(composed);
        groups.push({ name, description, strength:Number(card.querySelector(".anchor-strength").value), images });
      }
      return groups;
    }
    function scenesUi() {
      const scenes = [], capacity = sceneCapacityUi();
      for (const card of document.querySelectorAll(".scene-card")) {
        const text = card.querySelector(".scene-text").value.trim(), raw = card.querySelector(".scene-length").value.trim();
        if (!text && !raw) continue; if (!text) throw new Error("Every scene beat with a length needs prompt text.");
        const length = raw ? Number(raw) : null;
        if (length !== null && (!Number.isSafeInteger(length) || length < 1 || length > capacity)) throw new Error(`Every scene length must be from 1 to ${capacity} latent frames.`);
        scenes.push({ text, length });
      }
      const automatic = scenes.filter((scene) => scene.length === null).length, pinned = scenes.reduce((sum, scene) => sum + (scene.length || 0), 0);
      if (scenes.length > capacity || pinned > capacity || (automatic && capacity - pinned < automatic)) throw new Error(`Scene beats must fit within ${capacity} latent frames.`);
      return scenes;
    }
    function characterUi(payload) {
      payload.character_name = $ui("characterName").value; payload.character_appearance = $ui("characterAppearance").value;
      payload.character_wardrobe = $ui("characterWardrobe").value; payload.character_continuity = $ui("characterContinuity").value; payload.character_avoid = $ui("characterAvoid").value;
    }
    async function characterReferencesUi(payload) {
      const files = characterReferenceFilesUi();
      if (managedCharacterReferences.length && activeCharacterProfileId) payload.character_profile_id = activeCharacterProfileId;
      if (files.length) payload.character_reference_images = await Promise.all(files.map(fileDataUrlUi));
    }
    async function sharedConditionUi(payload) {
      const files = sharedActiveFilesUi();
      if (files.length + sharedComposedCountUi() + sharedManagedCharacterCountUi() > 4) throw new Error("Use at most 4 active images across Character Bible references, start, end, and timeline anchors.");
      const managedBytes = managedCharacterReferences.reduce((sum, reference) => sum + Number(reference.size || 0), 0);
      if (files.reduce((sum, file) => sum + file.size, 0) + managedBytes > 24 * 1048576) throw new Error("Character and timeline anchor images may total at most 24 MB.");
      payload.start_frame = await frameUi("start"); payload.end_frame = await frameUi("end"); payload.ingredients = await anchorGroupsUi();
      await characterReferencesUi(payload);
    }
    async function ingredientReferencesUi() {
      const cards = Array.from(document.querySelectorAll(".ingredient-reference")); if (!cards.length) throw new Error("Add at least one Ingredients reference.");
      const references = [], tags = new Set();
      for (const [index, card] of cards.entries()) {
        const file = card.querySelector(".ingredient-image").files[0] || null, label = card.querySelector(".ingredient-label").value.trim(), description = card.querySelector(".ingredient-description").value.trim();
        let tag = card.querySelector(".ingredient-tag").value.trim().toLowerCase(); if (tag && !tag.startsWith("@")) tag = `@${tag}`;
        if (!file) throw new Error(`Ingredient reference ${index + 1} needs an image.`);
        if (![".png", ".jpg", ".jpeg", ".webp"].includes(extensionUi(file.name)) || file.size < 1 || file.size > 8 * 1048576) throw new Error(`Ingredient reference ${index + 1} image must be PNG, JPEG, or WebP up to 8 MB.`);
        if (!label || !tag || !description) throw new Error(`Ingredient reference ${index + 1} needs a tag, display label, and description.`);
        if (!/^@[a-z][a-z0-9_]{0,31}$/.test(tag)) throw new Error(`Ingredient reference ${index + 1} tag must look like @maya and use lowercase letters, numbers, or underscores.`);
        if (tags.has(tag)) throw new Error(`Ingredient tag ${tag} must be unique.`); tags.add(tag);
        card.querySelector(".ingredient-tag").value = tag;
        references.push({ image:await fileDataUrlUi(file), tag, label, description });
      }
      return references;
    }

    function updateUnionUi() {
      const source = $ui("unionRoute").value === "source";
      if (source) $ui("unionControlType").value = "canny";
      $ui("unionControlType").disabled = source;
    }
    $ui("unionRoute").addEventListener("change", updateUnionUi); updateUnionUi();

    function displayReadinessUi() {
      const target = $ui("modeReadiness"), item = readinessByMode[modeUi];
      if (!item) { target.className = "readiness"; target.textContent = "Checking local workflow readiness…"; return; }
      target.className = `readiness ${item.ready ? "ready" : "needs"}`;
      target.textContent = `${item.ready ? "● Ready" : item.blocked ? "● Unavailable" : "● Setup needed"}${item.experimental ? " · Experimental" : ""} — ${item.note}`;
      submitUi.disabled = tutorialSessionActiveUi || renderUiBusy || Boolean(item.blocked);
    }
    function selectModeUi(mode) {
      modeUi = mode;
      document.querySelectorAll("[data-select-mode]").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.selectMode === mode)));
      document.querySelectorAll("[data-mode-panel]").forEach((panel) => { panel.hidden = panel.dataset.modePanel !== mode; });
      document.querySelectorAll("[data-modes]").forEach((element) => { element.hidden = !element.dataset.modes.split(" ").includes(mode); });
      submitUi.textContent = tutorialSessionActiveUi ? "Finish learning mode before rendering" : buttonLabels[mode]; $ui("promptLabel").textContent = promptLabels[mode];
      updateAnchorUiSlots(); updateSceneUiSlots(); frameComposerTargetUi(); displayReadinessUi(); noticeUi.textContent = "";
    }
    document.querySelectorAll("[data-select-mode]").forEach((button) => button.addEventListener("click", () => selectModeUi(button.dataset.selectMode)));

    bindDurationUi("generate"); bindDurationUi("a2v"); bindDurationUi("union"); bindDurationUi("motion");
    promptUi.addEventListener("input", () => { $ui("count").textContent = promptUi.value.length.toLocaleString(); });
    promptCoachInputUi.addEventListener("input", () => { $ui("promptCoachCount").textContent = promptCoachInputUi.value.length.toLocaleString(); });
    $ui("frameComposerPrompt").addEventListener("input", () => { $ui("frameComposerCount").textContent = $ui("frameComposerPrompt").value.length.toLocaleString(); });
    $ui("randomSeed").addEventListener("click", () => { const values = new Uint32Array(1); crypto.getRandomValues(values); $ui("seed").value = values[0]; });

    formUi.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (tutorialSessionActiveUi) { noticeUi.textContent = "Learning mode never starts a render. Finish or close the lesson, review the prepared setup, then press Generate when you choose."; submitUi.disabled = true; return; }
      noticeUi.textContent = ""; submitUi.disabled = true;
      try {
        const readiness = readinessByMode[modeUi];
        if (readiness && readiness.blocked) throw new Error(readiness.note);
        const seed = readNumberUi("seed", 0, 4294967295, "Seed", true);
        const payload = { mode:modeUi, prompt:promptUi.value, seed }; characterUi(payload);
        if (modeUi === "generate") {
          payload.profile = document.querySelector('input[name="profile"]:checked').value;
          payload.aspect_ratio = $ui("generateAspect").value;
          payload.gemma_pack = $ui("gemmaPack").value;
          payload.duration_seconds = readNumberUi("generateDuration", 1, 10, "Duration", true);
          payload.scenes = scenesUi(); payload.generated_keyframes = 0;
          await sharedConditionUi(payload);
        } else if (modeUi === "retake") {
          payload.source_video = await uploadUi(requiredFileUi("retakeVideo", "Source video", [".mp4",".mov",".m4v",".mkv",".avi",".webm"], 64 * 1048576));
          payload.start_seconds = readNumberUi("retakeStart", 0, 86400, "Retake start"); payload.end_seconds = readNumberUi("retakeEnd", 0, 86400, "Retake end"); payload.preserve_audio = $ui("retakePreserveAudio").checked;
        } else if (modeUi === "extend") {
          payload.source_video = await uploadUi(requiredFileUi("extendVideo", "Source video", [".mp4",".mov",".m4v",".mkv",".avi",".webm"], 64 * 1048576));
          payload.extend_seconds = Number($ui("extendSeconds").value); payload.direction = $ui("extendDirection").value;
        } else if (modeUi === "a2v") {
          payload.audio = await uploadUi(requiredFileUi("a2vAudio", "Audio input", [".wav",".mp3",".m4a",".aac",".flac",".ogg",".opus",".aif",".aiff",".caf"], 64 * 1048576));
          payload.audio_start = readNumberUi("a2vAudioStart", 0, 86400, "Audio start"); payload.profile = $ui("a2vProfile").value; payload.aspect_ratio = $ui("a2vAspect").value; payload.duration_seconds = readNumberUi("a2vDuration", 1, 10, "Duration", true);
          const start = optionalImageUi("a2vStartImage", "A2V start image"); payload.start_image = start ? await fileDataUrlUi(start) : null; payload.image_strength = readNumberUi("a2vImageStrength", 0, 1, "Image strength");
        } else {
          payload.profile = $ui(`${modeUi}Profile`).value; payload.duration_seconds = modeUi === "ingredients" ? 5 : readNumberUi(`${modeUi}Duration`, 1, 10, "Duration", true);
          payload.aspect_ratio = $ui(`${modeUi}Aspect`).value;
          payload.lora_strength = readNumberUi(`${modeUi}Lora`, 0, 2, "LoRA strength"); payload.conditioning_strength = readNumberUi(`${modeUi}Conditioning`, 0, 1, "Guide strength");
          await sharedConditionUi(payload);
          if (modeUi === "ingredients") { payload.ingredient_references = await ingredientReferencesUi(); payload.dialogue_guidance = $ui("ingredientDialogue").value; }
          else if (modeUi === "union") {
            payload.union_route = $ui("unionRoute").value; payload.control_type = $ui("unionControlType").value;
            payload.control_video = await uploadUi(requiredFileUi("unionVideo", "Control video", [".mp4",".mov",".m4v",".mkv",".avi",".webm"], 64 * 1048576));
          } else {
            try { payload.tracks = JSON.parse($ui("motionTracks").value); } catch (_) { throw new Error("Motion paths must be valid JSON."); }
          }
        }
        await apiUi("/api/render", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
        await refreshUi();
      } catch (error) { noticeUi.textContent = error.message; submitUi.disabled = tutorialSessionActiveUi || Boolean(readinessByMode[modeUi]?.blocked); }
    });
    cancelUi.addEventListener("click", async () => {
      noticeUi.textContent = ""; cancelUi.disabled = true;
      try { await apiUi("/api/cancel", { method:"POST", headers:{"Content-Type":"application/json"}, body:"{}" }); await refreshUi(); }
      catch (error) { noticeUi.textContent = error.message; cancelUi.disabled = false; }
    });

    async function startUpscaleUi(profile) {
      if (upscaleUiBusy) return;
      createUpscaleUi.disabled = true; createAiUpscaleUi.disabled = true;
      const ai = profile === "quality-ai";
      upscaleStatusUi.textContent = ai ? "Starting the local AI quality upscale… Your original video is unchanged." : "Creating a fast 720p resize… Your original video is unchanged.";
      try {
        const payload = ai ? {profile:"quality-ai", target_short_edge:1080} : {profile:"fast720"};
        await apiUi("/api/upscale", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        await refreshUpscaleUi();
      } catch (error) {
        upscaleStatusUi.textContent = error.message;
        createUpscaleUi.disabled = !captureUiAvailable;
        createAiUpscaleUi.disabled = !captureUiAvailable || !aiUpscaleReadyUi;
      }
    }
    createUpscaleUi.addEventListener("click", () => startUpscaleUi("fast720"));
    createAiUpscaleUi.addEventListener("click", () => startUpscaleUi("quality-ai"));
    cancelUpscaleUi.addEventListener("click", async () => {
      cancelUpscaleUi.disabled = true;
      try {
        await apiUi("/api/upscale/cancel", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
        await refreshUpscaleUi();
      } catch (error) {
        upscaleStatusUi.textContent = error.message;
        cancelUpscaleUi.disabled = false;
      }
    });

    function renderUpscaleUi(data) {
      const job = data.job, matchesCurrent = Boolean(job && currentMainJobIdUi && job.source_job_id === currentMainJobIdUi);
      const busy = Boolean(job && activeStates.has(job.status));
      const aiReadiness = data.ai_readiness || {};
      aiUpscaleReadyUi = Boolean(aiReadiness.ready);
      upscaleUiBusy = busy;
      createUpscaleUi.disabled = !captureUiAvailable || busy;
      createAiUpscaleUi.disabled = !captureUiAvailable || busy || !aiUpscaleReadyUi;
      createAiUpscaleUi.title = aiUpscaleReadyUi ? "" : (aiReadiness.note || "AI upscaler is not ready.");
      cancelUpscaleUi.hidden = !busy;
      cancelUpscaleUi.disabled = job?.status === "cancelling";
      downloadUpscaleUi.hidden = true;
      downloadUpscaleUi.removeAttribute("href");
      upscaleProgressUi.hidden = !busy;
      if (busy) {
        upscaleProgressStageUi.textContent = String(job.stage || "preparing").replace(/\b\w/g, (letter) => letter.toUpperCase());
        if (job.progress_percent == null) { upscaleProgressBarUi.removeAttribute("value"); upscaleProgressValueUi.textContent = "Working locally"; }
        else { upscaleProgressBarUi.value = job.progress_percent; upscaleProgressValueUi.textContent = `${job.progress_percent.toFixed(1)}% · ${job.completed_frames || 0}/${job.total_frames || "?"} frames`; }
      }
      if (busy && !matchesCurrent) {
        upscaleStatusUi.textContent = "An upscale for the previous completed video is still running.";
        return;
      }
      if (!matchesCurrent) {
        upscaleStatusUi.textContent = captureUiAvailable
          ? (aiUpscaleReadyUi ? "AI Quality Upscale is ready; 1080p is recommended. The original will be preserved." : `Fast resize is ready. AI Quality Upscale is unavailable: ${aiReadiness.note || "setup incomplete"}`)
          : "Finish a video to enable local upscaling.";
        return;
      }
      if (busy) {
        const label = job.profile === "quality-ai" ? "AI quality upscale" : "fast 720p resize";
        upscaleStatusUi.textContent = job.status === "cancelling" ? `Cancelling the ${label}…` : `Running the ${label} locally… Your original video is unchanged.`;
      } else if (job.status === "succeeded" && job.output_url) {
        downloadUpscaleUi.href = job.output_url;
        downloadUpscaleUi.textContent = job.profile === "quality-ai" ? "Download AI 1080p MP4" : "Download 720p MP4";
        downloadUpscaleUi.hidden = false;
        const label = job.profile === "quality-ai" ? "AI quality upscale" : "Fast resize";
        upscaleStatusUi.textContent = `${label} ready · ${job.output_width}×${job.output_height} · ${job.elapsed_seconds.toFixed(2)}s · original preserved`;
      } else if (job.status === "failed") {
        upscaleStatusUi.textContent = `Couldn’t complete the upscale. Your original is unchanged. ${job.error || ""}`.trim();
      } else if (job.status === "cancelled") {
        upscaleStatusUi.textContent = "Upscale cancelled. Your original video is unchanged.";
      }
    }
    async function refreshUpscaleUi() {
      if (upscalePollBusy) return;
      upscalePollBusy = true;
      try { renderUpscaleUi(await apiUi("/api/upscale/status", {cache:"no-store"})); }
      catch (error) { upscaleStatusUi.textContent = `Upscale status unavailable: ${error.message}`; }
      finally { upscalePollBusy = false; }
    }

    function renderUi(data) {
      const job = data.job, state = job ? job.status : "idle", busy = activeStates.has(state);
      currentMainJobIdUi = job ? job.id : null;
      renderUiBusy = busy;
      captureUiAvailable = Boolean(job && state === "succeeded" && job.output_url);
      $ui("captureCurrentFrame").disabled = captureUiBusy || !captureUiAvailable;
      statusUi.textContent = stateLabels[state] || state; dotUi.className = `dot ${state}`; submitUi.disabled = tutorialSessionActiveUi || busy || Boolean(readinessByMode[modeUi]?.blocked); cancelUi.hidden = !busy; cancelUi.disabled = state === "cancelling";
      metaUi.textContent = job ? `${job.mode || "generate"}${job.profile ? ` · ${job.profile}` : ""}${job.aspect_ratio && job.aspect_ratio !== "profile" ? ` · ${job.aspect_ratio}` : ""}${job.gemma_pack ? ` · ${job.gemma_pack}` : ""}${job.restored ? " · restored from disk" : (job.seed == null ? "" : ` · seed ${job.seed}`)}${!job.restored && job.elapsed_seconds != null ? ` · ${job.elapsed_seconds.toFixed(1)}s` : ""}` : "idle";
      $ui("progressWrap").hidden = !job || (!busy && state !== "succeeded");
      if (job) {
        $ui("progressBar").hidden = state === "succeeded";
        $ui("progressStage").textContent = (job.stage || "queued").replace(/\b\w/g, (letter) => letter.toUpperCase());
        if (job.progress_percent == null) { $ui("progressBar").removeAttribute("value"); $ui("progressValue").textContent = state === "succeeded" ? "Complete" : "Working locally · live log below"; }
        else { $ui("progressBar").value = job.progress_percent; $ui("progressValue").textContent = `${job.progress_percent.toFixed(1)}% · parsed from denoising output`; }
      }
      const nearBottom = logUi.scrollHeight - logUi.scrollTop - logUi.clientHeight < 45; logUi.textContent = job && job.log ? job.log : "Waiting for a render…"; if (nearBottom) logUi.scrollTop = logUi.scrollHeight;
      if (job && job.error) noticeUi.textContent = job.error;
      if (job && state === "succeeded" && job.output_url) {
        if (videoUiUrl !== job.output_url) { videoUiUrl = job.output_url; videoUi.src = job.output_url; downloadUi.href = job.output_url; startPresentedFrameTrackingUi(); }
        emptyUi.style.display = "none"; videoUi.style.display = "block"; resultUi.style.display = "block";
      } else if (!videoUiUrl || busy) {
        stopPresentedFrameTrackingUi(); videoUi.removeAttribute("src"); videoUi.load(); videoUiUrl = null; videoUi.style.display = "none"; resultUi.style.display = "none"; emptyUi.style.display = "block";
        emptyUi.lastChild.textContent = busy ? " Rendering locally. Live stages and logs appear below." : " Your completed video will appear here.";
      }
    }
    async function refreshUi() { if (pollUiBusy) return; pollUiBusy = true; try { renderUi(await apiUi("/api/status", {cache:"no-store"})); } catch (error) { noticeUi.textContent = `UI connection lost: ${error.message}`; } finally { pollUiBusy = false; } }
    function displayCurrentAvailabilityUi() {
      const target = $ui("currentAvailabilityList"); if (!target) return;
      const labels = {generate:"Generate", ingredients:"Ingredients", motion:"Motion Track", union:"Union Control", retake:"Retake", extend:"Extend", a2v:"Audio-to-Video"};
      const eligible = [], blocked = [];
      for (const [mode, label] of Object.entries(labels)) {
        const item = readinessByMode[mode];
        if (!item) continue;
        (item.blocked ? blocked : eligible).push(label);
      }
      target.replaceChildren();
      const add = (heading, copy) => { const li = document.createElement("li"), strong = document.createElement("strong"); strong.textContent = heading; li.append(strong, document.createTextNode(copy)); target.appendChild(li); };
      add("Installed and eligible: ", eligible.length ? eligible.join(", ") + "." : "none confirmed yet.");
      add("Currently blocked: ", blocked.length ? blocked.join(", ") + "." : "none.");
      add("Verification meaning: ", "this is an installed-asset check, not a render guarantee; full receipt, source and Metal preflight runs again on submit.");
      add("Helpers: ", "Character Bible, Frame Composer, anchors, Scene Beats, frame capture, Character Voices, AI Quality Upscale and Fast 720p remain separate supporting stages.");
    }
    async function loadReadinessUi() { try { const data = await apiUi("/api/readiness", {cache:"no-store"}); Object.assign(readinessByMode, data.modes || {}); document.querySelectorAll("[data-select-mode]").forEach((button) => { const item = readinessByMode[button.dataset.selectMode]; button.setAttribute("aria-disabled", String(Boolean(item?.blocked))); if (item?.blocked) button.title = item.note; else button.removeAttribute("title"); }); const packs = data.gemma_packs || {}, selector = $ui("gemmaPack"); for (const option of selector.options) { if (option.value !== "auto") option.disabled = packs[option.value] ? !packs[option.value].ready : true; } const automatic = packs.auto || {}; $ui("gemmaPackNote").textContent = automatic.note || "The render wrapper will run a full local verification before loading Gemma."; displayReadinessUi(); displayCurrentAvailabilityUi(); if (!tutorialPanelUi.hidden) renderTutorialUi(); } catch (_) { $ui("modeReadiness").textContent = "Readiness check unavailable; each wrapper will still run its full local preflight."; $ui("gemmaPackNote").textContent = "Pack readiness is unavailable; the render wrapper will still run its full verification."; const target = $ui("currentAvailabilityList"); if (target) target.textContent = "Live readiness unavailable; each workflow still performs full preflight on submit."; } }
    loadTutorialProgressUi(); addVoiceCueUi(); selectModeUi("generate"); refreshUi(); refreshFrameComposerUi(); refreshVoiceUi(); refreshUpscaleUi(); loadReadinessUi();
    refreshCharacterProfilesUi().catch((error) => { $ui("characterProfileStatus").className = "character-profile-status needs"; $ui("characterProfileStatus").textContent = `Character library unavailable: ${error.message}`; });
    setInterval(refreshUi, 900); setInterval(refreshFrameComposerUi, 1200); setInterval(refreshVoiceUi, 1200); setInterval(refreshUpscaleUi, 900);
  </script>
</body>
</html>
"""


class APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _image_dimensions(kind: str, data: bytes) -> tuple[int, int] | None:
    """Read dimensions from common image headers without invoking an image decoder."""
    if kind == "png":
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        return None
    if kind == "jpeg":
        if len(data) < 4 or data[:3] != b"\xff\xd8\xff":
            return None
        pos = 2
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            while pos < len(data) and data[pos] == 0xFF:
                pos += 1
            if pos >= len(data):
                return None
            marker = data[pos]
            pos += 1
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if pos + 2 > len(data):
                return None
            segment_length = int.from_bytes(data[pos : pos + 2], "big")
            if segment_length < 2 or pos + segment_length > len(data):
                return None
            if marker in sof and segment_length >= 7:
                height = int.from_bytes(data[pos + 3 : pos + 5], "big")
                width = int.from_bytes(data[pos + 5 : pos + 7], "big")
                return width, height
            if marker == 0xDA:
                return None
            pos += segment_length
        return None
    if kind == "webp":
        if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
            return None
        chunk = data[12:16]
        if chunk == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if chunk == b"VP8L" and data[20] == 0x2F:
            b1, b2, b3, b4 = data[21:25]
            return 1 + b1 + ((b2 & 0x3F) << 8), 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0F) << 10)
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
        return None
    return None


def _decode_image(value: Any, label: str) -> tuple[str, bytes] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} must be a base64 image data URL.")
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=\r\n]+)", value)
    if not match:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} must be a PNG, JPEG, or WebP data URL.")
    kind, encoded = match.groups()
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 8:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{label} is larger than 8 MB.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} contains invalid base64 data.") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{label} must be between 1 byte and 8 MB.")
    dimensions = _image_dimensions(kind, data)
    if dimensions is None:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} is not a valid {kind.upper()} file.")
    width, height = dimensions
    if width < 1 or height < 1 or width > 8192 or height > 8192 or width * height > MAX_IMAGE_PIXELS:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} dimensions are unsafe or unsupported.")
    return ("jpg" if kind == "jpeg" else kind), data


def _strict_image_container(kind: str, data: bytes, *, label: str) -> None:
    """Reject truncated or mislabeled image containers after cheap header parsing."""
    complete = False
    if kind == "png":
        complete = len(data) >= 45 and data.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82")
    elif kind == "jpg":
        complete = len(data) >= 4 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")
    elif kind == "webp":
        complete = (
            len(data) >= 20
            and data[:4] == b"RIFF"
            and data[8:12] == b"WEBP"
            and int.from_bytes(data[4:8], "little") + 8 == len(data)
        )
    if not complete or _image_dimensions(kind, data) is None:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} is truncated or has an invalid image container.")


def _decode_reference_image(value: Any) -> tuple[str, bytes] | None:
    """Backward-compatible decoder kept for old clients and helper tests."""
    return _decode_image(value, "Reference image")


def _validate_character_fields(payload: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, label, max_chars, max_bytes in CHARACTER_FIELDS:
        value = payload.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise APIError(HTTPStatus.BAD_REQUEST, f"{key} must be text.")
        value = value.strip()
        if "\x00" in value:
            raise APIError(HTTPStatus.BAD_REQUEST, f"{label} contains unsupported characters.")
        if len(value) > max_chars or len(value.encode("utf-8")) > max_bytes:
            raise APIError(HTTPStatus.BAD_REQUEST, f"{label} is too long (maximum {max_chars} characters).")
        values[key] = value
    return values


def _validate_character_profile_id(value: Any, *, field: str = "character_profile_id") -> str:
    if not isinstance(value, str) or not CHARACTER_PROFILE_ID_RE.fullmatch(value):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} is invalid.")
    return value


def _ensure_character_store() -> Path:
    try:
        if CHARACTERS_DIR.exists() and (CHARACTERS_DIR.is_symlink() or not CHARACTERS_DIR.is_dir()):
            raise OSError("unsafe character store")
        CHARACTERS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        if CHARACTERS_DIR.is_symlink():
            raise OSError("symbolic-link character store")
        CHARACTERS_DIR.chmod(0o700)
        return CHARACTERS_DIR.resolve(strict=True)
    except OSError:
        raise APIError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "The local character store is unavailable or unsafe.",
        ) from None


def _character_profile_directory(profile_id: str, *, must_exist: bool) -> tuple[Path, Path]:
    profile_id = _validate_character_profile_id(profile_id)
    root = _ensure_character_store()
    unresolved = root / profile_id
    if unresolved.is_symlink():
        raise APIError(HTTPStatus.CONFLICT, "The saved character profile is unsafe.")
    if not unresolved.exists():
        if must_exist:
            raise APIError(HTTPStatus.NOT_FOUND, "Saved character profile not found.")
        return root, unresolved
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError:
        raise APIError(HTTPStatus.NOT_FOUND, "Saved character profile not found.") from None
    if resolved.parent != root or not resolved.is_dir() or resolved.is_symlink():
        raise APIError(HTTPStatus.CONFLICT, "The saved character profile is unsafe.")
    return root, resolved


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_private_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short character reference write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _character_reference_filename(reference_id: str, kind: str) -> str:
    if not CHARACTER_REFERENCE_ID_RE.fullmatch(reference_id) or kind not in {"png", "jpg", "webp"}:
        raise APIError(HTTPStatus.CONFLICT, "A saved character reference entry is invalid.")
    return f"ref-{reference_id}.{kind}"


def _read_character_profile(profile_id: str, *, include_data: bool = False) -> dict[str, Any]:
    _, directory = _character_profile_directory(profile_id, must_exist=True)
    manifest_path = directory / "profile.json"
    if manifest_path.is_symlink():
        raise APIError(HTTPStatus.CONFLICT, "The saved character manifest is unsafe.")
    try:
        if not manifest_path.is_file() or manifest_path.stat().st_size > 64 * 1024:
            raise OSError("missing or oversized character manifest")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise APIError(HTTPStatus.CONFLICT, "The saved character manifest is missing or corrupt.") from None
    expected = {"schema", "id", "created_at", "updated_at", "fields", "references"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise APIError(HTTPStatus.CONFLICT, "The saved character manifest has an unsupported format.")
    if raw.get("schema") != CHARACTER_PROFILE_SCHEMA or raw.get("id") != profile_id:
        raise APIError(HTTPStatus.CONFLICT, "The saved character manifest identity does not match its directory.")
    if not isinstance(raw.get("created_at"), str) or not isinstance(raw.get("updated_at"), str):
        raise APIError(HTTPStatus.CONFLICT, "The saved character manifest timestamps are invalid.")
    fields = raw.get("fields")
    if not isinstance(fields, dict) or set(fields) != {key for key, _, _, _ in CHARACTER_FIELDS}:
        raise APIError(HTTPStatus.CONFLICT, "The saved character fields are invalid.")
    try:
        fields = _validate_character_fields(fields)
    except APIError:
        raise APIError(HTTPStatus.CONFLICT, "The saved character fields are invalid.") from None
    if not fields["character_name"]:
        raise APIError(HTTPStatus.CONFLICT, "The saved character has no name.")
    references = raw.get("references")
    if not isinstance(references, list) or len(references) > MAX_CHARACTER_REFERENCES:
        raise APIError(HTTPStatus.CONFLICT, "The saved character reference list is invalid.")
    checked_references: list[dict[str, Any]] = []
    total = 0
    seen: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "id",
            "kind",
            "size",
            "sha256",
            "original_name",
        }:
            raise APIError(HTTPStatus.CONFLICT, "A saved character reference entry is invalid.")
        reference_id = reference.get("id")
        kind = reference.get("kind")
        size = reference.get("size")
        digest = reference.get("sha256")
        original_name = reference.get("original_name")
        if (
            not isinstance(reference_id, str)
            or reference_id in seen
            or not CHARACTER_REFERENCE_ID_RE.fullmatch(reference_id)
            or kind not in {"png", "jpg", "webp"}
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_CHARACTER_REFERENCE_BYTES
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(original_name, str)
            or not 1 <= len(original_name) <= MAX_MEDIA_NAME
            or "\x00" in original_name
            or "/" in original_name
            or "\\" in original_name
        ):
            raise APIError(HTTPStatus.CONFLICT, "A saved character reference entry is invalid.")
        seen.add(reference_id)
        filename = _character_reference_filename(reference_id, kind)
        unresolved = directory / filename
        if unresolved.is_symlink():
            raise APIError(HTTPStatus.CONFLICT, "A saved character reference is unsafe.")
        try:
            path = unresolved.resolve(strict=True)
            if path.parent != directory or not path.is_file() or path.stat().st_size != size:
                raise OSError("unsafe saved reference")
            data = path.read_bytes() if include_data else None
        except OSError:
            raise APIError(HTTPStatus.CONFLICT, "A saved character reference is missing or corrupt.") from None
        if data is not None:
            if hashlib.sha256(data).hexdigest() != digest:
                raise APIError(HTTPStatus.CONFLICT, "A saved character reference failed integrity validation.")
            try:
                _strict_image_container(kind, data, label="Saved character reference")
            except APIError:
                raise APIError(HTTPStatus.CONFLICT, "A saved character reference failed integrity validation.") from None
        total += size
        if total > MAX_TOTAL_CHARACTER_REFERENCE_BYTES:
            raise APIError(HTTPStatus.CONFLICT, "Saved character references exceed the safe total size.")
        checked = dict(reference)
        checked["path"] = path
        if data is not None:
            checked["data"] = data
        checked_references.append(checked)
    return {
        "schema": raw["schema"],
        "id": profile_id,
        "created_at": raw["created_at"],
        "updated_at": raw["updated_at"],
        "fields": fields,
        "references": checked_references,
        "directory": directory,
    }


def _public_character_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": profile["id"],
        "created_at": profile["created_at"],
        "updated_at": profile["updated_at"],
        **profile["fields"],
        "references": [],
    }
    for reference in profile["references"]:
        reference_id = reference["id"]
        result["references"].append(
            {
                "id": reference_id,
                "name": reference["original_name"],
                "kind": reference["kind"],
                "size": reference["size"],
                "url": f"/character-media/{profile['id']}/{reference_id}",
            }
        )
    return result


def _list_character_profiles() -> list[dict[str, Any]]:
    root = _ensure_character_store()
    profiles: list[dict[str, Any]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read the local character store.") from None
    for entry in entries:
        if not CHARACTER_PROFILE_ID_RE.fullmatch(entry.name):
            continue
        profile = _read_character_profile(entry.name)
        profiles.append(_public_character_profile(profile))
    profiles.sort(key=lambda item: (item["character_name"].casefold(), item["id"]))
    return profiles


def _decode_character_profile_references(value: Any) -> list[tuple[str, bytes, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_CHARACTER_REFERENCES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"references must contain at most {MAX_CHARACTER_REFERENCES} images.",
        )
    decoded: list[tuple[str, bytes, str]] = []
    total = 0
    extensions = {"png": {".png"}, "jpg": {".jpg", ".jpeg"}, "webp": {".webp"}}
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != {"name", "data"}:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Character reference {index} must contain only name and data.",
            )
        name = item.get("name")
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= MAX_MEDIA_NAME
            or "\x00" in name
            or "/" in name
            or "\\" in name
        ):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Character reference {index} has an invalid file name.")
        image = _decode_image(item.get("data"), f"Character reference {index}")
        if image is None:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Character reference {index} is required.")
        kind, data = image
        _strict_image_container(kind, data, label=f"Character reference {index}")
        if Path(name).suffix.lower() not in extensions[kind]:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Character reference {index} file extension does not match its image data.",
            )
        total += len(data)
        if total > MAX_TOTAL_CHARACTER_REFERENCE_BYTES:
            raise APIError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Character references may total at most 24 MB.",
            )
        decoded.append((kind, data, name))
    return decoded


def _save_character_profile(payload: dict[str, Any], profile_id: str | None = None) -> dict[str, Any]:
    allowed = {key for key, _, _, _ in CHARACTER_FIELDS} | {
        "references",
        "replace",
        "copy_references_from",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, "Unknown field(s): " + ", ".join(unknown))
    fields = _validate_character_fields(payload)
    if not fields["character_name"]:
        raise APIError(HTTPStatus.BAD_REQUEST, "Character name is required before saving a profile.")
    replace = payload.get("replace", False)
    if not isinstance(replace, bool):
        raise APIError(HTTPStatus.BAD_REQUEST, "replace must be true or false.")
    if profile_id is None and replace:
        raise APIError(HTTPStatus.BAD_REQUEST, "A new character profile cannot replace an unspecified profile.")
    if profile_id is not None:
        profile_id = _validate_character_profile_id(profile_id)
        if not replace:
            raise APIError(HTTPStatus.CONFLICT, "Replacing a saved character requires explicit confirmation.")
    decoded = _decode_character_profile_references(payload.get("references"))
    copy_from = payload.get("copy_references_from")
    if copy_from not in {None, ""}:
        if decoded:
            raise APIError(HTTPStatus.BAD_REQUEST, "Use uploaded references or copy_references_from, not both.")
        copy_id = _validate_character_profile_id(copy_from, field="copy_references_from")
        source = _read_character_profile(copy_id, include_data=True)
        decoded = [
            (reference["kind"], reference["data"], reference["original_name"])
            for reference in source["references"]
        ]

    with CHARACTER_STORE_LOCK:
        root = _ensure_character_store()
        existing: dict[str, Any] | None = None
        if profile_id is not None:
            existing = _read_character_profile(profile_id)
        else:
            ids = [entry.name for entry in root.iterdir() if CHARACTER_PROFILE_ID_RE.fullmatch(entry.name)]
            if len(ids) >= MAX_CHARACTER_PROFILES:
                raise APIError(
                    HTTPStatus.INSUFFICIENT_STORAGE,
                    f"The local character library is limited to {MAX_CHARACTER_PROFILES} profiles.",
                )
        for saved in _list_character_profiles():
            if saved["character_name"].casefold() == fields["character_name"].casefold() and saved["id"] != profile_id:
                raise APIError(
                    HTTPStatus.CONFLICT,
                    "A character with this name is already saved. Select it and use Replace selected.",
                )
        if profile_id is None:
            profile_id = uuid.uuid4().hex
            _, directory = _character_profile_directory(profile_id, must_exist=False)
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                raise APIError(HTTPStatus.CONFLICT, "Could not allocate a unique character profile ID.") from None
            created_at = _utc_now()
            created_directory = True
        else:
            assert existing is not None
            directory = existing["directory"]
            created_at = existing["created_at"]
            created_directory = False

        new_paths: list[Path] = []
        new_entries: list[dict[str, Any]] = []
        try:
            for kind, data, original_name in decoded:
                reference_id = uuid.uuid4().hex
                filename = _character_reference_filename(reference_id, kind)
                path = directory / filename
                _write_private_file(path, data)
                new_paths.append(path)
                new_entries.append(
                    {
                        "id": reference_id,
                        "kind": kind,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "original_name": original_name,
                    }
                )
            manifest = {
                "schema": CHARACTER_PROFILE_SCHEMA,
                "id": profile_id,
                "created_at": created_at,
                "updated_at": _utc_now(),
                "fields": fields,
                "references": new_entries,
            }
            _atomic_write_json(directory / "profile.json", manifest)
        except OSError:
            for path in new_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if created_directory:
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save the character profile locally.") from None

        if existing is not None:
            for reference in existing["references"]:
                old_path = reference["path"]
                if old_path not in new_paths:
                    try:
                        old_path.unlink()
                    except OSError:
                        pass
        return _public_character_profile(_read_character_profile(profile_id))


def _delete_character_profile(profile_id: str) -> None:
    profile_id = _validate_character_profile_id(profile_id)
    with CHARACTER_STORE_LOCK:
        profile = _read_character_profile(profile_id)
        directory = profile["directory"]
        expected = {directory / "profile.json"} | {reference["path"] for reference in profile["references"]}
        try:
            actual = set(directory.iterdir())
        except OSError:
            raise APIError(HTTPStatus.CONFLICT, "Could not inspect the saved character directory.") from None
        if actual != expected or any(path.is_symlink() for path in actual):
            raise APIError(
                HTTPStatus.CONFLICT,
                "The saved character directory contains unexpected files and was not deleted.",
            )
        try:
            for reference in profile["references"]:
                reference["path"].unlink()
            (directory / "profile.json").unlink()
            directory.rmdir()
        except OSError:
            raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not delete the saved character profile.") from None


def _resolve_character_reference_group(
    payload: dict[str, Any],
    character: dict[str, str],
) -> dict[str, Any] | None:
    images: list[tuple[str, bytes]] = []
    profile_id = payload.get("character_profile_id")
    if profile_id not in {None, ""}:
        saved = _read_character_profile(_validate_character_profile_id(profile_id), include_data=True)
        images.extend((reference["kind"], reference["data"]) for reference in saved["references"])
    raw_images = payload.get("character_reference_images")
    if raw_images is None:
        raw_images = []
    if not isinstance(raw_images, list) or len(raw_images) > MAX_CHARACTER_REFERENCES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"character_reference_images must contain at most {MAX_CHARACTER_REFERENCES} images.",
        )
    for index, value in enumerate(raw_images, start=1):
        image = _decode_image(value, f"Character Bible reference {index}")
        if image is None:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Character Bible reference {index} is required.")
        _strict_image_container(image[0], image[1], label=f"Character Bible reference {index}")
        images.append(image)
    if len(images) > MAX_CHARACTER_REFERENCES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"Use at most {MAX_CHARACTER_REFERENCES} saved and newly selected character references combined.",
        )
    if sum(len(data) for _, data in images) > MAX_TOTAL_CHARACTER_REFERENCE_BYTES:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Character references may total at most 24 MB.")
    if not images:
        return None
    description_parts = [
        character[key]
        for key in ("character_appearance", "character_wardrobe", "character_continuity")
        if character[key]
    ]
    return {
        "name": character["character_name"] or "Saved character",
        "description": " ".join(description_parts),
        "strength": 0.9,
        "images": images,
    }


def _suggest_capture_aspect_ratio(width: int, height: int, source_aspect: Any) -> str:
    """Keep an explicit source ratio; otherwise select the closest UI preset."""
    if isinstance(source_aspect, str) and source_aspect in NAMED_ASPECT_RATIO_VALUES:
        return source_aspect
    ratio = width / height
    return min(
        NAMED_ASPECT_RATIO_VALUES,
        key=lambda name: abs(math.log(ratio / NAMED_ASPECT_RATIO_VALUES[name])),
    )


def _save_captured_video_frame(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - {"image", "source_media", "timestamp_seconds"})
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, "Unknown frame-capture field(s): " + ", ".join(unknown))
    source_media = payload.get("source_media")
    if not isinstance(source_media, str) or len(source_media) > 256:
        raise APIError(HTTPStatus.BAD_REQUEST, "source_media must identify the completed local video.")
    parsed = urlsplit(source_media)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not parsed.path.startswith("/media/"):
        raise APIError(HTTPStatus.BAD_REQUEST, "source_media must be a local completed-video URL.")
    try:
        media_name = unquote(parsed.path[len("/media/") :], errors="strict")
    except UnicodeError:
        raise APIError(HTTPStatus.BAD_REQUEST, "source_media is invalid.") from None
    if (
        not media_name
        or "/" in media_name
        or "\\" in media_name
        or not MEDIA_NAME_RE.fullmatch(media_name)
    ):
        raise APIError(HTTPStatus.BAD_REQUEST, "source_media is invalid.")
    current = STATE.snapshot().get("job")
    if (
        not isinstance(current, dict)
        or current.get("status") != "succeeded"
        or current.get("output_url") != source_media
    ):
        raise APIError(
            HTTPStatus.CONFLICT,
            "Capture is allowed only for the completed video currently shown by this local UI.",
        )
    timestamp = _finite_number(
        payload.get("timestamp_seconds"),
        field="timestamp_seconds",
        minimum=0,
        maximum=86_400,
    )
    decoded = _decode_image(payload.get("image"), "Captured frame")
    if decoded is None or decoded[0] != "png":
        raise APIError(HTTPStatus.BAD_REQUEST, "Captured frame must be a PNG image.")
    _, data = decoded
    _strict_image_container("png", data, label="Captured frame")
    dimensions = _image_dimensions("png", data)
    assert dimensions is not None
    if FRAME_GENERATED_DIR.exists() and (FRAME_GENERATED_DIR.is_symlink() or not FRAME_GENERATED_DIR.is_dir()):
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "Captured-frame output directory is unsafe.")
    try:
        FRAME_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        if FRAME_GENERATED_DIR.is_symlink():
            raise OSError("symbolic-link captured-frame root")
        root = FRAME_GENERATED_DIR.resolve(strict=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        token = uuid.uuid4().hex[:12]
        name = f"ltx25-capture-{stamp}-{token}.png"
        final_path = root / name
        temporary_path = root / f".{name}.{uuid.uuid4().hex}.tmp.png"
        if final_path.exists() or final_path.is_symlink():
            raise OSError("captured-frame name collision")
        _write_private_file(temporary_path, data)
        os.replace(temporary_path, final_path)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        try:
            if "temporary_path" in locals() and not temporary_path.is_symlink():
                temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save the captured frame locally.") from None
    width, height = dimensions
    return {
        "name": name,
        "url": "/frames/" + quote(name, safe=""),
        "timestamp_seconds": timestamp,
        "width": width,
        "height": height,
        "suggested_aspect_ratio": _suggest_capture_aspect_ratio(
            width,
            height,
            current.get("aspect_ratio"),
        ),
        "source_media": source_media,
    }


def _clean_text(value: Any, *, field: str, max_chars: int, max_bytes: int) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be text.")
    value = value.strip()
    if "\x00" in value:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} contains unsupported characters.")
    if len(value) > max_chars or len(value.encode("utf-8")) > max_bytes:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} is too long (maximum {max_chars} characters).")
    return value


def _validate_video_prompt(value: Any, *, empty_message: str) -> str:
    """Validate a renderer prompt without normalizing or truncating it."""
    if not isinstance(value, str):
        raise APIError(HTTPStatus.BAD_REQUEST, "prompt must be text.")
    if not value.strip():
        raise APIError(HTTPStatus.BAD_REQUEST, empty_message)
    if "\x00" in value:
        raise APIError(HTTPStatus.BAD_REQUEST, "Prompt contains an unsupported NUL character.")
    encoded_size = len(value.encode("utf-8"))
    if len(value) > MAX_PROMPT_CHARS or encoded_size > MAX_PROMPT_BYTES:
        raise APIError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "Prompt exceeds the local renderer safety ceiling "
            f"({MAX_PROMPT_CHARS:,} characters / {MAX_PROMPT_BYTES // 1024} KiB UTF-8). "
            "It was rejected without truncation.",
        )
    return value


def _validate_final_prompt_size(value: str, *, label: str) -> None:
    """Protect the local process argv after adding structured prompt context."""
    if len(value) > MAX_FINAL_PROMPT_CHARS or len(value.encode("utf-8")) > MAX_FINAL_PROMPT_BYTES:
        raise APIError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"{label} exceeds the local renderer safety ceiling after structured context was added. "
            "It was rejected without truncation.",
        )


def _validate_generated_keyframes(value: Any) -> int:
    if value is None:
        value = 0
    if isinstance(value, bool) or not isinstance(value, int) or value not in GENERATED_KEYFRAME_CHOICES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "generated_keyframes must be 0 for the distilled Q4 pack; use uploaded timeline anchors instead.",
        )
    return value


def _validate_duration_seconds(value: Any, *, field: str = "duration_seconds") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be an integer from 1 to 10.")
    return value


def _validate_aspect_ratio(value: Any) -> str:
    if value is None:
        return "profile"
    if not isinstance(value, str) or value not in ASPECT_RATIOS:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "aspect_ratio must be profile, 16:9, 9:16, 1:1, 4:3, or 3:4.",
        )
    return value


def _validate_scenes(
    value: Any,
    profile: str,
    *,
    total_frames: int | None = None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise APIError(HTTPStatus.BAD_REQUEST, "scenes must be a list of scene-beat objects.")

    frame_count = PROFILE_FRAMES[profile] if total_frames is None else total_frames
    latent_frames = (frame_count + 7) // 8
    if len(value) > latent_frames:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"This render has {latent_frames} latent frames and supports at most {latent_frames} scenes.",
        )

    scenes: list[dict[str, Any]] = []
    for index, raw_scene in enumerate(value, start=1):
        if not isinstance(raw_scene, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Scene {index} must be an object.")
        unknown = sorted(set(raw_scene) - {"text", "length"})
        if unknown:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown scene {index} field(s): " + ", ".join(unknown),
            )
        text = _clean_text(
            raw_scene.get("text"),
            field=f"Scene {index} text",
            max_chars=SCENE_TEXT_MAX_CHARS,
            max_bytes=SCENE_TEXT_MAX_BYTES,
        )
        if not text:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Scene {index} text cannot be empty.")
        length = raw_scene.get("length")
        if length is not None and (
            isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= latent_frames
        ):
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Scene {index} length must be null or an integer from 1 to {latent_frames} latent frames.",
            )
        scenes.append({"text": text, "length": length})

    automatic_count = sum(scene["length"] is None for scene in scenes)
    pinned_total = sum(scene["length"] or 0 for scene in scenes)
    if pinned_total > latent_frames:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"Scene lengths require {pinned_total} latent frames, but this render has {latent_frames}.",
        )
    remaining = latent_frames - pinned_total
    if automatic_count and remaining < automatic_count:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"Scene lengths leave {remaining} latent frames for {automatic_count} automatic scenes; "
            "each scene needs at least one.",
        )
    return scenes


def _finite_strength(value: Any, *, field: str, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be a number from 0 to 1.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be a finite number from 0 to 1.")
    return result


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    default: float | None = None,
) -> float:
    if value is None and default is not None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"{field} must be a number from {minimum:g} to {maximum:g}.",
        )
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"{field} must be a finite number from {minimum:g} to {maximum:g}.",
        )
    return result


def _validate_bool(value: Any, *, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be true or false.")
    return value


def _decode_uploaded_file(
    value: Any,
    *,
    label: str,
    extensions: set[str],
    max_bytes: int,
    media_prefix: str,
) -> tuple[str, bytes, str]:
    if not isinstance(value, dict):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} must be an uploaded file object.")
    unknown = sorted(set(value) - {"name", "data"})
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, f"Unknown {label} field(s): " + ", ".join(unknown))
    name = value.get("name")
    data_url = value.get("data")
    if not isinstance(name, str) or not name or len(name) > MAX_MEDIA_NAME:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} has an invalid filename.")
    if Path(name).name != name or "/" in name or "\\" in name or "\x00" in name:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} filename must not contain a path.")
    extension = Path(name).suffix.lower()
    if extension not in extensions:
        allowed = ", ".join(sorted(extensions))
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} must use one of these extensions: {allowed}.")
    if not isinstance(data_url, str):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} data must be a base64 data URL.")
    match = re.fullmatch(r"data:([^;,]*);base64,([A-Za-z0-9+/=\r\n]+)", data_url)
    if not match:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} must be a base64 data URL.")
    mime, encoded = match.groups()
    if mime and mime != "application/octet-stream" and not mime.startswith(media_prefix + "/"):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} has an unexpected media type.")
    if len(encoded) > ((max_bytes + 2) // 3) * 4 + 8:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{label} is too large.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} contains invalid base64 data.") from None
    if not data:
        raise APIError(HTTPStatus.BAD_REQUEST, f"{label} cannot be empty.")
    if len(data) > max_bytes:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{label} is too large.")
    return extension, data, name


_VOICE_READINESS_LOCK = threading.Lock()
_VOICE_READINESS_CACHE: dict[str, Any] = {"checked_at": 0.0, "value": None}


def _character_voice_module() -> Any:
    if not CHARACTER_VOICE_SCRIPT.is_file() or CHARACTER_VOICE_SCRIPT.is_symlink():
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The Character Voice wrapper is missing or unsafe.")
    try:
        import character_voice
    except Exception as exc:
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, f"Character Voice wrapper could not load: {exc}") from None
    return character_voice


def _character_voice_readiness(*, refresh: bool = False) -> dict[str, Any]:
    """Return cached, path-free readiness without repeatedly importing Torch."""
    now = time.monotonic()
    with _VOICE_READINESS_LOCK:
        cached = _VOICE_READINESS_CACHE.get("value")
        checked_at = float(_VOICE_READINESS_CACHE.get("checked_at") or 0.0)
        active_compute = False
        for state_name in ("STATE", "FRAME_STATE", "VOICE_STATE", "UPSCALE_STATE"):
            state = globals().get(state_name)
            if state is not None:
                try:
                    active_compute = active_compute or bool(state.is_busy())
                except Exception:
                    pass
        if not refresh and isinstance(cached, dict) and (now - checked_at < 300.0 or active_compute):
            return dict(cached)
        mux_ready = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        try:
            voice = _character_voice_module()
            status = voice.installer.readiness(full_hash=False)
            ready = bool(status.get("ready"))
            if ready:
                note = "Chatterbox Multilingual V3 is verified for offline local synthesis."
            else:
                # Installer diagnostics may contain absolute local paths.  The
                # browser only needs a stable, actionable summary; keep the
                # localhost API path-free as promised by this function.
                details = []
                if not status.get("runtime_ready"):
                    details.append("isolated runtime incomplete")
                if not status.get("model_ready"):
                    details.append("model files incomplete")
                if not status.get("receipt_ready"):
                    details.append("installation receipt incomplete")
                if not details:
                    details.append("local verification incomplete")
                note = "Synthesis is not installed yet — run install_character_voice.py. " + "; ".join(details)
            value = {
                "ready": ready,
                "registry_ready": True,
                "mux_ready": mux_ready,
                "note": note,
                "languages": dict(voice.SUPPORTED_LANGUAGES),
                "identity_note": "Speaker identity comes from the authorized canonical WAV; prose controls delivery only.",
            }
        except APIError as exc:
            value = {
                "ready": False,
                "registry_ready": False,
                "mux_ready": mux_ready,
                "note": exc.message,
                "languages": {},
                "identity_note": "Speaker identity requires an authorized canonical WAV.",
            }
        except Exception:
            value = {
                "ready": False,
                "registry_ready": CHARACTER_VOICE_SCRIPT.is_file(),
                "mux_ready": mux_ready,
                "note": "Character Voice readiness could not be verified; synthesis remains disabled.",
                "languages": {},
                "identity_note": "Speaker identity requires an authorized canonical WAV.",
            }
        _VOICE_READINESS_CACHE.update({"checked_at": now, "value": dict(value)})
        return value


def _public_voice_registry() -> dict[str, Any]:
    try:
        registry = _character_voice_module().load_registry()
    except APIError:
        return {"presets": [], "characters": {}}
    except Exception:
        return {"presets": [], "characters": {}, "error": "The local voice registry could not be read safely."}
    presets: list[dict[str, Any]] = []
    for name, raw in sorted(registry.get("presets", {}).items()):
        if not isinstance(name, str) or not isinstance(raw, dict):
            continue
        presets.append(
            {
                "name": name,
                "reference_language": str(raw.get("reference_language") or ""),
                "duration_seconds": raw.get("duration_seconds"),
                "voice_description": str(raw.get("voice_description") or ""),
                "consent_confirmed": raw.get("consent_confirmed") is True,
                "reference_sha256": str(raw.get("sha256") or ""),
            }
        )
    characters = {
        name: preset
        for name, preset in sorted(registry.get("characters", {}).items())
        if isinstance(name, str) and isinstance(preset, str)
    }
    return {"presets": presets, "characters": characters}


def _normalize_voice_character(value: Any, *, field: str = "character") -> str:
    if not isinstance(value, str):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{field} must be text.")
    normalized = value.strip().lower()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    if not VOICE_CHARACTER_RE.fullmatch(normalized):
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"{field} must contain 1–64 lowercase letters, digits, hyphens, or underscores.",
        )
    return normalized


def _decode_voice_reference(value: Any) -> bytes:
    if not isinstance(value, str):
        raise APIError(HTTPStatus.BAD_REQUEST, "Canonical voice reference must be a base64 WAV data URL.")
    match = re.fullmatch(
        r"data:(audio/(?:wav|x-wav|wave|vnd\.wave)|application/octet-stream);base64,([A-Za-z0-9+/=\r\n]+)",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise APIError(HTTPStatus.BAD_REQUEST, "Canonical voice reference must be a WAV data URL.")
    encoded = match.group(2)
    if len(encoded) > ((MAX_VOICE_REFERENCE_BYTES + 2) // 3) * 4 + 8:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Canonical voice reference is larger than 50 MB.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise APIError(HTTPStatus.BAD_REQUEST, "Canonical voice reference contains invalid base64 data.") from None
    if not 44 < len(data) <= MAX_VOICE_REFERENCE_BYTES:
        raise APIError(HTTPStatus.BAD_REQUEST, "Canonical voice reference must be a non-empty WAV no larger than 50 MB.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise APIError(HTTPStatus.BAD_REQUEST, "Canonical voice reference is not a RIFF/WAVE file.")
    return data


def _validate_voice_cues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_VOICE_CUES:
        raise APIError(HTTPStatus.BAD_REQUEST, f"cues must contain 1 to {MAX_VOICE_CUES} dialogue cues.")
    voice = _character_voice_module()
    cues: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Voice cue {index} must be an object.")
        unknown = sorted(set(raw) - {"character", "text", "language", "start_seconds", "voice_description", "seed"})
        if unknown:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Unknown voice cue {index} field(s): " + ", ".join(unknown))
        character = _normalize_voice_character(raw.get("character"), field=f"Voice cue {index} character")
        text = _clean_text(
            raw.get("text"),
            field=f"Voice cue {index} text",
            max_chars=MAX_VOICE_TEXT_CHARS,
            max_bytes=6_400,
        )
        description = _clean_text(
            raw.get("voice_description", ""),
            field=f"Voice cue {index} delivery note",
            max_chars=MAX_VOICE_DESCRIPTION_CHARS,
            max_bytes=2_000,
        )
        language = raw.get("language")
        seed = raw.get("seed", 1234)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Voice cue {index} seed must be an integer from 0 to 4,294,967,295.")
        start_seconds = _finite_number(
            raw.get("start_seconds", 0),
            field=f"Voice cue {index} start_seconds",
            minimum=0,
            maximum=MAX_VOICE_START_SECONDS,
        )
        try:
            text = voice.validate_text(text)
            language = voice.validate_language(language if isinstance(language, str) else "")
            description = voice.validate_description(description)
            _character, preset, preset_entry, _reference = voice.resolve_preset(character=character, preset=None)
        except (voice.UserInputError, voice.PreflightError, OSError) as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Voice cue {index}: {exc}") from None
        cues.append(
            {
                "character": character,
                "preset": preset,
                "reference_sha256": preset_entry["sha256"],
                "text": text,
                "language": language,
                "start_seconds": start_seconds,
                "voice_description": description,
                "seed": seed,
            }
        )
    return cues


def _voice_subprocess_environment() -> dict[str, str]:
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
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    return environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_voice_result(
    payload: Any,
    *,
    cue: dict[str, Any],
    expected_audio: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Character Voice returned an unsupported result schema.")
    if VOICE_GENERATED_DIR.is_symlink() or GENERATED_DIR.is_symlink():
        raise RuntimeError("Managed voice output directories became unsafe.")
    audio_path = expected_audio.resolve(strict=True)
    manifest_path = expected_audio.with_suffix(".voice.json").resolve(strict=True)
    root = VOICE_GENERATED_DIR.resolve(strict=True)
    if root not in audio_path.parents or root not in manifest_path.parents:
        raise RuntimeError("Character Voice output escaped the managed voice directory.")
    if expected_audio.is_symlink() or expected_audio.with_suffix(".voice.json").is_symlink():
        raise RuntimeError("Character Voice returned a symbolic-link artifact.")
    if manifest_path.stat().st_size > 64 * 1024:
        raise RuntimeError("Character Voice manifest is unexpectedly large.")
    if payload.get("audio_path") != str(audio_path) or payload.get("manifest_path") != str(manifest_path):
        raise RuntimeError("Character Voice returned an unexpected artifact path.")
    if (
        payload.get("offline") is not True
        or payload.get("watermarked") is not True
        or payload.get("character") != cue["character"]
        or payload.get("preset") != cue["preset"]
        or payload.get("reference_sha256") != cue["reference_sha256"]
        or payload.get("language") != cue["language"]
        or payload.get("text_sha256") != hashlib.sha256(cue["text"].encode("utf-8")).hexdigest()
        or "text" in payload
    ):
        raise RuntimeError("Character Voice result does not match the isolated cue assignment.")
    if _sha256_file(audio_path) != payload.get("audio_sha256"):
        raise RuntimeError("Character Voice WAV failed its published SHA-256 check.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("audio_sha256") != payload.get("audio_sha256")
        or manifest.get("offline") is not True
        or manifest.get("audio_path") != str(audio_path)
        or manifest.get("character") != cue["character"]
        or manifest.get("preset") != cue["preset"]
        or manifest.get("reference_sha256") != cue["reference_sha256"]
        or manifest.get("language") != cue["language"]
        or manifest.get("watermarked") is not True
        or manifest.get("text_sha256") != payload.get("text_sha256")
        or "text" in manifest
    ):
        raise RuntimeError("Character Voice manifest does not authenticate the generated WAV.")
    voice = _character_voice_module()
    try:
        wave_info = voice.inspect_wave(audio_path, reference=False)
    except (voice.UserInputError, voice.PreflightError, OSError) as exc:
        raise RuntimeError(f"Character Voice WAV failed local validation: {exc}") from None
    if payload.get("channels") != wave_info.channels or payload.get("sample_rate") != wave_info.sample_rate:
        raise RuntimeError("Character Voice result metadata does not match the generated WAV.")
    duration = payload.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) <= 0:
        raise RuntimeError("Character Voice returned an invalid cue duration.")
    if abs(float(duration) - wave_info.duration_seconds) > 0.01:
        raise RuntimeError("Character Voice duration metadata does not match the generated WAV.")
    return {
        "character": cue["character"],
        "preset": cue["preset"],
        "language": cue["language"],
        "start_seconds": cue["start_seconds"],
        "seed": cue["seed"],
        "audio_path": audio_path,
        "manifest_path": manifest_path,
        "audio_sha256": payload["audio_sha256"],
        "duration_seconds": float(duration),
        "watermarked": payload.get("watermarked") is True,
    }


def _probe_local_video(path: Path, *, require_audio: bool = False) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for local video processing.")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or f"exit {completed.returncode}").strip()[-1_000:]
        raise RuntimeError(f"ffprobe could not validate the video: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from None
    streams = result.get("streams") if isinstance(result, dict) else None
    video_stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    ) if isinstance(streams, list) else None
    audio_stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"),
        None,
    ) if isinstance(streams, list) else None
    if video_stream is None:
        raise RuntimeError("The selected local MP4 has no video stream.")
    if require_audio and audio_stream is None:
        raise RuntimeError("The dialogue mux has no audio stream.")
    width = video_stream.get("width")
    height = video_stream.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise RuntimeError("The selected local MP4 has invalid video dimensions.")
    raw_duration = (result.get("format") or {}).get("duration") if isinstance(result, dict) else None
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("The selected local MP4 has no positive finite duration.")
    return {
        "duration_seconds": duration,
        "has_audio": audio_stream is not None,
        "width": width,
        "height": height,
        "video_codec": str(video_stream.get("codec_name") or ""),
        "pixel_format": str(video_stream.get("pix_fmt") or ""),
        "audio_codec": str(audio_stream.get("codec_name") or "") if audio_stream is not None else "",
    }


def _ai_upscale_readiness() -> dict[str, Any]:
    """Cheap readiness check; the wrapper performs pinned hashes before every launch."""
    required = (
        (AI_UPSCALER_PYTHON, "isolated Python runtime"),
        (AI_UPSCALER_SCRIPT, "streaming video wrapper"),
        (AI_UPSCALER_ROOT / "upscale.py", "pinned Core ML runtime"),
        (AI_UPSCALER_MODEL / "Manifest.json", "Core ML model manifest"),
        (AI_UPSCALER_MODEL_WEIGHT, "Real-ESRGAN x4plus weights"),
        (AI_UPSCALER_MODEL / "Data" / "com.apple.CoreML" / "model.mlmodel", "Core ML graph"),
    )
    missing = [label for path, label in required if not path.is_file()]
    if missing:
        return {
            "ready": False,
            "backend": "Core ML Real-ESRGAN x4plus",
            "target_short_edge": AI_UPSCALE_SHORT_EDGE,
            "note": "Missing " + ", ".join(missing) + ".",
        }
    if any(path.is_symlink() for path in (AI_UPSCALER_SCRIPT, AI_UPSCALER_ROOT / "upscale.py", AI_UPSCALER_MODEL_WEIGHT)):
        return {
            "ready": False,
            "backend": "Core ML Real-ESRGAN x4plus",
            "target_short_edge": AI_UPSCALE_SHORT_EDGE,
            "note": "The AI upscaler contains an unexpected symbolic link.",
        }
    try:
        if AI_UPSCALER_MODEL_WEIGHT.stat().st_size != 33_440_896:
            raise ValueError("weight size does not match the pinned release")
    except (OSError, ValueError) as exc:
        return {
            "ready": False,
            "backend": "Core ML Real-ESRGAN x4plus",
            "target_short_edge": AI_UPSCALE_SHORT_EDGE,
            "note": f"The AI model is incomplete or changed: {exc}.",
        }
    return {
        "ready": True,
        "backend": "Core ML Real-ESRGAN x4plus",
        "target_short_edge": AI_UPSCALE_SHORT_EDGE,
        "note": "Ready — pinned x4plus model and isolated Core ML runtime are installed.",
    }


def _upscale_target_dimensions(
    width: int,
    height: int,
    short_edge: int = FAST_UPSCALE_SHORT_EDGE,
) -> tuple[int, int]:
    """Scale the short edge to a requested even-pixel target without stretching."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("source dimensions must be positive integers")
    if isinstance(short_edge, bool) or not isinstance(short_edge, int) or short_edge < 2:
        raise ValueError("target short edge must be a positive integer")
    if min(width, height) >= short_edge:
        raise ValueError(f"This video is already {short_edge}p or larger on its short edge.")

    def even(value: float) -> int:
        return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)

    if width >= height:
        return even(width * short_edge / height), short_edge
    return short_edge, even(height * short_edge / width)


def _build_upscale_command(
    ffmpeg: str,
    source_video: Path,
    temporary_output: Path,
    target_width: int,
    target_height: int,
) -> list[str]:
    """Build a shell-free, aspect-preserving presentation upscale command."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-n",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale={target_width}:{target_height}:flags=lanczos,setsar=1,unsharp=5:5:0.35:5:5:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temporary_output),
    ]


def _build_quality_upscale_command(
    source_video: Path,
    temporary_output: Path,
    target_short_edge: int,
    ffmpeg: str,
    ffprobe: str,
) -> list[str]:
    """Build the shell-free streaming Core ML Real-ESRGAN command."""
    return [
        str(AI_UPSCALER_PYTHON),
        str(AI_UPSCALER_SCRIPT),
        "--input",
        str(source_video),
        "--output",
        str(temporary_output),
        "--target-short-edge",
        str(target_short_edge),
        "--ffmpeg",
        ffmpeg,
        "--ffprobe",
        ffprobe,
    ]


def _build_dialogue_mux_command(
    ffmpeg: str,
    source_video: Path,
    rendered_cues: list[dict[str, Any]],
    temporary_output: Path,
) -> list[str]:
    if not rendered_cues:
        raise ValueError("at least one rendered cue is required for muxing")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-n", "-i", str(source_video)]
    for cue in rendered_cues:
        command.extend(["-i", str(cue["audio_path"])])
    delayed: list[str] = []
    filters: list[str] = []
    for index, cue in enumerate(rendered_cues, start=1):
        label = f"cue{index}"
        milliseconds = int(round(float(cue["start_seconds"]) * 1_000.0))
        filters.append(f"[{index}:a:0]adelay={milliseconds}:all=1[{label}]")
        delayed.append(f"[{label}]")
    if len(delayed) == 1:
        filters.append(f"{delayed[0]}apad[dialogue]")
    else:
        filters.append(
            "".join(delayed)
            + f"amix=inputs={len(delayed)}:duration=longest:dropout_transition=0,apad[dialogue]"
        )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[dialogue]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(temporary_output),
        ]
    )
    return command


def _validate_motion_tracks(value: Any) -> list[list[dict[str, float]]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_TRACKS:
        raise APIError(HTTPStatus.BAD_REQUEST, f"tracks must contain 1 to {MAX_TRACKS} motion tracks.")
    tracks: list[list[dict[str, float]]] = []
    for track_index, raw_track in enumerate(value, start=1):
        if not isinstance(raw_track, list) or not 2 <= len(raw_track) <= MAX_TRACK_POINTS:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Motion track {track_index} must contain 2 to {MAX_TRACK_POINTS} points.",
            )
        track: list[dict[str, float]] = []
        for point_index, raw_point in enumerate(raw_track, start=1):
            if not isinstance(raw_point, dict) or set(raw_point) != {"x", "y"}:
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    f"Motion track {track_index} point {point_index} must contain exactly x and y.",
                )
            point: dict[str, float] = {}
            for axis in ("x", "y"):
                coordinate = raw_point[axis]
                if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                    raise APIError(
                        HTTPStatus.BAD_REQUEST,
                        f"Motion track {track_index} point {point_index} {axis} must be a JSON number from 0 to 1.",
                    )
                numeric = float(coordinate)
                if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                    raise APIError(
                        HTTPStatus.BAD_REQUEST,
                        f"Motion track {track_index} point {point_index} {axis} must be from 0 to 1.",
                    )
                point[axis] = numeric
            track.append(point)
        tracks.append(track)
    return tracks


def _validate_frame_control(value: Any, *, key: str, label: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise APIError(HTTPStatus.BAD_REQUEST, f"{key} must be an object.")
    unknown = sorted(set(value) - {"image", "description", "strength"})
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, f"Unknown {key} field(s): " + ", ".join(unknown))
    return {
        "image": _decode_image(value.get("image"), f"{label} image"),
        "description": _clean_text(
            value.get("description", ""),
            field=f"{label} description",
            max_chars=FRAME_DESCRIPTION_MAX_CHARS,
            max_bytes=FRAME_DESCRIPTION_MAX_BYTES,
        ),
        "strength": _finite_strength(value.get("strength"), field=f"{label} strength", default=1.0),
    }


def _validate_anchor_groups(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise APIError(HTTPStatus.BAD_REQUEST, "ingredients must be a list of anchor-group objects.")
    if len(value) > MAX_ANCHOR_GROUPS:
        raise APIError(HTTPStatus.BAD_REQUEST, f"At most {MAX_ANCHOR_GROUPS} anchor groups are allowed.")
    groups: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(value, start=1):
        if not isinstance(raw_group, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Anchor group {group_index} must be an object.")
        unknown = sorted(set(raw_group) - {"name", "description", "strength", "images"})
        if unknown:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown anchor group {group_index} field(s): " + ", ".join(unknown),
            )
        raw_images = raw_group.get("images", [])
        if not isinstance(raw_images, list):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Anchor group {group_index} images must be a list.")
        if len(raw_images) > MAX_ACTIVE_IMAGES:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Anchor group {group_index} has too many images.")
        name = _clean_text(
            raw_group.get("name", ""),
            field=f"Anchor group {group_index} name",
            max_chars=ANCHOR_NAME_MAX_CHARS,
            max_bytes=ANCHOR_NAME_MAX_BYTES,
        )
        description = _clean_text(
            raw_group.get("description", ""),
            field=f"Anchor group {group_index} description",
            max_chars=ANCHOR_DESCRIPTION_MAX_CHARS,
            max_bytes=ANCHOR_DESCRIPTION_MAX_BYTES,
        )
        images = [
            _decode_image(image, f"Anchor group {group_index} image {image_index}")
            for image_index, image in enumerate(raw_images, start=1)
        ]
        groups.append(
            {
                "name": name,
                "description": description,
                "strength": _finite_strength(
                    raw_group.get("strength"), field=f"Anchor group {group_index} strength", default=0.75
                ),
                "images": [image for image in images if image is not None],
            }
        )
    return groups


def _validate_ingredient_references(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INGREDIENT_REFERENCES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"ingredient_references must contain 1 to {MAX_INGREDIENT_REFERENCES} references.",
        )
    references: list[dict[str, Any]] = []
    decoded_total = 0
    seen_tags: set[str] = set()
    for index, raw_reference in enumerate(value, start=1):
        if not isinstance(raw_reference, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Ingredient reference {index} must be an object.")
        unknown = sorted(set(raw_reference) - {"image", "label", "tag", "description"})
        if unknown:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown ingredient reference {index} field(s): " + ", ".join(unknown),
            )
        image = _decode_image(raw_reference.get("image"), f"Ingredient reference {index} image")
        if image is None:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Ingredient reference {index} image is required.")
        label = _clean_text(
            raw_reference.get("label"),
            field=f"Ingredient reference {index} label",
            max_chars=100,
            max_bytes=400,
        )
        description = _clean_text(
            raw_reference.get("description"),
            field=f"Ingredient reference {index} description",
            max_chars=500,
            max_bytes=2_000,
        )
        if not label or not description:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Ingredient reference {index} needs both a label and description.",
            )
        raw_tag = raw_reference.get("tag")
        if raw_tag is None or raw_tag == "":
            derived = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
            if not derived or not derived[0].isalpha():
                derived = "ref_" + derived
            tag = "@" + derived[:32]
        else:
            tag = _clean_text(
                raw_tag,
                field=f"Ingredient reference {index} tag",
                max_chars=33,
                max_bytes=132,
            ).lower()
            if not tag.startswith("@"):
                tag = "@" + tag
        if not INGREDIENT_TAG_RE.fullmatch(tag):
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"Ingredient reference {index} tag must look like @maya and use only lowercase letters, numbers, or underscores.",
            )
        if tag in seen_tags:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Ingredient tag {tag} must be unique.")
        seen_tags.add(tag)
        decoded_total += len(image[1])
        references.append(
            {
                "image": image,
                "label": label,
                "tag": tag,
                "prompt_label": f"{tag} — {label}",
                "description": description,
            }
        )
    if decoded_total > MAX_TOTAL_REFERENCE_BYTES:
        raise APIError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "Ingredient reference images may total at most 64 MB decoded.",
        )
    return references


def _round_robin_images(groups: list[dict[str, Any]]) -> list[tuple[int, int, tuple[str, bytes]]]:
    ordered: list[tuple[int, int, tuple[str, bytes]]] = []
    depth = 0
    while True:
        added = False
        for group_index, group in enumerate(groups):
            if depth < len(group["images"]):
                ordered.append((group_index, depth, group["images"][depth]))
                added = True
        if not added:
            return ordered
        depth += 1


def _plan_conditionings(
    profile: str,
    start_frame: dict[str, Any],
    end_frame: dict[str, Any],
    groups: list[dict[str, Any]],
    legacy_image: tuple[str, bytes] | None = None,
    *,
    total_frames: int | None = None,
) -> list[dict[str, Any]]:
    if legacy_image is not None and start_frame["image"] is not None:
        raise APIError(HTTPStatus.BAD_REQUEST, "Use either reference_image or start_frame.image, not both.")
    start_image = start_frame["image"] if start_frame["image"] is not None else legacy_image
    interior = _round_robin_images(groups)
    image_count = (1 if start_image else 0) + len(interior) + (1 if end_frame["image"] else 0)
    if image_count > MAX_ACTIVE_IMAGES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            f"At most {MAX_ACTIVE_IMAGES} active images are allowed across start, end, and all anchor groups.",
        )
    decoded_bytes = sum(
        len(image[1])
        for image in ([start_image] if start_image else [])
        + [item[2] for item in interior]
        + ([end_frame["image"]] if end_frame["image"] else [])
    )
    if decoded_bytes > MAX_TOTAL_IMAGE_BYTES:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Active images may total at most 24 MB decoded.")

    last_frame = (PROFILE_FRAMES[profile] if total_frames is None else total_frames) - 1
    plan: list[dict[str, Any]] = []
    if start_image is not None:
        plan.append(
            {
                "role": "start",
                "label": "Start frame" if legacy_image is None else "Legacy start reference",
                "frame": 0,
                "strength": start_frame["strength"] if legacy_image is None else 1.0,
                "image": start_image,
            }
        )
    for position, (group_index, image_index, image) in enumerate(interior, start=1):
        frame = round(position * last_frame / (len(interior) + 1))
        group = groups[group_index]
        group_label = group["name"] or f"Anchor group {group_index + 1}"
        plan.append(
            {
                "role": "anchor",
                "label": f"{group_label} · view {image_index + 1}",
                "frame": frame,
                "strength": group["strength"],
                "image": image,
            }
        )
    if end_frame["image"] is not None:
        plan.append(
            {
                "role": "end",
                "label": "End frame",
                "frame": last_frame,
                "strength": end_frame["strength"],
                "image": end_frame["image"],
            }
        )
    return plan


def _build_conditioned_prompt(
    prompt: str,
    character: dict[str, str],
    has_reference: bool,
    start_description: str = "",
    end_description: str = "",
    groups: list[dict[str, Any]] | None = None,
) -> str:
    populated = [(label, character[key]) for key, label, _, _ in CHARACTER_FIELDS if character[key]]
    groups = groups or []
    anchor_notes: list[str] = []
    if start_description:
        anchor_notes.append(f"- Opening frame: {start_description}")
    if end_description:
        anchor_notes.append(f"- Closing frame: {end_description}")
    for index, group in enumerate(groups, start=1):
        if group["name"] or group["description"]:
            label = group["name"] or f"Anchor group {index}"
            detail = group["description"] or "Use the uploaded timed reference views for this group."
            anchor_notes.append(f"- {label}: {detail}")
    if not populated and not anchor_notes:
        conditioned = prompt
    else:
        sections: list[str] = []
        if populated:
            lines = [
                "CHARACTER CONTINUITY:",
                "Treat the following as fixed facts and keep them unchanged across every frame.",
            ]
            if has_reference:
                lines.append("The uploaded reference image is the primary visual identity source; preserve it closely.")
            lines.extend(f"- {label}: {value}" for label, value in populated)
            sections.append("\n".join(lines))
        if anchor_notes:
            sections.append("TIMELINE AND ANCHOR NOTES:\n" + "\n".join(anchor_notes))
        conditioned = "\n\n".join(sections) + "\n\nSHOT DESCRIPTION:\n" + prompt
    _validate_final_prompt_size(
        conditioned,
        label="The shot, Character Bible, and anchor notes",
    )
    return conditioned


def _build_generator_command(job: dict[str, Any], prompt: str) -> list[str]:
    """Build one shell-free argv list for the validated job snapshot."""
    command = [
        str(GENERATOR_PYTHON),
        str(GENERATOR_SCRIPT),
        "--profile",
        job["profile"],
        "--seed",
        str(job["seed"]),
    ]
    aspect_ratio = job.get("aspect_ratio", "profile")
    if aspect_ratio != "profile":
        command.extend(["--aspect-ratio", aspect_ratio])
    gemma_pack = job.get("gemma_pack")
    if gemma_pack is not None:
        if gemma_pack not in {"official", "ddalcu-fallback"}:
            raise ValueError(f"unsupported Gemma pack: {gemma_pack}")
        command.extend(["--gemma-pack", gemma_pack])
    if job.get("duration_seconds") is not None:
        command.extend(["--duration-seconds", str(job["duration_seconds"])])
    command.extend(["--output", str(job["output_path"])])
    for conditioning in job["conditionings"]:
        if conditioning["role"] == "start":
            command.extend(
                ["--start-image", str(conditioning["path"]), "--start-strength", str(conditioning["strength"])]
            )
        elif conditioning["role"] == "end":
            command.extend(
                ["--end-image", str(conditioning["path"]), "--end-strength", str(conditioning["strength"])]
            )
        else:
            command.extend(
                [
                    "--anchor",
                    str(conditioning["path"]),
                    str(conditioning["frame"]),
                    str(conditioning["strength"]),
                ]
            )
    for scene in job.get("scenes", []):
        command.extend(["--scene", scene["text"]])
        if scene["length"] is not None:
            command.extend(["--scene-length", str(scene["length"])])
    command.extend(["--", prompt])
    return command


def _append_conditioning_args(command: list[str], conditionings: list[dict[str, Any]]) -> None:
    for conditioning in conditionings:
        if conditioning["role"] == "start":
            command.extend(
                ["--start-image", str(conditioning["path"]), "--start-strength", str(conditioning["strength"])]
            )
        elif conditioning["role"] == "end":
            command.extend(
                ["--end-image", str(conditioning["path"]), "--end-strength", str(conditioning["strength"])]
            )
        else:
            command.extend(
                ["--anchor", str(conditioning["path"]), str(conditioning["frame"]), str(conditioning["strength"])]
            )


def _build_edit_command(job: dict[str, Any], prompt: str) -> list[str]:
    """Build a shell-free edit/A2V wrapper command from a staged job."""
    mode = job["mode"]
    if mode not in EDIT_MODES:
        raise ValueError(f"unsupported edit mode: {mode}")
    command = [
        str(GENERATOR_PYTHON),
        str(EDIT_SCRIPT),
        mode,
        "--output",
        str(job["output_path"]),
        "--seed",
        str(job["seed"]),
    ]
    inputs = job.get("input_paths", {})
    if mode == "retake":
        command.extend(
            [
                "--video",
                str(inputs["source_video"]),
                "--start-seconds",
                str(job["start_seconds"]),
                "--end-seconds",
                str(job["end_seconds"]),
            ]
        )
        if job.get("preserve_audio"):
            command.append("--preserve-audio")
    elif mode == "extend":
        command.extend(
            [
                "--video",
                str(inputs["source_video"]),
                "--extend-seconds",
                str(job["extend_seconds"]),
                "--direction",
                job["direction"],
            ]
        )
    else:
        command.extend(
            [
                "--audio",
                str(inputs["audio"]),
                "--audio-start",
                str(job["audio_start"]),
                "--profile",
                job["profile"],
                "--frames",
                str(24 * job["duration_seconds"] + 1),
            ]
        )
        aspect_ratio = job.get("aspect_ratio", "profile")
        if aspect_ratio != "profile":
            command.extend(["--aspect-ratio", aspect_ratio])
        start_image = inputs.get("a2v_start_image")
        if start_image is not None:
            command.extend(
                ["--start-image", str(start_image), "--image-strength", str(job["image_strength"])]
            )
    command.extend(["--", prompt])
    return command


def _build_condition_command(job: dict[str, Any], prompt: str) -> list[str]:
    """Build a shell-free Ingredients/Union/Motion wrapper command."""
    mode = job["mode"]
    if mode not in CONDITION_MODES:
        raise ValueError(f"unsupported conditioning mode: {mode}")
    command = [
        str(GENERATOR_PYTHON),
        str(CONDITION_SCRIPT),
        mode,
        "--profile",
        job["profile"],
        "--output",
        str(job["output_path"]),
        "--seed",
        str(job["seed"]),
        "--lora-strength",
        str(job["lora_strength"]),
        "--conditioning-strength",
        str(job["conditioning_strength"]),
    ]
    if mode != "ingredients":
        command.extend(["--duration-seconds", str(job["duration_seconds"])])
    aspect_ratio = job.get("aspect_ratio", "profile")
    if aspect_ratio != "profile":
        command.extend(["--aspect-ratio", aspect_ratio])
    _append_conditioning_args(command, job.get("conditionings", []))
    inputs = job.get("input_paths", {})
    if mode == "ingredients":
        for reference in job["ingredient_references"]:
            command.extend(
                [
                    "--reference",
                    str(reference["path"]),
                    reference.get("prompt_label", reference["label"]),
                    reference["description"],
                ]
            )
    elif mode == "union":
        command.extend(["--control-type", job["control_type"]])
        option = "--source-video" if job["union_route"] == "source" else "--guide-video"
        command.extend([option, str(inputs["control_video"])])
    else:
        command.extend(["--tracks-json", str(inputs["tracks_json"])])
    command.extend(["--", prompt])
    return command


_DENOISE_FRACTION_RE = re.compile(
    r"(?:denois\w*|diffusion|sampl\w*|step)\D{0,40}(\d+)\s*/\s*(\d+)", re.IGNORECASE
)
_DENOISE_PERCENT_RE = re.compile(
    r"(?:denois\w*|diffusion|sampl\w*|step).*?(\d{1,3})%\|", re.IGNORECASE
)


def _update_progress_from_line(job: dict[str, Any], line: str) -> None:
    """Update honest stage/progress metadata from one real subprocess line."""
    lowered = line.lower()
    stage: str | None = None
    if any(token in lowered for token in ("preflight", "pre-flight", "verifying", "verify ", "checking asset")):
        stage = "preflight"
    elif any(token in lowered for token in ("loading", "load model", "load weight", "initializing model")):
        stage = "loading"
    elif any(token in lowered for token in ("preprocess", "reference sheet", "control guide", "canny")):
        stage = "preprocessing"
    elif any(token in lowered for token in ("denois", "diffusion", "sampling", "step")):
        stage = "denoising"
    elif any(token in lowered for token in ("decod", "vae decode")):
        stage = "decoding"
    elif any(token in lowered for token in ("encod", "ffmpeg", "mux")):
        stage = "encoding"
    elif any(token in lowered for token in ("saving", "saved", "writing output", "wrote ", "finaliz")):
        stage = "finalizing"
    if stage is not None and stage != job.get("stage"):
        job["stage"] = stage
        if stage != "denoising":
            job["progress_percent"] = None

    fraction = _DENOISE_FRACTION_RE.search(line)
    percent = _DENOISE_PERCENT_RE.search(line)
    parsed: float | None = None
    if fraction:
        current, total = (int(fraction.group(1)), int(fraction.group(2)))
        if total > 0 and 0 <= current <= total:
            parsed = current * 100.0 / total
    elif percent:
        candidate = int(percent.group(1))
        if 0 <= candidate <= 100:
            parsed = float(candidate)
    if parsed is not None:
        job["stage"] = "denoising"
        job["progress_percent"] = round(parsed, 1)


class JobState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in {"starting", "running", "cancelling"})

    def restore_latest_completed(self) -> dict[str, Any]:
        """Restore the newest validated primary render after a server restart."""
        with self._lock:
            if self._job is not None:
                return self.snapshot()
        if GENERATED_DIR.is_symlink():
            return self.snapshot()
        try:
            root = GENERATED_DIR.resolve(strict=True)
            entries = list(root.iterdir())
        except OSError:
            return self.snapshot()

        candidates: list[tuple[int, str, Path, re.Match[str]]] = []
        for entry in entries:
            match = RESTORABLE_OUTPUT_RE.fullmatch(entry.name)
            if match is None:
                continue
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                resolved = entry.resolve(strict=True)
                candidate_stat = entry.stat()
            except OSError:
                continue
            if resolved.parent != root or candidate_stat.st_size <= 0:
                continue
            candidates.append((candidate_stat.st_mtime_ns, entry.name, resolved, match))

        for _mtime_ns, _name, candidate, match in sorted(candidates, reverse=True):
            try:
                # Probe every candidate before exposing it in the player or to
                # the upscaler.  If the newest is damaged, continue to the next
                # safe primary render instead of making startup fail.
                info = _probe_local_video(candidate)
                candidate_stat = candidate.stat()
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            try:
                if candidate.is_symlink() or candidate.resolve(strict=True).parent != root:
                    continue
            except OSError:
                continue

            label = match.group("label")
            mode = label if label in WORKFLOW_MODES else "generate"
            restored_at = datetime.fromtimestamp(candidate_stat.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            now_mono = time.monotonic()
            restored_id = "restored-" + hashlib.sha256(candidate.name.encode("utf-8")).hexdigest()[:16]
            restored_job = {
                "id": restored_id,
                "status": "succeeded",
                "stage": "complete",
                "progress_percent": 100.0,
                "mode": mode,
                "profile": label,
                "aspect_ratio": _suggest_capture_aspect_ratio(
                    int(info["width"]), int(info["height"]), None
                ),
                "seed": None,
                "gemma_pack": None,
                "duration_seconds": float(info["duration_seconds"]),
                "started_at": restored_at,
                "finished_at": restored_at,
                "_started_mono": now_mono,
                "_finished_mono": now_mono,
                "log": f"[ui] Restored previous completed video: {candidate.name}\n",
                "error": None,
                "output_path": candidate,
                "public_output_path": candidate,
                "conditionings": [],
                "scenes": [],
                "generated_keyframes": 0,
                "temp_paths": [],
                "process": None,
                "cancel_requested": False,
                "restored": True,
            }
            with self._lock:
                if self._job is None:
                    self._job = restored_job
                return self.snapshot()
        return self.snapshot()

    def completed_output(self) -> tuple[str, Path]:
        """Return the latest completed UI video and job ID after confinement checks."""
        with self._lock:
            if self._job is None or self._job.get("status") != "succeeded":
                raise APIError(HTTPStatus.CONFLICT, "Generate a video successfully before using the completed result.")
            job_id = self._job.get("id")
            raw = self._job.get("public_output_path", self._job.get("output_path"))
        if not isinstance(job_id, str) or not isinstance(raw, Path):
            raise APIError(HTTPStatus.CONFLICT, "The latest completed video is unavailable.")
        if GENERATED_DIR.is_symlink() or raw.is_symlink():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The latest completed video path is unsafe.")
        try:
            root = GENERATED_DIR.resolve(strict=True)
            resolved = raw.resolve(strict=True)
        except OSError as exc:
            raise APIError(HTTPStatus.CONFLICT, f"The latest completed video is unavailable: {exc}") from None
        if resolved.parent != root or not MEDIA_NAME_RE.fullmatch(resolved.name) or not resolved.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The latest completed video is outside the managed output directory.")
        return job_id, resolved

    def completed_output_for_voice(self) -> Path:
        """Return the latest completed UI video for the optional dialogue mux."""
        return self.completed_output()[1]

    def _append(self, job_id: str, message: str) -> None:
        clean = ANSI_RE.sub("", message).replace("\x00", "")
        if len(clean) > 32_000:
            clean = clean[-32_000:]
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            self._job["log"] = (self._job["log"] + clean)[-MAX_LOG_CHARS:]
            _update_progress_from_line(self._job, clean)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {"job": None}
            job = self._job
            now = time.monotonic()
            elapsed = (job.get("_finished_mono") or now) - job["_started_mono"]
            public = {
                "id": job["id"],
                "status": job["status"],
                "mode": job.get("mode", "generate"),
                "profile": job.get("profile"),
                "aspect_ratio": job.get("aspect_ratio", "profile"),
                "seed": job["seed"],
                "gemma_pack": job.get("gemma_pack"),
                "stage": job.get("stage", "queued"),
                "progress_percent": job.get("progress_percent"),
                "duration_seconds": job.get("duration_seconds"),
                "started_at": job["started_at"],
                "finished_at": job.get("finished_at"),
                "elapsed_seconds": round(max(0.0, elapsed), 1),
                "log": job["log"],
                "error": job.get("error"),
                "restored": bool(job.get("restored")),
                "anchor_count": len(job.get("conditionings", [])),
                "anchor_plan": [
                    {"label": item["label"], "frame": item["frame"], "strength": item["strength"]}
                    for item in job.get("conditionings", [])
                ],
                "scene_count": len(job.get("scenes", [])),
                "generated_keyframes": job.get("generated_keyframes", 0),
                "output_url": (
                    "/media/" + quote(job.get("public_output_path", job["output_path"]).name)
                    if job["status"] == "succeeded"
                    else None
                ),
            }
            return {"job": public}

    @staticmethod
    def _remove_temp_paths(paths: list[Path]) -> list[str]:
        warnings: list[str] = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                warnings.append(path.name)
        return warnings

    def start(
        self,
        prompt: str,
        profile: str,
        seed: int,
        conditionings: list[dict[str, Any]],
        scenes: list[dict[str, Any]] | None = None,
        generated_keyframes: int = 0,
        duration_seconds: int | None = None,
        gemma_pack: str = "official",
        aspect_ratio: str = "profile",
    ) -> dict[str, Any]:
        _validate_generated_keyframes(generated_keyframes)
        aspect_ratio = _validate_aspect_ratio(aspect_ratio)
        if gemma_pack not in {"official", "ddalcu-fallback"}:
            raise APIError(HTTPStatus.BAD_REQUEST, "Gemma pack must be official or ddalcu-fallback.")
        if not GENERATOR_SCRIPT.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, f"Generator is missing: {GENERATOR_SCRIPT.name}")
        if not GENERATOR_PYTHON.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The LTX-2 MLX Python environment is missing.")
        with self._lock:
            if self._job and self._job["status"] in {"starting", "running", "cancelling"}:
                raise APIError(HTTPStatus.CONFLICT, "A video is already rendering. Cancel it before starting another.")
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = (GENERATED_DIR / f"ltx25-{profile}-{stamp}-{job_id[:8]}.mp4").resolve()
            temp_paths: list[Path] = []
            saved_conditionings: list[dict[str, Any]] = []
            try:
                for index, conditioning in enumerate(conditionings, start=1):
                    extension, payload = conditioning["image"]
                    handle = tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".ltx-anchor-{job_id[:8]}-{index}-",
                        suffix=f".{extension}",
                        dir=GENERATED_DIR,
                        delete=False,
                    )
                    path = Path(handle.name).resolve()
                    temp_paths.append(path)
                    try:
                        os.fchmod(handle.fileno(), 0o600)
                        handle.write(payload)
                        handle.flush()
                    finally:
                        handle.close()
                    saved = {key: value for key, value in conditioning.items() if key != "image"}
                    saved["path"] = path
                    saved_conditionings.append(saved)
            except Exception as exc:
                self._remove_temp_paths(temp_paths)
                raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not securely stage reference images: {exc}") from None
            plan_lines: list[str] = []
            for item in saved_conditionings:
                safe_label = re.sub(r"[\r\n\t]+", " ", ANSI_RE.sub("", item["label"]))
                plan_lines.append(
                    f"[ui] Planned {safe_label} at frame {item['frame']} (strength {item['strength']:.2f}).\n"
                )
            plan_log = "".join(plan_lines)
            scenes = [dict(scene) for scene in (scenes or [])]
            if scenes:
                plan_log += f"[ui] Planned {len(scenes)} Prompt Relay scene beat(s).\n"
            if generated_keyframes:
                plan_log += f"[ui] Planned {generated_keyframes} generated keyframe slot(s).\n"
            self._job = {
                "id": job_id,
                "status": "starting",
                "stage": "queued",
                "progress_percent": None,
                "mode": "generate",
                "profile": profile,
                "aspect_ratio": aspect_ratio,
                "seed": seed,
                "gemma_pack": gemma_pack,
                "duration_seconds": duration_seconds,
                "started_at": _utc_now(),
                "_started_mono": time.monotonic(),
                "finished_at": None,
                "_finished_mono": None,
                "log": f"[ui] Queued {profile} render with seed {seed}.\n" + plan_log,
                "error": None,
                "output_path": output_path,
                "public_output_path": output_path,
                "conditionings": saved_conditionings,
                "scenes": scenes,
                "generated_keyframes": generated_keyframes,
                "temp_paths": temp_paths,
                "process": None,
                "cancel_requested": False,
            }
        thread = threading.Thread(
            target=self._run, args=(job_id, prompt), name=f"ltx-render-{job_id[:8]}", daemon=True
        )
        thread.start()
        return self.snapshot()

    def start_workflow(self, prompt: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Stage validated local inputs and queue a non-generate workflow."""
        mode = spec["mode"]
        script = EDIT_SCRIPT if mode in EDIT_MODES else CONDITION_SCRIPT
        if not script.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, f"Workflow launcher is missing: {script.name}")
        if not GENERATOR_PYTHON.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The LTX-2 MLX Python environment is missing.")
        with self._lock:
            if self._job and self._job["status"] in {"starting", "running", "cancelling"}:
                raise APIError(HTTPStatus.CONFLICT, "A video is already rendering. Cancel it before starting another.")
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            if mode in EDIT_MODES:
                EDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            profile_label = spec.get("profile") or mode
            public_output_path = (GENERATED_DIR / f"ltx25-{mode}-{stamp}-{job_id[:8]}.mp4").resolve()
            output_root = EDIT_OUTPUT_DIR if mode in EDIT_MODES else GENERATED_DIR
            output_path = (output_root / public_output_path.name).resolve()
            temp_paths: list[Path] = []
            staged_inputs: dict[str, Path] = {}
            saved_conditionings: list[dict[str, Any]] = []
            saved_references: list[dict[str, Any]] = []
            payload_index = 0

            def stage_payload(extension: str, payload: bytes, kind: str) -> Path:
                nonlocal payload_index
                payload_index += 1
                suffix = extension if extension.startswith(".") else "." + extension
                handle = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".ltx-ui-{job_id[:8]}-{kind}-{payload_index}-",
                    suffix=suffix,
                    dir=GENERATED_DIR,
                    delete=False,
                )
                path = Path(handle.name).resolve()
                temp_paths.append(path)
                try:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(payload)
                    handle.flush()
                finally:
                    handle.close()
                return path

            try:
                decoded_total = 0
                for key, upload in spec.pop("uploads", {}).items():
                    extension, payload, _original_name = upload
                    decoded_total += len(payload)
                    staged_inputs[key] = stage_payload(extension, payload, key)
                for conditioning in spec.pop("conditionings", []):
                    extension, payload = conditioning["image"]
                    decoded_total += len(payload)
                    saved = {key: value for key, value in conditioning.items() if key != "image"}
                    saved["path"] = stage_payload(extension, payload, "anchor")
                    saved_conditionings.append(saved)
                for reference in spec.pop("ingredient_references", []):
                    extension, payload = reference["image"]
                    decoded_total += len(payload)
                    saved = {key: value for key, value in reference.items() if key != "image"}
                    saved["path"] = stage_payload(extension, payload, "ingredient")
                    saved_references.append(saved)
                tracks = spec.pop("tracks", None)
                if tracks is not None:
                    tracks_payload = json.dumps(tracks, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    decoded_total += len(tracks_payload)
                    staged_inputs["tracks_json"] = stage_payload(".json", tracks_payload, "tracks")
                if decoded_total > MAX_TOTAL_REQUEST_UPLOAD_BYTES:
                    raise APIError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "Uploaded files may total at most 96 MB decoded.",
                    )
            except APIError:
                self._remove_temp_paths(temp_paths)
                raise
            except Exception as exc:
                self._remove_temp_paths(temp_paths)
                raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not securely stage local inputs: {exc}") from None

            job = dict(spec)
            job.update(
                {
                    "id": job_id,
                    "status": "starting",
                    "stage": "queued",
                    "progress_percent": None,
                    "profile": spec.get("profile", profile_label),
                    "started_at": _utc_now(),
                    "_started_mono": time.monotonic(),
                    "finished_at": None,
                    "_finished_mono": None,
                    "log": f"[ui] Queued {mode} workflow with seed {spec['seed']}.\n",
                    "error": None,
                    "output_path": output_path,
                    "public_output_path": public_output_path,
                    "input_paths": staged_inputs,
                    "conditionings": saved_conditionings,
                    "ingredient_references": saved_references,
                    "temp_paths": temp_paths,
                    "process": None,
                    "cancel_requested": False,
                }
            )
            self._job = job
        thread = threading.Thread(
            target=self._run,
            args=(job_id, prompt),
            name=f"ltx-{mode}-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return self.snapshot()

    def _run(self, job_id: str, prompt: str) -> None:
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            job = self._job
            if job["cancel_requested"]:
                self._finish_locked(job, "cancelled", "Render cancelled before launch.")
                warnings = self._remove_temp_paths(job["temp_paths"])
                if warnings:
                    job["log"] += "[ui] Warning: temporary anchor image cleanup was incomplete.\n"
                return
            mode = job.get("mode", "generate")
            temp_paths = list(job["temp_paths"])
            output_path = job["output_path"]
            public_output_path = job.get("public_output_path", output_path)
            sandbox_block = _metal_sandbox_block_reason()
            if sandbox_block:
                self._finish_locked(job, "failed", sandbox_block)
                if self._remove_temp_paths(temp_paths):
                    job["log"] += "[ui] Warning: temporary input cleanup was incomplete.\n"
                return
            try:
                if mode == "generate":
                    command = _build_generator_command(job, prompt)
                elif mode in EDIT_MODES:
                    command = _build_edit_command(job, prompt)
                else:
                    command = _build_condition_command(job, prompt)
            except Exception as exc:
                self._finish_locked(job, "failed", f"Could not build the {mode} command: {exc}")
                if self._remove_temp_paths(temp_paths):
                    job["log"] += "[ui] Warning: temporary input cleanup was incomplete.\n"
                return
            job["stage"] = "preflight"

        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=HERE,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    self._terminate(process)
                    return
                self._job["process"] = process
                cancel_requested = self._job["cancel_requested"]
                self._job["status"] = "cancelling" if cancel_requested else "running"
            self._append(job_id, f"[ui] {mode} launcher started (PID {process.pid}).\n")
            if cancel_requested:
                self._terminate(process)
            assert process.stdout is not None
            for line in process.stdout:
                self._append(job_id, line)
            return_code = process.wait()
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    return
                job = self._job
                if job["cancel_requested"]:
                    self._finish_locked(job, "cancelled", "Render cancelled.")
                elif return_code != 0:
                    self._finish_locked(job, "failed", f"Generator exited with status {return_code}.")
                elif not output_path.is_file() or output_path.stat().st_size == 0:
                    self._finish_locked(job, "failed", "Generator finished without creating a video.")
                else:
                    if output_path != public_output_path:
                        try:
                            os.replace(output_path, public_output_path)
                        except OSError as exc:
                            self._finish_locked(job, "failed", f"Could not publish the completed video: {exc}")
                            return
                    self._finish_locked(job, "succeeded", None)
                    self._append(job_id, f"[ui] Video ready: {public_output_path.name}\n")
        except Exception as exc:
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "failed", f"Could not run the generator: {exc}")
        finally:
            try:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass
            if self._remove_temp_paths(temp_paths):
                self._append(job_id, "[ui] Warning: one or more temporary anchor images could not be removed.\n")
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    status = self._job["status"]
                    self._job["process"] = None
                    if status in {"failed", "cancelled"}:
                        for unfinished_path in {output_path, public_output_path}:
                            try:
                                unfinished_path.unlink(missing_ok=True)
                            except OSError:
                                pass

    def _finish_locked(self, job: dict[str, Any], status: str, error: str | None) -> None:
        job["status"] = status
        job["stage"] = "complete" if status == "succeeded" else status
        if status != "succeeded" or job.get("progress_percent") != 100.0:
            job["progress_percent"] = None
        job["error"] = error
        job["finished_at"] = _utc_now()
        job["_finished_mono"] = time.monotonic()
        if error:
            job["log"] = (job["log"] + f"[ui] {error}\n")[-MAX_LOG_CHARS:]

    @staticmethod
    def _terminate(process: subprocess.Popen[str], force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None or self._job["status"] not in {"starting", "running", "cancelling"}:
                raise APIError(HTTPStatus.CONFLICT, "There is no active render to cancel.")
            self._job["cancel_requested"] = True
            self._job["status"] = "cancelling"
            process = self._job["process"]
            job_id = self._job["id"]
            self._job["log"] = (self._job["log"] + "[ui] Cancellation requested.\n")[-MAX_LOG_CHARS:]
        if process is not None:
            self._terminate(process)

            def force_after_grace() -> None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append(job_id, "[ui] Generator did not stop; forcing termination.\n")
                    self._terminate(process, force=True)

            threading.Thread(target=force_after_grace, name=f"ltx-cancel-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            process = self._job.get("process") if self._job else None
        if process is not None:
            self._terminate(process)


STATE = JobState()


class _UpscaleCancelled(Exception):
    pass


class UpscaleState:
    """Create one validated presentation-size derivative without replacing its source."""

    ACTIVE = {"starting", "running", "cancelling"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None
        self._worker: threading.Thread | None = None

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in self.ACTIVE)

    def snapshot(self) -> dict[str, Any]:
        ai_readiness = _ai_upscale_readiness()
        with self._lock:
            if self._job is None:
                return {"job": None, "ai_readiness": ai_readiness}
            job = self._job
            elapsed = (job.get("_finished_mono") or time.monotonic()) - job["_started_mono"]
            return {
                "ai_readiness": ai_readiness,
                "job": {
                    "id": job["id"],
                    "source_job_id": job["source_job_id"],
                    "profile": job["profile"],
                    "backend": job["backend"],
                    "status": job["status"],
                    "stage": job["stage"],
                    "progress_percent": job.get("progress_percent"),
                    "completed_frames": job.get("completed_frames"),
                    "total_frames": job.get("total_frames"),
                    "source_width": job["source_width"],
                    "source_height": job["source_height"],
                    "output_width": job["output_width"],
                    "output_height": job["output_height"],
                    "started_at": job["started_at"],
                    "finished_at": job.get("finished_at"),
                    "elapsed_seconds": round(max(0.0, elapsed), 2),
                    "error": job.get("error"),
                    "log_tail": job.get("log", "")[-4_000:],
                    "output_url": (
                        "/media/" + quote(job["output_path"].name, safe="")
                        if job["status"] == "succeeded"
                        else None
                    ),
                }
            }

    def start(self, profile: str = "fast720", target_short_edge: int | None = None) -> dict[str, Any]:
        if not isinstance(profile, str) or profile not in UPSCALE_PROFILES:
            raise APIError(HTTPStatus.BAD_REQUEST, "profile must be fast720 or quality-ai.")
        expected_edge = AI_UPSCALE_SHORT_EDGE if profile == "quality-ai" else FAST_UPSCALE_SHORT_EDGE
        if target_short_edge is None:
            target_short_edge = expected_edge
        if isinstance(target_short_edge, bool) or not isinstance(target_short_edge, int) or target_short_edge != expected_edge:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"{profile} requires target_short_edge={expected_edge}.",
            )
        if profile == "quality-ai":
            readiness = _ai_upscale_readiness()
            if not readiness["ready"]:
                raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(readiness["note"]))
        with self._lock:
            if self._job and self._job["status"] in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "An upscale is already running.")
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "ffmpeg and ffprobe are required for video upscaling.")
        source_job_id, source_video = STATE.completed_output()
        if GENERATED_DIR.is_symlink() or source_video.is_symlink() or not source_video.is_file():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The completed video path is unsafe or unavailable.")
        try:
            generated_root = GENERATED_DIR.resolve(strict=True)
            resolved_source = source_video.resolve(strict=True)
        except OSError as exc:
            raise APIError(HTTPStatus.CONFLICT, f"The completed video is unavailable: {exc}") from None
        if resolved_source.parent != generated_root or not MEDIA_NAME_RE.fullmatch(resolved_source.name):
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "The completed video escaped the managed output directory.")
        try:
            source_info = _probe_local_video(resolved_source)
            target_width, target_height = _upscale_target_dimensions(
                int(source_info["width"]), int(source_info["height"]), target_short_edge
            )
            source_stat = resolved_source.stat()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None

        with self._lock:
            if self._job and self._job["status"] in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "An upscale is already running.")
            job_id = uuid.uuid4().hex
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            artifact_label = "ai-quality-1080" if profile == "quality-ai" else "fast-720p"
            output_path = (GENERATED_DIR / f"ltx25-{artifact_label}-{stamp}-{job_id[:8]}.mp4").resolve()
            temporary_path = (GENERATED_DIR / f".ltx25-{artifact_label}-{job_id}-{uuid.uuid4().hex}.partial.mp4").resolve()
            if (
                output_path.parent != generated_root
                or temporary_path.parent != generated_root
                or output_path.exists()
                or output_path.is_symlink()
                or temporary_path.exists()
                or temporary_path.is_symlink()
            ):
                raise APIError(HTTPStatus.CONFLICT, "Refusing to replace an existing upscale artifact.")
            self._job = {
                "id": job_id,
                "source_job_id": source_job_id,
                "profile": profile,
                "backend": "Core ML Real-ESRGAN x4plus" if profile == "quality-ai" else "FFmpeg Lanczos",
                "target_short_edge": target_short_edge,
                "source_path": resolved_source,
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_width": int(source_info["width"]),
                "source_height": int(source_info["height"]),
                "source_duration": float(source_info["duration_seconds"]),
                "source_has_audio": bool(source_info["has_audio"]),
                "output_width": target_width,
                "output_height": target_height,
                "output_path": output_path,
                "temporary_path": temporary_path,
                "ffmpeg": ffmpeg,
                "ffprobe": ffprobe,
                "status": "starting",
                "stage": "queued",
                "progress_percent": 0.0,
                "completed_frames": 0,
                "total_frames": None,
                "started_at": _utc_now(),
                "finished_at": None,
                "_started_mono": time.monotonic(),
                "_finished_mono": None,
                "error": None,
                "_worker_error": None,
                "log": (
                    f"[upscale] Queued {profile}: {source_info['width']}x{source_info['height']} -> "
                    f"{target_width}x{target_height}; original preserved.\n"
                ),
                "process": None,
                "cancel_requested": False,
            }
            worker = threading.Thread(
                target=self._run,
                args=(job_id,),
                name=f"ltx-upscale-{job_id[:8]}",
                daemon=True,
            )
            self._worker = worker
        worker.start()
        return self.snapshot()

    def _append(self, job_id: str, message: str) -> None:
        clean = ANSI_RE.sub("", message).replace("\x00", "")[-32_000:]
        with self._lock:
            if self._job is not None and self._job["id"] == job_id:
                self._job["log"] = (self._job["log"] + clean)[-MAX_LOG_CHARS:]

    def _consume_quality_event(self, job_id: str, line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        phase = event.get("phase")
        completed = event.get("completed")
        total = event.get("total")
        progress = event.get("progress")
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return True
            if isinstance(phase, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", phase):
                self._job["stage"] = phase.replace("_", "-")
            if isinstance(completed, int) and not isinstance(completed, bool) and completed >= 0:
                self._job["completed_frames"] = completed
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                self._job["total_frames"] = total
            if isinstance(progress, (int, float)) and not isinstance(progress, bool) and math.isfinite(progress):
                percent = float(progress) * 100.0 if float(progress) <= 1.0 else float(progress)
                self._job["progress_percent"] = round(min(99.0, max(0.0, percent)), 2)
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                clean = ANSI_RE.sub("", message).replace("\x00", "")[:1_000]
                self._job["log"] = (self._job["log"] + f"[ai-upscale] {clean}\n")[-MAX_LOG_CHARS:]
            error = event.get("error")
            if isinstance(error, str) and error.strip():
                clean_error = ANSI_RE.sub("", error).replace("\x00", "")[:2_000]
                self._job["_worker_error"] = clean_error
                self._job["log"] = (
                    self._job["log"] + f"[ai-upscale] Detail: {clean_error}\n"
                )[-MAX_LOG_CHARS:]
        return True

    @staticmethod
    def _remove_temporary(path: Path) -> bool:
        try:
            if path.is_symlink():
                return False
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _run(self, job_id: str) -> None:
        process: subprocess.Popen[str] | None = None
        temporary_path: Path | None = None
        try:
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    return
                job = self._job
                temporary_path = job["temporary_path"]
                if job["cancel_requested"]:
                    raise _UpscaleCancelled("Upscale cancelled before launch.")
                if job["profile"] == "quality-ai":
                    command = _build_quality_upscale_command(
                        job["source_path"],
                        temporary_path,
                        job["target_short_edge"],
                        job["ffmpeg"],
                        job["ffprobe"],
                    )
                    job["stage"] = "loading-ai-model"
                else:
                    command = _build_upscale_command(
                        job["ffmpeg"],
                        job["source_path"],
                        temporary_path,
                        job["output_width"],
                        job["output_height"],
                    )
                    job["stage"] = "encoding"
            process = subprocess.Popen(
                command,
                cwd=HERE,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    self._interrupt(process)
                    return
                self._job["process"] = process
                if self._job["cancel_requested"]:
                    self._job["status"] = "cancelling"
                    self._interrupt(process)
                else:
                    self._job["status"] = "running"
            assert process.stdout is not None
            for line in process.stdout:
                if not self._consume_quality_event(job_id, line):
                    self._append(job_id, line)
            return_code = process.wait()
            with self._lock:
                if self._job is None or self._job["id"] != job_id or self._job["cancel_requested"]:
                    raise _UpscaleCancelled("Upscale was cancelled.")
                if return_code != 0:
                    engine = "AI upscaler" if self._job["profile"] == "quality-ai" else "ffmpeg"
                    detail = self._job.get("_worker_error")
                    suffix = f" {detail}" if detail else ""
                    raise RuntimeError(f"The {engine} could not create the video (exit {return_code}).{suffix}")
                self._job["stage"] = "verifying"
                self._job["progress_percent"] = 99.0
                job = self._job
            if temporary_path.is_symlink() or not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
                raise RuntimeError("The upscaler returned success without a safe non-empty MP4.")
            output_info = _probe_local_video(temporary_path, require_audio=bool(job["source_has_audio"]))
            if (output_info["width"], output_info["height"]) != (job["output_width"], job["output_height"]):
                raise RuntimeError("The upscaled video has unexpected dimensions.")
            if abs(float(output_info["duration_seconds"]) - job["source_duration"]) > UPSCALE_DURATION_TOLERANCE_SECONDS:
                raise RuntimeError("The upscaled video duration does not match its source.")
            if job["source_has_audio"] and not output_info["has_audio"]:
                raise RuntimeError("The upscaled video lost the source audio stream.")
            source_stat = job["source_path"].stat()
            if source_stat.st_size != job["source_size"] or source_stat.st_mtime_ns != job["source_mtime_ns"]:
                raise RuntimeError("The source video changed while its upscale was being created.")
            with self._lock:
                if self._job is None or self._job["id"] != job_id or self._job["cancel_requested"]:
                    raise _UpscaleCancelled("Upscale was cancelled before publication.")
                os.replace(temporary_path, self._job["output_path"])
                self._job["progress_percent"] = 100.0
                self._finish_locked(self._job, "succeeded", None)
                self._job["log"] = (
                    self._job["log"] + f"[upscale] Video ready: {self._job['output_path'].name}\n"
                )[-MAX_LOG_CHARS:]
        except _UpscaleCancelled as exc:
            cleanup_ok = temporary_path is None or self._remove_temporary(temporary_path)
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "cancelled", str(exc))
                    if not cleanup_ok:
                        self._job["log"] += "[upscale] Warning: temporary-file cleanup was incomplete.\n"
        except Exception as exc:
            cleanup_ok = temporary_path is None or self._remove_temporary(temporary_path)
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "failed", str(exc))
                    if not cleanup_ok:
                        self._job["log"] += "[upscale] Warning: temporary-file cleanup was incomplete.\n"
        finally:
            try:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._job["process"] = None
                if self._worker is threading.current_thread():
                    self._worker = None

    @staticmethod
    def _finish_locked(job: dict[str, Any], status: str, error: str | None) -> None:
        job["status"] = status
        job["stage"] = "complete" if status == "succeeded" else status
        job["error"] = error
        job["finished_at"] = _utc_now()
        job["_finished_mono"] = time.monotonic()
        if error:
            job["log"] = (job["log"] + f"[upscale] {error}\n")[-MAX_LOG_CHARS:]

    @staticmethod
    def _interrupt(process: subprocess.Popen[str], *, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGINT)
        except ProcessLookupError:
            pass

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None or self._job["status"] not in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "There is no active upscale to cancel.")
            self._job["cancel_requested"] = True
            self._job["status"] = "cancelling"
            process = self._job.get("process")
            job_id = self._job["id"]
            self._job["log"] = (self._job["log"] + "[upscale] Cancellation requested.\n")[-MAX_LOG_CHARS:]
        if process is not None:
            self._interrupt(process)

            def force_after_grace() -> None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append(job_id, "[upscale] The worker did not stop; forcing termination.\n")
                    self._interrupt(process, force=True)

            threading.Thread(target=force_after_grace, name=f"upscale-cancel-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            job = self._job
            worker = self._worker
            process = job.get("process") if job else None
            temporary_path = job.get("temporary_path") if job else None
            if job is not None and job.get("status") in self.ACTIVE:
                job["cancel_requested"] = True
                job["status"] = "cancelling"
        if process is not None:
            self._interrupt(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._interrupt(process, force=True)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=6)
        if isinstance(temporary_path, Path):
            self._remove_temporary(temporary_path)


UPSCALE_STATE = UpscaleState()


def _frame_composer_readiness() -> dict[str, Any]:
    """Return an honest, cheap status for the isolated optional still-image lane."""
    if not FRAME_COMPOSER_SCRIPT.is_file():
        return {"ready": False, "note": "Frame Composer wrapper is missing."}
    try:
        import install_frame_composer as frame_installation

        status = frame_installation.readiness()
        ready = bool(status.get("ready"))
        detail = str(status.get("note") or "Setup incomplete.")
        if not ready:
            detail = "Not installed — run install_frame_composer.py --install. " + detail
        return {"ready": ready, "note": detail}
    except Exception:
        return {
            "ready": False,
            "note": "Run install_frame_composer.py --install to add the isolated runtime and exact Q4 model.",
        }


def _frame_composer_dimensions(profile: str, target_mode: str, aspect_ratio: str) -> tuple[int, int]:
    """Resolve exactly the same topology-specific canvas as the target LTX workflow."""
    width, height = FRAME_COMPOSER_PROFILES[profile]
    from generate_ltx25 import resolve_aspect_dimensions

    alignment = 32 if target_mode in {"ingredients", "union", "motion"} else 64
    try:
        return resolve_aspect_dimensions(width, height, aspect_ratio, alignment=alignment)
    except ValueError as exc:  # Defensive: public validation owns these finite choices.
        raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None


def _validate_frame_composer_request(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(
        set(payload) - {"prompt", "target_mode", "profile", "aspect_ratio", "seed", "references"}
    )
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, "Unknown Frame Composer field(s): " + ", ".join(unknown))
    prompt = _validate_video_prompt(
        payload.get("prompt"),
        empty_message="Enter a Frame Composer prompt.",
    )
    profile = payload.get("profile", "quality")
    target_mode = payload.get("target_mode")
    if target_mode is None:
        # Preserve the original profile-only API while allowing new callers to
        # identify the exact target topology explicitly.
        if profile in INGREDIENT_PROFILES:
            target_mode = "ingredients"
        elif profile == "motion3s-experimental":
            target_mode = "motion"
        elif profile in CONTROL_PROFILES:
            target_mode = "union"
        else:
            target_mode = "generate"
    if not isinstance(target_mode, str) or target_mode not in FRAME_COMPOSER_MODE_PROFILES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "Frame Composer target_mode must be generate, ingredients, union, or motion.",
        )
    if not isinstance(profile, str) or profile not in FRAME_COMPOSER_MODE_PROFILES[target_mode]:
        raise APIError(HTTPStatus.BAD_REQUEST, f"Frame Composer profile is invalid for {target_mode}.")
    aspect_ratio = _validate_aspect_ratio(payload.get("aspect_ratio"))
    width, height = _frame_composer_dimensions(profile, target_mode, aspect_ratio)
    seed = payload.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise APIError(HTTPStatus.BAD_REQUEST, "Frame Composer seed must be an integer from 0 to 4,294,967,295.")
    raw_references = payload.get("references")
    if not isinstance(raw_references, list) or not 1 <= len(raw_references) <= MAX_FRAME_COMPOSER_REFERENCES:
        raise APIError(HTTPStatus.BAD_REQUEST, "Frame Composer requires 1–4 active reference photos.")
    references: list[dict[str, Any]] = []
    tags: set[str] = set()
    total_bytes = 0
    for index, raw in enumerate(raw_references, start=1):
        if not isinstance(raw, dict) or set(raw) != {"tag", "image"}:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Frame Composer reference {index} must contain only tag and image.")
        tag_value = raw.get("tag")
        if not isinstance(tag_value, str):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Frame Composer reference {index} tag must be text.")
        tag = tag_value.strip().lower()
        if not tag.startswith("@"):
            tag = "@" + tag
        if not INGREDIENT_TAG_RE.fullmatch(tag):
            raise APIError(HTTPStatus.BAD_REQUEST, f"Frame Composer reference {index} tag must look like @maya.")
        decoded = _decode_image(raw.get("image"), f"Frame Composer reference {index}")
        assert decoded is not None
        tags.add(tag)
        total_bytes += len(decoded[1])
        references.append({"tag": tag, "image": decoded})
    if total_bytes > MAX_TOTAL_IMAGE_BYTES:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Frame Composer photos may total at most 24 MB.")
    mentioned = set(re.findall(r"(?<![A-Za-z0-9_])@[a-z][a-z0-9_]{0,31}(?![A-Za-z0-9_])", prompt))
    unknown_tags = sorted(mentioned - tags)
    if unknown_tags:
        raise APIError(HTTPStatus.BAD_REQUEST, "Frame Composer prompt uses unknown tag(s): " + ", ".join(unknown_tags))
    missing_tags = sorted(tags - mentioned)
    if missing_tags:
        raise APIError(HTTPStatus.BAD_REQUEST, "Mention every active ingredient tag in the prompt: " + ", ".join(missing_tags))
    return {
        "prompt": prompt,
        "target_mode": target_mode,
        "profile": profile,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "seed": seed,
        "references": references,
        "ingredient_count": len(tags),
    }


class FrameComposerState:
    """One serialized local FLUX still job, independent from the LTX Python environment."""

    ACTIVE = {"starting", "running", "cancelling"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in self.ACTIVE)

    def snapshot(self) -> dict[str, Any]:
        readiness = _frame_composer_readiness()
        with self._lock:
            if self._job is None:
                return {"job": None, "readiness": readiness}
            job = self._job
            public = {
                "id": job["id"],
                "status": job["status"],
                "target_mode": job["target_mode"],
                "profile": job["profile"],
                "aspect_ratio": job["aspect_ratio"],
                "width": job["width"],
                "height": job["height"],
                "seed": job["seed"],
                "reference_count": job["reference_count"],
                "ingredient_count": job["ingredient_count"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "error": job["error"],
                "log_tail": job["log"][-4_000:],
                "output_url": (
                    "/frames/" + quote(job["output_path"].name, safe="")
                    if job["status"] == "succeeded"
                    else None
                ),
            }
        return {"job": public, "readiness": readiness}

    def start(self, spec: dict[str, Any]) -> dict[str, Any]:
        ready = _frame_composer_readiness()
        if not ready["ready"]:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(ready["note"]))
        video_job = STATE.snapshot().get("job")
        if video_job and video_job.get("status") in {"starting", "running", "cancelling"}:
            raise APIError(HTTPStatus.CONFLICT, "A video is rendering. Wait for it to finish before composing a frame.")
        voice_state = globals().get("VOICE_STATE")
        if voice_state is not None and voice_state.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before composing a frame.")
        upscale_state = globals().get("UPSCALE_STATE")
        if upscale_state is not None and upscale_state.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before composing a frame.")
        if FRAME_GENERATED_DIR.is_symlink():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "Frame Composer output directory must not be a symbolic link.")
        FRAME_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._job and self._job["status"] in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "A frame is already being composed.")
            job_id = uuid.uuid4().hex
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = (FRAME_GENERATED_DIR / f"flux2-frame-{stamp}-{job_id[:8]}.png").resolve()
            temp_paths: list[Path] = []
            saved_references: list[dict[str, Any]] = []
            try:
                for index, reference in enumerate(spec["references"], start=1):
                    extension, data = reference["image"]
                    handle = tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".frame-composer-{job_id[:8]}-{index}-",
                        suffix="." + extension,
                        dir=FRAME_GENERATED_DIR,
                        delete=False,
                    )
                    path = Path(handle.name).resolve()
                    temp_paths.append(path)
                    try:
                        os.fchmod(handle.fileno(), 0o600)
                        handle.write(data)
                        handle.flush()
                    finally:
                        handle.close()
                    saved_references.append({"tag": reference["tag"], "path": path})
            except Exception as exc:
                JobState._remove_temp_paths(temp_paths)
                raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not stage Frame Composer inputs: {exc}") from None
            self._job = {
                "id": job_id,
                "status": "starting",
                "target_mode": spec["target_mode"],
                "profile": spec["profile"],
                "aspect_ratio": spec["aspect_ratio"],
                "width": spec["width"],
                "height": spec["height"],
                "seed": spec["seed"],
                "prompt": spec["prompt"],
                "references": saved_references,
                "reference_count": len(saved_references),
                "ingredient_count": len({reference["tag"] for reference in saved_references}),
                "started_at": _utc_now(),
                "finished_at": None,
                "error": None,
                "log": "[frame-composer] Queued isolated local still generation.\n",
                "output_path": output_path,
                "temp_paths": temp_paths,
                "process": None,
                "cancel_requested": False,
            }
        threading.Thread(target=self._run, args=(job_id,), name=f"frame-composer-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def _append(self, job_id: str, text: str) -> None:
        text = ANSI_RE.sub("", text).replace("\x00", "")[-32_000:]
        with self._lock:
            if self._job is not None and self._job["id"] == job_id:
                self._job["log"] = (self._job["log"] + text)[-MAX_LOG_CHARS:]

    def _run(self, job_id: str) -> None:
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            job = self._job
            sandbox_block = _metal_sandbox_block_reason()
            if sandbox_block:
                self._finish_locked(job, "failed", sandbox_block)
                blocked_temp_paths = list(job["temp_paths"])
            else:
                blocked_temp_paths = []
        if sandbox_block:
            JobState._remove_temp_paths(blocked_temp_paths)
            return
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            job = self._job
            command = [str(FRAME_COMPOSER_PYTHON), str(FRAME_COMPOSER_SCRIPT)]
            for reference in job["references"]:
                command.extend(["--reference", reference["tag"], str(reference["path"])])
            command.extend(
                [
                    "--profile",
                    job["profile"],
                    "--target-mode",
                    job["target_mode"],
                    "--aspect-ratio",
                    job["aspect_ratio"],
                    "--seed",
                    str(job["seed"]),
                    "--output",
                    str(job["output_path"]),
                    "--",
                    job["prompt"],
                ]
            )
            job["status"] = "running"
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
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=HERE,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    JobState._terminate(process)
                    return
                self._job["process"] = process
                if self._job["cancel_requested"]:
                    self._job["status"] = "cancelling"
                    JobState._terminate(process)
            assert process.stdout is not None
            for line in process.stdout:
                self._append(job_id, line)
            return_code = process.wait()
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    return
                job = self._job
                if job["cancel_requested"]:
                    self._finish_locked(job, "cancelled", "Frame composition cancelled.")
                elif return_code != 0:
                    self._finish_locked(job, "failed", f"Frame Composer exited with status {return_code}.")
                elif not job["output_path"].is_file() or job["output_path"].stat().st_size < 32:
                    self._finish_locked(job, "failed", "Frame Composer finished without a valid PNG.")
                else:
                    self._finish_locked(job, "succeeded", None)
                    self._append(job_id, f"[frame-composer] Frame ready: {job['output_path'].name}\n")
        except Exception as exc:
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "failed", f"Could not run Frame Composer: {exc}")
        finally:
            try:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    temp_paths = list(self._job["temp_paths"])
                    failed = self._job["status"] in {"failed", "cancelled"}
                    output_path = self._job["output_path"]
                    self._job["process"] = None
                else:
                    temp_paths, failed, output_path = [], False, None
            JobState._remove_temp_paths(temp_paths)
            if failed and output_path is not None:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _finish_locked(job: dict[str, Any], status: str, error: str | None) -> None:
        job["status"] = status
        job["error"] = error
        job["finished_at"] = _utc_now()
        if error:
            job["log"] = (job["log"] + f"[frame-composer] {error}\n")[-MAX_LOG_CHARS:]

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None or self._job["status"] not in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "There is no active frame composition to cancel.")
            self._job["cancel_requested"] = True
            self._job["status"] = "cancelling"
            process = self._job["process"]
            job_id = self._job["id"]
        if process is not None:
            JobState._terminate(process)

            def force_after_grace() -> None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append(job_id, "[frame-composer] Process did not stop; forcing termination.\n")
                    JobState._terminate(process, force=True)

            threading.Thread(target=force_after_grace, name=f"frame-cancel-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            process = self._job.get("process") if self._job else None
        if process is not None:
            JobState._terminate(process)


FRAME_STATE = FrameComposerState()


class _VoiceCancelled(RuntimeError):
    pass


class CharacterVoiceState:
    """Serialize independent per-character TTS calls and an optional safe mux."""

    ACTIVE = {"starting", "running", "cancelling"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in self.ACTIVE)

    def snapshot(self) -> dict[str, Any]:
        readiness = _character_voice_readiness()
        registry = _public_voice_registry()
        with self._lock:
            if self._job is None:
                return {"job": None, "readiness": readiness, "registry": registry}
            job = self._job
            public_cues = [
                {
                    "character": item["character"],
                    "preset": item["preset"],
                    "language": item["language"],
                    "start_seconds": item["start_seconds"],
                    "duration_seconds": item["duration_seconds"],
                    "audio_sha256": item["audio_sha256"],
                    "watermarked": item["watermarked"],
                    "audio_url": "/voices/" + quote(item["audio_path"].name, safe=""),
                }
                for item in job["rendered_cues"]
            ]
            public = {
                "id": job["id"],
                "status": job["status"],
                "stage": job["stage"],
                "progress_percent": job["progress_percent"],
                "current_cue": job.get("current_cue"),
                "cue_count": len(job["cues"]),
                "completed_cues": len(public_cues),
                "mux_latest": job["mux_latest"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "error": job["error"],
                "log_tail": job["log"][-4_000:],
                "cues": public_cues,
                "muxed_video_url": (
                    "/media/" + quote(job["muxed_video_path"].name, safe="")
                    if isinstance(job.get("muxed_video_path"), Path) and job["status"] == "succeeded"
                    else None
                ),
            }
        return {"job": public, "readiness": readiness, "registry": registry}

    def start(
        self,
        cues: list[dict[str, Any]],
        *,
        mux_latest: bool,
        source_video: Path | None,
        source_duration: float | None,
    ) -> dict[str, Any]:
        readiness = _character_voice_readiness(refresh=True)
        if not readiness["ready"]:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, str(readiness["note"]))
        if mux_latest and not readiness["mux_ready"]:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "ffmpeg and ffprobe are required for dialogue muxing.")
        if mux_latest and (source_video is None or source_duration is None):
            raise APIError(HTTPStatus.CONFLICT, "Generate a video successfully before requesting a dialogue mux.")
        if VOICE_GENERATED_DIR.is_symlink() or GENERATED_DIR.is_symlink():
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "Managed voice output directories must not be symbolic links.")
        VOICE_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            generated_root = GENERATED_DIR.resolve(strict=True)
            voice_root = VOICE_GENERATED_DIR.resolve(strict=True)
        except OSError as exc:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, f"Could not prepare managed voice output: {exc}") from None
        if voice_root.parent != generated_root:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "Managed voice output escapes the local generated directory.")
        with self._lock:
            if self._job and self._job["status"] in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "Character voices are already being generated.")
            job_id = uuid.uuid4().hex
            self._job = {
                "id": job_id,
                "status": "starting",
                "stage": "queued",
                "progress_percent": 0.0,
                "current_cue": None,
                "cues": [dict(cue) for cue in cues],
                "rendered_cues": [],
                "mux_latest": mux_latest,
                "source_video": source_video,
                "source_duration": source_duration,
                "muxed_video_path": None,
                "started_at": _utc_now(),
                "finished_at": None,
                "error": None,
                "log": f"[character-voice] Queued {len(cues)} isolated cue(s).\n",
                "process": None,
                "cancel_requested": False,
            }
        threading.Thread(target=self._run, args=(job_id,), name=f"character-voice-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def _append(self, job_id: str, message: str) -> None:
        clean = ANSI_RE.sub("", message).replace("\x00", "")[-32_000:]
        with self._lock:
            if self._job is not None and self._job["id"] == job_id:
                self._job["log"] = (self._job["log"] + clean)[-MAX_LOG_CHARS:]

    def _run_process(self, job_id: str, command: list[str], *, json_result: bool) -> Any:
        process: subprocess.Popen[str] | None = None
        output_parts: list[str] = []
        try:
            process = subprocess.Popen(
                command,
                cwd=HERE,
                env=_voice_subprocess_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                if self._job is None or self._job["id"] != job_id:
                    self._interrupt(process)
                    raise _VoiceCancelled("Voice job was replaced.")
                self._job["process"] = process
                if self._job["cancel_requested"]:
                    self._job["status"] = "cancelling"
                    self._interrupt(process)
                else:
                    self._job["status"] = "running"
            assert process.stdout is not None
            total_chars = 0
            for line in process.stdout:
                total_chars += len(line)
                if total_chars <= 256_000:
                    output_parts.append(line)
            return_code = process.wait()
            output = "".join(output_parts)
            with self._lock:
                cancelled = bool(
                    self._job is None
                    or self._job["id"] != job_id
                    or self._job["cancel_requested"]
                )
            if cancelled:
                raise _VoiceCancelled("Character Voice generation was cancelled.")
            if return_code != 0:
                detail = (output.strip() or f"exit {return_code}")[-3_000:]
                raise RuntimeError(f"Character Voice subprocess failed: {detail}")
            if json_result:
                try:
                    return json.loads(output)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Character Voice returned invalid result JSON: {exc}") from None
            return output
        finally:
            try:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass
            with self._lock:
                if self._job is not None and self._job["id"] == job_id and self._job.get("process") is process:
                    self._job["process"] = None

    def _run(self, job_id: str) -> None:
        with self._lock:
            if self._job is None or self._job["id"] != job_id:
                return
            job = self._job
            cues = [dict(cue) for cue in job["cues"]]
            mux_latest = bool(job["mux_latest"])
            source_video = job["source_video"]
            source_duration = job["source_duration"]
        total_steps = len(cues) + int(mux_latest)
        try:
            for index, cue in enumerate(cues, start=1):
                with self._lock:
                    if self._job is None or self._job["id"] != job_id or self._job["cancel_requested"]:
                        raise _VoiceCancelled("Character Voice generation was cancelled.")
                    self._job["stage"] = "synthesizing"
                    self._job["current_cue"] = f"@{cue['character']} · cue {index}/{len(cues)}"
                    self._job["progress_percent"] = round((index - 1) * 100.0 / total_steps, 1)
                safe_character = cue["character"][:32]
                filename = f"voice-{job_id[:8]}-{index:02d}-{safe_character}.wav"
                expected_audio = VOICE_GENERATED_DIR / filename
                expected_manifest = expected_audio.with_suffix(".voice.json")
                if (
                    expected_audio.exists()
                    or expected_audio.is_symlink()
                    or expected_manifest.exists()
                    or expected_manifest.is_symlink()
                ):
                    raise RuntimeError("Refusing to replace an existing Character Voice artifact.")
                command = [
                    str(sys.executable),
                    str(CHARACTER_VOICE_SCRIPT),
                    "synthesize",
                    cue["text"],
                    "--language",
                    cue["language"],
                    "--character",
                    cue["character"],
                    "--output",
                    filename,
                    "--style",
                    "auto",
                    "--seed",
                    str(cue["seed"]),
                    "--device",
                    "auto",
                ]
                if cue["voice_description"]:
                    command.extend(["--voice-description", cue["voice_description"]])
                try:
                    raw_result = self._run_process(job_id, command, json_result=True)
                    rendered = _validate_voice_result(raw_result, cue=cue, expected_audio=expected_audio)
                except Exception:
                    for artifact in (expected_audio, expected_manifest):
                        try:
                            if not artifact.is_symlink():
                                artifact.unlink(missing_ok=True)
                        except OSError:
                            pass
                    raise
                with self._lock:
                    if self._job is None or self._job["id"] != job_id:
                        raise _VoiceCancelled("Character Voice generation was cancelled.")
                    self._job["rendered_cues"].append(rendered)
                    self._job["progress_percent"] = round(index * 100.0 / total_steps, 1)
                self._append(job_id, f"[character-voice] Finished isolated cue {index}/{len(cues)} for @{cue['character']}.\n")

            if mux_latest:
                assert isinstance(source_video, Path) and isinstance(source_duration, (int, float))
                with self._lock:
                    if self._job is None or self._job["id"] != job_id or self._job["cancel_requested"]:
                        raise _VoiceCancelled("Dialogue mux was cancelled.")
                    self._job["stage"] = "muxing"
                    self._job["current_cue"] = None
                    rendered_cues = list(self._job["rendered_cues"])
                muxed = self._mux(job_id, source_video, float(source_duration), rendered_cues)
                with self._lock:
                    if self._job is None or self._job["id"] != job_id:
                        raise _VoiceCancelled("Dialogue mux was cancelled.")
                    self._job["muxed_video_path"] = muxed
                    self._job["progress_percent"] = 100.0
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "succeeded", None)
            self._append(job_id, "[character-voice] All isolated cues are ready.\n")
        except _VoiceCancelled as exc:
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "cancelled", str(exc))
        except Exception as exc:
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._finish_locked(self._job, "failed", str(exc))
        finally:
            with self._lock:
                if self._job is not None and self._job["id"] == job_id:
                    self._job["process"] = None

    def _mux(
        self,
        job_id: str,
        source_video: Path,
        source_duration: float,
        rendered_cues: list[dict[str, Any]],
    ) -> Path:
        if GENERATED_DIR.is_symlink() or VOICE_GENERATED_DIR.is_symlink():
            raise RuntimeError("Managed dialogue-mux directories became unsafe.")
        if source_video.is_symlink() or not source_video.is_file():
            raise RuntimeError("The video selected for dialogue mux is no longer safe or available.")
        generated_root = GENERATED_DIR.resolve(strict=True)
        if source_video.resolve(strict=True).parent != generated_root:
            raise RuntimeError("The video selected for dialogue mux escaped the managed output directory.")
        for cue in rendered_cues:
            audio = cue["audio_path"]
            if audio.is_symlink() or audio.resolve(strict=True).parent != VOICE_GENERATED_DIR.resolve(strict=True):
                raise RuntimeError("A generated cue escaped the managed voice output directory.")
            if float(cue["start_seconds"]) >= source_duration:
                raise RuntimeError(f"Cue for @{cue['character']} starts after the source video ends.")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for the safe dialogue mux.")
        safe_stem = source_video.stem[:112]
        final_path = GENERATED_DIR / f"{safe_stem}-voiced-{job_id[:8]}.mp4"
        temporary_path = GENERATED_DIR / f".{safe_stem}-voiced-{job_id[:8]}-{uuid.uuid4().hex}.tmp.mp4"
        if final_path.exists() or final_path.is_symlink() or temporary_path.exists() or temporary_path.is_symlink():
            raise RuntimeError("Refusing to replace an existing dialogue-mux output.")
        command = _build_dialogue_mux_command(ffmpeg, source_video, rendered_cues, temporary_path)
        try:
            self._run_process(job_id, command, json_result=False)
            if temporary_path.is_symlink() or not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
                raise RuntimeError("ffmpeg returned success without a safe non-empty MP4.")
            _probe_local_video(temporary_path, require_audio=True)
            with self._lock:
                if self._job is None or self._job["id"] != job_id or self._job["cancel_requested"]:
                    raise _VoiceCancelled("Dialogue mux was cancelled before publication.")
            os.replace(temporary_path, final_path)
            self._append(job_id, f"[character-voice] Published a new voiced MP4; original preserved: {final_path.name}\n")
            return final_path
        finally:
            try:
                if not temporary_path.is_symlink():
                    temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _finish_locked(job: dict[str, Any], status: str, error: str | None) -> None:
        job["status"] = status
        job["stage"] = "complete" if status == "succeeded" else status
        job["current_cue"] = None
        if status != "succeeded":
            job["progress_percent"] = None
        job["error"] = error
        job["finished_at"] = _utc_now()
        if error:
            job["log"] = (job["log"] + f"[character-voice] {error}\n")[-MAX_LOG_CHARS:]

    @staticmethod
    def _interrupt(process: subprocess.Popen[str], *, force: bool = False) -> None:
        """Interrupt the whole voice process group; SIGINT lets wrapper cleanup run."""
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGINT)
        except ProcessLookupError:
            pass

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None or self._job["status"] not in self.ACTIVE:
                raise APIError(HTTPStatus.CONFLICT, "There is no active Character Voice job to cancel.")
            self._job["cancel_requested"] = True
            self._job["status"] = "cancelling"
            process = self._job["process"]
            job_id = self._job["id"]
            self._job["log"] = (self._job["log"] + "[character-voice] Cancellation requested.\n")[-MAX_LOG_CHARS:]
        if process is not None:
            self._interrupt(process)

            def force_after_grace() -> None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append(job_id, "[character-voice] Process did not stop; forcing termination.\n")
                    self._interrupt(process, force=True)

            threading.Thread(target=force_after_grace, name=f"voice-cancel-{job_id[:8]}", daemon=True).start()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            process = self._job.get("process") if self._job else None
        if process is not None:
            self._interrupt(process)


VOICE_STATE = CharacterVoiceState()


def _cheap_phosphene_pack_issues(pack: Any, verification: Any) -> int:
    """Count incomplete artifacts for one official pack without hashing large weights."""
    issues = 0
    receipt_path = pack.path / verification.RECEIPT_NAME
    try:
        if receipt_path.is_symlink():
            raise OSError("receipt is a symbolic link")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = verification._receipt_expected(pack)
        if any(receipt.get(field) != value for field, value in expected.items()):
            issues += 1
        license_exception = receipt.get("license_exception")
        if (
            not isinstance(license_exception, dict)
            or license_exception.get("installed_sha256") != verification.CURRENT_LICENSE_SHA256
        ):
            issues += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        issues += 1
    for relative, artifact in pack.files.items():
        candidate = pack.path / relative
        try:
            if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size != artifact.size:
                issues += 1
        except OSError:
            issues += 1
    try:
        if pack.path.is_symlink() or not pack.path.is_dir():
            issues += 1
        elif any(
            path.is_file()
            and (
                path.name.endswith((".part", ".partial", ".incomplete", ".download", ".assembling"))
                or path.name.startswith(".tmp")
            )
            for path in pack.path.rglob("*")
        ):
            issues += 1
    except OSError:
        issues += 1
    return issues


def _cheap_split_pack_issues() -> int:
    """Backward-compatible official base + official Gemma readiness count."""
    try:
        import verify_ltx25 as verification
    except Exception:
        return 1
    return sum(_cheap_phosphene_pack_issues(pack, verification) for pack in verification.PACKS)


def _gemma_pack_readiness() -> dict[str, dict[str, Any]]:
    """Report official and explicitly separated fallback Gemma readiness."""
    try:
        import verify_ltx25 as verification

        official_ready = _cheap_phosphene_pack_issues(verification.GEMMA_PACK, verification) == 0
    except Exception:
        official_ready = False
    try:
        from import_ddalcu_gemma_fallback import verify_fallback_install

        verify_fallback_install(full_hash=False, report=lambda _message: None)
        fallback_ready = True
    except Exception:
        fallback_ready = False

    selected = "official" if official_ready else "ddalcu-fallback" if fallback_ready else None
    if selected == "official":
        auto_note = "Auto will use the exact official Phosphene Gemma pack."
    elif selected == "ddalcu-fallback":
        auto_note = (
            "Auto will use the separately validated ddalcu compatibility fallback; "
            "it is structurally verified but not byte-identical to the official tensor serialization."
        )
    else:
        auto_note = "No complete verified Gemma pack is installed yet."
    return {
        "auto": {"ready": selected is not None, "selected": selected, "note": auto_note},
        "official": {
            "ready": official_ready,
            "note": "Exact official Phosphene Gemma pack." if official_ready else "Official Gemma pack is incomplete.",
        },
        "ddalcu-fallback": {
            "ready": fallback_ready,
            "note": (
                "Pinned mirror identity plus official metadata and tensor schema verified."
                if fallback_ready
                else "Validated compatibility fallback is not installed."
            ),
        },
    }


def _resolve_gemma_pack(requested: object) -> str:
    if not isinstance(requested, str) or requested not in GEMMA_PACK_CHOICES:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "gemma_pack must be auto, official, or ddalcu-fallback.",
        )
    if requested != "auto":
        return requested
    selected = _gemma_pack_readiness()["auto"].get("selected")
    # Preserve the official fail-closed behavior while installation is still
    # incomplete; generate_ltx25.py performs the authoritative full preflight.
    return selected if isinstance(selected, str) else "official"


def _edit_mode_assets(mode: str) -> tuple[int, str]:
    """Return cheap Q8 readiness for an advanced edit workflow."""
    try:
        import edit_ltx25 as edit_wrapper

        missing = len(edit_wrapper.missing_mode_assets(mode))
        if missing:
            return missing, edit_wrapper.DEV_ONLY_MODE_NOTE
        return 0, "Q8 advanced assets are present; full receipt, model, source, and Metal preflight runs on submit."
    except Exception:
        return 1, (
            "This workflow requires the separate Q8 advanced model lane, which is not yet ready."
        )


def _control_mode_assets(mode: str) -> tuple[int, str]:
    """Return cheap Q8 + adapter readiness for Ingredients/Union/Motion."""
    try:
        import condition_ltx25 as control_wrapper

        missing = len(control_wrapper.missing_mode_assets(mode))
        if missing:
            return missing, (
                f"The Q8 distilled-control receipt or the {mode} adapter still needs "
                f"{missing} local asset(s)."
            )
        return 0, (
            "Q8 distilled-control receipt and assets are present; full receipt, adapter, source, and Metal "
            "preflight runs on submit."
        )
    except Exception:
        return 1, "The Q8 control wrapper or its local assets are not ready."


def _workflow_readiness(
    gemma_packs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a cheap, path-free readiness summary; full checks still run in each wrapper."""
    python_ready = GENERATOR_PYTHON.is_file()
    try:
        import verify_ltx25 as verification

        base_issues = _cheap_phosphene_pack_issues(verification.BASE_PACK, verification)
    except Exception:
        base_issues = 1
    gemma_packs = gemma_packs or _gemma_pack_readiness()
    selected_gemma = gemma_packs["auto"].get("selected")
    gemma_issues = 0 if gemma_packs["auto"]["ready"] else 1
    asset_issues = base_issues + gemma_issues
    generator_script_ready = GENERATOR_SCRIPT.is_file()
    generate_launcher_issues = int(not python_ready) + int(not generator_script_ready)
    generate_ready = generate_launcher_issues == 0 and asset_issues == 0
    edit_launcher = python_ready and EDIT_SCRIPT.is_file()
    base_note = (
        (
            "The Phosphene Q4 base and exact official Gemma pack are present; Generate uses distilled "
            "low-RAM inference. Full hash/source/Metal preflight runs on submit."
            if selected_gemma == "official"
            else "The Phosphene Q4 base and separately validated Gemma compatibility fallback are present; "
            "Generate uses distilled low-RAM inference. Full structural/source/Metal preflight runs on submit."
        )
        if asset_issues == 0
        else f"Needs attention in {asset_issues} base/Gemma verification item(s)."
    )
    runtime_note = ""
    if not python_ready:
        runtime_note = " The pinned LTX Python environment is missing."
    if not generator_script_ready:
        runtime_note += " The Generate launcher is missing."
    result: dict[str, dict[str, Any]] = {
        "generate": {
            "ready": generate_ready,
            "experimental": False,
            "blocked": not generate_ready,
            "missing_assets": asset_issues + generate_launcher_issues,
            "note": base_note + runtime_note,
        }
    }
    for mode in ("retake", "extend", "a2v"):
        mode_missing, mode_note = _edit_mode_assets(mode)
        launcher_issues = int(not edit_launcher)
        blocked = mode_missing > 0 or asset_issues > 0 or launcher_issues > 0
        launcher_note = " The edit workflow launcher or pinned Python environment is missing." if launcher_issues else ""
        result[mode] = {
            "ready": not blocked,
            "experimental": False,
            "blocked": blocked,
            "missing_assets": asset_issues + mode_missing + launcher_issues,
            "note": mode_note + " " + base_note + launcher_note,
        }
    control_launcher = python_ready and CONDITION_SCRIPT.is_file()
    for mode in ("ingredients", "union", "motion"):
        mode_missing, mode_note = _control_mode_assets(mode)
        launcher_issues = int(not control_launcher)
        blocked = mode_missing > 0 or asset_issues > 0 or launcher_issues > 0
        launcher_note = " The control workflow launcher or pinned Python environment is missing." if launcher_issues else ""
        result[mode] = {
            "ready": not blocked,
            "experimental": False,
            "blocked": blocked,
            "missing_assets": asset_issues + mode_missing + launcher_issues,
            "note": mode_note + " " + base_note + launcher_note,
        }
    return result


def _register_voice_preset(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(
        set(payload)
        - {"name", "reference", "reference_language", "voice_description", "consent_confirmed", "force"}
    )
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, "Unknown voice preset field(s): " + ", ".join(unknown))
    if payload.get("consent_confirmed") is not True:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "Explicit consent confirmation is required; use only your own voice or a recording you are authorized to clone.",
        )
    if not isinstance(payload.get("name"), str):
        raise APIError(HTTPStatus.BAD_REQUEST, "name must be text.")
    if not isinstance(payload.get("reference_language"), str):
        raise APIError(HTTPStatus.BAD_REQUEST, "reference_language must be text.")
    if payload.get("voice_description") is not None and not isinstance(payload.get("voice_description"), str):
        raise APIError(HTTPStatus.BAD_REQUEST, "voice_description must be text.")
    force = _validate_bool(payload.get("force"), field="force")
    data = _decode_voice_reference(payload.get("reference"))
    voice = _character_voice_module()
    if HERE.is_symlink():
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "Character Voice staging root is unsafe.")
    try:
        with tempfile.TemporaryDirectory(prefix=".voice-register-", dir=HERE) as temporary:
            staging = Path(temporary)
            reference = staging / "reference.wav"
            descriptor = os.open(reference, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            entry = voice.register_preset(
                payload["name"],
                reference,
                payload["reference_language"],
                consent_confirmed=True,
                voice_description=payload.get("voice_description"),
                force=force,
            )
    except (voice.UserInputError, voice.PreflightError) as exc:
        raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None
    except OSError as exc:
        raise APIError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not register the local voice preset: {exc}") from None
    return {
        "preset": {
            "name": entry["name"],
            "reference_language": entry["reference_language"],
            "duration_seconds": entry["duration_seconds"],
            "voice_description": entry["voice_description"],
            "consent_confirmed": entry["consent_confirmed"],
            "reference_sha256": entry["sha256"],
        }
    }


def _assign_voice_preset(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - {"character", "preset", "force"})
    if unknown:
        raise APIError(HTTPStatus.BAD_REQUEST, "Unknown voice assignment field(s): " + ", ".join(unknown))
    character = _normalize_voice_character(payload.get("character"))
    preset = payload.get("preset")
    if not isinstance(preset, str):
        raise APIError(HTTPStatus.BAD_REQUEST, "preset must be text.")
    voice = _character_voice_module()
    try:
        result = voice.assign_character(
            character,
            preset,
            force=_validate_bool(payload.get("force"), field="force"),
        )
    except (voice.UserInputError, voice.PreflightError, OSError) as exc:
        raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None
    return {"assignment": result}


class LTXRequestHandler(BaseHTTPRequestHandler):
    server_version = "LTXLocalUI/1.0"
    sys_version = ""

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write("[http] " + (format_string % args) + "\n")

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if urlsplit(self.path).path in {
            "/api/status",
            "/api/frame-composer/status",
            "/api/voice/status",
            "/api/upscale/status",
        } and str(code) == "200":
            return
        super().log_request(code, size)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A browser can cancel a polling request while navigating or
                # hiding a panel.  The response is already disposable and the
                # local server should not print a crash-looking traceback.
                return

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _request_host_allowed(self) -> bool:
        host_header = self.headers.get("Host", "")
        if not host_header or len(host_header) > 255:
            return False
        try:
            parsed = urlsplit("//" + host_header)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return False
        return (
            hostname is not None
            and hostname.lower() in ALLOWED_HOSTS
            and (port is None or port == self.server.server_port)
        )

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            parsed = urlsplit(origin)
            origin_port = parsed.port if parsed.port is not None else 80
            request_host = urlsplit("//" + self.headers.get("Host", "")).hostname
        except ValueError:
            return False
        return (
            parsed.scheme == "http"
            and parsed.hostname is not None
            and parsed.hostname.lower() in ALLOWED_HOSTS
            and request_host is not None
            and parsed.hostname.lower() == request_host.lower()
            and origin_port == self.server.server_port
        )

    def _check_request(self, mutating: bool = False) -> bool:
        if not self._request_host_allowed():
            self._send_error_json(HTTPStatus.MISDIRECTED_REQUEST, "Only localhost requests are accepted.")
            return False
        if mutating and not self._origin_allowed():
            self._send_error_json(HTTPStatus.FORBIDDEN, "Cross-origin requests are not accepted.")
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise APIError(HTTPStatus.BAD_REQUEST, "Chunked request bodies are not supported.")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise APIError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json.")
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            raise APIError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required.")
        try:
            length = int(length_text)
        except ValueError:
            raise APIError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length.") from None
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large.")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise APIError(HTTPStatus.BAD_REQUEST, "Incomplete request body.")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APIError(HTTPStatus.BAD_REQUEST, "Request body must contain valid UTF-8 JSON.") from None
        if not isinstance(value, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, "JSON body must be an object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_request():
            return
        path = urlsplit(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json(HTTPStatus.OK, STATE.snapshot())
        elif path == "/api/readiness":
            gemma_packs = _gemma_pack_readiness()
            self._send_json(
                HTTPStatus.OK,
                {
                    "modes": _workflow_readiness(gemma_packs),
                    "gemma_packs": gemma_packs,
                },
            )
        elif path == "/api/frame-composer/status":
            self._send_json(HTTPStatus.OK, FRAME_STATE.snapshot())
        elif path == "/api/voice/status":
            self._send_json(HTTPStatus.OK, VOICE_STATE.snapshot())
        elif path == "/api/upscale/status":
            self._send_json(HTTPStatus.OK, UPSCALE_STATE.snapshot())
        elif path == "/api/characters":
            try:
                self._send_json(HTTPStatus.OK, {"profiles": _list_character_profiles()})
            except APIError as exc:
                self._send_error_json(exc.status, exc.message)
        elif path.startswith("/api/characters/"):
            try:
                encoded_id = path[len("/api/characters/") :]
                if not encoded_id or "/" in encoded_id:
                    raise APIError(HTTPStatus.NOT_FOUND, "Not found.")
                profile_id = unquote(encoded_id, errors="strict")
                profile = _public_character_profile(_read_character_profile(profile_id))
                self._send_json(HTTPStatus.OK, {"profile": profile})
            except UnicodeError:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid character profile path.")
            except APIError as exc:
                self._send_error_json(exc.status, exc.message)
        elif path.startswith("/character-media/"):
            self._serve_character_media(path[len("/character-media/") :], head_only=False)
        elif path.startswith("/voices/"):
            self._serve_voice_media(path[len("/voices/") :], head_only=False)
        elif path.startswith("/frames/"):
            self._serve_frame_media(path[len("/frames/") :], head_only=False)
        elif path.startswith("/media/"):
            self._serve_media(path[len("/media/") :], head_only=False)
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._check_request():
            return
        path = urlsplit(self.path).path
        if path.startswith("/character-media/"):
            self._serve_character_media(path[len("/character-media/") :], head_only=True)
        elif path.startswith("/voices/"):
            self._serve_voice_media(path[len("/voices/") :], head_only=True)
        elif path.startswith("/frames/"):
            self._serve_frame_media(path[len("/frames/") :], head_only=True)
        elif path.startswith("/media/"):
            self._serve_media(path[len("/media/") :], head_only=True)
        else:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_request(mutating=True):
            return
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/characters":
                profile = _save_character_profile(payload)
                self._send_json(HTTPStatus.CREATED, {"profile": profile})
            elif path == "/api/frame-capture":
                frame = _save_captured_video_frame(payload)
                self._send_json(HTTPStatus.CREATED, {"frame": frame})
            elif path == "/api/frame-composer":
                spec = _validate_frame_composer_request(payload)
                with COMPUTE_ADMISSION_LOCK:
                    _reject_if_prompt_coach_active()
                    if VOICE_STATE.is_busy():
                        raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading Frame Composer.")
                    if UPSCALE_STATE.is_busy():
                        raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading Frame Composer.")
                    response = FRAME_STATE.start(spec)
                self._send_json(HTTPStatus.ACCEPTED, response)
            elif path == "/api/frame-composer/cancel":
                if payload:
                    raise APIError(HTTPStatus.BAD_REQUEST, "Frame Composer cancel request must be empty.")
                self._send_json(HTTPStatus.ACCEPTED, FRAME_STATE.cancel())
            elif path == "/api/voice/presets":
                with COMPUTE_ADMISSION_LOCK:
                    if VOICE_STATE.is_busy():
                        raise APIError(HTTPStatus.CONFLICT, "Wait for the active Character Voice job before changing presets.")
                    response = _register_voice_preset(payload)
                self._send_json(HTTPStatus.CREATED, response)
            elif path == "/api/voice/assign":
                with COMPUTE_ADMISSION_LOCK:
                    if VOICE_STATE.is_busy():
                        raise APIError(HTTPStatus.CONFLICT, "Wait for the active Character Voice job before changing assignments.")
                    response = _assign_voice_preset(payload)
                self._send_json(HTTPStatus.OK, response)
            elif path == "/api/voice/synthesize":
                self._handle_voice_synthesis(payload)
            elif path == "/api/voice/cancel":
                if payload:
                    raise APIError(HTTPStatus.BAD_REQUEST, "Character Voice cancel request must be empty.")
                self._send_json(HTTPStatus.ACCEPTED, VOICE_STATE.cancel())
            elif path == "/api/upscale":
                self._handle_upscale(payload)
            elif path == "/api/upscale/cancel":
                if payload:
                    raise APIError(HTTPStatus.BAD_REQUEST, "Upscale cancel request must be empty.")
                self._send_json(HTTPStatus.ACCEPTED, UPSCALE_STATE.cancel())
            elif path == "/api/prompt-coach":
                self._handle_prompt_coach(payload)
            elif path == "/api/render":
                self._handle_render(payload)
            elif path == "/api/generate":
                self._handle_generate(payload)
            elif path == "/api/cancel":
                if payload:
                    raise APIError(HTTPStatus.BAD_REQUEST, "Cancel request must be an empty JSON object.")
                self._send_json(HTTPStatus.ACCEPTED, STATE.cancel())
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except APIError as exc:
            self._send_error_json(exc.status, exc.message)

    def _handle_prompt_coach(self, payload: dict[str, Any]) -> None:
        """Run a local conversational coach without overlapping media compute."""
        claimed = False
        try:
            with COMPUTE_ADMISSION_LOCK:
                active_compute = any(
                    state.is_busy() for state in (STATE, FRAME_STATE, VOICE_STATE, UPSCALE_STATE)
                )
                if not active_compute:
                    claimed = _claim_prompt_coach()
            gemma_packs = _gemma_pack_readiness()
            pause_reason = (
                "AI inference is paused while local media compute is active"
                if active_compute
                else "another Prompt Coach AI reply is already running"
                if not claimed
                else ""
            )
            response = prompt_coach_backend.coach(
                payload,
                readiness=_workflow_readiness(gemma_packs),
                allow_model=claimed,
                pause_reason=pause_reason,
            )
        except prompt_coach_backend.CoachInputError as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        finally:
            if claimed:
                _release_prompt_coach()
        self._send_json(HTTPStatus.OK, response)

    def do_PUT(self) -> None:  # noqa: N802
        if not self._check_request(mutating=True):
            return
        path = urlsplit(self.path).path
        try:
            if not path.startswith("/api/characters/"):
                raise APIError(HTTPStatus.NOT_FOUND, "Not found.")
            encoded_id = path[len("/api/characters/") :]
            if not encoded_id or "/" in encoded_id:
                raise APIError(HTTPStatus.NOT_FOUND, "Not found.")
            profile_id = unquote(encoded_id, errors="strict")
            payload = self._read_json()
            profile = _save_character_profile(payload, profile_id)
            self._send_json(HTTPStatus.OK, {"profile": profile})
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid character profile path.")
        except APIError as exc:
            self._send_error_json(exc.status, exc.message)

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._check_request(mutating=True):
            return
        path = urlsplit(self.path).path
        try:
            if not path.startswith("/api/characters/"):
                raise APIError(HTTPStatus.NOT_FOUND, "Not found.")
            encoded_id = path[len("/api/characters/") :]
            if not encoded_id or "/" in encoded_id:
                raise APIError(HTTPStatus.NOT_FOUND, "Not found.")
            profile_id = unquote(encoded_id, errors="strict")
            payload = self._read_json()
            if payload != {"confirm": True}:
                raise APIError(HTTPStatus.BAD_REQUEST, "Deleting a character requires explicit confirmation.")
            _delete_character_profile(profile_id)
            self._send_json(HTTPStatus.OK, {"deleted": profile_id})
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid character profile path.")
        except APIError as exc:
            self._send_error_json(exc.status, exc.message)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._check_request():
            return
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "Cross-origin requests are not supported.")

    @staticmethod
    def _prompt_seed_character(payload: dict[str, Any]) -> tuple[str, int, dict[str, str]]:
        prompt = _validate_video_prompt(
            payload.get("prompt"),
            empty_message="Enter a prompt before rendering.",
        )
        seed = payload.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise APIError(HTTPStatus.BAD_REQUEST, "seed must be an integer from 0 to 4,294,967,295.")
        return prompt, seed, _validate_character_fields(payload)

    @staticmethod
    def _reject_unknown(payload: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise APIError(HTTPStatus.BAD_REQUEST, "Unknown field(s): " + ", ".join(unknown))

    def _handle_upscale(self, payload: dict[str, Any]) -> None:
        self._reject_unknown(payload, {"profile", "target_short_edge"})
        profile = payload.get("profile", "fast720")
        target_short_edge = payload.get("target_short_edge")
        with COMPUTE_ADMISSION_LOCK:
            _reject_if_prompt_coach_active()
            if STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A video is rendering. Wait before starting an upscale.")
            if FRAME_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait before starting an upscale.")
            if VOICE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before starting an upscale.")
            response = UPSCALE_STATE.start(profile=profile, target_short_edge=target_short_edge)
        self._send_json(HTTPStatus.ACCEPTED, response)

    def _handle_voice_synthesis(self, payload: dict[str, Any]) -> None:
        self._reject_unknown(payload, {"cues", "mux_latest"})
        cues = _validate_voice_cues(payload.get("cues"))
        mux_latest = _validate_bool(payload.get("mux_latest"), field="mux_latest")
        source_video: Path | None = None
        source_duration: float | None = None
        with COMPUTE_ADMISSION_LOCK:
            _reject_if_prompt_coach_active()
            if STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A video is rendering. Wait before generating character voices.")
            if FRAME_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait before generating character voices.")
            if UPSCALE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before generating character voices.")
            if mux_latest:
                source_video = STATE.completed_output_for_voice()
                try:
                    source_duration = float(_probe_local_video(source_video)["duration_seconds"])
                except RuntimeError as exc:
                    raise APIError(HTTPStatus.BAD_REQUEST, str(exc)) from None
                for cue in cues:
                    if cue["start_seconds"] >= source_duration:
                        raise APIError(
                            HTTPStatus.BAD_REQUEST,
                            f"Cue for @{cue['character']} starts at {cue['start_seconds']:.2f}s, after the "
                            f"latest video ends at {source_duration:.2f}s.",
                        )
            response = VOICE_STATE.start(
                cues,
                mux_latest=mux_latest,
                source_video=source_video,
                source_duration=source_duration,
            )
        self._send_json(HTTPStatus.ACCEPTED, response)

    def _handle_render(self, payload: dict[str, Any]) -> None:
        if FRAME_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait for it to finish before loading LTX.")
        if VOICE_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading LTX.")
        if UPSCALE_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading LTX.")
        mode = payload.get("mode", "generate")
        if not isinstance(mode, str) or mode not in WORKFLOW_MODES:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                "mode must be generate, retake, extend, a2v, ingredients, union, or motion.",
            )
        forwarded = dict(payload)
        forwarded.pop("mode", None)
        if mode == "generate":
            self._handle_generate(forwarded)
            return

        character_keys = {key for key, _, _, _ in CHARACTER_FIELDS}
        common = {"mode", "prompt", "seed"} | character_keys
        if mode == "retake":
            allowed = common | {"source_video", "start_seconds", "end_seconds", "preserve_audio"}
        elif mode == "extend":
            allowed = common | {"source_video", "extend_seconds", "direction"}
        elif mode == "a2v":
            allowed = common | {
                "audio",
                "audio_start",
                "profile",
                "aspect_ratio",
                "duration_seconds",
                "start_image",
                "image_strength",
            }
        else:
            allowed = common | {
                "profile",
                "duration_seconds",
                "lora_strength",
                "conditioning_strength",
                "start_frame",
                "end_frame",
                "ingredients",
                "character_profile_id",
                "character_reference_images",
            }
            if mode == "ingredients":
                allowed |= {"aspect_ratio", "ingredient_references", "dialogue_guidance"}
            elif mode == "union":
                allowed |= {"aspect_ratio", "union_route", "control_type", "control_video"}
            else:
                allowed |= {"aspect_ratio", "tracks"}
        self._reject_unknown(payload, allowed)
        if mode in EDIT_MODES:
            missing, note = _edit_mode_assets(mode)
            if missing:
                raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, note)
        prompt, seed, character = self._prompt_seed_character(payload)

        if mode in EDIT_MODES:
            conditioned_prompt = _build_conditioned_prompt(prompt, character, False)
            spec: dict[str, Any] = {
                "mode": mode,
                "seed": seed,
                "profile": payload.get("profile") if mode == "a2v" else mode,
                "conditionings": [],
                "uploads": {},
            }
            if mode in {"retake", "extend"}:
                spec["uploads"]["source_video"] = _decode_uploaded_file(
                    payload.get("source_video"),
                    label="Source video",
                    extensions=VIDEO_EXTENSIONS,
                    max_bytes=MAX_VIDEO_BYTES,
                    media_prefix="video",
                )
            if mode == "retake":
                start_seconds = _finite_number(
                    payload.get("start_seconds"),
                    field="start_seconds",
                    minimum=0,
                    maximum=86_400,
                )
                end_seconds = _finite_number(
                    payload.get("end_seconds"),
                    field="end_seconds",
                    minimum=0,
                    maximum=86_400,
                )
                if end_seconds <= start_seconds:
                    raise APIError(HTTPStatus.BAD_REQUEST, "end_seconds must be greater than start_seconds.")
                spec.update(
                    {
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "preserve_audio": _validate_bool(
                            payload.get("preserve_audio"), field="preserve_audio"
                        ),
                    }
                )
            elif mode == "extend":
                extend_seconds = payload.get("extend_seconds", 3)
                if isinstance(extend_seconds, bool) or not isinstance(extend_seconds, int) or not 1 <= extend_seconds <= 8:
                    raise APIError(HTTPStatus.BAD_REQUEST, "extend_seconds must be an integer from 1 to 8.")
                direction = payload.get("direction", "after")
                if direction not in {"before", "after"}:
                    raise APIError(HTTPStatus.BAD_REQUEST, "direction must be before or after.")
                spec.update({"extend_seconds": extend_seconds, "direction": direction})
            else:
                profile = payload.get("profile", "clip3s")
                if not isinstance(profile, str) or profile not in PROFILES:
                    raise APIError(HTTPStatus.BAD_REQUEST, "A2V profile is invalid.")
                duration_seconds = _validate_duration_seconds(payload.get("duration_seconds", 3))
                spec.update(
                    {
                        "profile": profile,
                        "aspect_ratio": _validate_aspect_ratio(payload.get("aspect_ratio")),
                        "duration_seconds": duration_seconds,
                        "audio_start": _finite_number(
                            payload.get("audio_start"),
                            field="audio_start",
                            minimum=0,
                            maximum=86_400,
                            default=0,
                        ),
                        "image_strength": _finite_strength(
                            payload.get("image_strength"), field="image_strength", default=1.0
                        ),
                    }
                )
                spec["uploads"]["audio"] = _decode_uploaded_file(
                    payload.get("audio"),
                    label="Audio input",
                    extensions=AUDIO_EXTENSIONS,
                    max_bytes=MAX_AUDIO_BYTES,
                    media_prefix="audio",
                )
                start_image = _decode_image(payload.get("start_image"), "A2V start image")
                if start_image is not None:
                    spec["uploads"]["a2v_start_image"] = (
                        "." + start_image[0],
                        start_image[1],
                        "start." + start_image[0],
                    )
            with COMPUTE_ADMISSION_LOCK:
                _reject_if_prompt_coach_active()
                if FRAME_STATE.is_busy():
                    raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait before loading LTX.")
                if VOICE_STATE.is_busy():
                    raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading LTX.")
                if UPSCALE_STATE.is_busy():
                    raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading LTX.")
                response = STATE.start_workflow(conditioned_prompt, spec)
            self._send_json(HTTPStatus.ACCEPTED, response)
            return

        profile = payload.get(
            "profile",
            "ingredients5s" if mode == "ingredients" else "control3s",
        )
        profile_choices = (
            INGREDIENT_PROFILES
            if mode == "ingredients"
            else MOTION_PROFILES
            if mode == "motion"
            else CONTROL_PROFILES
        )
        if not isinstance(profile, str) or profile not in profile_choices:
            choices = ", ".join(sorted(profile_choices))
            raise APIError(HTTPStatus.BAD_REQUEST, f"profile for {mode} must be one of: {choices}.")
        if mode == "ingredients":
            raw_duration = payload.get("duration_seconds", 5)
            if raw_duration != 5 or isinstance(raw_duration, bool):
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    "Ingredients duration is locked to 121 frames / 5.04 seconds.",
                )
            duration_seconds = 5
            total_frames = 121
        else:
            duration_seconds = _validate_duration_seconds(payload.get("duration_seconds", 3))
            total_frames = 24 * duration_seconds + 1
            if profile == "motion3s-experimental" and duration_seconds != 3:
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    "The experimental 640×384 control profile is restricted to three seconds on this 16 GB Mac.",
                )
        start_frame = _validate_frame_control(payload.get("start_frame"), key="start_frame", label="Start frame")
        end_frame = _validate_frame_control(payload.get("end_frame"), key="end_frame", label="End frame")
        groups = _validate_anchor_groups(payload.get("ingredients"))
        character_group = _resolve_character_reference_group(payload, character)
        if character_group is not None:
            groups = [character_group, *groups]
        conditionings = _plan_conditionings(
            profile,
            start_frame,
            end_frame,
            groups,
            total_frames=total_frames,
        )
        if mode == "motion" and start_frame["image"] is None:
            raise APIError(HTTPStatus.BAD_REQUEST, "Motion Track requires a start-frame image.")
        conditioned_prompt = _build_conditioned_prompt(
            prompt,
            character,
            start_frame["image"] is not None,
            start_frame["description"],
            end_frame["description"],
            groups,
        )
        spec = {
            "mode": mode,
            "profile": profile,
            "seed": seed,
            "duration_seconds": duration_seconds,
            "lora_strength": _finite_number(
                payload.get("lora_strength"),
                field="lora_strength",
                minimum=0,
                maximum=2,
                default=1,
            ),
            "conditioning_strength": _finite_strength(
                payload.get("conditioning_strength"),
                field="conditioning_strength",
                default=1,
            ),
            "conditionings": conditionings,
            "uploads": {},
        }
        spec["aspect_ratio"] = _validate_aspect_ratio(payload.get("aspect_ratio"))
        if mode == "ingredients":
            references = _validate_ingredient_references(payload.get("ingredient_references"))
            dialogue_guidance = _clean_text(
                payload.get("dialogue_guidance", ""),
                field="dialogue_guidance",
                max_chars=2_000,
                max_bytes=8_000,
            )
            if dialogue_guidance:
                conditioned_prompt += (
                    "\n\nSPEAKER-TAGGED DIALOGUE GUIDANCE:\n"
                    "Keep each quoted line associated with its explicit @ingredient tag; do not exchange speakers.\n"
                    + dialogue_guidance
                )
            augmented_text = conditioned_prompt + "".join(
                reference["prompt_label"] + reference["description"]
                for reference in references
            )
            _validate_final_prompt_size(
                augmented_text,
                label="The prompt plus Ingredient labels and descriptions",
            )
            spec["ingredient_references"] = references
        elif mode == "union":
            route = payload.get("union_route", "source")
            if route not in {"source", "guide"}:
                raise APIError(HTTPStatus.BAD_REQUEST, "union_route must be source or guide.")
            control_type = payload.get("control_type", "canny")
            if control_type not in {"canny", "depth", "pose", "prepared"}:
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    "control_type must be canny, depth, pose, or prepared.",
                )
            if route == "source" and control_type != "canny":
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    "Built-in source preprocessing supports Canny only; upload a prepared guide for depth or pose.",
                )
            spec.update({"union_route": route, "control_type": control_type})
            spec["uploads"]["control_video"] = _decode_uploaded_file(
                payload.get("control_video"),
                label="Control video",
                extensions=VIDEO_EXTENSIONS,
                max_bytes=MAX_VIDEO_BYTES,
                media_prefix="video",
            )
        else:
            spec["tracks"] = _validate_motion_tracks(payload.get("tracks"))
        missing, note = _control_mode_assets(mode)
        if missing:
            raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, note)
        with COMPUTE_ADMISSION_LOCK:
            _reject_if_prompt_coach_active()
            if FRAME_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait before loading LTX.")
            if VOICE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading LTX.")
            if UPSCALE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading LTX.")
            response = STATE.start_workflow(conditioned_prompt, spec)
        self._send_json(HTTPStatus.ACCEPTED, response)

    def _handle_generate(self, payload: dict[str, Any]) -> None:
        if FRAME_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait for it to finish before loading LTX.")
        if VOICE_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading LTX.")
        if UPSCALE_STATE.is_busy():
            raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading LTX.")
        allowed_keys = {
            "prompt",
            "profile",
            "seed",
            "reference_image",
            "start_frame",
            "end_frame",
            "ingredients",
            "scenes",
            "generated_keyframes",
            "duration_seconds",
            "gemma_pack",
            "aspect_ratio",
            "character_profile_id",
            "character_reference_images",
        } | {
            key for key, _, _, _ in CHARACTER_FIELDS
        }
        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            raise APIError(HTTPStatus.BAD_REQUEST, "Unknown field(s): " + ", ".join(unknown))
        prompt = _validate_video_prompt(
            payload.get("prompt"),
            empty_message="Enter a prompt before generating.",
        )
        profile = payload.get("profile", "quality")
        if not isinstance(profile, str) or profile not in PROFILES:
            raise APIError(HTTPStatus.BAD_REQUEST, "profile must be draft, balanced, quality, max, or clip3s.")
        seed = payload.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise APIError(HTTPStatus.BAD_REQUEST, "seed must be an integer from 0 to 4,294,967,295.")
        gemma_pack = _resolve_gemma_pack(payload.get("gemma_pack", "official"))
        aspect_ratio = _validate_aspect_ratio(payload.get("aspect_ratio"))
        raw_duration = payload.get("duration_seconds")
        duration_seconds = (
            _validate_duration_seconds(raw_duration) if raw_duration is not None else None
        )
        total_frames = 24 * duration_seconds + 1 if duration_seconds is not None else PROFILE_FRAMES[profile]
        scenes = _validate_scenes(payload.get("scenes"), profile, total_frames=total_frames)
        generated_keyframes = _validate_generated_keyframes(payload.get("generated_keyframes"))
        legacy_image = _decode_reference_image(payload.get("reference_image"))
        start_frame = _validate_frame_control(payload.get("start_frame"), key="start_frame", label="Start frame")
        end_frame = _validate_frame_control(payload.get("end_frame"), key="end_frame", label="End frame")
        groups = _validate_anchor_groups(payload.get("ingredients"))
        character = _validate_character_fields(payload)
        character_group = _resolve_character_reference_group(payload, character)
        if character_group is not None:
            groups = [character_group, *groups]
        conditionings = _plan_conditionings(
            profile,
            start_frame,
            end_frame,
            groups,
            legacy_image,
            total_frames=total_frames,
        )
        conditioned_prompt = _build_conditioned_prompt(
            prompt,
            character,
            legacy_image is not None or start_frame["image"] is not None,
            start_frame["description"],
            end_frame["description"],
            groups,
        )
        relay_text = conditioned_prompt + "".join("\n" + scene["text"] for scene in scenes)
        _validate_final_prompt_size(
            relay_text,
            label="The conditioned prompt plus scene beats",
        )
        start_kwargs: dict[str, Any] = {
            "scenes": scenes,
            "generated_keyframes": generated_keyframes,
            "gemma_pack": gemma_pack,
            "aspect_ratio": aspect_ratio,
        }
        if duration_seconds is not None:
            start_kwargs["duration_seconds"] = duration_seconds
        with COMPUTE_ADMISSION_LOCK:
            _reject_if_prompt_coach_active()
            if FRAME_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A frame is being composed. Wait before loading LTX.")
            if VOICE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "Character voices are being generated. Wait before loading LTX.")
            if UPSCALE_STATE.is_busy():
                raise APIError(HTTPStatus.CONFLICT, "A video upscale is running. Wait before loading LTX.")
            response = STATE.start(conditioned_prompt, profile, seed, conditionings, **start_kwargs)
        self._send_json(HTTPStatus.ACCEPTED, response)

    def _serve_character_media(self, encoded_path: str, head_only: bool) -> None:
        parts = encoded_path.split("/")
        if len(parts) != 2 or not all(parts):
            self._send_error_json(HTTPStatus.NOT_FOUND, "Character reference not found.")
            return
        try:
            profile_id, reference_id = (unquote(part, errors="strict") for part in parts)
            profile_id = _validate_character_profile_id(profile_id)
            if not CHARACTER_REFERENCE_ID_RE.fullmatch(reference_id):
                raise APIError(HTTPStatus.BAD_REQUEST, "Character reference ID is invalid.")
            profile = _read_character_profile(profile_id)
            reference = next(
                (item for item in profile["references"] if item["id"] == reference_id),
                None,
            )
            if reference is None:
                raise APIError(HTTPStatus.NOT_FOUND, "Character reference not found.")
            data = reference["path"].read_bytes()
            if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise APIError(HTTPStatus.CONFLICT, "Character reference failed integrity validation.")
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid character reference path.")
            return
        except OSError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Character reference not found.")
            return
        except APIError as exc:
            self._send_error_json(exc.status, exc.message)
            return
        content_type = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}[reference["kind"]]
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", '"' + reference["sha256"] + '"')
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _serve_voice_media(self, encoded_name: str, head_only: bool) -> None:
        try:
            name = unquote(encoded_name, errors="strict")
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid voice path.")
            return
        if len(name) > MAX_MEDIA_NAME or not VOICE_MEDIA_NAME_RE.fullmatch(name) or "/" in name or "\\" in name:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid voice path.")
            return
        try:
            if VOICE_GENERATED_DIR.is_symlink():
                raise OSError("symbolic-link voice root")
            root = VOICE_GENERATED_DIR.resolve(strict=True)
            unresolved = root / name
            if unresolved.is_symlink():
                raise OSError("symbolic-link voice output")
            candidate = unresolved.resolve(strict=True)
        except OSError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Voice cue not found.")
            return
        if candidate.parent != root or candidate.is_symlink() or not candidate.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Voice cue not found.")
            return
        size = candidate.stat().st_size
        if not 44 < size <= 512 * 1024 * 1024:
            self._send_error_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Voice cue is not a safe WAV artifact.")
            return
        try:
            with candidate.open("rb") as probe:
                header = probe.read(12)
        except OSError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Voice cue not found.")
            return
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            self._send_error_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Voice cue is not a RIFF/WAVE artifact.")
            return
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self._range_error(size)
                return
            first, last = match.groups()
            if not first and not last:
                self._range_error(size)
                return
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            else:
                suffix = int(last)
                if suffix <= 0:
                    self._range_error(size)
                    return
                start = max(0, size - suffix)
            if start >= size or end < start:
                self._range_error(size)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-cache")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with candidate.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_media(self, encoded_name: str, head_only: bool) -> None:
        try:
            name = unquote(encoded_name, errors="strict")
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid media path.")
            return
        if len(name) > MAX_MEDIA_NAME or not MEDIA_NAME_RE.fullmatch(name) or "/" in name or "\\" in name:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid media path.")
            return
        root = GENERATED_DIR.resolve()
        candidate = (root / name).resolve()
        if candidate.parent != root or candidate.is_symlink() or not candidate.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Video not found.")
            return
        size = candidate.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or size == 0:
                self._range_error(size)
                return
            first, last = match.groups()
            if not first and not last:
                self._range_error(size)
                return
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            else:
                suffix = int(last)
                if suffix <= 0:
                    self._range_error(size)
                    return
                start = max(0, size - suffix)
                end = size - 1
            if start >= size or end < start:
                self._range_error(size)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1) if size else 0
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-cache")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only or length == 0:
            return
        with candidate.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_frame_media(self, encoded_name: str, head_only: bool) -> None:
        try:
            name = unquote(encoded_name, errors="strict")
        except UnicodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid frame path.")
            return
        if len(name) > MAX_MEDIA_NAME or not FRAME_MEDIA_NAME_RE.fullmatch(name) or "/" in name or "\\" in name:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid frame path.")
            return
        root = FRAME_GENERATED_DIR.resolve()
        candidate = (root / name).resolve()
        if candidate.parent != root or candidate.is_symlink() or not candidate.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Composed frame not found.")
            return
        try:
            size = candidate.stat().st_size
        except OSError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Composed frame not found.")
            return
        if not 1 <= size <= MAX_IMAGE_BYTES:
            self._send_error_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Composed frame is not a safe anchor image.")
            return
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "private, no-cache")
        self.send_header("Content-Disposition", f'inline; filename="{name}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if head_only:
            return
        with candidate.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _range_error(self, size: int) -> None:
        body = b'{"error":"Invalid byte range."}'
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class LocalThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Ignore expected disconnects when a browser cancels a local request."""
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local LTX-2 MLX video UI on loopback only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"localhost port (default: {DEFAULT_PORT})")
    parser.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    return args


def main() -> None:
    args = _parse_args()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    STATE.restore_latest_completed()
    if FRAME_GENERATED_DIR.is_symlink():
        raise SystemExit(f"Unsafe Frame Composer output symlink: {FRAME_GENERATED_DIR}")
    FRAME_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        server = LocalThreadingHTTPServer((HOST, args.port), LTXRequestHandler)
    except OSError as exc:
        raise SystemExit(f"Could not start LTX Studio on {HOST}:{args.port}: {exc}") from None
    port = server.server_address[1]
    url = f"http://{HOST}:{port}/"
    print(f"LTX Studio is running at {url}", flush=True)
    print("Press Ctrl-C to stop it. Only this Mac can connect.", flush=True)
    if args.open:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    # Convert SIGTERM into the same orderly path as Ctrl-C so service managers
    # and terminal users can stop active render, upscale, frame, and voice
    # subprocess groups before the loopback server releases its port.
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_from_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_from_sigterm)
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\nStopping LTX Studio…", flush=True)
    finally:
        prompt_coach_backend.shutdown_local_runtime()
        UPSCALE_STATE.shutdown()
        VOICE_STATE.shutdown()
        FRAME_STATE.shutdown()
        STATE.shutdown()
        server.server_close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
