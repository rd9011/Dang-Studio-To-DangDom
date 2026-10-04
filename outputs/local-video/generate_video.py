#!/usr/bin/env python3
"""Run the pinned FastMetal-5B-QAD pipeline with safe local defaults."""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RUNTIME = WORKSPACE / "work" / "local-video" / "FastVideo"
VENV_PYTHON = RUNTIME / ".venv" / "bin" / "python"
MODEL = WORKSPACE / "work" / "local-video" / "models" / "FastMetal-5B-QAD"
CACHE = WORKSPACE / "work" / "local-video" / "cache"
ENTRYPOINT = RUNTIME / "examples" / "inference" / "basic" / "mlx_wan22_generate.py"

SOURCE_REVISION = "d995516da00c24105aa841df1e690d3cd8a6c173"
MODEL_REVISION = "5e4819ec9c8dd062443cf015bd9022c700eca458"
TAEHV_SHA256 = "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"

MODEL_FILES = {
    ".gitattributes": 1_580,
    "README.md": 2_034,
    "mlx_dit.json": 31_940,
    "mlx_dit.safetensors": 5_314_678_720,
    "model_index.json": 502,
    "scheduler/scheduler_config.json": 820,
    "text_encoder/config.json": 855,
    "text_encoder/model-00001-of-00003.safetensors": 4_935_812_536,
    "text_encoder/model-00002-of-00003.safetensors": 4_983_103_192,
    "text_encoder/model-00003-of-00003.safetensors": 1_442_935_480,
    "text_encoder/model.safetensors.index.json": 22_476,
    "tokenizer/special_tokens_map.json": 7_079,
    "tokenizer/spiece.model": 4_548_313,
    "tokenizer/tokenizer.json": 16_837_459,
    "tokenizer/tokenizer_config.json": 61_758,
    "vae/config.json": 1_701,
    "vae/diffusion_pytorch_model.safetensors": 2_818_777_808,
}

PROFILES = {
    "smoke": {
        "height": 448,
        "width": 832,
        "frames": 41,
        "refine": False,
        "decoder": "taehv",
        "note": "fast installation check (about 1.7 seconds of video)",
    },
    "quality": {
        "height": 704,
        "width": 1280,
        "frames": 81,
        "refine": True,
        "decoder": "taehv",
        "note": "best officially benchmarked 16 GB profile (about 3.4 seconds)",
    },
    "max": {
        "height": 704,
        "width": 1280,
        "frames": 121,
        "refine": True,
        "decoder": "taehv",
        "note": "five-second maximum attempt; more memory and time than the validated profile",
    },
    "full-vae": {
        "height": 448,
        "width": 832,
        "frames": 81,
        "refine": True,
        "decoder": "wan-vae",
        "note": "slower full Wan VAE experiment at a conservative resolution",
    },
}


def _bootstrap_venv() -> None:
    if os.environ.get("FASTMETAL_BOOTSTRAPPED") == "1":
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"FastVideo environment is missing: {VENV_PYTHON}")
    env = os.environ.copy()
    env.update(
        {
            "FASTMETAL_BOOTSTRAPPED": "1",
            "HF_HOME": str(WORKSPACE / "work" / "local-video" / "hf-home"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a silent local video with FastMetal-5B-QAD on Apple Silicon."
    )
    parser.add_argument("prompt", help="Detailed text description of the desired shot")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="quality",
        help="quality is the recommended final profile; smoke verifies the install",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--renoise-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="MP4 path; relative paths are placed beside this launcher")
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="add FastVideo's deterministic cinematic template (useful for short prompts)",
    )
    parser.add_argument("--compile", action="store_true", help="compile the MLX DiT after eager mode is proven")
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="also save denoised latents so another decoder can be compared without denoising again",
    )
    return parser.parse_args()


def _check_model() -> None:
    failures: list[str] = []
    for relative, expected_size in MODEL_FILES.items():
        path = MODEL / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif path.stat().st_size != expected_size:
            failures.append(f"wrong size {relative}: {path.stat().st_size} != {expected_size}")
    partials = list(MODEL.rglob("*.incomplete")) if MODEL.exists() else []
    if partials:
        failures.append(f"{len(partials)} incomplete download file(s) remain")
    if failures:
        details = "\n  - ".join(failures)
        raise SystemExit(
            "FastMetal model preflight failed:\n  - "
            + details
            + f"\nExpected pinned model revision: {MODEL_REVISION}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_runtime() -> None:
    sys.path.insert(0, str(RUNTIME))
    import truststore

    truststore.inject_into_ssl()

    # Upstream hard-codes ~/.cache for TAEHV. Keep this installation self-contained.
    from fastvideo.mlx_runtime import wan_vae

    taehv_dir = CACHE / "fastvideo" / "taehv"
    taehv_dir.mkdir(parents=True, exist_ok=True)
    wan_vae._cache_dir = lambda: taehv_dir  # type: ignore[attr-defined]

    # decode_latents_to_video imports this symbol lazily from diffusers.utils.
    # Patch that public module attribute so the later local import sees the
    # high-quality wrapper. Diffusers otherwise defaults to quality=5/10.
    import diffusers.utils as diffusers_utils

    original_export = diffusers_utils.export_to_video

    def high_quality_export(video_frames, output_video_path, fps=10, **kwargs):
        kwargs.setdefault("quality", 10.0)
        return original_export(video_frames, output_video_path, fps=fps, **kwargs)

    diffusers_utils.export_to_video = high_quality_export


def _ensure_taehv() -> None:
    from fastvideo.mlx_runtime.wan_vae import ensure_taehv_checkpoint

    checkpoint = ensure_taehv_checkpoint(z_dim=48)
    actual = _sha256(checkpoint)
    if actual != TAEHV_SHA256:
        raise SystemExit(f"TAEHV checksum mismatch: {actual} != {TAEHV_SHA256}")


def main() -> None:
    _bootstrap_venv()
    args = _parse_args()
    _check_model()
    _configure_runtime()

    profile = PROFILES[args.profile]
    if profile["decoder"] == "taehv":
        _ensure_taehv()

    free_gib = shutil.disk_usage(WORKSPACE).free / (1024**3)
    if free_gib < 8:
        raise SystemExit(f"Only {free_gib:.1f} GiB is free; leave at least 8 GiB for macOS swap and output.")
    if free_gib < 15:
        print(f"WARNING: only {free_gib:.1f} GiB is free; 15+ GiB is recommended before rendering.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or (HERE / f"fastmetal-{args.profile}-{timestamp}.mp4")
    if not output.is_absolute():
        output = HERE / output
    output = output.resolve()
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = output.with_suffix(".metrics.json")
    prompt_key = hashlib.sha256(
        f"{args.prompt}\0enhance={args.enhance_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    prompt_cache = CACHE / "prompt-embeds" / f"{prompt_key}.npy"
    prompt_cache.parent.mkdir(parents=True, exist_ok=True)

    upstream_args = [
        str(ENTRYPOINT),
        "--mlx-checkpoint",
        str(MODEL),
        "--text-encoder-root",
        str(MODEL),
        "--vae-root",
        str(MODEL / "vae"),
        "--text-encoder-device",
        "cpu",
        "--height",
        str(profile["height"]),
        "--width",
        str(profile["width"]),
        "--num-frames",
        str(profile["frames"]),
        "--fps",
        "24",
        "--seed",
        str(args.seed),
        "--renoise-seed",
        str(args.renoise_seed),
        "--decode-backend",
        str(profile["decoder"]),
        "--prompt-embeds-cache",
        str(prompt_cache),
        "--prompt",
        args.prompt,
        "--output-path",
        str(output),
        "--metrics-json",
        str(metrics),
    ]
    if profile["refine"]:
        upstream_args.append("--refine")
    if args.enhance_prompt:
        upstream_args.append("--enhance-prompt")
    if args.compile:
        upstream_args.append("--compile")
    if args.save_latents:
        upstream_args.extend(["--save-latents", str(output.with_suffix(".latents.npz"))])

    print(f"FastMetal source: {SOURCE_REVISION}")
    print(f"FastMetal model:  {MODEL_REVISION}")
    print(f"Profile: {args.profile} — {profile['note']}")
    print("Export quality: 10/10")
    print(f"Output: {output}")

    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        os.chdir(RUNTIME)
        sys.argv = upstream_args
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
