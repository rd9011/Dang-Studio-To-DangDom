#!/usr/bin/env python3
"""Encode one prompt through LTX-2.5 Gemma + connector, without rendering."""

from __future__ import annotations

import argparse
import os
import sys

from verify_ltx25 import (
    DDALCU_FALLBACK_GEMMA_MODEL,
    GEMMA_MODEL,
    MODEL,
    VerificationError,
    WORKSPACE,
    verify_install,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU smoke-test only the LTX-2.5 prompt-encoding path; no video is generated."
    )
    parser.add_argument("prompt", nargs="?", default="A red ball resting on a clean white table.")
    parser.add_argument(
        "--gemma-pack",
        choices=("official", "ddalcu-fallback"),
        default="ddalcu-fallback",
        help="which already-verified Gemma directory to exercise",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        choices=(128, 256),
        default=128,
        metavar="TOKENS",
        help=(
            "small connector-safe sequence for the smoke test; lengths must be multiples of "
            "the model's 128 register tokens, while production generation uses 1024"
        ),
    )
    parser.add_argument("--full-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prompt = args.prompt.strip()
    if not prompt:
        print("PROMPT SMOKE FAILED: prompt cannot be empty", file=sys.stderr)
        return 2
    gemma_model = (
        DDALCU_FALLBACK_GEMMA_MODEL if args.gemma_pack == "ddalcu-fallback" else GEMMA_MODEL
    )
    try:
        verify_install(full_hash=args.full_hash, require_gpu=True, gemma_model=gemma_model)
    except (VerificationError, OSError, ValueError) as exc:
        print(f"PROMPT SMOKE PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 2

    cache_root = WORKSPACE / "work" / "local-video" / "cache" / "ltx25"
    os.environ.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            "LTX2_GEMMA_MAX_LENGTH": str(args.max_length),
        }
    )

    try:
        import mlx.core as mx
        from ltx_pipelines_mlx.utils.blocks import PromptEncoder

        encoder = PromptEncoder(MODEL, gemma_model)
        video, audio = encoder(prompt)
        mx.eval(video, audio)
        finite = bool(mx.all(mx.isfinite(video))) and bool(mx.all(mx.isfinite(audio)))
        if not finite:
            raise RuntimeError("prompt embeddings contain NaN or infinity")
        print(
            "PROMPT ENCODING VERIFIED: "
            f"pack={args.gemma_pack}, video_shape={tuple(video.shape)}, "
            f"audio_shape={tuple(audio.shape)}, finite=True"
        )
    except Exception as exc:  # noqa: BLE001 - this is a user-facing hardware smoke test
        print(f"PROMPT SMOKE FAILED: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
