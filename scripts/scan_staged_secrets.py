#!/usr/bin/env python3
"""Fail a commit when staged content looks unsafe for the public repository."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


MAX_SOURCE_BYTES = 20 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".ckpt",
    ".dmg",
    ".gguf",
    ".key",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".p12",
    ".pem",
    ".pkg",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer credential": re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def staged_paths() -> list[str]:
    raw = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def staged_blob(path: str) -> bytes:
    return git("show", f":{path}")


def tracked_paths() -> list[str]:
    raw = git("ls-files", "-z")
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def tracked_blob(path: str) -> bytes:
    return git("show", f"HEAD:{path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="scan every file tracked by HEAD (intended for public-repository CI)",
    )
    args = parser.parse_args(argv)

    findings: list[str] = []
    paths = tracked_paths() if args.tracked else staged_paths()
    read_blob = tracked_blob if args.tracked else staged_blob
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix in BLOCKED_SUFFIXES:
            findings.append(f"{path}: blocked binary/model/media type {suffix}")
            continue
        try:
            content = read_blob(path)
        except subprocess.CalledProcessError:
            continue
        if len(content) > MAX_SOURCE_BYTES:
            findings.append(f"{path}: staged file is larger than 20 MiB")
            continue
        if b"\0" in content[:8192]:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path}: possible {label}")

    if findings:
        print("Public-repository safety check failed:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        print("No secret values were printed. Remove or unstage the affected files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
