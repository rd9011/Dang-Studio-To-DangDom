#!/usr/bin/env python3
"""Import one revision-pinned Gemma mirror as an explicit LTX-2.5 fallback.

This is deliberately *not* a downloader.  It accepts only the known complete
``ddalcu`` safetensors object, verifies its full byte count and SHA-256, parses
the safetensors header without pickle or executable model code, and proves the
tensor names against Phosphene's pinned official index.  Shapes and dtypes for
the loadable text tower are then derived from the pinned official config.

The mirror object is not byte-identical to Phosphene's official tensor file.
It is therefore installed into a separately named directory with a distinct
receipt and is selected only through an explicit launcher option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, NoReturn

from studio_paths import PATHS


HERE = PATHS.app_root
WORKSPACE = PATHS.workspace
MODELS_ROOT = PATHS.models_root
DEFAULT_SOURCE = MODELS_ROOT / ".hf-fallback" / "gemma-ddalcu-model.safetensors.partial"
OFFICIAL_MODEL = MODELS_ROOT / "gemma4-12b-ltx25-q4"
FALLBACK_MODEL = MODELS_ROOT / "gemma4-12b-ltx25-q4-ddalcu-fallback"

SOURCE_REPOSITORY = "ddalcu/LTX-2.5-MLX-Serve-4bit"
SOURCE_REVISION = "e8f8c97cd6e4ff9c997e0d382c027ad9b61776bb"
SOURCE_FILE = "gemma4-12b-ltx-v1/model.safetensors"
SOURCE_BYTES = 6_699_162_168
SOURCE_SHA256 = "be2d3f3d7f13b61a6ab391981b2c8c98f99135edfbde6adcae32d81f155744dd"

OFFICIAL_WEIGHT_NAME = "model-00001-of-00001.safetensors"
OFFICIAL_WEIGHT_BYTES = 6_699_142_782
OFFICIAL_WEIGHT_SHA256 = "a3e8b1629e3cd3974b9c03fd337315c02edbd517fe58300b8e409e2959547704"
SERIALIZATION_SIZE_DELTA = SOURCE_BYTES - OFFICIAL_WEIGHT_BYTES
RECEIPT_NAME = ".ddalcu-gemma-fallback-receipt.json"
RECEIPT_SCHEMA = "local-video-ddalcu-gemma-fallback-receipt/1"
COPY_CHUNK_BYTES = 16 << 20
MAX_SAFETENSORS_HEADER_BYTES = 64 << 20


@dataclass(frozen=True)
class Artifact:
    size: int
    sha256: str


# These are byte-for-byte copies of the official Phosphene pack's metadata,
# tokenizer and license.  The alternate source contributes the tensor file
# only.  Keeping the index is useful evidence: its exact key set must equal the
# alternate safetensors header before import is allowed.
PINNED_METADATA: dict[str, Artifact] = {
    "LICENSE-LTX-2.x-Community-License.md": Artifact(
        34_545, "a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654"
    ),
    "NOTICE.md": Artifact(4_884, "354c701af72ffc11f09bd6d17b8ac59e7021aad3731691e54a8a88daa82bf04e"),
    "chat_template.jinja": Artifact(
        18_683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
    ),
    "config.json": Artifact(4_439, "b1bf058616210d00b126236812396fc990282b377cdffad6d72d8db358039c16"),
    "generation_config.json": Artifact(
        255, "c70f87dc2995fc43406c0bcfb41b69c6d31c2d0c033fa09e536ffabc091ae24c"
    ),
    "model.safetensors.index.json": Artifact(
        108_762, "a49c77f68883ed9076ec2c8807a667425ad18382395960eb9cd35d4646780d0c"
    ),
    "phosphene_quant_manifest.json": Artifact(
        1_077, "72524ed82b106356af5ef60b1a14bce313537c11866bf65d697580b7b9e0a68e"
    ),
    "processor_config.json": Artifact(
        1_382, "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9"
    ),
    "tokenizer.json": Artifact(
        32_169_626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    ),
    "tokenizer_config.json": Artifact(
        3_736, "794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8"
    ),
}


class FallbackImportError(RuntimeError):
    """The alternate Gemma pack failed an identity or compatibility check."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _fail(message: str) -> NoReturn:
    raise FallbackImportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_regular(path) as handle:
        for block in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_regular(path: Path) -> BinaryIO:
    """Open a regular file read-only without following a final symlink."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FallbackImportError(f"cannot securely open {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"refusing non-regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symlink component."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _json_no_duplicates(raw: bytes | str, *, description: str) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FallbackImportError(f"{description} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FallbackImportError(f"invalid {description}: {exc}") from exc


def read_safetensors_schema(path: Path) -> dict[str, TensorInfo]:
    """Parse and bounds-check a safetensors header without loading tensor data."""
    with _open_regular(path) as handle:
        file_size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            _fail(f"{path} is too short to be a safetensors file")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
            _fail(f"unsafe safetensors header size in {path}: {header_size:,} bytes")
        if 8 + header_size > file_size:
            _fail(f"truncated safetensors header in {path}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            _fail(f"truncated safetensors header in {path}")

    parsed = _json_no_duplicates(raw_header, description=f"safetensors header {path}")
    if not isinstance(parsed, dict):
        _fail(f"safetensors header is not an object: {path}")
    metadata = parsed.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        _fail(f"invalid __metadata__ in {path}")

    data_bytes = file_size - 8 - header_size
    tensors: dict[str, TensorInfo] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, raw in parsed.items():
        if not isinstance(name, str) or not name or "\x00" in name:
            _fail(f"invalid tensor name in {path}: {name!r}")
        if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
            _fail(f"invalid descriptor for tensor {name!r} in {path}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            _fail(f"unsupported dtype {dtype!r} for tensor {name!r}")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        ):
            _fail(f"invalid shape for tensor {name!r}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in offsets
        ):
            _fail(f"invalid offsets for tensor {name!r}")
        start, end = offsets
        if not 0 <= start <= end <= data_bytes:
            _fail(f"out-of-bounds offsets for tensor {name!r}")
        elements = math.prod(shape)
        expected_bytes = elements * _DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            _fail(
                f"tensor {name!r} byte span {end - start:,} does not match "
                f"{dtype}{tuple(shape)} ({expected_bytes:,} bytes)"
            )
        info = TensorInfo(dtype, tuple(shape), start, end)
        tensors[name] = info
        ranges.append((start, end, name))

    if not tensors:
        _fail(f"safetensors file contains no tensors: {path}")
    cursor = 0
    for start, end, name in sorted(ranges, key=lambda item: (item[0], item[1], item[2])):
        if start != cursor:
            kind = "overlap" if start < cursor else "gap"
            _fail(f"safetensors data has a {kind} before tensor {name!r} in {path}")
        cursor = end
    if cursor != data_bytes:
        _fail(f"safetensors data has {data_bytes - cursor:,} trailing unclaimed bytes in {path}")
    return tensors


def _verify_artifact(path: Path, artifact: Artifact, *, label: str) -> None:
    try:
        observed_size = path.stat().st_size
    except OSError as exc:
        raise FallbackImportError(f"missing {label}: {path} ({exc})") from exc
    if path.is_symlink() or not path.is_file():
        _fail(f"refusing non-regular or symlinked {label}: {path}")
    if observed_size != artifact.size:
        _fail(f"wrong size for {label}: {observed_size:,} != {artifact.size:,} bytes")
    observed_hash = sha256_file(path)
    if observed_hash != artifact.sha256:
        _fail(f"SHA-256 mismatch for {label}: {observed_hash} != {artifact.sha256}")


def verify_metadata_source(path: Path) -> None:
    for name, artifact in PINNED_METADATA.items():
        _verify_artifact(path / name, artifact, label=f"official metadata {name}")


def _read_pinned_json(directory: Path, name: str) -> dict:
    raw = (directory / name).read_bytes()
    parsed = _json_no_duplicates(raw, description=name)
    if not isinstance(parsed, dict):
        _fail(f"{name} is not a JSON object")
    return parsed


def _validated_config(directory: Path) -> dict:
    config = _read_pinned_json(directory, "config.json")
    quantization = config.get("quantization")
    text = config.get("text_config")
    if config.get("model_type") != "gemma4_unified":
        _fail("config.json is not a gemma4_unified pack")
    if config.get("gemma_version") != "gemma4-12b-ltx-v1":
        _fail("config.json is not the LTX-2.5 Gemma 4 fine-tune")
    if quantization != {"group_size": 64, "bits": 4}:
        _fail(f"unexpected Gemma quantization block: {quantization!r}")
    if not isinstance(text, dict):
        _fail("config.json has no text_config")
    reference = {
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "intermediate_size": 15360,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        "head_dim": 256,
        "global_head_dim": 512,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "vocab_size": 262144,
    }
    drift = [f"{key}={text.get(key)!r} (expected {value!r})" for key, value in reference.items() if text.get(key) != value]
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != reference["num_hidden_layers"]:
        drift.append("layer_types is not an exact 48-entry list")
    elif any(
        value != ("full_attention" if (index + 1) % 6 == 0 else "sliding_attention")
        for index, value in enumerate(layer_types)
    ):
        drift.append("layer_types is not the pinned 5-sliding/1-full pattern")
    if drift:
        _fail("unsupported Gemma text configuration: " + "; ".join(drift))
    return config


def _expected_text_schema(config: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Derive the exact runtime-visible Q4 text-tower schema from config."""
    text = config["text_config"]
    bits = int(config["quantization"]["bits"])
    group_size = int(config["quantization"]["group_size"])
    hidden = int(text["hidden_size"])
    intermediate = int(text["intermediate_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    global_kv_heads = int(text["num_global_key_value_heads"])
    vocab = int(text["vocab_size"])
    schema: dict[str, tuple[str, tuple[int, ...]]] = {}

    def quantized(prefix: str, output: int, input_: int) -> None:
        if input_ % group_size or input_ * bits % 32:
            _fail(f"{prefix} dimensions are incompatible with declared Q{bits}/g{group_size}")
        schema[f"{prefix}.weight"] = ("U32", (output, input_ * bits // 32))
        qparam_shape = (output, input_ // group_size)
        schema[f"{prefix}.scales"] = ("BF16", qparam_shape)
        schema[f"{prefix}.biases"] = ("BF16", qparam_shape)

    quantized("embed_tokens", vocab, hidden)
    layer_types = text["layer_types"]
    for index, layer_type in enumerate(layer_types):
        prefix = f"layers.{index}"
        head_dim = int(text["global_head_dim"] if layer_type == "full_attention" else text["head_dim"])
        layer_kv_heads = global_kv_heads if layer_type == "full_attention" else kv_heads
        attention_width = heads * head_dim
        kv_width = layer_kv_heads * head_dim
        quantized(f"{prefix}.self_attn.q_proj", attention_width, hidden)
        quantized(f"{prefix}.self_attn.o_proj", hidden, attention_width)
        quantized(f"{prefix}.self_attn.k_proj", kv_width, hidden)
        if not (text["attention_k_eq_v"] and layer_type == "full_attention"):
            quantized(f"{prefix}.self_attn.v_proj", kv_width, hidden)
        schema[f"{prefix}.self_attn.q_norm.weight"] = ("BF16", (head_dim,))
        schema[f"{prefix}.self_attn.k_norm.weight"] = ("BF16", (head_dim,))
        quantized(f"{prefix}.mlp.gate_proj", intermediate, hidden)
        quantized(f"{prefix}.mlp.up_proj", intermediate, hidden)
        quantized(f"{prefix}.mlp.down_proj", hidden, intermediate)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            schema[f"{prefix}.{norm}.weight"] = ("BF16", (hidden,))
        schema[f"{prefix}.layer_scalar"] = ("BF16", (1,))
    schema["norm.weight"] = ("BF16", (hidden,))
    return schema


_NON_TEXT_PREFIXES = (
    "vision_tower",
    "audio_tower",
    "vision_model",
    "audio_model",
    "multi_modal_projector",
    "audio_projector",
    "embed_vision",
    "embed_audio",
    "vision_embedder",
    "audio_embedder",
)
_DROPPED_SUBSTRINGS = ("self_attn.rotary_emb", "input_max", "input_min", "output_max", "output_min")


def _runtime_name(raw_name: str, config: dict) -> str | None:
    name = raw_name
    for prefix in ("model.language_model.", "language_model.model.", "language_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith(_NON_TEXT_PREFIXES) or raw_name.startswith(_NON_TEXT_PREFIXES):
        return None
    if any(marker in name for marker in _DROPPED_SUBSTRINGS) or name.startswith("lm_head"):
        return None
    # The pinned 12B config has no KV-shared layers.  Gemma 4 full-attention
    # layers intentionally have no V projection when K == V.
    if ".self_attn.v_proj" in name:
        try:
            index = int(name.split("layers.", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            return name
        if config["text_config"]["layer_types"][index] == "full_attention":
            return None
    return name


def _official_index_keys(directory: Path) -> set[str]:
    index = _read_pinned_json(directory, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        _fail("official model.safetensors.index.json has no weight_map")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
        _fail("official weight_map is not string-to-string")
    if any(Path(value).name != value or value != OFFICIAL_WEIGHT_NAME for value in weight_map.values()):
        _fail("official weight_map points outside the pinned single-shard filename")
    return set(weight_map)


def tensor_schema_sha256(tensors: dict[str, TensorInfo]) -> str:
    canonical = [
        {"name": name, "dtype": info.dtype, "shape": list(info.shape)}
        for name, info in sorted(tensors.items())
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_tensor_compatibility(tensors: dict[str, TensorInfo], metadata_dir: Path) -> str:
    """Validate raw names against the index and loadable shapes against config."""
    config = _validated_config(metadata_dir)
    indexed = _official_index_keys(metadata_dir)
    observed = set(tensors)
    if observed != indexed:
        missing = sorted(indexed - observed)
        extra = sorted(observed - indexed)
        _fail(
            "mirror tensor names differ from the official index: "
            f"missing={missing[:8]!r}; extra={extra[:8]!r}"
        )

    normalized: dict[str, TensorInfo] = {}
    for raw_name, info in tensors.items():
        name = _runtime_name(raw_name, config)
        if name is None:
            continue
        if name in normalized:
            _fail(f"multiple source tensors normalize to runtime key {name!r}")
        normalized[name] = info

    expected = _expected_text_schema(config)
    if set(normalized) != set(expected):
        missing = sorted(set(expected) - set(normalized))
        extra = sorted(set(normalized) - set(expected))
        _fail(
            "mirror is not an exact loadable Gemma 4 text-tower schema: "
            f"missing={missing[:8]!r}; extra={extra[:8]!r}"
        )
    mismatches = []
    for name, (dtype, shape) in expected.items():
        actual = normalized[name]
        if actual.dtype != dtype or actual.shape != shape:
            mismatches.append(
                f"{name}: {actual.dtype}{actual.shape} != {dtype}{shape}"
            )
    if mismatches:
        _fail("mirror tensor shape/dtype mismatch: " + "; ".join(mismatches[:12]))
    return tensor_schema_sha256(tensors)


def inspect_source(source: Path, metadata_dir: Path, *, report: Callable[[str], None] = print) -> tuple[dict[str, TensorInfo], str]:
    """Fully authenticate and structurally validate an already-downloaded source."""
    verify_metadata_source(metadata_dir)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise FallbackImportError(f"mirror source is missing: {source} ({exc})") from exc
    if source.is_symlink() or not source.is_file():
        _fail(f"refusing non-regular or symlinked mirror source: {source}")
    if size != SOURCE_BYTES:
        _fail(
            f"mirror source is incomplete or wrong: {size:,} != {SOURCE_BYTES:,} bytes; "
            "the active .partial file was left untouched"
        )
    digest = sha256_file(source)
    if digest != SOURCE_SHA256:
        _fail(f"mirror source SHA-256 mismatch: {digest} != {SOURCE_SHA256}")
    report(f"OK mirror identity {SOURCE_REPOSITORY}@{SOURCE_REVISION} ({digest})")
    tensors = read_safetensors_schema(source)
    schema_digest = validate_tensor_compatibility(tensors, metadata_dir)
    report(f"OK {len(tensors):,} tensor names/shapes/dtypes against pinned official index/config")
    return tensors, schema_digest


def _copy_and_hash(source: Path, destination: Path) -> str:
    def identity(info: os.stat_result) -> tuple[int, int, int, int]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    digest = hashlib.sha256()
    with _open_regular(source) as source_handle:
        initial = os.fstat(source_handle.fileno())
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as destination_handle:
                while True:
                    block = source_handle.read(COPY_CHUNK_BYTES)
                    if not block:
                        break
                    digest.update(block)
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except Exception:
            try:
                destination.unlink()
            except OSError:
                pass
            raise
        final = os.fstat(source_handle.fileno())
    if identity(initial) != identity(final):
        _fail("mirror source changed while it was being copied; refusing import")
    return digest.hexdigest()


def _receipt(schema_digest: str, tensor_count: int) -> dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "file": SOURCE_FILE,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
        },
        "installed_weight": {
            "file": OFFICIAL_WEIGHT_NAME,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
        },
        "official_reference_weight": {
            "file": OFFICIAL_WEIGHT_NAME,
            "bytes": OFFICIAL_WEIGHT_BYTES,
            "sha256": OFFICIAL_WEIGHT_SHA256,
            "byte_identical": False,
            "serialization_size_delta": SERIALIZATION_SIZE_DELTA,
        },
        "compatibility": {
            "official_index_sha256": PINNED_METADATA["model.safetensors.index.json"].sha256,
            "official_config_sha256": PINNED_METADATA["config.json"].sha256,
            "tensor_count": tensor_count,
            "tensor_schema_sha256": schema_digest,
            "basis": "exact index keys plus config-derived runtime text-tower shapes and dtypes",
        },
        "metadata": {
            name: {"bytes": artifact.size, "sha256": artifact.sha256}
            for name, artifact in sorted(PINNED_METADATA.items())
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _cleanup_private_staging(path: Path, parent: Path) -> None:
    """Remove only the exact private directory created by this process."""
    try:
        resolved = path.resolve()
        resolved_parent = parent.resolve()
        if resolved.parent != resolved_parent or not resolved.name.startswith(".ddalcu-gemma-import-"):
            return
        for child in resolved.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
            elif child.is_symlink():
                child.unlink()
            else:
                return
        resolved.rmdir()
    except OSError:
        pass


def install_fallback(
    source: Path = DEFAULT_SOURCE,
    metadata_dir: Path = OFFICIAL_MODEL,
    destination: Path = FALLBACK_MODEL,
    *,
    report: Callable[[str], None] = print,
) -> None:
    """Copy, authenticate and atomically publish the alternate pack."""
    # Keep the lexical source name.  ``Path.resolve()`` would dereference a
    # symlink before the lstat/O_NOFOLLOW checks and defeat their purpose.
    source = _lexical_absolute(source)
    metadata_dir = _lexical_absolute(metadata_dir)
    destination = _lexical_absolute(destination)
    if destination.exists() or destination.is_symlink():
        _fail(f"destination already exists; verify it or choose another path: {destination}")

    try:
        source_info = source.lstat()
    except OSError as exc:
        raise FallbackImportError(f"mirror source is missing: {source} ({exc})") from exc
    if not stat.S_ISREG(source_info.st_mode):
        _fail(f"refusing non-regular or symlinked mirror source: {source}")
    source_size = source_info.st_size
    if source_size != SOURCE_BYTES:
        _fail(
            f"mirror source is incomplete or wrong: {source_size:,} != {SOURCE_BYTES:,} bytes; "
            "no data was modified"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    verify_metadata_source(metadata_dir)

    staging = Path(tempfile.mkdtemp(prefix=".ddalcu-gemma-import-", dir=destination.parent))
    os.chmod(staging, 0o700)
    try:
        installed_weight = staging / OFFICIAL_WEIGHT_NAME
        digest = _copy_and_hash(source, installed_weight)
        if digest != SOURCE_SHA256:
            _fail(f"mirror source SHA-256 mismatch after private copy: {digest} != {SOURCE_SHA256}")
        if installed_weight.stat().st_size != SOURCE_BYTES:
            _fail("private tensor copy has the wrong size")

        # The expensive identity check is complete.  Only now parse and trust
        # the safetensors structure.
        tensors = read_safetensors_schema(installed_weight)
        schema_digest = validate_tensor_compatibility(tensors, metadata_dir)

        for name, artifact in PINNED_METADATA.items():
            copied = staging / name
            copied_digest = _copy_and_hash(metadata_dir / name, copied)
            if copied.stat().st_size != artifact.size or copied_digest != artifact.sha256:
                _fail(f"metadata changed while copying: {name}")

        receipt = _receipt(schema_digest, len(tensors))
        receipt["installed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(staging / RECEIPT_NAME, receipt)
        for child in staging.iterdir():
            os.chmod(child, 0o444)
        os.chmod(staging, 0o755)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(staging, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        _cleanup_private_staging(staging, destination.parent)
        raise

    report(f"INSTALLED explicit fallback: {destination}")
    report(
        "NOTE: tensor values were authenticated to the pinned mirror, but the file is not "
        "byte-identical to Phosphene's official serialization."
    )


def _read_receipt(path: Path) -> dict:
    receipt_path = path / RECEIPT_NAME
    try:
        receipt = _json_no_duplicates(receipt_path.read_bytes(), description=RECEIPT_NAME)
    except OSError as exc:
        raise FallbackImportError(f"missing fallback receipt: {receipt_path} ({exc})") from exc
    if not isinstance(receipt, dict):
        _fail("fallback receipt is not a JSON object")
    return receipt


def verify_fallback_install(
    path: Path = FALLBACK_MODEL,
    *,
    full_hash: bool = False,
    report: Callable[[str], None] = print,
) -> None:
    """Verify the distinct alternate pack; never accepts an in-progress file."""
    path = _lexical_absolute(path)
    if path.is_symlink() or not path.is_dir():
        _fail(f"fallback Gemma directory is missing or unsafe: {path}")
    receipt = _read_receipt(path)
    expected = _receipt(
        str(receipt.get("compatibility", {}).get("tensor_schema_sha256", ""))
        if isinstance(receipt.get("compatibility"), dict)
        else "",
        int(receipt.get("compatibility", {}).get("tensor_count", -1))
        if isinstance(receipt.get("compatibility"), dict)
        and isinstance(receipt.get("compatibility", {}).get("tensor_count"), int)
        else -1,
    )
    for field in (
        "schema",
        "source",
        "installed_weight",
        "official_reference_weight",
        "metadata",
    ):
        if receipt.get(field) != expected[field]:
            _fail(f"fallback receipt identity mismatch in {field}")
    compatibility = receipt.get("compatibility")
    if not isinstance(compatibility, dict):
        _fail("fallback receipt has no compatibility record")
    for field in ("official_index_sha256", "official_config_sha256", "basis"):
        if compatibility.get(field) != expected["compatibility"][field]:
            _fail(f"fallback receipt compatibility mismatch in {field}")

    for name, artifact in PINNED_METADATA.items():
        _verify_artifact(path / name, artifact, label=f"fallback metadata {name}")
    weight = path / OFFICIAL_WEIGHT_NAME
    try:
        weight_size = weight.stat().st_size
    except OSError as exc:
        raise FallbackImportError(f"fallback tensor file is missing: {weight} ({exc})") from exc
    if weight.is_symlink() or not weight.is_file() or weight_size != SOURCE_BYTES:
        _fail(f"fallback tensor file has the wrong type or size: {weight}")
    if full_hash:
        digest = sha256_file(weight)
        if digest != SOURCE_SHA256:
            _fail(f"fallback tensor SHA-256 mismatch: {digest} != {SOURCE_SHA256}")
        report(f"OK fallback tensor SHA-256 {digest}")

    tensors = read_safetensors_schema(weight)
    schema_digest = validate_tensor_compatibility(tensors, path)
    if compatibility.get("tensor_count") != len(tensors):
        _fail("fallback receipt tensor_count does not match the installed header")
    if compatibility.get("tensor_schema_sha256") != schema_digest:
        _fail("fallback receipt tensor schema digest does not match the installed header")
    partials = sorted(
        child.name
        for child in path.iterdir()
        if child.is_file()
        and (child.name.endswith((".part", ".partial", ".incomplete", ".download", ".assembling")) or child.name.startswith(".tmp"))
    )
    if partials:
        _fail(f"incomplete files remain inside fallback directory: {partials!r}")
    report(
        f"OK explicit ddalcu fallback ({len(tensors):,} tensors; "
        f"{'full hash' if full_hash else 'size + receipt + structural schema'})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and import the pinned ddalcu Gemma file as a separate LTX-2.5 fallback."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-source", action="store_true", help="hash and structurally inspect the source; write nothing")
    mode.add_argument("--install", action="store_true", help="copy into a private directory and atomically publish the fallback")
    mode.add_argument("--verify", action="store_true", help="verify an already-installed fallback directory")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metadata-source", type=Path, default=OFFICIAL_MODEL)
    parser.add_argument("--destination", type=Path, default=FALLBACK_MODEL)
    parser.add_argument("--full-hash", action="store_true", help="with --verify, hash the 6.24 GiB tensor file again")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.inspect_source:
            inspect_source(args.source, args.metadata_source)
        elif args.install:
            install_fallback(args.source, args.metadata_source, args.destination)
        else:
            verify_fallback_install(args.destination, full_hash=args.full_hash)
    except (FallbackImportError, OSError, ValueError) as exc:
        print(f"FALLBACK IMPORT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
