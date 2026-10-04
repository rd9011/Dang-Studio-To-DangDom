from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import import_ddalcu_gemma_fallback as subject


def write_safetensors(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class SafetensorsHeaderTests(unittest.TestCase):
    def test_valid_header_is_read_without_tensor_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            write_safetensors(
                path,
                [
                    ("model.a", "U32", (1, 1), b"\0" * 4),
                    ("model.b", "BF16", (2,), b"\0" * 4),
                ],
            )
            schema = subject.read_safetensors_schema(path)
            self.assertEqual(schema["model.a"].shape, (1, 1))
            self.assertEqual(schema["model.b"].dtype, "BF16")

    def test_descriptor_byte_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.safetensors"
            header = json.dumps(
                {"a": {"dtype": "U32", "shape": [2], "data_offsets": [0, 4]}}
            ).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            with self.assertRaisesRegex(subject.FallbackImportError, "does not match"):
                subject.read_safetensors_schema(path)

    def test_gaps_overlaps_and_trailing_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gap.safetensors"
            header = json.dumps(
                {"a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}
            ).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(subject.FallbackImportError, "gap"):
                subject.read_safetensors_schema(path)

    def test_duplicate_header_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            header = (
                b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
                b'"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
            )
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")
            with self.assertRaisesRegex(subject.FallbackImportError, "repeats JSON key"):
                subject.read_safetensors_schema(path)


class CompatibilityTests(unittest.TestCase):
    def test_pins_distinguish_mirror_from_official_serialization(self) -> None:
        self.assertEqual(subject.SOURCE_REPOSITORY, "ddalcu/LTX-2.5-MLX-Serve-4bit")
        self.assertEqual(subject.SOURCE_REVISION, "e8f8c97cd6e4ff9c997e0d382c027ad9b61776bb")
        self.assertEqual(subject.SOURCE_BYTES, 6_699_162_168)
        self.assertEqual(subject.OFFICIAL_WEIGHT_BYTES, 6_699_142_782)
        self.assertEqual(subject.SERIALIZATION_SIZE_DELTA, 19_386)
        self.assertNotEqual(subject.SOURCE_SHA256, subject.OFFICIAL_WEIGHT_SHA256)
        receipt = subject._receipt("1" * 64, 123)
        self.assertFalse(receipt["official_reference_weight"]["byte_identical"])

    def test_expected_schema_tracks_gemma4_full_attention_k_equals_v(self) -> None:
        config = {
            "quantization": {"group_size": 64, "bits": 4},
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "attention_k_eq_v": True,
                "head_dim": 32,
                "global_head_dim": 64,
                "vocab_size": 128,
                "layer_types": ["sliding_attention", "full_attention"],
            },
        }
        schema = subject._expected_text_schema(config)
        self.assertIn("layers.0.self_attn.v_proj.weight", schema)
        self.assertNotIn("layers.1.self_attn.v_proj.weight", schema)
        self.assertEqual(schema["embed_tokens.weight"], ("U32", (128, 8)))
        self.assertEqual(schema["layers.1.self_attn.o_proj.scales"], ("BF16", (64, 2)))

    def test_exact_index_names_and_config_shapes_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "model_type": "gemma4_unified",
                "gemma_version": "gemma4-12b-ltx-v1",
                "quantization": {"group_size": 64, "bits": 4},
                "text_config": {
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
                    "layer_types": [
                        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
                        for index in range(48)
                    ],
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            expected = subject._expected_text_schema(config)
            tensors = {}
            weight_map = {}
            cursor = 0
            for name, (dtype, shape) in expected.items():
                raw = f"model.{name}"
                tensors[raw] = subject.TensorInfo(dtype, shape, cursor, cursor)
                weight_map[raw] = subject.OFFICIAL_WEIGHT_NAME
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
            )
            digest = subject.validate_tensor_compatibility(tensors, root)
            self.assertEqual(len(digest), 64)

            removed = next(iter(tensors))
            del tensors[removed]
            with self.assertRaisesRegex(subject.FallbackImportError, "official index"):
                subject.validate_tensor_compatibility(tensors, root)

            tensors[removed] = subject.TensorInfo(
                expected[removed.removeprefix("model.")][0],
                (1,),
                cursor,
                cursor,
            )
            with self.assertRaisesRegex(subject.FallbackImportError, "shape/dtype mismatch"):
                subject.validate_tensor_compatibility(tensors, root)


class AtomicImportTests(unittest.TestCase):
    def test_copy_and_hash_real_success_path_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "destination.bin"
            payload = (b"real-copy-path" * 1024) + b"end"
            source.write_bytes(payload)

            digest = subject._copy_and_hash(source, destination)

            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(source.read_bytes(), payload)

    def test_incomplete_active_partial_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "active.partial"
            source.write_bytes(b"still downloading")
            before = source.read_bytes()
            with patch.object(subject, "PINNED_METADATA", {}), patch.object(subject, "SOURCE_BYTES", 100):
                with self.assertRaisesRegex(subject.FallbackImportError, "left untouched"):
                    subject.inspect_source(source, root, report=lambda _message: None)
            self.assertEqual(source.read_bytes(), before)

    def test_install_copies_then_atomically_publishes_distinct_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "active.partial"
            write_safetensors(source, [("model.a", "U8", (4,), b"data")])
            source_bytes = source.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()

            metadata = root / "official-small-files"
            metadata.mkdir()
            (metadata / "config.json").write_text("{}")
            (metadata / "model.safetensors.index.json").write_text("{}")
            pins = {
                name: subject.Artifact(path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
                for name, path in (
                    ("config.json", metadata / "config.json"),
                    ("model.safetensors.index.json", metadata / "model.safetensors.index.json"),
                )
            }
            destination = root / "explicit-fallback"
            with (
                patch.object(subject, "SOURCE_BYTES", len(source_bytes)),
                patch.object(subject, "SOURCE_SHA256", source_hash),
                patch.object(subject, "PINNED_METADATA", pins),
                patch.object(subject, "validate_tensor_compatibility", return_value="c" * 64),
            ):
                subject.install_fallback(source, metadata, destination, report=lambda _message: None)
                self.assertTrue((destination / subject.RECEIPT_NAME).is_file())
                self.assertEqual((destination / subject.OFFICIAL_WEIGHT_NAME).read_bytes(), source_bytes)
                self.assertEqual(source.read_bytes(), source_bytes)
                subject.verify_fallback_install(destination, full_hash=True, report=lambda _message: None)

    def test_existing_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "existing"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("mine")
            with self.assertRaisesRegex(subject.FallbackImportError, "already exists"):
                subject.install_fallback(root / "source", root / "metadata", destination)
            self.assertEqual(marker.read_text(), "mine")

    def test_install_rejects_symlinked_source_before_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_source = root / "real.safetensors"
            write_safetensors(real_source, [("model.a", "U8", (4,), b"data")])
            source_link = root / "active.partial"
            source_link.symlink_to(real_source)
            destination = root / "fallback"

            with patch.object(subject, "SOURCE_BYTES", real_source.stat().st_size):
                with self.assertRaisesRegex(subject.FallbackImportError, "symlinked mirror source"):
                    subject.install_fallback(source_link, root / "metadata", destination)

            self.assertTrue(source_link.is_symlink())
            self.assertEqual(real_source.read_bytes(), source_link.read_bytes())
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
