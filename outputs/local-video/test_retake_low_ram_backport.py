"""GPU-free structural tests for the pinned Retake/Extend backport."""

from __future__ import annotations

import ast
import unittest

import verify_ltx25 as verification


class RetakeLowRamBackportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli_path = verification.RUNTIME / next(
            name for name in verification.LOW_RAM_EDIT_PATCHES if name.endswith("/cli.py")
        )
        cls.retake_path = verification.RUNTIME / next(
            name for name in verification.LOW_RAM_EDIT_PATCHES if name.endswith("/retake.py")
        )
        cls.cli_source = cls.cli_path.read_text(encoding="utf-8")
        cls.retake_source = cls.retake_path.read_text(encoding="utf-8")
        cls.cli_tree = ast.parse(cls.cli_source)
        cls.retake_tree = ast.parse(cls.retake_source)

    def test_backport_files_match_the_runtime_allowlist(self) -> None:
        verification.verify_low_ram_edit_patch(report=lambda _message: None)

    def test_retake_constructor_accepts_and_forwards_low_ram_streaming(self) -> None:
        classes = [
            node
            for node in ast.walk(self.retake_tree)
            if isinstance(node, ast.ClassDef) and node.name == "RetakePipeline"
        ]
        self.assertEqual(len(classes), 1)
        constructors = [
            node
            for node in classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
        ]
        self.assertEqual(len(constructors), 1)
        argument_names = [argument.arg for argument in constructors[0].args.args]
        self.assertIn("low_ram_streaming", argument_names)
        self.assertTrue(
            any(
                isinstance(node, ast.keyword)
                and node.arg == "low_ram_streaming"
                and isinstance(node.value, ast.Name)
                and node.value.id == "low_ram_streaming"
                for node in ast.walk(constructors[0])
            )
        )

    def test_cli_exposes_low_ram_for_both_edit_commands_and_forwards_it(self) -> None:
        self.assertGreaterEqual(self.cli_source.count('"--low-ram"'), 2)
        forwarded = [
            node
            for node in ast.walk(self.cli_tree)
            if isinstance(node, ast.keyword) and node.arg == "low_ram_streaming"
        ]
        self.assertGreaterEqual(len(forwarded), 2)

    def test_decode_forwards_recorded_source_frame_rate(self) -> None:
        calls = [
            node
            for node in ast.walk(self.cli_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_decode_and_save_video"
        ]
        self.assertTrue(
            any(any(keyword.arg == "frame_rate" for keyword in call.keywords) for call in calls)
        )


if __name__ == "__main__":
    unittest.main()
