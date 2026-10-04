from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import unittest
import sys
import tempfile
import wave
from http import HTTPStatus
from pathlib import Path
from unittest.mock import Mock, patch

import ltx_ui as ui
import character_voice as voice


PNG_1X1_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def write_voice_wave(path: Path, *, seconds: float = 3.2, sample_rate: int = 24_000) -> None:
    frames = int(seconds * sample_rate)
    payload = bytearray()
    for index in range(frames):
        value = int(4_000 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
        payload.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


class LtxUiFeatureTests(unittest.TestCase):
    def test_json_response_ignores_browser_disconnect_during_polling(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler.command = "GET"
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError
        handler.send_response = Mock()
        handler._security_headers = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        handler._send_json(HTTPStatus.OK, {"status": "ready"})

        handler.wfile.write.assert_called_once()

    def test_metal_sandbox_guard_detects_denial_without_importing_mlx(self):
        checker = Mock(return_value=1)
        library = Mock(sandbox_check=checker)
        with (
            patch.object(ui.sys, "platform", "darwin"),
            patch("ctypes.CDLL", return_value=library),
        ):
            reason = ui._metal_sandbox_block_reason()

        self.assertIn("macOS blocked Apple GPU/Metal access", reason)
        self.assertIn("launch_ui.command", reason)
        checker.assert_called_once_with(os.getpid(), b"iokit-open-user-client", 0)

    def test_metal_sandbox_guard_allows_normal_desktop_process(self):
        checker = Mock(return_value=0)
        library = Mock(sandbox_check=checker)
        with (
            patch.object(ui.sys, "platform", "darwin"),
            patch("ctypes.CDLL", return_value=library),
        ):
            self.assertIsNone(ui._metal_sandbox_block_reason())

    def test_video_job_fails_cleanly_before_subprocess_when_metal_is_sandboxed(self):
        state = ui.JobState()
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(ui, "GENERATED_DIR", Path(temporary)),
                patch.object(ui, "GENERATOR_SCRIPT", Path(__file__)),
                patch.object(ui, "GENERATOR_PYTHON", Path(sys.executable)),
                patch.object(ui.threading, "Thread"),
            ):
                state.start("one shot", "clip3s", 17, [])
                assert state._job is not None
                job_id = state._job["id"]
                with (
                    patch.object(ui, "_metal_sandbox_block_reason", return_value="Metal sandbox denial"),
                    patch.object(ui.subprocess, "Popen") as popen,
                ):
                    state._run(job_id, "one shot")

        completed = state.snapshot()["job"]
        self.assertEqual(completed["status"], "failed")
        self.assertEqual(completed["error"], "Metal sandbox denial")
        popen.assert_not_called()

    def test_clip3s_is_exposed_and_anchors_use_its_last_frame(self):
        self.assertIn("clip3s", ui.PROFILES)
        self.assertEqual(ui.PROFILE_FRAMES["clip3s"], 73)
        self.assertIn('name="profile" value="clip3s"', ui.PAGE)
        self.assertNotIn('id="generatedKeyframes"', ui.PAGE)
        self.assertIn("model-generated keyframe slots belong to the dev/HQ lane", ui.PAGE)

        start = {"image": None, "description": "", "strength": 1.0}
        end = {"image": ("png", b"image"), "description": "", "strength": 0.8}
        plan = ui._plan_conditionings("clip3s", start, end, [])
        self.assertEqual([(item["role"], item["frame"]) for item in plan], [("end", 72)])

    def test_scene_validation_preserves_order_and_optional_lengths(self):
        scenes = ui._validate_scenes(
            [
                {"text": " opening beat ", "length": 2},
                {"text": "middle beat", "length": None},
                {"text": "closing beat", "length": 2},
            ],
            "quality",
        )
        self.assertEqual(
            scenes,
            [
                {"text": "opening beat", "length": 2},
                {"text": "middle beat", "length": None},
                {"text": "closing beat", "length": 2},
            ],
        )

    def test_scene_validation_rejects_malformed_or_impossible_plans(self):
        invalid = [
            ("not-a-list", "scenes must be a list"),
            ([{"text": "", "length": None}], "text cannot be empty"),
            ([{"text": "one", "length": True}], "must be null or an integer"),
            ([{"text": "one", "length": 8}], "from 1 to 7"),
            ([{"text": "one", "length": 4}, {"text": "two", "length": 4}], "require 8 latent frames"),
            ([{"text": "one", "length": 7}, {"text": "two", "length": None}], "each scene needs at least one"),
            ([{"text": "one", "length": None, "extra": 1}], "Unknown scene 1 field"),
            ([{"text": str(index), "length": None} for index in range(8)], "supports at most 7 scenes"),
        ]
        for value, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ui.APIError, message):
                    ui._validate_scenes(value, "quality")

    def test_generated_keyframe_validation_is_exact(self):
        self.assertEqual(ui._validate_generated_keyframes(None), 0)
        self.assertEqual(ui._validate_generated_keyframes(0), 0)
        for value in (True, False, -1, 1, 2, 3, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ui.APIError, "must be 0 for the distilled Q4 pack"):
                    ui._validate_generated_keyframes(value)

    def test_duration_validation_accepts_only_integer_seconds_from_one_to_ten(self):
        for value in (1, 5, 10):
            with self.subTest(value=value):
                self.assertEqual(ui._validate_duration_seconds(value), value)

        for value in (None, True, False, 0, 11, 3.0, "3"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ui.APIError, "duration_seconds must be an integer from 1 to 10"):
                    ui._validate_duration_seconds(value)

    def test_aspect_ratio_ui_and_validation_cover_safe_workflows(self):
        for element_id in ("generateAspect", "a2vAspect", "ingredientsAspect", "unionAspect", "motionAspect"):
            self.assertIn(f'id="{element_id}"', ui.PAGE)
        for aspect in ("16:9", "9:16", "1:1", "4:3", "3:4"):
            self.assertIn(f'<option value="{aspect}">', ui.PAGE)
            self.assertEqual(ui._validate_aspect_ratio(aspect), aspect)
        self.assertEqual(ui._validate_aspect_ratio(None), "profile")
        self.assertIn("Extend always inherits the source video's aspect ratio and dimensions", ui.PAGE)
        self.assertIn("Trained landscape · Preferred", ui.PAGE)
        self.assertIn('value="9:16" data-experimental="ingredients-aspect"', ui.PAGE)
        for value in (True, 1, "21:9", "", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ui.APIError, "aspect_ratio must be"):
                    ui._validate_aspect_ratio(value)

    def test_experimental_options_open_local_coach_guardrail_without_starting_model(self):
        for marker in (
            'value="trained5s" data-experimental="ingredients-memory"',
            'value="9:16" data-experimental="ingredients-aspect"',
            'value="motion3s-experimental" data-experimental="motion-resolution"',
            "experimentalCoachWarningsSeenUi",
            "Coach · Experimental guardrail",
            "openPromptCoachUi({focusInput:false})",
            "if (event.isTrusted) warnExperimentalSelectionUi(event.currentTarget)",
            "Local safety guidance only—no model call or render was started.",
        ):
            self.assertIn(marker, ui.PAGE)
        warning_start = ui.PAGE.index("function warnExperimentalSelectionUi(select)")
        warning_end = ui.PAGE.index("function boundedPromptCoachHistoryUi", warning_start)
        warning_source = ui.PAGE[warning_start:warning_end]
        self.assertNotIn("askPromptCoachUi", warning_source)
        self.assertNotIn("/api/prompt-coach", warning_source)
        self.assertIn('$ui("motionDuration").value = "3"', warning_source)

    def test_prompt_coach_optional_questions_prepare_an_answer_without_sending(self):
        for marker in (
            "Ask Coach anything or continue the conversation",
            'id="promptCoachQuestionsLabel"',
            "Optional questions from Coach",
            "Choose one, then type your answer. Nothing is sent until you press Send.",
            'role="group" aria-labelledby="promptCoachQuestionsLabel"',
        ):
            self.assertIn(marker, ui.PAGE)

        suggestion_start = ui.PAGE.index("function renderPromptCoachSuggestionsUi(items)")
        suggestion_end = ui.PAGE.index("function renderPromptCoachResultUi(result)", suggestion_start)
        suggestion_source = ui.PAGE[suggestion_start:suggestion_end]
        self.assertIn("Answer to Coach’s question:\\n${question}\\n\\nMy answer: ", suggestion_source)
        self.assertIn("promptCoachMessageUi.focus()", suggestion_source)
        self.assertIn("promptCoachMessageUi.setSelectionRange(scaffold.length, scaffold.length)", suggestion_source)
        self.assertIn("label.hidden = !target.childElementCount", suggestion_source)
        self.assertNotIn("askPromptCoachUi", suggestion_source)
        self.assertNotIn("promptCoachSend", suggestion_source)
        self.assertIn(
            'if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); askPromptCoachUi(promptCoachMessageUi.value, promptCoachMessageUi.value, "chat"); }',
            ui.PAGE,
        )

    def test_prompt_coach_separates_deterministic_scoring_from_conversation(self):
        for marker in (
            'id="promptCoachLoad"',
            'id="promptCoachScoreButton"',
            '>Score prompt</button>',
            'id="promptCoachAnalyze"',
            '>Develop / review prompt</button>',
            'id="promptCoachMessage"',
            '>Ask Coach</button>',
            'aria-describedby="promptCoachActionHelp"',
            'id="promptCoachActionHelp"',
            "without adding a chat turn or rewriting your words",
            "Questions belong in the separate Ask Coach box below",
            'const scoreOnly = result.response_kind === "score_report"',
            'const conversationOnly = result.response_kind === "conversation"',
            'audit.hidden = conversationOnly',
            'audit.open = !guidanceOnly',
            'if (scoreOnly) audit.open = true',
            '"Prompt score & deterministic audit"',
            '$ui("promptCoachUse").hidden = scoreOnly',
            '$ui("promptCoachDraftBlock").hidden = scoreOnly',
            '$ui("promptCoachScoreButton").addEventListener("click", scorePromptCoachUi)',
            '$ui("promptCoachAnalyze").addEventListener("click", developPromptCoachUi)',
        ):
            self.assertIn(marker, ui.PAGE)

        score_start = ui.PAGE.index("async function scorePromptCoachUi()")
        score_end = ui.PAGE.index("function developPromptCoachUi()", score_start)
        score_source = ui.PAGE[score_start:score_end]
        self.assertIn('action:"score", message:"Score this prompt."', score_source)
        self.assertIn("history:[]", score_source)
        self.assertIn('scoreButton.textContent = "Scoring…"', score_source)
        self.assertIn('promptCoachPanelUi.setAttribute("aria-busy", "true")', score_source)
        self.assertIn('promptCoachPanelUi.removeAttribute("aria-busy")', score_source)
        self.assertIn("no chat turn, AI model, or media job was started", score_source)
        self.assertNotIn("appendPromptCoachChatUi", score_source)
        self.assertNotIn("promptCoachHistoryUi.push", score_source)
        self.assertNotIn("promptCoachLastPromptUi", score_source)

        develop_start = score_end
        develop_end = ui.PAGE.index("function openPromptCoachUi", develop_start)
        develop_source = ui.PAGE[develop_start:develop_end]
        self.assertIn("askPromptCoachUi", develop_source)
        self.assertIn("Develop this idea, or conservatively review it", develop_source)
        self.assertIn('"develop"', develop_source)
        self.assertNotIn("promptCoachInputIsStudioQuestionUi", ui.PAGE)

        send_binding = '$ui("promptCoachSend").addEventListener("click", () => askPromptCoachUi(promptCoachMessageUi.value, promptCoachMessageUi.value, "chat"));'
        self.assertIn(send_binding, ui.PAGE)
        ask_start = ui.PAGE.index("async function askPromptCoachUi(")
        ask_end = ui.PAGE.index("async function scorePromptCoachUi()", ask_start)
        ask_source = ui.PAGE[ask_start:ask_end]
        self.assertIn('action = "chat"', ask_source)
        self.assertIn("action !== \"chat\" && !raw.trim()", ask_source)
        self.assertIn("JSON.stringify({action, message:message.trim()", ask_source)

    def test_prompt_coach_client_signals_understand_locked_audio_performance(self):
        signals_start = ui.PAGE.index("function promptCoachSignalsUi(")
        signals_end = ui.PAGE.index("function promptCoachRewriteUi(", signals_start)
        signals_source = ui.PAGE[signals_start:signals_end]
        for marker in (
            "performs?|performing",
            "camera (?:remains?|stays?) (?:locked|static|stationary|fixed)",
            "|stage|",
            "supplied audio",
            "audio (?:owns|drives|determines) (?:the )?timing",
        ):
            self.assertIn(marker, signals_source)

    def test_motion_track_validation_normalizes_coordinates_and_rejects_bad_shapes(self):
        tracks = ui._validate_motion_tracks(
            [
                [{"x": 0, "y": 0.25}, {"x": 1, "y": 0.75}],
                [{"x": 0.2, "y": 0.3}, {"x": 0.4, "y": 0.5}, {"x": 0.6, "y": 0.7}],
            ]
        )
        self.assertEqual(tracks[0], [{"x": 0.0, "y": 0.25}, {"x": 1.0, "y": 0.75}])

        invalid = [
            [],
            [[{"x": 0, "y": 0}]],
            [[{"x": 0, "y": 0}, {"x": 1, "y": 1, "z": 0}]],
            [[{"x": True, "y": 0}, {"x": 1, "y": 1}]],
            [[{"x": "0.2", "y": 0}, {"x": 1, "y": 1}]],
            [[{"x": -0.01, "y": 0}, {"x": 1, "y": 1}]],
            [[{"x": 0, "y": 0}, {"x": 1.01, "y": 1}]],
            [[{"x": 0, "y": 0}, {"x": 1, "y": 1}]] * 9,
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ui.APIError):
                    ui._validate_motion_tracks(value)

    def test_ingredient_reference_tags_are_stable_normalized_and_unique(self):
        references = ui._validate_ingredient_references(
            [
                {
                    "image": PNG_1X1_DATA_URL,
                    "label": "Maya Chen",
                    "description": "lead actor in a red coat",
                },
                {
                    "image": PNG_1X1_DATA_URL,
                    "tag": "  HERO_2  ",
                    "label": "Second Hero",
                    "description": "supporting actor in blue",
                },
                {
                    "image": PNG_1X1_DATA_URL,
                    "label": "42 Blue Car",
                    "description": "compact electric vehicle",
                },
            ]
        )
        self.assertEqual(
            [(reference["tag"], reference["prompt_label"]) for reference in references],
            [
                ("@maya_chen", "@maya_chen — Maya Chen"),
                ("@hero_2", "@hero_2 — Second Hero"),
                ("@ref_42_blue_car", "@ref_42_blue_car — 42 Blue Car"),
            ],
        )

        duplicate = [
            {
                "image": PNG_1X1_DATA_URL,
                "tag": "@Maya",
                "label": "Maya front",
                "description": "front view",
            },
            {
                "image": PNG_1X1_DATA_URL,
                "tag": "maya",
                "label": "Maya profile",
                "description": "side view",
            },
        ]
        with self.assertRaisesRegex(ui.APIError, "Ingredient tag @maya must be unique"):
            ui._validate_ingredient_references(duplicate)

    def test_generator_command_keeps_anchors_and_adds_new_flags_as_single_argv_items(self):
        suspicious_scene = 'turn left; touch "/tmp/not-a-command"'
        job = {
            "profile": "clip3s",
            "seed": 99,
            "gemma_pack": "ddalcu-fallback",
            "output_path": Path("/tmp/result.mp4"),
            "conditionings": [
                {"role": "start", "path": Path("/tmp/start.png"), "strength": 1.0, "frame": 0},
                {"role": "anchor", "path": Path("/tmp/middle.png"), "strength": 0.65, "frame": 36},
                {"role": "end", "path": Path("/tmp/end.png"), "strength": 0.8, "frame": 72},
            ],
            "scenes": [
                {"text": "opening", "length": 2},
                {"text": suspicious_scene, "length": None},
            ],
            "generated_keyframes": 0,
        }

        command = ui._build_generator_command(job, "global prompt")

        self.assertEqual(command[:2], [str(ui.GENERATOR_PYTHON), str(ui.GENERATOR_SCRIPT)])
        self.assertIn("--start-image", command)
        self.assertIn("--anchor", command)
        self.assertIn("--end-image", command)
        self.assertEqual(
            command[command.index("--gemma-pack") : command.index("--gemma-pack") + 2],
            ["--gemma-pack", "ddalcu-fallback"],
        )
        self.assertEqual(
            command[command.index("--scene") : command.index("--")],
            [
                "--scene",
                "opening",
                "--scene-length",
                "2",
                "--scene",
                suspicious_scene,
            ],
        )
        self.assertNotIn("--generated-keyframes", command)
        self.assertEqual(command[-2:], ["--", "global prompt"])
        self.assertEqual(command.count(suspicious_scene), 1)

    def test_default_command_is_backward_compatible(self):
        job = {
            "profile": "quality",
            "seed": 42,
            "output_path": Path("/tmp/result.mp4"),
            "conditionings": [],
        }
        self.assertEqual(
            ui._build_generator_command(job, "global prompt"),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.GENERATOR_SCRIPT),
                "--profile",
                "quality",
                "--seed",
                "42",
                "--output",
                "/tmp/result.mp4",
                "--",
                "global prompt",
            ],
        )

    def test_fifty_thousand_character_prompt_reaches_job_command_exactly(self):
        prompt = "  OPENING " + ("precise beat · " * 3_850) + "UNIQUE-END-SENTINEL  "
        self.assertGreater(len(prompt), 50_000)
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting"}}

        with patch.object(ui.STATE, "start", return_value=response) as start:
            handler._handle_generate({"prompt": prompt, "profile": "quality", "seed": 42})

        validated_prompt = start.call_args.args[0]
        self.assertEqual(validated_prompt, prompt)
        job = {
            "profile": "quality",
            "seed": 42,
            "output_path": Path("/tmp/long-prompt.mp4"),
            "conditionings": [],
        }
        command = ui._build_generator_command(job, validated_prompt)
        self.assertEqual(command[-2:], ["--", prompt])
        self.assertTrue(command[-1].endswith("UNIQUE-END-SENTINEL  "))

    def test_prompt_editor_has_no_browser_cutoff_or_apply_slice(self):
        self.assertIn('<textarea id="prompt" required', ui.PAGE)
        self.assertNotIn('<textarea id="prompt" maxlength=', ui.PAGE)
        self.assertNotIn('<textarea id="promptCoachInput" maxlength=', ui.PAGE)
        self.assertNotIn('<textarea id="frameComposerPrompt" maxlength=', ui.PAGE)
        self.assertNotIn(".slice(0, 4000)", ui.PAGE)
        self.assertIn("characters · full text preserved", ui.PAGE)

    def test_absurd_prompt_is_rejected_without_truncation(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        oversized = "x" * (ui.MAX_PROMPT_CHARS + 1)
        with patch.object(ui.STATE, "start") as start:
            with self.assertRaisesRegex(ui.APIError, "rejected without truncation") as raised:
                handler._handle_generate({"prompt": oversized})
        self.assertEqual(raised.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        start.assert_not_called()

    def test_explicit_aspect_ratio_is_forwarded_to_supported_backends(self):
        generate_job = {
            "profile": "quality",
            "aspect_ratio": "9:16",
            "seed": 42,
            "output_path": Path("/tmp/vertical.mp4"),
            "conditionings": [],
        }
        generate_command = ui._build_generator_command(generate_job, "vertical shot")
        self.assertEqual(
            generate_command[generate_command.index("--aspect-ratio") : generate_command.index("--aspect-ratio") + 2],
            ["--aspect-ratio", "9:16"],
        )

        a2v_job = {
            "mode": "a2v",
            "profile": "clip3s",
            "aspect_ratio": "1:1",
            "seed": 43,
            "output_path": Path("/tmp/square.mp4"),
            "input_paths": {"audio": Path("/tmp/audio.wav")},
            "audio_start": 0,
            "duration_seconds": 3,
            "image_strength": 1,
        }
        a2v_command = ui._build_edit_command(a2v_job, "square performance")
        self.assertEqual(
            a2v_command[a2v_command.index("--aspect-ratio") : a2v_command.index("--aspect-ratio") + 2],
            ["--aspect-ratio", "1:1"],
        )

        motion_job = {
            "mode": "motion",
            "profile": "control3s",
            "aspect_ratio": "3:4",
            "seed": 44,
            "output_path": Path("/tmp/portrait.mp4"),
            "lora_strength": 1,
            "conditioning_strength": 1,
            "duration_seconds": 3,
            "conditionings": [],
            "input_paths": {"tracks_json": Path("/tmp/tracks.json")},
        }
        motion_command = ui._build_condition_command(motion_job, "portrait motion")
        self.assertEqual(
            motion_command[motion_command.index("--aspect-ratio") : motion_command.index("--aspect-ratio") + 2],
            ["--aspect-ratio", "3:4"],
        )

    def test_retake_command_forwards_interval_audio_policy_and_literal_prompt(self):
        prompt = 'repair this interval; ignore "--output /tmp/escape.mp4"'
        job = {
            "mode": "retake",
            "profile": "quality",
            "seed": 17,
            "output_path": Path("/tmp/retake.mp4"),
            "input_paths": {"source_video": Path("/tmp/source.mp4")},
            "start_seconds": 1.25,
            "end_seconds": 2.75,
            "preserve_audio": True,
        }
        self.assertEqual(
            ui._build_edit_command(job, prompt),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.EDIT_SCRIPT),
                "retake",
                "--output",
                "/tmp/retake.mp4",
                "--seed",
                "17",
                "--video",
                "/tmp/source.mp4",
                "--start-seconds",
                "1.25",
                "--end-seconds",
                "2.75",
                "--preserve-audio",
                "--",
                prompt,
            ],
        )

    def test_extend_command_forwards_duration_and_direction(self):
        job = {
            "mode": "extend",
            "profile": "quality",
            "seed": 18,
            "output_path": Path("/tmp/extend.mp4"),
            "input_paths": {"source_video": Path("/tmp/source.mp4")},
            "extend_seconds": 4,
            "direction": "before",
        }
        self.assertEqual(
            ui._build_edit_command(job, "lead into the original shot"),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.EDIT_SCRIPT),
                "extend",
                "--output",
                "/tmp/extend.mp4",
                "--seed",
                "18",
                "--video",
                "/tmp/source.mp4",
                "--extend-seconds",
                "4",
                "--direction",
                "before",
                "--",
                "lead into the original shot",
            ],
        )

    def test_a2v_command_maps_duration_to_ltx_frames_and_optional_start_image(self):
        job = {
            "mode": "a2v",
            "profile": "clip3s",
            "seed": 19,
            "output_path": Path("/tmp/a2v.mp4"),
            "input_paths": {
                "audio": Path("/tmp/performance.wav"),
                "a2v_start_image": Path("/tmp/opening.png"),
            },
            "audio_start": 0.5,
            "duration_seconds": 3,
            "image_strength": 0.85,
        }
        self.assertEqual(
            ui._build_edit_command(job, "movement follows the percussion"),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.EDIT_SCRIPT),
                "a2v",
                "--output",
                "/tmp/a2v.mp4",
                "--seed",
                "19",
                "--audio",
                "/tmp/performance.wav",
                "--audio-start",
                "0.5",
                "--profile",
                "clip3s",
                "--frames",
                "73",
                "--start-image",
                "/tmp/opening.png",
                "--image-strength",
                "0.85",
                "--",
                "movement follows the percussion",
            ],
        )

    def test_ingredients_command_forwards_references_and_timeline_anchor(self):
        job = {
            "mode": "ingredients",
            "profile": "ingredients5s",
            "seed": 20,
            "output_path": Path("/tmp/ingredients.mp4"),
            "lora_strength": 1.0,
            "conditioning_strength": 0.9,
            "conditionings": [
                {"role": "anchor", "path": Path("/tmp/timed.png"), "frame": 60, "strength": 0.7}
            ],
            "input_paths": {},
            "ingredient_references": [
                {
                    "path": Path("/tmp/pilot.png"),
                    "label": "pilot",
                    "description": "orange flight suit",
                },
                {
                    "path": Path("/tmp/ship.png"),
                    "label": "ship",
                    "description": "compact cobalt spacecraft",
                },
            ],
        }
        command = ui._build_condition_command(job, "the pilot enters the ship")
        self.assertEqual(
            command,
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.CONDITION_SCRIPT),
                "ingredients",
                "--profile",
                "ingredients5s",
                "--output",
                "/tmp/ingredients.mp4",
                "--seed",
                "20",
                "--lora-strength",
                "1.0",
                "--conditioning-strength",
                "0.9",
                "--anchor",
                "/tmp/timed.png",
                "60",
                "0.7",
                "--reference",
                "/tmp/pilot.png",
                "pilot",
                "orange flight suit",
                "--reference",
                "/tmp/ship.png",
                "ship",
                "compact cobalt spacecraft",
                "--",
                "the pilot enters the ship",
            ],
        )
        self.assertNotIn("--duration-seconds", command)

    def test_ingredients_command_emits_tagged_display_label_as_one_argv_item(self):
        job = {
            "mode": "ingredients",
            "profile": "ingredients5s",
            "seed": 23,
            "output_path": Path("/tmp/tagged-ingredients.mp4"),
            "lora_strength": 1.0,
            "conditioning_strength": 1.0,
            "conditionings": [],
            "input_paths": {},
            "ingredient_references": [
                {
                    "path": Path("/tmp/maya.png"),
                    "tag": "@maya",
                    "label": "Maya",
                    "prompt_label": "@maya — Maya",
                    "description": "lead actor in a red coat",
                }
            ],
        }

        command = ui._build_condition_command(job, "@maya walks into frame")
        reference_index = command.index("--reference")
        self.assertEqual(
            command[reference_index : reference_index + 4],
            ["--reference", "/tmp/maya.png", "@maya — Maya", "lead actor in a red coat"],
        )
        self.assertEqual(command[-2:], ["--", "@maya walks into frame"])

    def test_union_command_uses_raw_source_route_and_duration(self):
        job = {
            "mode": "union",
            "profile": "control3s-plus",
            "seed": 21,
            "output_path": Path("/tmp/union.mp4"),
            "lora_strength": 0.8,
            "conditioning_strength": 0.75,
            "duration_seconds": 6,
            "conditionings": [],
            "input_paths": {"control_video": Path("/tmp/source.mov")},
            "control_type": "canny",
            "union_route": "source",
        }
        self.assertEqual(
            ui._build_condition_command(job, "preserve the source structure"),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.CONDITION_SCRIPT),
                "union",
                "--profile",
                "control3s-plus",
                "--output",
                "/tmp/union.mp4",
                "--seed",
                "21",
                "--lora-strength",
                "0.8",
                "--conditioning-strength",
                "0.75",
                "--duration-seconds",
                "6",
                "--control-type",
                "canny",
                "--source-video",
                "/tmp/source.mov",
                "--",
                "preserve the source structure",
            ],
        )

    def test_motion_command_forwards_start_anchor_tracks_and_duration(self):
        job = {
            "mode": "motion",
            "profile": "control3s",
            "seed": 22,
            "output_path": Path("/tmp/motion.mp4"),
            "lora_strength": 1.0,
            "conditioning_strength": 1.0,
            "duration_seconds": 8,
            "conditionings": [
                {"role": "start", "path": Path("/tmp/start.png"), "frame": 0, "strength": 0.95}
            ],
            "input_paths": {"tracks_json": Path("/tmp/tracks.json")},
        }
        self.assertEqual(
            ui._build_condition_command(job, "the ball follows the path"),
            [
                str(ui.GENERATOR_PYTHON),
                str(ui.CONDITION_SCRIPT),
                "motion",
                "--profile",
                "control3s",
                "--output",
                "/tmp/motion.mp4",
                "--seed",
                "22",
                "--lora-strength",
                "1.0",
                "--conditioning-strength",
                "1.0",
                "--duration-seconds",
                "8",
                "--start-image",
                "/tmp/start.png",
                "--start-strength",
                "0.95",
                "--tracks-json",
                "/tmp/tracks.json",
                "--",
                "the ball follows the path",
            ],
        )

    def test_progress_parser_tracks_real_stages_and_denoise_values(self):
        job = {"stage": "queued", "progress_percent": None}

        ui._update_progress_from_line(job, "PRE-FLIGHT VERIFIED; render starting")
        self.assertEqual(job, {"stage": "preflight", "progress_percent": None})

        ui._update_progress_from_line(job, "[Loading transformer (transformer-dev.safetensors)] ...")
        self.assertEqual(job, {"stage": "loading", "progress_percent": None})

        ui._update_progress_from_line(job, "Denoising step 3/8")
        self.assertEqual(job, {"stage": "denoising", "progress_percent": 37.5})

        ui._update_progress_from_line(job, "Denoising (guided): 62%|######")
        self.assertEqual(job, {"stage": "denoising", "progress_percent": 62.0})

        ui._update_progress_from_line(job, "[VAE decode] ...")
        self.assertEqual(job, {"stage": "decoding", "progress_percent": None})

        ui._update_progress_from_line(job, "ffmpeg mux complete")
        self.assertEqual(job, {"stage": "encoding", "progress_percent": None})

        ui._update_progress_from_line(job, "Saving final video")
        self.assertEqual(job, {"stage": "finalizing", "progress_percent": None})

    def test_generate_api_forwards_features_without_merging_scene_text_into_global_prompt(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting"}}
        payload = {
            "prompt": "global continuity prompt",
            "profile": "clip3s",
            "seed": 7,
            "character_name": "Mara",
            "scenes": [
                {"text": "first local beat", "length": 3},
                {"text": "second local beat", "length": None},
            ],
            "generated_keyframes": 0,
        }

        with patch.object(ui.STATE, "start", return_value=response) as start:
            handler._handle_generate(payload)

        start.assert_called_once()
        positional = start.call_args.args
        keywords = start.call_args.kwargs
        self.assertIn("CHARACTER CONTINUITY:", positional[0])
        self.assertIn("Character name: Mara", positional[0])
        self.assertIn("SHOT DESCRIPTION:\nglobal continuity prompt", positional[0])
        self.assertEqual(positional[1:3], ("clip3s", 7))
        self.assertEqual(positional[3], [])
        self.assertNotIn("first local beat", positional[0])
        self.assertEqual(keywords["scenes"], payload["scenes"])
        self.assertEqual(keywords["generated_keyframes"], 0)
        self.assertEqual(keywords["gemma_pack"], "official")
        handler._send_json.assert_called_once_with(HTTPStatus.ACCEPTED, response)

    def test_generate_api_auto_selects_validated_gemma_fallback(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting"}}
        readiness = {
            "auto": {
                "ready": True,
                "selected": "ddalcu-fallback",
                "note": "validated fallback",
            },
            "official": {"ready": False, "note": "incomplete"},
            "ddalcu-fallback": {"ready": True, "note": "verified"},
        }
        with (
            patch.object(ui, "_gemma_pack_readiness", return_value=readiness),
            patch.object(ui.STATE, "start", return_value=response) as start,
        ):
            handler._handle_generate({"prompt": "one coherent shot", "gemma_pack": "auto"})

        self.assertEqual(start.call_args.kwargs["gemma_pack"], "ddalcu-fallback")
        handler._send_json.assert_called_once_with(HTTPStatus.ACCEPTED, response)

    def test_generate_modern_start_frame_is_primary_visual_identity_reference(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting"}}
        payload = {
            "prompt": "continue walking into the sunrise",
            "character_name": "Mara",
            "start_frame": {"image": PNG_1X1_DATA_URL, "description": "", "strength": 1.0},
        }

        with patch.object(ui.STATE, "start", return_value=response) as start:
            handler._handle_generate(payload)

        conditioned_prompt = start.call_args.args[0]
        self.assertIn("The uploaded reference image is the primary visual identity source", conditioned_prompt)
        self.assertEqual(start.call_args.args[3][0]["role"], "start")

    def test_generate_api_rejects_unknown_gemma_pack(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        with patch.object(ui.STATE, "start") as start:
            with self.assertRaisesRegex(ui.APIError, "gemma_pack must be auto") as raised:
                handler._handle_generate({"prompt": "one shot", "gemma_pack": "untrusted"})
        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        start.assert_not_called()

    def test_page_exposes_gemma_provenance_selector(self):
        self.assertIn('id="gemmaPack"', ui.PAGE)
        self.assertIn('value="ddalcu-fallback"', ui.PAGE)
        self.assertIn('payload.gemma_pack = $ui("gemmaPack").value', ui.PAGE)

    def test_single_job_guard_and_feature_metadata_are_preserved(self):
        state = ui.JobState()
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(ui, "GENERATED_DIR", Path(temporary)),
                patch.object(ui, "GENERATOR_SCRIPT", Path(__file__)),
                patch.object(ui, "GENERATOR_PYTHON", Path(sys.executable)),
                patch.object(ui.threading, "Thread") as thread,
            ):
                snapshot = state.start(
                    "prompt",
                    "clip3s",
                    8,
                    [],
                    scenes=[{"text": "local beat", "length": None}],
                    generated_keyframes=0,
                )
                self.assertEqual(snapshot["job"]["scene_count"], 1)
                self.assertEqual(snapshot["job"]["generated_keyframes"], 0)
                thread.return_value.start.assert_called_once_with()
                with self.assertRaisesRegex(ui.APIError, "already rendering"):
                    state.start("second", "draft", 9, [])

    def test_restore_latest_completed_selects_latest_safe_primary_render(self):
        state = ui.JobState()
        video_info = {
            "duration_seconds": 3.041667,
            "has_audio": True,
            "width": 448,
            "height": 768,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
        }
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            generated.mkdir()
            older = generated / "ltx25-quality-20260925-010000-11111111.mp4"
            newest = generated / "ltx25-max-20260925-020000-22222222.mp4"
            excluded_names = (
                "ltx25-720p-20260925-030000-33333333.mp4",
                "ltx25-fast-720p-20260925-040000-44444444.mp4",
                "ltx25-ai-quality-1080-20260925-050000-55555555.mp4",
                "ltx25-max-20260925-020000-22222222-voiced-66666666.mp4",
                ".ltx25-max-77777777.partial.mp4",
                "unmanaged.mp4",
            )
            paths = [older, newest, *(generated / name for name in excluded_names)]
            for index, path in enumerate(paths, start=1):
                path.write_bytes(b"video")
                os.utime(path, ns=(index * 1_000_000_000, index * 1_000_000_000))

            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "_probe_local_video", return_value=video_info) as probe,
            ):
                restored = state.restore_latest_completed()["job"]
                completed_id, completed_path = state.completed_output()

            self.assertEqual(restored["status"], "succeeded")
            self.assertTrue(restored["restored"])
            self.assertIsNone(restored["seed"])
            self.assertEqual(restored["mode"], "generate")
            self.assertEqual(restored["profile"], "max")
            self.assertEqual(restored["aspect_ratio"], "9:16")
            self.assertEqual(restored["output_url"], "/media/" + newest.name)
            self.assertEqual(completed_id, restored["id"])
            self.assertEqual(completed_path, newest.resolve())
            probe.assert_called_once_with(newest.resolve())

    def test_restore_latest_completed_skips_invalid_and_symlink_candidates(self):
        state = ui.JobState()
        video_info = {
            "duration_seconds": 3.0,
            "has_audio": False,
            "width": 768,
            "height": 448,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "audio_codec": "",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated"
            generated.mkdir()
            valid = generated / "ltx25-quality-20260925-010000-11111111.mp4"
            corrupt = generated / "ltx25-max-20260925-020000-22222222.mp4"
            outside = root / "outside.mp4"
            linked = generated / "ltx25-draft-20260925-030000-33333333.mp4"
            valid.write_bytes(b"valid")
            corrupt.write_bytes(b"corrupt")
            outside.write_bytes(b"outside")
            linked.symlink_to(outside)
            os.utime(valid, ns=(1_000_000_000, 1_000_000_000))
            os.utime(corrupt, ns=(2_000_000_000, 2_000_000_000))

            probed: list[str] = []

            def probe(path: Path, **_kwargs: object) -> dict[str, object]:
                probed.append(path.name)
                if path.name == corrupt.name:
                    raise RuntimeError("damaged")
                return video_info

            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "_probe_local_video", side_effect=probe),
            ):
                restored = state.restore_latest_completed()["job"]

            self.assertEqual(probed, [corrupt.name, valid.name])
            self.assertEqual(restored["output_url"], "/media/" + valid.name)
            self.assertNotIn(linked.name, probed)

    def test_restore_latest_completed_rejects_symlinked_output_root(self):
        state = ui.JobState()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            (actual / "ltx25-max-20260925-020000-22222222.mp4").write_bytes(b"video")
            linked_root = root / "generated"
            linked_root.symlink_to(actual, target_is_directory=True)
            with (
                patch.object(ui, "GENERATED_DIR", linked_root),
                patch.object(ui, "_probe_local_video") as probe,
            ):
                self.assertEqual(state.restore_latest_completed(), {"job": None})
            probe.assert_not_called()

    def test_restored_completed_video_can_start_ai_upscale(self):
        main_state = ui.JobState()
        upscale_state = ui.UpscaleState()
        video_info = {
            "duration_seconds": 3.041667,
            "has_audio": True,
            "width": 448,
            "height": 768,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
        }
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            generated.mkdir()
            source = generated / "ltx25-max-20260925-020000-22222222.mp4"
            source.write_bytes(b"video")
            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "STATE", main_state),
                patch.object(ui, "_probe_local_video", return_value=video_info),
                patch.object(ui, "_ai_upscale_readiness", return_value={"ready": True, "note": "ready"}),
                patch.object(ui.shutil, "which", side_effect=lambda name: f"/opt/homebrew/bin/{name}"),
                patch.object(ui.threading, "Thread") as thread,
            ):
                restored = main_state.restore_latest_completed()["job"]
                queued = upscale_state.start(profile="quality-ai", target_short_edge=1080)["job"]

            self.assertEqual(queued["status"], "starting")
            self.assertEqual(queued["source_job_id"], restored["id"])
            self.assertEqual((queued["source_width"], queued["source_height"]), (448, 768))
            self.assertEqual((queued["output_width"], queued["output_height"]), (1080, 1852))
            thread.return_value.start.assert_called_once_with()

    def test_edit_workflow_stages_private_upload_guards_and_promotes_output(self):
        state = ui.JobState()
        prompt = 'repair the gesture; ignore "--output /tmp/not-used.mp4"'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated"
            renders = root / "renders"
            spec = {
                "mode": "retake",
                "seed": 77,
                "profile": "retake",
                "conditionings": [],
                "uploads": {"source_video": (".mp4", b"private-source-bytes", "source.mp4")},
                "start_seconds": 0.5,
                "end_seconds": 1.25,
                "preserve_audio": True,
            }
            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "EDIT_OUTPUT_DIR", renders),
                patch.object(ui, "EDIT_SCRIPT", Path(__file__)),
                patch.object(ui, "GENERATOR_PYTHON", Path(sys.executable)),
                patch.object(ui.threading, "Thread") as thread,
            ):
                queued = state.start_workflow(prompt, spec)
                self.assertEqual(queued["job"]["status"], "starting")
                thread.return_value.start.assert_called_once_with()
                assert state._job is not None
                job = state._job
                job_id = job["id"]
                staged = job["input_paths"]["source_video"]
                output_path = job["output_path"]
                public_output_path = job["public_output_path"]

                self.assertEqual(staged.parent, generated.resolve())
                self.assertTrue(staged.name.startswith(f".ltx-ui-{job_id[:8]}-source_video-"))
                self.assertEqual(staged.suffix, ".mp4")
                self.assertEqual(staged.read_bytes(), b"private-source-bytes")
                self.assertEqual(staged.stat().st_mode & 0o777, 0o600)
                self.assertEqual(output_path.parent, renders.resolve())
                self.assertEqual(public_output_path.parent, generated.resolve())

                rejected_spec = {
                    "mode": "retake",
                    "seed": 78,
                    "profile": "retake",
                    "conditionings": [],
                    "uploads": {},
                }
                with self.assertRaisesRegex(ui.APIError, "already rendering") as raised:
                    state.start_workflow("second job", rejected_spec)
                self.assertEqual(raised.exception.status, HTTPStatus.CONFLICT)
                thread.return_value.start.assert_called_once_with()

                process = Mock()
                process.pid = 4321
                process.stdout = io.StringIO("PRE-FLIGHT VERIFIED\nDenoising step 8/8\n")

                def finish_render() -> int:
                    output_path.write_bytes(b"completed-edit-video")
                    return 0

                process.wait.side_effect = finish_render
                with (
                    patch.object(ui, "_metal_sandbox_block_reason", return_value=None),
                    patch.object(ui.subprocess, "Popen", return_value=process) as popen,
                ):
                    state._run(job_id, prompt)

                popen.assert_called_once()
                command = popen.call_args.args[0]
                options = popen.call_args.kwargs
                self.assertIsInstance(command, list)
                self.assertEqual(command[:3], [str(ui.GENERATOR_PYTHON), str(ui.EDIT_SCRIPT), "retake"])
                self.assertEqual(command[command.index("--video") + 1], str(staged))
                self.assertEqual(command[command.index("--output") + 1], str(output_path))
                self.assertEqual(command[-2:], ["--", prompt])
                self.assertNotIn("shell", options)
                self.assertTrue(options["start_new_session"])
                self.assertEqual(options["cwd"], ui.HERE)

                completed = state.snapshot()["job"]
                self.assertEqual(completed["status"], "succeeded")
                self.assertEqual(completed["stage"], "complete")
                self.assertEqual(completed["output_url"], "/media/" + public_output_path.name)
                self.assertFalse(staged.exists())
                self.assertFalse(output_path.exists())
                self.assertEqual(public_output_path.read_bytes(), b"completed-edit-video")

    def test_generate_api_rejects_combined_relay_prompt_before_starting_job(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        payload = {
            "prompt": "🎬" * 65_500,
            "profile": "quality",
            "character_name": "🎭" * 80,
            "character_appearance": "🎭" * 400,
            "character_wardrobe": "🎭" * 300,
            "character_continuity": "🎭" * 400,
            "character_avoid": "🎭" * 300,
            "scenes": [{"text": str(index) + "🎬" * 999, "length": 1} for index in range(7)],
        }
        with patch.object(ui.STATE, "start") as start:
            with self.assertRaisesRegex(ui.APIError, "conditioned prompt plus scene beats.*safety ceiling"):
                handler._handle_generate(payload)
        start.assert_not_called()

    def test_generate_api_rejects_dev_only_generated_keyframe_slots(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        with patch.object(ui.STATE, "start") as start:
            with self.assertRaisesRegex(ui.APIError, "distilled Q4 pack") as raised:
                handler._handle_generate({"prompt": "move", "generated_keyframes": 1})
        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        start.assert_not_called()

    def test_edit_api_is_blocked_before_upload_staging_without_dev_hq_lane(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        with (
            patch.object(ui, "_edit_mode_assets", return_value=(2, "dev/HQ lane is unavailable")),
            patch.object(ui.STATE, "start_workflow") as start,
        ):
            with self.assertRaisesRegex(ui.APIError, "dev/HQ lane is unavailable") as raised:
                handler._handle_render({"mode": "extend", "prompt": "continue"})
        self.assertEqual(raised.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        start.assert_not_called()

    def test_render_handler_rejects_unknown_fields_before_starting_workflow(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        payload = {
            "mode": "motion",
            "prompt": "move the subject",
            "extra_z": 1,
            "extra_a": 2,
        }

        with patch.object(ui.STATE, "start_workflow") as start:
            with self.assertRaisesRegex(ui.APIError, r"Unknown field\(s\): extra_a, extra_z") as raised:
                handler._handle_render(payload)

        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        start.assert_not_called()
        handler._send_json.assert_not_called()

    def test_render_handler_locks_ingredients_duration_and_requires_complete_references(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        invalid = [
            (
                {"mode": "ingredients", "prompt": "same pilot", "duration_seconds": 4},
                "Ingredients duration is locked",
            ),
            (
                {"mode": "ingredients", "prompt": "same pilot", "duration_seconds": 5},
                "ingredient_references must contain 1 to 8 references",
            ),
            (
                {
                    "mode": "ingredients",
                    "prompt": "same pilot",
                    "ingredient_references": [{"label": "pilot", "description": "orange suit"}],
                },
                "Ingredient reference 1 image is required",
            ),
            (
                {
                    "mode": "ingredients",
                    "prompt": "same pilot",
                    "ingredient_references": [
                        {"image": PNG_1X1_DATA_URL, "label": "pilot", "description": ""}
                    ],
                },
                "needs both a label and description",
            ),
        ]

        with patch.object(ui.STATE, "start_workflow") as start:
            for payload, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ui.APIError, message) as raised:
                        handler._handle_render(payload)
                    self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)

        start.assert_not_called()
        handler._send_json.assert_not_called()

    def test_render_handler_allows_only_canny_for_union_source_preprocessing(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        payload = {
            "mode": "union",
            "prompt": "follow this structure",
            "union_route": "source",
            "control_type": "depth",
        }

        with patch.object(ui.STATE, "start_workflow") as start:
            with self.assertRaisesRegex(ui.APIError, "source preprocessing supports Canny only") as raised:
                handler._handle_render(payload)

        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        start.assert_not_called()
        handler._send_json.assert_not_called()

    def test_render_handler_requires_motion_start_image_and_tracks(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        tracks = [[{"x": 0, "y": 0}, {"x": 1, "y": 1}]]
        invalid = [
            (
                {"mode": "motion", "prompt": "follow the curve", "tracks": tracks},
                "requires a start-frame image",
            ),
            (
                {
                    "mode": "motion",
                    "prompt": "follow the curve",
                    "start_frame": {"image": PNG_1X1_DATA_URL, "strength": 1.0},
                },
                "tracks must contain 1 to 8 motion tracks",
            ),
        ]

        with patch.object(ui.STATE, "start_workflow") as start:
            for payload, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ui.APIError, message) as raised:
                        handler._handle_render(payload)
                    self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)

        start.assert_not_called()
        handler._send_json.assert_not_called()

    def test_render_handler_starts_motion_when_q8_lane_is_ready(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting", "mode": "motion"}}
        payload = {
            "mode": "motion",
            "prompt": "move it along the curve",
            "profile": "control3s",
            "seed": 321,
            "duration_seconds": 4,
            "lora_strength": 1.25,
            "conditioning_strength": 0.6,
            "start_frame": {
                "image": PNG_1X1_DATA_URL,
                "description": "red ball at lower left",
                "strength": 0.8,
            },
            "tracks": [[{"x": 0, "y": 0.25}, {"x": 1, "y": 0.75}]],
        }

        with (
            patch.object(ui, "_control_mode_assets", return_value=(0, "ready")),
            patch.object(ui.STATE, "start_workflow", return_value=response) as start,
        ):
            handler._handle_render(payload)

        start.assert_called_once()
        prompt, spec = start.call_args.args
        self.assertIn("move it along the curve", prompt)
        self.assertEqual(spec["mode"], "motion")
        self.assertEqual(spec["duration_seconds"], 4)
        self.assertEqual(spec["tracks"][0][-1], {"x": 1.0, "y": 0.75})
        handler._send_json.assert_called_once_with(HTTPStatus.ACCEPTED, response)

    def test_render_handler_starts_ingredients_when_q8_lane_is_ready(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting", "mode": "ingredients"}}
        dialogue = '@maya: "We leave now."\n@leo: "Right behind you."'
        payload = {
            "mode": "ingredients",
            "prompt": "Maya and Leo face each other beside the ship",
            "seed": 404,
            "dialogue_guidance": dialogue,
            "ingredient_references": [
                {
                    "image": PNG_1X1_DATA_URL,
                    "tag": "MAYA",
                    "label": "Maya",
                    "description": "lead actor in a red coat",
                },
                {
                    "image": PNG_1X1_DATA_URL,
                    "tag": "@leo",
                    "label": "Leo",
                    "description": "supporting actor in a blue jacket",
                },
            ],
        }

        with (
            patch.object(ui, "_control_mode_assets", return_value=(0, "ready")),
            patch.object(ui.STATE, "start_workflow", return_value=response) as start,
        ):
            handler._handle_render(payload)

        start.assert_called_once()
        prompt, spec = start.call_args.args
        self.assertIn("@maya", prompt)
        self.assertIn("@leo", prompt)
        self.assertIn(dialogue, prompt)
        self.assertEqual(spec["mode"], "ingredients")
        self.assertEqual(len(spec["ingredient_references"]), 2)
        handler._send_json.assert_called_once_with(HTTPStatus.ACCEPTED, response)

    def test_workflow_readiness_has_stable_public_shape(self):
        with patch.object(
            ui,
            "_edit_mode_assets",
            return_value=(2, "The Q8 advanced edit lane needs two assets."),
        ), patch.object(
            ui,
            "_control_mode_assets",
            return_value=(3, "The Q8 advanced control lane needs three assets."),
        ):
            readiness = ui._workflow_readiness()
        self.assertEqual(set(readiness), ui.WORKFLOW_MODES)
        for mode, status in readiness.items():
            with self.subTest(mode=mode):
                self.assertIs(type(status["ready"]), bool)
                self.assertIs(type(status["experimental"]), bool)
                self.assertIs(type(status["blocked"]), bool)
                self.assertIsInstance(status["note"], str)
                self.assertTrue(status["note"])
                if mode in {"a2v", "ingredients", "union", "motion"}:
                    self.assertIs(type(status["missing_assets"]), int)
                    self.assertGreaterEqual(status["missing_assets"], 0)
        for mode in ui.EDIT_MODES:
            self.assertTrue(readiness[mode]["blocked"])
            self.assertIn("Q8", readiness[mode]["note"])
        for mode in ui.CONDITION_MODES:
            self.assertTrue(readiness[mode]["blocked"])
            self.assertIn("Q8", readiness[mode]["note"])

    def test_workflow_readiness_hard_blocks_shared_and_launcher_failures(self):
        gemma_missing = {"auto": {"ready": False, "selected": None, "note": "missing"}}
        with (
            patch.object(ui, "_cheap_phosphene_pack_issues", return_value=0),
            patch.object(ui, "_edit_mode_assets", return_value=(0, "Q8 edit assets ready.")),
            patch.object(ui, "_control_mode_assets", return_value=(0, "Q8 control assets ready.")),
        ):
            shared_failure = ui._workflow_readiness(gemma_missing)
        for mode in ui.WORKFLOW_MODES:
            with self.subTest(shared_failure=mode):
                self.assertFalse(shared_failure[mode]["ready"])
                self.assertTrue(shared_failure[mode]["blocked"])
                self.assertIn("base/Gemma", shared_failure[mode]["note"])

        gemma_ready = {"auto": {"ready": True, "selected": "official", "note": "ready"}}
        with (
            patch.object(ui, "_cheap_phosphene_pack_issues", return_value=0),
            patch.object(ui, "_edit_mode_assets", return_value=(0, "Q8 edit assets ready.")),
            patch.object(ui, "_control_mode_assets", return_value=(0, "Q8 control assets ready.")),
            patch.object(ui, "EDIT_SCRIPT", Path("/definitely/missing/edit_ltx25.py")),
        ):
            launcher_failure = ui._workflow_readiness(gemma_ready)
        for mode in ui.EDIT_MODES:
            with self.subTest(launcher_failure=mode):
                self.assertFalse(launcher_failure[mode]["ready"])
                self.assertTrue(launcher_failure[mode]["blocked"])
                self.assertIn("launcher", launcher_failure[mode]["note"])
        self.assertTrue(launcher_failure["generate"]["ready"])
        self.assertFalse(launcher_failure["generate"]["blocked"])
        for mode in ui.CONDITION_MODES:
            self.assertTrue(launcher_failure[mode]["ready"])
            self.assertFalse(launcher_failure[mode]["blocked"])


class CharacterProfileUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.characters = Path(self.temporary.name) / "work" / "local-video" / "characters"
        self.patch = patch.object(ui, "CHARACTERS_DIR", self.characters)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def payload(name: str = "Mara") -> dict[str, object]:
        return {
            "character_name": name,
            "character_appearance": "short black hair and amber eyes",
            "character_wardrobe": "red flight jacket",
            "character_continuity": "small scar above the left eyebrow",
            "character_avoid": "no hat or jewelry",
            "references": [{"name": "mara-front.png", "data": PNG_1X1_DATA_URL}],
        }

    def test_profiles_copy_references_and_survive_fresh_reads(self):
        saved = ui._save_character_profile(self.payload())
        self.assertRegex(saved["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(saved["character_name"], "Mara")
        self.assertEqual(len(saved["references"]), 1)
        self.assertTrue(saved["references"][0]["url"].startswith("/character-media/"))

        directory = self.characters / saved["id"]
        manifest = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        reference = directory / f"ref-{manifest['references'][0]['id']}.png"
        self.assertTrue(reference.is_file())
        self.assertEqual(reference.stat().st_mode & 0o777, 0o600)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

        loaded = ui._public_character_profile(ui._read_character_profile(saved["id"], include_data=True))
        self.assertEqual(loaded["character_appearance"], "short black hair and amber eyes")
        self.assertEqual(ui._list_character_profiles()[0]["id"], saved["id"])
        group = ui._resolve_character_reference_group(
            {"character_profile_id": saved["id"]},
            ui._validate_character_fields(saved),
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(len(group["images"]), 1)
        self.assertEqual(group["images"][0][0], "png")

    def test_duplicate_names_and_implicit_replacement_are_rejected(self):
        saved = ui._save_character_profile(self.payload())
        with self.assertRaisesRegex(ui.APIError, "already saved") as duplicate:
            ui._save_character_profile(self.payload())
        self.assertEqual(duplicate.exception.status, HTTPStatus.CONFLICT)

        replacement = self.payload()
        replacement["character_wardrobe"] = "navy coat"
        with self.assertRaisesRegex(ui.APIError, "explicit confirmation") as implicit:
            ui._save_character_profile(replacement, saved["id"])
        self.assertEqual(implicit.exception.status, HTTPStatus.CONFLICT)

        replacement["replace"] = True
        replaced = ui._save_character_profile(replacement, saved["id"])
        self.assertEqual(replaced["id"], saved["id"])
        self.assertEqual(replaced["character_wardrobe"], "navy coat")
        self.assertEqual(len(replaced["references"]), 1)

    def test_copy_references_delete_and_path_guards_are_exact(self):
        source = ui._save_character_profile(self.payload("Mara"))
        clone_payload = self.payload("Mara Stunt Double")
        clone_payload["references"] = []
        clone_payload["copy_references_from"] = source["id"]
        clone = ui._save_character_profile(clone_payload)
        self.assertEqual(len(clone["references"]), 1)
        self.assertNotEqual(clone["references"][0]["id"], source["references"][0]["id"])

        ui._delete_character_profile(source["id"])
        self.assertFalse((self.characters / source["id"]).exists())
        self.assertTrue((self.characters / clone["id"]).is_dir())
        with self.assertRaisesRegex(ui.APIError, "invalid"):
            ui._read_character_profile("../escape")

        unexpected = self.characters / clone["id"] / "do-not-delete.txt"
        unexpected.write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(ui.APIError, "unexpected files"):
            ui._delete_character_profile(clone["id"])
        self.assertTrue(unexpected.is_file())
        self.assertTrue((self.characters / clone["id"] / "profile.json").is_file())

    def test_symlinks_and_tampered_references_are_rejected(self):
        root = ui._ensure_character_store()
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        profile_id = "a" * 32
        (root / profile_id).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ui.APIError, "unsafe"):
            ui._read_character_profile(profile_id)
        (root / profile_id).unlink()

        saved = ui._save_character_profile(self.payload("Leo"))
        internal = ui._read_character_profile(saved["id"])
        internal["references"][0]["path"].write_bytes(b"tampered")
        with self.assertRaisesRegex(ui.APIError, "missing or corrupt|integrity"):
            ui._read_character_profile(saved["id"], include_data=True)

    def test_character_library_ui_is_local_explicit_and_auto_submitted(self):
        for element_id in (
            "characterProfileSelect",
            "loadCharacterProfile",
            "saveCharacterProfile",
            "replaceCharacterProfile",
            "deleteCharacterProfile",
            "characterReferenceImages",
            "characterReferencePreview",
        ):
            self.assertIn(f'id="{element_id}"', ui.PAGE)
        self.assertIn("work/local-video/characters", ui.PAGE)
        self.assertIn("payload.character_profile_id = activeCharacterProfileId", ui.PAGE)
        self.assertIn("Voice assignments remain separate", ui.PAGE)


class CapturedFrameUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.frames = Path(self.temporary.name) / "generated-frames"
        self.patch = patch.object(ui, "FRAME_GENERATED_DIR", self.frames)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.temporary.cleanup()

    def test_capture_persists_validated_private_png_without_touching_video(self):
        original = Path(self.temporary.name) / "completed.mp4"
        original.write_bytes(b"original-video-remains")
        snapshot = {
            "job": {
                "status": "succeeded",
                "output_url": "/media/completed.mp4",
                "aspect_ratio": "9:16",
            }
        }
        with patch.object(ui.STATE, "snapshot", return_value=snapshot):
            frame = ui._save_captured_video_frame(
                {
                    "image": PNG_1X1_DATA_URL,
                    "source_media": "/media/completed.mp4",
                    "timestamp_seconds": 1.375,
                }
            )

        self.assertRegex(frame["name"], r"^ltx25-capture-[0-9]{8}-[0-9]{6}-[0-9a-f]{12}\.png$")
        self.assertEqual(frame["url"], "/frames/" + frame["name"])
        self.assertEqual((frame["width"], frame["height"]), (1, 1))
        self.assertEqual(frame["suggested_aspect_ratio"], "9:16")
        saved = self.frames / frame["name"]
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertEqual(ui._image_dimensions("png", saved.read_bytes()), (1, 1))
        self.assertEqual(original.read_bytes(), b"original-video-remains")
        self.assertFalse(any(path.name.startswith(".") for path in self.frames.iterdir()))

    def test_capture_aspect_uses_explicit_source_then_nearest_safe_preset(self):
        self.assertEqual(ui._suggest_capture_aspect_ratio(384, 256, "9:16"), "9:16")
        self.assertEqual(ui._suggest_capture_aspect_ratio(1920, 1080, "profile"), "16:9")
        self.assertEqual(ui._suggest_capture_aspect_ratio(1080, 1920, None), "9:16")
        self.assertEqual(ui._suggest_capture_aspect_ratio(1000, 1000, "profile"), "1:1")
        self.assertEqual(ui._suggest_capture_aspect_ratio(1200, 900, "profile"), "4:3")
        self.assertEqual(ui._suggest_capture_aspect_ratio(900, 1200, "profile"), "3:4")

    def test_capture_rejects_path_escape_stale_source_and_unsafe_root(self):
        snapshot = {"job": {"status": "succeeded", "output_url": "/media/completed.mp4"}}
        with patch.object(ui.STATE, "snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ui.APIError, "Unknown frame-capture field"):
                ui._save_captured_video_frame(
                    {
                        "image": PNG_1X1_DATA_URL,
                        "source_media": "/media/completed.mp4",
                        "timestamp_seconds": 0,
                        "path": "/tmp/attacker-chosen.png",
                    }
                )
            with self.assertRaisesRegex(ui.APIError, "valid JPEG|Captured frame must be a PNG"):
                ui._save_captured_video_frame(
                    {
                        "image": PNG_1X1_DATA_URL.replace("image/png", "image/jpeg"),
                        "source_media": "/media/completed.mp4",
                        "timestamp_seconds": 0,
                    }
                )
            with self.assertRaisesRegex(ui.APIError, "source_media is invalid"):
                ui._save_captured_video_frame(
                    {
                        "image": PNG_1X1_DATA_URL,
                        "source_media": "/media/%2e%2e%2fevil.mp4",
                        "timestamp_seconds": 0,
                    }
                )
            with self.assertRaisesRegex(ui.APIError, "currently shown") as stale:
                ui._save_captured_video_frame(
                    {
                        "image": PNG_1X1_DATA_URL,
                        "source_media": "/media/other.mp4",
                        "timestamp_seconds": 0,
                    }
                )
            self.assertEqual(stale.exception.status, HTTPStatus.CONFLICT)

            outside = Path(self.temporary.name) / "outside-frames"
            outside.mkdir()
            self.frames.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ui.APIError, "unsafe"):
                ui._save_captured_video_frame(
                    {
                        "image": PNG_1X1_DATA_URL,
                        "source_media": "/media/completed.mp4",
                        "timestamp_seconds": 0,
                    }
                )

    def test_capture_ui_wires_exact_canvas_frame_and_separate_generate_handoff(self):
        for marker in (
            'id="captureCurrentFrame"',
            'id="capturePreview"',
            'id="useCapturedStart"',
            'id="downloadCapturedFrame"',
            'context.drawImage(videoUi, 0, 0, canvas.width, canvas.height)',
            'requestVideoFrameCallback',
            'await waitForVideoSeekUi(videoUi)',
            'displayedVideoTimestampUi(videoUi)',
            'apiUi("/api/frame-capture"',
            'composedFrameAssignments.start = latestCapturedFrameData',
            'selectModeUi("generate")',
            '$ui("startDescription").value = ""',
            '$ui("startStrength").value = "1"',
            'aspect.value = latestCapturedFrameAspectUi',
            "does not trim or stitch the rejected tail",
            "not hidden motion or audio state",
        ):
            self.assertIn(marker, ui.PAGE)


class UpscaleTests(unittest.TestCase):
    def test_720_target_dimensions_use_short_edge_and_even_rounding(self):
        cases = {
            (768, 448): (1234, 720),
            (448, 768): (720, 1234),
            (576, 576): (720, 720),
            (704, 512): (990, 720),
            (384, 256): (1080, 720),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                target = ui._upscale_target_dimensions(*source)
                self.assertEqual(target, expected)
                self.assertEqual(min(target), 720)
                self.assertEqual(target[0] % 2, 0)
                self.assertEqual(target[1] % 2, 0)
                self.assertLess(abs(target[0] / target[1] - source[0] / source[1]), 0.002)
        for source in ((1280, 720), (720, 1280), (900, 900)):
            with self.subTest(already_large=source):
                with self.assertRaisesRegex(ValueError, "already 720p or larger"):
                    ui._upscale_target_dimensions(*source)

    def test_1080_target_dimensions_use_short_edge_and_preserve_aspect(self):
        cases = {
            (768, 448): (1852, 1080),
            (448, 768): (1080, 1852),
            (576, 576): (1080, 1080),
            (704, 512): (1486, 1080),
            (384, 256): (1620, 1080),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                target = ui._upscale_target_dimensions(*source, ui.AI_UPSCALE_SHORT_EDGE)
                self.assertEqual(target, expected)
                self.assertEqual(min(target), 1080)
                self.assertEqual(target[0] % 2, 0)
                self.assertEqual(target[1] % 2, 0)
                self.assertLess(abs(target[0] / target[1] - source[0] / source[1]), 0.002)
        for source in ((1920, 1080), (1080, 1920), (1200, 1200)):
            with self.subTest(already_large=source):
                with self.assertRaisesRegex(ValueError, "already 1080p or larger"):
                    ui._upscale_target_dimensions(*source, ui.AI_UPSCALE_SHORT_EDGE)

    def test_upscale_command_is_shell_free_and_preserves_optional_audio(self):
        source = Path("/tmp/source.mp4")
        output = Path("/tmp/output.partial.mp4")
        command = ui._build_upscale_command("/usr/bin/ffmpeg", source, output, 1080, 720)
        self.assertIsInstance(command, list)
        self.assertEqual(command[:7], ["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-n", "-i"])
        self.assertEqual(command[7], str(source))
        self.assertIn("0:v:0", command)
        self.assertIn("0:a:0?", command)
        graph = command[command.index("-vf") + 1]
        self.assertIn("scale=1080:720:flags=lanczos", graph)
        self.assertIn("unsharp=5:5:0.35", graph)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertIn("+faststart", command)
        self.assertEqual(command[-1], str(output))

    def test_quality_upscale_command_is_shell_free_and_uses_pinned_runtime(self):
        source = Path("/tmp/source.mp4")
        output = Path("/tmp/output.partial.mp4")
        command = ui._build_quality_upscale_command(
            source,
            output,
            1080,
            "/opt/homebrew/bin/ffmpeg",
            "/opt/homebrew/bin/ffprobe",
        )

        self.assertIsInstance(command, list)
        self.assertEqual(command[0], str(ui.AI_UPSCALER_PYTHON))
        self.assertEqual(command[1], str(ui.AI_UPSCALER_SCRIPT))
        self.assertEqual(command[command.index("--input") + 1], str(source))
        self.assertEqual(command[command.index("--output") + 1], str(output))
        self.assertEqual(command[command.index("--target-short-edge") + 1], "1080")
        self.assertEqual(command[command.index("--ffmpeg") + 1], "/opt/homebrew/bin/ffmpeg")
        self.assertEqual(command[command.index("--ffprobe") + 1], "/opt/homebrew/bin/ffprobe")
        self.assertNotIn("sh", command[:2])
        self.assertNotIn("-c", command)

    def test_upscale_profiles_and_targets_are_strict(self):
        state = ui.UpscaleState()
        invalid = (
            ({"profile": "unknown"}, "profile must be fast720 or quality-ai"),
            ({"profile": 1}, "profile must be fast720 or quality-ai"),
            ({"profile": "fast720", "target_short_edge": 1080}, "requires target_short_edge=720"),
            ({"profile": "quality-ai", "target_short_edge": 720}, "requires target_short_edge=1080"),
            ({"profile": "quality-ai", "target_short_edge": True}, "requires target_short_edge=1080"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ui.APIError, message):
                    state.start(**kwargs)

        handler = object.__new__(ui.LTXRequestHandler)
        with self.assertRaisesRegex(ui.APIError, "Unknown field.*surprise"):
            handler._handle_upscale({"profile": "fast720", "surprise": True})

    def test_upscale_handler_forwards_quality_profile_and_target(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        response = {"job": {"status": "starting"}}
        with (
            patch.object(ui, "_prompt_coach_is_active", return_value=False),
            patch.object(ui.STATE, "is_busy", return_value=False),
            patch.object(ui.FRAME_STATE, "is_busy", return_value=False),
            patch.object(ui.VOICE_STATE, "is_busy", return_value=False),
            patch.object(ui.UPSCALE_STATE, "start", return_value=response) as start,
        ):
            handler._handle_upscale({"profile": "quality-ai", "target_short_edge": 1080})

        start.assert_called_once_with(profile="quality-ai", target_short_edge=1080)
        handler._send_json.assert_called_once_with(HTTPStatus.ACCEPTED, response)

    def test_quality_jsonl_events_update_real_progress_and_ignore_bad_values(self):
        state = ui.UpscaleState()
        state._job = {
            "id": "quality-job",
            "stage": "queued",
            "progress_percent": 0.0,
            "completed_frames": 0,
            "total_frames": None,
            "log": "",
        }

        consumed = state._consume_quality_event(
            "quality-job",
            json.dumps(
                {
                    "phase": "restoring_frames",
                    "completed": 37,
                    "total": 73,
                    "progress": 0.5068,
                    "message": "Restored frame 37 of 73",
                }
            ),
        )
        self.assertTrue(consumed)
        assert state._job is not None
        self.assertEqual(state._job["stage"], "restoring-frames")
        self.assertEqual(state._job["completed_frames"], 37)
        self.assertEqual(state._job["total_frames"], 73)
        self.assertEqual(state._job["progress_percent"], 50.68)
        self.assertIn("[ai-upscale] Restored frame 37 of 73", state._job["log"])

        self.assertTrue(
            state._consume_quality_event(
                "quality-job",
                '{"phase":"../../unsafe","completed":true,"total":0,"progress":150}',
            )
        )
        self.assertEqual(state._job["stage"], "restoring-frames")
        self.assertEqual(state._job["completed_frames"], 37)
        self.assertEqual(state._job["total_frames"], 73)
        self.assertEqual(state._job["progress_percent"], 99.0)
        self.assertFalse(state._consume_quality_event("quality-job", "not-json\n"))

    def test_upscale_state_atomically_publishes_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            generated.mkdir()
            source = generated / "ltx25-source.mp4"
            source.write_bytes(b"original-video")
            main_state = ui.JobState()
            main_state._job = {
                "id": "sourcejob123",
                "status": "succeeded",
                "output_path": source.resolve(),
                "public_output_path": source.resolve(),
            }
            state = ui.UpscaleState()
            probes = [
                {"duration_seconds": 5.0, "has_audio": True, "width": 384, "height": 256},
                {"duration_seconds": 5.0, "has_audio": True, "width": 1080, "height": 720},
            ]

            class Process:
                pid = 12345
                stdout = io.StringIO("")

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            def launch(command, **_kwargs):
                Path(command[-1]).write_bytes(b"upscaled-video")
                return Process()

            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "STATE", main_state),
                patch.object(ui, "_ai_upscale_readiness", return_value={"ready": True, "note": "ready"}),
                patch.object(ui.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
                patch.object(ui, "_probe_local_video", side_effect=probes),
                patch.object(ui.threading, "Thread") as thread,
                patch.object(ui.subprocess, "Popen", side_effect=launch),
            ):
                queued = state.start()
                self.assertEqual(queued["job"]["status"], "starting")
                with self.assertRaisesRegex(ui.APIError, "already running"):
                    state.start()
                thread.return_value.start.assert_called_once_with()
                assert state._job is not None
                state._run(state._job["id"])
                completed = state.snapshot()["job"]

            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["profile"], "fast720")
            self.assertEqual(completed["backend"], "FFmpeg Lanczos")
            self.assertEqual(completed["progress_percent"], 100.0)
            self.assertEqual((completed["output_width"], completed["output_height"]), (1080, 720))
            self.assertRegex(completed["output_url"], r"^/media/ltx25-fast-720p-.*\.mp4$")
            self.assertEqual(source.read_bytes(), b"original-video")
            published = [path for path in generated.iterdir() if path.name.startswith("ltx25-fast-720p-")]
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0].read_bytes(), b"upscaled-video")
            self.assertFalse(any(path.name.endswith(".partial.mp4") for path in generated.iterdir()))

    def test_ai_upscale_state_publishes_1080_copy_with_reported_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            generated.mkdir()
            source = generated / "ltx25-source.mp4"
            source.write_bytes(b"original-video")
            main_state = ui.JobState()
            main_state._job = {
                "id": "sourcejob-ai",
                "status": "succeeded",
                "output_path": source.resolve(),
                "public_output_path": source.resolve(),
            }
            state = ui.UpscaleState()
            probes = [
                {"duration_seconds": 5.0, "has_audio": True, "width": 384, "height": 256},
                {"duration_seconds": 5.0, "has_audio": True, "width": 1620, "height": 1080},
            ]

            class Process:
                pid = 54321
                stdout = io.StringIO(
                    '{"phase":"restoring","completed":73,"total":73,"progress":1.0,'
                    '"message":"All frames restored"}\n'
                )

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            def launch(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"ai-upscaled-video")
                return Process()

            readiness = {
                "ready": True,
                "backend": "Core ML Real-ESRGAN x4plus",
                "target_short_edge": 1080,
                "note": "ready",
            }
            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "STATE", main_state),
                patch.object(ui, "_ai_upscale_readiness", return_value=readiness),
                patch.object(ui.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
                patch.object(ui, "_probe_local_video", side_effect=probes),
                patch.object(ui.threading, "Thread") as thread,
                patch.object(ui.subprocess, "Popen", side_effect=launch),
            ):
                queued = state.start(profile="quality-ai", target_short_edge=1080)
                self.assertEqual(queued["job"]["status"], "starting")
                self.assertEqual(queued["job"]["profile"], "quality-ai")
                thread.return_value.start.assert_called_once_with()
                assert state._job is not None
                state._run(state._job["id"])
                completed = state.snapshot()["job"]

            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(completed["profile"], "quality-ai")
            self.assertEqual(completed["backend"], "Core ML Real-ESRGAN x4plus")
            self.assertEqual(completed["progress_percent"], 100.0)
            self.assertEqual((completed["completed_frames"], completed["total_frames"]), (73, 73))
            self.assertEqual((completed["output_width"], completed["output_height"]), (1620, 1080))
            self.assertRegex(completed["output_url"], r"^/media/ltx25-ai-quality-1080-.*\.mp4$")
            self.assertIn("All frames restored", completed["log_tail"])
            self.assertEqual(source.read_bytes(), b"original-video")
            published = [path for path in generated.iterdir() if path.name.startswith("ltx25-ai-quality-1080-")]
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0].read_bytes(), b"ai-upscaled-video")
            self.assertFalse(any(path.name.endswith(".partial.mp4") for path in generated.iterdir()))

    def test_upscale_failure_cleans_partial_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            generated.mkdir()
            source = generated / "ltx25-source.mp4"
            source.write_bytes(b"original-video")
            main_state = ui.JobState()
            main_state._job = {"id": "sourcejob", "status": "succeeded", "output_path": source.resolve()}
            state = ui.UpscaleState()

            class Process:
                pid = 222
                stdout = io.StringIO("encoder failed\n")

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            with (
                patch.object(ui, "GENERATED_DIR", generated),
                patch.object(ui, "STATE", main_state),
                patch.object(ui, "_ai_upscale_readiness", return_value={"ready": True, "note": "ready"}),
                patch.object(ui.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
                patch.object(ui, "_probe_local_video", return_value={"duration_seconds": 5.0, "has_audio": True, "width": 384, "height": 256}),
                patch.object(ui.threading, "Thread"),
                patch.object(ui.subprocess, "Popen", return_value=Process()),
            ):
                state.start()
                assert state._job is not None
                state._run(state._job["id"])
                failed = state.snapshot()["job"]

            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["output_url"])
            self.assertEqual(source.read_bytes(), b"original-video")
            self.assertEqual(list(generated.iterdir()), [source])

    def test_page_exposes_upscale_and_help_with_honest_copy(self):
        for marker in (
            'id="createUpscale"',
            'id="createAiUpscale"',
            'id="downloadUpscale"',
            'id="upscaleProgress"',
            'id="upscaleProgressBar"',
            'apiUi("/api/upscale"',
            'apiUi("/api/upscale/status"',
            "keeps the original",
            "AI Quality Upscale · 1080p",
            "Real‑ESRGAN",
            "Recommended",
            "does not reconstruct missing detail",
            "cannot repair anatomy, identity, motion or composition",
            "independent-frame enhancement can introduce shimmer",
            "Download MP4",
            "Download AI 1080p MP4",
            '{profile:"quality-ai", target_short_edge:1080}',
            "LTX‑2.5 help, prompt guide &amp; setup advisor",
            'id="recommendSetup"',
            'id="copyPromptTemplate"',
            'id="openHelp"',
            "not a “Gemma4 prompt.”",
            "recommendSetupUi",
            "Capture current frame",
            "Current availability on this Mac",
            "Dang Studio To DangDom",
            'id="promptCoachLauncher"',
            'id="promptCoachPanel"',
            'id="promptCoachAnalyze"',
            'id="promptCoachWalkthrough"',
            'id="promptCoachRewrite"',
            'id="promptCoachRewriteSafety"',
            'id="promptCoachUse"',
            "analysePromptCoachUi",
            "Apply it in Dang Studio",
            "The percentage measures prompt completeness—not the probability of a successful render.",
            "Coach · LTX‑2.5 Production Skill v1.4 loaded",
            "recommendation certainty",
            "If this is a rough or open-ended idea, develop it into one complete, coherent, concrete, production-ready shot",
            "promptCoachAssistantHistoryUi",
            "Current production-ready draft:",
            '{role:"assistant", content:promptCoachAssistantHistoryUi(result)}',
            "promptCoachInputUi.value = appliedPrompt",
            "Prompt Coach applied a complete draft",
            "finished draft readiness",
            "source idea ${result.score}%",
            'rewriteReview.strengths || result.strengths || []',
            'rewriteReview.improvements || result.improvements || []',
            'rewriteReview.risks || result.risks || []',
            "Finished draft · what is specified well",
            "Finished draft · remaining risks and improvements",
            "Best available workflow for the finished draft",
            "Finished-draft render risk",
            "Apply it in Dang Studio · finished draft",
            'id="currentAvailabilityList"',
            "displayCurrentAvailabilityUi",
            "Feature availability is evaluated separately.",
            "promptCoachInputUi.disabled = true",
            "loadButton.disabled = true",
            "useButton.disabled = true",
            "stale result was discarded",
            "restored from disk",
            "job.seed == null",
        ):
            self.assertIn(marker, ui.PAGE)
        self.assertNotIn("Replace any [bracketed] choices", ui.PAGE)
        self.assertNotIn(' · seed ${job.seed}${job.elapsed_seconds', ui.PAGE)
        help_start = ui.PAGE.index('<details id="helpPanel" class="help-panel"')
        help_end = ui.PAGE.index(">", help_start)
        self.assertIn(" open", ui.PAGE[help_start:help_end])

    def test_page_exposes_safe_interactive_learning_mode(self):
        for marker in (
            'id="openTutorial"',
            'aria-controls="tutorialPanel"',
            'aria-expanded="false"',
            'id="tutorialPanel"',
            'role="dialog"',
            'aria-labelledby="tutorialTitle"',
            'id="tutorialSafety"',
            'id="tutorialLessonGrid"',
            'id="tutorialProgress"',
            'id="tutorialStepLabel" aria-live="polite"',
            'id="tutorialShowControl"',
            'id="tutorialApplyPractice"',
            'id="tutorialRestoreSetup"',
            'id="tutorialAskCoach"',
            'id="tutorialBack"',
            'id="tutorialNext"',
            'id="closeTutorial"',
            'id="tutorialResume"',
            'type="button" aria-controls="tutorialPanel"',
            'dangStudio.tutorial.v1',
            'Learning mode is safe:',
            'it cannot start video, image, voice, capture, or upscale jobs',
            'Create my first clip',
            'Keep characters consistent',
            'Control motion or composition',
            'Continue or repair a clip',
            'Add speech or soundtrack',
            'Finish and upscale a video',
            'Installed eligibility is not a render guarantee',
            'full receipt, source and Metal preflight runs on submit',
            'Learning mode never starts or changes media jobs',
            'Finish learning mode before rendering',
            'Tutorial question prepared. Review it, then press Send when you choose.',
            'Not quite. Try again.',
            'if (!isCorrect)',
        ):
            self.assertIn(marker, ui.PAGE)

        ids = re.findall(r'id="([^"]+)"', ui.PAGE)
        self.assertEqual(len(ids), len(set(ids)), "Every static HTML id must be unique.")
        targets = re.findall(r'targetId:"([^"]+)"', ui.PAGE)
        self.assertGreaterEqual(len(set(targets)), 12)
        for target in targets:
            self.assertEqual(ids.count(target), 1, f"Tutorial target {target!r} must exist exactly once.")

        api_start = ui.PAGE.index("function apiUi(path, options = {})")
        api_end = ui.PAGE.index("const tutorialModeLabelsUi", api_start)
        api_source = ui.PAGE[api_start:api_end]
        self.assertIn('method !== "GET"', api_source)
        self.assertIn('path === "/api/prompt-coach"', api_source)
        self.assertIn('path === "/api/cancel"', api_source)
        self.assertNotIn("requestSubmit", ui.PAGE)

    def test_audio_tutorial_uses_prompt_tags_and_clears_only_stale_coach_context(self):
        self.assertNotIn("Keep each @speaker separate", ui.PAGE)
        question_start = ui.PAGE.index("function tutorialCoachQuestionUi(")
        question_end = ui.PAGE.index("function tutorialAskCoachUi(", question_start)
        question_source = ui.PAGE[question_start:question_end]
        for marker in (
            'lesson?.id !== "audio-dialogue"',
            'match(/@[A-Za-z][A-Za-z0-9_]*/g)',
            "No stable character tag is present",
            "do not invent a tag or speaker identity",
            "The existing stable ${tags.length === 1 ? \"tag is\" : \"tags are\"}",
            "identify which tagged character, if any, actually speaks",
        ):
            self.assertIn(marker, question_source)

        reset_start = ui.PAGE.index("function resetPromptCoachHistoryUi(")
        reset_end = ui.PAGE.index("function renderPromptCoachSuggestionsUi(", reset_start)
        reset_source = ui.PAGE[reset_start:reset_end]
        self.assertIn("promptCoachConversationUi.replaceChildren()", reset_source)
        self.assertIn('promptCoachLastPromptUi = ""', reset_source)
        self.assertIn('promptCoachTutorialContextUi = ""', reset_source)

        ask_start = ui.PAGE.index("function tutorialAskCoachUi(")
        ask_end = ui.PAGE.index("function showSetupAdviceUi(", ask_start)
        ask_source = ui.PAGE[ask_start:ask_end]
        for marker in (
            "tutorialCoachQuestionUi(lesson, coachPrompt)",
            "promptCoachTutorialContextUi !== tutorialContext",
            "hasPriorAnalysis",
            "promptCoachTutorialContextUi = tutorialContext",
            "promptCoachLastPromptUi = coachPrompt",
            "promptCoachMessageUi.value = coachQuestion",
        ):
            self.assertIn(marker, ask_source)
        self.assertNotIn("promptCoachMessageUi.value = lesson.coachQuestion", ask_source)

    def test_workflow_guidance_keeps_chat_primary_and_folds_the_prompt_audit(self):
        for marker in (
            'id="promptCoachAudit"',
            'id="promptCoachAuditSummary"',
            'const guidanceOnly = result.response_kind === "workflow_guidance"',
            'audit.open = !guidanceOnly',
            '"Optional prompt audit & production draft"',
            'promptCoachConversationUi.scrollTop = promptCoachConversationUi.scrollHeight',
        ):
            self.assertIn(marker, ui.PAGE)

        render_start = ui.PAGE.index("function renderPromptCoachResultUi(")
        render_end = ui.PAGE.index("async function askPromptCoachUi(", render_start)
        render_source = ui.PAGE[render_start:render_end]
        self.assertIn("if (guidanceOnly)", render_source)
        self.assertIn("promptCoachResultUi.scrollIntoView", render_source)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_audio_tutorial_question_materializes_exact_existing_tags(self):
        source_start = ui.PAGE.index("function tutorialCoachQuestionUi(")
        source_end = ui.PAGE.index("function tutorialAskCoachUi(", source_start)
        function_source = ui.PAGE[source_start:source_end]
        probe = function_source + r'''
const lesson = {
  id: "audio-dialogue",
  coachQuestion: "Teach me which audio workflow fits."
};
const result = {
  one: tutorialCoachQuestionUi(lesson, "A close portrait of @maya performing to music."),
  many: tutorialCoachQuestionUi(lesson, "@subject_a reacts while @subject_b speaks; preserve @subject_a."),
  none: tutorialCoachQuestionUi(lesson, "An unnamed singer performs on stage."),
  other: tutorialCoachQuestionUi({id:"first-clip", coachQuestion:"Explain this."}, "@maya"),
};
process.stdout.write(JSON.stringify(result));
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tutorial-question.js"
            path.write_text(probe, encoding="utf-8")
            completed = subprocess.run(
                [shutil.which("node") or "node", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        result = json.loads(completed.stdout)
        self.assertIn("tag is @maya", result["one"])
        self.assertNotIn("@speaker", result["one"])
        self.assertIn("tags are @subject_a and @subject_b", result["many"])
        self.assertEqual(result["many"].count("@subject_a"), 1)
        self.assertIn("No stable character tag is present", result["none"])
        self.assertNotIn("@", result["none"])
        self.assertEqual(result["other"], "Explain this.")

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_inline_ui_javascript_is_syntax_valid(self):
        script = ui.PAGE.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ltx-ui-inline.js"
            path.write_text(script, encoding="utf-8")
            completed = subprocess.run(
                [shutil.which("node") or "node", "--check", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
    def test_real_ffmpeg_upscale_landscape_and_portrait_preserves_audio(self):
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for width, height, expected in ((96, 64, (1080, 720)), (64, 96, (720, 1080))):
                with self.subTest(source=(width, height)):
                    source = root / f"source-{width}x{height}.mp4"
                    output = root / f"output-{width}x{height}.mp4"
                    subprocess.run(
                        [
                            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:r=24:d=0.25",
                            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.25",
                            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
                        ],
                        check=True,
                    )
                    target = ui._upscale_target_dimensions(width, height)
                    subprocess.run(ui._build_upscale_command(ffmpeg, source, output, *target), check=True)
                    source_info = ui._probe_local_video(source, require_audio=True)
                    output_info = ui._probe_local_video(output, require_audio=True)
                    self.assertEqual((output_info["width"], output_info["height"]), expected)
                    self.assertTrue(output_info["has_audio"])
                    self.assertLess(abs(output_info["duration_seconds"] - source_info["duration_seconds"]), 0.2)


class CharacterVoiceUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.voice_root = self.root / "managed-voices"
        self.reference_root = self.voice_root / "references"
        self.registry = self.voice_root / "registry.json"
        self.voice_outputs = self.root / "generated" / "voices"
        for target, value in (
            ("VOICE_ROOT", self.voice_root),
            ("REFERENCE_ROOT", self.reference_root),
            ("REGISTRY_PATH", self.registry),
            ("OUTPUT_ROOT", self.voice_outputs),
        ):
            patcher = patch.object(voice, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reference = self.root / "authorized.wav"
        write_voice_wave(self.reference)

    def register_and_assign(self, character: str = "aria", preset: str = "aria-voice") -> None:
        voice.register_preset(
            preset,
            self.reference,
            "en",
            consent_confirmed=True,
            voice_description="warm and calm",
        )
        voice.assign_character(character, preset)

    def test_page_exposes_honest_reference_anchored_voice_controls(self):
        voice_summary = "<summary><span>Character Voices"
        voice_summary_start = ui.PAGE.index(voice_summary)
        voice_details_start = ui.PAGE.rfind("<details", 0, voice_summary_start)
        voice_details_end = ui.PAGE.index(">", voice_details_start)
        self.assertNotIn(" open", ui.PAGE[voice_details_start:voice_details_end])
        self.assertIn("Character Voices", ui.PAGE)
        self.assertIn('id="voiceConsent"', ui.PAGE)
        self.assertIn('id="voiceCueList"', ui.PAGE)
        self.assertIn('id="voiceMuxLatest"', ui.PAGE)
        self.assertIn("each row is one separate TTS call", ui.PAGE)
        self.assertIn("cannot invent an exact persistent speaker identity", ui.PAGE)
        self.assertIn("23 languages", ui.PAGE)

    def test_ui_registration_requires_consent_and_copies_authorized_wave(self):
        encoded = base64.b64encode(self.reference.read_bytes()).decode("ascii")
        payload = {
            "name": "aria-voice",
            "reference": "data:audio/wav;base64," + encoded,
            "reference_language": "en",
            "voice_description": "calm and measured",
            "consent_confirmed": False,
            "force": False,
        }
        with patch.object(ui, "_character_voice_module", return_value=voice):
            with self.assertRaisesRegex(ui.APIError, "Explicit consent"):
                ui._register_voice_preset(payload)
            self.assertFalse(self.registry.exists())
            payload["consent_confirmed"] = True
            result = ui._register_voice_preset(payload)
            assignment = ui._assign_voice_preset(
                {"character": "@aria", "preset": "aria-voice", "force": False}
            )

        self.assertEqual(result["preset"]["name"], "aria-voice")
        self.assertTrue(result["preset"]["consent_confirmed"])
        self.assertEqual(assignment["assignment"], {"character": "aria", "preset": "aria-voice"})
        registry = voice.load_registry()
        canonical = self.voice_root / registry["presets"]["aria-voice"]["reference"]
        self.assertTrue(canonical.is_file())
        self.assertNotEqual(canonical, self.reference)

    def test_voice_cues_require_explicit_assigned_speakers_and_remain_separate(self):
        self.register_and_assign("aria", "aria-voice")
        voice.register_preset("dev-voice", self.reference, "en", consent_confirmed=True)
        voice.assign_character("dev", "dev-voice")
        payload = [
            {
                "character": "@aria",
                "text": "We leave now.",
                "language": "en",
                "start_seconds": 0.25,
                "voice_description": "calm but urgent",
                "seed": 10,
            },
            {
                "character": "dev",
                "text": "I am right behind you.",
                "language": "hi",
                "start_seconds": 1.5,
                "voice_description": "dramatic",
                "seed": 11,
            },
        ]
        with patch.object(ui, "_character_voice_module", return_value=voice):
            cues = ui._validate_voice_cues(payload)
            with self.assertRaisesRegex(ui.APIError, "has no assigned voice preset"):
                ui._validate_voice_cues(
                    [{**payload[0], "character": "@unassigned"}]
                )
        self.assertEqual([(item["character"], item["preset"], item["language"]) for item in cues], [
            ("aria", "aria-voice", "en"),
            ("dev", "dev-voice", "hi"),
        ])
        self.assertEqual([item["text"] for item in cues], ["We leave now.", "I am right behind you."])

    def test_voice_state_invokes_one_argv_safe_process_per_speaker_cue(self):
        generated = self.root / "generated"
        voice_generated = generated / "voices"
        cues = [
            {"character": "aria", "preset": "aria-voice", "text": "Aria only", "language": "en", "start_seconds": 0.0, "voice_description": "calm", "seed": 1},
            {"character": "dev", "preset": "dev-voice", "text": "Dev only", "language": "hi", "start_seconds": 1.0, "voice_description": "dramatic", "seed": 2},
        ]
        state = ui.CharacterVoiceState()
        commands: list[list[str]] = []

        def fake_process(_job_id, command, *, json_result):
            self.assertTrue(json_result)
            commands.append(list(command))
            return {"schema_version": 1}

        def fake_validation(_payload, *, cue, expected_audio):
            return {
                "character": cue["character"],
                "preset": cue["preset"],
                "language": cue["language"],
                "start_seconds": cue["start_seconds"],
                "seed": cue["seed"],
                "audio_path": expected_audio,
                "manifest_path": expected_audio.with_suffix(".voice.json"),
                "audio_sha256": "0" * 64,
                "duration_seconds": 0.5,
                "watermarked": True,
            }

        ready = {"ready": True, "registry_ready": True, "mux_ready": True, "note": "ready", "languages": {}}
        with (
            patch.object(ui, "GENERATED_DIR", generated),
            patch.object(ui, "VOICE_GENERATED_DIR", voice_generated),
            patch.object(ui, "_character_voice_readiness", return_value=ready),
            patch.object(ui, "_public_voice_registry", return_value={"presets": [], "characters": {}}),
            patch.object(ui.threading, "Thread") as thread,
            patch.object(state, "_run_process", side_effect=fake_process),
            patch.object(ui, "_validate_voice_result", side_effect=fake_validation),
        ):
            queued = state.start(cues, mux_latest=False, source_video=None, source_duration=None)
            self.assertEqual(queued["job"]["cue_count"], 2)
            thread.return_value.start.assert_called_once_with()
            assert state._job is not None
            state._run(state._job["id"])
            completed = state.snapshot()["job"]

        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][2:4], ["synthesize", "Aria only"])
        self.assertEqual(commands[1][2:4], ["synthesize", "Dev only"])
        self.assertIn("aria", commands[0])
        self.assertNotIn("dev", commands[0])
        self.assertIn("dev", commands[1])
        self.assertNotIn("Aria only", commands[1])
        self.assertNotIn("shell", commands[0])
        self.assertEqual([(item["character"], item["language"]) for item in completed["cues"]], [("aria", "en"), ("dev", "hi")])

    def test_dialogue_mux_command_delays_each_cue_and_replaces_native_audio(self):
        source = Path("/tmp/source.mp4")
        output = Path("/tmp/new.mp4")
        rendered = [
            {"audio_path": Path("/tmp/aria.wav"), "start_seconds": 0.25},
            {"audio_path": Path("/tmp/dev.wav"), "start_seconds": 1.5},
        ]
        command = ui._build_dialogue_mux_command("/usr/bin/ffmpeg", source, rendered, output)
        self.assertEqual(command[:7], ["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-n", "-i"])
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("[1:a:0]adelay=250:all=1[cue1]", graph)
        self.assertIn("[2:a:0]adelay=1500:all=1[cue2]", graph)
        self.assertIn("amix=inputs=2:duration=longest", graph)
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")
        self.assertNotIn("0:a", command)
        self.assertEqual(command[-1], str(output))

    def test_voice_cancel_uses_sigint_before_forced_kill_for_wrapper_cleanup(self):
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        with patch.object(ui.os, "killpg") as killpg:
            ui.CharacterVoiceState._interrupt(process)
            ui.CharacterVoiceState._interrupt(process, force=True)
        self.assertEqual(
            killpg.call_args_list,
            [unittest.mock.call(4321, ui.signal.SIGINT), unittest.mock.call(4321, ui.signal.SIGKILL)],
        )

    def test_ui_revalidates_voice_manifest_and_audio_hash_before_exposing_cue(self):
        self.voice_outputs.mkdir(parents=True)
        audio = self.voice_outputs / "voice-test.wav"
        write_voice_wave(audio, seconds=0.25)
        digest = ui._sha256_file(audio)
        manifest = audio.with_suffix(".voice.json")
        cue = {
            "character": "aria",
            "preset": "aria-voice",
            "reference_sha256": "a" * 64,
            "language": "en",
            "text": "We leave now.",
            "start_seconds": 0.5,
            "seed": 9,
        }
        payload = {
            "schema_version": 1,
            "character": "aria",
            "preset": "aria-voice",
            "reference_sha256": "a" * 64,
            "language": "en",
            "offline": True,
            "audio_path": str(audio.resolve()),
            "manifest_path": str(manifest.resolve()),
            "audio_sha256": digest,
            "text_sha256": ui.hashlib.sha256(cue["text"].encode("utf-8")).hexdigest(),
            "sample_rate": 24_000,
            "channels": 1,
            "duration_seconds": 0.25,
            "watermarked": True,
        }
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with patch.object(ui, "VOICE_GENERATED_DIR", self.voice_outputs):
            result = ui._validate_voice_result(payload, cue=cue, expected_audio=audio)
            self.assertEqual(result["audio_sha256"], digest)
            manifest.write_text(json.dumps({**payload, "character": "dev"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not authenticate"):
                ui._validate_voice_result(payload, cue=cue, expected_audio=audio)

    def test_mux_atomically_publishes_new_video_and_preserves_original(self):
        generated = self.root / "generated"
        voice_generated = generated / "voices"
        voice_generated.mkdir(parents=True)
        source = generated / "ltx25-test.mp4"
        source.write_bytes(b"original-video")
        cue_path = voice_generated / "voice.wav"
        cue_path.write_bytes(b"RIFF0000WAVEaudio")
        rendered = [{"character": "aria", "audio_path": cue_path, "start_seconds": 0.25}]
        state = ui.CharacterVoiceState()
        state._job = {"id": "abc12345", "cancel_requested": False, "process": None, "log": ""}
        commands: list[list[str]] = []

        def fake_run(_job_id, command, *, json_result):
            self.assertFalse(json_result)
            commands.append(list(command))
            Path(command[-1]).write_bytes(b"muxed-video")
            return ""

        with (
            patch.object(ui, "GENERATED_DIR", generated),
            patch.object(ui, "VOICE_GENERATED_DIR", voice_generated),
            patch.object(ui.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
            patch.object(ui, "_probe_local_video", return_value={"duration_seconds": 3.0, "has_audio": True}),
            patch.object(state, "_run_process", side_effect=fake_run),
        ):
            published = state._mux("abc12345", source.resolve(), 3.0, rendered)

        self.assertEqual(source.read_bytes(), b"original-video")
        self.assertEqual(published.read_bytes(), b"muxed-video")
        self.assertNotEqual(published, source)
        self.assertIn("-filter_complex", commands[0])
        self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in generated.iterdir()))

    def test_uninstalled_voice_backend_is_reported_without_blocking_registry_ui(self):
        status = {
            "ready": False,
            "runtime_ready": False,
            "model_ready": False,
            "receipt_ready": False,
            "runtime_note": "isolated voice runtime is missing",
            "model_note": "model directory is missing: /Users/example/private/model",
            "receipt_note": "installation receipt is missing",
        }
        with (
            patch.object(ui, "_character_voice_module", return_value=voice),
            patch.object(voice.installer, "readiness", return_value=status),
            patch.object(ui.shutil, "which", return_value=None),
        ):
            ui._VOICE_READINESS_CACHE.update({"checked_at": 0.0, "value": None})
            readiness = ui._character_voice_readiness(refresh=True)
        self.assertFalse(readiness["ready"])
        self.assertTrue(readiness["registry_ready"])
        self.assertFalse(readiness["mux_ready"])
        self.assertIn("install_character_voice.py", readiness["note"])
        self.assertNotIn("/Users/", readiness["note"])


if __name__ == "__main__":
    unittest.main()
