from __future__ import annotations

import json
import os
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.request import ProxyHandler

import ltx_ui as ui
import prompt_coach as coach
import prompt_coach_skill as coach_skill


def readiness(*, ingredients: bool = True) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for mode in coach.WORKFLOW_MODES:
        ready = ingredients if mode == "ingredients" else True
        result[mode] = {
            "ready": ready,
            "blocked": not ready,
            "note": "verified and ready" if ready else "verified Q8 assets are missing",
        }
    return result


def request_payload(prompt: str = "@maya walks through a neon alley") -> dict[str, object]:
    return {
        "message": "Review this prompt and give me exact settings.",
        "prompt": prompt,
        "history": [],
        "context": {
            "mode": "generate",
            "profile": "quality",
            "aspect_ratio": "16:9",
            "duration_seconds": 3,
            "seed": 42,
            "references": {"start_frame": True, "ingredient_references": 0},
        },
    }


class PromptCoachTests(unittest.TestCase):
    def test_score_action_is_explicit_and_defaults_to_interaction(self):
        interactive = coach.validate_request(request_payload())
        self.assertEqual(interactive["action"], "interact")

        scored_payload = request_payload()
        scored_payload["action"] = "score"
        scored = coach.validate_request(scored_payload)
        self.assertEqual(scored["action"], "score")

        for action in ("develop", "chat"):
            explicit_payload = request_payload()
            explicit_payload["action"] = action
            self.assertEqual(coach.validate_request(explicit_payload)["action"], action)

        invalid_payload = request_payload()
        invalid_payload["action"] = "rewrite"
        with self.assertRaisesRegex(coach.CoachInputError, "interact or score"):
            coach.validate_request(invalid_payload)

    def test_explicit_chat_keeps_arbitrary_answer_and_never_rewrites_prompt(self):
        source = (
            "One continuous 3-second shot. @maya crosses a softly lit stage while the "
            "camera stays locked; preserve her identity and avoid cuts."
        )
        payload = request_payload(source)
        payload["action"] = "chat"
        payload["message"] = (
            "Suppose I bring three photographs from different angles—what trade-off would "
            "that create if I also want the view completely stationary?"
        )

        def conversational_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "Use the references for identity and keep the camera lock as a separate shot instruction.",
                # Chat is answer-only, so even an unrelated provider rewrite
                # must neither replace nor invalidate the useful answer.
                "rewrite": "A 10-second montage starring Bob in three locations.",
                "follow_ups": ["Are the three photos all of @maya?"],
            }

        result = coach.coach(
            payload,
            readiness=readiness(),
            model_runner=conversational_model,
        )

        self.assertEqual(result["response_kind"], "conversation")
        self.assertEqual(result["provider"], "test-ai")
        self.assertIn("camera lock", result["answer"])
        self.assertEqual(result["rewrite"], source)
        self.assertEqual(result["rewrite_safety_note"], "")
        self.assertNotIn("Bob", result["rewrite"])

    def test_explicit_chat_allows_empty_video_prompt_and_preserves_question_mode(self):
        payload = request_payload("")
        payload["action"] = "chat"
        payload["message"] = "Explain Motion Track to me without jargon"

        def conversational_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "Motion Track makes a chosen point follow a path you draw.",
                "rewrite": "__UNCHANGED__",
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=conversational_model)

        self.assertEqual(result["response_kind"], "conversation")
        self.assertEqual(result["rewrite"], "")
        self.assertIn("follow a path", result["answer"])
        self.assertEqual(result["creative_completion"]["task"], "conversation")

    def test_explicit_chat_model_failure_never_becomes_score_or_prompt_template(self):
        payload = request_payload("@maya waits beside a window.")
        payload["action"] = "chat"
        payload["message"] = "Talk me through a completely different way to approach this"
        payload["context"]["references"] = {}

        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(result["response_kind"], "conversation")
        self.assertEqual(result["rewrite"], payload["prompt"])
        self.assertIn("couldn't run the local conversational model", result["answer"])
        self.assertNotIn("% ready", result["answer"])
        self.assertNotIn("best available route", result["answer"].lower())
        self.assertEqual(result["follow_ups"], [])

    def test_chat_fact_check_blocks_per_photo_tags_identity_promises_and_fake_extend(self):
        payload = request_payload("@maya waits beside a window.")
        payload["action"] = "chat"
        payload["message"] = "How should several photos of @maya work?"

        def inaccurate_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": (
                    "Assign a different tag to each photo; this locks the identity. "
                    "A captured frame can also start an Extend clip."
                ),
                "rewrite": "__UNCHANGED__",
                "follow_ups": ["Ready?"],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=inaccurate_model)

        self.assertEqual(result["response_kind"], "conversation")
        self.assertIn("withheld", result["answer"])
        self.assertEqual(result["rewrite"], payload["prompt"])
        self.assertEqual(result["follow_ups"], [])
        self.assertIn("assemble the views into one clean reference-sheet image", result["answer_safety_note"])
        self.assertIn("never lock or guarantee", result["answer_safety_note"])
        self.assertIn("not native Extend", result["answer_safety_note"])

    def test_score_action_bypasses_models_and_preserves_prompt_verbatim(self):
        original = "  @maya turns once beneath a blue spotlight.\nKeep the camera locked.  "
        payload = request_payload(original)
        payload["action"] = "score"
        payload["message"] = "Score this prompt."
        forbidden_runner = Mock(side_effect=AssertionError("model runner must not start"))

        with patch.object(
            coach,
            "_apple_prompt",
            side_effect=AssertionError("Apple model input must not be built"),
        ):
            result = coach.coach(
                payload,
                readiness=readiness(),
                allow_model=True,
                model_runner=forbidden_runner,
            )

        forbidden_runner.assert_not_called()
        self.assertEqual(result["response_kind"], "score_report")
        self.assertEqual(result["provider"], "deterministic-score")
        self.assertEqual(result["rewrite"], original)
        self.assertEqual(result["follow_ups"], [])
        self.assertEqual(result["rewrite_safety_note"], "")
        self.assertIn(f"{result['score']}/100", result["answer"])
        self.assertIn("not the probability", result["answer"])
        self.assertIn("Highest-value gap", result["answer"])
        self.assertNotIn("best available route", result["answer"].lower())
        self.assertIn("no conversational model or media job", result["provider_note"].lower())

    def test_versioned_skill_context_is_relevant_and_explicit_about_score(self):
        payload = coach.validate_request(
            request_payload(
                "A storyboard and @maya identity references guide one continuous shot."
            )
        )
        payload["context"]["mode"] = "ingredients"
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())
        skill = coach_skill.build_skill_context(payload, analysis)

        self.assertEqual(skill["version"], "1.4.0")
        self.assertIn("completeness", skill["score_definition"].lower())
        self.assertIn("ingredients", skill["relevant_workflows"])
        self.assertIn("union", skill["relevant_workflows"])
        self.assertIn("generate", skill["relevant_workflows"])
        self.assertEqual(skill["local_workflow_status"]["ingredients"]["availability"], "eligible")
        self.assertIn(
            "cannot be combined",
            " ".join(skill["relevant_workflows"]["ingredients"]["limits"]),
        )
        self.assertIn(
            "static comic strip",
            " ".join(skill["relevant_workflows"]["union"]["limits"]).lower(),
        )

    def test_model_transcripts_receive_the_same_versioned_skill(self):
        payload = coach.validate_request(request_payload())
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())

        ollama_report = json.loads(
            coach._model_messages(payload, analysis)[1]["content"].split("\n", 1)[1]
        )
        apple_report = json.loads(coach._apple_prompt(payload, analysis))

        self.assertEqual(ollama_report["ltx_production_skill"]["version"], "1.4.0")
        self.assertEqual(apple_report["ltx_production_skill"]["version"], "1.4.0")
        self.assertEqual(
            ollama_report["ltx_production_skill"], apple_report["ltx_production_skill"]
        )

    def test_general_chat_retrieves_capability_facts_without_exact_question_routes(self):
        payload = coach.validate_request(request_payload("@maya stands by a window."))
        payload["action"] = "chat"
        payload["message"] = (
            "If I pause around second two and carry on from that picture, what continuity do I lose?"
        )
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())

        ollama_report = json.loads(
            coach._model_messages(payload, analysis)[1]["content"].split("\n", 1)[1]
        )
        capabilities = ollama_report["ltx_production_skill"]["relevant_studio_capabilities"]

        self.assertIn("frame_capture", capabilities)
        self.assertIn("does not trim or stitch", capabilities["frame_capture"]["limits"])
        self.assertEqual(ollama_report["interaction_mode"], "conversation")
        self.assertEqual(ollama_report["creative_completion"]["task"], "conversation")

    def test_guided_coach_develops_the_reported_open_idea_into_a_complete_prompt(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["message"] = "I want you to expand this idea and build a prompt."

        result = coach.coach(payload, readiness=readiness(), allow_model=False)
        rewrite = result["rewrite"]
        signals = coach._signals(rewrite, coach.validate_request(payload)["context"])

        self.assertEqual(result["provider"], "guided")
        self.assertIn("Rowan Kite is the search target", result["answer"])
        self.assertFalse(signals["unresolved_placeholders"], rewrite)
        self.assertNotRegex(rewrite.lower(), r"\b(?:do you know|i gotta)\b")
        self.assertIn("rowan kite", rewrite.lower())
        self.assertRegex(rewrite.lower(), r"\brace(?:s|d|way|track|course|ing)?\b")
        for signal in (
            "subject_basic",
            "action",
            "camera",
            "setting",
            "look",
            "chronological",
            "continuity",
        ):
            with self.subTest(signal=signal):
                self.assertTrue(signals[signal], rewrite)
        self.assertGreaterEqual(result["rewrite_analysis"]["score"], 70)
        draft_analysis = coach.analyse_prompt(
            rewrite,
            coach.validate_request(payload)["context"],
            readiness(),
        )
        for key in (
            "strengths",
            "improvements",
            "risks",
            "render_risk",
            "recommendation_certainty",
            "workflow",
            "semantic_best",
            "workflow_reason",
            "workflow_conflicts",
            "availability_notes",
            "walkthrough",
        ):
            with self.subTest(rewrite_analysis_field=key):
                self.assertEqual(result["rewrite_analysis"][key], draft_analysis[key])
        self.assertTrue(
            any("Camera framing" in item for item in result["rewrite_analysis"]["strengths"]),
            result["rewrite_analysis"]["strengths"],
        )
        self.assertFalse(
            any("Choose one framing" in item for item in result["rewrite_analysis"]["improvements"]),
            result["rewrite_analysis"]["improvements"],
        )

    def test_guided_emergency_writer_preserves_named_meeting_and_location(self):
        source = (
            "no I specifically mentioned that Arden Vale meets Rowan Kite and Mira Sol "
            "in 5 seconds next to the finish marker, this will be the whole scene"
        )
        payload = request_payload(source)
        payload["message"] = "Develop this idea into a production-ready prompt."
        payload["context"]["duration_seconds"] = 5

        result = coach.coach(payload, readiness=readiness(), allow_model=False)
        rewrite = result["rewrite"]

        self.assertEqual(result["provider"], "guided")
        self.assertIn("Arden Vale", result["answer"])
        self.assertIn("Rowan Kite", result["answer"])
        self.assertIn("Mira Sol", result["answer"])
        self.assertNotIn("unnamed", result["answer"].lower())
        for fact in ("Arden Vale", "Rowan Kite", "Mira Sol", "finish marker"):
            with self.subTest(fact=fact):
                self.assertIn(fact, rewrite)
        self.assertIn("One continuous 5-second", rewrite)
        self.assertIn("0.0–0.8 seconds", rewrite)
        self.assertIn("3.8–5.0 seconds", rewrite)
        self.assertIn("locked medium-wide eye-level three-shot", rewrite)
        self.assertIn("no dialogue", rewrite.lower())
        self.assertNotRegex(
            rewrite.lower(),
            r"\b(?:no i specifically|main subject|unnamed|raceway|racetrack)\b",
        )
        self.assertFalse(
            coach._signals(rewrite, coach.validate_request(payload)["context"])[
                "unresolved_placeholders"
            ]
        )

    def test_guided_classic_cartoon_meeting_followups_only_ask_unresolved_choices(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene. Use a classic theatrical 2D "
            "cel-cartoon look with hand-painted backgrounds, clean inked outlines, "
            "expressive squash-and-stretch, warm painted colour, and subtle film grain."
        )
        payload = request_payload(source)
        payload["message"] = "Develop this idea into a complete production-ready prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )

        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(len(result["follow_ups"]), 3)
        follow_ups = " ".join(result["follow_ups"])
        self.assertNotIn("Which visual treatment", follow_ups)
        self.assertIn("preserving the specified classic theatrical 2D cel treatment", follow_ups)
        for question in result["follow_ups"]:
            with self.subTest(question=question):
                self.assertIn("Arden Vale", question)
                self.assertIn("Rowan Kite", question)
                self.assertIn("Mira Sol", question)
                self.assertIn("finish marker", question)
        self.assertIn("leave that visual treatment intact", result["answer"])

    def test_guided_meeting_rewrite_recognizes_locked_medium_wide_three_shot(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. Use a classic theatrical 2D cel-cartoon look."
        )
        payload = request_payload(source)
        payload["message"] = "Develop this idea into a complete production-ready prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )

        result = coach.coach(payload, readiness=readiness(), allow_model=False)
        signals = coach._signals(
            result["rewrite"], coach.validate_request(payload)["context"]
        )

        self.assertTrue(signals["camera"])
        self.assertTrue(signals["camera_lock"])
        self.assertTrue(
            any(
                "Camera framing or movement is specified" in strength
                for strength in result["rewrite_analysis"]["strengths"]
            )
        )
        self.assertFalse(
            any(
                "Choose one framing" in improvement
                for improvement in result["rewrite_analysis"]["improvements"]
            )
        )

    def test_qwen_meeting_draft_is_not_rejected_by_generic_named_character_instruction(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene."
        )
        payload = request_payload(source)
        payload["message"] = (
            "If this is a rough or open-ended idea, develop it into one complete, coherent, "
            "concrete, production-ready shot. Preserve every named character and stated intent, "
            "and do not invent additional named characters, dialogue, lore or scene cuts."
        )
        payload["context"]["duration_seconds"] = 5
        payload["context"]["references"] = {}
        candidate = (
            "One continuous 5-second 16:9 cartoon shot beside a sunlit finish marker. "
            "0.0–1.0 seconds: locked medium-wide three-shot establishes Arden Vale approaching "
            "from screen left while Rowan Kite stands beside the finish marker and Mira Sol "
            "waits at screen right. 1.0–3.8 seconds: Arden Vale stops between them; Rowan Kite "
            "turns his eyes first and then his head toward Arden, while Mira Sol makes one "
            "restrained suspicious glance. Keep all three identities, anatomy, scale, wardrobe, "
            "screen positions and rabbit-hole geography stable. 3.8–5.0 seconds: the three hold "
            "a readable awkward meeting pose so the comic tension lands. Warm afternoon light, "
            "clean cel-animation colour and soft outdoor ambience. No dialogue, narration, music "
            "or offscreen voices. Avoid cuts, duplicate characters, identity swaps, extra limbs, "
            "pose snapping, changing subject scale, camera movement and shifting geography."
        )

        def qwen_like_model(_request, _analysis):
            return {
                "provider": "ollama",
                "provider_label": "Local Qwen",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I preserved all three characters, the five-second timing and rabbit-hole location.",
                "rewrite": candidate,
                "follow_ups": [],
            }

        validated = coach.validate_request(payload)
        contract_names = coach._stable_named_phrases(
            coach._creator_authored_contract(validated)
        )
        self.assertCountEqual(
            contract_names,
            ["arden vale", "rowan kite", "mira sol"],
        )
        self.assertNotIn("character", contract_names)

        result = coach.coach(payload, readiness=readiness(), model_runner=qwen_like_model)

        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["rewrite"], candidate)
        self.assertEqual(result["rewrite_safety_note"], "")

    def test_meeting_contract_still_rejects_genuine_named_character_removal(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene."
        )
        payload = request_payload(source)
        payload["message"] = "Develop it, preserving every named character."
        payload["context"]["duration_seconds"] = 5
        payload["context"]["references"] = {}
        missing_named_subject = (
            "One continuous 5-second 16:9 cartoon shot beside the finish marker. Arden Vale "
            "approaches Rowan Kite and a generic spectator in a locked medium-wide shot. They "
            "exchange one restrained silent reaction and hold the final pose. Warm afternoon "
            "cel-animation light and soft outdoor ambience. Preserve anatomy and geography; "
            "avoid cuts, duplicate characters, extra limbs and camera movement."
        )

        def genericizing_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I simplified the cast.",
                "rewrite": missing_named_subject,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=genericizing_model)

        self.assertNotEqual(result["rewrite"], missing_named_subject)
        self.assertIn("Mira Sol", result["rewrite"])
        self.assertRegex(
            result["rewrite_safety_note"].lower(),
            r"removed or genericized.*mira sol",
        )

    def test_model_placeholder_draft_is_rejected_for_a_rough_idea(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete prompt."
        unsafe = (
            "One continuous 3-second 16:9 shot. Rowan Kite waits on a festival raceway. "
            "Subject continuity: [fixed appearance]. Action: [one clear action]. "
            "Camera: [choose a framing]. Setting and light: [choose a style]."
        )

        def placeholder_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "Here is the expanded prompt.",
                "rewrite": unsafe,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=placeholder_model)

        self.assertNotEqual(result["rewrite"], unsafe)
        self.assertFalse(
            coach._signals(result["rewrite"], coach.validate_request(payload)["context"])[
                "unresolved_placeholders"
            ]
        )
        self.assertIn("placeholder", result["rewrite_safety_note"].lower())

    def test_model_cannot_drop_or_genericize_an_explicit_named_target(self):
        source = "A determined racer searches for Rowan Kite on a festival raceway."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete prompt."
        genericized = (
            "One continuous 3-second 16:9 shot. A determined racer searches for a cheerful rabbit "
            "on a sunlit festival raceway, beginning at the start line, following one clue, and ending "
            "beside a painted tunnel. Medium-wide locked camera, saturated cel-animation lighting; "
            "preserve the racer and avoid cuts or duplicate subjects."
        )

        def genericizing_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I made the target more generic.",
                "rewrite": genericized,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=genericizing_model)

        self.assertNotEqual(result["rewrite"], genericized)
        self.assertIn("Rowan Kite", result["rewrite"])
        self.assertRegex(result["rewrite_safety_note"].lower(), r"(?:name|named|target|identity)")

    def test_model_cannot_replace_search_target_with_merchandise_or_invent_plot(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."
        unsafe = (
            "One continuous 3-second 16:9 cartoon shot. BOB drives onto a bright raceway, "
            "crashes into a wooden crate, and finds a glass jar of tiny Rowan Kite figures. "
            "A medium tracking camera follows the action in warm cel-animated light. "
            "Preserve BOB and the vehicle; avoid cuts, duplicate subjects, and warped wheels."
        )

        def premise_changing_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I completed the idea.",
                "rewrite": unsafe,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=premise_changing_model)

        self.assertNotEqual(result["rewrite"], unsafe)
        self.assertIn("Rowan Kite", result["rewrite"])
        safety = result["rewrite_safety_note"].lower()
        self.assertIn("proxy object or representation", safety)
        self.assertIn("invented a new named character", safety)
        self.assertIn("crash or collision", safety)

    def test_semantic_gate_allows_concrete_completion_that_preserves_open_premise(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."
        completion = (
            "One continuous 3-second 16:9 shot in saturated hand-drawn cartoon style. "
            "An unnamed determined racer searches for Rowan Kite while steering along a sunlit "
            "desert raceway. 0.0-0.7 seconds: begin from the supplied start frame and keep the "
            "racer's appearance, vehicle geometry, lane position, and scale stable. "
            "0.7-2.3 seconds: a medium-wide lateral tracking camera follows as the racer scans "
            "roadside clues without leaving the race, making one controlled steering correction. "
            "2.3-3.0 seconds: the racer spots a fresh star-shaped dust trail pointing beyond the "
            "finish banner, reacts with focused recognition, and continues the search; Rowan Kite "
            "remains unrevealed. Preserve coherent geography, anatomy, wheels, and direction of "
            "travel. Warm afternoon light, crisp cel shading, restrained squash-and-stretch, faint "
            "engine whirr and tire hum only; no dialogue, narration, or music. Avoid cuts, crashes, "
            "duplicate racers, extra limbs, warped wheels, changing vehicle design, readable text, "
            "or unintended camera moves."
        )

        def premise_preserving_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I kept the search inside the cartoon race and made reversible choices.",
                "rewrite": completion,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=premise_preserving_model)

        self.assertEqual(result["rewrite"], completion)
        self.assertEqual(result["rewrite_safety_note"], "")

    def test_semantic_gate_accepts_creator_authored_followup_identity_ending_tag_and_dialogue(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["history"] = [
            {"role": "user", "content": "Develop this rough idea into a complete shot."},
            {
                "role": "assistant",
                "content": "Current production-ready draft: an unnamed racer follows a clue.",
            },
        ]
        payload["message"] = (
            'The lead is @arden, Arden Vale. End on a comic near-miss, and have him say '
            '"Missed him again!"'
        )
        completion = (
            "One continuous 3-second 16:9 cartoon shot. @arden, Arden Vale, searches for "
            "Rowan Kite while racing along a bright desert raceway. 0.0-0.6 seconds: a "
            "medium-wide lateral tracking camera establishes Arden scanning the road while "
            "his vehicle, wardrobe, bill, anatomy, scale, and lane direction remain stable. "
            "0.6-2.2 seconds: Arden's eyes find a star-shaped dust clue; his gaze leads one "
            "controlled steering correction and his body follows with restrained cartoon "
            "secondary motion. 2.2-3.0 seconds: he narrowly avoids a harmless road marker in "
            'the requested comic near-miss, settles safely, and says "Missed him again!" while '
            "Rowan Kite remains unrevealed. Hold the readable frustrated final pose. Warm "
            "afternoon light, saturated hand-drawn cel animation, coherent raceway geography, "
            "faint engine whirr and tire hum. Exactly one voice, Arden Vale only. Avoid cuts, "
            "crashes, duplicate racers, extra limbs, warped wheels, identity drift, changing "
            "travel direction, unreadable text, extra voices, or additional camera moves."
        )

        def followup_aware_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I applied your identity, ending beat, tag, and exact line.",
                "rewrite": completion,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=followup_aware_model)

        self.assertEqual(result["provider"], "test-ai")
        self.assertEqual(result["rewrite"], completion)
        self.assertEqual(result["rewrite_safety_note"], "")
        self.assertIn("Arden Vale", result["rewrite"])
        self.assertIn("near-miss", result["rewrite"])
        self.assertIn('@arden', result["rewrite"])
        self.assertIn('"Missed him again!"', result["rewrite"])

    def test_assistant_history_cannot_authorize_its_invented_bob_identity(self):
        source = "do you know festival races? I gotta find Rowan Kite."
        payload = request_payload(source)
        payload["history"] = [
            {"role": "user", "content": "Develop this rough idea into a complete shot."},
            {
                "role": "assistant",
                "content": "Current production-ready draft: Bob is the lead racer.",
            },
        ]
        payload["message"] = "The lead is Arden Vale and the final beat is a comic near-miss."
        invented = (
            "One continuous 3-second 16:9 cartoon shot. Bob races along a sunlit desert "
            "raceway searching for Rowan Kite. A medium tracking camera follows Bob through "
            "one bend before he narrowly avoids a barrier in a comic near-miss and holds a "
            "stable final pose. Preserve the vehicle and course geography; avoid cuts, extra "
            "limbs, duplicate racers, dialogue, and unintended camera movement."
        )

        def assistant_contaminated_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I used the prior draft.",
                "rewrite": invented,
                "follow_ups": [],
            }

        result = coach.coach(
            payload,
            readiness=readiness(),
            model_runner=assistant_contaminated_model,
        )

        self.assertNotEqual(result["rewrite"], invented)
        self.assertNotIn("Bob", result["rewrite"])
        self.assertIn("invented a new named character: bob", result["rewrite_safety_note"].lower())
        self.assertIn("Arden Vale", coach._creator_authored_contract(coach.validate_request(payload)))
        self.assertNotIn("Bob", coach._creator_authored_contract(coach.validate_request(payload)))

    def test_live_sparse_idea_regression_rejects_invented_cast_prop_twist_and_camera_pileup(self):
        source = "do you know festival races? I gotta find Rowan Kite"
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(source, validated["context"], readiness())
        fallback = analysis["rewrite"]
        bad_live_output = (
            "One continuous 3-second 16:9 shot. BOB (40s), a lanky cartoon racer in a red "
            "jumpsuit, speeds along a sunny desert raceway while clutching a glass jar full of "
            "tiny Rowan Kite figures. 0.0–0.7 seconds: a close-up push-in reveals the jar. "
            "0.7–2.2 seconds: the camera whip-pans, dollies alongside Bob, then orbits as he "
            "swerves and crashes through a barrier, scattering the figures. 2.2–3.0 seconds: "
            "zoom into Bob staring at the broken jar. Saturated cel-animation light, stable "
            "identity, no cuts. Avoid extra limbs, duplicate characters and unreadable text."
        )

        rewrite, note = coach._safe_provider_rewrite(
            bad_live_output,
            source_prompt=source,
            fallback=fallback,
            original_analysis=analysis,
            context=validated["context"],
            readiness=readiness(),
        )

        self.assertEqual(rewrite, fallback)
        self.assertIn("invented a new named character", note.lower())
        self.assertIn("incoherent stack of camera moves", note.lower())
        self.assertNotRegex(rewrite.lower(), r"\bbob\b|\bjar\b|\bcrash(?:es|ed|ing)?\b")
        self.assertIn("Rowan Kite", rewrite)
        self.assertIn("0.0–0.5 seconds", rewrite)
        self.assertIn("2.2–3.0 seconds", rewrite)
        self.assertIn("one smooth tracking move", rewrite)
        self.assertFalse(coach._signals(rewrite, validated["context"])["unresolved_placeholders"])
        self.assertGreaterEqual(
            coach.analyse_prompt(rewrite, validated["context"], readiness())["score"],
            70,
        )

    def test_rejected_development_replaces_contaminated_answer_and_questions_atomically(self):
        source = "do you know festival races? I gotta find Rowan Kite"
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."
        bad_live_output = (
            "One continuous 3-second 16:9 shot. BOB (40s) reaches for a jar of Rowan Kite "
            "figures in a living room, crashes into the table, and falls backward. Camera pushes "
            "in, pans right, then orbits. Warm natural light; avoid duplicate subjects."
        )

        def drifting_model(_request, _analysis):
            return {
                "provider": "apple-foundation-model",
                "provider_label": "Apple on-device AI",
                "provider_note": "Generated privately.",
                "provider_action": "",
                "answer": "Bob's jar-and-crash scene is ready.",
                "rewrite": bad_live_output,
                "follow_ups": [
                    "What emotional tone should Bob's crash have?",
                    "Should the camera end on Bob or the broken jar?",
                ],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=drifting_model)

        self.assertEqual(result["provider"], "guided")
        self.assertIn("Guarded local development", result["provider_label"])
        self.assertNotRegex(result["answer"].lower(), r"\bbob\b|\bjar\b|\bcrash")
        self.assertTrue(result["follow_ups"])
        self.assertNotRegex(" ".join(result["follow_ups"]).lower(), r"\bbob\b|\bjar\b|\bcrash")
        self.assertIn("Who is searching", result["follow_ups"][0])
        self.assertIn("semantic gate rejected", result["provider_note"])
        self.assertIn("Rowan Kite", result["rewrite"])
        self.assertNotEqual(result["rewrite"], bad_live_output)

    def test_rejected_qwen_candidate_fails_over_to_validated_apple_once(self):
        source = "A determined racer searches for Rowan Kite on a festival raceway."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."
        qwen_candidate = {
            "provider": "ollama",
            "provider_label": "Local AI · test-model",
            "provider_note": "Qwen generated privately.",
            "provider_action": "",
            "answer": "Bob's crash scene is ready.",
            "rewrite": (
                "One continuous 3-second 16:9 shot. BOB crashes into a jar of "
                "Rowan Kite figurines while the camera pans, orbits and zooms."
            ),
            "follow_ups": ["How hard should Bob crash?"],
        }

        def valid_apple(_request, analysis):
            return {
                "provider": "apple-foundation-model",
                "provider_label": "Apple on-device AI",
                "provider_note": "Apple generated privately.",
                "provider_action": "",
                "answer": "I kept Rowan Kite as the search target and staged one readable beat.",
                "rewrite": analysis["rewrite"],
                "follow_ups": ["Which reference should define the unnamed racer?"],
            }

        with (
            patch.object(coach, "_run_apple_agent", side_effect=valid_apple) as apple,
            patch.object(coach, "_run_ollama_agent", return_value=qwen_candidate) as ollama,
        ):
            result = coach.coach(payload, readiness=readiness())

        apple.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual(result["provider"], "apple-foundation-model")
        self.assertIn("Rowan Kite", result["rewrite"])
        self.assertNotRegex(result["rewrite"].lower(), r"\bbob\b|\bjar\b|\bcrash")
        self.assertIn("Failover audit", result["provider_note"])
        self.assertIn("creator-contract gate rejected", result["provider_note"])
        self.assertIn("passed the same gate", result["provider_note"])
        self.assertEqual(result["rewrite_safety_note"], "")

    def test_all_provider_candidates_rejected_falls_back_with_public_audit(self):
        source = "A determined racer searches for Rowan Kite on a festival raceway."
        payload = request_payload(source)
        payload["message"] = "Expand this idea into a complete production prompt."

        def unsafe(provider, label, rewrite):
            return {
                "provider": provider,
                "provider_label": label,
                "provider_note": "private",
                "provider_action": "",
                "answer": "Done.",
                "rewrite": rewrite,
                "follow_ups": [],
            }

        apple_candidate = unsafe(
            "apple-foundation-model",
            "Apple on-device AI",
            "One continuous 3-second 16:9 shot. BOB crashes into a jar of rabbit figurines.",
        )
        ollama_candidate = unsafe(
            "ollama",
            "Local AI · test-model",
            "One continuous 3-second 16:9 shot. A driver searches for a generic rabbit.",
        )
        with (
            patch.object(coach, "_run_apple_agent", return_value=apple_candidate) as apple,
            patch.object(coach, "_run_ollama_agent", return_value=ollama_candidate) as ollama,
        ):
            result = coach.coach(payload, readiness=readiness())

        apple.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual(result["provider"], "guided")
        self.assertIn("Rowan Kite", result["rewrite"])
        self.assertIn("Provider audit", result["provider_note"])
        self.assertEqual(result["provider_note"].count("creator-contract gate rejected"), 2)
        self.assertNotRegex(result["answer"].lower(), r"\bbob\b|\bjar\b|\bcrash")

    def test_unavailable_qwen_fails_over_to_apple_without_second_model_call(self):
        payload = request_payload(
            "One continuous 3-second 16:9 shot. A brass robot in a red coat turns toward "
            "a window in a warm studio. Locked medium camera; preserve its geometry and avoid cuts."
        )

        def valid_apple(_request, analysis):
            return {
                "provider": "apple-foundation-model",
                "provider_label": "Apple on-device AI",
                "provider_note": "Apple generated privately.",
                "provider_action": "",
                "answer": "This is ready for a short locked-camera test.",
                "rewrite": analysis["rewrite"],
                "follow_ups": [],
            }

        with (
            patch.object(
                coach, "_run_ollama_agent", side_effect=RuntimeError("local model unavailable")
            ) as ollama,
            patch.object(coach, "_run_apple_agent", side_effect=valid_apple) as apple,
        ):
            result = coach.coach(payload, readiness=readiness())

        apple.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual(result["provider"], "apple-foundation-model")
        self.assertIn("Qwen local AI was unavailable", result["provider_note"])
        self.assertIn("passed the same gate", result["provider_note"])

    def test_qwen_context_limit_does_not_prevent_builtin_apple_failover(self):
        payload = request_payload(
            "A brass robot searches for Rowan Kite on a festival raceway. "
            + ("warm textured atmosphere " * 1_100)
        )
        payload["message"] = "Expand this into a production prompt."
        self.assertGreater(
            len(
                coach._apple_prompt(
                    coach.validate_request(payload),
                    coach.analyse_prompt(
                        payload["prompt"],
                        coach.validate_request(payload)["context"],
                        readiness(),
                    ),
                ).encode("utf-8")
            ),
            coach.MAX_CONVERSATIONAL_MODEL_INPUT_BYTES,
        )

        def valid_apple(_request, analysis):
            return {
                "provider": "apple-foundation-model",
                "provider_label": "Apple on-device AI",
                "provider_note": "Apple generated privately.",
                "provider_action": "",
                "answer": "I preserved the search target.",
                "rewrite": analysis["rewrite"],
                "follow_ups": [],
            }

        with (
            patch.object(
                coach, "_run_ollama_agent", side_effect=RuntimeError("compact Qwen context exceeded")
            ) as ollama,
            patch.object(coach, "_run_apple_agent", side_effect=valid_apple) as apple,
        ):
            result = coach.coach(payload, readiness=readiness())

        apple.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual(result["provider"], "apple-foundation-model")
        self.assertIn("Failover audit", result["provider_note"])

    def test_model_cannot_violate_explicit_silence_or_locked_camera(self):
        source = (
            "One continuous 3-second 16:9 shot. A silent brass robot turns toward a red product "
            "in a warm studio. Locked medium camera; no camera movement and no dialogue, speech, "
            "music, narration, or offscreen voices. Preserve its geometry and avoid cuts."
        )
        payload = request_payload(source)
        unsafe = (
            'One continuous 3-second 16:9 shot. A brass robot turns toward a red product and says "Hello" '
            "as the camera tracks around it in a warm cinematic studio. Preserve its geometry and avoid cuts."
        )

        def constraint_breaking_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I added energy.",
                "rewrite": unsafe,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=constraint_breaking_model)

        self.assertNotEqual(result["rewrite"], unsafe)
        self.assertIn("no camera movement", result["rewrite"].lower())
        self.assertIn("no dialogue", result["rewrite"].lower())
        self.assertRegex(result["rewrite_safety_note"].lower(), r"camera")
        self.assertRegex(result["rewrite_safety_note"].lower(), r"(?:silence|silent|dialogue|speech)")

    def test_model_grounding_excludes_fallback_draft_and_declares_creative_mode(self):
        payload = request_payload("do you know festival races? I gotta find Rowan Kite.")
        payload["message"] = "I want you to expand this idea and build a prompt."
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(validated["prompt"], validated["context"], readiness())

        ollama_report = json.loads(
            coach._model_messages(validated, analysis)[1]["content"].split("\n", 1)[1]
        )
        apple_report = json.loads(coach._apple_prompt(validated, analysis))

        for report in (ollama_report, apple_report):
            with self.subTest(provider=report):
                self.assertNotIn("rewrite", report["verified_analysis"])
                self.assertEqual(report["creative_completion"]["task"], "develop")
                self.assertTrue(report["creative_completion"]["enabled"])
                rules = " ".join(report["creative_completion"]["rules"]).lower()
                self.assertIn("concrete", rules)
                self.assertIn("no square-bracket", rules)
                self.assertIn("preserve every explicit name", rules)

    def test_both_local_model_prompts_require_fact_grounded_precision_development(self):
        payload = request_payload("do you know festival races? I gotta find Rowan Kite.")
        payload["message"] = "Expand this idea and build a complete production prompt."
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        ollama_instructions = coach._model_messages(validated, analysis)[0]["content"]
        observed: dict[str, str] = {}

        def fake_run(argv, **_kwargs):
            observed["apple_instructions"] = argv[argv.index("--instructions") + 1]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "answer": "I completed the rough premise using only reversible assumptions.",
                        "rewrite": analysis["rewrite"],
                        "follow_ups": [],
                    }
                ),
                stderr="",
            )

        with (
            patch.object(coach, "_apple_fm_executable", return_value="/usr/bin/fm"),
            patch.object(coach, "_apple_fm_license_ready"),
            patch.object(coach, "_apple_fm_available"),
            patch.object(coach.subprocess, "run", side_effect=fake_run),
        ):
            coach._run_apple_agent(validated, analysis)

        for provider, instructions in (
            ("ollama", ollama_instructions),
            ("apple", observed["apple_instructions"]),
        ):
            with self.subTest(provider=provider):
                lowered = instructions.lower()
                self.assertIn("fact ledger", lowered)
                self.assertIn("named search target stays the target", lowered)
                self.assertIn("unnamed functional role", lowered)
                self.assertIn("non-overlapping chronological beats", lowered)
                self.assertIn("readable final hold", lowered)
                self.assertIn("exact audio ownership", lowered)
                self.assertIn("no invented speech", lowered)
                self.assertIn("never claim a loaded start/end frame", lowered)

        schema = json.loads(coach._APPLE_RESPONSE_SCHEMA)
        rewrite_contract = schema["properties"]["rewrite"]["description"].lower()
        self.assertIn("preserve the premise", rewrite_contract)
        self.assertIn("named entities and their roles", rewrite_contract)
        self.assertIn("invent no proper names, dialogue or reference claims", rewrite_contract)

    def test_sparse_development_grounding_leaves_room_for_a_precision_draft(self):
        payload = coach.validate_request(
            request_payload("do you know festival races? I gotta find Rowan Kite.")
        )
        payload["message"] = "Expand this idea and build a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 3, "aspect_ratio": "profile", "references": {}}
        )
        analysis = coach.analyse_prompt(
            payload["prompt"], payload["context"], readiness()
        )

        grounded = coach._apple_prompt(payload, analysis).encode("utf-8")

        # Apple's system model has a 4,096-token session budget shared by
        # instructions, prompt and output. Keep this report compact enough to
        # leave meaningful generation space for the 220–450 word draft.
        self.assertLess(len(grounded), 12_000)

    def test_sparse_named_character_draft_audit_flags_skeletal_invented_identity_details(self):
        payload = request_payload(
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene."
        )
        payload["message"] = "Develop this idea into a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        skeletal = (
            "A locked 5-second shot beside the finish marker. Arden Vale, wearing a blue jacket "
            "and round goggles, looks toward Rowan Kite in a green hat while Mira Sol holds a "
            'ceremonial sword. Arden says "Hello." They react, then hold. Warm light and outdoor ambience; no cuts.'
        )

        issues = coach._sparse_develop_quality_issues(
            skeletal, validated, analysis
        )

        self.assertTrue(any("skeletal" in issue for issue in issues))
        self.assertTrue(any("time ranges" in issue for issue in issues))
        self.assertTrue(any("not supplied" in issue for issue in issues))
        joined = " ".join(issues).lower()
        self.assertIn("jacket", joined)
        self.assertIn("sword", joined)
        self.assertIn("invented spoken line", joined)

    def test_browser_develop_command_cannot_bypass_sparse_draft_audit(self):
        payload = request_payload(
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene. Use a classic theatrical 2D "
            "cel-cartoon look with hand-painted backgrounds and clean inked outlines."
        )
        payload["message"] = (
            "If this is a rough or open-ended idea, develop it into one complete, coherent, "
            "concrete, production-ready shot by making sensible creative choices for action, "
            "chronology, camera, setting, lighting, style and ambience; do not leave bracketed "
            "placeholders. Preserve every named character and stated intent, and do not invent "
            "additional named characters, dialogue, lore or scene cuts unless requested. If the "
            "prompt is already detailed, review it conservatively and preserve its decisions. In "
            "either case, explain the best available workflow and exact settings for this Studio."
        )
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        skeletal = (
            "A 5-second profile-default shot beside the finish marker. Arden Vale, Rowan Kite "
            "and Mira Sol stand together in warm cel-animation light, react, and hold. "
            "The camera is locked. No dialogue; avoid cuts and duplicate characters."
        )

        issues = coach._sparse_develop_quality_issues(skeletal, validated, analysis)

        self.assertEqual(coach._coach_task(validated, analysis), "develop")
        self.assertTrue(any("skeletal" in issue for issue in issues))
        self.assertTrue(any("time ranges" in issue for issue in issues))

    def test_classic_cartoon_guardrail_rejects_magic_and_environmental_warping(self):
        authority = (
            "Arden Vale meets Rowan Kite and Mira Sol beside the finish marker. "
            "Use a classic theatrical 2D cel-cartoon look with hand-painted backgrounds, "
            "clean inked outlines and expressive squash-and-stretch."
        )
        rewrite = (
            "Use the requested classic theatrical 2D cel-cartoon treatment. Arden Vale "
            "approaches the finish marker and stops. The finish marker begins to glow and the "
            "ground warps, causing Rowan Kite to lean back while Mira Sol blinks. Hold "
            "the resulting reaction; no dialogue or added props."
        )

        issues = coach._classic_cartoon_development_issues(rewrite, authority)

        self.assertTrue(any("magic" in issue and "warping" in issue for issue in issues))
        self.assertFalse(any("cause-and-effect" in issue for issue in issues))

    def test_classic_cartoon_guardrail_accepts_one_grounded_physical_reaction_gag(self):
        authority = (
            "Arden Vale meets Rowan Kite and Mira Sol beside the finish marker. "
            "Use a classic theatrical 2D cel-cartoon look with hand-painted backgrounds, "
            "clean inked outlines and expressive squash-and-stretch."
        )
        rewrite = (
            "Use the requested classic theatrical 2D cel-cartoon treatment with hand-painted "
            "backgrounds, clean inked outlines and restrained squash-and-stretch. Arden Vale "
            "takes one brisk step beside the finish marker and stops in a compact squash. The "
            "sudden stop causes Rowan Kite to lean back and Mira Sol to blink one beat later. "
            "All three settle into the resulting off-balance pose and hold; no dialogue or props."
        )

        self.assertEqual(
            coach._classic_cartoon_development_issues(rewrite, authority),
            [],
        )

    def test_guided_classic_cartoon_meeting_uses_one_physical_reaction_gag(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene. Use a classic theatrical 2D "
            "cel-cartoon look with hand-painted backgrounds, clean inked outlines, "
            "expressive squash-and-stretch, warm painted colour, and subtle film grain."
        )
        payload = request_payload(source)
        payload["message"] = "Develop this idea into a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )

        result = coach.coach(payload, readiness=readiness(), allow_model=False)
        rewrite = result["rewrite"].lower()

        self.assertIn("squash-and-settle", rewrite)
        self.assertIn("causes rowan kite and mira sol to lean back", rewrite)
        self.assertIn("double-take", rewrite)
        self.assertNotRegex(
            rewrite,
            r"\b(?:magic|magical|portal|warp(?:s|ed|ing)?|glowing aura)\b",
        )

    def test_question_word_what_is_not_an_invented_named_character(self):
        source = (
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene. Use a classic theatrical 2D "
            "cel-cartoon look."
        )
        payload = request_payload(source)
        payload["message"] = "Develop this idea into a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        candidate = analysis["rewrite"] + " What is happening remains readable throughout."

        rewrite, note = coach._safe_provider_rewrite(
            candidate,
            source_prompt=source,
            creator_contract=coach._creator_authored_contract(validated),
            fallback=analysis["rewrite"],
            original_analysis=analysis,
            context=validated["context"],
            readiness=readiness(),
        )

        self.assertNotIn(
            "what", coach._high_confidence_named_identities(candidate)
        )
        self.assertEqual(note, "")
        self.assertEqual(rewrite, candidate)

    def test_corrective_pass_masks_and_restores_famous_character_names(self):
        source = "Arden Vale meets Rowan Kite and Mira Sol beside the finish marker."

        masked, restore = coach._masked_identity_contract(
            source, coach._stable_named_phrases(source)
        )
        restored = coach._restore_masked_identities(
            {
                "rewrite": "SUBJECT_1 notices Subject 2; subject-3 reacts.",
                "follow_ups": ["Should SUBJECT 1 lead the final hold?"],
            },
            restore,
        )

        self.assertEqual(
            masked,
            "SUBJECT_1 meets SUBJECT_2 and SUBJECT_3 beside the finish marker.",
        )
        self.assertEqual(
            restored["rewrite"],
            "Arden Vale notices Rowan Kite; Mira Sol reacts.",
        )
        self.assertEqual(
            restored["follow_ups"],
            ["Should Arden Vale lead the final hold?"],
        )

    def test_production_heading_and_first_name_alias_are_not_invented_characters(self):
        source = "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds beside the finish marker."
        payload = request_payload(source)
        payload["message"] = "Develop it while preserving every named character."
        payload["context"].update({"duration_seconds": 5, "references": {}})
        candidate = (
            "One continuous 5-second 16:9 shot. Arden Vale approaches Rowan Kite and Mira Sol "
            "beside the finish marker. Arden stops at the shared mark and exchanges one restrained "
            "recognition beat with both characters before a readable final hold. Lighting: warm "
            "late-afternoon cel-animation light. Camera: locked medium-wide three-shot. Preserve "
            "identity, scale and geography; avoid cuts, duplicates and unintended camera motion."
        )

        def qwen_like_model(_request, _analysis):
            return {
                "provider": "ollama",
                "provider_label": "Local Qwen",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I kept the full named cast.",
                "rewrite": candidate,
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=readiness(), model_runner=qwen_like_model)

        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["rewrite"], candidate)
        self.assertEqual(result["rewrite_safety_note"], "")

    def test_common_multiword_production_headings_are_not_named_characters(self):
        production_copy = (
            "Visual Style: classic theatrical cel animation.\n"
            "Sound Design: light outdoor ambience.\n"
            "Character Blocking: keep the trio geographically distinct.\n"
            "Background is a hand-painted desert landscape."
        )

        self.assertEqual(coach._high_confidence_named_identities(production_copy), set())

    def test_scenery_and_classic_frame_wording_do_not_override_aspect(self):
        context = coach.validate_request(request_payload())["context"]
        context["aspect_ratio"] = "profile"
        style_copy = (
            "Background is a hand-painted desert landscape inside a classic frame, "
            "with clean inked outlines."
        )

        self.assertIsNone(coach._explicit_prompt_aspect(style_copy))
        self.assertEqual(coach._signals(style_copy, context)["aspect"], "profile")
        self.assertEqual(coach._explicit_prompt_aspect("Use landscape orientation."), "16:9")
        self.assertEqual(coach._explicit_prompt_aspect("One continuous 4:3 shot."), "4:3")

    def test_multiple_visible_characters_alone_does_not_select_ingredients(self):
        payload = request_payload(
            "Keep multiple characters readable beside the finish marker in one locked shot."
        )
        payload["context"]["references"] = {}
        context = coach.validate_request(payload)["context"]
        signals = coach._signals(payload["prompt"], context)

        self.assertFalse(signals["multi_reference"])
        self.assertEqual(coach._semantic_workflow(signals, context)[0], "generate")

        context["references"]["ingredient_references"] = 2
        referenced = coach._signals(payload["prompt"], context)
        self.assertTrue(referenced["multi_reference"])
        self.assertEqual(coach._semantic_workflow(referenced, context)[0], "ingredients")

    def test_quoted_production_term_is_not_mistaken_for_dialogue(self):
        context = coach.validate_request(request_payload())["context"]
        production_copy = (
            "Use the phrase “classic theatrical cartoon” only as a visual direction; "
            "no dialogue."
        )

        self.assertFalse(coach._signals(production_copy, context)["dialogue"])
        self.assertFalse(
            coach._signals('Use “hand-painted cel” styling.', context)["dialogue"]
        )
        self.assertTrue(coach._signals('Arden says "Hello."', context)["dialogue"])
        self.assertTrue(coach._signals('Exactly one voice says “Hello.”', context)["dialogue"])

    def test_holding_a_final_pose_is_not_treated_as_an_invented_prop(self):
        payload = request_payload(
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds beside the finish marker."
        )
        payload["message"] = "Develop this into a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        final_pose = (
            "Arden Vale, Rowan Kite and Mira Sol finish by holding their final pose "
            "beside the finish marker."
        )
        held_prop = (
            "Arden Vale, Rowan Kite and Mira Sol finish with Arden holding a brass telescope "
            "beside the finish marker."
        )

        pose_issues = coach._sparse_develop_quality_issues(
            final_pose, validated, analysis
        )
        prop_issues = coach._sparse_develop_quality_issues(
            held_prop, validated, analysis
        )
        self.assertFalse(any("not supplied" in issue for issue in pose_issues))
        self.assertTrue(any("not supplied" in issue for issue in prop_issues))

    def test_decimal_timeline_beats_are_not_read_as_requested_duration(self):
        timeline = (
            "0.0–0.8 seconds: establish the trio. "
            "0.8–2.2 seconds: Arden approaches. "
            "2.2–3.8 seconds: they react. "
            "3.8–5.0 seconds: hold the final pose."
        )
        context = coach.validate_request(request_payload())["context"]
        context["duration_seconds"] = 5

        self.assertIsNone(coach._explicit_prompt_duration(timeline))
        self.assertEqual(coach._signals(timeline, context)["duration"], 5)
        self.assertIsNone(coach._explicit_prompt_duration("At 1.5 seconds, Arden turns."))
        self.assertEqual(
            coach._explicit_prompt_duration("One continuous 5-second shot. " + timeline),
            5,
        )
        self.assertEqual(coach._explicit_prompt_duration("Duration is 5.0 seconds."), 5)

        word_ranges = (
            "0.0 to 1.0 seconds: establish the trio. "
            "1.0 to 2.5 seconds: Arden approaches. "
            "2.5 to 4.0 seconds: they react. "
            "4.0 to 5.0 seconds: hold the final pose."
        )
        self.assertIsNone(coach._explicit_prompt_duration(word_ranges))
        self.assertEqual(coach._signals(word_ranges, context)["duration"], 5)

    def test_negative_camera_exclusions_do_not_create_conflicting_moves(self):
        prompt = (
            "One continuous 5-second shot with a locked medium camera. "
            "No zoom, pan, tilt, orbit, reframing or lens change."
        )
        context = coach.validate_request(request_payload())['context']

        signals = coach._signals(prompt, context)
        analysis = coach.analyse_prompt(prompt, context, readiness())

        self.assertTrue(signals["camera_lock"])
        self.assertFalse(signals["camera_move"])
        self.assertLessEqual(len(signals["camera_terms"]), 2)
        self.assertFalse(
            any("camera instructions may conflict" in risk for risk in analysis["risks"])
        )

    def test_qwen_sparse_development_gets_one_bounded_quality_revision(self):
        payload = request_payload(
            "Arden Vale meets Rowan Kite and Mira Sol in 5 seconds next to the "
            "finish marker. This is the whole scene."
        )
        payload["message"] = "Develop this idea into a complete production prompt."
        payload["context"].update(
            {"duration_seconds": 5, "aspect_ratio": "profile", "references": {}}
        )
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(
            validated["prompt"], validated["context"], readiness()
        )
        first = {
            "answer": "I filled the scene.",
            "rewrite": (
                "A locked 5-second shot. Arden Vale wears a blue scarf beside Rowan Kite "
                "while Mira Sol holds a brass telescope at the finish marker. They react and hold."
            ),
            "follow_ups": ["What style should it use?"],
        }
        rich = (
            "One continuous 5-second profile-default trained-bucket shot beside the finish marker. "
            "Preserve Arden Vale, Rowan Kite and Mira Sol as three recognizable, geographically "
            "distinct characters and keep the landmark fixed throughout. 0.0–0.7 seconds: begin "
            "with a medium-wide eye-level three-shot that immediately establishes Arden Vale "
            "approaching from screen left, Rowan Kite nearest the finish marker and Mira Sol on "
            "screen right. Give the audience time to register all three identities before the "
            "meeting begins. 0.7–2.0 seconds: Arden reaches the shared mark and stops cleanly; his "
            "gaze finds Rowan first, then shifts toward Mira. Rowan notices the arrival through one "
            "small eye-led turn while Mira registers it a beat later. Keep feet, subject scale and "
            "screen direction stable. 2.0–3.8 seconds: let the three exchange one restrained, "
            "readable recognition beat without changing places or introducing another plot event. "
            "Use only subtle posture settling and small natural secondary motion so anatomy remains "
            "stable. 3.8–5.0 seconds: resolve into a balanced shared tableau beside the finish marker "
            "and hold long enough for the comic tension to land. Camera remains completely locked: "
            "no pan, tilt, orbit, zoom, reframing or lens change. Clean late-afternoon directional "
            "light separates the silhouettes from a softly textured outdoor background while the "
            "rabbit-hole geography remains unchanged. Synchronized audio consists only of a light "
            "breeze, distant birds and quiet movement settling; no dialogue, narration, music or "
            "offscreen voices. Avoid cuts, duplicate characters, extra limbs, identity swaps, pose "
            "snapping, changing scale, crossed screen direction, moving landmarks and unintended "
            "camera movement."
        )
        revised = {
            "answer": "I replaced unsupported identity details with precise staging and timing.",
            "rewrite": rich,
            "follow_ups": [
                "Should Arden Vale, Rowan Kite and Mira Sol end on comic surprise or suspicion?",
                "Should the rabbit-hole scene use classic cel animation or modern 3D?",
                "Should the meeting remain silent or use one exact named-speaker line?",
            ],
        }
        chats = [first, revised]
        observed_payloads = []

        def fake_ollama(path, body, *, timeout):
            if path == "/api/tags":
                return {"models": []}
            self.assertEqual(path, "/api/chat")
            observed_payloads.append(body)
            content = chats.pop(0)
            return {"message": {"content": json.dumps(content)}}

        with (
            patch.object(coach, "_start_private_ollama", return_value=object()),
            patch.object(coach, "_stop_private_ollama") as stop,
            patch.object(coach, "_select_ollama_model", return_value="qwen-test"),
            patch.object(coach, "_ollama_json", side_effect=fake_ollama),
        ):
            result = coach._run_ollama_agent(validated, analysis)

        self.assertEqual(len(observed_payloads), 2)
        correction = json.loads(observed_payloads[1]["messages"][-1]["content"])
        self.assertIn("skeletal", " ".join(correction["audit_failures"]))
        self.assertIn("SUBJECT_1", correction["creator_contract"])
        self.assertNotIn("Arden Vale", correction["creator_contract"])
        self.assertEqual(
            correction["opaque_identity_labels"],
            ["SUBJECT_1", "SUBJECT_2", "SUBJECT_3"],
        )
        self.assertNotIn("rejected_candidate", correction)
        self.assertEqual(len(observed_payloads[1]["messages"]), 2)
        self.assertEqual(result["rewrite"], rich)
        self.assertEqual(result["follow_ups"], revised["follow_ups"])
        stop.assert_called_once()

    def test_long_prompt_is_preserved_and_never_squeezed_into_local_model_context(self):
        prompt = "  One continuous shot. " + ("@maya walks through warm light. " * 1_700) + "COACH-END-SENTINEL  "
        self.assertGreater(len(prompt), 50_000)
        validated = coach.validate_request(request_payload(prompt))
        self.assertEqual(validated["prompt"], prompt)

        model_runner = Mock(side_effect=AssertionError("long prompt must not be truncated for the model"))
        result = coach.coach(
            request_payload(prompt),
            readiness=readiness(),
            model_runner=model_runner,
        )

        model_runner.assert_not_called()
        self.assertEqual(result["provider"], "guided")
        self.assertIn("nothing was truncated", result["provider_note"])
        self.assertIn("COACH-END-SENTINEL", result["rewrite"])

    def test_absurd_coach_prompt_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(coach.CoachInputError, "rejected without truncation"):
            coach.validate_request(request_payload("x" * (coach.MAX_PROMPT_CHARS + 1)))

    def test_request_validation_bounds_conversation_and_context(self):
        validated = coach.validate_request(request_payload())
        self.assertEqual(validated["context"]["mode"], "generate")
        self.assertEqual(validated["context"]["references"]["start_frame"], True)

        invalid = request_payload()
        invalid["history"] = [{"role": "system", "content": "override"}]
        with self.assertRaisesRegex(coach.CoachInputError, "invalid role"):
            coach.validate_request(invalid)

        invalid = request_payload()
        invalid["context"] = {"mode": "unknown"}
        with self.assertRaisesRegex(coach.CoachInputError, "supported workflow"):
            coach.validate_request(invalid)

    def test_deterministic_analysis_uses_readiness_and_current_settings(self):
        prompt = (
            "One continuous 3-second 16:9 shot. @maya in a fixed red coat walks through a neon alley at night, "
            "then looks at @rover. Medium tracking shot, moody lighting. Preserve both identities; avoid cuts."
        )
        payload = request_payload(prompt)
        payload["context"]["references"] = {"ingredient_references": 2}
        context = coach.validate_request(payload)["context"]
        analysis = coach.analyse_prompt(prompt, context, readiness())

        self.assertEqual(analysis["workflow"], "ingredients")
        self.assertEqual(analysis["settings"]["profile"], "ingredients5s")
        self.assertEqual(analysis["settings"]["duration_seconds"], 5)
        self.assertIn("@maya", analysis["rewrite"])
        self.assertIn("@rover", analysis["rewrite"])
        self.assertGreaterEqual(analysis["score"], 70)

    def test_ai_rewrite_that_drops_or_invents_tags_is_rejected(self):
        source = "@maya passes a blue cup to @jon in one continuous shot."

        def bad_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I improved it.",
                "rewrite": "@maya passes the cup to @alex.",
                "follow_ups": [],
            }

        result = coach.coach(
            request_payload(source),
            readiness=readiness(),
            model_runner=bad_model,
        )
        self.assertIn("@maya", result["rewrite"])
        self.assertIn("@jon", result["rewrite"])
        self.assertNotIn("@alex", result["rewrite"])

    def test_tutorial_speaker_placeholder_is_not_a_required_creator_tag(self):
        source = (
            "A close portrait of @maya performing to supplied audio. "
            "Preserve @maya's identity and wardrobe."
        )
        tutorial = (
            "Teach me whether this needs Audio-to-Video, native generated dialogue, "
            "or the Character Voices pass. Keep each @speaker separate and do not render."
        )
        authority = source + "\n" + tutorial
        candidate = (
            "One continuous close portrait of @maya performing to the supplied audio. "
            "Preserve @maya's identity and wardrobe."
        )
        fallback = "verified fallback"

        self.assertEqual(
            coach._stable_creator_tags(source, authority),
            ["@maya", "@maya"],
        )
        self.assertEqual(coach._stable_named_phrases(authority), [])
        self.assertEqual(
            coach._preserve_prompt_tags(
                candidate,
                source,
                fallback,
                creator_contract=authority,
            ),
            candidate,
        )
        self.assertEqual(
            coach._preserve_prompt_tags(
                "One continuous close portrait of a performer.",
                source,
                fallback,
                creator_contract=authority,
            ),
            fallback,
        )

    def test_workflow_labels_are_not_extracted_as_named_characters(self):
        tutorial = (
            "Teach me whether this prompt needs Audio-to-Video, native generated dialogue, "
            "or the Character Voices pass. Compare Motion Track and Union Control too."
        )

        self.assertEqual(coach._stable_named_phrases(tutorial), [])

    def test_ai_rewrite_must_preserve_duplicate_tag_count_and_order(self):
        source = "@maya speaks, then @maya hands the key to @jon."

        def reordered_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "Done.",
                "rewrite": "@maya speaks to @jon, then @maya hands over the key.",
                "follow_ups": [],
            }

        result = coach.coach(request_payload(source), readiness=readiness(), model_runner=reordered_model)
        self.assertEqual(
            coach.re.findall(r"@[A-Za-z][A-Za-z0-9_]*", result["rewrite"]),
            ["@maya", "@maya", "@jon"],
        )

    def test_ai_conversation_survives_but_unsafe_rewrite_fails_closed(self):
        source = (
            "One continuous 3-second 16:9 shot. @maya, a woman in a red coat, walks through a neon alley at night. "
            "Locked medium shot with cinematic lighting. Preserve her face and wardrobe; avoid cuts."
        )
        unsafe_rewrite = "A 10-second 9:16 montage. @maya jumps, fights, and runs through three locations."

        def unsafe_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "The shot is strong, but keep the action physically simple.",
                "rewrite": unsafe_rewrite,
                "follow_ups": [],
            }

        result = coach.coach(request_payload(source), readiness=readiness(), model_runner=unsafe_model)
        expected = coach.analyse_prompt(
            source,
            coach.validate_request(request_payload(source))["context"],
            readiness(),
        )["rewrite"]

        self.assertEqual(result["provider"], "test-ai")
        self.assertEqual(result["answer"], "The shot is strong, but keep the action physically simple.")
        self.assertEqual(result["rewrite"], expected)
        self.assertNotEqual(result["rewrite"], unsafe_rewrite)
        self.assertNotIn("10-second", result["rewrite"])
        self.assertNotIn("montage", result["rewrite"].lower())
        self.assertEqual(result["rewrite_analysis"]["settings"]["duration_seconds"], 3)
        self.assertEqual(result["rewrite_analysis"]["settings"]["aspect_ratio"], "16:9")
        self.assertIn("not applied", result["rewrite_safety_note"])
        self.assertIn("verified safe draft", result["rewrite_safety_note"])

    def test_active_media_compute_uses_guided_fallback_without_model_call(self):
        model_runner = Mock(side_effect=AssertionError("must not load a model"))
        result = coach.coach(
            request_payload(),
            readiness=readiness(),
            allow_model=False,
            model_runner=model_runner,
        )
        self.assertEqual(result["provider"], "guided")
        self.assertIn("paused while local media compute is active", result["provider_note"])
        model_runner.assert_not_called()

    def test_guided_hallucination_followup_uses_prompt_specific_risks(self):
        prompt = (
            "A 10-second shot of a gymnast fighting in a crowd, then running at night under cinematic light. "
            "Tracking close-up camera, fixed red uniform; avoid cuts."
        )
        payload = request_payload(prompt)
        payload["message"] = "How can I reduce hallucinations further?"
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(result["provider"], "guided")
        self.assertIn("cannot eliminate", result["answer"])
        self.assertIn("For this prompt", result["answer"])
        self.assertTrue("Crowds" in result["answer"] or "complex take" in result["answer"])
        self.assertIn("change only one variable", result["answer"])
        self.assertNotIn("Follow the exact Studio walkthrough below", result["answer"])

    def test_audio_tutorial_question_routes_to_direct_contextual_guidance(self):
        prompt = (
            "A close portrait of @maya performing to the supplied audio on a softly lit stage. "
            "The audio owns timing and visible rhythm; the camera stays locked while her expression "
            "and subtle head movement follow the performance. Preserve @maya's identity and wardrobe. "
            "Avoid cuts, extra voices and background performers."
        )
        payload = request_payload(prompt)
        payload["message"] = (
            "Teach me whether this prompt needs Audio-to-Video, native generated dialogue, "
            "or the Character Voices pass. Keep each @speaker separate and do not render."
        )
        payload["context"].update(
            {"mode": "generate", "profile": "draft", "references": {}}
        )
        mode_readiness = readiness()
        mode_readiness["a2v"] = {
            "ready": False,
            "blocked": True,
            "note": "verified Q8 audio assets are missing",
        }
        model_runner = Mock(side_effect=AssertionError("known workflow guidance must not load a model"))

        result = coach.coach(
            payload,
            readiness=mode_readiness,
            model_runner=model_runner,
        )

        model_runner.assert_not_called()
        self.assertEqual(result["provider"], "workflow-guidance")
        self.assertEqual(result["response_kind"], "workflow_guidance")
        self.assertEqual(result["semantic_best"], "a2v")
        self.assertEqual(result["workflow"], "generate")
        self.assertIn("A visible main action is described.", result["strengths"])
        self.assertIn("Camera framing or movement is specified.", result["strengths"])
        self.assertIn("Setting and lighting/style are both defined.", result["strengths"])
        self.assertNotIn(
            "Choose one framing and one camera move—or a locked camera.",
            result["improvements"],
        )
        self.assertIn("Audio-to-Video own the picture timing", result["answer"])
        self.assertIn("contains no authored spoken line", result["answer"])
        self.assertIn("Character Voices as a separate pass", result["answer"])
        self.assertIn("Use @maya", result["answer"])
        self.assertIn("@speaker", result["answer"])
        self.assertIn("currently blocked", result["answer"])
        self.assertIn("will not follow exact beats or phonemes", result["answer"])
        self.assertIn("No render or media job was started", result["answer"])
        self.assertNotIn("% ready", result["answer"])
        self.assertIn("final audio", result["follow_ups"][0])
        self.assertIn("real @character tags", result["follow_ups"][2])
        self.assertEqual(result["rewrite"], prompt)
        self.assertNotIn("@speaker", result["rewrite"])

    def test_audio_guidance_reports_loaded_eligible_track_without_score_first_copy(self):
        prompt = (
            "@maya performs to the supplied audio; the audio owns timing and visible rhythm. "
            "Locked close-up on a softly lit stage; preserve @maya and avoid cuts."
        )
        payload = request_payload(prompt)
        payload["message"] = (
            "Compare Audio-to-Video, native dialogue, and Character Voices for this prompt."
        )
        payload["context"]["references"] = {"audio": True}

        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(result["response_kind"], "workflow_guidance")
        self.assertEqual(result["workflow"], "a2v")
        self.assertTrue(
            any("supplied audio owns timing" in item.lower() for item in result["strengths"])
        )
        self.assertIn("audio file is loaded", result["answer"])
        self.assertIn("eligible for submit-time preflight", result["answer"])
        self.assertNotIn("prompt is", result["answer"].lower())

    def test_image_creation_question_routes_to_frame_composer_guidance(self):
        payload = request_payload("How do you create an image?")
        payload["message"] = "How do you create an image?"
        model_runner = Mock(side_effect=AssertionError("Studio help must not load a model"))

        result = coach.coach(payload, readiness=readiness(), model_runner=model_runner)

        model_runner.assert_not_called()
        self.assertEqual(result["provider"], "workflow-guidance")
        self.assertEqual(result["response_kind"], "workflow_guidance")
        self.assertIn("Use Frame Composer", result["answer"])
        self.assertIn("one to four reference photos", result["answer"])
        self.assertIn("Start, End, or Interior frame", result["answer"])
        self.assertIn("not a blank text-only image generator", result["answer"])
        self.assertIn("No image or media job was started", result["answer"])
        self.assertEqual(result["rewrite"], payload["prompt"])
        self.assertIn("final aspect ratio", result["follow_ups"][2])

    def test_audio_workflow_question_does_not_authorize_literal_speaker_tag(self):
        payload = request_payload("@maya performs on a softly lit stage while the audio owns timing.")
        payload["history"] = [
            {
                "role": "user",
                "content": (
                    "Teach me whether this needs Audio-to-Video or Character Voices. "
                    "Keep each @speaker separate."
                ),
            },
            {"role": "user", "content": "Keep @jon silent in the background."},
        ]
        payload["message"] = "Review the updated shot."

        contract = coach._creator_authored_contract(coach.validate_request(payload))

        self.assertIn("@maya", contract)
        self.assertIn("@jon", contract)
        self.assertNotIn("@speaker", contract)

    def test_guided_why_followup_explains_verified_workflow_for_this_prompt(self):
        prompt = (
            "One continuous shot with multiple reference images for @maya and @rover. "
            "They walk through a neon alley at night in fixed wardrobe; locked medium camera, avoid cuts."
        )
        payload = request_payload(prompt)
        payload["message"] = "Why is this workflow better than Generate?"
        payload["context"]["references"] = {"ingredient_references": 2}
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(result["workflow"], "ingredients")
        self.assertIn("multiple identity/object references must be preserved", result["answer"])
        self.assertIn("Ingredients is installed and eligible for submit-time preflight", result["answer"])
        self.assertIn("ingredients5s", result["answer"])
        self.assertEqual(
            result["follow_ups"][0],
            "Why is Ingredients better than Generate for this prompt?",
        )

    def test_guided_settings_followup_returns_exact_prompt_specific_steps(self):
        prompt = (
            "One continuous 3-second 9:16 shot. @maya in a fixed red coat walks through a neon alley at night. "
            "Locked medium camera with cinematic lighting; preserve her identity and avoid cuts."
        )
        payload = request_payload(prompt)
        payload["message"] = "Walk me through these settings."
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertIn("quality", result["answer"])
        self.assertIn("3 seconds", result["answer"])
        self.assertIn("9:16", result["answer"])
        self.assertIn("seed 42", result["answer"])
        self.assertIn("Select Generate", result["answer"])
        self.assertIn("Character Bible", result["answer"])

    def test_guided_followup_keeps_exact_tag_order_and_duplicate_count(self):
        prompt = '@maya says "Wait", then @maya hands a key to @jon in one continuous shot; avoid cuts.'
        payload = request_payload(prompt)
        payload["message"] = "How can I reduce hallucinations for this prompt?"
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertEqual(
            coach.re.findall(r"@[A-Za-z][A-Za-z0-9_]*", result["rewrite"]),
            ["@maya", "@maya", "@jon"],
        )

    def test_model_not_ready_has_truthful_os_managed_action(self):
        def not_ready(_request, _analysis):
            raise RuntimeError("Apple Foundation Models system model is not ready (modelNotReady)")

        result = coach.coach(
            request_payload(),
            readiness=readiness(),
            model_runner=not_ready,
        )
        self.assertEqual(result["provider"], "guided")
        self.assertIn("on-device conversational model gets ready", result["provider_note"])
        self.assertNotIn("modelNotReady", result["provider_note"])
        self.assertNotIn("RuntimeError", result["provider_note"])
        self.assertIn("Siri Language", result["provider_action"])
        self.assertIn("English (India)", result["provider_action"])
        self.assertIn("powered", result["provider_action"])
        self.assertIn("on Wi-Fi", result["provider_action"])
        self.assertIn("automatic model download", result["provider_action"])
        self.assertNotIn("restart", result["provider_action"].lower())

    def test_existing_duration_and_aspect_inside_one_shot_is_not_duplicated(self):
        source = (
            "One continuous 3-second 16:9 shot. @maya, a woman in a red coat, walks through a neon alley at night. "
            "Locked medium shot with cinematic lighting. Preserve her face and wardrobe; avoid cuts."
        )
        context = coach.validate_request(request_payload(source))["context"]
        rewrite = coach.analyse_prompt(source, context, readiness())["rewrite"]

        self.assertEqual(rewrite.lower().count("one continuous 3-second 16:9 shot."), 1)
        self.assertTrue(rewrite.startswith("One continuous 3-second 16:9 shot."))

    def test_apple_fm_uses_stdin_secure_schema_file_and_cleans_it(self):
        payload = request_payload("@maya turns toward the window in a locked medium shot")
        validated = coach.validate_request(payload)
        analysis = coach.analyse_prompt(validated["prompt"], validated["context"], readiness())
        generated = {
            "answer": "Use Generate with a short locked shot.",
            "rewrite": "One continuous shot. @maya turns toward the window. Locked medium camera.",
            "follow_ups": ["Why a locked camera?"],
        }
        observed: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            observed["argv"] = list(argv)
            observed["input"] = kwargs.get("input")
            schema_path = Path(argv[argv.index("--schema") + 1])
            observed["schema_path"] = schema_path
            observed["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
            observed["schema_mode"] = schema_path.stat().st_mode & 0o777
            return SimpleNamespace(returncode=0, stdout=json.dumps(generated), stderr="")

        with (
            patch.object(coach, "_apple_fm_executable", return_value="/usr/bin/fm"),
            patch.object(coach, "_apple_fm_license_ready"),
            patch.object(coach, "_apple_fm_available"),
            patch.object(coach.subprocess, "run", side_effect=fake_run) as run,
        ):
            result = coach._run_apple_agent(validated, analysis)

        run.assert_called_once()
        argv = observed["argv"]
        self.assertEqual(argv[:4], ["/usr/bin/fm", "respond", "-m", "system"])
        self.assertNotIn(validated["prompt"], argv)
        self.assertIn(validated["prompt"], observed["input"])
        self.assertEqual(observed["schema_mode"], 0o600)
        self.assertFalse(observed["schema_path"].exists())
        self.assertEqual(observed["schema"]["required"], ["answer", "rewrite", "follow_ups"])
        self.assertEqual(result["provider"], "apple-foundation-model")

    def test_apple_fm_samples_creative_development_but_keeps_review_deterministic(self):
        def command_for(payload):
            validated = coach.validate_request(payload)
            analysis = coach.analyse_prompt(
                validated["prompt"], validated["context"], readiness()
            )
            observed: list[str] = []

            def fake_run(argv, **_kwargs):
                observed.extend(argv)
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "answer": "Grounded local advice.",
                            "rewrite": analysis["rewrite"],
                            "follow_ups": [],
                        }
                    ),
                    stderr="",
                )

            with (
                patch.object(coach, "_apple_fm_executable", return_value="/usr/bin/fm"),
                patch.object(coach, "_apple_fm_license_ready"),
                patch.object(coach, "_apple_fm_available"),
                patch.object(coach.subprocess, "run", side_effect=fake_run),
            ):
                coach._run_apple_agent(validated, analysis)
            return observed

        develop = request_payload("do you know festival races? I gotta find Rowan Kite.")
        develop["message"] = "Expand this idea and build a complete production prompt."
        review = request_payload(
            "One continuous 3-second 16:9 shot. A brass robot in a fixed red jacket turns "
            "toward a window in a warm studio, then holds still. Locked medium camera, soft "
            "directional cinematic light; preserve its geometry and wardrobe, avoid cuts and duplicates."
        )

        self.assertNotIn("--greedy", command_for(develop))
        self.assertIn("--greedy", command_for(review))

    def test_apple_schema_tempfile_error_becomes_safe_fallback_error(self):
        payload = coach.validate_request(request_payload())
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())
        with (
            patch.object(coach, "_apple_fm_executable", return_value="/usr/bin/fm"),
            patch.object(coach, "_apple_fm_license_ready"),
            patch.object(coach, "_apple_fm_available"),
            patch.object(coach.tempfile, "NamedTemporaryFile", side_effect=OSError("disk unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "private schema file"):
                coach._run_apple_agent(payload, analysis)

    def test_ollama_disables_redirects_and_environment_proxies(self):
        response = Mock()
        response.read.return_value = b'{"models":[]}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(coach, "build_opener", return_value=opener) as build:
            self.assertEqual(coach._ollama_json("/api/tags", None, timeout=1), {"models": []})
        handlers = build.call_args.args
        proxy = next(item for item in handlers if isinstance(item, ProxyHandler))
        redirect = next(item for item in handlers if isinstance(item, coach.HTTPRedirectHandler))
        self.assertEqual(proxy.proxies, {})
        self.assertIsNone(redirect.redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_dialogue_speaker_tags_without_references_do_not_trigger_ingredients(self):
        prompt = '@maya says "Wait." Then @jon says "Go." One continuous medium shot in a lit room; avoid cuts.'
        payload = request_payload(prompt)
        payload["context"]["references"] = {}
        context = coach.validate_request(payload)["context"]
        analysis = coach.analyse_prompt(prompt, context, readiness())
        self.assertEqual(analysis["workflow"], "generate")
        self.assertEqual(analysis["score"], 92)
        self.assertTrue(any("identity is still loose" in item for item in analysis["strengths"]))
        self.assertTrue(any("Character Voice" in step for step in analysis["walkthrough"]))

    def test_natural_action_morphology_locked_shot_and_wardrobe_are_understood(self):
        context = coach.validate_request(request_payload())["context"]
        for phrase in (
            "A red ball is rolling across the room in a locked shot.",
            "A red ball rolled across the room in a locked shot.",
            "A dog was running through the garden in a locked shot.",
            "A dog ran through the garden in a locked shot.",
            "A woman in a red coat walked through the street in a locked shot.",
        ):
            with self.subTest(phrase=phrase):
                signals = coach._signals(phrase, context)
                self.assertTrue(signals["subject_basic"])
                self.assertTrue(signals["action"])
                self.assertTrue(signals["camera"])
        self.assertTrue(coach._signals("A woman in a red coat walked away.", context)["identity_detail"])

    def test_adjective_only_slogan_is_not_credited_as_a_subject(self):
        prompt = "Beautiful magical cinematic emotional masterpiece."
        payload = request_payload(prompt)
        payload["context"]["references"] = {}
        context = coach.validate_request(payload)["context"]
        analysis = coach.analyse_prompt(prompt, context, readiness())
        self.assertTrue(any("Name the main subject" in item for item in analysis["improvements"]))
        self.assertFalse(any("subject" in item.lower() for item in analysis["strengths"]))

    def test_model_instruction_makes_constraints_authoritative_but_semantics_advisory(self):
        validated = coach.validate_request(request_payload())
        analysis = coach.analyse_prompt(validated["prompt"], validated["context"], readiness())
        system = coach._model_messages(validated, analysis)[0]["content"]
        self.assertIn("hard workflow constraints", system)
        self.assertIn("semantic score, strengths and risks are advisory", system)

    def test_ingredients_forces_trained_aspect_and_effective_five_second_rewrite(self):
        prompt = "One continuous 3-second 9:16 shot. @maya walks toward camera with fixed wardrobe in neon light; avoid cuts."
        payload = request_payload(prompt)
        payload["context"]["mode"] = "ingredients"
        payload["context"]["profile"] = "ingredients5s"
        payload["context"]["aspect_ratio"] = "9:16"
        payload["context"]["references"] = {"ingredient_references": 1}
        context = coach.validate_request(payload)["context"]
        analysis = coach.analyse_prompt(prompt, context, readiness())
        self.assertEqual(analysis["workflow"], "ingredients")
        self.assertEqual(analysis["settings"]["aspect_ratio"], "profile")
        self.assertEqual(analysis["settings"]["duration_seconds"], 5)
        self.assertIn("use 5 seconds and the profile-default trained-bucket aspect", analysis["rewrite"])
        self.assertNotIn("9:16", analysis["rewrite"])

    def test_complex_ten_second_prompt_rewrite_uses_effective_three_seconds(self):
        prompt = (
            "A 10-second shot of a gymnast fighting, then running, finally jumping at night with cinematic light, "
            "tracking close-up camera, fixed red wardrobe; avoid cuts."
        )
        context = coach.validate_request(request_payload(prompt))["context"]
        analysis = coach.analyse_prompt(prompt, context, readiness())
        self.assertEqual(analysis["settings"]["duration_seconds"], 3)
        self.assertIn("3-second", analysis["rewrite"])
        self.assertNotRegex(analysis["rewrite"], r"\b10[- ]?seconds?\b")

    def test_no_runnable_workflow_does_not_advise_rendering(self):
        unavailable = {
            mode: {"ready": False, "blocked": True, "note": "missing"}
            for mode in coach.WORKFLOW_MODES
        }
        payload = coach.validate_request(request_payload())
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], unavailable)
        self.assertEqual(analysis["workflow"], "unavailable")
        self.assertFalse(any("render a short test" in step.lower() for step in analysis["walkthrough"]))

    def test_ui_endpoint_passes_actual_workflow_readiness_and_busy_gate(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        mode_readiness = readiness()
        response = {"provider": "guided", "answer": "ok"}
        with (
            patch.object(ui, "_gemma_pack_readiness", return_value={"auto": {"ready": True}}) as gemma,
            patch.object(ui, "_workflow_readiness", return_value=mode_readiness) as workflows,
            patch.object(ui.STATE, "is_busy", return_value=False),
            patch.object(ui.FRAME_STATE, "is_busy", return_value=False),
            patch.object(ui.VOICE_STATE, "is_busy", return_value=False),
            patch.object(ui.UPSCALE_STATE, "is_busy", return_value=False),
            patch.object(ui, "_claim_prompt_coach", return_value=True) as claim,
            patch.object(ui, "_release_prompt_coach") as release,
            patch.object(ui.prompt_coach_backend, "coach", return_value=response) as run,
        ):
            handler._handle_prompt_coach(request_payload())

        gemma.assert_called_once_with()
        workflows.assert_called_once_with({"auto": {"ready": True}})
        claim.assert_called_once_with()
        release.assert_called_once_with()
        run.assert_called_once_with(request_payload(), readiness=mode_readiness, allow_model=True, pause_reason="")
        handler._send_json.assert_called_once_with(HTTPStatus.OK, response)

    def test_active_coach_makes_new_compute_fail_fast(self):
        handler = object.__new__(ui.LTXRequestHandler)
        handler._send_json = Mock()
        with (
            patch.object(ui, "_prompt_coach_is_active", return_value=True),
            patch.object(ui.STATE, "is_busy", return_value=False),
            patch.object(ui.FRAME_STATE, "is_busy", return_value=False),
            patch.object(ui.VOICE_STATE, "is_busy", return_value=False),
            patch.object(ui.UPSCALE_STATE, "start") as start,
        ):
            with self.assertRaisesRegex(ui.APIError, "Prompt Coach is finishing") as raised:
                handler._handle_upscale({})
        self.assertEqual(raised.exception.status, HTTPStatus.CONFLICT)
        start.assert_not_called()

    def test_page_exposes_conversational_coach_and_backend_route(self):
        for element_id in (
            "promptCoachConversation",
            "promptCoachMessage",
            "promptCoachSend",
            "promptCoachProvider",
            "promptCoachProviderAction",
        ):
            self.assertIn(f'id="{element_id}"', ui.PAGE)
        self.assertIn('apiUi("/api/prompt-coach"', ui.PAGE)
        self.assertIn("boundedPromptCoachHistoryUi", ui.PAGE)
        self.assertIn("resetPromptCoachHistoryUi", ui.PAGE)
        self.assertIn("Prompt Coach applied a complete draft", ui.PAGE)
        self.assertNotIn("Replace any [bracketed] choices", ui.PAGE)


if __name__ == "__main__":
    unittest.main()
