from __future__ import annotations

import json
import unittest

import prompt_coach as coach
import prompt_coach_skill as skill


def readiness(**overrides: bool) -> dict[str, dict[str, object]]:
    """Build a complete server-owned readiness snapshot for focused tests."""

    result: dict[str, dict[str, object]] = {}
    for mode in coach.WORKFLOW_MODES:
        ready = overrides.get(mode, True)
        result[mode] = {
            "ready": ready,
            "blocked": not ready,
            "note": f"{mode} verified and ready" if ready else f"{mode} assets are missing",
        }
    return result


def request_payload(
    prompt: str,
    *,
    mode: str = "generate",
    references: dict[str, int | bool] | None = None,
    duration: int = 3,
) -> dict[str, object]:
    profile = {
        "ingredients": "ingredients5s",
        "motion": "control3s",
        "union": "control3s",
    }.get(mode, "quality")
    return {
        "message": "Review this prompt and recommend the grounded Studio workflow.",
        "prompt": prompt,
        "history": [],
        "context": {
            "mode": mode,
            "profile": profile,
            "aspect_ratio": "16:9",
            "duration_seconds": duration,
            "seed": 42,
            "references": references or {},
        },
    }


def analyse(
    prompt: str,
    *,
    mode: str = "generate",
    references: dict[str, int | bool] | None = None,
    duration: int = 3,
    mode_readiness: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    payload = request_payload(prompt, mode=mode, references=references, duration=duration)
    validated = coach.validate_request(payload)
    return coach.analyse_prompt(
        validated["prompt"],
        validated["context"],
        mode_readiness or readiness(),
    )


class PromptCoachSkillRegressionTests(unittest.TestCase):
    def test_real_skill_package_is_versioned_and_injected(self):
        instructions = skill.load_skill_instructions()
        self.assertEqual(skill.SKILL_VERSION, "1.4.0")
        self.assertTrue(skill.SKILL_PACKAGE_DIR.is_dir())
        self.assertIn("# LTX-2.5 Prompt Coach", instructions)
        self.assertIn("Recommendation certainty", instructions)

        modes = readiness()
        payload = coach.validate_request(request_payload("A red ball rolls across a lit studio."))
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], modes)
        apple_payload = json.loads(coach._apple_prompt(payload, analysis))

        self.assertEqual(apple_payload["ltx_production_skill"]["version"], "1.4.0")
        self.assertEqual(
            apple_payload["ltx_production_skill"]["skill_entrypoint"],
            instructions,
        )

    def test_v14_skill_distinguishes_creative_development_from_conservative_review(self):
        instructions = " ".join(skill.load_skill_instructions().split()).lower()

        self.assertIn("development mode", instructions)
        self.assertTrue(
            any(term in instructions for term in ("sparse", "open-ended", "rough idea")),
            instructions,
        )
        self.assertIn("coherent", instructions)
        self.assertIn("concrete", instructions)
        self.assertIn("square-bracket placeholders", instructions)
        self.assertIn("named character", instructions)
        self.assertIn("dialogue", instructions)
        self.assertIn("lore", instructions)
        self.assertRegex(instructions, r"follow-up questions?.{0,80}optional")

    def test_first_last_frame_reference_is_scoped_and_injected_conditionally(self):
        reference = skill.load_first_last_frame_reference()
        self.assertIn("# First-frame / last-frame prompting reference", reference)
        self.assertIn("does not inspect the image pixels", " ".join(reference.split()))
        for project_specific_name in ("ProjectOnlyNameA", "ProjectOnlyNameB", "ProjectOnlyNameC", "ProjectOnlyNameD", "ProjectOnlyNameE"):
            self.assertNotIn(project_specific_name, reference)

        both = coach.validate_request(
            request_payload(
                "A woman turns toward a window in one continuous shot.",
                references={"start_frame": True, "end_frame": True},
            )
        )
        both_analysis = coach.analyse_prompt(both["prompt"], both["context"], readiness())
        both_context = skill.build_skill_context(both, both_analysis)
        conditional = both_context["conditional_reference"]
        self.assertEqual(conditional["name"], "first-last-frame")
        self.assertTrue(conditional["loaded_start_frame"])
        self.assertTrue(conditional["loaded_end_frame"])
        self.assertEqual(conditional["instructions"], reference)

        ollama_report = json.loads(
            coach._model_messages(both, both_analysis)[1]["content"].split("\n", 1)[1]
        )
        apple_report = json.loads(coach._apple_prompt(both, both_analysis))
        self.assertEqual(
            ollama_report["ltx_production_skill"]["conditional_reference"],
            apple_report["ltx_production_skill"]["conditional_reference"],
        )

        start_only = coach.validate_request(
            request_payload(
                "Use the supplied Start frame for a slow turn.",
                references={"start_frame": True},
            )
        )
        start_analysis = coach.analyse_prompt(
            start_only["prompt"], start_only["context"], readiness()
        )
        self.assertNotIn(
            "conditional_reference",
            skill.build_skill_context(start_only, start_analysis),
        )

        discussed = coach.validate_request(
            request_payload("How should Start and End act as the sequence boundaries?")
        )
        discussed_analysis = coach.analyse_prompt(
            discussed["prompt"], discussed["context"], readiness()
        )
        discussed_reference = skill.build_skill_context(discussed, discussed_analysis)[
            "conditional_reference"
        ]
        self.assertFalse(discussed_reference["loaded_start_frame"])
        self.assertFalse(discussed_reference["loaded_end_frame"])

    def test_dynamic_readiness_is_embedded_in_the_model_skill_context(self):
        modes = readiness(ingredients=False)
        payload = coach.validate_request(
            request_payload(
                "Use multiple reference images for @subject_a and @subject_b in a neon room.",
                mode="ingredients",
                references={"ingredient_references": 2},
            )
        )
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], modes)

        apple_payload = json.loads(coach._apple_prompt(payload, analysis))
        skill = apple_payload["ltx_production_skill"]

        self.assertEqual(analysis["workflow_readiness"]["ingredients"]["availability"], "blocked")
        self.assertEqual(skill["local_workflow_status"]["ingredients"]["availability"], "blocked")
        self.assertIn("ingredients assets are missing", skill["local_workflow_status"]["ingredients"]["note"])
        self.assertEqual(
            skill["local_workflow_status"]["ingredients"],
            analysis["workflow_readiness"]["ingredients"],
        )

    def test_start_image_i2v_does_not_require_identity_redescription(self):
        result = analyse(
            "The woman turns toward the window as the camera slowly pushes in through warm bedroom light.",
            references={"start_frame": True},
        )

        improvements = " ".join(result["improvements"]).lower()
        self.assertNotIn("fixed identity", improvements)
        self.assertNotIn("face/object details", improvements)
        self.assertNotIn("subject continuity:", result["rewrite"].lower())

    def test_loaded_start_and_end_frames_get_a_grounded_bridge_plan(self):
        prompt = (
            "A woman in a red coat begins facing the window, then slowly turns toward camera in warm bedroom light. "
            "Locked medium shot; preserve her identity and avoid duplicates."
        )
        result = analyse(
            prompt,
            references={"start_frame": True, "end_frame": True},
        )

        self.assertTrue(result["boundary_frame_guidance"]["active"])
        self.assertIn("Start frame as the exact opening composition", result["rewrite"])
        self.assertIn("End frame as the exact final composition", result["rewrite"])
        self.assertIn("smallest physically plausible continuous changes", result["rewrite"])
        self.assertNotIn("Subject continuity: [", result["rewrite"])
        walkthrough = " ".join(result["walkthrough"])
        self.assertIn("authoritative opening and final targets", walkthrough)
        self.assertIn("cannot inspect their pixels", " ".join(result["risks"]))

        payload = request_payload(
            prompt,
            references={"start_frame": True, "end_frame": True},
        )
        payload["message"] = "How should I use my loaded Start and End as the boundaries?"
        guided = coach.coach(payload, readiness=readiness(), allow_model=False)
        self.assertIn("authoritative endpoint targets", guided["answer"])
        self.assertIn("cannot inspect their pixels", guided["answer"])

    def test_boundary_pair_camera_lock_and_intentional_move_contract(self):
        references = {"start_frame": True, "end_frame": True}

        locked = analyse(
            "A woman begins neutral, then smiles in a warm room. Medium shot; preserve identity and avoid cuts.",
            references=references,
        )
        locked_guidance = locked["boundary_frame_guidance"]
        self.assertTrue(locked_guidance["camera_lock_default"])
        self.assertFalse(locked_guidance["deliberate_camera_move"])
        self.assertIn("camera position and height", locked["rewrite"])
        self.assertIn("lens/field of view", locked["rewrite"])
        self.assertIn("subject-to-camera distance", locked["rewrite"])
        self.assertIn("subject scale", locked["rewrite"])
        self.assertNotIn("Camera: [", locked["rewrite"])

        unquantified = analyse(
            "A woman begins neutral, then smiles during a slow push-in in a warm room. Preserve identity and avoid cuts.",
            references=references,
        )
        unquantified_guidance = unquantified["boundary_frame_guidance"]
        self.assertFalse(unquantified_guidance["camera_lock_default"])
        self.assertTrue(unquantified_guidance["deliberate_camera_move"])
        self.assertFalse(unquantified_guidance["quantified_camera_move"])
        self.assertNotIn("Keep camera position and height", unquantified["rewrite"])
        self.assertTrue(
            any("quantitatively and chronologically" in item for item in unquantified["improvements"]),
            unquantified["improvements"],
        )

        quantified = analyse(
            "A woman begins neutral, then smiles during a slow 10% push-in over 3 seconds in a warm room. Preserve identity and avoid cuts.",
            references=references,
        )
        quantified_guidance = quantified["boundary_frame_guidance"]
        self.assertTrue(quantified_guidance["deliberate_camera_move"])
        self.assertTrue(quantified_guidance["quantified_camera_move"])
        self.assertIn("10% push-in over 3 seconds", quantified["rewrite"])
        self.assertFalse(
            any("quantitatively and chronologically" in item for item in quantified["improvements"]),
            quantified["improvements"],
        )

    def test_boundary_duration_staging_uses_the_effective_duration(self):
        references = {"start_frame": True, "end_frame": True}
        three = analyse(
            "A robot begins still, then looks up in a lit studio. Locked camera; preserve its shape.",
            references=references,
            duration=3,
        )
        five = analyse(
            "A robot begins still, then looks up in a lit studio. Locked camera; preserve its shape.",
            mode="ingredients",
            references={**references, "ingredient_references": 2},
            duration=3,
        )

        self.assertEqual(three["settings"]["duration_seconds"], 3)
        self.assertIn("For this three-second take", three["rewrite"])
        self.assertIn("brief final hold", three["rewrite"])
        self.assertEqual(five["settings"]["duration_seconds"], 5)
        self.assertIn("For this five-second take", five["rewrite"])
        self.assertIn("readable final hold", five["rewrite"])

    def test_ai_rewrite_cannot_drop_voice_contract_or_silent_character_rule(self):
        prompt = (
            'Exactly one voice exists. @maya, a young adult with a low pitch, warm airy texture, '
            'measured pacing and clear articulation, says "Wait." Never use a nasal voice. '
            '@jon remains completely silent and must not mouth, echo or repeat the line. '
            "They stand in a warm room in a locked medium shot; preserve identity and avoid cuts."
        )
        payload = request_payload(prompt)

        def contract_dropping_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I shortened the direction.",
                "rewrite": (
                    '@maya says "Wait" while @jon watches in a warm room. '
                    "Locked medium shot; preserve identity and avoid cuts."
                ),
                "follow_ups": [],
            }

        result = coach.coach(
            payload,
            readiness=readiness(),
            model_runner=contract_dropping_model,
        )

        self.assertIn("Exactly one voice exists", result["rewrite"])
        self.assertIn("must not mouth", result["rewrite"])
        self.assertIn("voice-count contract", result["rewrite_safety_note"])
        self.assertIn("voice specification", result["rewrite_safety_note"])
        self.assertIn("silent-character", result["rewrite_safety_note"])

    def test_full_injected_skill_context_is_project_agnostic(self):
        payload = coach.validate_request(
            request_payload(
                "A generic robot turns toward a window in one continuous shot.",
                references={"start_frame": True, "end_frame": True},
            )
        )
        analysis = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())
        serialized = json.dumps(
            skill.build_skill_context(payload, analysis),
            ensure_ascii=False,
        )

        for project_specific_name in ("ProjectOnlyNameA", "ProjectOnlyNameB", "ProjectOnlyNameC", "ProjectOnlyNameD", "ProjectOnlyNameE"):
            self.assertNotIn(project_specific_name, serialized)
        self.assertIn("camera position", serialized)
        self.assertIn("lens/field of view", serialized)
        self.assertIn("recurring prose voice specification", serialized)

    def test_model_rewrite_cannot_drop_a_loaded_boundary_contract(self):
        prompt = (
            "A woman in a red coat turns toward the window in warm bedroom light. "
            "Locked medium shot; preserve her identity."
        )
        payload = request_payload(
            prompt,
            references={"start_frame": True, "end_frame": True},
        )

        def boundary_dropping_model(_request, _analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "I simplified the prompt.",
                "rewrite": "A woman in a red coat turns toward the window in warm bedroom light.",
                "follow_ups": [],
            }

        result = coach.coach(
            payload,
            readiness=readiness(),
            model_runner=boundary_dropping_model,
        )

        self.assertIn("Start frame as the exact opening composition", result["rewrite"])
        self.assertIn("End frame as the exact final composition", result["rewrite"])
        self.assertIn("removed the exact Start/End boundary contract", result["rewrite_safety_note"])

    def test_frame_descriptions_without_images_do_not_claim_exact_anchors(self):
        prompt = (
            "Use a first frame of a woman by a window and a last frame where she faces camera. "
            "Warm bedroom light and a locked medium shot."
        )
        payload = request_payload(prompt, references={})
        payload["message"] = "How should I use the first and last frames?"
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertFalse(result["boundary_frame_guidance"]["active"])
        self.assertTrue(result["boundary_frame_guidance"]["discussed"])
        self.assertIn("text descriptions alone do not create exact visual anchors", result["answer"])
        self.assertIn("Upload both actual images", result["answer"])
        self.assertLessEqual(result["rewrite_analysis"]["score"], 89)
        context = coach.validate_request(payload)["context"]
        rewrite_analysis = coach.analyse_prompt(result["rewrite"], context, readiness())
        self.assertTrue(coach._signals(result["rewrite"], context)["unresolved_placeholders"])
        self.assertTrue(
            any("square-bracket placeholder" in item for item in rewrite_analysis["improvements"]),
            rewrite_analysis["improvements"],
        )

    def test_start_only_and_end_only_rewrites_use_the_correct_boundary(self):
        start_only = analyse(
            "A woman turns toward the window in warm light with a locked medium camera.",
            references={"start_frame": True},
        )["rewrite"]
        end_only = analyse(
            "A woman turns toward the window in warm light with a locked medium camera.",
            references={"end_frame": True},
        )["rewrite"]

        self.assertIn("Start frame for appearance and the exact opening composition", start_only)
        self.assertNotIn("End frame as the exact final composition", start_only)
        self.assertIn("End frame as the exact final composition", end_only)
        self.assertNotIn("opening appearance/composition", end_only)

    def test_boundary_pair_preserves_an_intentional_multishot_sequence(self):
        prompt = (
            "An 8-second three-shot sequence. Shot 1: a pilot enters a hangar. "
            "Cut to her hand on the console. Cut to the ship lifting into dawn light."
        )
        result = analyse(
            prompt,
            references={"start_frame": True, "end_frame": True},
            duration=8,
        )

        self.assertTrue(result["boundary_frame_guidance"]["multiple_cuts"])
        self.assertIn("Cut to", result["rewrite"])
        self.assertIn("boundaries of the whole requested sequence", " ".join(result["walkthrough"]))
        self.assertNotIn("One continuous", result["rewrite"])
        self.assertNotIn("a cut, teleport", result["rewrite"])
        context = coach.validate_request(
            request_payload(
                prompt,
                references={"start_frame": True, "end_frame": True},
                duration=8,
            )
        )["context"]
        rewritten = coach.analyse_prompt(result["rewrite"], context, readiness())
        self.assertEqual(rewritten["settings"]["duration_seconds"], 8)

    def test_generated_avoid_clause_does_not_invent_complex_body_motion(self):
        prompt = (
            "An 8-second three-shot sequence. Shot 1: a pilot enters a hangar. "
            "Cut to the glowing console. Cut to the ship lifting into dawn light. "
            "Avoid duplicate subjects, extra limbs or fingers, and identity drift."
        )
        context = coach.validate_request(request_payload(prompt, duration=8))["context"]

        self.assertFalse(coach._signals(prompt, context)["complex_body"])
        self.assertEqual(
            coach.analyse_prompt(prompt, context, readiness())["settings"]["duration_seconds"],
            8,
        )

    def test_a2v_with_supplied_audio_does_not_require_motion_chronology(self):
        result = analyse(
            "Close portrait of a dancer on a neon stage, cinematic blue lighting, locked medium camera.",
            mode="a2v",
            references={"audio": True},
        )

        self.assertEqual(result["workflow"], "a2v")
        improvements = " ".join(result["improvements"]).lower()
        self.assertNotIn("main action", improvements)
        self.assertNotIn("begins", improvements)
        self.assertNotIn("action: [", result["rewrite"].lower())

    def test_motion_with_tracks_warns_against_a_competing_prompt_trajectory(self):
        result = analyse(
            "The red ball moves from the lower left to the upper right, then loops back. "
            "Locked camera in a bright studio.",
            mode="motion",
            references={"start_frame": True, "motion_tracks": 1},
        )

        self.assertEqual(result["workflow"], "motion")
        warnings = " ".join([*result["improvements"], *result["risks"]]).lower()
        self.assertIn("trajectory", warnings)
        self.assertTrue(
            any(term in warnings for term in ("competing", "contradict", "path owns", "tracks own")),
            warnings,
        )

    def test_exact_curved_path_routes_to_motion_track(self):
        result = analyse(
            "A red ball follows an exact curved path across a warm studio while a locked camera watches; "
            "preserve its shape and avoid cuts.",
            references={"start_frame": True, "motion_tracks": 1},
        )

        self.assertEqual(result["semantic_best"], "motion")
        self.assertEqual(result["workflow"], "motion")
        self.assertEqual(result["settings"]["mode"], "motion")
        self.assertTrue(
            any("motion track" in step.lower() for step in result["walkthrough"]),
            result["walkthrough"],
        )

    def test_explicit_multishot_request_is_not_forced_into_one_continuous_shot(self):
        prompt = (
            "An 8-second three-shot sequence. Shot 1: a pilot enters a hangar. "
            "Cut to a close-up of her hand on the console. Cut to the ship lifting into warm dawn light."
        )
        result = analyse(prompt, duration=8)

        self.assertIn("three-shot sequence", result["rewrite"].lower())
        self.assertIn("cut to", result["rewrite"].lower())
        self.assertFalse(result["rewrite"].lower().startswith("one continuous"), result["rewrite"])

    def test_negated_crowd_and_text_do_not_create_positive_risk_signals(self):
        result = analyse(
            "One continuous 3-second shot of a woman walking alone through an empty station at night. "
            "Locked camera, cinematic lighting. No crowd and no readable text; preserve her appearance."
        )

        risks = " ".join(result["risks"]).lower()
        self.assertNotIn("crowd", risks)
        self.assertNotIn("readable text", risks)

    def test_explicit_silence_does_not_invent_dialogue_or_camera_movement(self):
        prompt = (
            "A silent brass robot turns toward a product in a warm studio. Medium shot. "
            "No dialogue, speech, music, or offscreen voices. No camera movement; "
            "preserve geometry and avoid cuts."
        )
        references = {"start_frame": True, "end_frame": True}
        payload = request_payload(prompt, references=references)
        validated = coach.validate_request(payload)
        source_signals = coach._signals(prompt, validated["context"])
        analysis = coach.analyse_prompt(prompt, validated["context"], readiness())
        rewrite_signals = coach._signals(analysis["rewrite"], validated["context"])
        result = coach.coach(payload, readiness=readiness(), allow_model=False)

        self.assertFalse(source_signals["dialogue"])
        self.assertFalse(source_signals["camera_move"])
        self.assertFalse(rewrite_signals["dialogue"])
        self.assertFalse(rewrite_signals["camera_move"])
        self.assertNotIn("Dialogue:", analysis["rewrite"])
        self.assertFalse(
            any("Character Voice cue" in step for step in analysis["walkthrough"]),
            analysis["walkthrough"],
        )
        self.assertEqual(result["rewrite_safety_note"], "")

    def test_lip_syncs_inflection_is_detected_as_audio_driven(self):
        prompt = "A singer lip-syncs the supplied song on a neon stage in a locked medium shot."
        payload = coach.validate_request(request_payload(prompt))

        self.assertTrue(coach._signals(prompt, payload["context"])["audio_driven"])
        result = coach.analyse_prompt(payload["prompt"], payload["context"], readiness())
        self.assertEqual(result["workflow"], "a2v")

    def test_ingredients_flags_invalid_machine_tags_but_accepts_lowercase_tags(self):
        invalid = analyse(
            "@Subject_A and @subject_b walk together through a neon room.",
            mode="ingredients",
            references={"ingredient_references": 2},
        )
        valid = analyse(
            "@subject_a and @subject_b walk together through a neon room.",
            mode="ingredients",
            references={"ingredient_references": 2},
        )

        invalid_findings = " ".join([*invalid["improvements"], *invalid["risks"]]).lower()
        valid_findings = " ".join([*valid["improvements"], *valid["risks"]]).lower()
        self.assertIn("@subject_a", invalid_findings)
        self.assertTrue(
            any(term in invalid_findings for term in ("invalid", "lowercase", "machine-safe")),
            invalid_findings,
        )
        self.assertNotIn("invalid tag", valid_findings)
        self.assertNotIn("must be lowercase", valid_findings)

    def test_ai_claim_that_blocked_workflow_is_ready_uses_grounded_fallback(self):
        prompt = (
            "Use multiple reference images for @subject_a and @subject_b. They walk through a neon room "
            "in fixed wardrobe with a locked medium camera; preserve both identities and avoid cuts."
        )
        payload = request_payload(prompt, references={"ingredient_references": 2})
        modes = readiness(ingredients=False)

        def ungrounded_model(_request, deterministic_analysis):
            return {
                "provider": "test-ai",
                "provider_label": "Test AI",
                "provider_note": "test",
                "provider_action": "",
                "answer": "Ingredients is ready now, so select it and render immediately.",
                "rewrite": deterministic_analysis["rewrite"],
                "follow_ups": [],
            }

        result = coach.coach(payload, readiness=modes, model_runner=ungrounded_model)

        self.assertNotIn("Ingredients is ready now", result["answer"])
        grounded = " ".join(
            [result["answer"], result.get("provider_note", ""), *result.get("availability_notes", [])]
        ).lower()
        self.assertTrue(any(term in grounded for term in ("not ready", "unavailable", "blocked")), grounded)


if __name__ == "__main__":
    unittest.main()
