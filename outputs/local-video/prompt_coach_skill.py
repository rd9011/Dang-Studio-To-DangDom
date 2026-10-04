#!/usr/bin/env python3
"""Versioned LTX production knowledge supplied to Prompt Coach models.

This module is intentionally compact.  The deterministic analysis in
``prompt_coach.py`` remains authoritative for scores, settings and local asset
readiness; this skill gives the conversational model enough domain knowledge
to explain those facts without improvising feature semantics.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from studio_paths import PATHS


SKILL_ID = "dang-studio-ltx25-production"
SKILL_VERSION = "1.4.0"
SKILL_LABEL = "LTX‑2.5 Production Skill v1.4"
SKILL_PACKAGE_DIR = PATHS.app_root / "skills" / "ltx25-prompt-coach"
SKILL_ENTRYPOINT = SKILL_PACKAGE_DIR / "SKILL.md"
FIRST_LAST_FRAME_REFERENCE = SKILL_PACKAGE_DIR / "references" / "first-last-frame.md"
MAX_SKILL_ENTRYPOINT_BYTES = 16 * 1024
MAX_SKILL_REFERENCE_BYTES = 12 * 1024


def _load_markdown(path: Path, *, maximum_bytes: int) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        return ""
    if not payload or len(payload) > maximum_bytes:
        return ""
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""


@lru_cache(maxsize=1)
def load_skill_instructions() -> str:
    """Load the inspectable skill entrypoint supplied to every Coach model.

    The Python rules below remain a compact structured representation used for
    routing.  The Markdown package is the human-auditable operating procedure
    and is deliberately loaded at runtime so the displayed skill is also the
    skill the conversational model receives.
    """

    return _load_markdown(SKILL_ENTRYPOINT, maximum_bytes=MAX_SKILL_ENTRYPOINT_BYTES)


@lru_cache(maxsize=1)
def load_first_last_frame_reference() -> str:
    """Load the optional, user-supplied boundary-frame prompting playbook."""

    return _load_markdown(
        FIRST_LAST_FRAME_REFERENCE,
        maximum_bytes=MAX_SKILL_REFERENCE_BYTES,
    )


CORE_RULES = (
    "Treat verified local readiness and effective settings as facts; wording cannot unlock a missing asset.",
    "The percentage is prompt-specification completeness, never the probability or quality of a render.",
    "A 100% complete prompt can still suffer anatomy, identity, motion, text, lip-sync or audio errors.",
    "Choose the coaching mode before rewriting: develop a sparse or open-ended idea into one coherent concrete production prompt, but review an already detailed prompt conservatively.",
    "In development mode, make sensible reversible choices for action, chronology, camera, setting, lighting, style and ambience instead of returning a fill-in-the-blank scaffold.",
    "Preserve every explicit name, identity fact, relationship, dialogue line, @tag and constraint. Do not add a new named character, dialogue, lore or biography unless the creator asks for it.",
    "For a 3–5 second development draft, write a precision shot: actual duration and aspect, one-shot contract, loaded-reference authority only when verified, supplied subject/prop/geography continuity, timed or clearly staged micro-beats, explicit camera behavior, physically plausible restrained performance, audio ownership and targeted negative constraints.",
    "When identity or story choices are missing, use the least committal coherent assumption and disclose it; do not fabricate exact face, body, wardrobe, biography, relationship or dialogue details.",
    "Ask at most three optional follow-up questions and reserve them for consequential choices: who or identity, the desired final beat, and style, reference or audio direction.",
    "Dense detail must remain internally consistent and must not overload the action, camera or audio capacity of a short shot.",
    "Never leave square-bracket placeholders, TODO or TBD markers in a production draft. State material creative assumptions, and offer follow-up questions only as optional variations rather than blockers.",
    "Use one short continuous shot as the safest local default. LTX-2.5 can produce 2–4 explicit shots, but each transition, recurring identity, framing and audio continuity must be stated and enough duration must be available.",
    "Preserve every stable @tag exactly, including case, spelling, order and repeated speaker bindings.",
    "Recommend only controls that exist in the supplied Studio context and state clearly when the best control is unavailable.",
    "A prompt @tag does not prove that a saved Character Bible profile, image, audio file or control input is loaded. Treat supplied reference counts as truth and phrase optional setup conditionally.",
    "Camera framing and movement are prompt instructions in this Studio, not separate camera-setting switches.",
    "For same-shot Start/End references, default to matching camera position, lens/field of view, horizon, perspective, subject distance, framing and scale unless the creator explicitly requests a deliberate move.",
    "Keep reference identity, timed composition, structural guidance, motion paths and audio timing conceptually separate.",
    "When a request is too complex, reduce simultaneous actions or split it into clips; do not promise that more prose removes model risk.",
    "Favor one primary action plus one supporting reaction. For a three-second shot establish immediately and hold the ending briefly; for five seconds allow a short settle, one build and a readable final hold. Live duration always wins.",
    "Treat every Studio option labelled Experimental as a canary, explain why it is outside the preferred model or hardware envelope, and recommend the selected workflow's Preferred setting as the reliable fallback.",
)


PROMPT_BLUEPRINT = (
    "Shot contract: live duration, aspect ratio, one continuous shot and no-cut rule; grant authority to Start/End/Interior or other references only when the live context verifies they are loaded.",
    "Continuity: preserve supplied subject identity, props, scale, wardrobe, relationships, geography and required @tags without inventing exact identity attributes.",
    "Micro-beats: use timestamps or an equally clear opening → action → final-hold progression, with one primary action plus at most one supporting reaction.",
    "Camera: state framing, viewpoint and either one compatible movement or a lock; keep it consistent with reference authority and geography.",
    "Motion and performance: keep movement physically plausible and expressions, gestures and secondary motion restrained enough for the short duration.",
    "World: location, lighting source, atmosphere, materials and visual style.",
    "Audio ownership: identify ambience, effects, music, source audio and explicitly tagged dialogue; keep each supplied line assigned to one speaker and never invent dialogue.",
    "Targeted avoid clause: protect the requested identity, props, geography, anatomy, framing and audio from the most likely failure modes without adding a generic negative-prompt dump.",
)


WORKFLOW_RULES: dict[str, dict[str, Any]] = {
    "generate": {
        "label": "Generate",
        "use_when": "creating a new short shot from text, optionally with boundary or timed image anchors",
        "controls": "prompt, quality profile, 1–10 second selector, supported aspect ratios, seed, Character Bible, Scene Beats and Start/End/Interior anchors",
        "limits": (
            "Start, End and Interior images guide particular times; they are not a global identity embedding.",
            "Frame capture creates a visual hand-off for another Generate clip, not latent or audio-aware continuation.",
            "For image-conditioned generation, the image owns appearance and composition; prompt the change, motion, camera and audio instead of redundantly redescribing every visible detail.",
        ),
    },
    "ingredients": {
        "label": "Ingredients",
        "use_when": "several reference identities or objects must remain distinguishable throughout one shot",
        "controls": "one clean reference panel image and one unique stable @tag per active identity/object, using the dedicated Q8 Ingredients lane; when one identity needs several views, assemble those views into one clean character-sheet panel rather than uploading duplicate-tag cards; Studio builds the official Reference sheet / Generated video prompt wrapper automatically",
        "limits": (
            "The official trained bucket is 768×448 at 121 frames/24 fps. The local 384×224 profile is a 16 GB memory compromise and may reduce reference fidelity; use the trained profile only when memory permits.",
            "Profile-default landscape is preferred. Other aspect ratios are out-of-training experiments and may weaken identity, anatomy, reference placement or composition, especially in portrait.",
            "Prefer clean black-background character sheets with a face close-up and body views, product-style prop views and one uncluttered location panel.",
            "Ingredients and Motion Track cannot be combined in the same render in this Studio.",
            "Reference quality and a precise tag description improve consistency but do not guarantee a perfect identity lock.",
        ),
    },
    "motion": {
        "label": "Motion Track",
        "use_when": "a subject or point must follow an explicit spatial trajectory",
        "controls": "required Start image, normalized path JSON and a short Q8 control render; begin with 3–4 physically plausible points per path",
        "limits": (
            "The Start image establishes the scene and paths own movement. Prompt subject/content/style without describing a competing trajectory.",
            "512×320 is the preferred local quality profile. The 640×384 profile is a three-second-only 16 GB experiment with materially higher attention, swapping, runtime and Python/Metal termination risk.",
            "Motion Track and Ingredients cannot be combined in the same render in this Studio.",
        ),
    },
    "union": {
        "label": "Union Control",
        "use_when": "a timed Canny, depth or pose guide should control whole-frame structure",
        "controls": "a timed guide video plus prompt appearance guidance in the Q8 control lane",
        "limits": (
            "A static comic strip or storyboard is not itself a timed control video; use Start/Interior/End anchors or first turn it into a timed guide.",
            "Do not fight the guide with contradictory camera or geometry instructions.",
        ),
    },
    "extend": {
        "label": "Extend",
        "use_when": "new material must be generated immediately before or after an existing clip",
        "controls": "source video, before/after direction and a prompt describing only the new interval",
        "limits": (
            "It inherits source dimensions and needs the separate Q8/HQ edit lane.",
            "Using a captured frame with Generate is only a visual approximation and does not preserve hidden velocity or audio state.",
        ),
    },
    "retake": {
        "label": "Retake",
        "use_when": "a selected interval inside an existing clip must be replaced",
        "controls": "source video, exact interval, replacement action and optional source-audio preservation",
        "limits": (
            "Describe only the replacement interval and its boundary continuity.",
            "It needs the separate Q8/HQ edit lane.",
        ),
    },
    "a2v": {
        "label": "Audio-to-Video",
        "use_when": "audio timing, rhythm, speech or performance must drive the visible motion",
        "controls": "a precisely trimmed local audio segment, time range and an optional Start image in the Q8 audio lane",
        "limits": (
            "The supplied audio remains unchanged and owns timing/motion; prompt only the visual subject, scene, style and camera rather than competing timing instructions.",
            "Ordinary Generate cannot be assumed to follow exact beats or phonemes.",
            "Use Character Voices as a separate speaker-controlled pass when dependable voice identity matters.",
        ),
    },
}


# Non-render-lane Studio features need first-class grounding too.  Keeping
# these facts structured lets the conversational Coach answer arbitrary
# paraphrases without hard-coding one response per question.
STUDIO_CAPABILITIES: dict[str, dict[str, Any]] = {
    "frame_composer": {
        "label": "Frame Composer",
        "purpose": "compose a local still from one to four active reference photos, then assign it as a Start, End or Interior frame",
        "controls": "add reference views under stable @tags, mention every active tag once in the composition prompt, choose the matched LTX canvas and click Compose frame locally",
        "limits": "the installed FLUX.2 Klein Q4 composer is reference-based and requires at least one photo; it is not blank text-only image generation",
    },
    "character_bible": {
        "label": "Character Bible",
        "purpose": "save reusable visual identity text and managed references locally across Studio restarts",
        "controls": "name, fixed appearance, wardrobe, continuity rules, avoid rules and up to four local reference images / 24 MB total",
        "limits": "a saved profile is not the true Ingredients IC-LoRA workflow and voice assignments remain separate",
    },
    "timed_anchors": {
        "label": "Start, End and Interior anchors",
        "purpose": "condition exact boundary moments or evenly spaced interior moments with supplied images",
        "controls": "upload or assign a composed still, add a description and strength, and keep the render aspect consistent",
        "limits": "descriptions alone do not synthesize stills; interior anchors are temporal conditioning, not a global identity embedding or Ingredients",
    },
    "scene_beats": {
        "label": "Scene Beats & Motion",
        "purpose": "relay a short chronological action plan into the video prompt",
        "controls": "use concise beats whose timing fits the selected duration and one coherent shot",
        "limits": "beats are prompt structure, not a motion path, pose guide or independent render control",
    },
    "character_voices": {
        "label": "Character Voices",
        "purpose": "apply dependable speaker-separated voices after the picture is generated",
        "controls": "give every cue one real stable @character tag, one voice preset, one line and explicit timing",
        "limits": "native generated dialogue can exchange or blend voices; voice assignments are separate from visual Character Bible profiles",
    },
    "frame_capture": {
        "label": "Capture current frame",
        "purpose": "save the exact displayed frame from a completed video as a local PNG for a new image-to-video clip",
        "controls": "pause or seek to the desired hand-off time, then click Capture current frame",
        "limits": "this does not trim or stitch the rejected tail, carry audio or hidden velocity, or perform native Extend",
    },
    "ai_upscale": {
        "label": "AI Quality Upscale to 1080p",
        "purpose": "use local Real-ESRGAN on the Apple GPU to reconstruct edge and texture detail while preserving aspect ratio, frame rate and audio",
        "controls": "finish a video, then click AI upscale to 1080p; the original is preserved",
        "limits": "it cannot repair anatomy, identity, motion or composition, and independent-frame enhancement can make fine texture shimmer",
    },
    "fast_resize": {
        "label": "Fast 720p resize",
        "purpose": "quickly enlarge the short edge with Lanczos and mild sharpening while preserving aspect ratio and audio",
        "controls": "finish a video, then create the separate 720p copy",
        "limits": "it adds pixels but does not reconstruct missing detail",
    },
    "render_settings": {
        "label": "Quality, duration, aspect and seed",
        "purpose": "choose the compute/detail profile, 1–10 second duration, output shape and repeatable random seed",
        "controls": "test at three seconds with Draft or Conservative, retain a useful seed, then move to Quality for the final take; named aspects use the nearest safe 64-pixel LTX grid",
        "limits": "durations above five seconds carry a complexity warning; aspect/profile choices change native dimensions and may be experimental in control lanes",
    },
    "local_compute": {
        "label": "Local compute behavior",
        "purpose": "keep prompts, references, models and outputs on this Mac",
        "controls": "one media or conversational inference job runs at a time so the 16 GB unified-memory machine is not overloaded",
        "limits": "heavy control modes, high resolutions and several references increase swapping, runtime and Python/Metal termination risk",
    },
}


_CAPABILITY_PATTERNS = {
    "frame_composer": r"\b(?:frame composer|compose|create|make|generate)\b.{0,50}\b(?:image|still|picture|photo|frame)\b|\b(?:image|still|picture|photo|frame)\b.{0,50}\b(?:compose|create|make|generate)\b",
    "character_bible": r"\b(?:character bible|saved character|character profile|persist|restart|identity library|reuse (?:the )?(?:same )?character|same character across (?:clips?|videos?|sessions?))\b",
    "timed_anchors": r"\b(?:start frame|end frame|first frame|last frame|interior anchor|timeline anchor|keyframe|boundary frame)\b",
    "scene_beats": r"\b(?:scene beats?|prompt relay|chronolog(?:y|ical)|timed beats?)\b",
    "character_voices": r"\b(?:character voices?|speaker separation|voice preset|voices? intermix|multilingual voice|keep (?:the )?voices? separate|different voice for each|same voice across clips?|voice in another language)\b",
    "frame_capture": r"\b(?:capture (?:the )?(?:current )?frame|screenshot|continue from|carry on from|hand[- ]off frame|pause (?:at|around))\b",
    "ai_upscale": r"\b(?:ai (?:quality )?upscal|real[- ]?esrgan|1080p|youtube quality|reconstruct detail|higher resolution|make (?:it|the video) sharper|improve (?:the )?(?:detail|resolution)|blurry output)\b",
    "fast_resize": r"\b(?:fast 720p|720p resize|lanczos|quick resize)\b",
    "render_settings": r"\b(?:quality profile|draft|balanced|max detail|conservative|aspect ratio|16:9|9:16|duration|seconds?|seed|resolution)\b",
    "local_compute": r"\b(?:local|offline|privacy|compute|memory|ram|swap|metal|python (?:quit|crash)|one job)\b",
}


def _mentioned_capabilities(text: str) -> list[str]:
    return [
        capability
        for capability, pattern in _CAPABILITY_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    ]


REFERENCE_RULES = (
    "Character Bible profiles persist locally across restarts and can supply text plus managed views, but they are not the same as the true Ingredients identity workflow.",
    "Start and End frames anchor boundaries; Interior anchors guide named moments. Their influence is temporal, not a guaranteed full-clip identity lock.",
    "With a Start image, avoid wasting prompt space on appearance already fixed by the image; describe the continuous visible change. With Start plus End, describe the bridge between the two endpoints.",
    "With both Start and End loaded, prompt them as authoritative endpoint targets and describe the smallest physically plausible continuous bridge; conditioning still cannot guarantee a pixel-exact match.",
    "A performance change does not require a camera change. Strengthen gaze, expression, gesture or posture first; if a move is intentional, make it chronological and quantify only an amount the creator supplied.",
    "Frame Composer combines up to four reference sets into one still; assign that result as Start, End or Interior rather than calling it a video identity embedding.",
    "Use machine-safe tags such as @subject_a and @subject_b. Each Ingredients reference card must have a unique tag; combine several views of one identity into one clean reference-sheet image under one tag. Mention each active Ingredients tag and give each spoken line exactly one @speaker.",
    "Character Voices is the dependable speaker-separation layer after picture generation; native generated dialogue may exchange or blend voices.",
    "For native speech, give every line one named speaker, state the allowed voice count, identify silent characters when useful, and keep any recurring prose voice specification stable across clips; prose is direction, not a persistent voice embedding.",
)


_MODE_PATTERNS = {
    "generate": r"\b(?:generate|text[ -]to[ -]video|new shot)\b",
    "ingredients": r"\b(?:ingredients?|multi[ -]reference|identity references?|several (?:photos?|images?|views?)|(?:two|three|four|multiple) (?:photos?|images?|views?)|keep (?:the )?same (?:person|character|object|identity))\b",
    "motion": r"\b(?:motion track|trajectory|path json|exact path|follow (?:this|a|the) (?:line|path|route)|drawn path)\b",
    "union": r"\b(?:union control|canny|depth guide|pose guide|control video|storyboard|comic strip)\b",
    "extend": r"\b(?:extend|continuation|continue the clip)\b",
    "retake": r"\b(?:retake|replace (?:an? )?interval|repair the clip)\b",
    "a2v": r"\b(?:audio[ -]to[ -]video|audio driven|soundtrack|to the beat|lip[ -]?sync|sound controls? (?:the )?(?:timing|movement)|move to (?:my|the) audio)\b",
}


def _mentioned_modes(text: str) -> list[str]:
    return [mode for mode, pattern in _MODE_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]


def _needs_first_last_frame_reference(request: dict[str, Any]) -> bool:
    context = request.get("context", {}) if isinstance(request, dict) else {}
    references = context.get("references", {}) if isinstance(context, dict) else {}
    if isinstance(references, dict) and bool(references.get("start_frame")) and bool(
        references.get("end_frame")
    ):
        return True
    discussion = f"{request.get('prompt', '')}\n{request.get('message', '')}"
    if re.search(r"\bboundary frames?\b", discussion, re.IGNORECASE):
        return True
    mentions_start = re.search(
        r"\b(?:first|start|opening)[ -]?frames?\b",
        discussion,
        re.IGNORECASE,
    ) is not None
    mentions_end = re.search(
        r"\b(?:last|end|ending|final)[ -]?frames?\b",
        discussion,
        re.IGNORECASE,
    ) is not None
    if mentions_start and mentions_end:
        return True
    mentions_boundary = re.search(
        r"\b(?:boundary|boundaries|endpoint|endpoints)\b",
        discussion,
        re.IGNORECASE,
    ) is not None
    mentions_start_term = re.search(
        r"\b(?:first|start|opening)\b",
        discussion,
        re.IGNORECASE,
    ) is not None
    mentions_end_term = re.search(
        r"\b(?:last|end|ending|final)\b",
        discussion,
        re.IGNORECASE,
    ) is not None
    return mentions_boundary and mentions_start_term and mentions_end_term


def build_skill_context(request: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Return the smallest relevant slice of the production skill.

    Generate is included as the baseline.  The selected mode, deterministic
    recommendation and any explicitly discussed workflow are added, capped by
    the seven known modes.  This keeps Apple's compact context focused.
    """

    context = request.get("context", {}) if isinstance(request, dict) else {}
    selected = str(context.get("mode", "generate"))
    recommended = str(analysis.get("workflow", "generate"))
    discussion = f"{request.get('prompt', '')}\n{request.get('message', '')}"
    ordered = [recommended, selected, *_mentioned_modes(discussion), "generate"]
    modes: list[str] = []
    for mode in ordered:
        if mode in WORKFLOW_RULES and mode not in modes:
            modes.append(mode)

    capabilities = _mentioned_capabilities(discussion)
    references = context.get("references", {}) if isinstance(context, dict) else {}
    if isinstance(references, dict):
        contextual_capabilities = []
        if int(references.get("character_references", 0) or 0) > 0:
            contextual_capabilities.append("character_bible")
        if (
            references.get("start_frame")
            or references.get("end_frame")
            or int(references.get("interior_anchors", 0) or 0) > 0
        ):
            contextual_capabilities.append("timed_anchors")
        if int(references.get("scene_beats", 0) or 0) > 0:
            contextual_capabilities.append("scene_beats")
        for capability in contextual_capabilities:
            if capability not in capabilities:
                capabilities.append(capability)

    result = {
        "id": SKILL_ID,
        "version": SKILL_VERSION,
        "scope": "LTX-2.5 prompting and this Mac's Dang Studio workflow selection",
        "authority": "Verified analysis/readiness wins over this explanatory skill and over model memory.",
        "skill_entrypoint": load_skill_instructions(),
        "score_definition": "Prompt completeness only; never render-success probability.",
        "core_rules": CORE_RULES,
        "prompt_blueprint": PROMPT_BLUEPRINT,
        "relevant_workflows": {mode: WORKFLOW_RULES[mode] for mode in modes},
        "relevant_studio_capabilities": {
            capability: STUDIO_CAPABILITIES[capability]
            for capability in capabilities
        },
        "local_workflow_status": {
            mode: analysis.get("workflow_readiness", {}).get(
                mode,
                {"availability": "unknown", "note": "No verified readiness result was supplied."},
            )
            for mode in modes
        },
        "reference_rules": REFERENCE_RULES,
    }
    if _needs_first_last_frame_reference(request):
        playbook = load_first_last_frame_reference()
        if playbook:
            result["conditional_reference"] = {
                "name": "first-last-frame",
                "scope": "Advisory only for prompts using or discussing Start and End frame boundaries.",
                "loaded_start_frame": bool(
                    isinstance(references, dict) and references.get("start_frame")
                ),
                "loaded_end_frame": bool(
                    isinstance(references, dict) and references.get("end_frame")
                ),
                "instructions": playbook,
            }
    return result


__all__ = [
    "SKILL_ID",
    "SKILL_LABEL",
    "SKILL_PACKAGE_DIR",
    "SKILL_VERSION",
    "STUDIO_CAPABILITIES",
    "build_skill_context",
    "load_first_last_frame_reference",
    "load_skill_instructions",
]
