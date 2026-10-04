#!/usr/bin/env python3
"""Local, model-optional conversational coach for LTX video prompts.

The installed LTX-2.5 Gemma 4 checkpoint is a text *encoder*.  It deliberately
does not contain the language-model head or KV-cache implementation needed to
write a reply. Prompt Coach therefore prefers the pinned private Qwen/Ollama
writer, uses Apple's OS-managed on-device Foundation Model as failover, and
falls back to deterministic, auditable production analysis when neither is
available. Runtime inference is local and this module never pulls a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from prompt_coach_skill import SKILL_ID, SKILL_LABEL, SKILL_VERSION, build_skill_context
from studio_paths import PATHS


MAX_MESSAGE_CHARS = 2_000
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_BYTES = 256 * 1024
MAX_REWRITE_CHARS = 110_000
MAX_REWRITE_BYTES = 288 * 1024
MAX_CONVERSATIONAL_MODEL_PROMPT_BYTES = 24 * 1024
MAX_CONVERSATIONAL_MODEL_INPUT_BYTES = 24 * 1024
MAX_HISTORY_ITEMS = 10
MAX_HISTORY_CHARS = 8_000
MAX_MODEL_RESPONSE_BYTES = 128 * 1024
MAX_LOCAL_MODEL_BYTES = 6 * 1024**3
RICH_DEVELOP_MIN_WORDS = 190
RICH_DEVELOP_MAX_WORDS = 520
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 180
OLLAMA_START_TIMEOUT_SECONDS = 15
APPLE_FM_TIMEOUT_SECONDS = 45
WORKFLOW_MODES = {"generate", "retake", "extend", "a2v", "ingredients", "union", "motion"}
COACH_ACTIONS = {"interact", "score", "develop", "chat"}
KNOWN_PROFILES = {
    "draft",
    "balanced",
    "quality",
    "max",
    "clip3s",
    "ingredients5s",
    "trained5s",
    "control3s",
    "control3s-plus",
    "motion3s-experimental",
}
KNOWN_ASPECTS = {"profile", "16:9", "9:16", "1:1", "4:3", "3:4"}
MODEL_NAME_PREFERENCE = (
    "qwen3:4b-instruct-2507-q8_0",
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:8b",
    "qwen3:4b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "llama3.2:latest",
    "gemma3:4b",
    "phi4-mini:latest",
)
_PROJECT_ROOT = PATHS.workspace
_PINNED_OLLAMA_EXECUTABLE = PATHS.ollama_runtime / "ollama"
_MODEL_CALL_LOCK = threading.Lock()
_OLLAMA_PROCESS_LOCK = threading.Lock()
_OWNED_OLLAMA_PROCESSES: set[subprocess.Popen[Any]] = set()
_UNRESOLVED_PLACEHOLDER_RE = re.compile(
    r"\[[^\[\]\r\n]{1,500}\]|<(?:insert|describe|choose|name|add)[^>]{0,200}>|\b(?:TODO|TBD)\b",
    flags=re.IGNORECASE,
)
_DEVELOP_REQUEST_RE = re.compile(
    r"\b(?:expand|develop|build|flesh out|fill (?:it|this) (?:in|out)|complete (?:it|this)|"
    r"turn (?:it|this|the idea) into|write (?:me )?(?:a |the )?(?:full |complete |production[- ]ready )?prompt|"
    r"make (?:it|this) (?:a |more )?(?:full|complete|creative|cinematic)|take creative liberty|surprise me)\b",
    flags=re.IGNORECASE,
)

_DEVELOP_MODE_MODEL_DIRECTIVE = (
    "DEVELOP MODE IS COMPLETION, NOT REVIEW. First silently build a fact ledger from "
    "prompt_under_review: the premise, every explicit name and its role or objective, relationships, "
    "actions, dialogue, @tags, duration, aspect, shot structure, camera, style, audio and constraints "
    "are immutable. Treat chatty wording as a visual brief, not a question to answer. A named search "
    "target stays the target; never recast it as the lead. Fill only missing craft choices. If the "
    "acting lead is unspecified, use an unnamed functional role such as 'the unnamed racer/searcher' "
    "and disclose that assumption; never invent a proper name, exact appearance, wardrobe, biography, "
    "canon or lore, relationship, dialogue, or a character-owned prop. Do not fill a famous character's "
    "unstated design from memory: say to preserve that named character's established design instead of "
    "supplying colours, clothes, facial traits or weapons that the creator did not authorize. Never claim "
    "a loaded Start/End frame, reference image, "
    "audio file or motion guide unless current_studio_context verifies it. "
    "Write one finished, causally staged precision prompt rather than a padded template: (1) the "
    "effective duration and aspect from verified settings, plus a shot/no-cut contract that preserves "
    "any explicit multi-shot intent; if aspect is profile, call it the profile-default trained-bucket "
    "aspect rather than inventing a ratio; (2) continuity for only known subjects, props, scale and "
    "geography; (3) non-overlapping chronological beats spanning the full duration—opening state, one "
    "feasible primary action or reaction, its consequence, and a readable final hold; (4) one clear "
    "framing/viewpoint and either a lock or one simple compatible move; (5) setting, light, style and "
    "physically plausible restrained secondary motion; (6) exact audio ownership, with no invented "
    "speech; and (7) a short avoid clause targeted to likely identity, anatomy, prop, geography, camera "
    "and audio failures. Precision means cause-and-effect and stable state, not adjective volume. "
    "When the creator explicitly asks for a classic theatrical, 2D cel-cartoon treatment, treat that "
    "as rendering and performance language—not permission to import franchise lore or add magic. For "
    "a sparse encounter, stage exactly one readable cause-and-effect bodily reaction gag using only the "
    "named characters and location already supplied. Do not invent supernatural effects, weapons, "
    "costumes, dialogue or handheld props. "
    "Return usable prose with no brackets, TODOs, TBDs or questions. After completing it, ask at most "
    "three nonblocking questions about missing lead identity/reference, final reveal or hold, and "
    "visual/audio direction."
)


class CoachInputError(ValueError):
    """Raised for a malformed browser request."""


class _LocalProviderExhausted(RuntimeError):
    """All private model providers failed or returned an unsafe candidate.

    ``public_audit`` is deliberately separate from the diagnostic exception
    text. The former is safe to show in the Coach panel and explains which
    candidates were rejected without dumping a subprocess error or model
    response into the browser.
    """

    def __init__(self, diagnostic: str, *, public_audit: str = "") -> None:
        super().__init__(diagnostic)
        self.public_audit = public_audit


def _clean_text(value: Any, *, field: str, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CoachInputError(f"{field} must be text.")
    value = value.strip()
    if not allow_empty and not value:
        raise CoachInputError(f"{field} cannot be empty.")
    if "\x00" in value or len(value) > maximum or len(value.encode("utf-8")) > maximum * 4:
        raise CoachInputError(f"{field} is too long.")
    return value


def _clean_model_answer(value: Any) -> str:
    """Normalize compact-model prose for the panel's plain-text renderer."""

    answer = _clean_text(value, field="model answer", maximum=2_500)
    answer = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", answer)
    answer = re.sub(r"__([^_\n]+)__", r"\1", answer)
    answer = re.sub(r"`([^`\n]+)`", r"\1", answer)
    answer = re.sub(r"(?m)^#{1,6}\s+", "", answer)
    return answer


def _clean_prompt(value: Any, *, allow_empty: bool = False) -> str:
    """Validate the reviewed prompt without stripping or shortening it."""
    if not isinstance(value, str):
        raise CoachInputError("prompt must be text.")
    if not allow_empty and not value.strip():
        raise CoachInputError("prompt cannot be empty.")
    if "\x00" in value:
        raise CoachInputError("prompt contains an unsupported NUL character.")
    if len(value) > MAX_PROMPT_CHARS or len(value.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise CoachInputError(
            "prompt exceeds the local process safety ceiling "
            f"({MAX_PROMPT_CHARS:,} characters / {MAX_PROMPT_BYTES // 1024} KiB UTF-8); "
            "it was rejected without truncation."
        )
    return value


def _clean_rewrite(value: Any, *, field: str = "model rewrite") -> str:
    cleaned = _clean_text(value, field=field, maximum=MAX_REWRITE_CHARS)
    if len(cleaned.encode("utf-8")) > MAX_REWRITE_BYTES:
        raise CoachInputError(f"{field} is too long.")
    return cleaned


def _bounded_string(value: Any, *, maximum: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:maximum]


def _validate_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoachInputError("context must be an object.")
    unknown = sorted(
        set(value)
        - {
            "mode",
            "profile",
            "aspect_ratio",
            "duration_seconds",
            "seed",
            "references",
        }
    )
    if unknown:
        raise CoachInputError("Unknown context field(s): " + ", ".join(unknown))
    mode = value.get("mode", "generate")
    if mode not in WORKFLOW_MODES:
        raise CoachInputError("context.mode is not a supported workflow.")
    profile = value.get("profile", "quality")
    if profile not in KNOWN_PROFILES:
        raise CoachInputError("context.profile is not supported.")
    aspect = value.get("aspect_ratio", "profile")
    if aspect not in KNOWN_ASPECTS:
        raise CoachInputError("context.aspect_ratio is not supported.")
    duration = value.get("duration_seconds", 3)
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 10:
        raise CoachInputError("context.duration_seconds must be an integer from 1 to 10.")
    seed = value.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4_294_967_295:
        raise CoachInputError("context.seed must be a whole number from 0 to 4294967295.")
    references = value.get("references", {})
    if not isinstance(references, dict):
        raise CoachInputError("context.references must be an object.")
    clean_references: dict[str, int | bool] = {}
    for key, raw in references.items():
        if key not in {
            "start_frame",
            "end_frame",
            "interior_anchors",
            "character_references",
            "ingredient_references",
            "control_video",
            "audio",
            "motion_tracks",
            "scene_beats",
        }:
            continue
        if isinstance(raw, bool):
            clean_references[key] = raw
        elif isinstance(raw, int) and 0 <= raw <= 16:
            clean_references[key] = raw
    return {
        "mode": mode,
        "profile": profile,
        "aspect_ratio": aspect,
        "duration_seconds": duration,
        "seed": seed,
        "references": clean_references,
    }


def _validate_history(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_HISTORY_ITEMS:
        raise CoachInputError(f"history must contain at most {MAX_HISTORY_ITEMS} messages.")
    history: list[dict[str, str]] = []
    total = 0
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise CoachInputError(f"history message {index} must contain exactly role and content.")
        role = item.get("role")
        if role not in {"user", "assistant"}:
            raise CoachInputError(f"history message {index} has an invalid role.")
        content = _clean_text(
            item.get("content"), field=f"history message {index}", maximum=MAX_MESSAGE_CHARS
        )
        total += len(content)
        if total > MAX_HISTORY_CHARS:
            raise CoachInputError("history is too long.")
        history.append({"role": role, "content": content})
    return history


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CoachInputError("Prompt Coach request must be an object.")
    unknown = sorted(set(payload) - {"action", "message", "prompt", "history", "context"})
    if unknown:
        raise CoachInputError("Unknown Prompt Coach field(s): " + ", ".join(unknown))
    action = payload.get("action", "interact")
    if not isinstance(action, str) or action not in COACH_ACTIONS:
        raise CoachInputError("action must be either interact or score, develop, or chat.")
    return {
        "action": action,
        "message": _clean_text(payload.get("message"), field="message", maximum=MAX_MESSAGE_CHARS),
        "prompt": _clean_prompt(payload.get("prompt", ""), allow_empty=True),
        "history": _validate_history(payload.get("history")),
        "context": _validate_context(payload.get("context", {})),
    }


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _conversation_intent(message: str) -> str:
    """Route narrow Studio-help questions before prompt scoring.

    These are questions *about* which control should own a production task,
    not requests to add their wording (or generic examples such as
    ``@speaker``) to the video prompt. Keeping this deliberately narrow lets
    genuine follow-up facts continue to extend the creator-authored contract.
    """

    asks_how_to_make_still = _has(
        r"^\s*(?:how (?:do|can|should|would)|what(?:'s| is| are)|where|which|can (?:i|we)|"
        r"does (?:this|it|the studio)|is there)\b",
        message,
    ) and _has(
        r"\b(?:create|make|generate|compose|build)\b.{0,45}\b(?:an? )?(?:image|still|picture|photo|frame)\b|"
        r"\b(?:image|still|picture|photo|frame)\b.{0,45}\b(?:create|make|generate|compose|build)\b",
        message,
    )
    if asks_how_to_make_still:
        return "frame_composer_guidance"
    asks_for_comparison = _has(
        r"\b(?:teach|explain|compare|which|whether|difference)\b", message
    ) or ("?" in message and _has(r"\b(?:should|need)\b", message))
    if not asks_for_comparison:
        return ""
    audio_choices = sum(
        bool(_has(pattern, message))
        for pattern in (
            r"\b(?:audio[ -]to[ -]video|a2v)\b",
            r"\bnative(?: generated)? (?:dialogue|speech|voice|audio)\b",
            r"\bcharacter voices?\b",
        )
    )
    return "audio_workflow_guidance" if audio_choices >= 2 else ""


def _message_requests_prompt_work(message: str) -> bool:
    """Return true when the creator is asking Coach to change the production brief.

    This deliberately detects *actions on the prompt* rather than attempting
    to enumerate every possible Studio question.  It keeps an instruction
    such as ``make @maya's coat red`` in editing mode while allowing arbitrary
    questions to remain genuine conversation turns.
    """

    text = str(message or "").strip()
    if not text:
        return False
    return bool(
        _has(
            r"^\s*(?:please\s+)?(?:develop|review|rewrite|revise|improve|expand|"
            r"polish|shorten|simplify|change|add|remove|replace|edit|draft)\b",
            text,
        )
        or _has(
            r"^\s*(?:can|could|would|will) you\s+(?:develop|review|rewrite|revise|"
            r"improve|expand|polish|shorten|simplify|change|add|remove|replace|edit|draft)\b",
            text,
        )
        or _has(
            r"\b(?:develop|review|rewrite|revise|improve|expand|polish)\s+"
            r"(?:it|this|the|my|our)\b",
            text,
        )
        or _has(r"\b(?:use|apply)\s+(?:this|that)\s+(?:answer|change|idea)\b", text)
        or _has(r"^\s*answer to coach(?:'|’)?s question\s*:", text)
    )


def _message_is_general_question(message: str) -> bool:
    """Recognize open conversational turns without a feature-question allowlist."""

    text = str(message or "").strip()
    if not text or _message_requests_prompt_work(text):
        return False
    if "?" in text:
        return True
    return _has(
        r"^\s*(?:how|what|why|when|where|which|who|can|could|should|would|will|"
        r"do|does|did|is|are|was|were|explain|compare|teach|tell me|show me|"
        r"help me understand|walk me through|what about|how about)\b",
        text,
    )


def _interaction_mode(request: dict[str, Any]) -> str:
    """Choose scoring, prompt work, or ordinary conversation by user intent.

    Verified deterministic answers may still fast-path a few safety-critical
    workflows, but general conversation is intentionally open-ended.  It is
    not an allowlist of supported questions.
    """

    if request.get("action") == "score":
        return "score"
    if request.get("action") == "develop":
        return "prompt_work"
    if request.get("action") == "chat":
        return "conversation"
    message = str(request.get("message", ""))
    if _conversation_intent(message):
        return "verified_guidance"
    if _message_requests_prompt_work(message):
        return "prompt_work"
    # Legacy ``interact`` requests retain their historical prompt-work
    # behavior.  The current UI sends the explicit ``chat`` action, so general
    # conversation no longer depends on punctuation or keyword guessing.
    return "prompt_work"


def _coach_task(request: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Choose conservative review or creative concept development.

    An explicitly open brief should not receive a fill-in-the-blank worksheet.
    Conversely, a detailed production prompt should not gain invented story
    beats merely because it was sent through Coach.
    """

    if _interaction_mode(request) == "conversation":
        return "conversation"
    recent_user_text = "\n".join(
        item["content"] for item in request.get("history", []) if item.get("role") == "user"
    )
    discussion = f"{recent_user_text}\n{request.get('message', '')}"
    if _DEVELOP_REQUEST_RE.search(discussion):
        return "develop"
    if int(analysis.get("score", 0)) < 45:
        return "develop"
    return "review"


def _creative_completion_policy(
    request: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    task = _coach_task(request, analysis)
    if task == "conversation":
        return {
            "task": "conversation",
            "enabled": False,
            "reason": "The creator asked a conversational question rather than requesting a prompt rewrite.",
            "rules": [
                "Answer the current question directly and naturally using the verified Studio context and production skill.",
                "Do not score, review or rewrite the prompt unless the creator explicitly asks for that action.",
                "Preserve the current prompt byte-for-byte and use recent turns as conversational context.",
                "Never invent an installed feature, loaded reference or successful render guarantee.",
            ],
        }
    return {
        "task": task,
        "enabled": task == "develop",
        "reason": (
            "The creator asked for concept development or supplied an open brief."
            if task == "develop"
            else "The supplied prompt is detailed enough for conservative refinement."
        ),
        "rules": (
            [
                "Make one coherent interpretation now; complete the draft before asking optional questions.",
                "Treat conversational wording as a visual brief. Extract the premise plus every explicit name, role, objective and constraint before choosing craft details.",
                "Preserve every explicit name, identity anchor, role, relationship, dialogue line, @tag and production constraint; a named search target must remain the target, not become the lead.",
                "Fill missing chronology, camera, setting, lighting, atmosphere, style and ambience with concrete reversible choices. If the lead is missing, use an unnamed functional role rather than inventing identity, wardrobe, dialogue, lore or biography.",
                "Build a precision draft in order: shot contract, non-overlapping timed micro-beats spanning the duration, camera behavior, continuity and prop geography, physically plausible restrained performance, audio ownership, final hold and targeted failure exclusions.",
                "Never claim a Start frame, End frame, reference image, audio file or motion guide is supplied unless current_studio_context verifies it.",
                "Do not copy conversational framing such as 'do you know' into the video prompt; translate it into visual intent.",
                "Return finished prose with no square-bracket fields, TODOs, TBDs or angle-bracket instructions.",
                "State the main creative assumptions briefly in the answer; ask up to three specific follow-up questions only about consequential unresolved identity, final-beat, style/reference or audio choices. They are not blockers.",
            ]
            if task == "develop"
            else [
                "Refine conservatively and preserve supplied story facts.",
                "Do not introduce new named characters, dialogue, relationships or plot events.",
                "Return finished prose with no unresolved placeholders.",
            ]
        ),
    }


def _creator_authored_contract(request: dict[str, Any]) -> str:
    """Return facts the creator actually supplied across the conversation.

    The Studio prompt remains the root contract. Genuine follow-up answers
    can add an identity, line, ending beat, or other production fact, so user
    turns are appended in order. Assistant turns are deliberately excluded:
    they may contain a useful draft, but they can never authorize their own
    invented cast, dialogue, or plot event.
    """

    parts = [str(request.get("prompt", "")).strip()]
    parts.extend(
        content
        for item in request.get("history", [])
        if item.get("role") == "user"
        and (content := str(item.get("content", "")).strip())
        and not _conversation_intent(content)
        and not _message_is_general_question(content)
    )
    current = str(request.get("message", "")).strip()
    if (
        current
        and not _conversation_intent(current)
        and not _message_is_general_question(current)
    ):
        parts.append(current)
    return "\n".join(part for part in parts if part)


def _analysis_for_model(analysis: dict[str, Any]) -> dict[str, Any]:
    """Remove the deterministic fallback draft from conversational grounding.

    That draft is a safety fallback, not a model-authored answer. Supplying it
    as verified context caused compact models to copy its placeholders.
    """

    return {key: value for key, value in analysis.items() if key != "rewrite"}


def _skill_context_for_model(
    request: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Keep the real skill while removing duplicate structured prose.

    ``SKILL.md`` already contains the operating rules, precision-draft contract
    and coaching procedure.  Repeating CORE_RULES, PROMPT_BLUEPRINT and
    REFERENCE_RULES beside it consumed most of Apple's 4,096-token session
    window and left too little room for the actual precision draft.
    """

    full = build_skill_context(request, analysis)
    keys = (
        "id",
        "version",
        "scope",
        "authority",
        "skill_entrypoint",
        "score_definition",
        "relevant_workflows",
        "relevant_studio_capabilities",
        "local_workflow_status",
        "conditional_reference",
    )
    return {key: full[key] for key in keys if key in full}


def _stable_named_phrases(prompt: str) -> list[str]:
    """Extract explicit identity/target phrases that a rewrite must retain."""

    # Capitalised Studio feature names occur in tutorial/advisory questions but
    # are not people or story targets.  Keep this list deliberately limited to
    # product vocabulary that is rendered by this application; a broad title-
    # case stop-list would risk weakening protection for real character names.
    production_labels = {
        "audio to video",
        "character bible",
        "character voices",
        "frame composer",
        "motion track",
        "prompt coach",
        "scene beats",
        "union control",
    }
    generic_production_targets = {
        "action",
        "audio",
        "camera",
        "character",
        "movement",
        "performance",
        "prompt",
        "rhythm",
        "scene",
        "shot",
        "subject",
        "timing",
    }

    phrases: list[str] = []
    for match in re.finditer(
        r"\b(find|search(?:ing)? for|look(?:ing)? for|meet|follow|protect|chase|race against|"
        r"starring|featuring|named)\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3})",
        prompt,
        flags=re.IGNORECASE,
    ):
        phrase = re.split(
            r"\s+(?:in|at|on|with|while|when|then|and|but|who|that)\b",
            match.group(2),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,!?:;\"'“”")
        folded_target = re.sub(
            r"^(?:the|a|an)\s+", "", phrase.casefold().replace("-", " ")
        )
        # ``named`` is also ordinary production vocabulary: the Studio's own
        # instruction says “preserve every named character”.  That generic
        # noun is not a character called “character” and must not become a
        # mandatory identity anchor in the creator-contract gate.
        if (
            folded_target in production_labels
            or folded_target in generic_production_targets
            or (
                match.group(1).casefold() == "named"
                and phrase.casefold()
                in {
                    "character",
                    "characters",
                    "identity",
                    "identities",
                    "person",
                    "people",
                    "subject",
                    "subjects",
                    "target",
                    "targets",
                }
            )
        ):
            continue
        if phrase:
            phrases.append(phrase)
    for match in re.finditer(r"\b([A-Z][A-Za-z'’–]+(?:\s+[A-Z][A-Za-z'’–]+)+)\b", prompt):
        phrase = match.group(1).strip()
        folded_phrase = phrase.casefold().replace("-", " ")
        if folded_phrase in production_labels or any(
            folded_phrase == f"{prefix} {label}"
            for prefix in {
                "choose",
                "compare",
                "disable",
                "enable",
                "need",
                "needs",
                "open",
                "select",
                "the",
                "try",
                "use",
            }
            for label in production_labels
        ):
            continue
        phrases.append(phrase)
    return list(dict.fromkeys(phrase.casefold() for phrase in phrases))


def _masked_identity_contract(text: str, names: list[str]) -> tuple[str, dict[str, str]]:
    """Hide famous-name priors during a corrective creative pass.

    Small local writers often ignore an explicit "do not invent wardrobe or
    props" instruction as soon as they recognize a familiar franchise name.
    Neutral labels let the model solve staging and chronology instead of
    recalling canon. The labels are restored locally before any candidate is
    audited or displayed; no creator-authored identity is removed.
    """

    masked = text
    restore: dict[str, str] = {}
    for index, folded_name in enumerate(names, start=1):
        match = re.search(
            r"(?<!\w)" + re.escape(folded_name) + r"(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        token = f"SUBJECT_{index}"
        restore[token] = match.group(0)
        masked = re.sub(
            r"(?<!\w)" + re.escape(folded_name) + r"(?!\w)",
            token,
            masked,
            flags=re.IGNORECASE,
        )
    return masked, restore


def _restore_masked_identities(value: Any, restore: dict[str, str]) -> Any:
    if isinstance(value, str):
        restored = value
        for token, name in restore.items():
            number = token.rsplit("_", 1)[-1]
            restored = re.sub(
                rf"(?<!\w)SUBJECT[ _-]?{re.escape(number)}(?!\w)",
                name,
                restored,
                flags=re.IGNORECASE,
            )
        return restored
    if isinstance(value, list):
        return [_restore_masked_identities(item, restore) for item in value]
    if isinstance(value, dict):
        return {
            key: _restore_masked_identities(item, restore)
            for key, item in value.items()
        }
    return value


def _has_non_negated(pattern: str, text: str) -> bool:
    """Match an intent/risk term while ignoring nearby plain-language negation."""

    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if not _match_is_negated(text, match.start()):
            return True
    return False


def _match_is_negated(text: str, start: int) -> bool:
    """Return whether a term at ``start`` belongs to a nearby negative clause."""

    prefix = text[max(0, start - 64) : start]
    clause_prefix = re.split(r"[.!?;\n]", prefix)[-1]
    if re.search(
        r"\b(?:no|without|avoid(?:ing)?|exclude(?:d|s|ing)?|never|do not|does not|don't|must not)\b",
        clause_prefix,
        flags=re.IGNORECASE,
    ):
        return True
    return re.search(
        r"\b(?:no|without|avoid(?:ing)?|exclude(?:d|s|ing)?|never|do not|does not|don't|must not)"
        r"(?:\s+\w+){0,4}\s*$",
        prefix,
        flags=re.IGNORECASE,
    ) is not None


def _explicit_duration_value(text: str) -> int | None:
    """Return an overall integer duration without reading timeline decimals.

    A beat such as ``0.0–0.8 seconds`` previously exposed the trailing ``8``
    to the old integer regex and could therefore turn a five-second draft into
    an apparent eight-second request. Whole-second decimals remain valid for
    an actual duration (``5.0 seconds``), but a value on the right-hand side of
    a numeric timeline range is ignored.
    """

    pattern = re.compile(
        r"(?<![\d.])\b(10|[1-9])(?:\.0+)?\s*(?:-\s*)?"
        r"(?:s|sec|secs|second|seconds)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 32) : match.start()]
        if re.search(r"\d(?:\.\d+)?\s*(?:[-–—]|\bto)\s*$", prefix, flags=re.IGNORECASE):
            continue
        return int(match.group(1))
    return None


def _explicit_aspect_value(text: str) -> str | None:
    """Return an explicitly requested aspect without treating scenery as one."""

    stripped = text.strip(" \t\r\n.,:;-").casefold()
    if stripped == "landscape":
        return "16:9"
    for pattern, aspect in (
        (r"\b(?:9:16|vertical|reels?|shorts|tiktok|phone screen)\b", "9:16"),
        (r"\b(?:1:1|square|square post)\b", "1:1"),
        (r"\b(?:3:4|portrait photo|portrait post)\b", "3:4"),
        (r"\b(?:4:3|academy|classic television)\b", "4:3"),
        (
            r"\b(?:16:9|widescreen|youtube|cinematic frame|"
            r"landscape\s+(?:aspect(?:\s+ratio)?|orientation|format|video|canvas)|"
            r"(?:aspect(?:\s+ratio)?|orientation|format|video|canvas)\s*"
            r"(?:is|=|:|to)?\s*landscape)\b",
            "16:9",
        ),
    ):
        if _has(pattern, text):
            return aspect
    return None


def _has_quoted_dialogue(text: str) -> bool:
    """Distinguish quoted speech from quoted production vocabulary.

    Bare quotes remain conservative dialogue signals. The narrow exemption is
    for a quoted term explicitly framed as a phrase, style, look, label or
    other production treatment; nearby speech verbs always win.
    """

    quote_pattern = re.compile(r'["“]([^"”\n]{1,240})["”]')
    matches = list(quote_pattern.finditer(text))
    if not matches:
        return bool(re.search(r'["“”]', text))
    speech_pattern = (
        r"\b(?:say(?:s|ing|said)?|speak(?:s|ing|spoke)?|whisper(?:s|ed|ing)?|"
        r"shout(?:s|ed|ing)?|blurt(?:s|ed|ing)?|ask(?:s|ed|ing)?|"
        r"repl(?:y|ies|ied|ying)|utter(?:s|ed|ing)?|dialogue|line)\b"
    )
    production_before = re.compile(
        r"\b(?:phrase|term|word|words|label|title|style|look|aesthetic|"
        r"treatment|preset|visual direction)\s*(?:is|=|:|called|named)?\s*$",
        flags=re.IGNORECASE,
    )
    production_after = re.compile(
        r"^\s*(?:(?:as|for)\s+(?:the|a)\s+)?(?:phrase|term|label|title|"
        r"style|styling|look|aesthetic|treatment|preset|visual direction)\b",
        flags=re.IGNORECASE,
    )
    for match in matches:
        nearby = text[max(0, match.start() - 72) : min(len(text), match.end() + 72)]
        if _has_non_negated(speech_pattern, nearby):
            return True
        before = text[max(0, match.start() - 72) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 72)]
        if production_before.search(before) or production_after.search(after):
            continue
        return True
    return False


def _has_continuous_shot_declaration(text: str) -> bool:
    """Recognize a shot declaration even when settings sit inside it."""
    return _has(r"\b(?:one|single)\s+continuous(?:\s+[^\s,.;!?]+){0,6}\s+shot\b", text)


def _has_one_shot_constraint(text: str) -> bool:
    """Recognize either a continuous-shot declaration or an explicit no-cut rule."""
    return _has_continuous_shot_declaration(text) or _has(
        r"\b(?:continuous take|one take|single take|avoid cuts?|no cuts?|without cuts?)\b",
        text,
    )


def _signals(prompt: str, context: dict[str, Any]) -> dict[str, Any]:
    # Coach-generated square-bracket fields are unresolved choices, not facts.
    # Excluding them from signal detection prevents a draft such as
    # ``[fixed wardrobe]`` from receiving credit as if the creator supplied it.
    unresolved_placeholders = bool(_UNRESOLVED_PLACEHOLDER_RE.search(prompt))
    text = _UNRESOLVED_PLACEHOLDER_RE.sub(" ", prompt).lower()
    references = context["references"]
    words = re.findall(r"\S+", prompt)
    tag_mentions = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", prompt)
    tags = list(dict.fromkeys(tag_mentions))
    invalid_machine_tags = [
        tag for tag in tags if re.fullmatch(r"@[a-z][a-z0-9_]{0,31}", tag) is None
    ]
    explicit_duration = _explicit_duration_value(text)
    duration = explicit_duration if explicit_duration is not None else context["duration_seconds"]
    aspect = _explicit_aspect_value(text) or context["aspect_ratio"]
    identity_detail = _has(
        r"\b(?:wearing|dressed in|wardrobe|hair|eyes|face|facial|age|build|scar|freckles|tattoo|consistent|unchanged|same character|same person|identity|colou?r(?:ed)? (?:shirt|jacket|dress|coat|suit)|(?:woman|man|girl|boy|person|character|child|actor|hero|pilot) in (?:an? )?(?:[a-z-]+ ){0,3}(?:shirt|jacket|dress|coat|suit|uniform))\b",
        text,
    )
    action = _has(
        r"\b(?:walk(?:s|ed|ing)?|ran|run(?:s|ning)?|turn(?:s|ed|ing)?|look(?:s|ed|ing)?|find|finds|finding|found|search(?:es|ed|ing)?|meet|meets|meeting|met|race|races|raced|racing|chase|chases|chased|chasing|rais(?:es|ed|ing)|lower(?:s|ed|ing)?|open(?:s|ed|ing)?|clos(?:es|ed|ing)|enter(?:s|ed|ing)?|leave|leaves|leaving|left|speak(?:s|ing)?|spoke|say|says|saying|said|perform(?:s|ed|ing)?|smil(?:es|ed|ing)|cry|cries|cried|crying|drive|drives|driving|drove|fly|flies|flying|flew|fall|falls|falling|fell|jump(?:s|ed|ing)?|danc(?:es|ed|ing)|fight|fights|fighting|fought|reach(?:es|ed|ing)?|hold|holds|holding|held|mov(?:es|ed|ing)|cross(?:es|ed|ing)?|approach(?:es|ed|ing)?|pull(?:s|ed|ing)?|push(?:es|ed|ing)?|sit|sits|sitting|sat|stand|stands|standing|stood|begin|begins|beginning|began|end|ends|ending|ended|roll(?:s|ed|ing)?)\b",
        text,
    )
    concrete_subject = _has(
        r"\b(?:woman|man|girl|boy|person|character|child|baby|actor|hero|pilot|warrior|dancer|singer|athlete|gymnast|chef|doctor|robot|android|creature|animal|dog|cat|rabbit|bunny|horse|bird|fish|dragon|monster|racer|driver|car|truck|vehicle|motorcycle|bicycle|train|aircraft|plane|spaceship|boat|ship|ball|kite|cup|bottle|book|phone|camera|flower|tree|building|house|object|product)\b",
        text,
    )
    start_frame_supplied = bool(references.get("start_frame"))
    end_frame_supplied = bool(references.get("end_frame"))
    boundary_pair = start_frame_supplied and end_frame_supplied
    mentions_start_frame = _has(r"\b(?:first|start|opening)[ -]?frame\b", text)
    mentions_end_frame = _has(r"\b(?:last|end|ending|final)[ -]?frame\b", text)
    mentions_boundary_terms = _has(r"\b(?:boundary|boundaries|endpoint|endpoints)\b", text)
    mentions_start_term = _has(r"\b(?:first|start|opening)\b", text)
    mentions_end_term = _has(r"\b(?:last|end|ending|final)\b", text)
    boundary_pair_discussed = (mentions_start_frame and mentions_end_frame) or (
        mentions_boundary_terms and mentions_start_term and mentions_end_term
    )
    exact_start_boundary = _has(
        r"\b(?:exact (?:first|start|opening) frame|(?:first|start|opening) frame.{0,28}\bexact\b|begin(?:s|ning)? exactly (?:from|on).{0,28}\bframe)\b",
        text,
    )
    exact_end_boundary = _has(
        r"\b(?:exact (?:last|end|ending|final) frame|(?:last|end|ending|final) frame.{0,28}\bexact\b|(?:finish|end|arrive|match)(?:es|ing)? exactly (?:at|on|with).{0,28}\bframe)\b",
        text,
    )
    reference_subject = any(
        bool(references.get(key))
        for key in ("start_frame", "end_frame", "character_references", "ingredient_references")
    )
    audio_control_supplied = bool(references.get("audio")) and (
        context["mode"] == "a2v"
        or _has_non_negated(
            r"\b(?:audio-driven|soundtrack|music video|to the beat|song|singing|dance to|"
            r"lip.?sync(?:s|ed|ing)?|supplied audio|audio (?:owns|drives|determines) (?:the )?timing|"
            r"(?:visible )?(?:motion|performance|rhythm) (?:follows?|driven by) (?:the )?audio)\b",
            text,
        )
    )
    motion_control_supplied = bool(references.get("start_frame")) and bool(
        references.get("motion_tracks")
    ) and (
        context["mode"] == "motion"
        or _has_non_negated(
            r"\b(?:motion track|trajectory|follow(?:s|ed|ing)? (?:an? )?(?:exact )?(?:[a-z-]+ ){0,3}(?:path|trajectory)|moves? from .+ to|camera path|exact(?:\s+[a-z-]+){0,3}\s+(?:path|trajectory))\b",
            text,
        )
    )
    camera_pattern = (
        r"\b(?:close[- ]?up|medium (?:shot|framing|camera)|"
        r"medium[- ]wide(?:\s+eye[- ]level)?(?:\s+(?:one|two|three|group)[- ]shot)?|"
        r"(?:one|two|three|group)[- ]shot|wide shot|full shot|macro|overhead|"
        r"low angle|high angle|pov|tracking|dolly|push[- ]?in|pull[- ]?back|pan|tilt|crane|"
        r"orbit|zoom|handheld|static camera|stationary camera|fixed camera|"
        r"locked (?:[a-z-]+ )?(?:camera|shot)|rack focus|shallow depth|lens)\b"
    )
    camera_terms = [
        match.group(0)
        for match in re.finditer(camera_pattern, text, flags=re.IGNORECASE)
        if not _match_is_negated(text, match.start())
    ]
    camera_move = _has_non_negated(
        r"\b(?:tracking(?: shot)?|doll(?:y|ies|ied|ying)|push(?:es|ed|ing)?[- ]?in|pull(?:s|ed|ing)?[- ]?back|pan(?:s|ned|ning)?|tilt(?:s|ed|ing)?|crane(?:s|d|ing)?|orbit(?:s|ed|ing)?|zoom(?:s|ed|ing)?|camera (?:moves|travels|tracks))\b",
        text,
    )
    camera_lock = _has_non_negated(
        r"\b(?:(?:locked|static|stationary|fixed)(?:\s+[a-z-]+){0,3}\s+"
        r"(?:camera|shot|(?:one|two|three|group)[- ]shot)|"
        r"camera (?:remains?|stays?) (?:locked|static|stationary|fixed))\b",
        text,
    )
    quantified_camera_move = camera_move and _has(
        r"\b(?:\d{1,3}(?:\.\d+)?\s*%|over\s+(?:\d+(?:\.\d+)?\s*)?(?:seconds?|secs?|s)\b|from\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)?\s+(?:to|through)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)?)",
        text,
    )
    voice_spec_terms = {
        "age": _has(r"\b(?:toddler|child|young|teen(?:age|aged)?|adult|middle[- ]aged|elderly)\b", text),
        "pitch": _has(r"\b(?:high|low|mid)[- ]?(?:pitched|pitch)|\bregister\b", text),
        "texture": _has(r"\b(?:airy|breathy|bright|dark|raspy|smooth|warm|soft|gravelly|nasal|vocal weight|vocal texture)\b", text),
        "delivery": _has(r"\b(?:pacing|pace|articulation|articulate|pronunciation|delivery|resonance)\b", text),
        "exclusion": _has(r"\b(?:no|not|never use|avoid)\b.{0,50}\b(?:voice|tone|pitch|rasp|nasal|falsetto|squeak|resonance)\b", text),
    }
    signals = {
        "prompt": prompt,
        "text": text,
        "unresolved_placeholders": unresolved_placeholders,
        "words": words,
        "tags": tags,
        "invalid_machine_tags": invalid_machine_tags,
        "duration": duration,
        "aspect": aspect,
        "identity_detail": identity_detail,
        "reference_subject": reference_subject,
        "start_frame_supplied": start_frame_supplied,
        "end_frame_supplied": end_frame_supplied,
        "boundary_pair": boundary_pair,
        "boundary_pair_discussed": boundary_pair_discussed,
        "boundary_contract": boundary_pair and exact_start_boundary and exact_end_boundary,
        "audio_control_supplied": audio_control_supplied,
        "motion_control_supplied": motion_control_supplied,
        "subject_basic": identity_detail or bool(tags) or concrete_subject or reference_subject,
        "action": action,
        "camera_terms": camera_terms,
        "camera": bool(camera_terms) or camera_lock,
        "camera_move": camera_move,
        "camera_lock": camera_lock,
        "quantified_camera_move": quantified_camera_move,
        "setting": _has(
            r"\b(?:interior|exterior|room|street|alley|forest|beach|desert|mountain|city|village|office|home|kitchen|bedroom|studio|stage|station|airport|space|underwater|rooftop|warehouse|garden|field|monument|burrow|racetrack|raceway|race course|road course|rain|snow|fog|day|night|sunset|sunrise|dawn|dusk)\b",
            text,
        ),
        "look": _has(
            r"\b(?:light|lighting|lit|sunlight|moonlight|neon|shadow|cinematic|realistic|photoreal|anime|comic|cartoon|animated|animation|cel[- ]shaded|illustration|film grain|colour|color|palette|atmosphere|moody|warm|cool|soft|hard light|volumetric)\b",
            text,
        ),
        "chronological": _has(
            r"\b(?:begins?|first|then|while|afterwards?|finally|ends?|at the end|throughout|one continuous shot|single continuous shot)\b",
            text,
        )
        or _has_continuous_shot_declaration(prompt),
        "continuity": _has(
            r"\b(?:avoid|no cuts?|without cuts?|preserve|remains?|unchanged|consistent|continuity|do not|must not|same (?:face|hair|clothes|wardrobe|character)|no (?:text|subtitles|duplicates?|extra limbs?))\b",
            text,
        ),
        "dialogue": _has_quoted_dialogue(prompt)
        or _has_non_negated(
            r"\b(?:dialogue|speech|says|speaks|voiceover|voice-over|narrat|lip.?sync|talking)\b",
            text,
        ),
        "voice_count": _has(r"\bexactly\s+(?:zero|one|two|three|four|\d+)\s+voices?\b", text),
        "voice_specification": sum(bool(value) for value in voice_spec_terms.values()) >= 2,
        "silent_voice_guard": _has(
            r"\b(?:remain(?:s)? (?:completely )?silent|must not (?:mouth|echo|repeat|speak)|no (?:additional|other|offscreen|background) voices?)\b",
            text,
        ),
        "explicit_silence": _has(
            r"\b(?:silent|silently|no dialogue|without dialogue|no speech|without speech|no voices?|without voices?)\b",
            text,
        ),
        "continuation": _has_non_negated(
            r"\b(?:continue|continuation|extend|next shot|next clip|after this|picks? up|last good frame)\b",
            text,
        ),
        "retake": _has_non_negated(
            r"\b(?:retake|replace an interval|replace the|fix the (?:first|middle|last)|regenerate .*seconds?|repair .*seconds?)\b",
            text,
        ),
        "audio_driven": _has_non_negated(
            r"\b(?:audio-driven|soundtrack|music video|to the beat|song|singing|dance to|"
            r"lip.?sync(?:s|ed|ing)?|supplied audio|audio (?:owns|drives|determines) (?:the )?timing|"
            r"(?:visible )?(?:motion|performance|rhythm) (?:follows?|driven by) (?:the )?audio)\b",
            text,
        ),
        "structural": _has_non_negated(
            r"\b(?:canny|depth guide|pose guide|control video|storyboard|comic strip|match the structure|same composition)\b",
            text,
        ),
        "timed_guide": _has_non_negated(r"\b(?:canny|depth guide|pose guide|control video|timed guide)\b", text),
        "tracked": _has_non_negated(
            r"\b(?:motion track|trajectory|follow(?:s|ed|ing)? (?:an? )?(?:exact )?(?:[a-z-]+ ){0,3}(?:path|trajectory)|moves? from .+ to|camera path|exact(?:\s+[a-z-]+){0,3}\s+(?:path|trajectory))\b",
            text,
        ),
        "timed_anchors": _has(
            r"\b(?:exact opening|opening frame|start frame|exact ending|final frame|end frame|key composition|interior anchor)\b",
            text,
        ),
        "multi_beat": _has(r"\b(?:then|afterwards?|finally|next,|at the end)\b", text),
        "multi_reference": _has_non_negated(
            r"\b(?:multi-reference|multiple reference|reference images|reference photos|character consistency)\b",
            text,
        )
        or int(context["references"].get("ingredient_references", 0) or 0) > 1,
        "multiple_cuts": _has_non_negated(
            r"\b(?:montage|multiple cuts?|several cuts?|cut to|smash cut|scene change|different locations?|transition to|(?:two|three|four|\d+) (?:locations?|scenes?))\b",
            text,
        ),
        "crowd": _has_non_negated(r"\b(?:crowd|dozens|hundreds|packed audience|many people)\b", text),
        "readable_text": _has_non_negated(
            r"\b(?:readable text|sign says|title card|exact logo|spells? out|subtitles?|on-screen text)\b",
            text,
        ),
        "complex_body": _has_non_negated(
            r"\b(?:fingers?|hands?|acrobat|fight|wrestl|complex dance|gymnast|juggle)\b",
            text,
        ),
        "testing": _has(r"\b(?:test|draft|preview|rough|try|experiment)\b", text),
    }
    return signals


def _availability(mode: str, readiness: dict[str, Any]) -> tuple[bool, str]:
    item = readiness.get(mode)
    if not isinstance(item, dict):
        return False, "readiness is still being checked"
    blocked = bool(item.get("blocked", not item.get("ready", False)))
    note = _bounded_string(item.get("note"), maximum=700)
    return not blocked, note or ("ready now" if not blocked else "required verified assets are missing")


def _readiness_snapshot(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize server-owned installation checks for model grounding."""

    snapshot: dict[str, dict[str, Any]] = {}
    for mode in sorted(WORKFLOW_MODES):
        eligible, note = _availability(mode, readiness)
        snapshot[mode] = {
            "availability": "eligible" if eligible else "blocked",
            "note": note,
            "verification_scope": (
                "Installed-asset eligibility only; receipt, source and Metal preflight run again on submit."
            ),
        }
    return snapshot


def _semantic_workflow(
    signals: dict[str, Any], context: dict[str, Any]
) -> tuple[str, str]:
    """Choose the workflow that best matches intent, independent of install state."""

    requested = "generate"
    reason = "a new shot from a text description"
    if signals["retake"]:
        requested, reason = "retake", "the request targets an interval inside existing footage"
    elif signals["continuation"]:
        requested, reason = "extend", "the request continues an existing clip"
    elif signals["audio_driven"]:
        requested, reason = "a2v", "audio timing is central to the shot"
    elif signals["tracked"]:
        requested, reason = "motion", "an exact trajectory is requested"
    elif signals["timed_guide"]:
        requested, reason = "union", "a timed structural guide is requested"
    elif signals["multi_reference"]:
        requested, reason = "ingredients", "multiple identity/object references must be preserved"
    elif context["mode"] != "generate":
        requested, reason = context["mode"], "that workflow is selected in Studio"
    return requested, reason


def _choose_workflow(
    signals: dict[str, Any], context: dict[str, Any], readiness: dict[str, Any]
) -> tuple[str, str, list[str]]:
    requested, reason = _semantic_workflow(signals, context)

    ready, note = _availability(requested, readiness)
    notes = [
        f"{requested.title()} is {'installed and eligible for submit-time preflight' if ready else 'blocked'}: {note}"
    ]
    if ready:
        return requested, reason, notes
    if requested == "extend":
        notes.append("Use frame capture plus Generate for a visual hand-off; it will not preserve latent motion or audio state.")
    elif requested == "retake":
        notes.append("Generate a separate replacement shot from an exported boundary frame, then trim and stitch externally.")
    elif requested == "a2v":
        notes.append("Generate the picture first and align audio afterward; ordinary Generate will not follow exact beats or phonemes.")
    elif requested in {"ingredients", "motion", "union"}:
        notes.append("Use Generate with a clean Start frame as an approximation; exact identity/trajectory/structure control is unavailable there.")
    generate_ready, generate_note = _availability("generate", readiness)
    notes.append(
        f"Generate fallback is {'installed and eligible for submit-time preflight' if generate_ready else 'blocked'}: {generate_note}"
    )
    if not generate_ready:
        notes.append("No local video workflow can run until the base Generate assets pass verification.")
        return "unavailable", f"{requested} is the semantic match, but neither it nor Generate can run", notes
    return "generate", f"{requested} is the semantic match but is currently unavailable", notes


def _production_concept_source(prompt: str) -> str:
    """Translate conversational framing into visual production intent.

    This is deliberately modest: it does not invent identity details or a new
    named cast.  Its job is to make the model-optional fallback usable instead
    of echoing questions such as ``do you know ...?`` into an LTX prompt.
    """

    source = re.sub(r"\s+", " ", prompt.strip())
    # A follow-up correction is production intent, not dialogue to echo into
    # the generated prompt.  Remove only the conversational wrapper; every
    # name, action, place and duration after it remains creator-authored.
    source = re.sub(
        r"^(?:no[,.!]?\s+)?(?:i\s+)?(?:specifically\s+)?"
        r"(?:mentioned|said|meant|asked)(?:\s+(?:that|for))?\s+",
        "",
        source,
        flags=re.IGNORECASE,
    )
    source = re.sub(
        r"\s*[,;:-]?\s*(?:and\s+)?this (?:will|should) be the whole scene[.!]?\s*$",
        "",
        source,
        flags=re.IGNORECASE,
    )
    topic_match = re.match(
        r"^(?:do you know|have you (?:seen|heard of)|what about)\s+([^?]+)\?\s*(.*)$",
        source,
        flags=re.IGNORECASE,
    )
    if topic_match:
        topic, remainder = topic_match.groups()
        if _has(r"\bcartoon\b", topic) and _has(r"\brace", topic):
            topic_copy = "A fast-paced cartoon road race"
        else:
            topic_copy = f"The visual concept is {topic.strip()}"
        source = f"{topic_copy}. {remainder.strip()}".strip()

    lead = "A determined lead racer" if _has(r"\brace", source) else "A determined protagonist"
    source = re.sub(
        r"\b(?:i\s+(?:have\s+|['’]ve\s+)?(?:gotta|got to|need to|want to)|we\s+(?:gotta|got to|need to|want to))\s+find\s+",
        lead + " searches for ",
        source,
        flags=re.IGNORECASE,
    )
    source = re.sub(r"\s+", " ", source).strip(" ,;:-")
    if source and source[-1] not in ".!?":
        source += "."
    return source


def _meeting_scene_brief(source: str) -> dict[str, str] | None:
    """Extract a compact ``A meets B and C near X`` creator contract.

    This deliberately handles only an explicit, visually useful construction.
    It is not a general named-entity recognizer and therefore cannot silently
    invent a cast or location when the creator did not provide one.
    """

    match = re.search(
        r"\b(?P<lead>[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3})\s+"
        r"(?:meets?|encounters?|joins?)\s+"
        r"(?P<others>[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3}"
        r"(?:\s*(?:,|\band\b)\s*[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3})*)\s+"
        r"(?:in\s+(?:10|[1-9])(?:\s*[- ]\s*|\s*)(?:s|sec|secs|second|seconds)\s+)?"
        r"(?:at|by|beside|near|next\s+to)\s+"
        r"(?P<location>(?:the\s+)?[^,.;!?]+)",
        source,
    )
    if not match:
        return None
    location = re.sub(
        r"\s+in\s+(?:10|[1-9])(?:\s*[- ]\s*|\s*)(?:s|sec|secs|second|seconds)\b.*$",
        "",
        match.group("location"),
        flags=re.IGNORECASE,
    ).strip()
    if not location:
        return None
    return {
        "lead": match.group("lead").strip(),
        "others": re.sub(r"\s+", " ", match.group("others")).strip(),
        "location": location,
    }


def _classic_cartoon_style_requested(text: str) -> bool:
    """Recognize an explicit old-school cel-cartoon art direction."""

    return _has(
        r"\b(?:classic|theatrical|traditional)\b[^.!?\n]{0,90}"
        r"\b(?:2d|cel(?:[- ](?:cartoon|animation))?|cartoon|animation)\b|"
        r"\b(?:hand[- ]painted backgrounds?|clean inked outlines?|squash[- ]and[- ]stretch)\b",
        text,
    )


def _concrete_default_chronology(
    signals: dict[str, Any], *, precision: bool = False, source: str = ""
) -> str:
    text = signals["text"]
    duration = max(1, min(10, int(signals["duration"])))
    first_end = 0.5 if duration <= 3 else 0.8
    final_start = 2.2 if duration <= 3 else round(duration * 0.76, 1)
    meeting = _meeting_scene_brief(source)
    if meeting:
        lead = meeting["lead"]
        others = meeting["others"]
        location = meeting["location"]
        if precision:
            if _classic_cartoon_style_requested(source):
                middle = round(duration * 0.48, 1)
                return (
                    f"0.0–{first_end:.1f} seconds: establish {others} in distinct, readable positions beside {location} "
                    f"as {lead} takes one brisk final step into the shared frame. "
                    f"{first_end:.1f}–{middle:.1f} seconds: {lead} stops abruptly in a compact squash-and-settle; "
                    f"the sudden stop causes {others} to lean back in one staggered physical reaction, with the second response landing a fraction later. "
                    f"{middle:.1f}–{final_start:.1f} seconds: let the trio recover only enough to register one silent, eye-led double-take "
                    "without changing positions or adding another gag. "
                    f"{final_start:.1f}–{duration:.1f} seconds: hold the resulting off-balance three-character tableau beside {location} "
                    "long enough for the physical joke to read."
                )
            return (
                f"0.0–{first_end:.1f} seconds: establish {lead} approaching {location} while {others} "
                "are already visible in distinct, readable positions; let the audience register all named characters before the interaction. "
                f"{first_end:.1f}–{final_start:.1f} seconds: {lead} reaches {others}; their eyes meet first, then each gives one restrained "
                "recognition reaction without crossing positions or adding a second plot beat. "
                f"{final_start:.1f}–{duration:.1f} seconds: resolve the meeting in a clear shared three-character tableau beside {location} "
                "and hold the composition long enough to read."
            )
        return (
            f"Begin with {lead} approaching {location}, then let {lead} meet {others} through one readable exchange of glances, "
            "and end on a stable shared tableau beside the same landmark."
        )
    if _has(r"\b(?:find|search|look for|looking for)\b", text):
        if precision:
            return (
                f"0.0–{first_end:.1f} seconds: establish the unnamed lead racer already scanning the course for the named target; "
                "keep the pose readable and let the eyes lead the search before any larger turn. "
                f"{first_end:.1f}–{final_start:.1f} seconds: one clear visual clue catches the lead's attention; the gaze shifts first, "
                "then the head and body follow in one controlled, physically plausible turn toward the route. "
                f"{final_start:.1f}–{duration:.1f} seconds: the lead commits to that direction and settles into a determined final pose; "
                "hold long enough for the discovery beat to read. The named target is not revealed yet."
            )
        return (
            "The shot begins with the lead scanning the course, then one clear visual clue draws them into a decisive turn, "
            "and it ends on a readable determined pose aimed down the route."
        )
    if _has(r"\b(?:race|racing|chase|chasing)\b", text):
        if precision:
            return (
                f"0.0–{first_end:.1f} seconds: establish the lead in a poised, readable start. "
                f"{first_end:.1f}–{final_start:.1f} seconds: build through one clean burst of speed and a single controlled bend, "
                "with body weight and secondary motion responding naturally. "
                f"{final_start:.1f}–{duration:.1f} seconds: resolve the move and hold a stable final pose without adding another beat."
            )
        return (
            "The shot begins on a poised start, builds through one clean burst of speed and a single bend, "
            "then ends with the subject settling into a readable final pose."
        )
    if signals["action"]:
        return (
            "Begin with a brief readable setup, perform the stated action as one physically plausible change, "
            "then end with a short stable hold."
        )
    return (
        "The main subject begins still, notices something just off camera, takes one deliberate step toward it, "
        "then settles into a clear final pose."
    )


def _concrete_default_camera(signals: dict[str, Any], *, source: str = "") -> str:
    if _meeting_scene_brief(source):
        return (
            "Camera: a locked medium-wide eye-level three-shot that keeps every named character and the meeting landmark visible; "
            "no zoom, pan, orbit, reframing or lens change."
        )
    if _has(r"\b(?:race|racing|chase|chasing|find|search)\b", signals["text"]):
        return "Camera: medium-wide eye-level framing with one smooth tracking move that keeps the lead readable."
    return "Camera: medium eye-level framing with one subtle slow push-in and no change of angle."


def _concrete_default_setting_and_light(signals: dict[str, Any], *, source: str = "") -> str:
    text = signals["text"]
    meeting = _meeting_scene_brief(source)
    if meeting:
        return (
            f"Setting and light: preserve the geography around {meeting['location']} as one stable landmark, with clean late-afternoon "
            "directional light, readable character separation and polished cinematic cartoon rendering."
        )
    if _has(r"\b(?:race|racing|racetrack|raceway|race course)\b", text):
        return (
            "Setting and light: a sunlit stylized desert raceway with bold readable shapes, warm directional light, "
            "clean saturated cel-animation colour and light airborne dust."
        )
    if _has(r"\b(?:cartoon|animated|animation)\b", text):
        return (
            "Setting and light: a clear stylized environment with bold readable shapes, warm directional light, "
            "clean saturated animation colour and restrained atmospheric motion."
        )
    if signals["setting"]:
        return "Light and look: soft directional cinematic light, clean subject separation, natural contrast and restrained atmosphere."
    if signals["look"]:
        return "Setting: an uncluttered environment appropriate to the stated style, with a clear foreground, middle ground and background."
    return (
        "Setting and light: an uncluttered practical location in late-afternoon light, with soft directional contrast, "
        "clear depth separation and a polished cinematic treatment."
    )


def _rewrite(
    signals: dict[str, Any],
    *,
    duration: int | None = None,
    aspect: str | None = None,
) -> str:
    source = _production_concept_source(signals["prompt"])
    meeting_scene = _meeting_scene_brief(source)
    # A visual-style clause can add many adjectives without adding any story
    # blocking. Keep an otherwise sparse classic-cartoon meeting eligible for
    # the concrete four-beat writer instead of mistaking the style vocabulary
    # for a developed action plan.
    sparse_word_limit = (
        90
        if meeting_scene and _classic_cartoon_style_requested(source)
        else 40
    )
    precision_open_brief = len(signals["words"]) <= sparse_word_limit and sum(
        not bool(signals[key])
        for key in ("identity_detail", "camera", "setting", "chronological", "continuity")
    ) >= 3
    effective_duration = int(duration if duration is not None else signals["duration"])
    effective_aspect = aspect if aspect is not None else signals["aspect"]
    if meeting_scene:
        source = re.sub(
            r"\s+in\s+(?:10|[1-9])(?:\s*[- ]\s*|\s*)(?:s|sec|secs|second|seconds)\b",
            "",
            source,
            flags=re.IGNORECASE,
        )
    if effective_duration != signals["duration"]:
        source = re.sub(
            r"\b(?:10|[1-9])(?:\s*[- ]\s*|\s*)(?:s|sec|secs|second|seconds)\b",
            "",
            source,
            flags=re.IGNORECASE,
        )
    if effective_aspect != signals["aspect"]:
        source = re.sub(r"\b(?:16:9|9:16|1:1|4:3|3:4)\b", "", source, flags=re.IGNORECASE)
    source = re.sub(r"\s+", " ", source).strip(" ,;:-")
    if source and source[-1] not in ".!?":
        source += "."
    parts: list[str] = []
    if signals["multiple_cuts"]:
        aspect_copy = effective_aspect if effective_aspect != "profile" else "profile-default trained-bucket"
        parts.append(
            f"A structured {effective_duration}-second {aspect_copy} sequence with two to four explicit shots; name each transition and preserve recurring identity, framing logic and audio continuity."
        )
    elif not _has_continuous_shot_declaration(source):
        aspect_copy = effective_aspect if effective_aspect != "profile" else "profile-default trained-bucket"
        parts.append(f"One continuous {effective_duration}-second {aspect_copy} shot.")
    elif effective_duration != signals["duration"] or effective_aspect != signals["aspect"]:
        aspect_copy = effective_aspect if effective_aspect != "profile" else "profile-default trained-bucket"
        parts.append(
            f"Production constraint: use {effective_duration} seconds and the {aspect_copy} aspect; this overrides conflicting timing or ratio wording below."
        )
    if source:
        parts.append(source)
    else:
        parts.append("A clearly defined main subject performs one simple visible action.")
    if signals["boundary_pair"] and signals["multiple_cuts"]:
        parts.append(
            "Use the supplied Start frame as the exact opening of the sequence and the supplied End frame as its exact final composition. "
            "Preserve only the identities, anatomy, scale, geography, props, viewpoint, environment, lighting and style that remain consistent across both; "
            "encode only the visible changes required by the End frame and keep every requested transition chronological."
        )
    elif signals["boundary_pair"] and not signals["boundary_contract"]:
        parts.append(
            "Use the supplied Start frame as the exact opening composition and the supplied End frame as the exact final composition. "
            "Preserve only the identities, anatomy, scale, subject geography, props, camera viewpoint, environment, lighting and visual style that remain consistent across both; "
            "encode only the visible changes required by the End frame and "
            "use the smallest physically plausible continuous changes needed to bridge them, begin exactly on Start and finish with a brief hold on End."
        )
    if signals["boundary_pair"] and not signals["multiple_cuts"]:
        if not signals["camera_move"]:
            parts.append(
                "Keep camera position and height, lens/field of view, perspective, horizon, subject-to-camera distance, framing, subject scale and fixed environmental anchors matched from Start to End."
            )
        else:
            parts.append(
                "Use the requested deliberate camera move as one continuous path from Start to End; the supplied End frame owns its final framing."
            )
        parts.append(
            "Do not invent a different opening or ending, a cut, teleport, identity swap or camera move that contradicts either boundary frame."
        )
        if effective_duration == 3:
            parts.append(
                "For this three-second take, establish the opening immediately, perform one compact primary action with at most one supporting reaction, and reserve a brief final hold."
            )
        elif effective_duration == 5:
            parts.append(
                "For this five-second take, allow a brief opening settle, build one clear primary action or reveal with at most one supporting reaction, and land on a readable final hold."
            )
    elif signals["boundary_pair"]:
        parts.append(
            "Do not invent different boundary compositions, extra transitions, teleports, identity swaps or camera changes that contradict the supplied frames."
        )
    elif signals["start_frame_supplied"] and not signals["identity_detail"]:
        parts.append(
            "Use the supplied Start frame for appearance and the exact opening composition; describe only the continuous visible change without reinventing it."
        )
    elif signals["end_frame_supplied"] and not signals["identity_detail"]:
        parts.append(
            "Use the supplied End frame as the exact final composition and describe only the physically plausible action that arrives there."
        )
    elif not signals["identity_detail"] and not signals["reference_subject"]:
        parts.append(
            "Subject continuity: keep the principal subject and every explicitly named character recognizable, "
            "with stable silhouettes, colours, proportions and wardrobe throughout."
        )
    elif signals["reference_subject"] and not signals["identity_detail"]:
        parts.append("Use the supplied visual reference for appearance and opening composition; describe only visible changes without reinventing it.")
    else:
        parts.append("Preserve every stated identity, wardrobe, colour and object detail in every frame.")
    if not signals["action"] and not (
        signals["audio_control_supplied"] or signals["motion_control_supplied"]
    ):
        parts.append(
            _concrete_default_chronology(
                signals, precision=precision_open_brief, source=source
            )
        )
    elif signals["audio_control_supplied"] and not signals["action"]:
        parts.append("Let the supplied audio own timing and motion; describe only its visual interpretation.")
    elif signals["motion_control_supplied"] and not signals["action"]:
        parts.append("Let the supplied Motion Track own the trajectory; do not add a competing path in prose.")
    if signals["action"] and not signals["chronological"] and not (
        signals["audio_control_supplied"] or signals["motion_control_supplied"]
    ):
        parts.append(
            _concrete_default_chronology(
                signals, precision=precision_open_brief, source=source
            )
        )
    if not signals["camera"] and not (signals["boundary_pair"] and not signals["multiple_cuts"]):
        parts.append(_concrete_default_camera(signals, source=source))
    if not (signals["setting"] and signals["look"]):
        parts.append(_concrete_default_setting_and_light(signals, source=source))
    if signals["boundary_pair_discussed"] and not signals["boundary_pair"]:
        parts.append(
            "[Upload the actual Start and End images before treating the described boundary frames as exact visual anchors.]"
        )
    if signals["dialogue"] and not signals["tags"]:
        parts.append("Dialogue: assign every line to an explicit stable speaker tag.")
    if signals["dialogue"] and signals["tags"] and not signals["voice_count"]:
        parts.append("State exactly how many voices exist and identify every silent character.")
    if precision_open_brief:
        if meeting_scene:
            parts.append(
                f"Performance and continuity: keep {meeting_scene['lead']}, {meeting_scene['others']} recognizable and "
                f"geographically distinct, preserve their scale and screen positions, and keep {meeting_scene['location']} fixed throughout. "
                "Use restrained eye-led reactions and physically plausible secondary motion; do not turn the meeting into a chase, collision or unrelated event."
            )
        elif _has(r"\b(?:race|racing|racetrack|raceway|race course|find|search)\b", signals["text"]):
            parts.append(
                "Performance and continuity: keep the action restrained and readable; let gaze lead head and body movement, "
                "preserve the lead's scale and screen direction, and keep the raceway geography stable throughout."
            )
        else:
            parts.append(
                "Performance and continuity: keep the action restrained and readable; let gaze lead head and body movement, "
                "preserve every subject's scale and screen direction, and keep the setting geography stable throughout."
            )
        if not signals["dialogue"]:
            if meeting_scene:
                parts.append(
                    "Synchronized audio consists only of faint outdoor ambience, a soft breeze and subtle movement sounds during the approach; "
                    "no dialogue, narration, music or offscreen voices."
                )
            elif _has(r"\b(?:race|racing|racetrack|raceway|race course|find|search)\b", signals["text"]):
                parts.append(
                    "Synchronized audio consists only of light raceway ambience, soft wind movement and one subtle motion accent "
                    "on the decisive turn; no dialogue, narration, music or offscreen voices."
                )
            else:
                parts.append(
                    "Synchronized audio consists only of natural ambience and restrained movement sounds; "
                    "no dialogue, narration, music or offscreen voices."
                )
    avoid = "duplicate subjects, extra limbs or fingers, identity or wardrobe drift, unreadable text and unintended camera movement"
    if not signals["multiple_cuts"]:
        avoid = "cuts, " + avoid
    if precision_open_brief:
        avoid += ", abrupt pose snapping, character swaps, changing subject scale, contradictory screen direction and shifting background geography"
    parts.append("Avoid: " + avoid + ".")
    rewrite = " ".join(parts)
    if len(rewrite) > MAX_REWRITE_CHARS or len(rewrite.encode("utf-8")) > MAX_REWRITE_BYTES:
        raise CoachInputError("the full prompt plus Coach guidance exceeds the local process safety ceiling")
    return rewrite


def analyse_prompt(
    prompt: str,
    context: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    signals = _signals(prompt, context)
    reference_owns_identity = signals["reference_subject"] or bool(
        context["references"].get("end_frame")
    )
    audio_owns_timing = signals["audio_control_supplied"]
    motion_owns_timing = signals["motion_control_supplied"]
    strengths: list[str] = []
    improvements: list[str] = []
    risks: list[str] = []
    score = 0
    if signals["identity_detail"]:
        score += 20
        strengths.append("Subject identity or fixed visual details are explicit.")
    elif signals["boundary_pair"]:
        score += 20
        strengths.append("Loaded Start and End frames own the two boundary compositions, so redundant endpoint appearance prose is not required.")
    elif signals["start_frame_supplied"]:
        score += 20
        strengths.append("A loaded Start frame owns the opening appearance/composition, so redundant identity prose is not required.")
    elif signals["end_frame_supplied"]:
        score += 20
        strengths.append("A loaded End frame owns the final appearance/composition, so redundant endpoint identity prose is not required.")
    elif reference_owns_identity:
        score += 20
        strengths.append("A loaded visual reference supplies subject appearance, so redundant identity prose is not required.")
    elif signals["subject_basic"]:
        score += 12
        strengths.append("A subject is present, but its fixed identity is still loose.")
        improvements.append("Add fixed face/object details, wardrobe, colours and what must remain unchanged.")
    else:
        improvements.append("Name the main subject and its fixed visual identity.")
    if signals["action"]:
        score += 20
        strengths.append("A visible main action is described.")
    elif audio_owns_timing:
        score += 20
        strengths.append("The supplied audio owns timing and motion; the prompt can stay focused on visual interpretation.")
    elif motion_owns_timing:
        score += 20
        strengths.append("The supplied Motion Track owns the trajectory; a competing motion description is not required.")
    else:
        improvements.append("Add one physically achievable main action.")
    if signals["camera"]:
        score += 15
        strengths.append("Camera framing or movement is specified.")
    elif signals["boundary_pair"] and not signals["multiple_cuts"]:
        improvements.append(
            "Use camera-lock continuity by default: match position, lens/field of view, horizon, perspective, framing, subject distance and scale from Start to End."
        )
    else:
        improvements.append("Choose one framing and one camera move—or a locked camera.")
    if signals["setting"] and signals["look"]:
        score += 15
        strengths.append("Setting and lighting/style are both defined.")
    elif signals["setting"] or signals["look"]:
        score += 8
        strengths.append("The setting or visual treatment is partly defined.")
        improvements.append("Add the missing location, time, light source or visual treatment.")
    else:
        improvements.append("Specify location, time of day, lighting, atmosphere and style.")
    if signals["chronological"]:
        score += 15
        strengths.append("Timing or chronological flow is explicit.")
    elif audio_owns_timing:
        score += 15
        strengths.append("The supplied audio owns timing and supplies the shot chronology.")
    elif motion_owns_timing:
        score += 15
        strengths.append("Chronology is supplied by the timed Motion Track.")
    elif signals["action"] and not signals["multi_beat"]:
        score += 9
        improvements.append("State how the shot begins and ends, even for one action.")
    else:
        improvements.append("Order the action as begins → then → ends within one shot.")
    if signals["boundary_pair"] and not signals["boundary_contract"]:
        improvements.append(
            "State that the loaded Start frame is the exact opening target and the loaded End frame is the exact final target, then describe only the smallest physically plausible bridge between them."
        )
    if signals["boundary_pair_discussed"] and not signals["start_frame_supplied"]:
        improvements.append(
            "Upload the actual Start image or create it in Frame Composer; a Start description alone conditions the prompt and does not create an exact visual anchor."
        )
    if signals["boundary_pair_discussed"] and not signals["end_frame_supplied"]:
        improvements.append(
            "Upload the actual End image or create it in Frame Composer; an End description alone conditions the prompt and does not create an exact visual anchor."
        )
    if signals["continuity"]:
        score += 10
        strengths.append("Continuity or avoid constraints are present.")
    else:
        improvements.append("Add continuity rules and a short avoid clause.")
    if len(signals["camera_terms"]) <= 2:
        score += 5
    else:
        improvements.append("Resolve conflicting camera instructions; keep one framing/move per explicit shot.")
    if signals["multiple_cuts"]:
        if signals["duration"] <= 5:
            risks.append("LTX-2.5 supports explicit multi-shot prompts, but several shots in five seconds or less leave little usable time; use one shot or lengthen/split the sequence.")
        else:
            risks.append("Multi-shot is supported, but name each of two to four transitions and re-establish recurring identity, framing and audio continuity.")
    if len(signals["camera_terms"]) > 2:
        risks.append("Several camera instructions may conflict; keep one framing plus one movement.")
    if signals["boundary_pair"] and not signals["multiple_cuts"]:
        if signals["camera_move"] and not signals["quantified_camera_move"]:
            improvements.append(
                "State the deliberate camera move quantitatively and chronologically, and confirm that the End frame supports its final framing."
            )
            risks.append(
                "An unquantified camera move can drift away from the supplied End composition; Prompt Coach cannot inspect the pixels to verify compatibility."
            )
        elif signals["camera_move"] and signals["quantified_camera_move"]:
            strengths.append("The deliberate boundary-to-boundary camera move has an explicit amount or timing.")
        elif signals["camera_lock"]:
            strengths.append("The same-shot boundary camera is explicitly locked.")
    if signals["camera_move"] and signals["camera_lock"]:
        improvements.append("Resolve the conflicting locked-camera and camera-movement instructions.")
        risks.append("The prompt requests both a camera lock and a camera move, which can destabilize framing.")
    if signals["crowd"]:
        risks.append("Crowds increase duplicate-person and face drift risk; reduce the visible cast.")
    if signals["readable_text"]:
        risks.append("Exact readable text and logos are unreliable; composite them afterward.")
    if signals["duration"] > 5 and (
        signals["multi_beat"] or signals["complex_body"] or signals["multi_reference"]
    ):
        risks.append("This is a complex take over five seconds; split it into three-second clips and continue or stitch.")
    if context["mode"] == "ingredients" and context["aspect_ratio"] != "profile":
        risks.append(
            "The selected Ingredients aspect is experimental and outside its 768×448 landscape training bucket; identity, anatomy, reference placement and composition may weaken."
        )
        improvements.append(
            "Compare one fixed-seed low-profile canary against the Preferred profile-default landscape before relying on this Ingredients aspect."
        )
    if context["mode"] == "ingredients" and context["profile"] == "trained5s":
        risks.append(
            "The official 768×448 Ingredients bucket is model-valid but has high peak memory and Python/Metal termination risk on this 16 GB Mac."
        )
    if context["mode"] == "motion" and context["profile"] == "motion3s-experimental":
        risks.append(
            "The selected 640×384 Motion profile is a three-second-only experiment with materially higher attention, swapping, runtime and Python/Metal termination risk than Preferred 512×320."
        )
        improvements.append(
            "Use one simple path and no unnecessary anchors for the canary; return to Preferred 512×320 plus upscaling when reliability matters."
        )
    if signals["dialogue"] and not signals["tags"]:
        score -= 5
        risks.append("Dialogue has no explicit @speaker tag, so voices can be intermixed.")
    elif signals["dialogue"]:
        if not signals["voice_count"]:
            improvements.append("State exactly how many voices are permitted and identify silent characters when needed.")
        if not signals["voice_specification"]:
            improvements.append(
                "For native speech, add a stable reusable voice direction covering pitch/register, weight or texture, pacing/articulation and scene delivery; use Character Voices for dependable recurring identity."
            )
    if signals["complex_body"]:
        risks.append("Detailed hands and complex body motion can hallucinate even with a complete prompt.")
    if signals["boundary_pair"]:
        risks.append(
            "Start and End frames anchor endpoint compositions, not persistent identity between them; use Ingredients when several referenced identities must remain distinguishable throughout."
        )
        risks.append(
            "Prompt Coach can verify that both boundary files are loaded but cannot inspect their pixels; describe any important pose, gaze, crop or prop difference, and prepare both stills for the selected aspect ratio."
        )
    if motion_owns_timing and signals["tracked"]:
        risks.append(
            "The supplied Motion Tracks own the trajectory; remove or soften any competing prose path that could contradict them."
        )
    if signals["unresolved_placeholders"]:
        score = min(score, 89)
        improvements.append(
            "Replace every square-bracket placeholder with a concrete choice before treating the prompt as complete."
        )
    score = min(100, max(0, score))
    if not strengths:
        strengths.append("The core idea is present, but it needs production details before rendering.")
    if not improvements and not risks:
        improvements.append("Render a three-second test; 100% readiness still cannot guarantee the model output.")

    workflow, workflow_reason, availability_notes = _choose_workflow(signals, context, readiness)
    semantic_best, _semantic_reason = _semantic_workflow(signals, context)
    workflow_conflicts: list[str] = []
    if signals["tracked"] and signals["multi_reference"]:
        conflict = "Ingredients and Motion Track cannot run in the same render; choose identity priority or trajectory priority, then use the other as a separate stage/approximation."
        workflow_conflicts.append(conflict)
        risks.append(conflict)
    if (workflow == "ingredients" or semantic_best == "ingredients") and signals["invalid_machine_tags"]:
        invalid_copy = ", ".join(signals["invalid_machine_tags"])
        improvements.append(
            f"Replace invalid Ingredients tag(s) {invalid_copy} with lowercase machine-safe forms such as @subject_a."
        )
        risks.append("Ingredients accepts only lowercase tags containing letters, numbers or underscores after @.")
    complex_long = signals["duration"] > 5 and not signals["multiple_cuts"] and (
        signals["multi_beat"] or signals["complex_body"] or signals["multi_reference"]
    )
    duration = 3 if complex_long else max(1, min(10, signals["duration"]))
    aspect = signals["aspect"]
    if workflow == "ingredients":
        profile = "ingredients5s"
        duration = 5
        aspect = "profile"
    elif workflow == "motion":
        profile = "control3s-plus"
    elif workflow == "union":
        profile = "control3s"
    elif workflow == "generate":
        profile = "draft" if signals["testing"] else "quality"
    else:
        profile = context["profile"]
    walkthrough: list[str] = []
    if workflow == "ingredients":
        walkthrough.extend(
            [
                "Select Ingredients and add one clean reference per character or object; give each a stable @tag.",
                "Mention every active @tag and start with the Preferred 384×224 profile-default landscape. Other aspects and the official 768×448 memory tier are experimental on this Mac.",
            ]
        )
    elif workflow == "motion":
        walkthrough.extend(
            [
                "Select Motion Track, upload the required Start frame and enter the normalized path JSON.",
                "Keep the first test to three seconds at Preferred 512×320 and describe the subject that follows the path; treat 640×384 as a one-track canary only.",
            ]
        )
    elif workflow == "union":
        walkthrough.extend(
            [
                "Select Union Control and upload a timed guide video, not a static storyboard strip.",
                "Start with Canny and three seconds; describe appearance without fighting the guide's structure.",
            ]
        )
    elif workflow == "extend":
        walkthrough.extend(
            [
                "Select Extend, upload the source and choose whether the new material belongs before or after it.",
                "Describe only what happens in the new interval; Extend inherits source dimensions and aspect ratio.",
            ]
        )
    elif workflow == "retake":
        walkthrough.extend(
            [
                "Select Retake, upload the source and mark the exact start and end of the unwanted interval.",
                "Describe only the replacement action and preserve source audio when it must remain unchanged.",
            ]
        )
    elif workflow == "a2v":
        walkthrough.extend(
            [
                "Select Audio-to-Video, upload the audio and set its start offset and clip duration.",
                "Add a Start frame when identity/composition matters, then describe visible motion caused by the sound.",
            ]
        )
    elif workflow == "generate":
        walkthrough.append(
            f"Select Generate, use {profile}, {duration} second{'s' if duration != 1 else ''}, and {aspect if aspect != 'profile' else 'the profile-default aspect ratio'}."
        )
        if signals["boundary_pair"]:
            walkthrough.append(
                "In Start & End Frames, keep both images loaded as the authoritative opening and final targets, composed for the same selected aspect; add an Interior anchor only for one genuinely critical midpoint."
            )
            if signals["multiple_cuts"]:
                walkthrough.append(
                    "Treat those frames as the boundaries of the whole requested sequence, keep only its named cuts, and describe each transition chronologically toward End."
                )
            else:
                walkthrough.append(
                    "Write the smallest physically plausible chronological bridge and let the action settle on the End frame. Keep camera position, lens/FOV, perspective, horizon, framing, distance and subject scale matched unless you intentionally request a chronological move supported by End."
                )
        elif signals["boundary_pair_discussed"]:
            walkthrough.append(
                "Upload both actual images under Start & End Frames—or create each still in Frame Composer—because text descriptions alone do not create visual anchors."
            )
        elif signals["identity_detail"] or signals["tags"]:
            walkthrough.append("Load the Character Bible entry or a clean Start frame and preserve the same name/tag.")
        if signals["multi_beat"]:
            walkthrough.append("Add short chronological Scene Beats whose total remains one continuous shot.")
    else:
        walkthrough.append("No video workflow is runnable right now; resolve the base Generate readiness issue shown above before rendering.")
    if signals["dialogue"] and workflow != "unavailable":
        walkthrough.append("After the picture works, create one Character Voice cue per @speaker and mux those cues into a separate copy.")
    if workflow != "unavailable":
        walkthrough.append("Keep the seed, render a short test and change only one variable on the next attempt.")

    verdict = (
        "Strong specification"
        if score >= 90
        else "Usable, with meaningful gaps"
        if score >= 70
        else "Risky and under-specified"
        if score >= 45
        else "Too open-ended for a controlled take"
    )
    render_risk = "high" if any(
        marker in " ".join(risks).lower()
        for marker in ("cannot run", "five seconds or less", "complex take", "crowds", "detailed hands")
    ) else "medium" if risks else "low"
    recommendation_certainty = (
        "low" if workflow_conflicts else "medium" if workflow != semantic_best else "high"
    )
    return {
        "score": score,
        "score_meaning": "prompt readiness, not probability of render success",
        "verdict": verdict,
        "strengths": strengths,
        "improvements": improvements,
        "risks": risks,
        "workflow": workflow,
        "semantic_best": semantic_best,
        "best_runnable": workflow,
        "workflow_reason": workflow_reason,
        "workflow_conflicts": workflow_conflicts,
        "boundary_frame_guidance": {
            "active": signals["boundary_pair"],
            "discussed": signals["boundary_pair_discussed"],
            "start_loaded": signals["start_frame_supplied"],
            "end_loaded": signals["end_frame_supplied"],
            "prompt_has_exact_boundary_contract": signals["boundary_contract"],
            "multiple_cuts": signals["multiple_cuts"],
            "camera_lock_default": signals["boundary_pair"]
            and not signals["multiple_cuts"]
            and not signals["camera_move"],
            "deliberate_camera_move": signals["camera_move"],
            "quantified_camera_move": signals["quantified_camera_move"],
        },
        "render_risk": render_risk,
        "recommendation_certainty": recommendation_certainty,
        "availability_notes": availability_notes,
        "workflow_readiness": _readiness_snapshot(readiness),
        "settings": {
            "mode": workflow,
            "profile": profile,
            "aspect_ratio": aspect,
            "duration_seconds": duration,
            "seed": context["seed"],
        },
        "walkthrough": walkthrough,
        "rewrite": _rewrite(signals, duration=duration, aspect=aspect),
    }


def _fallback_answer(
    message: str,
    analysis: dict[str, Any],
    context: dict[str, Any],
    *,
    task: str = "review",
    source_prompt: str = "",
) -> str:
    lower = message.lower()
    score = analysis["score"]
    workflow = analysis["workflow"]
    workflow_label = "no runnable local workflow" if workflow == "unavailable" else workflow.title()
    settings = analysis["settings"]
    aspect_label = (
        "the profile-default trained-bucket aspect"
        if settings["aspect_ratio"] == "profile"
        else settings["aspect_ratio"]
    )
    if workflow == "unavailable":
        return (
            f"Your prompt is {score}% ready, but no local video workflow is runnable right now. "
            f"{analysis['workflow_reason'].capitalize()}. Resolve the verified base-model readiness issue shown below; "
            "changing prompt wording cannot bypass missing or incomplete assets."
        )
    if task == "develop":
        concept_source = _production_concept_source(source_prompt)
        meeting = _meeting_scene_brief(concept_source)
        if meeting:
            if _classic_cartoon_style_requested(concept_source):
                question_summary = (
                    "The optional questions below leave that visual treatment intact and ask only about unresolved "
                    "reference-image, final-payoff and dialogue choices."
                )
            else:
                question_summary = (
                    "The optional questions below ask only about the unresolved visual treatment, final payoff and dialogue choice."
                )
            return (
                f"I expanded {meeting['lead']}'s meeting with {meeting['others']} at {meeting['location']} into a complete "
                f"{settings['duration_seconds']}-second continuous shot for {workflow_label} at {aspect_label}. "
                f"As a reversible staging choice, {meeting['lead']} approaches while {meeting['others']} are already readable beside the landmark; "
                "the camera stays in one medium-wide three-shot and the final beat holds their shared reaction. "
                "No reference images, exact dialogue or audio contract were supplied, so the draft preserves each named identity, uses restrained silent acting and does not pretend assets are loaded. "
                + question_summary
            )
        named_targets = _stable_named_phrases(concept_source)
        target_copy = named_targets[0] if named_targets else "the requested target"
        if named_targets:
            # Stable-name comparison is intentionally case-insensitive, but
            # creator-facing prose should retain the normalized display case
            # (for example, ``Rowan Kite`` rather than ``rowan kite``).
            display_match = re.search(
                r"(?<!\w)" + re.escape(named_targets[0]) + r"(?!\w)",
                concept_source,
                flags=re.IGNORECASE,
            )
            if display_match:
                target_copy = display_match.group(0)
        search_assumption = ""
        if _has(r"\b(?:find|search|look(?:ing)? for)\b", source_prompt):
            search_assumption = (
                f" I assumed {target_copy} is the search target but remains unrevealed in this short take, "
                "so the final beat lands on a clue and a clear new direction rather than an overcrowded meeting."
            )
        return (
            f"I developed the rough idea into a complete {settings['duration_seconds']}-second continuous shot for "
            f"{workflow_label} at {aspect_label}.{search_assumption} Because no lead identity, reference image or audio "
            "contract was supplied, the draft uses an unnamed stable lead, a restrained three-beat performance, one "
            "camera move, simple ambience and no dialogue; it does not pretend that Start/End frames or character "
            "references are loaded. The draft below is usable now. The three follow-ups target the choices that would "
            "materially improve identity, the ending and visual/audio direction."
        )
    boundary_guidance = analysis.get("boundary_frame_guidance", {})
    if (boundary_guidance.get("active") or boundary_guidance.get("discussed")) and _has(
        r"\b(?:first|start|opening|last|end|final)[ -]?frames?\b|\b(?:bridge|boundary|boundaries|endpoint|endpoints)\b",
        lower,
    ):
        if not boundary_guidance.get("active"):
            return (
                "Your prompt discusses Start and End frames, but text descriptions alone do not create exact visual anchors. "
                "Upload both actual images in Start & End Frames—or create and assign each still with Frame Composer—then "
                "describe only the smallest physically plausible bridge. Compose both stills for the selected aspect ratio; "
                "Studio center-crops mismatched sources. Prompt Coach can verify file presence but cannot inspect their pixels."
            )
        if boundary_guidance.get("multiple_cuts"):
            return (
                "Use the loaded Start frame as the exact opening target for the whole sequence and the End frame as its "
                "exact final target. Keep the creator's explicitly named cuts, describe each transition chronologically, "
                "preserve only attributes that remain invariant across both endpoints, and encode the visible deltas required "
                "by End. Do not add extra transitions. Coach can confirm that the files are loaded but cannot inspect their "
                "pixels, so describe important pose, crop, lighting or prop differences in the prompt."
            )
        return (
            "Use the loaded Start and End frames as authoritative endpoint targets: begin on the Start composition, "
            "describe only the smallest physically plausible continuous change between them, and finish with a brief hold "
            "on the End composition. Preserve attributes that remain invariant across both frames and explicitly describe only "
            "the visible deltas required by End. If no deliberate camera move is requested, match camera position and height, "
            "lens/field of view, horizon, perspective, framing, subject distance and scale throughout. If a move is intended, "
            "state its amount and timing without inventing values, and confirm that your End image supports its final framing. "
            "If the endpoints require a cut, teleport, identity swap or "
            "abrupt camera change, split the shot or add one critical Interior anchor. These controls guide boundaries but "
            "cannot guarantee pixel-exact reproduction or persistent identity between them. Coach can confirm that the files "
            "are loaded but cannot inspect their pixels, so describe any important pose, crop or prop difference in the prompt."
        )
    if _has(
        r"\b(?:hallucinat(?:e|es|ed|ing|ion|ions)|artifact|deform(?:ed|ity|ities)?|extra limbs?|extra fingers?|"
        r"anatomy|identity drift|wardrobe drift|consisten(?:cy|t)|duplicate subjects?)\b",
        lower,
    ):
        # Use only findings produced by the deterministic prompt analysis. This
        # keeps the guided answer prompt-specific without guessing at model
        # capabilities or promising that wording can eliminate generation risk.
        advice = [str(item).strip().rstrip(".") for item in analysis["risks"][:2] if str(item).strip()]
        if not advice:
            advice = [
                str(item).strip().rstrip(".")
                for item in analysis["improvements"][:2]
                if str(item).strip()
            ]
        if advice:
            specific = "; ".join(advice) + "."
        else:
            specific = "No major structural conflict was detected in this prompt."
        return (
            "Prompt wording can reduce hallucination risk, but it cannot eliminate it. "
            f"For this prompt: {specific} Use the verified {workflow_label} setup at "
            f"{settings['duration_seconds']} second{'s' if settings['duration_seconds'] != 1 else ''} with "
            f"{aspect_label}; keep the seed fixed, test once, and change only one variable per retry."
        )
    if _has(r"\b(?:why|compare|instead|difference)\b", lower):
        availability = " ".join(analysis["availability_notes"][:2])
        recommendation = (
            "Generate is already the recommended route"
            if workflow == "generate" and "currently unavailable" not in analysis["workflow_reason"]
            else f"I recommend {workflow_label}"
        )
        return (
            f"For this prompt, {recommendation} because {analysis['workflow_reason']}. {availability} "
            f"Use {settings['profile']} at {settings['duration_seconds']} second"
            f"{'s' if settings['duration_seconds'] != 1 else ''} with {aspect_label}. "
            f"The {score}% figure measures prompt completeness, not the chance of a successful render."
        )
    if _has(r"\b(?:rewrite|improve|fix|better prompt|make it)\b", lower):
        return (
            f"I rewrote it as one feasible continuous take and preserved its core idea. It is currently {score}% ready; "
            "review the stated creative assumptions before using the complete draft in Studio."
        )
    if _has(r"\b(?:how|walk me|steps?|apply|where|setup|settings?|profile|aspect|duration|seed)\b", lower):
        steps = [str(item).strip() for item in analysis["walkthrough"][:3] if str(item).strip()]
        step_copy = " ".join(f"{index}) {step}" for index, step in enumerate(steps, start=1))
        return (
            f"For this prompt, set {workflow_label} to {settings['profile']}, "
            f"{settings['duration_seconds']} second{'s' if settings['duration_seconds'] != 1 else ''}, "
            f"{aspect_label}, and seed {settings['seed']}. {step_copy}"
        )
    if not analysis["risks"]:
        risk_copy = "No major structural conflict was detected, although anatomy and identity can still drift."
    else:
        risk_copy = analysis["risks"][0]
    return (
        f"This prompt is {score}% ready ({analysis['verdict'].lower()}). "
        f"The best available route is {workflow_label} because {analysis['workflow_reason']}. "
        f"Main concern: {risk_copy} I have put the exact settings, workflow and safer draft below."
    )


def _audio_workflow_guidance(
    analysis: dict[str, Any], context: dict[str, Any], source_prompt: str
) -> str:
    """Answer the audio tutorial as a production decision, not a score report."""

    a2v_state = analysis.get("workflow_readiness", {}).get("a2v", {})
    a2v_eligible = isinstance(a2v_state, dict) and a2v_state.get("availability") == "eligible"
    a2v_note = str(a2v_state.get("note", "")).strip() if isinstance(a2v_state, dict) else ""
    audio_loaded = bool(context.get("references", {}).get("audio"))
    actual_tags = list(dict.fromkeys(re.findall(r"@[A-Za-z][A-Za-z0-9_]*", source_prompt)))
    tag_example = actual_tags[0] if actual_tags else "an actual character tag such as @maya"
    has_dialogue = _has_quoted_dialogue(source_prompt) or _has_non_negated(
        r"\b(?:dialogue|speech|says|speaks|voiceover|voice-over|narrat|talking)\b",
        source_prompt,
    )

    if a2v_eligible and audio_loaded:
        availability = (
            "The audio file is loaded and Audio-to-Video is installed and eligible for submit-time preflight."
        )
    elif a2v_eligible:
        availability = (
            "Audio-to-Video is installed, but this learning setup has not loaded an audio file; "
            "trim and upload the final track before rendering."
        )
    else:
        detail = f" ({a2v_note})" if a2v_note else ""
        availability = (
            "Audio-to-Video is the semantic match, but it is currently blocked on this Mac"
            f"{detail}. Generate can rehearse the picture, but it will not follow exact beats or phonemes."
        )

    dialogue_choice = (
        "The prompt includes speech, but native generated dialogue is only the quick-preview option: "
        "speaker tags can guide it, not guarantee separation."
        if has_dialogue
        else "Native generated dialogue is not needed here because the prompt contains no authored spoken line."
    )
    return (
        "For this shot, let Audio-to-Video own the picture timing: the supplied track is meant to drive "
        "the performer's rhythm, expression and head movement. "
        f"{dialogue_choice} If you later add distinct spoken lines and voices must never intermix, finish the picture first, "
        "then use Character Voices as a separate pass with one timed cue and one assigned preset per real character tag. "
        f"Use {tag_example}; “@speaker” in the question is an example pattern, not a literal character to add. "
        f"{availability} No render or media job was started."
    )


def _frame_composer_guidance() -> str:
    """Explain the Studio's real still-image workflow instead of inventing footage."""

    return (
        "Use Frame Composer for a still image in this Studio. Expand Frame Composer, add one to four "
        "reference photos, give each ingredient a stable tag such as @maya, and mention every active tag "
        "once in the composition prompt. Choose the matched LTX canvas, then click Compose frame locally. "
        "When the still is ready, assign it as the video’s Start, End, or Interior frame. This local composer "
        "is reference-based: it requires at least one photo, so it is not a blank text-only image generator. "
        "Character Bible saves reusable identity; Ingredients keeps several references consistent through "
        "a video. No image or media job was started."
    )


def _workflow_guidance_agent(
    intent: str,
    request: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any] | None:
    """Return verified direct guidance for known control-comparison questions."""

    if intent == "frame_composer_guidance":
        return {
            "provider": "workflow-guidance",
            "provider_label": f"Verified Studio workflow guidance · {SKILL_LABEL}",
            "provider_note": (
                "Answered directly from the verified Studio controls; no conversational model "
                "or media job was started."
            ),
            "provider_action": "",
            "answer": _frame_composer_guidance(),
            "rewrite": request["prompt"] or analysis["rewrite"],
            "follow_ups": [
                "Is this still meant to become a Start, End, or Interior video frame?",
                "Which character or object reference photos should define it?",
                "Which final aspect ratio should the composition match?",
            ],
        }
    if intent != "audio_workflow_guidance":
        return None
    return {
        "provider": "workflow-guidance",
        "provider_label": f"Verified Studio workflow guidance · {SKILL_LABEL}",
        "provider_note": (
            "Answered directly from the verified Studio controls and current local readiness; "
            "no conversational model or media job was started."
        ),
        "provider_action": "",
        "answer": _audio_workflow_guidance(
            analysis, request["context"], request["prompt"]
        ),
        # A control-comparison question did not ask for a rewrite. Preserve the
        # creator's prompt verbatim instead of manufacturing a generic draft.
        "rewrite": request["prompt"] or analysis["rewrite"],
        "follow_ups": [
            "Is the supplied track final audio, or should it guide motion only?",
            "Will any character speak or sing, and do you need exact lip-sync?",
            "Which real @character tags need separate Character Voices cues?",
        ],
    }


def _retrieved_chat_fallback(
    request: dict[str, Any], analysis: dict[str, Any]
) -> tuple[str, list[str]] | None:
    """Answer from retrieved product facts when the local writer cannot pass QA.

    This is topic retrieval over the capability registry—not a list of
    question-shaped canned responses.  It deliberately prefers accurate,
    composable product facts over falling back to prompt scoring.
    """

    skill = _skill_context_for_model(request, analysis)
    capabilities = skill.get("relevant_studio_capabilities", {})
    workflows = skill.get("relevant_workflows", {})
    statuses = skill.get("local_workflow_status", {})
    paragraphs: list[str] = []

    if isinstance(capabilities, dict):
        for capability in list(capabilities.values())[:2]:
            if not isinstance(capability, dict):
                continue
            label = str(capability.get("label", "Relevant Studio control")).strip()
            purpose = str(capability.get("purpose", "")).strip()
            controls = str(capability.get("controls", "")).strip()
            limits = str(capability.get("limits", "")).strip()
            natural_purpose = purpose[:1].upper() + purpose[1:] if purpose else ""
            paragraph = f"{label}: {natural_purpose}." if purpose else f"Use {label}."
            if controls:
                paragraph += f" In Studio: {controls[:1].upper() + controls[1:]}."
            if limits:
                paragraph += f" Important limitation: {limits}."
            paragraphs.append(paragraph)

    relevant_modes = []
    if isinstance(workflows, dict):
        relevant_modes = [mode for mode in workflows if mode != "generate"]
        if (
            not relevant_modes
            and not paragraphs
            and "generate" in workflows
            and _has(r"\b(?:generate|text[ -]to[ -]video|new shot)\b", request["message"])
        ):
            # Generate is always present as the baseline, so use it only when
            # no more specific retrieved topic exists.
            relevant_modes = ["generate"]
    for mode in relevant_modes[:1]:
        workflow = workflows.get(mode, {})
        if not isinstance(workflow, dict):
            continue
        label = str(workflow.get("label", mode.replace("_", " ").title())).strip()
        use_when = str(workflow.get("use_when", "")).strip()
        controls = str(workflow.get("controls", "")).strip()
        limits = workflow.get("limits", ())
        paragraph = f"{label} is the relevant render lane when {use_when}."
        if controls:
            paragraph += f" Set it up with {controls}."
        if isinstance(limits, (list, tuple)) and limits:
            paragraph += " Important limitation: " + " ".join(
                str(item).strip() for item in limits[:2] if str(item).strip()
            )
        state = statuses.get(mode, {}) if isinstance(statuses, dict) else {}
        if isinstance(state, dict):
            availability = str(state.get("availability", "unknown")).strip()
            note = str(state.get("note", "")).strip()
            if availability == "eligible":
                paragraph += " It is installed and eligible for submit-time preflight on this Mac."
            elif availability == "blocked":
                paragraph += f" It is currently blocked on this Mac{': ' + note if note else '.'}"
        paragraphs.append(paragraph)

    if not paragraphs:
        return None
    paragraphs.append(
        "I preserved your video prompt exactly; this answer changes no prompt text and starts no media job."
    )
    return "\n\n".join(paragraphs), []


def _score_report_agent(
    request: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Return a deterministic audit without rewriting or starting an AI model.

    Scoring and conversation are deliberately separate actions. A creator who
    presses Score is asking for the current prompt to be measured as written,
    not for a conversational provider to reinterpret it or silently improve
    the text before displaying the result.
    """

    score = int(analysis["score"])
    gap_candidates = [
        str(item).strip()
        for item in [*analysis.get("improvements", []), *analysis.get("risks", [])]
        if str(item).strip()
    ]
    gaps = list(dict.fromkeys(gap_candidates))[:2]
    if score >= 100 and not analysis.get("risks"):
        gap_copy = (
            " No structural prompt gap was detected; a complete prompt still cannot "
            "guarantee anatomy, identity, motion or audio in the generated video."
        )
    elif gaps:
        gap_copy = " Highest-value gap" + ("s" if len(gaps) != 1 else "") + ": "
        gap_copy += " ".join(
            f"{index}) {gap}" for index, gap in enumerate(gaps, start=1)
        )
    else:
        gap_copy = " No additional structural gap was detected."
    return {
        "provider": "deterministic-score",
        "provider_label": f"Deterministic prompt score · {SKILL_LABEL}",
        "provider_note": (
            "Scored locally from the prompt as written and the verified Studio context; "
            "no conversational model or media job was started."
        ),
        "provider_action": "",
        "answer": (
            f"{score}/100 measures prompt readiness, not the probability of a successful render. "
            f"{analysis['verdict']}.{gap_copy}"
        ),
        # Preserve every byte of the creator's prompt. Score never rewrites.
        "rewrite": request["prompt"],
        "follow_ups": [],
    }


def _fallback_follow_ups(
    analysis: dict[str, Any], *, task: str = "review", source_prompt: str = ""
) -> list[str]:
    """Offer questions that remain truthful for the verified workflow."""
    if task == "develop":
        source = _production_concept_source(source_prompt)
        meeting = _meeting_scene_brief(source)
        if meeting:
            cast = f"{meeting['lead']}, {meeting['others']}"
            questions: list[str] = []
            if _classic_cartoon_style_requested(source):
                if not _has(r"\b(?:reference images?|reference photos?|start frame|end frame)\b", source):
                    questions.append(
                        f"Do you want clean reference images to define {cast} beside {meeting['location']} while preserving the specified classic theatrical 2D cel treatment?"
                    )
            else:
                questions.append(
                    f"Which visual treatment should the meeting at {meeting['location']} use—classic 2D cel, modern 3D, or a supplied reference style?"
                )
            if not _has(
                r"\b(?:final (?:beat|hold|pose|tableau)|ends? (?:on|with)|finishes? (?:on|with))\b",
                source,
            ):
                questions.append(
                    f"What exact emotional or comic reaction should {cast} land on beside {meeting['location']} in the final hold?"
                )
            if not (
                _has_quoted_dialogue(source)
                or _has(
                    r"\b(?:no dialogue|without dialogue|silent|silently|audio|sound|music|voice|narration)\b",
                    source,
                )
            ):
                questions.append(
                    f"Should {cast} remain silent beside {meeting['location']}, or should one exact line be assigned to a named speaker?"
                )
            return questions[:3]
        if _has(r"\b(?:find|search|look(?:ing)? for)\b", source_prompt):
            return [
                "Who is searching, and which fixed appearance or reference must define that lead racer?",
                "Should the final beat reveal the target, find a clue, or end in a comic near-miss?",
                "Which cartoon visual style, reference images and audio tone should this race use?",
            ]
        return [
            "Who is the main subject, and which fixed appearance or reference must define them?",
            "What exact emotional or visual beat should the final second land on?",
            "Which visual style, reference images and audio tone should this shot use?",
        ]
    workflow = str(analysis["workflow"])
    if workflow == "unavailable":
        workflow_question = "What must be ready before I can render this prompt?"
    elif workflow == "generate":
        workflow_question = "Why is Generate the best fit for this prompt?"
    else:
        workflow_question = f"Why is {workflow.title()} better than Generate for this prompt?"
    if (
        analysis.get("boundary_frame_guidance", {}).get("active")
        or analysis.get("boundary_frame_guidance", {}).get("discussed")
    ):
        workflow_question = "How should I bridge these Start and End frames safely?"
    return [
        workflow_question,
        "Walk me through these exact settings.",
        "How can I reduce hallucinations for this prompt?",
    ]


_ANSWER_WORKFLOW_PATTERNS = {
    "generate": r"\bgenerate\b",
    "retake": r"\bretake\b",
    "extend": r"\bextend\b",
    "a2v": r"\b(?:audio[ -]to[ -]video|a2v)\b",
    "ingredients": r"\bingredients?\b",
    "union": r"\b(?:union control|union)\b",
    "motion": r"\b(?:motion track|motion control)\b",
}


def _answer_readiness_contradiction(answer: str, analysis: dict[str, Any]) -> str:
    """Detect a model answer that presents a blocked workflow as usable now."""

    readiness = analysis.get("workflow_readiness", {})
    for mode, pattern in _ANSWER_WORKFLOW_PATTERNS.items():
        state = readiness.get(mode, {}) if isinstance(readiness, dict) else {}
        if not isinstance(state, dict) or state.get("availability") != "blocked":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
            if not re.search(pattern, sentence, flags=re.IGNORECASE):
                continue
            if re.search(
                r"\b(?:not ready|not available|unavailable|blocked|missing|cannot|can't|isn't ready|is not installed|needs? .+ before)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(
                r"\b(?:is|are)\s+(?:installed(?: and)?\s+)?(?:ready|available|usable)|"
                r"\b(?:you can|can now|go ahead and)\s+(?:use|select|choose|run)|"
                r"\b(?:select|choose|use|run)\s+(?:the\s+)?" + pattern,
                sentence,
                flags=re.IGNORECASE,
            ):
                return mode
    return ""


def _conversation_answer_issues(
    answer: str,
    request: dict[str, Any],
) -> list[str]:
    """Audit high-impact Studio claims without grading conversational style.

    The local writer may phrase an answer freely, but it may not turn a
    reference into a guarantee or merge controls that have different
    semantics.  These checks are product invariants, not question templates.
    """

    issues: list[str] = []
    if answer.strip().strip("_*").casefold() == "unchanged":
        issues.append(
            "The answer field must contain the corrected conversational answer; only rewrite may contain __UNCHANGED__."
        )
    absolute_claim_re = re.compile(
        r"\b(?:guarantee(?:s|d)?|perfect(?:ly)? consistent|will consistently|"
        r"locks? (?:the )?identity|identity (?:is|stays|remains|becomes) locked|identity lock|"
        r"will (?:maintain|preserve|keep|ensure) (?:a |the )?(?:same|consistent)|"
        r"will use.{0,60}to maintain consistent|ensur(?:e|es|ed) identity consistency|"
        r"only reliable identity signal|no (?:other|further|additional) changes are needed)\b",
        flags=re.IGNORECASE,
    )
    for match in absolute_claim_re.finditer(answer):
        prefix = answer[max(0, match.start() - 35) : match.start()]
        if re.search(
            r"\b(?:not|never|cannot|can't|doesn't|does not|won't|will not)\s+"
            r"(?:fully\s+|always\s+)?$",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        issues.append(
            "Reference images and tags improve identity consistency but never lock or guarantee it; remove absolute success claims."
        )
        break
    if _has(
        r"\b(?:tag|label|assign)\s+(?:each|every)\s+(?:photo|image|view|reference)\b|"
        r"\b(?:separate|different|unique)\s+(?:@?tags?|labels?)\s+(?:for|to)\s+"
        r"(?:each|every)\s+(?:photo|image|view|reference)\b",
        answer,
    ):
        issues.append(
            "Do not tag each photo of one identity as a separate Ingredients card; assemble the views into one clean reference-sheet image under one unique @tag."
        )
    if _has(
        r"\b(?:each|every)\s+(?:photo|image|view|reference)\s+"
        r"(?:must|needs? to|should)\s+(?:have|use|get|receive)\s+"
        r"(?:a\s+)?(?:unique|different|separate)(?:[\s,]+\w+){0,3}\s+(?:@?tags?|labels?)\b",
        answer,
    ):
        issues.append(
            "One identity must not receive a different tag per view; assemble its views into one reference-sheet image under one unique @tag."
        )
    if _has(
        r"\b(?:assign|apply|use|give)\b.{0,35}\b(?:same|single|one|consistent)\b"
        r".{0,20}\b@?tag\b.{0,35}\b(?:all|each|every)\b.{0,15}\b(?:photos?|images?|views?|references?)\b|"
        r"\b(?:all|each|every)\b.{0,15}\b(?:photos?|images?|views?|references?)\b"
        r".{0,35}\b(?:same|single|one|consistent)\b.{0,20}\b@?tag\b",
        answer,
    ):
        issues.append(
            "The Ingredients UI requires every uploaded reference card tag to be unique; several views of one identity belong in one combined reference-sheet image, not duplicate-tag cards."
        )
    if _has(r"@[A-Za-z][A-Za-z0-9_]*:[A-Za-z0-9_]+", answer):
        issues.append(
            "Use machine-safe tags such as @maya or @subject_a; colon forms such as @tag:maya are invalid."
        )
    if _has(r"\bknown behavior in (?:ltx|ltx[- ]?2(?:\.5)?)\b", answer):
        issues.append(
            "Do not label an explanation as a known LTX behavior unless that claim appears in the verified Studio skill."
        )
    authority = f"{request.get('prompt', '')}\n{request.get('message', '')}"
    invented_example_terms: set[str] = set()
    if _has(r"\b(?:explicitly state|write|put|add)\b.{0,40}[\"“']", answer):
        visual_terms = (
            "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink",
            "coat", "jacket", "dress", "shirt", "trousers", "pants", "hat", "umbrella", "cane", "bag", "weapon",
        )
        for term in visual_terms:
            if _has(rf"\b{re.escape(term)}\b", answer) and not _has(
                rf"\b{re.escape(term)}\b", authority
            ):
                invented_example_terms.add(term)
    if invented_example_terms:
        issues.append(
            "Do not insert unsupplied appearance, wardrobe, colour or prop details into an example prompt: "
            + ", ".join(sorted(invented_example_terms))
            + "."
        )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if not (
            _has(r"\b(?:captur(?:e|ed|ing)|paused?)\b.{0,60}\bframe\b|\bframe\b.{0,60}\bcaptur(?:e|ed|ing)\b", sentence)
            and _has(r"\bextend\b", sentence)
        ):
            continue
        if _has(r"\b(?:not|cannot|can't|doesn't|does not|isn't|is not|rather than|instead of)\b", sentence):
            continue
        issues.append(
            "A captured playback frame can seed a new image-conditioned Generate clip, not native Extend."
        )
        break
    return list(dict.fromkeys(issues))


def _ollama_base_url() -> str | None:
    raw = os.environ.get("PROMPT_COACH_OLLAMA_URL", OLLAMA_DEFAULT_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return raw


def _ollama_json(path: str, payload: dict[str, Any] | None, *, timeout: float) -> dict[str, Any]:
    base = _ollama_base_url()
    if base is None:
        raise RuntimeError("Prompt Coach Ollama URL must be a localhost HTTP URL.")
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    class _RejectRedirects(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
            return None

    try:
        # Redirects are deliberately disabled. Checking the final URL after a
        # redirect would be too late: urllib may already have sent local prompt
        # data to a remote Location target.
        with build_opener(ProxyHandler({}), _RejectRedirects()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise RuntimeError(f"local Ollama service is unavailable ({type(exc).__name__})") from None
    if len(raw) > MAX_MODEL_RESPONSE_BYTES:
        raise RuntimeError("local Ollama response is too large")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("local Ollama returned invalid JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError("local Ollama returned an unexpected response")
    return value


def _ollama_executable() -> Path:
    override = os.environ.get("PROMPT_COACH_OLLAMA_EXECUTABLE", "").strip()
    candidate = Path(override).expanduser() if override else _PINNED_OLLAMA_EXECUTABLE
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    raise RuntimeError("the pinned private Ollama runtime is not installed")


def _start_private_ollama() -> subprocess.Popen[Any] | None:
    """Return an owned localhost server, or ``None`` for an existing one."""

    try:
        _ollama_json("/api/tags", None, timeout=0.8)
        return None
    except RuntimeError:
        pass
    base = _ollama_base_url()
    if base is None:
        raise RuntimeError("Prompt Coach Ollama URL must be a localhost HTTP URL.")
    parsed = urlsplit(base)
    executable = _ollama_executable()
    runtime_env = os.environ.copy()
    runtime_env.update(
        {
            "OLLAMA_HOST": parsed.netloc,
            "OLLAMA_MODELS": str(PATHS.ollama_models_root),
            "OLLAMA_CONTEXT_LENGTH": "8192",
            "OLLAMA_KEEP_ALIVE": "0",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_NO_CLOUD": "1",
        }
    )
    try:
        process = subprocess.Popen(
            [str(executable), "serve"],
            cwd=str(executable.parent),
            env=runtime_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"the pinned private Ollama runtime could not start ({type(exc).__name__})"
        ) from None
    with _OLLAMA_PROCESS_LOCK:
        _OWNED_OLLAMA_PROCESSES.add(process)
    deadline = time.monotonic() + OLLAMA_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            returncode = process.returncode
            _stop_private_ollama(process)
            raise RuntimeError(
                f"the pinned private Ollama runtime exited during startup ({returncode})"
            )
        try:
            _ollama_json("/api/tags", None, timeout=0.8)
            return process
        except RuntimeError:
            time.sleep(0.15)
    _stop_private_ollama(process)
    raise RuntimeError("the pinned private Ollama runtime did not become ready")


def _stop_private_ollama(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    finally:
        with _OLLAMA_PROCESS_LOCK:
            _OWNED_OLLAMA_PROCESSES.discard(process)


def shutdown_local_runtime() -> None:
    """Stop any private Coach server still owned during app shutdown."""

    with _OLLAMA_PROCESS_LOCK:
        processes = list(_OWNED_OLLAMA_PROCESSES)
    for process in processes:
        _stop_private_ollama(process)


def _model_sort_key(name: str) -> tuple[int, str]:
    lowered = name.lower()
    for index, preferred in enumerate(MODEL_NAME_PREFERENCE):
        if lowered == preferred or lowered.startswith(preferred + ":"):
            return index, lowered
    return len(MODEL_NAME_PREFERENCE), lowered


def _select_ollama_model(tags: dict[str, Any]) -> str | None:
    forced = os.environ.get("PROMPT_COACH_OLLAMA_MODEL", "").strip()
    models = tags.get("models")
    if not isinstance(models, list):
        return None
    candidates: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        size = item.get("size", 0)
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            continue
        if size <= 0 or size > MAX_LOCAL_MODEL_BYTES:
            continue
        candidates.append(name.strip())
    if forced:
        return forced if forced in candidates else None
    return min(candidates, key=_model_sort_key) if candidates else None


def _model_messages(
    request: dict[str, Any], analysis: dict[str, Any]
) -> list[dict[str, str]]:
    creative_completion = _creative_completion_policy(request, analysis)
    interaction_mode = _interaction_mode(request)
    develop_directive = (
        _DEVELOP_MODE_MODEL_DIRECTIVE
        if creative_completion["task"] == "develop"
        else ""
    )
    conversation_directive = (
        "This turn is ordinary conversation, not prompt editing. Answer the user's actual question "
        "in warm, specific prose, taking account of recent turns and the current Studio state. Do not "
        "lead with a readiness percentage, audit the prompt, or manufacture a video draft. If the user "
        "asks how or why, explain the reasoning and give concrete next steps. If a requested capability "
        "is absent or uncertain, say so plainly. Treat every purpose and limit in relevant_studio_capabilities "
        "as literal product truth and never blend similarly named controls. In particular, a frame captured "
        "from playback can seed a new image-conditioned Generate clip but cannot be used as native Extend, "
        "carry audio, or preserve hidden motion. In the dedicated Ingredients lane, each uploaded reference "
        "card must have a unique stable @tag. Several views of one identity must first be assembled into one "
        "clean reference-sheet image under one tag; do not assign either different identity tags or a duplicate "
        "tag to separate photos of the same identity. References improve consistency but never guarantee or "
        "lock identity. Use valid tags such as @maya, never @tag:maya. Character Bible references are useful "
        "but are not the true Ingredients identity workflow. "
        "Do not invent example face, wardrobe, colour or prop facts that the creator did not provide. "
        "Set rewrite to the literal token __UNCHANGED__; the "
        "application will preserve the prompt itself. Follow-up questions must be useful for the current "
        "topic, not the generic identity/camera/style checklist. "
        if interaction_mode == "conversation"
        else ""
    )
    output_contract = (
        "Return only a JSON object with three keys: answer (a direct plain-text reply, at most 300 words), "
        "rewrite (the literal token __UNCHANGED__), and follow_ups (up to three natural questions or next "
        "steps relevant to this conversation). Do not use Markdown in answer."
        if interaction_mode == "conversation"
        else (
            "Return only a JSON object with three keys: answer (plain text, at most 170 words and briefly "
            "naming any creative assumptions), rewrite (one complete production-ready prompt), and follow_ups "
            "(up to three specific, optional questions about consequential unresolved identity, final-beat, "
            "style/reference or audio choices). Do not use Markdown in answer."
        )
    )
    system = (
        "You are Dang Studio's local LTX-2.5 Prompt Coach. Be a warm, concise production partner, not a generic chatbot. "
        "Treat installed-feature readiness, hard workflow constraints and actual Studio controls as ground truth. "
        "Use the supplied ltx_production_skill as the domain procedure; verified readiness and settings override it if they ever differ. "
        "If that skill includes a conditional first-last-frame reference, use it only for the current boundary-frame question and never import example character names or voices. "
        "The deterministic semantic score, strengths and risks are advisory: use natural-language judgment to explain nuances they miss, without changing verified availability. "
        "Never claim that a prompt or 100% readiness guarantees a successful render. Never invent an installed feature. "
        "Distinguish reference identity, timed structure, motion paths and audio timing. Prefer one short continuous shot for a local test, but preserve an explicit two-to-four-shot request and structure its transitions honestly. "
        "Obey creative_completion.task. "
        f"{conversation_directive} "
        f"{develop_directive} "
        "For a sparse three-to-five-second develop request, the rewrite must still be a richly specific 220-380-word working draft unless the creator asks for brevity. "
        "Make reversible, disclosed creative decisions about blocking, acting/reaction, one camera plan, environment, light, secondary motion, sound and targeted failure prevention; do not merely paraphrase the idea or return a skeletal outline. "
        "Keep the physical action feasible for the duration. Follow-up questions must name the creator's actual characters, landmark or action and target choices that would materially change the shot—such as the comic/emotional payoff, visual era/reference and exact dialogue/audio contract. Never ask for a generic 'main subject' when the cast is already named. "
        "In review mode, refine conservatively. For a follow-up, revise the latest Current production draft in recent conversation rather than restarting from the rough original. "
        "Treat the Studio prompt, user-role history and current user message as the creator-authored fact contract. Assistant-role history is draft context only and cannot authorize a new name, @tag, line of dialogue, plot event or production constraint. "
        f"{output_contract}"
    )
    report = json.dumps(
        {
            "prompt_under_review": request["prompt"],
            "interaction_mode": interaction_mode,
            "current_studio_context": request["context"],
            "verified_analysis": _analysis_for_model(analysis),
            "creative_completion": creative_completion,
            "ltx_production_skill": _skill_context_for_model(request, analysis),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Verified local production context:\n" + report},
        {
            "role": "assistant",
            "content": "I will keep all recommendations consistent with that verified local context.",
        },
    ]
    messages.extend(request["history"][-6:])
    messages.append({"role": "user", "content": request["message"]})
    return messages


def _classic_cartoon_development_issues(rewrite: str, authority: str) -> list[str]:
    """Keep a sparse classic-cartoon completion physical and creator-grounded.

    A classic cartoon style is an art/performance instruction, not permission
    to import remembered franchise business.  This narrow audit deliberately
    applies only when the creator explicitly requested that treatment.
    """

    if not _classic_cartoon_style_requested(authority):
        return []

    issues: list[str] = []
    if not _classic_cartoon_style_requested(rewrite):
        issues.append(
            "preserve the creator's explicit classic 2D cel-cartoon visual treatment"
        )

    unrequested_families = {
        "magic, supernatural effects or environmental warping": (
            r"\b(?:magic(?:al)?|enchanted|spell|supernatural|mystical|portal|teleport(?:s|ed|ing)?|"
            r"morph(?:s|ed|ing)?|transform(?:s|ed|ing)?|energy field|glowing aura)\b|"
            r"\b(?:monument|ground|earth|landscape|air|reality)\s+"
            r"(?:begins?\s+to\s+)?(?:glows?|warps?|distorts?|ripples?|bends?)\b"
        ),
        "franchise lore or an invented relationship": (
            r"\b(?:old|long[- ]time|historic|familiar) rivals?\b|\b(?:nemesis|sworn enem(?:y|ies)|"
            r"shared history|familiar feud|hunter and prey|old friends?|long[- ]lost|reunion)\b"
        ),
        "a weapon": (
            r"\b(?:weapon|rifle|shotgun|gun|pistol|sword|knife|mallet|hammer|anvil|dynamite|bomb)s?\b"
        ),
        "a costume or wardrobe detail": (
            r"\b(?:costume|uniform|hat|cap|coat|jacket|vest|shirt|trousers|pants|dress|"
            r"boots|shoes|gloves|bow[- ]tie)s?\b"
        ),
        "a handheld or gag prop": (
            r"\b(?:carrot|placard|sign|cane|umbrella|book|bag|box|crate|rope|net|"
            r"chair|table|vehicle)s?\b"
        ),
    }
    introduced = [
        label
        for label, pattern in unrequested_families.items()
        if not _has_non_negated(pattern, authority)
        and _has_non_negated(pattern, rewrite)
    ]
    if introduced:
        issues.append(
            "remove unrequested " + ", ".join(introduced)
            + "; classic-cartoon styling does not authorize new story facts or objects"
        )

    has_cause = _has(
        r"\b(?:when|as soon as|after|caus(?:e|es|ed|ing)|prompt(?:s|ed|ing)|"
        r"in response|which makes|so that)\b",
        rewrite,
    )
    has_physical_action = _has(
        r"\b(?:approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|enter(?:s|ed|ing)?|"
        r"step(?:s|ped|ping)?|stop(?:s|ped|ping)?|halt(?:s|ed|ing)?|"
        r"brake(?:s|d|ing)?|lean(?:s|ed|ing)?|turn(?:s|ed|ing)?|"
        r"move(?:s|d|ing)?|reach(?:es|ed|ing)?)\b",
        rewrite,
    )
    has_visible_reaction = _has(
        r"\b(?:recoil(?:s|ed|ing)?|flinch(?:es|ed|ing)?|startl(?:e|es|ed|ing)|"
        r"double[- ]take|lean(?:s|ed|ing)?\s+(?:back|away)|blink(?:s|ed|ing)?|"
        r"freez(?:e|es|ing)|jolt(?:s|ed|ing)?|stiffen(?:s|ed|ing)?|"
        r"squash(?:es|ed|ing)?|stretch(?:es|ed|ing)?|wobbl(?:e|es|ed|ing)|"
        r"stagger(?:s|ed|ing)?|eyes?\s+widen(?:s|ed|ing)?)\b",
        rewrite,
    )
    if not (has_cause and has_physical_action and has_visible_reaction):
        issues.append(
            "stage exactly one concrete cause-and-effect physical reaction gag: one supplied "
            "character's simple movement must visibly trigger another supplied character's "
            "bodily reaction, followed by a readable hold"
        )
    return issues


def _sparse_develop_quality_issues(
    rewrite: str,
    request: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    """Audit a short sparse-idea completion before Qwen can return it.

    This is intentionally narrower than the creator-contract gate.  It does
    not judge taste or require adjective volume; it catches the two concrete
    failure modes seen with compact local writers: returning a skeletal draft
    after being asked to develop an idea, and padding named characters with
    exact appearance, wardrobe or held-prop facts the creator never supplied.
    """

    if _coach_task(request, analysis) != "develop":
        return []
    duration = int(analysis.get("settings", {}).get("duration_seconds", 0) or 0)
    if not 3 <= duration <= 5:
        return []
    authority = _creator_authored_contract(request)
    # Measure sparsity from the idea in the Studio field, not from the UI's
    # intentionally detailed "develop or review" command. That command is
    # submitted as the current user message and used to push a short idea over
    # the old 90-word cutoff in the real browser, silently skipping this audit.
    # Follow-up facts remain in ``authority`` for grounding and for the
    # explicit-brevity check below.
    source_words = re.findall(r"\b[\w'’-]+\b", request.get("prompt", ""))
    if len(source_words) > 90 or _has(
        r"\b(?:brief|briefly|concise|concisely|minimal|one paragraph|under\s+\d+\s+words?)\b",
        authority,
    ):
        return []

    issues: list[str] = []
    rewrite_words = re.findall(r"\b[\w'’-]+\b", rewrite)
    if len(rewrite_words) < 190:
        issues.append(
            f"the production prompt is skeletal ({len(rewrite_words)} words); expand it to 220–380 useful words"
        )

    # A long paragraph can still be the same generic template with more
    # adjectives. For an open 3–5 second idea, require the precision pattern
    # demonstrated by the production skill: explicit, non-overlapping ranges
    # that cover the opening, causal middle beat and readable ending. This is
    # deliberately limited to sparse develop mode and never rewrites a user's
    # already detailed prompt.
    timed_ranges = [
        (float(start), float(end))
        for start, end in re.findall(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:–|—|-|to)\s*"
            r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
            rewrite,
            flags=re.IGNORECASE,
        )
    ]
    useful_ranges = [(start, end) for start, end in timed_ranges if end > start]
    covers_open = bool(useful_ranges) and min(start for start, _ in useful_ranges) <= 0.15
    covers_end = bool(useful_ranges) and max(end for _, end in useful_ranges) >= duration - 0.15
    ordered = all(
        useful_ranges[index][0] >= useful_ranges[index - 1][1] - 0.05
        for index in range(1, len(useful_ranges))
    )
    if len(useful_ranges) < 3 or not covers_open or not covers_end or not ordered:
        issues.append(
            f"add three or four explicit non-overlapping time ranges spanning 0.0–{duration}.0 seconds "
            "(opening state, causal action/reaction, consequence and final hold)"
        )

    # Only inspect clauses containing an explicit creator-supplied name.  This
    # keeps reversible setting, light, camera and ambience choices free while
    # preventing a local model from inventing a blue scarf, telescope, facial hair,
    # etc. merely to make a familiar named character sound specific.
    identity_detail_re = re.compile(
        r"\b(?:wear(?:s|ing)?|dress(?:ed|es|ing)?|clad|holding|holds|carrying|carries|"
        r"armed\s+with|mustache|moustache|beard|hair|hat|cap|jacket|coat|vest|"
        r"shirt|trousers|pants|dress|rifle|gun|pistol|sword|cane)\b",
        flags=re.IGNORECASE,
    )
    authority_detail_terms = {
        match.group(0).casefold() for match in identity_detail_re.finditer(authority)
    }
    names = _stable_named_phrases(authority)
    invented_identity_terms: set[str] = set()
    for clause in re.split(r"(?<=[.!?;])\s+|\n+", rewrite):
        if not any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause, re.IGNORECASE)
            for name in names
        ):
            continue
        for match in identity_detail_re.finditer(clause):
            term = match.group(0).casefold()
            if term in {"holds", "holding"} and re.match(
                r"\s+(?:(?:the|a|an|their|his|her|its|one)\s+)?"
                r"(?:(?:final|ending|end|last|shared|stable|still)\s+){0,2}"
                r"(?:pose|tableau|position|composition|expression|reaction|beat|stillness)\b|"
                r"\s+(?:still|steady)\b",
                clause[match.end() :],
                flags=re.IGNORECASE,
            ):
                continue
            if term not in authority_detail_terms:
                invented_identity_terms.add(term)
    if invented_identity_terms:
        issues.append(
            "remove exact named-character appearance, wardrobe or held-prop details not supplied by the creator "
            f"({', '.join(sorted(invented_identity_terms))})"
        )
    source_signals = _signals(authority, request["context"])
    rewrite_signals = _signals(rewrite, request["context"])
    if rewrite_signals["dialogue"] and not source_signals["dialogue"]:
        issues.append(
            "remove every invented spoken line, quoted utterance, voiceover and narration; "
            "use unquoted ambience or effects only"
        )
    issues.extend(_classic_cartoon_development_issues(rewrite, authority))
    return issues


def _parse_ollama_chat_content(raw: dict[str, Any]) -> dict[str, Any]:
    message = raw.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("local Ollama reply is missing content")
    try:
        content = json.loads(message["content"])
    except json.JSONDecodeError:
        raise RuntimeError("local Ollama reply did not follow the JSON contract") from None
    if not isinstance(content, dict):
        raise RuntimeError("local Ollama reply has an invalid shape")
    return content


def _run_ollama_agent(request: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    owned_service = _start_private_ollama()
    try:
        tags = _ollama_json("/api/tags", None, timeout=1.2)
        model = _select_ollama_model(tags)
        if model is None:
            raise RuntimeError("no complete approved Ollama chat model is installed")
        messages = _model_messages(request, analysis)
        if (
            len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            > MAX_CONVERSATIONAL_MODEL_PROMPT_BYTES
        ):
            raise RuntimeError(
                "full grounded request exceeds the compact Ollama model context; the prompt remains preserved"
            )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "keep_alive": 0,
            "options": {
                "temperature": 0.7 if _coach_task(request, analysis) == "develop" else 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "num_ctx": 8192,
                "num_predict": 1_800 if _coach_task(request, analysis) == "develop" else 800,
            },
        }
        raw = _ollama_json("/api/chat", payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        content = _parse_ollama_chat_content(raw)
        if _interaction_mode(request) == "conversation":
            conversation_issues = _conversation_answer_issues(
                str(content.get("answer", "")), request
            )
            if conversation_issues:
                correction_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are correcting one Dang Studio answer. Return only JSON with exactly "
                            "answer, rewrite and follow_ups. Answer the user's question directly in warm "
                            "plain text with no Markdown. The supplied Studio capability facts and audit "
                            "failures are authoritative. Correct every failure without changing the topic, "
                            "inventing controls, promising success or turning the question into a video "
                            "prompt. Set rewrite to the literal token __UNCHANGED__. Ask at most three "
                            "natural follow-ups relevant to this question. The answer field must contain "
                            "the complete corrected advice—never the word UNCHANGED. Example shape: "
                            '{"answer":"Corrected advice here.","rewrite":"__UNCHANGED__","follow_ups":[]}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": request["message"],
                                "current_studio_context": request["context"],
                                "verified_skill": {
                                    key: value
                                    for key, value in _skill_context_for_model(request, analysis).items()
                                    if key
                                    in {
                                        "score_definition",
                                        "relevant_workflows",
                                        "relevant_studio_capabilities",
                                        "local_workflow_status",
                                        "conditional_reference",
                                    }
                                },
                                "rejected_answer": str(content.get("answer", "")),
                                "audit_failures": conversation_issues,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ]
                correction_payload = dict(payload)
                correction_payload["messages"] = correction_messages
                correction_payload["options"] = dict(payload["options"])
                correction_payload["options"]["temperature"] = 0.15
                correction_payload["options"]["num_predict"] = 900
                corrected_raw = _ollama_json(
                    "/api/chat",
                    correction_payload,
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
                content = _parse_ollama_chat_content(corrected_raw)
                remaining_issues = _conversation_answer_issues(
                    str(content.get("answer", "")), request
                )
                if remaining_issues:
                    raise RuntimeError(
                        "Qwen's factual self-revision still conflicted with verified Studio capability rules: "
                        + "; ".join(remaining_issues)
                    )
        first_rewrite = _clean_rewrite(content.get("rewrite"))
        quality_issues = _sparse_develop_quality_issues(
            first_rewrite,
            request,
            analysis,
        )
        if quality_issues:
            exact_duration = int(
                analysis.get("settings", {}).get("duration_seconds", 0) or 0
            )
            creator_contract = _creator_authored_contract(request)
            masked_contract, identity_restore = _masked_identity_contract(
                creator_contract,
                _stable_named_phrases(creator_contract),
            )
            masked_prompt, _ = _masked_identity_contract(
                request["prompt"],
                _stable_named_phrases(creator_contract),
            )
            classic_cartoon_correction = ""
            if _classic_cartoon_style_requested(creator_contract):
                classic_cartoon_correction = (
                    "The creator's classic theatrical 2D cel-cartoon direction controls only "
                    "rendering and performance. Build exactly one concrete physical reaction gag: "
                    "one SUBJECT's simple movement visibly causes another SUBJECT's bodily reaction, "
                    "then hold the result. Preserve every supplied style cue. Add no magic, glow or "
                    "environmental warping, franchise lore, relationship, weapon, costume, dialogue "
                    "or handheld prop unless it appears in creator_contract. "
                )
            # Start the bounded correction in a compact fresh session. Appending
            # the full skill report, the rejected answer and another long
            # instruction pushed real browser requests beyond the 24 KiB local
            # context gate, so the correction never ran even though the first
            # draft was properly rejected. The creator contract below is the
            # only factual authority; the first candidate is explicitly
            # untrusted diagnostic material.
            revision_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Dang Studio's final LTX-2.5 prompt editor. Return only a JSON object "
                        "with exactly answer, rewrite and follow_ups. The creator_contract and "
                        "verified_settings are authoritative. The audit_failures describe facts that "
                        "must be removed: never invent a costume, body detail, prop, weapon, lore, "
                        "relationship, named entity or spoken line. Preserve every "
                        "creator-supplied SUBJECT label, location, duration, aspect, action, style and "
                        "constraint. SUBJECT labels are opaque identities: repeat them exactly and never "
                        "infer canon, appearance, wardrobe, accessories, weapons or biography from them. "
                        "The rewrite must be a genuinely production-ready 220–380 words and at least "
                        "sixteen complete sentences, never a short summary. Organize it as: (1) three "
                        "sentences for shot contract, known identities and continuity; (2) four labeled, "
                        "non-overlapping timeline blocks, each with two concrete sentences and a literal "
                        "start–end range, beginning at 0.0 seconds and ending at "
                        f"{exact_duration}.0 seconds; (3) camera and visual treatment; (4) audio; and "
                        "(5) targeted avoid rules. The four blocks must stage an opening state, one "
                        "causal action/reaction, its consequence and a readable final hold. Keep stable "
                        "geography and scale, the creator's exact visual treatment, restrained secondary "
                        "motion and explicit audio ownership. If no dialogue was supplied, use no speech. "
                        f"{classic_cartoon_correction}"
                        "Do not claim any loaded "
                        "reference that verified_settings does not show. The answer briefly explains the "
                        "safe creative assumptions. Ask exactly three optional, contextual questions using "
                        "the actual character and location names."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "creator_contract": masked_contract,
                            "prompt_under_review": masked_prompt,
                            "opaque_identity_labels": list(identity_restore),
                            "verified_settings": {
                                "mode": request["context"]["mode"],
                                "profile": request["context"]["profile"],
                                "aspect_ratio": request["context"]["aspect_ratio"],
                                "duration_seconds": exact_duration,
                                "references": request["context"]["references"],
                            },
                            "audit_failures": quality_issues,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
            if (
                len(
                    json.dumps(
                        revision_messages,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                > MAX_CONVERSATIONAL_MODEL_PROMPT_BYTES
            ):
                raise RuntimeError(
                    "the sparse-draft correction exceeds the compact Ollama model context"
                )
            revised_payload = dict(payload)
            revised_payload["messages"] = revision_messages
            revised_payload["options"] = dict(payload["options"])
            revised_payload["options"]["temperature"] = 0.45
            revised_payload["options"]["num_predict"] = 2_200
            revised_raw = _ollama_json(
                "/api/chat",
                revised_payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            content = _parse_ollama_chat_content(revised_raw)
            content = _restore_masked_identities(content, identity_restore)
            revised_rewrite = _clean_rewrite(content.get("rewrite"))
            remaining_issues = _sparse_develop_quality_issues(
                revised_rewrite,
                request,
                analysis,
            )
            if remaining_issues:
                raise RuntimeError(
                    "Qwen's bounded self-revision did not satisfy the sparse-development quality contract: "
                    + "; ".join(remaining_issues)
                )
    finally:
        _stop_private_ollama(owned_service)
    answer = _clean_model_answer(content.get("answer"))
    # Return the model's candidate verbatim (after size/type validation). The
    # provider orchestrator must see an invalid tag edit or semantic drift so
    # it can reject this provider and try the next one. Silently replacing it
    # here with the deterministic draft would make an unsafe candidate look as
    # though it passed creator-contract validation.
    rewrite = _clean_rewrite(content.get("rewrite"))
    follow_ups = content.get("follow_ups", [])
    if not isinstance(follow_ups, list):
        follow_ups = []
    clean_follow_ups = []
    for item in follow_ups[:3]:
        if isinstance(item, str) and item.strip():
            clean_follow_ups.append(item.strip()[:160])
    return {
        "provider": "ollama",
        "provider_label": f"Local AI · {model} · {SKILL_LABEL}",
        "provider_note": f"Generated privately using the versioned local {SKILL_LABEL}; Qwen and its private localhost runtime were unloaded after this reply.",
        "provider_action": "",
        "answer": answer,
        "rewrite": rewrite,
        "follow_ups": clean_follow_ups,
    }


_APPLE_RESPONSE_SCHEMA = json.dumps(
    {
        "title": "CoachReply",
        "type": "object",
        "x-order": ["answer", "rewrite", "follow_ups"],
        "properties": {
            "answer": {
                "type": "string",
                "description": "Warm, concrete production advice in at most 170 words, without Markdown.",
            },
            "rewrite": {
                "type": "string",
                "description": "Use the literal token __UNCHANGED__ for an ordinary conversation turn. Otherwise return one complete LTX precision prompt; preserve the premise, named entities and their roles. In develop mode use verified duration/aspect, causal non-overlapping timed beats through a final hold, one camera plan, continuity/props/geography, feasible motion, exact audio ownership and targeted avoid rules. Invent no proper names, dialogue or reference claims; no brackets, TODOs or TBDs.",
            },
            "follow_ups": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Up to three short, specific questions about consequential unresolved identity, final beat, style/reference or audio choices; never blockers to completing the current draft.",
            },
        },
        "required": ["answer", "rewrite", "follow_ups"],
        "additionalProperties": False,
    },
    separators=(",", ":"),
)


def _apple_fm_executable() -> str:
    candidate = "/usr/bin/fm"
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    discovered = shutil.which("fm")
    if discovered:
        return discovered
    raise RuntimeError("Apple Foundation Models CLI is not installed")


def _apple_fm_license_ready(executable: str) -> None:
    try:
        result = subprocess.run(
            [executable, "license", "--status"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Apple AI license status could not be checked ({type(exc).__name__})") from None
    if result.returncode == 0:
        return
    combined = (result.stdout + " " + result.stderr).strip().lower()
    if result.returncode == 69 or "not agreed" in combined or "have not agreed" in combined:
        raise RuntimeError(
            "Apple Foundation Models CLI terms are not agreed; review and accept them manually with `sudo fm license`"
        )
    raise RuntimeError(f"Apple Foundation Models CLI is unavailable (status {result.returncode})")


def _apple_fm_available(executable: str) -> None:
    try:
        result = subprocess.run(
            [executable, "available", "--model", "system"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Apple AI availability could not be checked ({type(exc).__name__})") from None
    combined = (result.stdout + " " + result.stderr).strip()
    lowered = combined.lower()
    if result.returncode == 0 and not any(
        marker in lowered for marker in ("unavailable", "not ready", "modelnotready")
    ):
        return
    if "modelnotready" in lowered or "not ready" in lowered:
        raise RuntimeError(
            "Apple Foundation Models system model is not ready (modelNotReady); macOS manages its download automatically"
        )
    suffix = f": {combined.splitlines()[-1][:240]}" if combined else ""
    raise RuntimeError(f"Apple Foundation Models system model is unavailable{suffix}")


def _apple_prompt(request: dict[str, Any], analysis: dict[str, Any]) -> str:
    # Keep the one-shot transcript compact enough for the on-device context
    # window. The deterministic report remains the authority; prior turns are
    # useful conversational context, not an unbounded memory store.
    history: list[dict[str, str]] = []
    history_chars = 0
    for item in reversed(request["history"][-6:]):
        remaining = 4_000 - history_chars
        if remaining <= 0:
            break
        content = item["content"][:remaining]
        history.append({"role": item["role"], "content": content})
        history_chars += len(content)
    history.reverse()
    allowed_analysis_keys = (
        "score",
        "score_meaning",
        "verdict",
        "strengths",
        "improvements",
        "risks",
        "workflow",
        "workflow_reason",
        "availability_notes",
        "settings",
        "walkthrough",
    )
    compact_analysis = {
        key: analysis[key] for key in allowed_analysis_keys if key in analysis
    }
    return json.dumps(
        {
            "prompt_under_review": request["prompt"],
            "interaction_mode": _interaction_mode(request),
            "current_studio_context": request["context"],
            "verified_analysis": compact_analysis,
            "creative_completion": _creative_completion_policy(request, analysis),
            "ltx_production_skill": _skill_context_for_model(request, analysis),
            "recent_conversation": history,
            "user_message": request["message"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _ordered_subsequence(required: list[str], observed: list[str]) -> bool:
    """Return whether ``required`` occurs in order inside ``observed``."""

    cursor = iter(observed)
    return all(any(item == expected for item in cursor) for expected in required)


_GENERIC_INSTRUCTION_TAGS = {
    "@character",
    "@ingredient",
    "@object",
    "@person",
    "@speaker",
    "@subject",
    "@tag",
    "@voice",
}


def _stable_creator_tags(source_prompt: str, creator_contract: str) -> list[str]:
    """Return concrete creator tags, excluding later instructional examples.

    Every tag used by the actual Studio prompt is authoritative, even when it
    happens to have a generic-looking name.  In later coaching/tutorial turns,
    however, tokens such as ``@speaker`` and ``@character`` are documentation
    placeholders rather than newly introduced cast identities.  Excluding only
    those known placeholders prevents a question like “keep each @speaker
    separate” from forcing the generated video prompt to contain a fictional
    speaker, while a real later addition such as ``@maya`` remains protected.
    """

    source_tags = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", source_prompt)
    source_tag_keys = {tag.casefold() for tag in source_tags}
    authority_tags = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", creator_contract)
    return [
        tag
        for tag in authority_tags
        if tag.casefold() in source_tag_keys
        or tag.casefold() not in _GENERIC_INSTRUCTION_TAGS
    ]


def _preserve_prompt_tags(
    candidate: str,
    source_prompt: str,
    fallback: str,
    *,
    creator_contract: str | None = None,
) -> str:
    """Reject a rewrite that drops, reorders, or invents stable ``@tags``.

    Tags already present in the Studio prompt must survive in their original
    order, including deliberate repeated speaker bindings. A creator may add
    a new concrete tag in a later user turn; that tag then becomes required and
    may be used by the rewrite. Known tutorial placeholders such as
    ``@speaker`` are ignored when they appear only in those later instructions.
    Assistant-authored tags never enter ``creator_contract`` and therefore
    remain unauthorized.
    """

    authority = creator_contract if creator_contract is not None else source_prompt
    source_tags = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", source_prompt)
    authority_tags = _stable_creator_tags(source_prompt, authority)
    candidate_tags = re.findall(r"@[A-Za-z][A-Za-z0-9_]*", candidate)
    source_tag_set = set(source_tags)
    authorized = set(authority_tags)
    added_required = list(
        dict.fromkeys(tag for tag in authority_tags if tag not in source_tag_set)
    )
    valid = (
        _ordered_subsequence(source_tags, candidate_tags)
        and all(tag in candidate_tags for tag in added_required)
        and all(tag in authorized for tag in candidate_tags)
    )
    return candidate if valid else fallback


def _explicit_prompt_duration(prompt: str) -> int | None:
    return _explicit_duration_value(prompt)


def _explicit_prompt_aspect(prompt: str) -> str | None:
    return _explicit_aspect_value(prompt)


def _semantic_premise_anchors(text: str) -> set[str]:
    """Return explicit action premises that creative completion must preserve.

    These are deliberately broad semantic families rather than exact words, so
    a useful rewrite may turn ``find`` into ``searches for`` or ``race`` into a
    concrete racetrack action.  The gate only protects premises the creator
    actually supplied; it does not require every finished prompt to contain all
    of these actions.
    """

    families = {
        "search/find": r"\b(?:find|finds|finding|found|locate|locates|located|locating|seek|seeks|seeking|sought|search|searches|searched|searching|look(?:s|ed|ing)? for|hunt(?:s|ed|ing)? for)\b",
        "race": r"\b(?:race|races|raced|racing|racer|racers|racetrack|raceway|race course|finish line|checkered flag)\b",
        "near-miss final beat": r"\b(?:near[- ]miss|narrowly avoid(?:s|ed|ing)?|almost (?:crash(?:es|ed|ing)?|collid(?:e|es|ed|ing)|hit(?:s|ting)?|miss(?:es|ed|ing)?))\b",
        "chase": r"\b(?:chase|chases|chased|chasing|pursue|pursues|pursued|pursuing|pursuit)\b",
        "escape": r"\b(?:escape|escapes|escaped|escaping|flee|flees|fled|fleeing)\b",
        "dance": r"\b(?:dance|dances|danced|dancing|choreograph(?:y|ed|ing)?)\b",
        "fight": r"\b(?:fight|fights|fought|fighting|duel|duels|duelled|dueling)\b",
        "fly": r"\b(?:fly|flies|flew|flying|take(?:s)? off|airborne)\b",
        "transform": r"\b(?:transform|transforms|transformed|transforming|morph|morphs|morphed|morphing)\b",
    }
    return {
        label for label, pattern in families.items() if _has_non_negated(pattern, text)
    }


def _high_confidence_named_identities(text: str) -> set[str]:
    """Extract names used as actors, without treating every capitalised word as one."""

    names: list[str] = []
    name_pattern = r"[A-Z][A-Za-z'’-]{1,30}(?:\s+[A-Z][A-Za-z'’-]{1,30}){0,2}"
    for pattern in (
        rf"\b(?:named|called)\s+({name_pattern})\b",
        rf"\b({name_pattern})\s+(?=(?:is|has|wears|drives|steers|runs|walks|searches|finds|looks|turns|races|crashes|holds|stands|sits|enters|leaves|speaks|says)\b)",
        # Compact models often introduce screenplay-style cast facts such as
        # ``BOB (40s)`` without using "named" or a nearby verb.
        r"\b([A-Z][A-Z'’-]{1,30})\s*\(\s*(?:age\s*)?(?:\d{1,3}s?|child|teen|adult|elderly)\s*\)",
        rf"(?m)^\s*({name_pattern})\s*:",
    ):
        names.extend(match.group(1) for match in re.finditer(pattern, text))

    generic = {
        "a",
        "an",
        "the",
        "camera",
        "visual style",
        "sound design",
        "character blocking",
        "background",
        "lighting",
        "setting",
        "style",
        "sound",
        "performance",
        "continuity",
        "environment",
        "world",
        "avoid",
        "subject",
        "character",
        "lead",
        "hero",
        "racer",
        "driver",
        "shot",
        "audio",
        "action",
        "what",
    }
    normalized: set[str] = set()
    for name in names:
        cleaned = re.sub(r"^(?:The|A|An)\s+", "", name).strip().casefold()
        if cleaned and cleaned not in generic:
            normalized.add(cleaned)
    return normalized


def _explicit_dialogue_lines(text: str) -> set[str]:
    """Extract quoted lines that the creator explicitly assigns as speech."""

    lines: set[str] = set()
    speech = (
        r"(?:say(?:s|ing|said)?|speak(?:s|ing|spoke)?|whisper(?:s|ed|ing)?|"
        r"shout(?:s|ed|ing)?|blurt(?:s|ed|ing)?|ask(?:s|ed|ing)?|"
        r"repl(?:y|ies|ied|ying)|utter(?:s|ed|ing)?|dialogue|line)"
    )
    for pattern in (
        rf"{speech}[^\n.!?]{{0,50}}?[\"“]([^\"”\n]{{1,240}})[\"”]",
        rf"[\"“]([^\"”\n]{{1,240}})[\"”][^\n.!?]{{0,30}}?{speech}",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            normalized = re.sub(r"\s+", " ", match.group(1)).strip().casefold()
            if normalized:
                lines.add(normalized)
    return lines


def _target_occurrence_is_representation(text: str, start: int, end: int) -> bool:
    """Detect when a named character became merchandise, an image, or a label."""

    before = text[max(0, start - 90) : start]
    after = text[end : min(len(text), end + 70)]
    represented_before = re.search(
        r"(?:\b(?:jar|box|bag|crate|case|shelf|collection|display)\b(?:\s+\w+){0,3}\s+"
        r"(?:of|with|containing)|\b(?:photo|image|poster|picture|drawing|painting|statue|"
        r"figurine|figure|toy|doll|costume|mascot|portrait|representation)\s+of|"
        r"\b(?:labelled|labeled|marked|showing|depicting|printed with))\s*$",
        before,
        flags=re.IGNORECASE,
    )
    represented_after = re.match(
        r"\s*(?:[-–—]\s*)?(?:themed\s+)?(?:figure|figurine|toy|doll|statue|poster|"
        r"photo|image|picture|drawing|costume|mascot|merchandise|collectible)s?\b",
        after,
        flags=re.IGNORECASE,
    )
    return bool(represented_before or represented_after)


def _named_target_was_retyped(source: str, candidate: str, target: str) -> bool:
    """Return true when an explicit person/character target became a proxy object."""

    target_re = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)", re.IGNORECASE)
    source_matches = list(target_re.finditer(source))
    if source_matches and all(
        _target_occurrence_is_representation(source, match.start(), match.end())
        for match in source_matches
    ):
        # The creator explicitly requested the representation; preserve it via
        # normal named-target checks rather than reinterpreting it as a person.
        return False
    candidate_matches = list(target_re.finditer(candidate))
    return bool(candidate_matches) and all(
        _target_occurrence_is_representation(candidate, match.start(), match.end())
        for match in candidate_matches
    )


def _introduced_material_events(source: str, candidate: str) -> set[str]:
    """Find newly invented, premise-changing events rather than harmless detail."""

    families = {
        "a crash or collision": r"\b(?:crash|crashes|crashed|crashing|collide|collides|collided|colliding|wreck|wrecks|wrecked|wrecking|smash(?:es|ed|ing) into)\b",
        "an explosion or destructive event": r"\b(?:explode|explodes|exploded|exploding|explosion|detonate|detonates|detonated|destroy|destroys|destroyed|destroying)\b",
        # ``shot`` is intentionally excluded: in this domain it overwhelmingly
        # means a camera shot, not the past tense of shooting somebody.
        "an attack or injury": r"\b(?:attack|attacks|attacked|attacking|stab|stabs|stabbed|shoot|shoots|shooting|injure|injures|injured|injuring|kill|kills|killed|killing)\b",
        "a capture or confinement event": r"\b(?:kidnap|kidnaps|kidnapped|kidnapping|cage|cages|caged|caging|trap|traps|trapped|trapping)\b",
    }
    introduced: set[str] = set()
    for label, pattern in families.items():
        if _has_non_negated(pattern, source) or not _has_non_negated(pattern, candidate):
            continue
        # Near misses remain a reversible visual embellishment, unlike an
        # actual crash/explosion/attack added to an unrelated idea.
        material = False
        for match in re.finditer(pattern, candidate, flags=re.IGNORECASE):
            prefix = candidate[max(0, match.start() - 45) : match.start()]
            if re.search(
                r"\b(?:almost|nearly|nearly avoids?|narrowly avoids?|avoids?|prevents?|"
                r"without|no|never|do not|don't)\s+(?:\w+\s+){0,3}$",
                prefix,
                flags=re.IGNORECASE,
            ):
                continue
            material = True
            break
        if material:
            introduced.add(label)
    return introduced


def _camera_move_families(text: str) -> set[str]:
    """Count distinct directed camera moves, excluding framing and lens prose."""

    families = {
        "tracking": r"\b(?:tracking(?: shot)?|camera tracks?)\b",
        "dolly": r"\bdoll(?:y|ies|ied|ying)\b",
        "push-in": r"\bpush(?:es|ed|ing)?[- ]?in\b",
        "pull-back": r"\bpull(?:s|ed|ing)?[- ]?back\b",
        "pan": r"\b(?:whip[- ]?)?pan(?:s|ned|ning)?\b",
        "tilt": r"\btilt(?:s|ed|ing)?\b",
        "crane": r"\bcrane(?:s|d|ing)?\b",
        "orbit": r"\borbit(?:s|ed|ing)?\b",
        "zoom": r"\bzoom(?:s|ed|ing)?\b",
    }
    return {
        label for label, pattern in families.items() if _has_non_negated(pattern, text)
    }


def _safe_provider_rewrite(
    candidate: str,
    *,
    source_prompt: str,
    creator_contract: str | None = None,
    fallback: str,
    original_analysis: dict[str, Any],
    context: dict[str, Any],
    readiness: dict[str, Any],
) -> tuple[str, str]:
    """Fail closed when generated prose contradicts verified production facts."""
    authority = creator_contract if creator_contract is not None else source_prompt
    preserved = _preserve_prompt_tags(
        candidate,
        source_prompt,
        fallback,
        creator_contract=authority,
    )
    if preserved == fallback and candidate != fallback:
        return fallback, "The AI rewrite changed, dropped or reordered a stable @tag, so the verified safe draft was retained."

    if candidate != fallback and _UNRESOLVED_PLACEHOLDER_RE.search(candidate):
        return (
            fallback,
            "The AI rewrite returned unresolved bracket, TODO or TBD placeholders instead of a finished prompt, so the verified concrete draft was retained.",
        )

    candidate_analysis = analyse_prompt(preserved, context, readiness)
    candidate_signals = _signals(preserved, context)
    reasons: list[str] = []
    if candidate_analysis["score"] < int(original_analysis["score"]) - 10:
        reasons.append("materially reduced prompt readiness")
    expected_duration = int(original_analysis["settings"]["duration_seconds"])
    explicit_duration = _explicit_prompt_duration(preserved)
    if explicit_duration is not None and explicit_duration != expected_duration:
        reasons.append(f"changed the effective duration from {expected_duration}s to {explicit_duration}s")
    expected_aspect = str(original_analysis["settings"]["aspect_ratio"])
    explicit_aspect = _explicit_prompt_aspect(preserved)
    if explicit_aspect is not None and explicit_aspect != expected_aspect:
        reasons.append(f"changed the effective aspect from {expected_aspect} to {explicit_aspect}")
    source_signals = _signals(authority, context)
    source_premises = _semantic_premise_anchors(authority)
    candidate_premises = _semantic_premise_anchors(preserved)
    missing_premises = sorted(source_premises - candidate_premises)
    if missing_premises:
        reasons.append(
            "removed the creator's explicit " + ", ".join(missing_premises) + " premise"
        )
    missing_named_phrases = [
        phrase
        for phrase in _stable_named_phrases(authority)
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", preserved, re.IGNORECASE)
        is None
    ]
    if missing_named_phrases:
        reasons.append(
            "removed or genericized the explicit named character/target "
            + ", ".join(missing_named_phrases)
        )
    retyped_named_targets = [
        phrase
        for phrase in _stable_named_phrases(authority)
        if not _named_target_was_retyped(authority, authority, phrase)
        and _named_target_was_retyped(authority, preserved, phrase)
    ]
    if retyped_named_targets:
        reasons.append(
            "changed the explicit character/target into a proxy object or representation: "
            + ", ".join(retyped_named_targets)
        )
    stable_names = set(_stable_named_phrases(authority))
    source_names = _high_confidence_named_identities(authority) | stable_names
    # Compact writers naturally use an already-introduced character's first
    # name in later timed beats (``Arden turns`` after ``Arden Vale``).  Treat
    # only unambiguous first-token aliases as authorized; this still rejects a
    # genuinely new proper name such as Bob.
    first_name_counts: dict[str, int] = {}
    for name in stable_names:
        parts = name.split()
        if len(parts) > 1:
            first_name_counts[parts[0]] = first_name_counts.get(parts[0], 0) + 1
    source_names.update(
        name for name, count in first_name_counts.items() if count == 1
    )
    invented_names = sorted(_high_confidence_named_identities(preserved) - source_names)
    if invented_names:
        reasons.append("invented a new named character: " + ", ".join(invented_names))
    introduced_events = sorted(_introduced_material_events(authority, preserved))
    if introduced_events:
        reasons.append("invented " + ", ".join(introduced_events))
    source_camera_moves = _camera_move_families(authority)
    candidate_camera_moves = _camera_move_families(preserved)
    if len(candidate_camera_moves) > max(2, len(source_camera_moves)):
        reasons.append(
            "introduced an incoherent stack of camera moves: "
            + ", ".join(sorted(candidate_camera_moves))
        )
    if candidate_signals["dialogue"] and not source_signals["dialogue"]:
        reasons.append("invented dialogue that the creator did not request")
    if candidate_signals["multiple_cuts"] and not source_signals["multiple_cuts"]:
        reasons.append("introduced cuts, a montage or multiple locations")
    if source_signals["multiple_cuts"] and not candidate_signals["multiple_cuts"]:
        reasons.append("removed the original explicit multi-shot structure")
    if (
        original_analysis.get("boundary_frame_guidance", {}).get("active")
        and not candidate_signals["boundary_contract"]
    ):
        reasons.append("removed the exact Start/End boundary contract")
    if source_signals["boundary_pair"]:
        if not source_signals["camera_move"] and candidate_signals["camera_move"]:
            reasons.append("introduced an unrequested boundary-frame camera move")
        if source_signals["camera_move"] and not candidate_signals["camera_move"]:
            reasons.append("removed the deliberate boundary-frame camera move")
        if (
            source_signals["quantified_camera_move"]
            and not candidate_signals["quantified_camera_move"]
        ):
            reasons.append("removed the camera move amount or timing")
    if source_signals["dialogue"] and not candidate_signals["dialogue"]:
        reasons.append("removed the original dialogue")
    missing_dialogue = sorted(
        _explicit_dialogue_lines(authority) - _explicit_dialogue_lines(preserved)
    )
    if missing_dialogue:
        reasons.append(
            "removed or changed creator-authored dialogue: "
            + ", ".join(f'\"{line}\"' for line in missing_dialogue)
        )
    if source_signals["voice_count"] and not candidate_signals["voice_count"]:
        reasons.append("removed the explicit voice-count contract")
    if (
        source_signals["voice_specification"]
        and not candidate_signals["voice_specification"]
    ):
        reasons.append("removed the recurring voice specification")
    if source_signals["silent_voice_guard"] and not candidate_signals["silent_voice_guard"]:
        reasons.append("removed the silent-character voice or mouth rule")
    if source_signals["explicit_silence"] and (
        not candidate_signals["explicit_silence"] or candidate_signals["dialogue"]
    ):
        reasons.append("removed or contradicted the explicit silence/no-dialogue rule")
    if source_signals["camera_lock"] and (
        not candidate_signals["camera_lock"] or candidate_signals["camera_move"]
    ):
        reasons.append("removed or contradicted the locked-camera rule")
    source_requires_one_shot = _has_one_shot_constraint(authority)
    candidate_keeps_one_shot = _has_one_shot_constraint(preserved)
    if source_requires_one_shot and not candidate_keeps_one_shot:
        reasons.append("removed the original one-shot/no-cuts constraint")
    if candidate_analysis["workflow"] != original_analysis["workflow"]:
        reasons.append(
            f"changed the verified workflow from {original_analysis['workflow']} to {candidate_analysis['workflow']}"
        )
    if reasons:
        return fallback, "The AI rewrite was not applied because it " + ", ".join(reasons) + ". The verified safe draft was retained."
    return preserved, ""


def _parse_agent_content(
    content: Any,
    analysis: dict[str, Any],
    *,
    source: str,
    source_prompt: str,
    creator_contract: str | None = None,
) -> dict[str, Any]:
    if isinstance(content, str):
        if len(content.encode("utf-8")) > MAX_MODEL_RESPONSE_BYTES:
            raise RuntimeError(f"{source} response is too large")
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            raise RuntimeError(f"{source} reply did not follow the JSON contract") from None
    if not isinstance(content, dict):
        raise RuntimeError(f"{source} reply has an invalid shape")
    answer = _clean_model_answer(content.get("answer"))
    # Keep the raw candidate for the central creator-contract gate. Provider
    # parsing is intentionally not a repair step: repairing here would hide a
    # bad candidate and prevent clean failover to the next local model.
    rewrite = _clean_rewrite(content.get("rewrite"))
    follow_ups = content.get("follow_ups", [])
    if not isinstance(follow_ups, list):
        follow_ups = []
    clean_follow_ups = []
    for item in follow_ups[:3]:
        if isinstance(item, str) and item.strip():
            clean_follow_ups.append(item.strip()[:160])
    return {"answer": answer, "rewrite": rewrite, "follow_ups": clean_follow_ups}


def _run_apple_agent(request: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    executable = _apple_fm_executable()
    _apple_fm_license_ready(executable)
    _apple_fm_available(executable)
    conversation_directive = (
        "This is an ordinary conversation turn, not a prompt rewrite. Answer the user's actual question "
        "directly and naturally from the verified report and recent conversation. Do not lead with a "
        "score or turn the question into footage. Give concrete reasoning and steps where useful. Never "
        "invent a feature or loaded asset. Treat relevant_studio_capabilities purposes and limits as literal "
        "product truth and never blend similarly named controls. A captured playback frame can seed a new "
        "image-conditioned Generate clip but cannot be used as native Extend or preserve audio/hidden motion. "
        "In the Ingredients lane each uploaded reference card must have a unique stable @tag. Assemble several "
        "views of one identity into one clean reference-sheet image under one tag; do not assign either different "
        "identity tags or a duplicate tag to separate photos of the same identity. References improve consistency "
        "but never guarantee or lock identity. Use valid tags such as @maya, never @tag:maya. Character Bible "
        "references are not the true Ingredients workflow. Do not invent example face, wardrobe, colour or prop facts. "
        "Set rewrite to the literal token __UNCHANGED__; the application "
        "will preserve the full prompt. Use follow_ups only for this topic, never a generic prompt checklist. "
        if _interaction_mode(request) == "conversation"
        else ""
    )
    develop_directive = (
        _DEVELOP_MODE_MODEL_DIRECTIVE
        if _coach_task(request, analysis) == "develop"
        else ""
    )
    instructions = (
        "You are Dang Studio's private on-device LTX-2.5 Prompt Coach. Be a warm, concise production partner. "
        "Installed-feature readiness, hard workflow constraints and actual Studio controls in the supplied report are authoritative. "
        "Use ltx_production_skill as the LTX domain procedure, while verified readiness and settings always win. "
        "Use any conditional first-last-frame reference only for the current boundary-frame question; never import its example character names or voices. "
        "Its semantic score, strengths and risks are advisory; use natural-language judgment to notice nuances while never inventing an "
        "installed feature, never call prompt readiness a render-success probability, and never promise 100% output "
        "quality. Answer the user's current question directly, then improve the prompt only where useful. Prefer a "
        "short continuous shot for a local test, but preserve explicit two-to-four-shot intent and name its transitions. "
        "Obey creative_completion.task. "
        f"{conversation_directive} "
        f"{develop_directive} "
        "For a three-to-five-second develop request, aim for roughly 220-450 useful words when the facts support that precision, without overloading the action. State assumptions briefly in answer. "
        "For review, refine conservatively. For follow-ups, revise the latest Current production draft in conversation instead of restarting. "
        "Treat the Studio prompt, user-role history and current user message as the creator-authored fact contract. Assistant-role history is draft context only and cannot authorize a new name, @tag, line of dialogue, plot event or production constraint. "
        "Keep motion physically feasible. Output only the requested JSON."
    )
    grounded_input = _apple_prompt(request, analysis)
    if len(grounded_input.encode("utf-8")) > MAX_CONVERSATIONAL_MODEL_INPUT_BYTES:
        raise RuntimeError(
            "full grounded request exceeds the compact conversational model context; the prompt remains preserved"
        )
    schema_path = ""
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                prefix="dang-studio-coach-schema-",
                dir=tempfile.gettempdir(),
                delete=False,
            ) as schema_file:
                schema_file.write(_APPLE_RESPONSE_SCHEMA)
                schema_path = schema_file.name
            os.chmod(schema_path, 0o600)
        except OSError as exc:
            raise RuntimeError(f"Prompt Coach could not create its private schema file ({type(exc).__name__})") from None
        command = [
            executable,
            "respond",
            "-m",
            "system",
            "--no-stream",
            "--use-case",
            "general",
            "--instructions",
            instructions,
            "--schema",
            schema_path,
        ]
        if _coach_task(request, analysis) == "review":
            command.insert(command.index("--use-case"), "--greedy")
        try:
            completed = subprocess.run(
                command,
                input=grounded_input,
                capture_output=True,
                text=True,
                timeout=APPLE_FM_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Apple on-device AI timed out") from None
        except OSError as exc:
            raise RuntimeError(f"Apple on-device AI could not start ({type(exc).__name__})") from None
    finally:
        if schema_path:
            try:
                os.unlink(schema_path)
            except OSError:
                pass
    if completed is None:
        raise RuntimeError("Apple on-device AI did not return a response")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1][:240]}" if detail else ""
        raise RuntimeError(f"Apple on-device AI exited with status {completed.returncode}{suffix}")
    parsed = _parse_agent_content(
        completed.stdout.strip(),
        analysis,
        source="Apple on-device AI",
        source_prompt=request["prompt"],
        creator_contract=_creator_authored_contract(request),
    )
    return {
        "provider": "apple-foundation-model",
        "provider_label": f"Apple on-device AI · {SKILL_LABEL}",
        "provider_note": (
            "Generated privately and offline by macOS Foundation Models using the versioned local "
            f"{SKILL_LABEL}; the one-shot model process has exited."
        ),
        "provider_action": "",
        **parsed,
    }


def _validate_local_provider_candidate(
    candidate: dict[str, Any],
    *,
    request: dict[str, Any],
    analysis: dict[str, Any],
    readiness: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Validate one model turn before it can become the selected provider.

    Selection is atomic: answer, rewrite and follow-up questions all come from
    the same provider only after its draft passes the creator-authored fact
    contract. This prevents a rejected Apple interpretation from leaking its
    cast or plot into the chat while the deterministic draft is shown below.
    """

    contradicted_mode = _answer_readiness_contradiction(
        str(candidate.get("answer", "")), analysis
    )
    if contradicted_mode:
        return (
            None,
            f"its answer contradicted verified {contradicted_mode} readiness",
        )
    if _interaction_mode(request) == "conversation":
        factual_issues = _conversation_answer_issues(
            str(candidate.get("answer", "")), request
        )
        if factual_issues:
            return (
                None,
                "its conversational answer conflicted with verified Studio capability rules: "
                + "; ".join(factual_issues),
            )
        # A chat turn is answer-only.  The provider's schema still carries a
        # rewrite field for compatibility with compact local model tooling,
        # but it is never allowed to mutate—or invalidate—the production
        # prompt.  This is the key separation between conversation and Develop.
        selected = dict(candidate)
        selected["rewrite"] = request["prompt"]
        return selected, ""
    try:
        raw_rewrite = _clean_rewrite(candidate.get("rewrite", analysis["rewrite"]))
    except CoachInputError as exc:
        return None, str(exc)
    safe_rewrite, safety_note = _safe_provider_rewrite(
        raw_rewrite,
        source_prompt=request["prompt"],
        creator_contract=_creator_authored_contract(request),
        fallback=analysis["rewrite"],
        original_analysis=analysis,
        context=request["context"],
        readiness=readiness,
    )
    if safety_note:
        # The explanatory sentence is already written for the UI. Remove the
        # fallback suffix because failover is still in progress at this point.
        reason = re.sub(
            r"\s*The verified (?:safe|concrete) draft was retained\.?\s*$",
            "",
            safety_note,
            flags=re.IGNORECASE,
        ).strip()
        return None, reason
    selected = dict(candidate)
    selected["rewrite"] = safe_rewrite
    return selected, ""


def _run_local_agent(
    request: dict[str, Any],
    analysis: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    """Select the first *validated* private provider candidate.

    Each provider is invoked at most once. The caller holds
    ``_MODEL_CALL_LOCK`` for this entire sequence, and it only calls this
    function after media-compute admission has succeeded, so Apple and Ollama
    can neither overlap one another nor a render/upscale job.
    """

    diagnostics: list[str] = []
    public_attempts: list[str] = []
    for label, runner in (
        ("Qwen local AI", _run_ollama_agent),
        ("Apple on-device AI", _run_apple_agent),
    ):
        try:
            candidate = runner(request, analysis)
        except (CoachInputError, RuntimeError, ValueError) as exc:
            diagnostics.append(f"{label}: {exc}")
            public_attempts.append(f"{label} was unavailable for this reply.")
            continue
        selected, rejection = _validate_local_provider_candidate(
            candidate,
            request=request,
            analysis=analysis,
            readiness=readiness,
        )
        if selected is None:
            diagnostics.append(f"{label} candidate rejected: {rejection}")
            public_attempts.append(
                f"{label} returned a draft, but Studio's creator-contract gate rejected it: {rejection}"
            )
            continue
        if public_attempts:
            failover_note = " ".join(public_attempts)
            selected["provider_note"] = (
                str(selected.get("provider_note", "")).rstrip()
                + " Failover audit: "
                + failover_note
                + f" {label} was used only after its candidate passed the same gate."
            ).strip()
        return selected
    raise _LocalProviderExhausted(
        "; ".join(diagnostics) or "no private model provider was available",
        public_audit=" ".join(public_attempts),
    )


def coach(
    payload: Any,
    *,
    readiness: dict[str, Any],
    allow_model: bool = True,
    pause_reason: str = "",
    model_runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return one conversational response plus deterministic production facts.

    ``allow_model`` is false while any video/voice/still/upscale compute is
    active.  The caller owns that admission decision so an LLM can never be
    loaded alongside a render by accident.
    """

    request = validate_request(payload)
    analysis = analyse_prompt(request["prompt"], request["context"], readiness)
    creative_completion = _creative_completion_policy(request, analysis)
    interaction_mode = _interaction_mode(request)
    conversation_intent = _conversation_intent(request["message"])
    score_only = request["action"] == "score"
    agent = (
        _score_report_agent(request, analysis)
        if score_only
        else _workflow_guidance_agent(conversation_intent, request, analysis)
        if interaction_mode == "verified_guidance"
        else None
    )
    failure_note = ""
    provider_failure_audit = ""
    grounded_model_input_too_large = False
    if agent is None and allow_model and model_runner is not None:
        grounded_model_input_too_large = (
            len(_apple_prompt(request, analysis).encode("utf-8"))
            > MAX_CONVERSATIONAL_MODEL_INPUT_BYTES
        )
    # An injected runner has no provider-specific context negotiation, so keep
    # the conservative preflight used by tests/callers. Built-in providers do
    # their own bounded-context checks: an Apple-size rejection must not block
    # an already-installed Ollama model from being tried.
    if agent is not None:
        pass
    elif allow_model and model_runner is not None and grounded_model_input_too_large:
        failure_note = "full grounded request exceeds the compact conversational model context"
    elif allow_model and _MODEL_CALL_LOCK.acquire(blocking=False):
        try:
            try:
                if model_runner is not None:
                    agent = model_runner(request, analysis)
                else:
                    agent = _run_local_agent(request, analysis, readiness)
            except _LocalProviderExhausted as exc:
                failure_note = str(exc)
                provider_failure_audit = exc.public_audit
            except (CoachInputError, RuntimeError, ValueError) as exc:
                failure_note = str(exc)
        finally:
            _MODEL_CALL_LOCK.release()
    elif not allow_model:
        failure_note = pause_reason or "AI inference is paused while local media compute is active"
    else:
        failure_note = "another Prompt Coach AI reply is already running"
    if agent is None:
        guided_kind = (
            "conversation"
            if interaction_mode == "conversation"
            else "guided development"
            if creative_completion["task"] == "develop"
            else "guided review"
        )
        if "compact conversational model context" in failure_note:
            guided_note = (
                f"I’m using {guided_kind} because the prompt plus verified LTX skill context is larger than "
                "the compact conversational model’s working context. The full prompt was preserved; nothing was truncated."
            )
        elif "modelNotReady" in failure_note:
            guided_note = f"I’m using {guided_kind} while Apple’s private on-device conversational model gets ready."
        elif "terms are not agreed" in failure_note:
            guided_note = f"I’m using {guided_kind} until Apple’s on-device model terms are reviewed and accepted."
        elif "local media compute is active" in failure_note:
            guided_note = f"I’m using {guided_kind} because conversational AI is paused while local media compute is active."
        elif "another Prompt Coach AI reply" in failure_note:
            guided_note = f"I’m using {guided_kind} while another local Coach reply finishes."
        else:
            guided_note = f"I’m using {guided_kind} because the private conversational model is not available right now."
        if provider_failure_audit:
            guided_note = (guided_note.rstrip() + " Provider audit: " + provider_failure_audit).strip()
        if interaction_mode == "conversation":
            known_guidance = _workflow_guidance_agent(
                _conversation_intent(request["message"]), request, analysis
            )
            retrieved_guidance = _retrieved_chat_fallback(request, analysis)
            chat_answer = (
                known_guidance["answer"]
                if known_guidance is not None
                else retrieved_guidance[0]
                if retrieved_guidance is not None
                else _fallback_answer(
                    request["message"],
                    analysis,
                    request["context"],
                    task="review",
                    source_prompt=request["prompt"],
                )
                if request["action"] == "interact"
                else (
                    "I couldn't run the local conversational model for this reply, so I won't guess or "
                    "turn your question into a video prompt. Your prompt is unchanged. Try the question "
                    "again when local AI is available; if it concerns an installed feature, I will answer "
                    "from the verified Studio controls and current readiness."
                )
            )
            chat_follow_ups = (
                known_guidance.get("follow_ups", [])
                if known_guidance is not None
                else retrieved_guidance[1]
                if retrieved_guidance is not None
                else _fallback_follow_ups(
                    analysis,
                    task="review",
                    source_prompt=request["prompt"],
                )
                if request["action"] == "interact"
                else []
            )
        else:
            chat_answer = _fallback_answer(
                request["message"],
                analysis,
                request["context"],
                task=creative_completion["task"],
                source_prompt=request["prompt"],
            )
            chat_follow_ups = _fallback_follow_ups(
                analysis,
                task=creative_completion["task"],
                source_prompt=request["prompt"],
            )
        agent = {
            "provider": "guided",
            "provider_label": (
                f"Verified Studio knowledge · {SKILL_LABEL}"
                if interaction_mode == "conversation" and retrieved_guidance is not None
                else f"Conversation paused · {SKILL_LABEL}"
                if interaction_mode == "conversation"
                else f"Guided local fallback · {SKILL_LABEL}"
            ),
            "provider_note": guided_note,
            "provider_action": (
                "For fully conversational offline replies, open Terminal, run `sudo fm license`, review Apple's terms, "
                "and agree only if you accept them; then try Coach again."
                if "terms are not agreed" in failure_note
                else "Apple's on-device system model is not ready yet. First make Siri Language match the Mac language "
                "(on this Mac: System Settings → Apple Intelligence & Siri → Siri Language → English (India)). Then keep "
                "the Mac powered, on Wi-Fi and under light load while macOS completes its automatic model download; "
                "try Coach again afterward. Dang Studio will not download it."
                if "modelNotReady" in failure_note
                else ""
            ),
            "answer": chat_answer,
            "rewrite": (
                request["prompt"]
                if interaction_mode == "conversation"
                else analysis["rewrite"]
            ),
            "follow_ups": chat_follow_ups,
        }
    answer_safety_note = ""
    conversation_fact_issues = (
        _conversation_answer_issues(str(agent.get("answer", "")), request)
        if interaction_mode == "conversation"
        else []
    )
    if conversation_fact_issues:
        agent["answer"] = (
            "The local reply conflicted with verified Studio capability rules, so I withheld it instead "
            "of showing confident but inaccurate advice. Your prompt is unchanged; please ask again."
        )
        agent["follow_ups"] = []
        answer_safety_note = (
            "Conversation fact check withheld the reply: "
            + "; ".join(conversation_fact_issues)
        )
        agent["provider_note"] = (
            str(agent.get("provider_note", "")).rstrip()
            + " "
            + answer_safety_note
        ).strip()
    contradicted_mode = (
        ""
        if score_only
        else _answer_readiness_contradiction(str(agent.get("answer", "")), analysis)
    )
    if contradicted_mode:
        agent["answer"] = (
            f"The local reply conflicted with the verified {contradicted_mode} readiness, so I withheld "
            "that answer instead of guessing. Your prompt is unchanged; ask again and I’ll answer from "
            "the current Studio state."
            if interaction_mode == "conversation"
            else _fallback_answer(
                request["message"],
                analysis,
                request["context"],
                task=creative_completion["task"],
                source_prompt=request["prompt"],
            )
        )
        answer_safety_note = (
            f"The AI answer contradicted verified {contradicted_mode} readiness, so Studio replaced it "
            "with the deterministic grounded answer."
        )
        agent["provider_note"] = (str(agent.get("provider_note", "")).rstrip() + " " + answer_safety_note).strip()
    # This is intentionally enforced after *every* provider, including an
    # injected/test provider. The conversational answer remains useful, but
    # an unsafe model-authored draft can never silently contradict the
    # verified duration, aspect, one-shot, workflow, or stable @tag bindings.
    if score_only or interaction_mode == "conversation":
        agent["rewrite"] = request["prompt"]
        rewrite_safety_note = ""
    else:
        agent["rewrite"], rewrite_safety_note = _safe_provider_rewrite(
            _clean_rewrite(agent.get("rewrite", analysis["rewrite"])),
            source_prompt=request["prompt"],
            creator_contract=_creator_authored_contract(request),
            fallback=analysis["rewrite"],
            original_analysis=analysis,
            context=request["context"],
            readiness=readiness,
        )
    if rewrite_safety_note and creative_completion["task"] == "develop":
        # In development mode the answer and suggested questions are normally
        # derived from the same creative interpretation as the draft.  If the
        # semantic gate rejected that interpretation, retaining its chat copy
        # would leak the same invented cast, plot or premise back into the UI
        # (for example, questions about a crash that the creator never asked
        # for).  Replace the whole creative turn atomically with the grounded
        # skill-guided development while preserving a transparent audit note.
        rejected_provider = str(agent.get("provider_label", "Local AI")).strip()
        agent["answer"] = _fallback_answer(
            request["message"],
            analysis,
            request["context"],
            task=creative_completion["task"],
            source_prompt=request["prompt"],
        )
        agent["follow_ups"] = _fallback_follow_ups(
            analysis,
            task=creative_completion["task"],
            source_prompt=request["prompt"],
        )
        agent["provider"] = "guided"
        agent["provider_label"] = f"Guarded local development · {SKILL_LABEL}"
        rejection_note = (
            f"{rejected_provider} proposed a draft, but Studio's semantic gate rejected its "
            "creative interpretation. The visible answer, draft and follow-up questions come "
            "from the verified local development rules instead."
        )
        agent["provider_note"] = (
            str(agent.get("provider_note", "")).rstrip() + " " + rejection_note
        ).strip()
    rewrite_analysis = analyse_prompt(agent["rewrite"], request["context"], readiness)
    result = dict(analysis)
    result["rewrite"] = agent["rewrite"]
    result["creative_completion"] = creative_completion
    result["response_kind"] = (
        "score_report"
        if score_only
        else "workflow_guidance"
        if interaction_mode == "verified_guidance"
        else "conversation"
        if interaction_mode == "conversation"
        else "prompt_review"
    )
    result.update(
        {
            "provider": agent["provider"],
            "provider_label": agent["provider_label"],
            "provider_note": agent["provider_note"],
            "provider_action": agent.get("provider_action", ""),
            "answer": agent["answer"],
            "follow_ups": agent["follow_ups"],
            "rewrite_safety_note": rewrite_safety_note,
            "answer_safety_note": answer_safety_note,
            "rewrite_analysis": {
                "score": rewrite_analysis["score"],
                "verdict": rewrite_analysis["verdict"],
                "strengths": rewrite_analysis["strengths"],
                "improvements": rewrite_analysis["improvements"],
                "risks": rewrite_analysis["risks"],
                "workflow": rewrite_analysis["workflow"],
                "semantic_best": rewrite_analysis["semantic_best"],
                "best_runnable": rewrite_analysis["best_runnable"],
                "workflow_reason": rewrite_analysis["workflow_reason"],
                "workflow_conflicts": rewrite_analysis["workflow_conflicts"],
                "render_risk": rewrite_analysis["render_risk"],
                "recommendation_certainty": rewrite_analysis["recommendation_certainty"],
                "availability_notes": rewrite_analysis["availability_notes"],
                "settings": rewrite_analysis["settings"],
                "walkthrough": rewrite_analysis["walkthrough"],
            },
            "skill": {
                "id": SKILL_ID,
                "version": SKILL_VERSION,
                "label": SKILL_LABEL,
            },
        }
    )
    return result


__all__ = [
    "CoachInputError",
    "analyse_prompt",
    "coach",
    "validate_request",
]
