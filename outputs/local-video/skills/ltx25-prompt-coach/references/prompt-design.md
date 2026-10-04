# Prompt design

## Development versus review

First decide whether the creator supplied a rough concept or a substantially
specified shot.

For a sparse, conversational, or open-ended concept, develop it into one
coherent production prompt. Choose a physically achievable primary action and
chronology, one compatible camera treatment, a concrete setting, lighting,
visual style, and ambience. These should be sensible, reversible choices that
serve the supplied idea—not placeholders for the creator to fill in. Briefly
state material assumptions and keep follow-up questions as optional variations
after the completed draft.

For a detailed prompt, review conservatively. Preserve the creator's decisions
and change only omissions, contradictions, overload, or workflow mismatches
that materially affect controllability.

In both modes, preserve exact explicit names, identity facts, relationships,
dialogue, `@tags`, and constraints. Do not add a new named character, dialogue,
lore, or biography unless requested. A production draft must not contain
square-bracket placeholders, `TODO`, `TBD`, or other fill-in-the-blank
instructions.

When a sparse idea omits a consequential identity or story choice, choose the
least committal assumption that permits a coherent shot and disclose it. Do not
fill the space by inventing exact facial features, body details, wardrobe,
biography, relationships, dialogue, or a claim that reference frames were
supplied. The completed draft comes first. After it, ask at most three optional
questions focused on: who or identity; the desired final beat; and the desired
style, reference, or audio direction. Generic clarification questions must not
block the draft.

## Precision structure for 3–5 seconds

Use this structure when developing or materially repairing a short shot:

1. **Shot contract:** state the live duration, aspect ratio, one continuous
   shot, and no cuts unless cuts were requested. Treat Start, End, Interior,
   Ingredients, motion, structural, or audio references as authoritative only
   when the live context verifies the corresponding input is loaded.
2. **Stable continuity:** preserve the supplied subject facts, props, scale,
   wardrobe, relationships, screen direction, and scene geography. Do not
   manufacture exact identity attributes.
3. **Micro-beats:** use timestamps when they clarify timing, or an equally
   explicit opening → compact action → readable final-hold progression. Keep to
   one primary action and at most one supporting reaction.
4. **Camera:** define framing, viewpoint, and one compatible movement or a
   lock. The camera must not contradict loaded boundary frames, scene
   geography, or the action timing.
5. **Motion and performance:** require physically plausible weight, contact,
   gaze, gesture, and secondary motion. Restrain expression and movement to
   what can read cleanly in the available seconds.
6. **World and look:** make coherent choices for setting, lighting source,
   atmosphere, materials, and visual style without overriding supplied facts or
   reference-owned appearance.
7. **Audio ownership:** distinguish ambience, sound effects, music, source
   audio, and supplied dialogue. Assign every supplied line to one speaker;
   never create dialogue merely to complete the blueprint.
8. **Targeted negatives:** protect this shot's identity, props, anatomy,
   geography, camera, text, and audio from its most likely failures. Avoid a
   long generic negative-prompt inventory.

Precision is not the same as density. Every detail must agree with the others,
and the combined action, camera, performance, and audio demands must remain
small enough for a 3–5 second shot.

## Short-shot blueprint

Cover only the fields that materially define the requested shot:

- Shot contract: duration, aspect, continuous-shot or explicit cut structure.
- Identity: stable visible traits, scale, wardrobe, relationships, and active
  `@tags` not already owned by reference inputs.
- Chronology: opening state, visible change, then final pose or composition.
- Camera: framing, viewpoint, and either one compatible movement or a lock.
- World: location, lighting source, atmosphere, materials, and style.
- Audio: ambience and explicitly assigned dialogue.
- Continuity: critical invariants and a short avoid clause.

Concrete visible language is more useful than abstract praise such as
"beautiful," "viral," or "masterpiece." Avoid stacking several character
actions, camera moves, transformations, and dialogue beats into a three-second
shot.

## Motion and duration

Favor one primary action plus one supporting reaction. When appropriate, move
eyes before the head, shift posture before the whole body, keep hand or arm
gestures simple, and use hair, cloth, breathing, wings, or environmental motion
as subtle secondary detail. A changed performance does not require a changed
camera.

For a three-second clip, establish the state immediately, perform one compact
beat, and leave a brief final hold. For a five-second clip, allow a short settle,
build one action or reveal, and land on a readable payoff. The live Studio
duration remains authoritative; these are staging heuristics rather than fixed
model limits.

## Shot structure

One continuous shot is the safest default for a short local test, not a model
limitation. LTX-2.5 can attempt two to four explicit shots when the duration is
long enough. Name each cut or transition, recurring identity, framing, and audio
continuity. Do not rewrite an intentional multi-shot request into one shot.

## References and tags

Preserve valid tags exactly, including spelling and repeated speaker binding.
Use lowercase letters, digits, and underscores for Ingredients tags. Flag an
invalid tag instead of silently changing it, because renaming can detach the
prompt from its reference.

Start/End/Interior anchors own timed composition. Ingredients owns persistent
multi-reference identity. Motion paths own trajectories. Union guides own timed
structure. Source audio owns timing. Do not duplicate or contradict information
already owned by those controls.

## Dialogue

Assign each spoken line to exactly one explicit speaker and state how many
voices may exist in the shot. Name silent and offscreen characters when that
prevents accidental speech; require silent characters not to mouth, echo, or
inherit another character's line. Keep dialogue short enough to fit the chosen
duration naturally.

For native generated speech, a reusable voice specification may include age or
presentation, pitch/register, vocal weight and texture, tonal warmth or
brightness, pacing, articulation, emotional baseline, performance style, and
explicit exclusions. Reuse the stable wording for a recurring character and
change only scene-specific delivery. This prose improves direction but is not
a persistent voice embedding. Native generated dialogue can still blend or
swap voices, so recommend Character Voices when dependable recurring voice
identity or speaker separation matters more than generation in a single pass.

Distinguish dialogue, nonverbal vocalization, ambience, sound effects, and
music. State when audio will be added later so the video prompt can focus on a
clean visual performance instead of inventing synchronized sound.
