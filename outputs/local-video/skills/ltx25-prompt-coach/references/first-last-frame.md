# First-frame / last-frame prompting reference

Use this reference only when the creator supplies or explicitly discusses both
a Start frame and an End frame. It is an advisory prompt-writing playbook, not
a new render mode and not a claim that conditioning guarantees exact pixels.

Prompt Coach receives only whether each boundary image is loaded; it does not
inspect the image pixels, poses, gaze, crop, or image-strength values. Do not
claim to have compared the endpoints. Ask the creator to describe a material
difference when compatibility cannot be judged from the prompt.

## Boundary contract

- Treat the Start frame as the authoritative target for frame zero and the End
  frame as the authoritative target for the final frame.
- Preserve only the identities, anatomy, scale, subject geography, props,
  camera viewpoint, environment, lighting, and visual style that remain
  invariant across both references. Encode the visible deltas required by the
  End frame rather than freezing a deliberate transformation or lighting move.
- Describe the smallest physically plausible continuous set of changes that can
  connect the two endpoints. Do not invent a different opening or ending.
- Begin explicitly from the Start composition and finish explicitly on the End
  composition. Give the ending enough time to settle instead of introducing a
  new action in the final instant.
- If prompt prose contradicts either image, identify the conflict and make the
  images authoritative after confirming the creator's intent. Do not silently
  pretend incompatible endpoints can both be followed.

A typed Start or End description conditions the main video prompt; it does not
create or load a still. If either image is absent, say that no exact visual
anchor exists and recommend uploading one or creating it in Frame Composer.
Prepare both stills on the selected aspect-ratio canvas: Studio center-crops
anchors to the render aspect, so inconsistent source crops can change framing.

## Plan the bridge

For the safest default, write one chronological shot: opening state, transition
or performance beat, then final state. Use one compatible camera instruction.
A locked camera is usually safest when both references share a viewpoint;
otherwise describe only the restrained move needed to reconcile the endpoints.
If the creator intentionally requested a multi-shot sequence, keep its named
cuts and treat the two frames as the boundaries of the whole sequence instead
of forcing a continuous no-cut bridge.

### Camera-lock continuity default

When the two images represent the same shot and no deliberate camera move was
requested, explicitly keep these attributes matched from Start to End:

- camera position and height;
- lens or field of view;
- horizon, eye level, and perspective;
- subject-to-camera distance, framing, and apparent subject scale; and
- fixed environmental anchors in the same screen positions.

A changed expression or performance does not imply a changed camera. Prefer
gaze, facial expression, gesture, posture, weight shift, or restrained
environmental motion before inventing a push-in or reframe. If a camera move is
intentional, state it chronologically and quantitatively when the creator has
provided the amount—for example, a slow 10% push-in over three seconds—and ask
the creator to confirm that the End image visibly supports its final framing.
Do not invent a percentage or claim to verify that support from pixels the
Coach cannot see.

Before finalizing, check that subject size, lens/field of view, eye level,
horizon, perspective, and fixed anchors remain compatible. If no push, zoom,
dolly, pan, tilt, or orbit was requested, the rewrite must not imply one.

### Motion design

Keep interpolation deliberate and low-complexity: one primary action plus one
supporting reaction. Where appropriate, move the eyes before the head, shift
posture before the whole body, give gestures clear start and end states, and
keep hair, cloth, wings, breathing, or background motion subtle. Let a final
reaction or punchline hold long enough to read. Do not stack unrelated movement
into a short boundary-frame shot.

Live Studio duration and aspect settings are authoritative. When neither is
chosen, 3 seconds is a useful first test for a quick reaction or punchline; use
5 seconds only when a reveal, buildup, or transformation genuinely needs more
interpolation time. For three seconds, establish immediately, perform one
compact action, and reserve a brief final hold. For five seconds, allow a short
settle, build one clear action or reveal, and land on a readable final state.
These are recommendations, not fixed model limits.

If the frames differ only slightly, use a restrained expression, prop, light,
or camera beat instead of inventing unrelated action. If they differ so much
that a credible bridge would require a cut, teleport, identity swap, or sudden
camera change, warn the creator and recommend splitting the shot or adding one
critical Interior anchor.

## Identity, dialogue, and sound

Boundary images control timed compositions; they are not a persistent identity
embedding. Recommend Ingredients when multiple referenced identities or objects
must stay distinguishable throughout the clip. Apply the shared dialogue and
speaker-tag rules in `prompt-design.md`; do not import character names or fixed
voice descriptions from examples unless the creator included them in the
current prompt or saved project context.

## Compact prompt pattern

Use this shape for a single-shot rewrite. For an intentional multi-shot request,
replace the continuous-shot clause with the creator's explicit shot structure
and treat Start/End as the sequence boundaries:

> One continuous [duration] [aspect] shot, using the supplied Start frame as
> the exact opening composition and the supplied End frame as the exact final
> composition. Preserve [only the established invariants]. Begin exactly from
> the Start frame. [Small chronological bridge.] [Match camera position,
> lens/FOV, perspective, horizon, framing, subject distance and scale—or state
> the requested deliberate move chronologically.] Finish on and briefly hold
> the End-frame state. [Explicit
> dialogue ownership and sound, if any.] Avoid cuts, identity swaps, duplicate
> subjects, extra anatomy, prop fusion, unintended speech, and camera motion
> that contradicts the references.

Keep the final rewrite proportional to the shot. More prose does not make an
impossible transition reliable.
