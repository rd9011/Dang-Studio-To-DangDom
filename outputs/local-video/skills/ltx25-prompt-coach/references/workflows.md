# Workflow selection

Choose the workflow by the kind of control the request needs, not by keywords
alone. The live readiness report remains authoritative.

## Generate

Use for a new text-driven shot. Start, End, and Interior images anchor particular
times. They guide composition at those times; they are not a global identity
embedding. A captured frame is a visual hand-off into another Generate clip and
does not retain hidden velocity, latent state, or audio continuity.

For image-to-video, let the image own visible appearance and initial composition.
Prompt the visible change, subject motion, camera, environment response, and
audio. With Start plus End, describe the plausible bridge between them. When
they represent one viewpoint and no deliberate move is requested, explicitly
keep camera position, lens/field of view, perspective, horizon, framing,
subject distance, and scale matched; the Coach cannot inspect pixels to verify
whether the references are compatible.

## Ingredients

Use when several referenced characters, objects, or a location must remain
distinguishable throughout the shot. Use one stable lowercase machine-safe tag
per item, such as `@subject_a`, and mention every active tag. Clean reference sheets
are more useful than busy photographs.

The local Ingredients lane uses the Q8 transformer and its adapter and runs a
fixed 5.04-second bucket. Its profile-default landscape aspect is preferred:
the adapter was trained at 768x448, while 384x224 is the lower-memory 16 GB
profile. Other ratios are available only as out-of-training experiments and
may weaken identity, anatomy, reference placement, or composition. The official
768x448 profile itself is model-valid but experimental for memory on this 16 GB
Mac. Ingredients cannot be combined with Motion Track in one render.

## Motion Track

Use when a subject or point must follow an explicit spatial trajectory. A Start
image establishes the scene and normalized path JSON owns the movement. Keep the
prompt focused on subject, setting, style, and physically plausible response;
do not add a competing prose trajectory. Start with a few clear path points.
The 512x320 profile is the preferred local quality setting. A 640x384 profile
is a three-second-only experiment with substantially higher attention, swap,
runtime, and Python/Metal termination risk. Motion Track cannot be combined
with Ingredients in one render.

## Union Control

Use when a timed Canny, depth, or pose video should control whole-frame structure.
A comic strip or static storyboard is not a timed guide. Use its panels as
Start/Interior/End anchors, or first convert it into an appropriate guide video.
Do not contradict the guide's camera or geometry in prose.

## Extend

Use to generate new material directly before or after an existing clip. It
inherits source dimensions. Describe only the new interval and the desired
boundary continuity. It requires the Q8/HQ editing lane.

## Retake

Use to replace a selected interval inside an existing clip. Describe the
replacement interval and how both boundaries should join. Source-audio
preservation is optional. It requires the Q8/HQ editing lane.

## Audio-to-Video and voices

Use Audio-to-Video when supplied audio must own timing, rhythm, or performance.
The audio is unchanged; prompt the visible performer, setting, style, and camera
without competing timing instructions. Ordinary Generate should not be assumed
to follow exact beats or phonemes.

Prompt tags can clarify speakers but do not guarantee native voice separation.
Use Character Voices as the separate speaker-controlled pass when dependable
voice identity matters.
