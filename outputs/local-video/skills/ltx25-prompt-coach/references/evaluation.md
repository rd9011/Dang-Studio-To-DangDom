# Calibration and training gates

Treat reviewed render outcomes as evaluation data, not automatic online
training. A single failure can be seed noise, model variance, bad input media,
resource pressure, or a genuine prompting problem.

For each reviewed case, retain locally:

- original prompt and exact `@tags`;
- selected and recommended workflows;
- live readiness plus effective duration, aspect, profile, and seed;
- loaded reference counts and control inputs;
- accepted or rejected rewrite;
- outcome ratings for identity, composition, motion, anatomy, text, dialogue,
  audio, and artifacts;
- a short human explanation of the most important failure or success.

Keep a frozen, balanced evaluation set covering Generate, Ingredients, Motion
Track, Union Control, Extend, Retake, Audio-to-Video, blocked-feature fallbacks,
multi-shot requests, and conflicting controls.

Every skill revision must preserve:

- zero recommendations that claim a blocked workflow is runnable;
- 100% retention of valid `@tags` and speaker bindings;
- 100% preservation of the user's full prompt input;
- exact compliance with fixed duration/aspect/profile constraints;
- no regression in semantic workflow selection on the frozen set.

Revise rules only when repeated reviewed cases support the change. Consider
weight training only after a sufficiently large labeled set exists and this
skill-plus-validation approach measurably plateaus.
