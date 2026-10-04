# Dang Studio localhost quick start

This guide runs the complete browser Studio directly from the public source
checkout. The native macOS wrapper and DMG assembly process are not required.

## One-time base installation

Use an Apple Silicon Mac running macOS 13 or newer with at least 16 GB unified
memory. Install Xcode Command Line Tools, Git, Python 3.11, `uv` and FFmpeg.
For example, after installing Homebrew:

```bash
xcode-select --install
brew install git python@3.11 uv ffmpeg
```

From the repository root, inspect and install the pinned LTX runtime:

```bash
python3.11 outputs/local-video/install_ltx25_runtime.py --plan
python3.11 outputs/local-video/install_ltx25_runtime.py --install
```

The base Q4 video and Gemma text packs download about 27.47 GB. Start with at
least 40 GB free, inspect the network/disk plan, and review the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
before recording acceptance:

```bash
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --preflight
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --plan
python3.11 outputs/local-video/install_phosphene_ltx25_release.py \
  --install --accept-license
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --verify
python3.11 outputs/local-video/verify_ltx25.py
```

The weight installer verifies every shard and can resume an interrupted
transfer. Keep the Mac awake, connected to power and online during setup.

## Open the studio

Double-click `launch_ui.command` in `outputs/local-video`, or run:

```bash
./outputs/local-video/launch_ui.command
```

The studio opens at `http://127.0.0.1:7860/`. It is reachable only from this
Mac. Keep the Terminal window open while using the studio; press `Control-C` in
that window to stop it.

Always start GPU work through `launch_ui.command` in Finder or a normal macOS
Terminal. Some restricted or background automation shells cannot open Apple's
Metal user clients; MLX may otherwise abort and macOS may show “Python quit
unexpectedly.” The UI detects that condition before loading MLX and explains
how to relaunch. CPU-only diagnostics from a restricted shell must use
`python3 outputs/local-video/verify_ltx25.py --skip-gpu`.

The workflow cards are readiness-gated. Generate needs the Q4 base and Gemma;
Extend, Retake, and Audio-to-Video need the selective Q8 overlay; Ingredients,
Union, and Motion also need their matching adapter. A card stays disabled only
while its exact local assets are absent or invalid.

## One-time advanced setup

Ordinary Generate is unchanged and can be used without this section. For
Extend, Audio-to-Video, Ingredients, or Motion, first free at least **60 GiB**
and install the selective Q8 overlay:

```bash
python3 outputs/local-video/install_ltx25_advanced_q8.py --preflight
python3 outputs/local-video/install_ltx25_advanced_q8.py --plan
python3 outputs/local-video/install_ltx25_advanced_q8.py --install --accept-license
python3 outputs/local-video/install_ltx25_advanced_q8.py --verify
```

The physical Q8 additions are about 41.19 GB. The 60 GiB target also covers a
temporary shard, the 6 GiB safety reserve, the control adapters, and macOS
headroom. The install is resumable.

For Ingredients and Motion, accept the Ingredients repository terms in a
browser, authenticate once, and install the two adapters:

```bash
hf auth login
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle ingredients
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle motion
```

Install `--bundle union` as well only if Canny/depth/pose control is wanted.
These controls run one at a time on Q8; they are not stacked into a single
render.

## Recommended first workflow

1. Stay on **Generate**.
2. Write one continuous shot with one primary action. State the subject count,
   wardrobe, left/right placement, camera movement, and anything that must not
   change.
3. Choose **Aspect ratio**. Use 16:9 for landscape, 9:16 for vertical, 1:1 for
   square, or 4:3 / 3:4; **Profile default** keeps the profile's original canvas.
   Named ratios resolve to the nearest model-safe pixel grid.
4. Leave **Duration** at 3 seconds for exploration. The slider supports 1–10
   seconds and shows a 16 GB memory/identity-drift warning above 5 seconds.
5. Start with **Draft** or **Balanced**. Re-run the chosen seed at **Quality**
   after the composition works.
6. Fill **Character Bible** when a named character must remain consistent.
7. Add reference-led anchors when identity or composition matters. Up to four
   active images may be mixed across Start, End, and Interior positions.
8. Click **Generate video**. The right-hand progress bar reflects real model
   stages; the render log remains available for precise diagnostics.

When asking another agent to write the prompt, name the video model rather than
its internal text encoder. A reliable request is:

```text
Write one production-ready prompt for LTX-2.5 video generation. Duration: 3
seconds. Aspect ratio: 16:9. Describe a single continuous shot in chronological
order, including subject identity and appearance, setting, action, camera
framing and movement, lighting, visual style, and synchronized audio or
dialogue. Keep motion physically plausible, avoid scene cuts, and return only
the final prompt.
```

Gemma 4 is the local text encoder/prompt interpreter used inside this setup; it
is not the video model name the prompt writer should target.

Example visual prompt:

```text
One continuous locked-camera medium shot of Maya standing alone beside a blue
vintage car at dusk. Maya keeps the same face, shoulder-length black hair,
olive jacket, and silver pendant throughout. She looks toward the car, takes
one step forward, and stops. Exactly one person and one car; stable lighting;
no cuts, duplicates, wardrobe changes, text, or extra limbs.
```

After a render, pause on a frame you want to keep and click **Capture current
frame**, then **Use as next clip's start frame**. The UI carries the nearest
matching aspect ratio, clears stale Start instructions, and sets the image
strength to 1. This starts a separate continuation clip. It does not trim the
old clip, stitch the clips, carry its audio, or preserve hidden motion state.

## Multi-reference ingredients and frame composition

There are two complementary reference workflows:

- Use **Frame Composer** when several references should be combined into one
  reliable still, then assign it as Start, End, or Interior on ordinary Q4
  Generate.
- Use **Ingredients** when those references should condition the video-level
  Q8 render directly. One tag may have multiple views. Ingredients is locked
  to 121 frames / 5.04 seconds because that is the native adapter bucket.

For Frame Composer:

1. Add an ingredient tag such as `@maya`, then attach one or more views of that
   same character. Add other ingredients such as `@jacket` or `@car`.
2. Mention every active tag in the Frame Composer prompt. Multiple photos under
   one tag are treated as complementary views of one ingredient, not separate
   subjects.
3. There may be at most four active reference photos in one composition.
4. Compose the still, then choose **Use as Start**, **Use as End**, or **Add as
   Interior**.
5. Generate the video normally. Frame Composer and video rendering are
   serialized so both large models are never loaded together.

Example composition prompt:

```text
Keep @maya's face, hair, and age from both views; dress @maya in the exact
@jacket; place her beside @car in one coherent cinematic medium shot. Exactly
one person, no duplicates, no text.
```

For native Ingredients, open its workflow card, add the character, wardrobe,
prop, and location references, reuse their stable `@tags` in the prompt, and
submit. Start, End, and Interior anchors may be mixed with Ingredients when a
scene needs both reference identity and timeline composition. The Studio uses
the native LTX-2.5 Ingredients adapter on the Q8 distilled transformer.
**Profile default** is the recommended Ingredients aspect. Other aspect presets
are intentionally marked experimental: selecting one keeps the 121-frame
duration lock and similar pixel budget, but may reduce identity and composition
consistency. The studio explains that risk before an experimental render.

## Extend a successful take

1. Generate and accept a short take, or select another compatible local MP4.
2. Open **Extend**, choose **Before** or **After**, and leave the first attempt
   at three seconds.
3. Describe only the continuation while explicitly preserving identity,
   wardrobe, lighting, and camera motion.
4. Submit. The original source is preserved; the extension is written as a new
   MP4.

The Studio supports up to about ten seconds total on this 16 GB Mac. Longer
source clips and higher resolutions cost more memory and time, so chaining
several short accepted extensions is safer than one large attempt.

## Audio-to-Video

Open **Audio-to-Video**, supply a local audio track and a concise motion/shot
prompt, then optionally add a start image for identity and composition. The
audio after its selected start offset must be long enough for the requested
video duration. The job uses Q8 dev stage one and the pre-fused Q8 distilled
stage two in low-RAM mode.

## Motion Track

Open **Motion Track**, add the required start image, and draw or supply the
normalized motion path. The backend interpolates each path through the clip
and prepares the guide at the adapter's required scale. Motion uses the
official LTX-2.3 adapter in the LTX-2.5 workflow; it does not need a separate
LTX-2.3 base model. The preferred local quality profile is 512x320. A
640x384 three-second profile is exposed only as an experiment because its
attention, swapping, runtime, and Python/Metal termination risks are materially
higher on 16 GB; selecting it opens Prompt Coach guidance before any render.

## Character dialogue

Visual prompts and dialogue are deliberately separate.

1. Open **Character Voices**.
2. Register a clean 3–30 second PCM16 WAV that is your voice or a voice you
   have explicit permission to clone. The reference establishes identity;
   prose only adjusts coarse delivery.
3. Assign the preset to an explicit character tag such as `@maya`.
4. Add one sentence-sized cue per row. Each cue has one character, language,
   start time, text, delivery note, and seed. Speakers are synthesized in
   separate calls and cannot be intermixed accidentally.
5. Generate WAV cues independently, or opt in to publish a new dialogue-muxed
   MP4. The original video is always preserved.

The local voice lane supports Arabic, Danish, German, Greek, English, Spanish,
Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch,
Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, and Chinese.

## Where results go

- Videos: `outputs/local-video/renders/`
- Frame Composer stills: `outputs/local-video/generated-frames/`
- Character voice WAVs: `outputs/local-video/generated/voices/`

Outputs are published atomically and existing results are not overwritten
silently.

## Practical quality guidance

- Iterate at 3 seconds. Extend a successful take in additional three-second
  segments instead of forcing a difficult 10-second first attempt.
- Reuse a successful seed while changing only one variable at a time.
- Clean reference photos and explicit object counts reduce drift more reliably
  than a long decorative prompt.
- Use Start + one Interior + End as the practical high-control arrangement on
  this 16 GB Mac.
- Local Q4 generation can still make visual mistakes; references and concise
  shot design reduce them but cannot guarantee the hallucination rate of a
  large hosted service such as Seedance.

Advanced Q8 modes are much slower than Q4 Generate on an M1 Pro with 16 GB.
Expect tens of minutes: the first three-second Extend is realistically about
25–60 minutes and may run longer depending on source resolution, steps,
thermal state, and other open apps. This does not slow ordinary Q4 jobs.
