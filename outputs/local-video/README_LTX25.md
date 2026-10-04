# Local LTX-2.5 studio

This launcher runs LTX-2.5 locally on Apple Silicon with MLX. The default
Generate path is the distilled two-stage pipeline: Q4 transformer, separate Q4
Gemma 4 text tower, native video/audio, low-RAM block streaming, and no network
access during a render. A separate selective Q8 overlay provides Extend,
Retake, Audio-to-Video, Ingredients, Union, and Motion without changing or
slowing the Q4 Generate path.

For the short operator workflow after installation, start with
[`QUICKSTART.md`](QUICKSTART.md).

## Pinned installation

- Runtime: `mrbizarro/ltx-2-mlx`
- Tag: `v0.14.19+ltx25.7`
- Commit: `bf419b6f76e7753993eb7c3df3b15e1451310409`
- Python package: `ltx-pipelines-mlx==0.14.19+ltx25.7`
- Weight release: `mrbizarro/Phosphene@weights-ltx25-v1`
- Base pack: `work/local-video/models/ltx-2.5-mlx-q4`
- Text pack: `work/local-video/models/gemma4-12b-ltx25-q4`
- Advanced overlay: `work/local-video/models/ltx-2.5-mlx-q8`

The packs are separate by design. Do not merge Gemma or Q8 weights into the Q4
base directory, and do not replace a Phosphene transformer/upscaler with older
`dgrauet` files. `verify_ltx25.py` checks both base install receipts, every
exact byte size, small-file hashes, model identities, Q4 metadata, the pinned
runtime commit/tag plus the exact allowlisted low-RAM edit backport, package
version, ffmpeg, and Metal. `--full-hash` also streams every large base weight
through SHA-256.

```bash
python3 outputs/local-video/verify_ltx25.py
python3 outputs/local-video/verify_ltx25.py --full-hash   # optional, reads ~25.6 GiB
```

Use `--skip-gpu` only for a diagnostic CPU-side verification; a real render
still requires Metal.

### Advanced Q8 overlay

Free at least **60 GiB** before starting the Q8 install. The overlay stores
about 41.19 GB of selected physical files and reuses byte-identical Q4 sidecars
through verified relative links. This avoids duplicating the connector, VAE,
vocoder, and upscalers, and deliberately omits the unneeded 8.9 GB distilled
LoRA.

```bash
python3 outputs/local-video/install_ltx25_advanced_q8.py --preflight
python3 outputs/local-video/install_ltx25_advanced_q8.py --plan
python3 outputs/local-video/install_ltx25_advanced_q8.py --install --accept-license
python3 outputs/local-video/install_ltx25_advanced_q8.py --verify
```

The installer is shard-resumable and publishes its receipt only after every
selected file and link verifies. Advanced controls also need their individual
adapters; see [`README_LTX25_OPTIONAL_ASSETS.md`](README_LTX25_OPTIONAL_ASSETS.md).

### Explicit compatibility fallback

The official Phosphene Gemma pack above remains the default and preferred
source. A separately pinned `ddalcu` tensor serialization can be imported only
into `gemma4-12b-ltx25-q4-ddalcu-fallback`, after full-file authentication and
structural compatibility checks. It is never accepted as the official pack and
must be selected explicitly with `--gemma-pack ddalcu-fallback`. See
`README_DDALCU_GEMMA_FALLBACK.md` for the exact provenance, commands, and the
important value-equivalence limitation.

The UI exposes the same choice under **Text encoder provenance**. Its default
**Auto** option prefers the complete official pack and selects the separately
validated fallback only when the official pack is unavailable; the resolved
pack remains visible in the job status and render log.

## Generate

```bash
python3 outputs/local-video/generate_ltx25.py \
  "A single continuous cinematic shot of ..." \
  --profile quality --seed 1234 \
  --output renders/my-video.mp4
```

The resolved upstream command always uses:

```text
generate --model <base-q4> --gemma <gemma4-q4> --distilled --low-ram
```

Distilled inference uses trained schedule presets, not CFG/STG knobs:

- `draft`: 512x256x33, fast `5+2` schedule.
- `balanced`: 640x384x41, graded `8+2` schedule.
- `quality` (default): 768x448x49, graded `8+2` schedule.
- `max`: 768x448x25, graded `8+2` schedule; short high-detail take.
- `clip3s`: 512x256x73, graded `8+2` schedule; conservative 3.04-second take.

The UI's quality choices select these resolution/length/schedule combinations.
They do not synthesize CFG or STG values. CLI `--schedule-preset fast|default`
can override a profile; compatibility `--steps 5|8` maps to the same presets.
`--cfg-scale` and `--stg-scale` are rejected so they cannot be silently ignored.

The duration slider and `--duration-seconds N` accept integer values from 1 to
10 and map to `24*N+1` frames. Past five seconds, attention memory and identity
drift rise sharply on a 16 GB Mac. The attempt remains enabled, but separate
three-second takes followed by native Extend or an edit/stitch are safer.

The UI also offers Profile default, 16:9, 9:16, 1:1, 4:3, and 3:4 aspect
choices. Named ratios preserve the selected profile's approximate pixel budget
and resolve to the nearest valid LTX grid rather than promising an unsafe
arbitrary pixel size. Extend inherits the source video's dimensions.

## Uploaded image anchors

The distilled path supports a start frame, end frame, and timed interior
anchors. These may be mixed, up to four active images:

```bash
python3 outputs/local-video/generate_ltx25.py \
  "One continuous transformation with stable composition" \
  --start-image /absolute/start.png --start-strength 1.0 \
  --anchor /absolute/middle.png 24 0.65 \
  --end-image /absolute/end.png --end-strength 0.8 \
  --profile quality --output renders/multi-anchor.mp4
```

`--image` is the backward-compatible alias for `--start-image`. Frames are
zero-based and unique; strengths range from 0 to 1. Frame 0 is a hard starting
replacement while nonzero images consume additional keyframe tokens.

The standalone model-generated keyframe pipeline is not exposed by this local
wrapper. Use uploaded start/interior/end anchors instead.

## Prompt Relay

Successive sections of one continuous clip can have local scene prompts while
the main prompt remains global:

```bash
python3 outputs/local-video/generate_ltx25.py \
  "One continuous locked-off shot; preserve subject and lighting" \
  --scene "The subject sits and looks toward camera" --scene-length 2 \
  --scene "The subject stands and turns toward the window" \
  --scene "The subject pauses at the window" --scene-length 2 \
  --profile quality --output renders/prompt-relay.mp4
```

Scene lengths use latent frames. Omitted lengths divide the remaining timeline
automatically. Invalid or zero-length layouts are rejected before model load.

## Workflow availability

- Generate: supported by the installed Phosphene Q4 base + Gemma packs.
- Ingredients / Union / Motion: supported on the separate Q8 distilled lane
  after the matching pinned adapter is installed. Ingredients uses the native
  LTX-2.5 adapter and is fixed at 121 frames / 5.04 seconds. Motion uses the
  official LTX-2.3 adapter in Lightricks' LTX-2.5 workflow; no 2.3 base model
  is needed. Each control runs separately with `--single-stage --low-ram`.
- Retake / Extend / Audio-to-Video: supported on the Q8 advanced lane after its
  exact overlay receipt verifies. Extend and Retake use the dev transformer;
  A2V also uses the pre-fused Q8 distilled transformer in stage two.
- The UI keeps an advanced button disabled only while its exact Q8/adaptor
  assets are absent. The API and wrapper enforce the same gate.
- Character Voices: optional isolated Chatterbox Multilingual V3 lane. The UI
  registers an explicitly authorized canonical WAV, assigns one stable preset to
  each `@character`, and renders every multilingual dialogue cue separately so
  speakers are never combined in one TTS request. An opt-in post-render pass can
  replace native audio with timed cues in a new MP4 while preserving the original.
  Prose controls coarse delivery only; it does not create speaker identity. See
  `README_CHARACTER_VOICE.md`.

See `README_LTX25_CONTROLS.md`, `README_LTX25_EDIT_MODES.md`, and
`README_LTX25_OPTIONAL_ASSETS.md` for those modes.

On an M1 Pro with 16 GB unified memory, expect Q8 advanced jobs to take tens of
minutes. A first three-second Extend is roughly a 25–60 minute estimate and may
take longer with a high-resolution source or memory pressure. That estimate
does not apply to the unchanged Q4 Generate path.

## Persistent visual characters and frame hand-offs

The studio's **Character Bible** can save a named visual profile containing its
five text fields and up to four PNG/JPEG/WebP reference views. **Save as new**
creates a private managed copy, while replacing or deleting an existing profile
requires a separate explicit action. Profiles survive browser, UI-server, and
Mac restarts under `work/local-video/characters/`; they are never sent to an
external service. Loading a profile restores its text and previews and submits
its managed views automatically in Generate, Ingredients, Union, and Motion
jobs. Those views share the four-image / 24 MB timeline-anchor budget. Character
Voice presets and assignments remain a separate registry.

After a render completes, seek in the player and choose **Capture current
frame**. The browser pauses on the displayed frame, captures it at the video's
native decoded dimensions, and the localhost API validates and atomically saves
the PNG under `outputs/local-video/generated-frames/`. The preview can be
downloaded or assigned as the next Generate job's Start frame. This is a new
image-to-video shot: the UI selects the nearest matching aspect ratio, clears
any stale Start description, resets image strength to 1, and then expects the
new prompt to describe only the continuation. Trim the rejected tail and stitch
the clips separately. The hand-off does not carry source audio or hidden motion
state and is not a replacement for native Extend.

## Prompting for fewer visual mistakes

Describe one camera shot and one primary action. State identity, wardrobe,
object count, left/right placement, and camera behavior explicitly. Use clean
reference images whenever exact identity or composition matters. Text and
anchors improve consistency but cannot guarantee the hallucination rate of a
large hosted service such as Seedance.

Existing outputs are never replaced unless `--force` is explicit.
