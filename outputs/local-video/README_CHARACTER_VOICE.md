# Local character voices

This subsystem adds the non-UI foundation for local, multilingual character
speech. It uses the official **Chatterbox Multilingual V3** 500M model because
V3 is the current 23-language release, explicitly supports Apple MPS, improves
speaker similarity, and is designed to reduce unwanted continuation and
repetition relative to V2.

It is integrated into `ltx_ui.py` as an optional, separately installed lane. It
does not add packages to the LTX environment. Voice cues are rendered after the
video as isolated jobs; an explicit UI checkbox can publish a new dialogue-muxed
MP4 while preserving the original render.

## Honest identity model

A persistent character voice always comes from one named preset containing one
explicit canonical reference clip. Characters are assigned to presets. The same
reference hash is reused for each cue, while every cue records its own explicit
seed and synthesis settings.

“Stable” here means the same reference-conditioned speaker identity. It does not
promise bit-identical waveforms across different PyTorch/macOS/hardware versions;
the immutable runtime pins and fixed seed make runs as reproducible as the backend
allows.

A prose description such as “warm, deep, calm older narrator” **cannot create an
exact persistent speaker identity in Chatterbox**. In this wrapper it is saved as
audition metadata and can select only coarse delivery settings:

- calm/gentle/soft/measured language selects the calm delivery preset;
- dramatic/expressive/energetic/intense language selects dramatic delivery;
- everything else selects neutral delivery;
- age, gender, pitch, timbre, and accent words do not manufacture an identity.

The canonical reference—not the prose—is the identity anchor. Use only your own
voice or a recording you are authorized to clone. Registration requires an
explicit consent confirmation.

## Exact immutable pins

- Source: `https://github.com/resemble-ai/chatterbox.git`
- Source commit: `65b18437192794391a0308a8f705b1e33e633948`
- Source package version: `0.1.7`
- Model: `ResembleAI/chatterbox`
- Model revision: `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`
- Python: `3.11.x`, isolated by `uv`
- Important runtime pins: PyTorch/Torchaudio `2.6.0`, Transformers `5.2.0`,
  Diffusers `0.29.0`, `resemble-perth` `1.0.1`, `s3tokenizer` `0.3.0`
- License: MIT for the official source/model repository

The checked-in `character_voice_runtime/uv.lock` pins every transitive package
artifact and hash. The installer checks out the exact source commit and refuses
a dirty checkout.

Only five model files are installed:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `t3_mtl23ls_v3.safetensors` | 2,143,989,928 | `5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953` |
| `s3gen.pt` | 1,057,165,844 | `9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3` |
| `ve.pt` | 5,698,626 | `4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1` |
| `grapheme_mtl_merged_expanded_v1.json` | 69,989 | `69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf` |
| `Cangjie5_TC.json` | 1,920,163 | `7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c` |

The exact model payload is 3,208,844,550 bytes (2.99 GiB). Expect roughly
4–5 GiB persistent use for model plus environment and another 1–2 GiB for the
reusable package cache. Keep **at least 8 GiB free during installation**.

The two official `.pt` state dictionaries are pinned by SHA-256 and upstream
loads them with PyTorch's `weights_only=True`. The much larger legacy/V2,
single-language, and built-in anonymous voice files are not downloaded.

## Install and verify

No model weights were downloaded while this foundation was built or tested.
Run the installer when ready:

```bash
python3 outputs/local-video/install_character_voice.py
```

The model is public; no Hugging Face token or “Agree and Access” step is needed.
The installer intentionally has no token option and removes token variables from
its network subprocesses.

Model transport uses macOS system `curl` and the immutable revision URLs above.
User curl configuration is disabled; the initial request and every redirect are
restricted to HTTPS with TLS 1.2 or newer, no authorization header is sent, and
curl retries transient failures while displaying its real transfer progress.
Every completed file must match both the declared byte count and SHA-256 before
it is atomically published.

This machine's Hugging Face Xet bridge has returned whole-file HTTP 200 responses
to Range requests, so the installer deliberately does **not** resume model files.
If an older `.partial` exists, it is preserved while a separate complete copy is
downloaded. It is removed only after the fresh copy verifies and is published
(or published directly if the old partial is already complete and hash-exact).
Consequently, an incomplete partial does not reduce the free-space requirement.

Read-only checks:

```bash
python3 outputs/local-video/install_character_voice.py --check-only
python3 outputs/local-video/install_character_voice.py --check-only --full-hash
python3 outputs/local-video/character_voice.py status
```

`--full-hash` reads all 3.21 GB, so normal inference uses verified sizes plus the
signed installation receipt and the hashes established during installation.

## Register stable voices

Prepare a clean, single-speaker, uncompressed signed-16-bit PCM WAV. The wrapper
accepts mono or stereo, 16–96 kHz, 3–30 seconds; a clean 6–10 second clip in the
same language as the target line is ideal. Chatterbox uses at most ten seconds.

```bash
python3 outputs/local-video/character_voice.py register-preset aria-voice \
  --reference /absolute/path/aria-reference.wav \
  --reference-language en \
  --voice-description "warm, calm, measured delivery" \
  --consent-confirmed

python3 outputs/local-video/character_voice.py assign aria aria-voice
```

The clip is copied to a private managed directory under
`work/local-video/character-voices/references/`, named by its hash, and checked
again before every render. Replacing a preset or character assignment requires
the deliberate `--force` option.

List the registry:

```bash
python3 outputs/local-video/character_voice.py list
```

## Render a cue

```bash
python3 outputs/local-video/character_voice.py synthesize \
  "We should leave before the storm arrives." \
  --language en \
  --character aria \
  --voice-description "calm but increasingly urgent" \
  --output scene-01/aria-001.wav
```

Outputs are confined to `outputs/local-video/generated/voices/`. The model writes
into a private staging directory; the wrapper validates the WAV and publishes it
with an atomic rename. A sibling `.voice.json` records the exact preset hash,
language, settings, source/model revisions, output hash, duration, MPS/CPU used,
and the hash—not the content—of the spoken text.

All 23 language IDs are available locally:

`ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`

For cross-language cloning, a reference recorded in the requested language is
best. Upstream warns that mismatched-language references can transfer their
accent; reducing `--cfg-weight` toward `0` can mitigate this.

Long narration should be split into sentence-sized cues below 1,600 characters.
This limits unwanted continuation and lets every cue retain an explicit character
assignment.

## Local Studio UI

Open **Character Voices** in the localhost Studio:

1. Register a named preset from a 3–30 second PCM16 WAV and check the explicit
   authorization confirmation. Replacing an existing preset is a separate,
   deliberate checkbox.
2. Assign that preset to an `@character` tag. Reassigning an existing character
   is also deliberate.
3. Add one sentence-sized cue per row. Every row requires its own assigned
   character, language, start time, text, and seed. The delivery note is optional
   and never replaces the reference identity.
4. Select **replace the latest completed video’s native audio** only when you
   want a post-render dialogue pass. Otherwise the UI publishes independent WAVs
   that can be auditioned and downloaded.

The browser never parses dialogue out of the visual prompt or an ingredient tag.
The backend validates the assignment and launches one shell-free
`character_voice.py synthesize` argv per cue, sequentially. Two characters are
therefore never passed to the model in one request. Video, Frame Composer, and
voice-model compute are mutually admitted so the 16 GB machine does not load
these models concurrently.

The UI remains useful before the optional model is installed: preset registration
and character assignment work, while synthesis reports **Setup needed** and is
disabled. Readiness is verified locally and cached briefly so status polling does
not repeatedly import the model stack.

## Offline enforcement

Inference never calls `from_pretrained`. It calls `from_local` with the exact
model directory. Before importing PyTorch/Chatterbox, the isolated runner:

- removes token variables;
- enables Hub/Transformers/Diffusers offline modes and disables telemetry;
- installs a Python audit hook that rejects DNS and socket connections;
- redirects upstream's otherwise-online Chinese Cangjie lookup to the exact
  pinned local `Cangjie5_TC.json`.

A network attempt fails the job instead of silently downloading or falling back.

## UI and post-render integration contract

The UI should treat video prompt text and spoken dialogue as separate data. Never
send dialogue for two characters in one TTS call and never infer a voice from an
ingredient tag alone.

For each cue, the UI must provide:

```json
{
  "character": "aria",
  "text": "We should leave before the storm arrives.",
  "language": "en",
  "start_seconds": 1.25,
  "voice_description": "calm but increasingly urgent",
  "seed": 1234
}
```

`character`, `text`, and `language` are required. `start_seconds` belongs to the
editor/mux layer and is intentionally not interpreted by the voice model. The UI
can call `character_voice.synthesize(...)` directly or invoke the CLI with an
argument array (never a shell string). Parse the JSON result only when the process
returns zero. Stable fields for mux are:

```json
{
  "schema_version": 1,
  "character": "aria",
  "preset": "aria-voice",
  "audio_path": "/absolute/confined/path/aria-001.wav",
  "manifest_path": "/absolute/confined/path/aria-001.voice.json",
  "audio_sha256": "...",
  "sample_rate": 24000,
  "channels": 1,
  "duration_seconds": 2.4,
  "watermarked": true,
  "offline": true
}
```

Recommended post-render sequence:

1. Render each tagged character cue independently.
2. Verify its `.voice.json` and audio SHA-256.
3. Delay each mono cue by `start_seconds` in the editor/ffmpeg filter graph.
4. Mix character cues into one dialogue bus. Distinct identity remains explicit
   because each cue came from its assigned preset.
5. Either replace the LTX audio track with that bus, or mix the bus with native
   ambience at a deliberately lower level.
6. Mux into a temporary MP4 beside the final output, validate streams/duration,
   then atomically rename it over the requested final path.

The Studio implements the conservative replacement variant. It delays each
independently authenticated mono cue, mixes the cues into one dialogue bus,
copies the video stream, encodes only the new audio as 192 kb/s AAC, validates
the temporary MP4 with `ffprobe`, and atomically publishes a uniquely named
`*-voiced-*.mp4`. It never overwrites the original video. Cues that start at or
after the video end are rejected before voice inference. Mixing native ambience
is intentionally not automatic because the source may contain swapped or
hallucinated speech; add ambience later in an editor if desired.

For a simple one-cue replacement, the mux layer is equivalent to:

```text
ffmpeg -i video.mp4 -i aria-001.wav -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -shortest temporary.mp4
```

Use `subprocess` argument arrays rather than shell interpolation. Multi-cue mixes
should use one ffmpeg input per cue plus `adelay` and `amix`; do not concatenate
different characters into a single TTS request.

## Current validation boundary

The installer, registry, path confinement, WAV checks, offline job contract,
style resolution, exact manifests, UI consent/assignment/cue validation,
one-process-per-speaker scheduling, cancellation, ffmpeg argv construction, and
atomic mux publication have GPU-free tests. No weights were downloaded and no
real render was started while building this integration, so real MPS synthesis
has not yet been exercised on this laptop. The first post-install canary should
be a short English line with a 6–10 second authorized reference, followed by
Hindi and Chinese canaries. Chatterbox embeds Resemble AI's PerTh watermark in
generated audio; the wrapper records this in every result.
