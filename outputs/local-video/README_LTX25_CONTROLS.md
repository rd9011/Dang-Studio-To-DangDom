# LTX-2.5 Ingredients and control modes

`condition_ltx25.py` provides three local IC-LoRA workflows on the selective
Q8 advanced lane:

- Ingredients combines several character, wardrobe, prop, and location images.
- Union Control follows a Canny, depth, or pose guide video.
- Motion Track animates a still along normalized motion paths.

The fast Generate path remains on the verified Q4 model and is not slowed or
replaced. IC-LoRA is deliberately routed to the separate Q8 distilled
transformer because the Q4 transformer retained too little of the adapter
delta for trustworthy control. The pinned runtime is `mrbizarro/ltx-2-mlx`
tag `v0.14.19+ltx25.7`, commit
`bf419b6f76e7753993eb7c3df3b15e1451310409`.

The model paths are:

```text
work/local-video/models/ltx-2.5-mlx-q4       # fast Generate
work/local-video/models/ltx-2.5-mlx-q8       # advanced Q8 overlay
work/local-video/models/gemma4-12b-ltx25-q4 # shared text encoder
```

The Q8 overlay stores only the two Q8 transformers and Q8 metadata. It safely
links byte-identical sidecars from the verified Q4 pack, so it does not
duplicate the connector, VAE, vocoder, or upscalers. Install it only after at
least **60 GiB is free**:

```bash
python3 outputs/local-video/install_ltx25_advanced_q8.py --preflight
python3 outputs/local-video/install_ltx25_advanced_q8.py --plan
python3 outputs/local-video/install_ltx25_advanced_q8.py --install --accept-license
python3 outputs/local-video/install_ltx25_advanced_q8.py --verify
```

The selected physical download is about 41.19 GB (38.36 GiB). The 60 GiB
starting target leaves room for a temporary release shard, the installer's
6 GiB reserve, the required adapters, and normal macOS operation. Downloads
are resumable and the final receipt is written only after every selected file
and shared link verifies.

Each control render emits `ic-lora --model <q8-overlay> --gemma <gemma>
--single-stage --low-ram`. Exactly one control adapter is loaded for a render;
the modes are not arbitrarily stacked. Do not copy Q8 files into the Q4 base
directory or install an older transformer over either verified pack.

## Optional adapters

Adapters are installed under:

```text
work/local-video/models/LTX-2.5-control-adapters/
```

Plan first, then install the required adapters. Ingredients is the native
LTX-2.5 adapter. Motion intentionally uses Lightricks' LTX-2.3 Motion adapter,
as their official LTX-2.5 workflow does; it does **not** require a 2.3 base
model or another text encoder.

```bash
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle ingredients --plan
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle motion --plan
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle ingredients
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle motion
```

Install `--bundle union` separately if Canny/depth/pose control is wanted, or
use `--bundle controls` for all three. The Ingredients repository has a
separate access gate. Accept its terms in a browser and run `hf auth login`
before downloading it. See `README_LTX25_OPTIONAL_ASSETS.md` for exact
revisions, sizes, authentication, and atomic verification behavior.

## Ingredients: multi-reference images and timeline anchors

Each repeatable `--reference` is one panel. Repeat a stable label/tag for
multiple views of one character, or use different labels for cast, wardrobe,
props, and locations:

```bash
python3 outputs/local-video/condition_ltx25.py ingredients \
  "@pilot walks around @ship in one continuous medium shot" \
  --reference /absolute/pilot-face.png "@pilot — pilot" "front-facing close-up" \
  --reference /absolute/pilot-body.png "@pilot — pilot" "full body, orange suit" \
  --reference /absolute/ship.png "@ship — ship" "compact cobalt spacecraft" \
  --start-image /absolute/opening.png --start-strength 1.0 \
  --end-image /absolute/closing.png --end-strength 0.8 \
  --profile ingredients5s --output ingredients/pilot-and-ship.mp4
```

The studio preserves stable `@tags` as part of each display label and can add
speaker-tagged dialogue guidance, preventing character lines from being
silently reassigned in the prompt. This is prompt association only; the video
model is not a character-specific TTS engine.

The wrapper builds a local reference sheet and training-compatible reference
video. Start, end, and repeatable `--anchor IMAGE FRAME STRENGTH` images may be
mixed with Ingredients. At most four active timeline images are accepted, with
unique pixel-frame indices.

Ingredients is intentionally fixed at 121 frames / 5.04 seconds. That is the
adapter's required reference/output bucket, so the UI locks its duration
instead of offering a misleading three-second preview:

- `ingredients5s`: 384x224x121, conservative for 16 GB.
- `trained5s`: 768x448x121, official bucket and much higher memory risk.

**Profile default** is the recommended Ingredients aspect. The other aspect
presets are available for deliberate experiments and preserve the selected
profile's approximate pixel budget on the 32-pixel single-stage LTX grid. For
example, `ingredients5s --aspect-ratio 9:16` becomes 224x384 while remaining
locked to 121 frames. The backend prints an experimental warning because a
non-default aspect can reduce character identity, reference placement, and
composition consistency. Return to Profile default when reliability matters.

## Union Control

Built-in source preprocessing is Canny only:

```bash
python3 outputs/local-video/condition_ltx25.py union \
  "Same movement and framing, cinematic live-action lighting" \
  --control-type canny --source-video /absolute/source.mp4 \
  --profile control3s --output controls/canny.mp4
```

Depth and pose require an already annotated guide:

```bash
python3 outputs/local-video/condition_ltx25.py union \
  "A dancer follows the exact body movement" \
  --control-type pose --guide-video /absolute/dwpose-guide.mp4 \
  --profile control3s --output controls/pose.mp4
```

## Motion Track

```bash
python3 outputs/local-video/condition_ltx25.py motion \
  "The red ball follows the path while the camera remains locked" \
  --start-image /absolute/scene.png \
  --tracks-json /absolute/tracks.json \
  --profile control3s --output controls/motion.mp4
```

Tracks use normalized `x`/`y` values from 0 to 1. The backend interpolates
them to one point per frame and builds the guide locally. The guide is
automatically prepared at the adapter's required half-resolution scale.

For Union and Motion, `control3s` is 384x256x73 and `control3s-plus` is
512x320x73. Motion also exposes `motion3s-experimental` at 640x384x73 as a
three-second-only canary; it has substantially higher attention, swap, runtime,
and Python/Metal termination risk on 16 GB and is not a routine quality tier.
`--duration-seconds 1..10` maps to `24*N+1` frames for the standard control
profiles. The UI warns past five seconds because memory and continuity risk rise
sharply.

## Limits

- Exactly one IC-LoRA adapter is active per render.
- Uploaded timeline anchors can be layered on that one adapter.
- Prompt Relay/Scene Beats belong to Generate; `ic-lora` has no `--segment`.
- IC-LoRA is single-stage here. Extend and A2V are separate Q8 workflows; a
  control adapter is not applied to those jobs.
- Low-RAM streams weights; it cannot remove quadratic reference-attention cost.
- On an M1 Pro with 16 GB unified memory, advanced Q8 jobs are expected to take
  tens of minutes. A first three-second Extend is realistically about 25–60
  minutes and may take longer under memory pressure. This does not change the
  speed of ordinary three-second Q4 Generate jobs.

Use `--dry-run` for GPU-free command validation and `--preflight-only` for
model/adapter/hash/Metal checks without rendering.
