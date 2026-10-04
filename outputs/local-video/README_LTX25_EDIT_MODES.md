# LTX-2.5 edit/input modes

Retake, Extend, and Audio-to-Video use a separate selective Q8 advanced lane.
The ordinary Generate workflow stays on Q4, so installing or using an advanced
mode does not change the normal three-second generation path.

The model layout is:

```text
work/local-video/models/ltx-2.5-mlx-q4       # fast distilled Generate
work/local-video/models/ltx-2.5-mlx-q8       # selective advanced overlay
work/local-video/models/gemma4-12b-ltx25-q4 # shared text encoder
```

`edit_ltx25.py` always points advanced operations at the Q8 overlay and never
reuses, overwrites, or silently reinterprets the Q4 transformer. Retake and
Extend use `transformer-dev.safetensors`. A2V uses the dev transformer for
stage one and the pre-fused Q8 distilled transformer for its low-RAM stage two.
The separate 8.9 GB distilled LoRA is therefore deliberately excluded.

Until the Q8 overlay has a valid receipt, the studio disables these submit
buttons and the direct wrappers stop at preflight with an exact missing-asset
message. After installation, every request still verifies the receipt, local
model/source contract, low-RAM runtime patch, disk, ffmpeg, and Metal before
loading weights.

## Install and verify the advanced lane

Start with at least **60 GiB free**. This is a safe operating target for the
41.19 GB selective download, the largest temporary release shard, the
installer's 6 GiB reserve, and normal macOS headroom.

```bash
python3 outputs/local-video/install_ltx25_advanced_q8.py --preflight
python3 outputs/local-video/install_ltx25_advanced_q8.py --plan
python3 outputs/local-video/install_ltx25_advanced_q8.py --install --accept-license
python3 outputs/local-video/install_ltx25_advanced_q8.py --verify
```

The installer uses immutable manifests, resumable verified shards, exact
relative links to the byte-identical Q4 sidecars, and an atomic overlay
receipt. `--preflight` checks manifests/CDN without downloading weights;
`--plan` reports exact missing bytes and the current conservative disk need;
`--verify` hash-checks the completed overlay without downloading weights.

## Why these controls are not mapped onto Q4 Generate

Generate uses `--distilled` with a trained `5+2` or `8+2` schedule and does not
accept CFG/STG values. Retake, Extend, and A2V use dev inference and the
upstream edit defaults (typically 30 steps, CFG 3, STG 1). Sending those flags
to the distilled transformer would be a different, unsupported pipeline, not a
quality preset.

Uploaded start, end, and interior image anchors remain supported by normal
distilled Generate. The standalone keyframe pipeline is not exposed by this
local wrapper.

## Extend

The default extension adds nine latent frames, which is 72 pixel frames or
three seconds at 24 fps:

```bash
python3 outputs/local-video/edit_ltx25.py extend \
  "Continue the same shot; preserve the person, wardrobe, lighting, and camera motion" \
  --video /absolute/source.mp4 \
  --extend-seconds 3 --direction after \
  --output edits/extended.mp4
```

`--direction before` extends the opening instead. `--extend-frames` addresses
latent frames directly, and `--target-frames` accepts a target on the `8k+1`
pixel-frame grid. The wrapper caps the result at 241 frames (about ten seconds)
for this 16 GB machine. Source footage must fit the wrapper's resolution and
frame-count limits; run `--preflight-only` to validate it without rendering.

## Audio-to-Video

```bash
python3 outputs/local-video/edit_ltx25.py a2v \
  "Close cinematic portrait, subtle natural speech gestures, stable face" \
  --audio /absolute/dialogue.wav \
  --profile clip3s --frames 73 \
  --start-image /absolute/portrait.png \
  --output edits/dialogue.mp4
```

The selected audio must contain enough material after `--audio-start` for the
requested frames. `--frames` must satisfy `8k+1` and cannot exceed 241. The
start image is optional but useful for identity and composition.

## Retake

Retake regenerates a selected latent interval while keeping the surrounding
source video:

```bash
python3 outputs/local-video/edit_ltx25.py retake \
  "Repair the hand movement while preserving the same shot" \
  --video /absolute/source.mp4 \
  --start-seconds 1.0 --end-seconds 2.0 \
  --preserve-audio \
  --output edits/retake.mp4
```

Use either the seconds pair or `--start-frame`/`--end-frame`, not both.

## Runtime expectations on M1 Pro 16 GB

Q8 is substantially heavier than Q4 even with low-RAM block streaming. Plan
on tens of minutes per advanced render: a first three-second Extend is a
realistic **25–60 minute estimate**, and it can take longer depending on source
resolution, step count, thermal state, and memory pressure. Longer source clips
also increase the total attention workload. Start with three seconds and low
resolution, then raise quality only after the motion is correct.

This cost applies only to an advanced Q8 job. Ordinary Q4 Generate speed is
unchanged. Low-RAM reduces peak memory; it does not make a 22B Q8 transformer
fast or guarantee that every request will fit alongside other open apps.

Use `--dry-run` for GPU-free request/argv validation and `--preflight-only` for
the complete model, input, disk, ffmpeg, source-patch, and Metal checks without
rendering. Outputs are confined and published atomically; an existing file is
not replaced unless `--force` is explicit.

Do not copy individual dev weights into the Q4 base directory. The old
`dgrauet` optional bundle is not compatible with this contract and is not
offered by `install_optional_ltx25_assets.py`.
