# Revision-pinned Gemma fallback

The normal and preferred text encoder remains the exact Phosphene pack at
`work/local-video/models/gemma4-12b-ltx25-q4`.

`ddalcu/LTX-2.5-MLX-Serve-4bit` revision
`e8f8c97cd6e4ff9c997e0d382c027ad9b61776bb` publishes a Gemma tensor file with
the expected model family, but it is **not byte-identical** to Phosphene's
official file: it is 19,386 bytes larger and has a different SHA-256. The
importer therefore never installs it under the official pack identity.

## What the importer proves

- exact source size `6,699,162,168` and SHA-256
  `be2d3f3d7f13b61a6ab391981b2c8c98f99135edfbde6adcae32d81f155744dd`;
- a valid, fully bounded safetensors file parsed without pickle or model code;
- an exact tensor-name match with Phosphene's pinned official index;
- every runtime-visible Gemma text tensor has the Q4 shape and dtype derived
  from the pinned `gemma4-12b-ltx-v1` config;
- every tokenizer/config/license file is a byte-for-byte pinned copy from the
  official pack; and
- a distinct atomic receipt records both identities and the non-equivalence.

The source `.partial` is opened read-only and is never moved, renamed, linked,
or deleted. Import is refused until it has the exact final size and hash. A
private full copy is verified before the fallback directory is atomically made
visible at `work/local-video/models/gemma4-12b-ltx25-q4-ddalcu-fallback`.

## Commands

After the source download and the small official Gemma metadata files finish:

```bash
python3 outputs/local-video/import_ddalcu_gemma_fallback.py --inspect-source
python3 outputs/local-video/import_ddalcu_gemma_fallback.py --install
python3 outputs/local-video/import_ddalcu_gemma_fallback.py --verify --full-hash
```

Run the real Gemma-plus-connector prompt smoke test (Metal, but no video):

```bash
work/local-video/ltx-2-mlx/.venv/bin/python \
  outputs/local-video/smoke_ltx25_gemma_prompt.py \
  "A red ball resting on a clean white table." \
  --gemma-pack ddalcu-fallback
```

Then select it explicitly for a render:

```bash
work/local-video/ltx-2-mlx/.venv/bin/python \
  outputs/local-video/generate_ltx25.py \
  "A coherent single-shot scene." \
  --profile draft \
  --gemma-pack ddalcu-fallback
```

Without `--gemma-pack ddalcu-fallback`, verification and generation continue to
require the official Phosphene Gemma pack.

In the browser UI, **Text encoder provenance → Auto** checks both verified
local packs, prefers the exact official pack, and resolves to this fallback
only when the official pack is incomplete. The resolved pack is shown in the
job metadata and the fallback notice remains in the render log.

## Limitation

The official index lists names, not tensor shapes or values. The importer can
therefore prove exact names and config-derived loader compatibility, and the
source pin proves the mirror's exact bytes, but it cannot prove value-for-value
equality with the unavailable official tensor file. The prompt-encoding smoke
test is the final functional check; the official Phosphene pack remains the
stronger provenance choice whenever its download succeeds.
