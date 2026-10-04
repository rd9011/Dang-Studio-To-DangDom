# Optional LTX-2.5 control adapters

`install_optional_ltx25_assets.py` now installs only the pinned IC-LoRA
adapters used by Ingredients, Union Control, and Motion Track. It deliberately
does not install or replace base transformer, Gemma, upscaler, dev, HQ, A2V,
or standalone keyframe weights.

The adapters run only against the verified selective Q8 advanced overlay. They
are not applied to the Q4 transformer because Q4 quantization retains too
little of the LoRA delta for reliable control. The model layout is:

```text
work/local-video/models/ltx-2.5-mlx-q4        # ordinary fast Generate
work/local-video/models/ltx-2.5-mlx-q8        # advanced control/edit lane
work/local-video/models/gemma4-12b-ltx25-q4  # shared text encoder
```

Install and verify the Q8 overlay first:

```bash
python3 outputs/local-video/install_ltx25_advanced_q8.py --plan
python3 outputs/local-video/install_ltx25_advanced_q8.py --install --accept-license
python3 outputs/local-video/install_ltx25_advanced_q8.py --verify
```

Start that install with at least **60 GiB free**. Do not mix older `dgrauet` Q4
extras into any model directory.

## Plan and install

`--plan` is read-only. It prints exact sources/destinations, verified existing
files, resumable bytes, remaining transfer, and the disk preflight without
creating directories or contacting the network.

```bash
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle union --plan
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle motion --plan
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle ingredients --plan
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle controls --plan
```

Bundles can be repeated; duplicates are downloaded once. With nothing already
present, adapter transfers are approximately:

| Bundle | Download | Disk requirement including 6 GiB reserve |
| --- | ---: | ---: |
| `union` | 0.61 GiB | 6.61 GiB |
| `motion` | 0.30 GiB | 6.30 GiB |
| `ingredients` | 1.22 GiB | 7.22 GiB |
| `controls` / `all` | 2.13 GiB | 8.13 GiB |

After reviewing the plan:

```bash
hf auth login
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle ingredients
python3 outputs/local-video/install_optional_ltx25_assets.py --bundle motion
```

Those two bundles are the required additions for Ingredients and Motion Track:
1,636,096,786 bytes total (about 1.52 GiB). Install `--bundle union` only if
Canny/depth/pose control is wanted. `--bundle controls` installs all three.

Ingredients is the native LTX-2.5 adapter. Motion intentionally uses the
LTX-2.3 Motion Track adapter because that is the adapter referenced by the
official LTX-2.5 Motion workflow; no LTX-2.3 base model or Gemma 3 pack is
required. Each control mode runs as a separate Q8 `--single-stage --low-ram`
job, not as an arbitrary stack of adapters.

The installer can read `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`,
or the standard Hugging Face token file. Tokens are never command-line
arguments or printed, and credentials are removed on redirects to a different
signed-CDN origin.

The Ingredients repository is gated. Accept its terms in the browser before
installing that bundle. A 401/403 response identifies the repository whose
authorization is missing.

## Integrity and recovery

Each source URL includes an immutable 40-character revision. Downloads use a
hidden partial next to the final destination. Resume occurs only with matching
metadata, a stable server validator, an exact HTTP 206 range, and the expected
total length. Otherwise the partial safely restarts.

Before atomic promotion, the installer validates exact byte length and
safetensors structure, plus the published SHA-256 for every adapter. Existing
invalid destinations remain untouched unless `--replace-invalid` is explicit.
The gated Ingredients adapter is pinned to its immutable revision, exact size,
and published LFS SHA-256, and the installer records that digest in its local
receipt for every later check.

TLS verification is enabled by default. `--insecure-tls` exists only as an
explicit last resort for a known intercepting corporate proxy; it weakens
transport authenticity and should not be the normal CDN workaround. Exact
SHA-256 verification remains mandatory.

When a corporate proxy installs its own trusted root, prefer supplying that
root through the standard `SSL_CERT_FILE` environment mechanism instead of
disabling TLS verification.

## Pinned adapter manifest

| Adapter | Repository revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Union | `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control@b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff` | 654,465,352 | `a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18` |
| Motion | `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control@572bb9c9a1ba3d8e8724cce69783ffc2422386db` | 327,309,314 | `e279807ee3aa3db1ce60188d665ff83342860367dcd6bac19f8bd5a99a9e1dca` |
| Ingredients | `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients@12040e4091ac2008d3906a594e31a7fb1ab9d546` | 1,308,787,472 | `ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95` |

Adapters are stored in
`work/local-video/models/LTX-2.5-control-adapters/`; the receipt is
`work/local-video/models/.ltx25-optional-assets.json` with private permissions.
