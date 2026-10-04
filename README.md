# Dang Studio To DangDom

Dang Studio To DangDom is a local, browser-based video studio for LTX-2.5 on
Apple Silicon Macs. The complete localhost application source, installers,
Prompt Coach, tests and operator documentation live in this repository. It
binds to `127.0.0.1`, performs inference on the Mac and keeps prompts,
references, character profiles and generated media local.

> **Public-source pre-release; licence pending.** The owner has not selected a
> root project licence yet. You may inspect and run this repository, but its
> publication does not currently grant permission to copy, modify or
> redistribute the project. This is therefore not legally open source yet.
> Third-party models and runtimes have their own terms.

## Choose how to run it

| Route | Best for | What you receive |
| --- | --- | --- |
| **Clone and run on localhost** | Developers, auditing and full control | The complete public Python/browser application in this repository |
| **Install the convenience DMG** | People who want a guided, plug-and-play setup | A prebuilt macOS app that downloads selected models after showing storage and licence information |

The ready-made DMG is an additional release product, not a replacement for the
localhost source. The native macOS wrapper and the internal process used to
assemble the `.app` and DMG are intentionally not part of this public
repository. No private packaging code is required to use the complete Studio
in a browser.

## Source quick start

### Requirements

- Apple Silicon Mac; Intel Macs are not supported
- macOS 13 or newer
- 16 GB unified memory minimum
- Xcode Command Line Tools, Git, Python 3.11, `uv` and FFmpeg
- Internet access while installing runtimes and models
- About 40 GB free for the base Q4 installation; leave additional room for
  generated media

One Homebrew-based way to install the command-line prerequisites is:

```bash
xcode-select --install
brew install git python@3.11 uv ffmpeg
```

Then clone the repository and install the pinned runtime:

```bash
git clone https://github.com/rd9011/Dang-Studio-To-DangDom.git
cd Dang-Studio-To-DangDom

python3.11 outputs/local-video/install_ltx25_runtime.py --plan
python3.11 outputs/local-video/install_ltx25_runtime.py --install
```

The base video and text-encoder packs are about 27.47 GB. First inspect the
network and disk plan. Before installing, review the linked
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x):

```bash
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --preflight
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --plan
python3.11 outputs/local-video/install_phosphene_ltx25_release.py \
  --install --accept-license
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --verify
python3.11 outputs/local-video/verify_ltx25.py
```

`--accept-license` records your confirmation; it does not replace reading the
publisher's terms. The installer is resumable and verifies every downloaded
shard before publishing it to the model directory.

Start the browser Studio:

```bash
./outputs/local-video/launch_ui.command
```

It opens at [http://127.0.0.1:7860/](http://127.0.0.1:7860/). Keep the Terminal
window open; press `Control-C` there to stop the server. See
[`outputs/local-video/QUICKSTART.md`](outputs/local-video/QUICKSTART.md) for the
first render, optional Q8 workflows and troubleshooting.

## Convenience DMG

Download the current Apple Silicon pre-release:

- [Dang Studio To DangDom 0.1.0-pre DMG](https://github.com/rd9011/Dang-Studio-To-DangDom/releases/download/v0.1.0-pre/Dang-Studio-To-DangDom-0.1.0-arm64.dmg)
- [Published SHA-256 file](https://github.com/rd9011/Dang-Studio-To-DangDom/releases/download/v0.1.0-pre/SHA256SUMS.txt)

Expected SHA-256:

```text
48df0de1f05c9f65b2c914a31baaa98a0a1f6e85401d8a731bc8061027dfee24  Dang-Studio-To-DangDom-0.1.0-arm64.dmg
```

Verify the download before opening it:

```bash
shasum -a 256 ~/Downloads/Dang-Studio-To-DangDom-0.1.0-arm64.dmg
```

Open the DMG, drag **Dang Studio To DangDom.app** to `/Applications`, and launch
it. This pre-release is ad-hoc signed and not Apple-notarized. If Gatekeeper
blocks the first launch, use Finder's **Control-click → Open** flow after
checking the hash; do not disable Gatekeeper system-wide.

The DMG deliberately omits large model checkpoints. First-run setup offers
Essential, Custom and Full Studio plans, shows the current download/storage
estimate and third-party terms, and installs the chosen components under
`~/Library/Application Support/Dang Studio To DangDom/`. Removing the app alone
does not remove those downloaded components or user data.

## What the Studio includes

- Q4 text/image-to-video generation with native audio
- Start, end and timed interior frame anchors
- Saved visual character profiles and multiple reference views
- Optional Q8 Retake, Extend, Audio-to-Video, Ingredients, Union Control and
  Motion Track workflows
- Optional FLUX.2 frame composition and multilingual character voices
- Local Prompt Coach guidance and prompt-development skill
- Frame capture for creating a continuation shot
- Fast resize and local AI-quality video upscaling
- Aspect-ratio presets, quality profiles and 1–10 second duration controls

Optional features have separate model, memory, storage and licence
requirements. Their buttons remain unavailable until the exact local assets
pass verification.

## Storage and performance

| Plan | Approximate download | Practical free space before setup |
| --- | ---: | ---: |
| Base Q4 / Essential | 27.9 GB | 40 GB recommended |
| Full Studio | 83.9 GB | 100 GB recommended |

These are current pre-release estimates; the setup screen or a source
installer's `--plan` output is authoritative. Keep the Mac awake and plugged
in during large downloads. On 16 GB Macs, Q4 is the practical default. Q8,
multiple references and experimental high-resolution controls may swap
heavily, run for tens of minutes or terminate under memory pressure.

## Network and privacy boundary

Installation is networked. Depending on the components selected, installers
can contact GitHub Releases and `codeload.github.com`, Python package indexes,
Hugging Face, ModelScope and the Ollama registry. Gated models may require the
user to accept publisher terms and authenticate with that publisher.

Once the required components are installed, video inference and user media
remain local. The Studio server listens only on `127.0.0.1`; there is no cloud
inference service or analytics integration. Component installation, updates
and external help/licence links still need network access.

## Public source boundary

Included here:

- `outputs/local-video/ltx_ui.py` — localhost server and browser interface
- `outputs/local-video/*_ltx25.py` — generation, edit and conditioning launchers
- `outputs/local-video/install_*.py` — source-side component installers
- `outputs/local-video/prompt_coach.py` and `skills/` — local prompt guidance
- `outputs/local-video/test_*.py` — portable unit and integration tests
- `.github/workflows/source-validation.yml` — validation for the public source

Excluded from Git are model weights, downloaded third-party trees, virtual
environments, caches, local receipts, user media, generated outputs,
credentials, built applications and disk images. The native wrapper and
`.app`/DMG assembly implementation are also a private release-engineering
layer. Users do not need that layer to run the browser Studio from source.

## Development

Enable the staged-content safety hook after cloning:

```bash
git config core.hooksPath .githooks
```

Run the same source validation as CI:

```bash
python3.11 scripts/scan_staged_secrets.py --tracked
bash .github/scripts/prepare_ltx_source_fixture.sh
python3.11 -m pip install 'Pillow==11.3.0'
python3.11 -m compileall -q outputs/local-video scripts
python3.11 -m unittest discover -s outputs/local-video -p 'test_*.py'
```

The fixture command downloads only the small pinned runtime source needed by
installer tests, not model weights. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Caveats and responsible use

- This is a pre-release. The DMG is not notarized and there is no automatic
  updater.
- Local generation can hallucinate, change identity or anatomy, produce
  unstable motion, miss lip-sync, or create unintended audio. Review every
  result before publishing it.
- AI upscaling can restore edges and texture but cannot repair composition,
  anatomy, identity or motion; fine details may shimmer between frames.
- Reference images and detailed prompts improve consistency but do not
  guarantee it.
- Users are responsible for the rights to prompts, reference media, character
  likenesses, voices and generated output, and for compliance with applicable
  model terms and law.

Read [CAVEATS.md](CAVEATS.md) before a long install or production render.

## Project and model licences

This project currently has no root `LICENSE`. Public source visibility permits
inspection and local evaluation but does not yet grant general reuse rights.
The project owner must choose a licence before this can accurately be called
open source.

This repository also does not grant rights to LTX, Gemma-derived encoders,
FLUX, Chatterbox, Real-ESRGAN or other third-party components. Review their
publisher terms separately; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
