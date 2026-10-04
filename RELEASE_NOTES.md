# Release notes

## 0.1.0-pre — 2026-10-04

This is the first public-source pre-release of **Dang Studio To DangDom** for
Apple Silicon Macs running macOS 13 or newer.

> The repository is publicly inspectable and runnable, but it has no root
> `LICENSE` yet. General permission to copy, modify or redistribute has not
> been granted. Third-party components retain their own terms.

### Two supported entry points

- **Localhost source:** clone the repository, install the pinned runtime and
  model packs, then run `outputs/local-video/launch_ui.command`. This is the
  complete browser application and opens at `http://127.0.0.1:7860/`.
- **Convenience DMG:** install the prebuilt app for a guided first-run download.
  The native wrapper and internal app/DMG assembly process are not published;
  they are not needed for source use.

### Included application features

- Q4 LTX-2.5 Generate with text, image, start/end and interior anchors
- Aspect-ratio, duration and quality controls
- Character profiles, reference views and frame-to-next-clip capture
- Optional Q8 Retake, Extend, Audio-to-Video, Ingredients, Union Control and
  Motion Track workflows
- Optional frame composition, character voices, Prompt Coach and local AI
  video upscaling
- Restart-safe, hash-verifying source installers with explicit third-party
  terms

### Source installation

The base path requires Python 3.11, `uv`, FFmpeg, Git and Xcode Command Line
Tools. From a clone:

```bash
python3.11 outputs/local-video/install_ltx25_runtime.py --plan
python3.11 outputs/local-video/install_ltx25_runtime.py --install
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --preflight
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --plan
python3.11 outputs/local-video/install_phosphene_ltx25_release.py \
  --install --accept-license
python3.11 outputs/local-video/install_phosphene_ltx25_release.py --verify
python3.11 outputs/local-video/verify_ltx25.py
./outputs/local-video/launch_ui.command
```

Review the LTX-2.x Community License before using `--accept-license`. See the
root README and `outputs/local-video/QUICKSTART.md` for details.

### Convenience DMG

- [Download `Dang-Studio-To-DangDom-0.1.0-arm64.dmg`](https://github.com/rd9011/Dang-Studio-To-DangDom/releases/download/v0.1.0-pre/Dang-Studio-To-DangDom-0.1.0-arm64.dmg)
- SHA-256:
  `48df0de1f05c9f65b2c914a31baaa98a0a1f6e85401d8a731bc8061027dfee24`

The DMG contains no large generative model checkpoints. First-run setup
downloads selected components after showing their terms, storage estimate and
destination.

### Capacity

| Plan | Approximate download | Recommendation |
| --- | ---: | ---: |
| Base Q4 / Essential | 27.9 GB | Start with 40 GB free |
| Full Studio | 83.9 GB | Start with 100 GB free |

The current setup screen or source installer's `--plan` output is authoritative.
Leave extra room for temporary work and generated media.

### Important caveats

- The arm64 DMG is ad-hoc signed, not Apple Developer ID-signed and not
  notarized. It has no automatic updater.
- A 16 GB Mac meets the minimum, but Q8 and experimental high-resolution
  workflows can swap heavily, become very slow or terminate under pressure.
- Setup can contact GitHub, Python package indexes, Hugging Face, ModelScope and
  the Ollama registry. Inference and user media remain local after installation.
- Gated model access may require separate publisher acceptance and
  authentication. Studio consent does not grant external access.
- Model output may hallucinate or drift in identity, anatomy, motion, dialogue
  or audio. Upscaling cannot repair those generation errors and may shimmer.
- Users are responsible for rights to references, voices, characters and
  generated content, and for reviewing output before publication.

See [CAVEATS.md](CAVEATS.md) for the full pre-release limitations.
