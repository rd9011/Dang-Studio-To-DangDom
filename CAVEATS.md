# Pre-release caveats

Read this before installing large components or relying on a generated result.

## Platform and release status

- The supported platform is Apple Silicon with macOS 13 or newer. Intel Macs
  are not supported.
- The minimum supported memory is 16 GB unified memory. That minimum is for the
  practical Q4 path; it does not make every advanced workflow safe.
- The convenience DMG is an early arm64 pre-release. It is ad-hoc signed, not
  Apple Developer ID-signed, not notarized and has no automatic updater.
- Verify the release SHA-256 before using Finder's **Control-click → Open**
  flow. Never disable Gatekeeper system-wide.

## Storage, heat and runtime

- Begin with about 40 GB free for the base Q4/Essential setup or 100 GB for the
  current Full Studio plan. Installer `--plan` output and the setup UI are
  authoritative when catalog sizes change.
- Leave additional room for download staging, swap and generated media. A
  nearly full disk can fail even when final model sizes appear to fit.
- Keep laptops awake, ventilated and connected to power during downloads and
  renders.
- Q8 controls, multiple references and experimental high-resolution profiles
  can swap heavily, take tens of minutes or terminate under memory pressure on
  16 GB systems. Test at three seconds and low resolution first.
- Only one compute-heavy Studio job runs at a time.

## Model behaviour

- Generated video can hallucinate objects or details; change identity,
  anatomy, wardrobe or scale; destabilize motion; miss lip-sync; or create
  unintended speech and sound.
- Start/end frames, Ingredients, character profiles, stable tags and concise
  chronological prompts reduce drift but cannot guarantee consistency.
- A successful seed is not a guarantee that a longer duration or higher
  profile will preserve the same result.
- AI upscaling can reconstruct edges and texture but cannot repair bad anatomy,
  identity, motion, framing or composition. Frame-wise enhancement may shimmer.
- Review every result, including audio, before publishing it.

## Downloads, privacy and credentials

- Initial installation and optional component setup require network access and
  may contact GitHub, Python package indexes, Hugging Face, ModelScope or the
  Ollama registry.
- Some gated components require separate publisher acceptance and login. A
  Studio confirmation does not grant access to an external account.
- Completed local inference does not upload prompts, reference images or
  generated media to a cloud inference service. The browser server listens on
  `127.0.0.1` only.
- Do not share install logs that contain local paths, account names or tokens.

## Rights and licensing

- You are responsible for having the rights to reference media, character
  likenesses, voice samples, prompts and generated output.
- Model and runtime terms are separate from this project and can restrict use
  or redistribution.
- This repository currently has no root project `LICENSE`. It is public source
  for inspection and local evaluation, but it is not yet legally open source.

## Source and DMG boundary

The public repository contains the complete localhost/browser application and
its source installers. The convenience DMG is an additional binary release.
The native wrapper and the internal process for assembling the `.app` and DMG
are maintained privately and are not required to run the browser Studio.
