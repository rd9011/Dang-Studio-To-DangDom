# Contributing to Dang Studio To DangDom

Thank you for helping improve the complete localhost/browser Studio. This
public repository contains the application source, component installers,
Prompt Coach, tests and documentation needed to run it from a clone. It must
remain safe to clone: no private creative work, model payloads, credentials or
developer-machine state belong in Git.

## Repository boundary

Commit application source, tests, documentation, small configuration files and
original source assets only. Do **not** commit:

- model weights or converted model packages (`.safetensors`, `.gguf`, `.ckpt`,
  `.pt`, `.pth`, `.mlpackage` or `.mlmodelc`);
- generated video, audio, frames, upscales or user reference images;
- character libraries, voice samples, prompts or other user-created content;
- virtual environments, downloaded third-party trees, caches, logs, component
  receipts or resumable-download fragments;
- built applications, DMGs, PKGs or release-engineering artifacts;
- access tokens, cookies, signing identities, provisioning profiles, private
  keys, `.env` files or machine-specific configuration.

The native macOS wrapper and internal `.app`/DMG assembly process are not part
of this public source project. Contributions should target the complete
localhost experience or its portable Python components; do not add proprietary
packaging material to a public pull request.

Downloaded and generated data belongs in the ignored `work/` and output paths.
Third-party models and runtimes remain subject to their own licences and should
be installed only after the user reviews the applicable terms.

Enable the staged-content safety check after cloning:

```bash
git config core.hooksPath .githooks
```

Do not bypass that hook to publish a blocked file. If it reports a false
positive, explain the file and proposed exception in the pull request. Report
suspected credentials or security issues through [SECURITY.md](SECURITY.md),
not a public issue.

CI applies the same tracked-file scan:

```bash
python3 scripts/scan_staged_secrets.py --tracked
```

## Development setup

The supported host is an Apple Silicon Mac with macOS 13 or newer, Xcode
Command Line Tools, Git and Python 3.11. The portable test suite does not need
model weights, Metal access or a complete Studio installation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'Pillow==11.3.0'
```

One runtime-installer test verifies two exact files from the pinned LTX source.
Prepare its small ignored fixture with:

```bash
bash .github/scripts/prepare_ltx_source_fixture.sh
```

This downloads source code only. It does not download model weights, install a
runtime or write outside the ignored `work/` directory. It refuses to replace
an existing fixture with mismatched revision or hashes.

## Run validation

Run the same public-source validation used by CI:

```bash
python -m compileall -q outputs/local-video scripts
python -m unittest discover -s outputs/local-video -p 'test_*.py'
```

Tests that exercise a real model download, render, Metal runtime or binary
release remain separate release checks and do not belong in ordinary pull
request CI.

## Local application rules

- Resolve writable locations through `studio_paths.py`; never embed a home
  directory, checkout path or developer username.
- Pin downloaded code and data to immutable revisions and expected hashes.
  Verify downloads before atomically publishing them to an installed path.
- Keep the web service bound to loopback, preserve Host-header, request-size,
  upload-type and path validation, and do not add analytics or external
  prompt/media uploads.
- Never weaken explicit licence consent for gated or restricted components.
- Preserve user media and installed models during updates. Destructive cleanup
  requires an explicit, narrowly scoped user action.
- Add or update tests when changing a component manifest, download contract,
  filesystem layout, API, browser workflow or rendering command.
- Keep ordinary Q4 generation independent from optional Q8, voice, frame
  composition and upscaling components.

## Pull requests and commits

Keep commits small and purpose-specific. Before opening a pull request:

1. Rebase on `main` and inspect `git status` for accidental local or generated
   files.
2. Run the tracked-file scan, compile check and Python tests above.
3. Describe user-visible behaviour, disk/network impact, model or licence
   changes and clean-checkout testing performed.
4. Include no generated model or user-media output as proof; summarize results
   or use intentionally small, original test fixtures.

The project licence has not yet been selected. Until a root `LICENSE` exists,
do not assume contribution or redistribution terms, and do not import code or
assets whose terms are incompatible or unclear. Preserve all required
third-party notices and identify provenance in the pull request.
