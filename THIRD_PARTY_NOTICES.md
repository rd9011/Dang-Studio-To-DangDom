# Third-party notices

Dang Studio To DangDom is a local workflow application. It does not relicense
software, model weights or datasets installed through it. Every third-party
component remains subject to its publisher's terms.

The public repository does not include large model checkpoints. The
convenience DMG also keeps large models separate: first-run setup downloads
only the components the user selects after displaying storage and licence
information.

| Component | Purpose | Project / terms |
| --- | --- | --- |
| Python | Application and installer runtime | [Python licence](https://docs.python.org/3/license.html) |
| uv | Reproducible Python environment management | [Astral uv](https://github.com/astral-sh/uv); see `third_party_licenses/uv-MIT.txt` |
| FFmpeg | Local decode, encode and mux | [FFmpeg legal information](https://ffmpeg.org/legal.html) |
| LTX-2 MLX | Apple-Silicon inference runtime | [mrbizarro/ltx-2-mlx](https://github.com/mrbizarro/ltx-2-mlx) |
| LTX-Video / Gemma encoder packs | Video generation and text encoding | Publisher terms are shown by the relevant installer before download |
| Ollama | Optional private local Prompt Coach runtime | [Ollama](https://github.com/ollama/ollama); see `third_party_licenses/Ollama-MIT.txt` |
| Qwen3 | Optional Prompt Coach language model | [Qwen model page and licence](https://ollama.com/library/qwen3) |
| MLX-GEN / FLUX.2 | Optional Frame Composer | Publisher terms are shown before model download |
| Chatterbox | Optional character-voice workflow | Publisher licence is preserved with the installed runtime |
| Real-ESRGAN Core ML | Optional AI video upscaler | Publisher licence is preserved with the installed runtime |

Source installers pin required revisions and hashes where the upstream
distribution provides them. A component manager acceptance screen or
`--accept-license` flag records user confirmation; it is not legal advice and
does not grant access to a gated publisher repository.

Users must review the linked terms and confirm they have the rights required
for model use, reference media, character likenesses, voices and generated
output. The project itself currently has no root `LICENSE`; third-party notices
do not grant rights to the first-party Studio source.
