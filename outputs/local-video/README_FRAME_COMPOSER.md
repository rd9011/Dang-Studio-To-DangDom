# Optional local Frame Composer

Frame Composer uses **FLUX.2 Klein 4B Q4** to turn 1–4 tagged reference
photos into a still image at an exact LTX anchor resolution. One user-facing
ingredient tag can hold multiple views: for example, two photos tagged
`@maya` plus one tagged `@jacket` use three of the four photo slots but only
two prompt tags. The result can be assigned to the video UI as the Start
frame, End frame, or a new Interior timeline anchor.

Frame Composer remains the lightest way to build a precise start, end, or
interior anchor for ordinary Q4 Generate. It is also useful when the desired
composition needs several references but not the native video-level
Ingredients adapter.

Native Ingredients and Motion are now available through the separate selective
Q8 advanced lane. They do not run against Q4: the Q8 overlay preserves enough
adapter influence while leaving the fast Generate path unchanged. Native
Ingredients holds multiple references through the video itself and is fixed at
121 frames / 5.04 seconds; Frame Composer instead produces one still anchor.
Choose between them based on scene complexity, or compose a still and also use
native Ingredients when both forms of guidance are useful.

## What is pinned

- Runtime: `lpalbou/mlx-gen` tag `v0.38.0`, exact commit `99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3`
- Model: `AbstractFramework/flux.2-klein-4b-4bit`, exact revision `4540c0084d0bcfdd29b19a2b0f62c77bef18f5bd`
- Model manifest: 14 files, 4,619,712,832 bytes total, with an exact SHA-256 for every file
- Inference: four distilled steps, guidance `1.0`, low-RAM mode with a 1 GB
  MLX cache, and `exact-resize` canvas policy so portrait references cannot
  override the selected LTX anchor dimensions

The pinned [Hugging Face revision tree](https://huggingface.co/AbstractFramework/flux.2-klein-4b-4bit/tree/4540c0084d0bcfdd29b19a2b0f62c77bef18f5bd)
reports 4,619,599,193 bytes of storage because that figure counts its six LFS
objects. The eight small Git-tracked files add 113,639 bytes, producing the
installer's complete 14-file total of 4,619,712,832 bytes. All six LFS hashes
were checked against the revision API, and all eight small files were streamed
from that revision and independently SHA-256 checked.

The installer creates `work/local-video/frame-composer/.venv`. It uses the runtime repository's committed `uv.lock`; it never installs into or imports packages from the pinned LTX virtual environment.
The source checkout is pinned and sparse (`src` plus required root metadata),
so large repository documentation/output artifacts are not downloaded.

## Install

No model weights are bundled with the studio. From the workspace root:

```bash
python3 outputs/local-video/install_frame_composer.py --plan
python3 outputs/local-video/install_frame_composer.py --install
python3 outputs/local-video/install_frame_composer.py --verify
```

The model itself is about 4.30 GiB. The installer requires roughly 9.3 GiB free before starting so the completed model and 5 GiB of operating headroom coexist safely. This pinned model revision is public, so the installer sends no account token.

The public model files are fetched with macOS system `curl`, restricted to
verified HTTPS redirects and the immutable revision URL; TLS verification is
never disabled and no token is accepted on the command line. Downloads go to
a private staging directory. The final model directory is published only after
all 14 files match their pinned byte sizes and SHA-256 hashes and the staged
tree contains no extra files, directories, or symlinks. A receipt binds the
model revision and manifest digest to the exact runtime commit.

After installation, generation is forced offline with the standard Hugging Face, Transformers, and Diffusers offline flags. Only the verified local model path is passed to MLX-Gen.

## Use in the UI

1. Open **Frame Composer** in the Generate screen.
2. Add an ingredient with a lowercase tag such as `@maya`, `@jacket`, or
   `@car`, then select one or more reference views for it. Add more ingredients
   as needed. There may be at most four photos total across all ingredients.
3. Mention every active ingredient tag in the frame prompt. Mentioning a tag
   addresses all of its views. Example: `Keep @maya's face and hair; dress
   @maya in the exact jacket from @jacket; one coherent person, no duplicates,
   no text.`
4. Choose the LTX size that matches the video profile. The generated PNG uses those exact pixel dimensions.
5. Click **Compose frame locally**.
6. When the PNG appears, choose **Use as Start**, **Use as End**, or **Add as Interior**.
7. Finish the main video prompt and generate normally.

The UI's broader ingredient library may contain eight entries, but FLUX.2
Klein accepts at most four active reference images for one composition. Every
photo counts, including multiple views of the same character. Frame
composition and LTX rendering are serialized so both large models are not
loaded into the Mac's unified memory at the same time.

## Direct wrapper use

Outputs are intentionally confined to `outputs/local-video/generated-frames` and are published atomically:

```bash
work/local-video/frame-composer/.venv/bin/python \
  outputs/local-video/compose_ltx_frame.py \
  --reference @maya /absolute/path/maya-front.png \
  --reference @maya /absolute/path/maya-profile.png \
  --reference @jacket /absolute/path/jacket.png \
  --profile quality \
  --seed 4242 \
  --output frame-4242.png \
  "Keep @maya's identity; dress @maya in the exact @jacket; one person, no text."
```

Tags are UI/wrapper aliases rather than native model syntax. The wrapper
passes `--image` arguments in exactly the supplied order. It assigns every
view a stable internal alias such as `maya__view_1` and `maya__view_2`, then
expands `@maya` into one reference group containing both positional images.
The compiled prompt explicitly says that views in a group are complementary
evidence for one ingredient, not separate subjects. Unknown, invalid, missing,
or unmentioned ingredient tags fail before the model loads; references are
never silently truncated.

## Honest limitations

- This composes a strong still anchor; it does not guarantee perfect identity across every generated video frame.
- Grouping is deterministic prompt translation, not a trained face-ID system.
  Complementary, well-lit views generally provide cleaner evidence than
  contradictory outfits, ages, or appearances, and the model can still blend
  or ignore details.
- Four photos is a model/runtime limit, not a UI preference. A base canvas or extra character view also consumes one slot.
- The optional Frame Composer model and the LTX model are separate downloads and separate runtimes.
- A generated anchor PNG is capped at 8 MB because that is the studio's safe per-image upload limit.
