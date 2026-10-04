# Phosphene LTX-2.5 weights from GitHub

This installer bypasses the failing Hugging Face CDN and retrieves the two
required LTX-2.5 packs from the official
[`mrbizarro/Phosphene` `weights-ltx25-v1` release](https://github.com/mrbizarro/Phosphene/releases/tag/weights-ltx25-v1).
It does not need or read a Hugging Face or GitHub token.

## What it installs

| Pack | Destination | Verified payload |
|---|---|---:|
| `q4_25` | `work/local-video/models/ltx-2.5-mlx-q4` | 20.74 GB |
| `gemma4_25` | `work/local-video/models/gemma4-12b-ltx25-q4` | 6.73 GB |

The clean installation is 27.47 GB (decimal). The installer conservatively
requires the remaining files, one temporary 1.9 GB shard, and 2 GiB of untouched
reserve space. It intentionally leaves the older incomplete
`LTX-2.5-MLX-q4-dev` directory alone.

## Run it

From the workspace root:

```bash
python3 outputs/local-video/install_phosphene_ltx25_release.py --preflight
python3 outputs/local-video/install_phosphene_ltx25_release.py --plan
```

Review the binding
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x),
then install both packs:

```bash
python3 outputs/local-video/install_phosphene_ltx25_release.py --install --accept-license
```

The transfer is about 27.47 GB. Keep the Mac awake and connected to power. If
the command or network is interrupted, run the identical command again. It
will revalidate completed work and resume the current shard; it will not start
the pack over.

`--jobs 1` is the default and safest setting. If a *future resumed run* remains
consistently slow, it can use up to four independent file workers:

```bash
python3 outputs/local-video/install_phosphene_ltx25_release.py \
  --install --accept-license --jobs 3
```

Do not stop a healthy transfer merely to change this setting. Parallelism is
bounded to 1–4; each target still has exactly one writer and its shards remain
ordered. The disk preflight conservatively charges one extra 1.9 GB staging
shard per worker.

On a connection that repeatedly stalls during long responses, an opt-in range
mode keeps each request short while preserving the same shard and full-file
hash checks:

```bash
python3 outputs/local-video/install_phosphene_ltx25_release.py \
  --install --accept-license --jobs 1 --range-chunk-mib 4 --range-workers 8 \
  --attempts 20 --timeout 30 --ipv6
```

Range mode resumes at the exact current private-partial size. Every response
must be HTTPS `206`, match the requested `Content-Range` and `Content-Length`,
and retain the same strong ETag or Last-Modified validator. The 4 MiB setting
is the empirically stable choice for the GitHub CDN's long-response throttling;
each chunk is downloaded into a private temporary file, checked before append,
then the shard partial is flushed and fsynced. `--range-chunk-mib 0` is the
default and keeps the normal streaming behavior. `--range-workers 1` is also
the default. A higher value downloads non-overlapping chunks concurrently, but
only one coordinator appends them to the shard partial, strictly by byte
offset, after every chunk in the batch validates. The disk plan charges one
range temp per range worker and file worker. Total concurrency
`--jobs * --range-workers` is capped at 16. In range mode, `--timeout` also
bounds the whole chunk request so a trickling connection is retried before it
can hold a worker indefinitely. `--ipv6` is optional; keep it only after a
successful IPv6 range probe on the current network.

After completion:

```bash
python3 outputs/local-video/install_phosphene_ltx25_release.py --verify
python3 outputs/local-video/verify_ltx25.py
```

To operate on one pack, add `--pack q4_25` or `--pack gemma4_25`.

### Import a shard downloaded in a browser

If a completed official release shard was downloaded outside the installer,
move it into the installer's authenticated private staging layout with:

```bash
python3 outputs/local-video/import_phosphene_release_shard.py \
  /absolute/path/to/q4_25__transformer-distilled.safetensors.part004
```

The pack is detected by an exact lookup in a byte-pinned release manifest. You
may make the expectation explicit with `--pack q4_25`, `--pack gemma4_25`,
`--pack q8_25`, or `--pack hq_25`. For the selective Q8 overlay, the importer
accepts only the three Q8 metadata files, the eleven distilled-transformer
shards, and the eleven HQ dev-transformer shards that
`install_ltx25_advanced_q8.py` actually consumes. It deliberately rejects the
excluded 8.9 GB distilled LoRA and shared Q4 sidecars. The basename must be
unchanged. Browser working names such as `.crdownload`, `.download`, `.part`,
`.partial`, and `.tmp` are rejected; wait for the browser download to finish
first.

The importer checks the exact byte count and full SHA-256 before it changes
anything. The source and models directory must be on the same clone-capable
APFS filesystem. It then creates a distinct copy-on-write inode with macOS
`fclonefileat`, full-hashes that private clone again, publishes the installer's
exact identity sidecar, and removes the original path only after the staged
clone is durable. This isolates staging from a browser that still has the
source inode open without allocating another physical multi-gigabyte copy. By
default, existing installed files, shard partials, sidecars, or already-used
assembly positions are never replaced. Re-run the normal installer afterward;
it will independently accept and consume the verified staged shard.

The importer also journals that full-hash result on its read-only private
clone. The selective Q8 installer's disk preflight credits only exact-size
staged shards whose manifest identity, journaled SHA-256, and read-only mode all
agree; it still preserves the normal one-shard assembly allowance and 6 GiB
reserve. The installer hashes every shard again before assembly, so this space
credit does not weaken the final integrity boundary.

If that same asset already has a trusted incomplete installer partial, opt in
to replacing only that partial:

```bash
python3 outputs/local-video/import_phosphene_release_shard.py \
  --pack q4_25 --replace-partial \
  /absolute/path/to/q4_25__transformer-distilled.safetensors.part004
```

Replacement is allowed only after the external file passes its full pinned
hash. Under the shared installer lock, the old file must be a regular file
smaller than the pinned shard, byte-for-byte equal to the corresponding prefix
of the verified external file, and paired with the installer's exact identity
sidecar (an authenticated resume validator is also allowed). Full-size,
oversized, corrupt, symlinked, unbound, weak-validator, or already-assembled
conflicts remain hard failures. The independently full-hashed CoW clone
atomically replaces the old partial; the original external path is removed only
after that commit. This uses no second physical multi-gigabyte allocation.

Do not use an older already-running installer process for this handoff. The
current installer and importer share a per-model staging lock, so all processes
started from these files serialize safely; a process launched from code predating
the lock cannot participate in that guarantee.

The render verifier also pins the paired runtime to
`mrbizarro/ltx-2-mlx` tag `v0.14.19+ltx25.7`, commit
`bf419b6f76e7753993eb7c3df3b15e1451310409`, with package version
`0.14.19+ltx25.7`. Generate keeps the two pack paths separate via `--model`
and `--gemma`.

## Integrity and recovery design

- The exact release-manifest bytes are pinned locally before their JSON is
  trusted:

  - `q4_25`: 9,430 bytes, SHA-256
    `0013bfab7524d61d7c0e3d95733e6453cbab9dde944eb5a9100db8bf531ab424`
  - `gemma4_25`: 5,387 bytes, SHA-256
    `b995e479a8e3d2c08af35572fccf9412fad8b3c4f6ef3f2934d733f6596472d5`

- Every downloaded shard must match both its declared byte count and SHA-256.
  Every reassembled file must independently match its full-file byte count and
  SHA-256.
- Downloads and assemblies live under
  `work/local-video/models/.phosphene-release-staging`. A loader-visible file is
  created only by an atomic rename after its complete hash passes. An existing
  destination—even an invalid one—stays in place until that moment.
- The installer and external-shard importer share a per-model advisory lock
  across shard download, assembly, and promotion. An import fails closed if
  that model file is currently owned by another process.
- Resume metadata is bound to the pack, pinned manifest, asset name, byte count,
  and hash. Complete staged shard boundaries are rehashed after a restart; a
  torn assembly tail is discarded.
- Optional `--jobs` parallelism is across independent files only. A manifest
  filename is submitted once, each file's shards remain sequential, and the
  same full-file verification and atomic publish boundary applies.
- Optional fixed-size range mode rejects ignored or shifted ranges, changed
  validators, malformed lengths, and short/oversized bodies before they can be
  appended to a shard partial.
- Optional `--range-workers` parallelism is inside one shard. Workers can only
  write private mode-0600 chunk files; they never open the shard partial. A
  failed, missing, or validator-mismatched chunk cancels the batch before its
  chunks are appended, and successful chunks are appended in offset order.
- The transport is macOS system-trust `curl`, with user curl configuration
  disabled, redirects restricted to HTTPS, and no authorization header. This
  remains compatible with managed enterprise trust roots without weakening
  TLS verification.
- A successful pack receives `.phosphene-install-receipt.json` containing its
  immutable release identity and verified totals.

### Publisher license-asset exception

The publisher replaced the shared license asset on 2026-09-06 but did not
regenerate the two release manifests. The manifests still declare a 30,938-byte
license with SHA-256 `4e7dd26d...`, while the live release serves the current
34,545-byte license with SHA-256 `a6622b0e...`.

The installer does not relax manifest checking. It applies an exception only
when the old manifest entry—including its sole shard—exactly matches the known
stale tuple. It excludes only that legal-text entry, then downloads the current
release license under a separate embedded size/SHA-256 pin:

```text
bytes   34545
sha256  a6622b0e003d7b6aa6ca47afa1695cc9cbba7efe189e70217c54053dfc1fe654
```

No model, tokenizer, configuration, notice, or weight shard receives this
exception.

## Checks performed here

The live preflight passed for both pinned manifests, both one-byte GitHub CDN
range probes (`HTTP 206`), and the separately pinned 34 KB current license. No
weight shard was downloaded. The offline suite covers manifest tampering,
unsafe paths, no-token requests, normal resume, exact fixed-size ranges,
arbitrary-offset range resume, wrong ranges, changed validators,
malformed/short chunks, out-of-order parallel completion, strict ordered
append, failed/missing parallel chunks, parallel space accounting and CLI
bounds, corrupt shards, torn assemblies, full-file mismatch, and atomic
replacement:

```bash
cd outputs/local-video
python3 -m unittest -v test_install_phosphene_ltx25_release.py
```

The optional live integration check downloads only a 101-byte config asset,
truncates its private temporary copy, and proves the curl resume path:

```bash
cd outputs/local-video
PHOSPHENE_LIVE_TEST=1 python3 -m unittest -v \
  test_install_phosphene_ltx25_release.InstallerTests.test_live_system_curl_fresh_and_resumed_tiny_asset
```
