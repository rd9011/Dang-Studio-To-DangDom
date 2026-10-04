#!/usr/bin/env bash
# Prepare the small, revision-pinned source fixture used by runtime installer
# tests. This downloads source code only; it never downloads model weights or
# creates a Python runtime.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
destination="${1:-${repository_root}/work/local-video/ltx-2-mlx}"

if [[ "${destination}" != /* ]]; then
  destination="${repository_root}/${destination}"
fi

source_url="https://github.com/mrbizarro/ltx-2-mlx.git"
source_revision="bf419b6f76e7753993eb7c3df3b15e1451310409"
upstream_url="https://github.com/dgrauet/ltx-2-mlx.git"
backport_commits=(
  "add0142ece318bb09ad4bf295214724859fb241b"
  "99c9c7f124d3a56b5bfd63ebb8e96cf76cc7cd0a"
)
patch_paths=(
  "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py"
  "packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/retake.py"
)
expected_hashes=(
  "07e7de7e27cb9f003dd82bfb9d1b8e8702be9e241b28e96552a3228a76977a7c"
  "1870e51a1b1de94996d75c63d44d27598bf7ea0d9f2b4cf6e98b2b04adde43d3"
)

verify_fixture() {
  local root="$1"
  local revision
  local index
  local actual

  [[ -d "${root}/.git" ]] || return 1
  revision="$(git -C "${root}" rev-parse HEAD 2>/dev/null)" || return 1
  [[ "${revision}" == "${source_revision}" ]] || return 1

  for index in "${!patch_paths[@]}"; do
    [[ -f "${root}/${patch_paths[$index]}" ]] || return 1
    actual="$(shasum -a 256 "${root}/${patch_paths[$index]}" | awk '{print $1}')"
    [[ "${actual}" == "${expected_hashes[$index]}" ]] || return 1
  done
}

if [[ -e "${destination}" ]]; then
  if verify_fixture "${destination}"; then
    echo "Pinned LTX source fixture is already verified: ${destination}"
    exit 0
  fi
  echo "ERROR: refusing to replace an existing, unverified fixture: ${destination}" >&2
  exit 2
fi

destination_parent="$(dirname "${destination}")"
mkdir -p "${destination_parent}"
staging="$(mktemp -d "${destination_parent}/.ltx-source-fixture.XXXXXX")"

cleanup() {
  if [[ -n "${staging:-}" && -d "${staging}" ]]; then
    rm -rf -- "${staging}"
  fi
}
trap cleanup EXIT

git -C "${staging}" init -q
git -C "${staging}" remote add source "${source_url}"
git -C "${staging}" fetch -q --depth 1 --filter=blob:none source "${source_revision}"
git -C "${staging}" checkout -q --detach "${source_revision}"

for commit in "${backport_commits[@]}"; do
  echo "Applying pinned LTX test backport ${commit}"
  git -C "${staging}" fetch -q --no-tags "${upstream_url}" "${commit}"
  git -C "${staging}" show --format= --binary "${commit}" -- "${patch_paths[@]}" \
    | git -C "${staging}" apply --3way --whitespace=nowarn -
done

# The pinned fork carries the same two backports with a defensive getattr and
# stable formatting/comments. Reproduce that tiny reviewed adaptation so the
# fixture matches the hashes enforced by the source runtime installer.
git -C "${staging}" apply \
  --unidiff-zero \
  "${repository_root}/.github/fixtures/ltx-runtime-fork-adaptation.patch"

# `git apply --3way` writes the two reviewed backports into the temporary
# index. Runtime verification deliberately accepts them only as exact
# working-tree edits on top of the pristine pinned HEAD. Unstage those known
# paths without changing their verified bytes.
git -C "${staging}" reset -q HEAD -- "${patch_paths[@]}"

if ! verify_fixture "${staging}"; then
  echo "ERROR: the prepared LTX source fixture failed revision/hash verification" >&2
  exit 3
fi

mv "${staging}" "${destination}"
staging=""
echo "Prepared verified LTX source fixture: ${destination}"
