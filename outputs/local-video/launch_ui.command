#!/bin/zsh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WORKSPACE_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$WORKSPACE_DIR/work/local-video/ltx-2-mlx/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "LTX-2 MLX environment not found: $PYTHON_BIN"
  echo "Run the installation/verification steps first."
  read -r "?Press Return to close…"
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/ltx_ui.py" --open
