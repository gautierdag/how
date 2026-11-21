#!/bin/bash
set -euo pipefail

# Install uv if not already available
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    fi
fi

# Install tmux if not already available (used by experiments.sh)
if ! command -v tmux >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y tmux
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y tmux
        else
            echo "Warning: tmux is not installed and sudo is unavailable. Please install tmux manually."
        fi
    else
        echo "Warning: tmux is not installed. Please install tmux using your system package manager."
    fi
fi

# Install/sync Python dependencies (flash-attn is built without build isolation after base deps)
uv sync --no-install-package flash-attn
uv sync
