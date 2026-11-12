#!/usr/bin/env bash
set -Eeuo pipefail

# Add a voice from an audio or video file: ref_voices/<name>/{ref.wav,ref.txt}.
# Usage: add_voice.sh FILE [extra add_voice.py options, e.g. --name NAME]

if (($# < 1)); then
    echo "Usage: $(basename "$0") FILE [--name NAME] [--force]" >&2
    exit 2
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CODE_DIR/add_voice.py" "$@"
