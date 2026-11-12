#!/usr/bin/env bash
set -Eeuo pipefail

# Synthesize one text file with a chosen voice.
# Usage: run_single_text.sh TEXT_FILE VOICE
# Output: audio/text2speech/output/<text_file_stem>/<voice>.wav

if (($# != 2)); then
    echo "Usage: $(basename "$0") TEXT_FILE VOICE" >&2
    exit 2
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

text="$(realpath "$1")"
voice="$2"

exec "$CODE_DIR/run_tts.sh" --voice "$voice" --text "$text"
