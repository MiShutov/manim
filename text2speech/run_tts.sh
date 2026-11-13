#!/usr/bin/env bash
set -Eeuo pipefail

# Russian text-to-speech with VoxCPM2 (OpenBMB).
#
# Code lives in text2speech/, media lives in audio/text2speech/.
# Each voice is a folder with ref.wav and ref.txt under ref_voices/.

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE_DIR/.." && pwd)"
AUDIO_DIR="${AUDIO_DIR:-$REPO_ROOT/audio/text2speech}"
VOICES_DIR="${VOICES_DIR:-$AUDIO_DIR/ref_voices}"

usage() {
    cat <<'EOF'
Usage: run_tts.sh [options]

Options:
  -v, --voice NAME   voice folder under ref_voices/ (default: winter)
  -t, --text FILE    input text file (default: input_text.txt)
  -l, --list         list available voices and exit
  -h, --help         show this help

Environment:
  AUDIO_DIR       media root, default <repo>/audio/text2speech
  VOICES_DIR      voice library, default <AUDIO_DIR>/ref_voices
  CONDA_ENV       conda environment with voxcpm, default voxcpm
  DEVICE          auto, cuda, cuda:N, mps or cpu; default auto
  OUTPUT_FILE     output file name inside output/<text>/, default <voice>.wav
  TIMESTEPS       diffusion steps, default 10
  CFG_VALUE       guidance strength, default 2.0
  CHUNK_CHARS     characters per model call, default 300
  JOIN_FADE       fade between chunks in seconds, default 0.02
  STRESS          strip (default), keep or accent; see README, "Ударения"
  CONTROL         style instruction, e.g. "sad, slow" (see README, "Эмоции")
  KEEP_STRESS     1 is a legacy alias for STRESS=keep
  NORMALIZE       1 to expand numbers and dates before synthesis
  DENOISE         1 to denoise the reference audio before cloning
  SEED            base seed for reproducible generation
EOF
}

list_voices() {
    local dir name length text
    if [[ ! -d "$VOICES_DIR" ]]; then
        echo "No voices directory: $VOICES_DIR" >&2
        return 1
    fi
    echo "Voices in $VOICES_DIR:"
    for dir in "$VOICES_DIR"/*/; do
        [[ -f "$dir/ref.wav" ]] || continue
        name="$(basename "$dir")"
        length="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$dir/ref.wav" 2>/dev/null || echo "?")"
        text="$(head -n1 "$dir/ref.txt" 2>/dev/null || true)"
        printf "  %-20s %6.2f s  %s\n" "$name" "${length:-0}" "$text"
    done
}

VOICE="${VOICE:-}"
TEXT_FILE="${TEXT_FILE:-}"

while (($#)); do
    case "$1" in
        -v|--voice)
            VOICE="${2:?--voice requires a name}"
            shift 2
            ;;
        -t|--text)
            TEXT_FILE="${2:?--text requires a path}"
            shift 2
            ;;
        -l|--list)
            list_voices
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Conda environment that has voxcpm installed.
CONDA_ENV="${CONDA_ENV:-voxcpm}"
CONDA_SH="${CONDA_SH:-$HOME/Utils/miniconda3/etc/profile.d/conda.sh}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if [[ ! -f "$CONDA_SH" ]]; then
        echo "ERROR: conda.sh not found: $CONDA_SH" >&2
        echo "Set CONDA_SH or activate the environment manually." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    echo "Conda environment already active: $CONDA_ENV"
fi

PYTHON="${PYTHON:-$(command -v python)}"
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python not found in '$CONDA_ENV'." >&2
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/VoxCPM2}"

# Default voice: winter if present, otherwise the only available folder.
if [[ -z "$VOICE" ]]; then
    if [[ -d "$VOICES_DIR/winter" ]]; then
        VOICE="winter"
    else
        mapfile -t available < <(find "$VOICES_DIR" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null)
        if [[ "${#available[@]}" -eq 1 ]]; then
            VOICE="$(basename "${available[0]}")"
        else
            echo "ERROR: --voice is required, available voices:" >&2
            list_voices >&2
            exit 1
        fi
    fi
fi

VOICE_DIR="$VOICES_DIR/$VOICE"
REF_AUDIO="$VOICE_DIR/ref.wav"
REF_TEXT_FILE="$VOICE_DIR/ref.txt"
INPUT_TEXT="${TEXT_FILE:-$AUDIO_DIR/input_text.txt}"
INPUT_TEXT="$(realpath "$INPUT_TEXT")"

OUTPUT_DIR="$AUDIO_DIR/output/$(basename "${INPUT_TEXT%.*}")"
OUTPUT_FILE="${OUTPUT_FILE:-${VOICE}.wav}"
GEN_TEXT_FILE="$OUTPUT_DIR/${VOICE}.prepared.txt"

DEVICE="${DEVICE:-auto}"
CHUNK_CHARS="${CHUNK_CHARS:-300}"
JOIN_FADE="${JOIN_FADE:-0.02}"
TIMESTEPS="${TIMESTEPS:-10}"
CFG_VALUE="${CFG_VALUE:-2.0}"

# Stress handling: "strip" removes legacy "+" marks, "keep" passes them to the
# model (it will say "plus"), "accent" turns them into combining acute accents.
STRESS_MODE="${STRESS:-strip}"
if [[ "${KEEP_STRESS:-}" == "1" ]]; then
    STRESS_MODE="keep"
fi
case "$STRESS_MODE" in
    strip|keep|accent) ;;
    *)
        echo "ERROR: STRESS must be strip, keep or accent (got '$STRESS_MODE')" >&2
        exit 2
        ;;
esac

EXTRA_ARGS=()
[[ -n "${SEED:-}" ]] && EXTRA_ARGS+=(--seed "$SEED")
[[ -n "${NORMALIZE:-}" ]] && EXTRA_ARGS+=(--normalize)
[[ -n "${DENOISE:-}" ]] && EXTRA_ARGS+=(--denoise)
[[ -n "${CONTROL:-}" ]] && EXTRA_ARGS+=(--control "$CONTROL")

if [[ ! -d "$VOICE_DIR" ]]; then
    echo "ERROR: voice folder not found: $VOICE_DIR" >&2
    list_voices >&2
    exit 1
fi

for file in "$PYTHON" "$REF_AUDIO" "$INPUT_TEXT"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: file not found: $file" >&2
        exit 1
    fi
done

REF_ARGS=()
if [[ -f "$REF_TEXT_FILE" ]]; then
    REF_ARGS+=(--ref-text "$REF_TEXT_FILE")
else
    echo "WARNING: $REF_TEXT_FILE is missing; cloning the timbre only." >&2
    echo "         Add the reference transcript for hi-fi cloning." >&2
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "ERROR: local VoxCPM2 checkpoint not found: $MODEL_DIR" >&2
    echo "Download it once (see README) into models/VoxCPM2." >&2
    exit 1
fi

REF_LENGTH="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REF_AUDIO")"
if awk "BEGIN{exit !($REF_LENGTH < 5)}"; then
    echo "WARNING: reference is ${REF_LENGTH}s, VoxCPM2 clones better from 5-30 s" >&2
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON" - "$INPUT_TEXT" "$GEN_TEXT_FILE" "$STRESS_MODE" <<'PY'
import re
import sys

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"

source, target, mode = sys.argv[1:4]
text = re.sub(r"\s+", " ", open(source, encoding="utf-8").read()).strip()
if not text:
    sys.exit("ERROR: input text is empty")

marked = 0
if mode == "accent":
    # VoxCPM2 has no phoneme input, but the tokenizer keeps combining marks,
    # so an F5-style "+" marker is moved behind the stressed vowel as U+0301.
    text, marked = re.subn(
        r"[+]([" + VOWELS + "])", lambda m: m.group(1) + "\u0301", text
    )
elif mode == "keep":
    pass
else:
    # Plain text model: "+" would be pronounced as the word "plus".
    text = re.sub(r"[+](?=[" + VOWELS + "])", "", text)

open(target, "w", encoding="utf-8").write(text + "\n")
if mode == "accent":
    print(f"Text ({len(text)} chars, {marked} stress marks): {text[:80]}...")
else:
    print(f"Text ({len(text)} chars): {text[:80]}...")
PY

echo "Voice:    $VOICE ($REF_LENGTH s)"
echo "Model:    $MODEL_DIR"
echo "Output:   $OUTPUT_DIR/$OUTPUT_FILE"

"$PYTHON" "$CODE_DIR/synthesize.py" \
    --ref-audio "$REF_AUDIO" \
    "${REF_ARGS[@]}" \
    --gen-file "$GEN_TEXT_FILE" \
    --out "$OUTPUT_DIR/$OUTPUT_FILE" \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --chunk-chars "$CHUNK_CHARS" \
    --join-fade "$JOIN_FADE" \
    --timesteps "$TIMESTEPS" \
    --cfg-value "$CFG_VALUE" \
    "${EXTRA_ARGS[@]}"

echo
echo "Done: $OUTPUT_DIR/$OUTPUT_FILE"
