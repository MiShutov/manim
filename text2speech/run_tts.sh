#!/usr/bin/env bash
set -Eeuo pipefail

# Russian text-to-speech with the F5-TTS v4_winter finetune.
#
# Code lives in text2speech/, media lives in audio/text2speech/.
# Each voice is a folder in ref_voices/ with ref.wav and ref.txt.

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${AUDIO_DIR:-$(cd "$CODE_DIR/.." && pwd)/audio/text2speech}"
VOICES_DIR="$AUDIO_DIR/ref_voices"

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
  CONDA_ENV       conda environment with f5-tts, default f5tts
  DEVICE          cuda or cpu, default cuda
  AUTO_STRESS     1 to add "+" stress marks, 0 to keep the text as is
  OUTPUT_FILE     output file name inside output/<text>/, default <voice>.wav
  NFE_STEP        diffusion steps, default 32
  CFG_STRENGTH    guidance strength, default 2.0
  SPEED           speech rate, default 1.0
  SWAY, FIX_DURATION, TARGET_RMS, REMOVE_SILENCE, SEED
                  optional extras, passed to the driver when set
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

# Conda environment that has f5-tts installed.
CONDA_ENV="${CONDA_ENV:-f5tts}"
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

# Use the interpreters from the activated environment unless overridden.
PYTHON="${PYTHON:-$(command -v python)}"

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python not found in '$CONDA_ENV'." >&2
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-$(cd "$CODE_DIR/.." && pwd)/models/F5TTS_v1_Base_v4_winter}"
MODEL_CFG="$MODEL_DIR/model_cfg.yaml"
CHECKPOINT="$MODEL_DIR/model_212000.safetensors"
VOCAB="$MODEL_DIR/vocab.txt"

# Default voice: the first available folder, so a single voice needs no flags.
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

VOCODER_DIR="${VOCODER_DIR:-$(cd "$CODE_DIR/.." && pwd)/models/vocos-mel-24khz}"

OUTPUT_DIR="$AUDIO_DIR/output/$(basename "${INPUT_TEXT%.*}")"
OUTPUT_FILE="${OUTPUT_FILE:-${VOICE}.wav}"
GEN_TEXT_FILE="$OUTPUT_DIR/${VOICE}.stressed.txt"

DEVICE="${DEVICE:-cuda}"
AUTO_STRESS="${AUTO_STRESS:-1}"
NFE_STEP="${NFE_STEP:-32}"
CFG_STRENGTH="${CFG_STRENGTH:-2.0}"
SPEED="${SPEED:-1.0}"

# Optional knobs: only passed to the driver when set.
EXTRA_ARGS=()
[[ -n "${SWAY:-}" ]] && EXTRA_ARGS+=(--sway "$SWAY")
[[ -n "${FIX_DURATION:-}" ]] && EXTRA_ARGS+=(--fix-duration "$FIX_DURATION")
[[ -n "${TARGET_RMS:-}" ]] && EXTRA_ARGS+=(--target-rms "$TARGET_RMS")
[[ -n "${REMOVE_SILENCE:-}" ]] && EXTRA_ARGS+=(--remove-silence)
[[ -n "${SEED:-}" ]] && EXTRA_ARGS+=(--seed "$SEED")

if [[ ! -d "$VOICE_DIR" ]]; then
    echo "ERROR: voice folder not found: $VOICE_DIR" >&2
    list_voices >&2
    exit 1
fi

for file in "$PYTHON" "$MODEL_CFG" "$CHECKPOINT" "$VOCAB" \
            "$REF_AUDIO" "$REF_TEXT_FILE" "$INPUT_TEXT"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: file not found: $file" >&2
        exit 1
    fi
done

if [[ ! -s "$REF_TEXT_FILE" ]]; then
    echo "ERROR: reference transcript is empty: $REF_TEXT_FILE" >&2
    exit 1
fi

if [[ ! -f "$VOCODER_DIR/pytorch_model.bin" ]]; then
    echo "ERROR: local vocoder not found: $VOCODER_DIR" >&2
    echo "Download it once into models/vocos-mel-24khz (see README)." >&2
    exit 1
fi

REF_LENGTH="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REF_AUDIO")"
if awk "BEGIN{exit !($REF_LENGTH > 12)}"; then
    echo "WARNING: reference is ${REF_LENGTH}s, the model was trained on 8-12s" >&2
    echo "         prompts; generation may be slower and less stable." >&2
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$AUTO_STRESS" == "1" ]]; then
    # Adds "+" before stressed vowels, keeping manual marks untouched.
    "$PYTHON" "$CODE_DIR/add_stress.py" "$INPUT_TEXT" "$GEN_TEXT_FILE"
else
    "$PYTHON" - "$INPUT_TEXT" "$GEN_TEXT_FILE" <<'PY'
import re
import sys

source, target = sys.argv[1:3]
text = re.sub(r"\s+", " ", open(source, encoding="utf-8").read()).strip()
if not text:
    sys.exit("ERROR: input text is empty")
open(target, "w", encoding="utf-8").write(text + "\n")
print(f"Text ({len(text)} chars, stress marks kept as is): {text[:80]}...")
PY
fi

echo "Voice:    $VOICE ($REF_LENGTH s)"
echo "Model:    $CHECKPOINT"
echo "Output:   $OUTPUT_DIR/$OUTPUT_FILE"

"$PYTHON" "$CODE_DIR/synthesize.py" \
    --ref-audio "$REF_AUDIO" \
    --ref-text "$REF_TEXT_FILE" \
    --gen-file "$GEN_TEXT_FILE" \
    --out "$OUTPUT_DIR/$OUTPUT_FILE" \
    --model-cfg "$MODEL_CFG" \
    --ckpt "$CHECKPOINT" \
    --vocab "$VOCAB" \
    --vocoder-dir "$VOCODER_DIR" \
    --device "$DEVICE" \
    --nfe-step "$NFE_STEP" \
    --cfg-strength "$CFG_STRENGTH" \
    --speed "$SPEED" \
    "${EXTRA_ARGS[@]}"

echo
echo "Done: $OUTPUT_DIR/$OUTPUT_FILE"
