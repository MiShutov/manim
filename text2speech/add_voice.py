#!/usr/bin/env python3
"""Create a voice folder from any audio or video file.

VoxCPM2 clones the voice of a short reference clip (5-12 s is the sweet spot)
together with its exact transcript, which enables hi-fi cloning. This helper
turns an arbitrary file into the layout ``run_tts.sh`` expects:

    ref_voices/<name>/ref.wav   24 kHz mono PCM
    ref_voices/<name>/ref.txt   exact transcript of ref.wav

Long recordings are cut at a word boundary using the timings from the local
Qwen3 ASR + forced aligner, and the cut clip is transcribed again so the text
always matches the audio. Pass ``--max-seconds 0`` to keep the whole
recording as the reference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from env_tools import conda_python

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASR = REPO_ROOT / "speech2text" / "transcribe.py"
DEFAULT_VOICES = REPO_ROOT / "audio" / "text2speech" / "ref_voices"

SAMPLE_RATE = 24000
MAX_SECONDS = 12.0


def run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        sys.exit(f"ERROR: command failed ({error.returncode}): {' '.join(command)}")


def duration(path: Path) -> float:
    output = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(output)


def extract_audio(source: Path, target: Path) -> None:
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(target),
    ])


def cut_audio(source: Path, target: Path, start: float, end: float) -> None:
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(target),
    ])


def transcribe(audio: Path, asr_script: Path, asr_python: str, workdir: Path) -> dict:
    result = workdir / f"{audio.stem}.words.json"
    run([asr_python, str(asr_script), str(audio), "-o", str(result)])
    return json.loads(result.read_text(encoding="utf-8"))


def choose_boundary(words: list[dict], target_end: float) -> tuple[float, float]:
    """Return (start, end) around the first full words of the recording."""
    usable = [word for word in words if word["end"] <= target_end]
    if not usable:
        usable = words[:1]
    start = max(0.0, usable[0]["start"] - 0.15)
    end = usable[-1]["end"] + 0.25
    return start, max(end, start + 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="audio or video file")
    parser.add_argument(
        "--name",
        default=None,
        help="voice folder name, defaults to the source file stem",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=DEFAULT_VOICES,
        help=f"voice root, default {DEFAULT_VOICES}",
    )
    parser.add_argument(
        "--asr-script",
        type=Path,
        default=DEFAULT_ASR,
        help=f"transcription script, default {DEFAULT_ASR}",
    )
    parser.add_argument(
        "--asr-python",
        default=None,
        help="interpreter with transformers and torch, default is the voxcpm env",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_SECONDS,
        help=(
            "cut longer recordings to about this length; "
            "0 keeps the whole recording as the reference"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing voice folder",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        sys.exit(f"ERROR: no such file: {source}")
    if source.stat().st_size == 0:
        sys.exit(f"ERROR: file is empty (0 bytes): {source}")

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"ERROR: {tool} is required")

    name = args.name or source.stem
    asr_python = args.asr_python or conda_python()
    voice_dir = args.voices_dir / name
    if voice_dir.exists() and not args.force:
        sys.exit(f"ERROR: {voice_dir} already exists, use --force to overwrite")

    with tempfile.TemporaryDirectory(prefix="voice_") as tmp:
        workdir = Path(tmp)
        full = workdir / "full.wav"
        print(f"Extracting audio from {source.name}...")
        extract_audio(source, full)

        length = duration(full)
        print(f"Source duration: {length:.2f} s")

        clip = full
        if args.max_seconds > 0 and length > args.max_seconds:
            print("Transcribing to find a word boundary...")
            data = transcribe(full, args.asr_script, asr_python, workdir)
            if not data.get("words"):
                sys.exit("ERROR: no speech found in the source file")
            start, end = choose_boundary(data["words"], args.max_seconds - 1.5)
            print(f"Cutting reference clip: {start:.2f}-{end:.2f} s")
            clip = workdir / "clip.wav"
            cut_audio(full, clip, start, end)
        else:
            print("Keeping the full recording as the reference")

        print("Transcribing the reference clip...")
        data = transcribe(clip, args.asr_script, asr_python, workdir)
        transcript = data.get("transcript", "").strip()
        if not transcript:
            sys.exit("ERROR: no speech recognized in the reference clip")

        voice_dir.mkdir(parents=True, exist_ok=True)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(clip),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(voice_dir / "ref.wav"),
        ])
        (voice_dir / "ref.txt").write_text(transcript + "\n", encoding="utf-8")

    clip_length = duration(voice_dir / "ref.wav")
    print(f"\nVoice ready: {voice_dir}")
    print(f"Clip: {clip_length:.2f} s")
    print(f"Text: {transcript}")


if __name__ == "__main__":
    main()
