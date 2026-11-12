
#!/usr/bin/env python3

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
)

ASR_MODEL = "/home/msst/repo/manim/models/Qwen3-ASR-1.7B-hf"
ALIGN_MODEL = "/home/msst/repo/manim/models/Qwen3-ForcedAligner-0.6B-hf"

# Speech models consume 16 kHz mono audio, so convert to that format.
MP3_SAMPLE_RATE = 16000
MP3_BITRATE = "64k"


def convert_to_mp3(audio_path: Path) -> Path:
    """Convert an input file to a 16 kHz mono mp3 next to the source.

    mp3 input is returned unchanged. Any other format is decoded by ffmpeg.
    """
    if audio_path.suffix.lower() == ".mp3":
        return audio_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit(
            f"ffmpeg is required to convert {audio_path.suffix} to mp3. "
            "Install ffmpeg or pass an mp3 file."
        )

    mp3_path = audio_path.with_suffix(".mp3")
    print(f"[0/3] Converting to mp3: {mp3_path}")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(audio_path),
        "-vn",
        "-ac", "1",
        "-ar", str(MP3_SAMPLE_RATE),
        "-codec:a", "libmp3lame",
        "-b:a", MP3_BITRATE,
        str(mp3_path),
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"ffmpeg failed (exit code {exc.returncode}): {audio_path}")

    if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
        sys.exit(f"ffmpeg produced no output: {mp3_path}")

    return mp3_path


def main():
    parser = argparse.ArgumentParser(
        description="Russian speech transcription with word timestamps"
    )
    parser.add_argument(
        "audio",
        help="Path to audio file (non-mp3 input is converted with ffmpeg)"
    )
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional vocabulary/context for better recognition"
    )
    args = parser.parse_args()

    source_path = Path(args.audio).expanduser().resolve()

    if not source_path.is_file():
        sys.exit(f"Audio file not found: {source_path}")

    if not torch.cuda.is_available():
        sys.exit("CUDA is unavailable. Check your NVIDIA driver and PyTorch.")

    output_path = (
        Path(args.output)
        if args.output
        else source_path.with_suffix(".words.json")
    )

    audio_path = convert_to_mp3(source_path)

    print(f"Audio: {source_path}")
    if audio_path != source_path:
        print(f"mp3: {audio_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16

    # 1. Speech recognition
    print("\n[1/3] Loading ASR model...")

    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL)

    asr_model = AutoModelForMultimodalLM.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map="auto",
    ).eval()

    print("[1/3] Recognizing Russian speech...")

    inputs = asr_processor.apply_transcription_request(
        audio=str(audio_path),
        language="Russian",
        prompt=args.prompt,
    ).to(asr_model.device, asr_model.dtype)

    with torch.inference_mode():
        output_ids = asr_model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    parsed = asr_processor.decode(
        generated_ids,
        return_format="parsed",
    )[0]

    transcript = parsed["transcription"]

    if not transcript.strip():
        sys.exit("No speech recognized.")

    print("\nRecognized text:")
    print(transcript)

    # Free ASR model memory before loading aligner
    del asr_model, asr_processor, inputs, output_ids, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Word-level forced alignment
    print("\n[2/3] Loading forced alignment model...")

    align_processor = AutoProcessor.from_pretrained(ALIGN_MODEL)

    align_model = AutoModelForTokenClassification.from_pretrained(
        ALIGN_MODEL,
        dtype=dtype,
        device_map="auto",
    ).eval()

    print("[2/3] Aligning every word with audio...")

    align_inputs, word_lists = (
        align_processor.prepare_forced_aligner_inputs(
            audio=str(audio_path),
            transcript=transcript,
            language="Russian",
        )
    )

    align_inputs = align_inputs.to(
        align_model.device,
        align_model.dtype,
    )

    with torch.inference_mode():
        align_outputs = align_model(**align_inputs)

    timestamps = align_processor.decode_forced_alignment(
        logits=align_outputs.logits,
        input_ids=align_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=align_model.config.timestamp_token_id,
    )[0]

    # 3. Export JSON
    print("\n[3/3] Saving JSON...")

    words = [
        {
            "word": item["text"],
            "start": round(float(item["start_time"]), 3),
            "end": round(float(item["end_time"]), 3),
        }
        for item in timestamps
    ]

    result = {
        "audio": audio_path.name,
        "language": "ru",
        "model": ASR_MODEL,
        "aligner": ALIGN_MODEL,
        "transcript": transcript,
        "word_count": len(words),
        "words": words,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(words)} words aligned.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
