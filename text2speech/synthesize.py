#!/usr/bin/env python3
"""Run VoxCPM2 generation with a local checkpoint and a cloned voice.

VoxCPM2 is a diffusion-autoregressive TTS that supports 30 languages
(including Russian), clones a voice from a short reference clip, and outputs
48 kHz audio. It is trained on plain text, so any ``+`` stress marks left over
from older F5-style scripts are removed before synthesis.

VoxCPM2 can continue a prompt, which gives the most faithful clone when the
transcript of the reference clip is known:

* ``prompt_wav_path`` + ``prompt_text`` continue the reference (hi-fi cloning);
* ``reference_wav_path`` additionally provides the timbre.

Very long inputs make the KV cache grow until quality degrades or the GPU runs
out of memory, so the text is split into sentence-aligned segments and the
waveforms are joined with a short fade instead of a crossfade.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM

from chunking import join_chunks, normalize_text, plan_chunks


def seed_everything(seed: int) -> None:
    """Seed every RNG the sampler may touch.

    ``generate()`` has no seed argument, so reproducibility is set once for the
    whole run; chunk boundaries stay deterministic either way.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", type=Path, default=None)
    parser.add_argument("--gen-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-chars", type=int, default=300)
    parser.add_argument("--join-fade", type=float, default=0.02)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument(
        "--control",
        default=None,
        help=(
            "style instruction such as 'sad, slow' or 'cheerful and energetic'; "
            "prepended to every chunk in parentheses"
        ),
    )
    parser.add_argument(
        "--no-retry",
        action="store_false",
        dest="retry_badcase",
        help="disable automatic retries when the audio length looks wrong",
    )
    args = parser.parse_args()

    if args.chunk_chars <= 0:
        sys.exit("ERROR: --chunk-chars must be positive")
    if args.join_fade < 0:
        sys.exit("ERROR: --join-fade must not be negative")
    if not args.ref_audio.is_file():
        sys.exit(f"ERROR: reference audio not found: {args.ref_audio}")

    gen_text = normalize_text(args.gen_file.read_text(encoding="utf-8"))
    if not gen_text:
        sys.exit(f"ERROR: input text is empty: {args.gen_file}")

    prompt_text = None
    if args.ref_text is not None and args.ref_text.is_file():
        prompt_text = normalize_text(args.ref_text.read_text(encoding="utf-8"))
        if not prompt_text:
            sys.exit(f"ERROR: reference transcript is empty: {args.ref_text}")

    control = (args.control or "").strip().strip("()").strip()
    if control and prompt_text is not None:
        # VoxCPM2 steers style either by continuing a transcript or by an
        # instruction, not both: the instruction replaces prompt continuation,
        # so the timbre comes from the reference clip alone.
        print(
            "Note: --control disables prompt continuation "
            "(timbre-only cloning instead of hi-fi cloning)."
        )
        prompt_text = None

    model = VoxCPM.from_pretrained(
        str(args.model_dir),
        load_denoiser=args.denoise,
        device=args.device,
    )
    sample_rate = model.tts_model.sample_rate

    if args.seed is not None:
        seed_everything(args.seed)

    # Plan the long-form generation: VoxCPM2 degrades when one call has to
    # produce minutes of speech, so it is fed sentence-aligned segments.
    chunks = plan_chunks(gen_text, args.chunk_chars)
    print(
        f"Plan:     {len(chunks)} chunks, max {args.chunk_chars} chars each, "
        f"{sample_rate} Hz output"
    )
    if control:
        print(f"Control:  ({control})")
    for index, chunk in enumerate(chunks, 1):
        print(f"  chunk {index:>2}/{len(chunks)}  {len(chunk):>3} chars  {chunk}")

    waves = []
    for index, chunk in enumerate(chunks, 1):
        print(f"Generating chunk {index}/{len(chunks)}...", flush=True)
        wav = model.generate(
            text=f"({control}){chunk}" if control else chunk,
            reference_wav_path=str(args.ref_audio),
            prompt_wav_path=str(args.ref_audio) if prompt_text else None,
            prompt_text=prompt_text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.timesteps,
            normalize=args.normalize,
            denoise=args.denoise,
            retry_badcase=args.retry_badcase,
        )
        if wav is None or len(wav) == 0:
            print(f"WARNING: chunk {index} produced no audio, skipping")
            continue
        waves.append(wav)

    if not waves:
        sys.exit("ERROR: nothing generated")

    wav = join_chunks(waves, args.join_fade, sample_rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
