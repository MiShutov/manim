#!/usr/bin/env python3
"""Run one F5-TTS generation with a local model and a local vocoder.

Compared with ``f5-tts_infer-cli`` this driver does not preprocess the
reference audio: no silence trimming and no 12-second clipping. Whatever
``ref.wav`` contains is what the model conditions on, so a voice folder may
hold a longer reference than the usual 10 seconds.

References longer than ~12 s are supported but not recommended: the finetune
was trained on short prompts, and the duration heuristic assumes the prompt
text and audio follow the same speech rate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    remove_silence_for_generated_wav,
)
from f5_tts.model.utils import seed_everything


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def finish_sentence(text: str) -> str:
    """Match the punctuation the f5 CLI appends to reference transcripts."""
    if not text.endswith(". ") and not text.endswith("。"):
        text += " " if text.endswith(".") else ". "
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", type=Path, required=True)
    parser.add_argument("--gen-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--vocoder-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--fix-duration", type=float, default=None)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--remove-silence", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ref_text = normalize_text(args.ref_text.read_text(encoding="utf-8"))
    if not ref_text:
        sys.exit(f"ERROR: reference transcript is empty: {args.ref_text}")

    gen_text = normalize_text(args.gen_file.read_text(encoding="utf-8"))
    if not gen_text:
        sys.exit(f"ERROR: input text is empty: {args.gen_file}")

    if args.seed is not None:
        seed_everything(args.seed)

    config = OmegaConf.load(args.model_cfg)
    model_cls = get_class(f"f5_tts.model.{config.model.backbone}")

    vocoder = load_vocoder(
        "vocos", is_local=True, local_path=str(args.vocoder_dir), device=args.device
    )
    model = load_model(
        model_cls,
        config.model.arch,
        str(args.ckpt),
        mel_spec_type="vocos",
        vocab_file=str(args.vocab),
        device=args.device,
    )

    wav, sample_rate, _ = infer_process(
        str(args.ref_audio),
        finish_sentence(ref_text),
        gen_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway,
        speed=args.speed,
        fix_duration=args.fix_duration,
        target_rms=args.target_rms,
        device=args.device,
    )

    if wav is None:
        sys.exit("ERROR: nothing generated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, sample_rate)
    if args.remove_silence:
        remove_silence_for_generated_wav(str(args.out))


if __name__ == "__main__":
    main()
