#!/usr/bin/env python3
"""Automatic Russian stress marking in the ``+`` format used by the model.

The v4_winter finetune expects stress as ``+`` right before the stressed
vowel (``с+интеза``, ``нейр+онной``). Silero stress predicts exactly that
format, so the script keeps the text as close to the source as possible:

* words that already contain a manual ``+`` are left untouched;
* one-syllable words are not marked, the stress is unambiguous there;
* ``ё`` is never marked either, it is always stressed.

Homographs (``за́мок`` / ``замо́к``) are resolved from context, but the model
is not perfect, so review the result with ``--dry-run`` before synthesizing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
WORD = re.compile(r"[А-Яа-яЁё+]+")
PLACEHOLDER = "QVX{}QVX"


def protect_manual_marks(text: str) -> tuple[str, dict[str, str]]:
    """Replace manually marked words with placeholders."""
    saved: dict[str, str] = {}
    for index, word in enumerate(WORD.findall(text)):
        if "+" not in word:
            continue
        placeholder = PLACEHOLDER.format(index)
        saved[placeholder] = word
        text = text.replace(word, placeholder, 1)
    return text, saved


def restore_manual_marks(text: str, saved: dict[str, str]) -> str:
    for placeholder, word in saved.items():
        text = text.replace(placeholder, word)
    return text


def tidy_word(word: str) -> str:
    """Drop redundant marks: one-syllable words and ``ё``."""
    if word.count("+") == 0:
        return word
    word = word.replace("+ё", "ё").replace("+Ё", "Ё")
    if sum(1 for char in word if char in VOWELS) <= 1:
        word = word.replace("+", "")
    return word


def tidy(text: str) -> str:
    return WORD.sub(lambda match: tidy_word(match.group(0)), text)


def validate(text: str) -> list[str]:
    problems = []
    for index, char in enumerate(text):
        if char != "+":
            continue
        following = text[index + 1] if index + 1 < len(text) else ""
        if following not in VOWELS:
            problems.append(f"'+' before {following!r} at position {index}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="input text file (UTF-8)")
    parser.add_argument("target", type=Path, help="accented text file to write")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the result instead of writing the target file",
    )
    args = parser.parse_args()

    from env_tools import ensure_module

    ensure_module("silero_stress")

    text = args.source.read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        sys.exit("ERROR: input text is empty")

    text, saved = protect_manual_marks(text)

    # Loaded lazily: the model costs a second and is only needed here.
    from silero_stress.accentor import load_accentor

    accentor = load_accentor("ru")
    text = restore_manual_marks(accentor(text), saved)
    text = tidy(text)

    for problem in validate(text):
        print(f"WARNING: {problem}", file=sys.stderr)

    print(f"Text ({len(text)} chars, {text.count('+')} stress marks): {text[:80]}...")
    if saved:
        print("Kept manual stress marks: " + ", ".join(saved.values()))

    if args.dry_run:
        print(text)
        return

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
