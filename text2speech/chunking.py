"""Sentence-aligned chunk planning for long narration.

VoxCPM2 is autoregressive, so a minute-long input grows the KV cache until it
either runs out of memory or drifts (the upstream docs list speed-up, buzzing
and never-ending generations as the usual symptoms). The documented workaround
is to split the text into shorter segments and concatenate the waveforms; this
module decides where the splits go so that every boundary lands on punctuation.

Everything here is pure Python (plus numpy for the join), independent of the
model runtime.
"""

from __future__ import annotations

import re

import numpy as np

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…;:])\s+")
CLAUSE_BOUNDARY = re.compile(r"(?<=[,—–-])\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace so the chunk sizes match what the model receives."""
    return re.sub(r"\s+", " ", text).strip()


def is_initialism(piece: str) -> bool:
    """True when a piece ends with a one-letter abbreviation such as ``Б.``"""
    words = piece.split()
    if not words:
        return False
    last = words[-1]
    return len(last) == 2 and last[0].isalpha() and last.endswith(".")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries without breaking ``Б. П. Е.`` apart."""
    sentences: list[str] = []
    for sentence in SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentences and is_initialism(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {sentence}"
        else:
            sentences.append(sentence)
    return sentences


def word_atoms(text: str) -> list[str]:
    """Group consecutive one-letter abbreviations into single atoms."""
    atoms: list[str] = []
    for word in text.split():
        if atoms and is_initialism(atoms[-1]) and is_initialism(word):
            atoms[-1] = f"{atoms[-1]} {word}"
        else:
            atoms.append(word)
    return atoms


def split_oversized(unit: str, max_bytes: int) -> list[str]:
    """Split one over-long sentence on word boundaries."""
    parts: list[str] = []
    current = ""
    for atom in word_atoms(unit):
        candidate = f"{current} {atom}".strip()
        if current and len(candidate.encode("utf-8")) > max_bytes:
            parts.append(current)
            current = atom
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def text_units(text: str, max_bytes: int, sentence_overshoot: float = 1.6) -> list[str]:
    """Break text into sentence-sized units that fit ``max_bytes``.

    Sentences are kept whole whenever possible: a sentence may exceed the
    budget by ``sentence_overshoot`` times, because cutting a sentence in the
    middle makes autoregressive models add a filler or drift at the end of the
    segment. Only genuinely long sentences are split, first at commas and
    dashes and then at word boundaries.
    """
    sentence_limit = int(max_bytes * sentence_overshoot)
    units: list[str] = []
    for sentence in split_sentences(text):
        if not sentence:
            continue
        if len(sentence.encode("utf-8")) <= sentence_limit:
            units.append(sentence)
            continue
        for clause in CLAUSE_BOUNDARY.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause.encode("utf-8")) <= max_bytes:
                units.append(clause)
            else:
                units.extend(split_oversized(clause, max_bytes))
    return units


def pack_units(units: list[str], max_bytes: int) -> list[str]:
    """Greedily pack consecutive units into chunks within ``max_bytes``."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate.encode("utf-8")) > max_bytes:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_in_two(text: str) -> list[str]:
    """Split text into two chunks near the middle, preferring punctuation."""
    atoms = word_atoms(text)
    if len(atoms) < 2:
        return [text]

    total = len(text.encode("utf-8"))
    size = 0
    best_score = float("inf")
    best_cut = len(atoms) // 2
    for index, atom in enumerate(atoms[:-1], start=1):
        size += len(atom.encode("utf-8"))
        if is_initialism(atom):
            weight = 0.3
        elif atom.endswith(("!", "?", "…", ".")):
            weight = 0.0
        elif atom.endswith((",", ";", ":", "—")):
            weight = 0.15
        else:
            weight = 0.5
        score = weight + abs(size - total / 2) / total
        if score < best_score:
            best_score = score
            best_cut = index
        size += 1  # the space before the next atom
    return [" ".join(atoms[:best_cut]), " ".join(atoms[best_cut:])]


def rebalance(chunks: list[str], max_bytes: int) -> list[str]:
    """Merge undersized fragments back into their neighbours.

    Splitting a long sentence at a comma can leave a two-word tail; on its own
    it costs a full model call and sounds like an isolated phrase. Such a
    fragment is merged with the previous chunk and the pair is split again near
    the middle, so no chunk falls below a third of the budget.
    """
    min_bytes = max(max_bytes // 3, 40)
    chunks = list(chunks)
    index = 1
    while index < len(chunks):
        if len(chunks[index].encode("utf-8")) < min_bytes:
            merged = f"{chunks[index - 1]} {chunks[index]}"
            pieces = (
                split_in_two(merged)
                if len(merged.encode("utf-8")) > max_bytes
                else [merged]
            )
            chunks[index - 1 : index + 1] = pieces
            index = max(index - 1, 1)
        else:
            index += 1
    return chunks


def bytes_for_chars(text: str, max_chars: int) -> int:
    """Translate a character budget into the byte budget used internally."""
    if not text:
        return 0
    return max(int(max_chars * len(text.encode("utf-8")) / len(text)), 40)


def plan_chunks(text: str, max_chars: int) -> list[str]:
    """Plan sentence-aligned chunks of roughly ``max_chars`` characters each."""
    max_bytes = bytes_for_chars(text, max_chars)
    return rebalance(pack_units(text_units(text, max_bytes), max_bytes), max_bytes)


def join_chunks(waves: list[np.ndarray], fade: float, sample_rate: int) -> np.ndarray:
    """Concatenate chunks with a short fade instead of an overlap.

    Boundaries fall on punctuation, so the chunks already end with the natural
    pause of the sentence. A 10-20 ms fade only removes the click of a hard
    cut.
    """
    if not waves:
        raise ValueError("no audio to join")

    total = np.asarray(waves[0], dtype=np.float32).copy()
    for wave in waves[1:]:
        wave = np.asarray(wave, dtype=np.float32).copy()
        samples = min(int(fade * sample_rate), len(total), len(wave))
        if samples > 0:
            total[-samples:] *= np.linspace(1.0, 0.0, samples, dtype=np.float32)
            wave[:samples] *= np.linspace(0.0, 1.0, samples, dtype=np.float32)
        total = np.concatenate([total, wave])
    return total
