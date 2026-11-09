"""Data for the SentencePiece scene: corpus, learned merges, and encoding.

SentencePiece (Kudo & Richardson, 2018) is the subword library behind many
LLMs (LLaMA, T5, ALBERT). It runs byte-pair encoding directly on the raw text,
replacing every space with the meta-symbol ``\u2581`` so that the whitespace
becomes an ordinary, mergeable symbol at the start of each word.
"""

from __future__ import annotations

from collections import Counter

TITLE = "BPE: how a tokenizer learns subwords (SentencePiece)"

# The space meta-symbol. SentencePiece rewrites spaces to this block character
# before training, so "▁" is just another symbol that can be merged like "l"
# or "o". It is what keeps the leading space attached to a word.
SPACE = "\u2581"

# The training corpus as one running text. The scene shows these lines first,
# then folds them into the counted (word, frequency) table below. Each line
# is short enough to fit inside the corpus panel at the "caption" font size.
RAW_CORPUS_LINES = (
    "low lower newest widest",
    "newest low widest low",
    "newest lower low newest",
    "widest newest low newest",
)
RAW_CORPUS = " ".join(RAW_CORPUS_LINES)

# Word order of the counted table. Frequencies come from RAW_CORPUS, so the
# running text and the table can never disagree.
CORPUS_WORDS = ("low", "lower", "newest", "widest")
_CORPUS_FREQ = Counter(RAW_CORPUS.split())

# Training corpus: word -> (symbols, frequency). Every word starts with the
# space meta-symbol, exactly as SentencePiece sees it.
CORPUS = tuple(
    (SPACE + word, (SPACE, *word), _CORPUS_FREQ[word]) for word in CORPUS_WORDS
)

# Base vocabulary: every character that occurs in the corpus, with the space
# marker first and the remaining symbols in alphabetical order. BPE starts
# from this alphabet, and each merge appends exactly one token to it.
BASE_SYMBOLS = (
    SPACE,
    *sorted(
        {symbol for _, symbols, _ in CORPUS for symbol in symbols} - {SPACE}
    ),
)

# Ordered merge rules learned by BPE: (left, right, merged, count). The order
# is the priority used later during encoding. The "9 -> 9" chain compresses the
# nine "est" suffix occurrences; "7 -> 7 -> 7" glues the space onto "low", and
# so on.
MERGES = (
    ("e", "s", "es", 9),
    ("es", "t", "est", 9),
    ("\u2581", "l", "\u2581l", 7),
    ("\u2581l", "o", "\u2581lo", 7),
    ("\u2581lo", "w", "\u2581low", 7),
    ("\u2581", "n", "\u2581n", 6),
    ("\u2581n", "e", "\u2581ne", 6),
    ("\u2581ne", "w", "\u2581new", 6),
    ("\u2581new", "est", "\u2581newest", 6),
    ("\u2581", "w", "\u2581w", 3),
    ("\u2581w", "i", "\u2581wi", 3),
    ("\u2581wi", "d", "\u2581wid", 3),
    ("\u2581wid", "est", "\u2581widest", 3),
    ("\u2581low", "e", "\u2581lowe", 2),
)

# The first five merges are animated step by step in the corpus. The remaining
# rules are appended as the loop "keeps going".
ANIMATED_MERGES = 5

# Rules that are drawn before the "..." continuation: the animated ones plus a
# few appended rows. The vocabulary panel mirrors exactly these merges.
EXTRA_MERGES = 3
VISIBLE_MERGES = ANIMATED_MERGES + EXTRA_MERGES

# Stopping condition shown when the loop finishes.
VOCAB_TARGET = r"|V| = 32{,}000"

# Encode example: a word never seen during training, split into symbols.
NEW_WORD = "lowest"
NEW_SYMBOLS = ("\u2581", "l", "o", "w", "e", "s", "t")

# The two subword tokens "▁lowest" reduces to, and their ids.
RESULT_TOKENS = ("\u2581low", "est")
TOKEN_IDS = ("1024", "2048")

# Code snippet shown during the fast-encode act. ``sp`` is the conventional
# alias for SentencePiece's processor.
CODE_ENCODE = (
    ("tokens = ", None),
    ("sp.encode", "compute"),
    ("(", "muted"),
    ('"lowest"', "activation"),
    (")", "muted"),
)
