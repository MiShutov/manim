"""Data for the tokenization scene: text, subword tokens, ids, and vocab."""

from __future__ import annotations

TITLE = "Tokenization: text to tokens and back"

# The example contains a subword split: "sleeping" -> "sleep" + "ing".
TEXT = "The cat is sleeping."
TOKEN_PIECES = ("The", "cat", "is", "sleep", "ing", ".")
TOKEN_IDS = ("464", "3797", "318", "5396", "278", "13")

# The two chips that together form one word in the source text.
SUBWORD_SPAN = (3, 4)
SUBWORD_WORD = "sleeping"

# Visible window into the vocabulary, sorted by token id. The six ids used by
# the example are highlighted; the others are neutral decoy rows.
VOCAB = (
    ("13", "."),
    ("262", "the"),
    ("278", "ing"),
    ("318", "is"),
    ("319", "on"),
    ("464", "The"),
    ("2603", "mat"),
    ("3797", "cat"),
    ("5396", "sleep"),
    ("7623", "sofa"),
)

# A longer strip used by the right-hand scrolling panel. The ten example rows
# stay contiguous in the middle; extra rows above and below make the scroll
# visibly move through a much larger table.
SCROLL_VOCAB = (
    ("1", "a"),
    ("2", "I"),
    ("3", "s"),
    ("4", "t"),
    ("5", "ing"),
    ("6", "ed"),
    ("7", "er"),
    ("8", "ly"),
    ("9", "un"),
    ("10", "re"),
    ("11", "pre"),
    ("12", "non"),
) + VOCAB + (
    ("8001", "so"),
    ("8123", "fa"),
    ("9002", "tion"),
    ("10234", "token"),
    ("12000", "ization"),
    ("14567", "vocab"),
    ("20001", "large"),
    ("24567", "window"),
    ("30000", "scroll"),
    ("40000", "huge"),
    ("50000", "table"),
    ("60000", "json"),
    ("70000", "encode"),
    ("80000", "decode"),
    ("90000", "text"),
    ("100255", "..."),
)

# Code segments are (text, semantic color key) pairs. ``None`` means the
# ordinary source-text color.
CODE_ENCODE = (
    ("ids", "muted"),
    (" = ", None),
    ("tokenizer.encode", "compute"),
    ("(", "muted"),
    ("text", "activation"),
    (")", "muted"),
)
CODE_DECODE = (
    ("text", "activation"),
    (" = ", None),
    ("tokenizer.decode", "compute"),
    ("(", "muted"),
    ("ids", "muted"),
    (")", "muted"),
)

# Shorter labels for the tokenizer blocks themselves.
ENCODE_LABEL = (
    ("tokenizer.encode", None),
    ("(", "muted"),
    ("text", "activation"),
    (")", "muted"),
)
DECODE_LABEL = (
    ("tokenizer.decode", None),
    ("(", "muted"),
    ("ids", "muted"),
    (")", "muted"),
)

ENCODE_DIRECTION = "encode: string \u2192 id"
DECODE_DIRECTION = "decode: id \u2192 string"
