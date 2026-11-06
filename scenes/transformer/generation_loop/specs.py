"""Data for the autoregressive generation loop scene.

The scene continues the transformer-decoder anatomy: a five-token prompt is
run through the model, the vocabulary logits are scored, the argmax token is
appended to the context, and the loop repeats until the context holds seven
tokens.

Shared labels and badges come from the anatomy scene, so both diagrams stay
visually identical.
"""

from __future__ import annotations

from scenes.transformer.model_anatomy.specs import (
    BRACE_LABEL,
    HIDDEN,
    LM_HEAD_MATMUL,
    LM_HEAD_NAME,
    LOGITS,
    LOOKUP_FORMULA,
    LOOKUP_NAME,
    NOTATION_ROWS,
    NOTATION_TITLE,
    PRIME_MARK,
    PROMPT,
    TOKEN_IDS,
    TOKEN_PIECES,
)

TITLE = "Transformer-decoder autoregressive generation (without KV-Cache)"

# Vocabulary used for the demo, including the prompt tokens.
# Each entry is ``(token_text, token_id)``.
VOCAB = (
    ("The", "464"),
    ("cat", "3797"),
    ("is", "318"),
    ("sleep", "5396"),
    ("ing", "278"),
    ("on", "319"),
    ("the", "262"),
    ("sofa", "7623"),
    ("mat", "2603"),
    (".", "13"),
)

# Logit values per generation round, in the same order as ``VOCAB``.
# The argmax of each round is appended to the context.
LOGIT_ROUNDS = (
    (-1.2, -0.8, -0.5, -1.6, -2.0, 6.2, 5.4, 1.4, 0.6, -0.3),
    (0.4, 0.2, 0.1, -0.5, -0.9, 4.9, 6.6, 3.2, 2.4, 1.9),
)

# Tokens produced by the two demonstrated rounds.
GENERATED = (
    ("on", "319"),
    ("the", "262"),
)

# Explanations shown next to the live sequence length.
SEQUENCE_WORDS = {
    5: "five tokens in a sequence",
    6: "six tokens in a sequence",
    7: "seven tokens in a sequence",
}
