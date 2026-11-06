"""Data for the LLM anatomy scene: embeddings, N blocks, and lm_head.

The scene shows a Transformer-decoder in one horizontal line: a prompt
becomes token chips, an ``Embedding`` lookup turns token ids into token
embeddings, N identical transformer blocks produce hidden states, and the
``lm_head`` matmul turns the hidden states into next-token logits.

Visual grammar:
    - purple outlines: computations (embedding lookup, transformer blocks,
      lm_head matmul);
    - green outlines, shapes, activations;
    - yellow symbols: learned weights, here W_lm_head.
"""

from __future__ import annotations

from schematics.specs import BadgeSpec, NodeSpec

TITLE = "Transformer-decoder architecture"
PROMPT = "The cat is sleeping"
TOKEN_PIECES = ("The", "cat", "is", "sleep", "ing")
TOKEN_IDS = ("464", "3797", "318", "5396", "278")
# Token whose chip and activation row are highlighted together.
HIGHLIGHT_TOKEN = 1

# Left-hand legend. The symbol column stays on the left edge.
NOTATION_TITLE = "Notation"
NOTATION_ROWS = (
    (r"B", "batch", "activation"),
    (r"S", "sequence", "activation"),
    (r"d_{\mathrm{model}}", "hidden", "activation"),
    (r"N", "blocks", "activation"),
    (r"W", "learned weights", "weights"),
)

# The wide block shown before the stack unfolds, and one member of the
# unfolded stack.
SINGLE_BLOCK = NodeSpec(label="Transformer", shape="", color="compute")
BLOCK = NodeSpec(
    label="Block",
    shape="",
    color="compute",
    label_size="badge_title",
)

LOOKUP_FORMULA = r"x_i = \mathrm{Embedding}(\mathrm{id}_i)"
LOOKUP_NAME = "embedding"

EMBED_NAME = "token embeddings"
HIDDEN_NAME = "hidden states"
PRIME_MARK = "\u2032"
ACTIV_SHAPE = r"(B,\ S,\ d_{\mathrm{model}})"

HIDDEN = BadgeSpec(
    title=HIDDEN_NAME,
    shape=ACTIV_SHAPE,
    color="activation",
    shape_tex=True,
)

LM_HEAD_MATMUL = BadgeSpec(
    title=r"h' \mathbin{@} W_{\mathrm{lm\_head}}",
    shape=r"(B,\ S,\ \mathrm{vocab})",
    color="compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
    title_highlights=((r"W_{\mathrm{lm\_head}}", "weights"),),
)
LM_HEAD_NAME = r"\mathrm{lm\_head}"

LOGITS = BadgeSpec(
    title="logits",
    shape=r"(B,\ S,\ \mathrm{vocab})",
    color="activation",
    shape_tex=True,
)

BRACE_LABEL = r"N \text{ transformer blocks}"
