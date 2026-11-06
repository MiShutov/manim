"""Data for the causal multi-head attention diagram.

The scene follows the computation graph from "Attention Is All You Need"
(Vaswani et al., 2017): input -> Q/K/V projections -> RoPE on Q and K ->
per-head scaled dot-product attention with a causal mask -> head
concatenation -> output projection.

Visual grammar:
    - purple: computation blocks and asymptotic cost;
    - green: activations, inputs, outputs, and tensor shapes;
    - yellow: learned weight matrices;
    - white: the causal-mask callout and structural annotations.
"""

from __future__ import annotations

from schematics.specs import BadgeSpec

# Dimensions of the base Transformer from the paper.
D_MODEL = 512
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS

# Tensor shapes used throughout the diagram. Keep subscripts in LaTeX.
INPUT_SHAPE = r"(B, S, d_{\mathrm{model}})"
HEAD_SHAPE = r"(B, h, S, d_k)"
SCORE_SHAPE = r"(B, h, S, S)"
OUTPUT_SHAPE = r"(B, S, d_{\mathrm{model}})"

# Cost of each stage. The per-head scores and the weighted sum use
# B * h * S^2 * d_k, which equals B * S^2 * d_model because h * d_k = d_model.
ROPE_COST = r"O(B \cdot S \cdot d_{\mathrm{model}})"
PROJECTION_COST = r"O(B \cdot S \cdot d_{\mathrm{model}}^2)"
SCORE_COST = r"O(B \cdot S^2 \cdot d_{\mathrm{model}})"
SOFTMAX_COST = r"O(B \cdot h \cdot S^2)"
CONCAT_COST = r"O(B \cdot S \cdot d_{\mathrm{model}})"
OUTPUT_COST = r"O(B \cdot S \cdot d_{\mathrm{model}}^2)"

TITLE = "Multi-head attention"

# Left-hand legend. Mathematical symbols use LaTeX; descriptions stay short
# so the legend fits beside the graph without stealing vertical space.
NOTATION_TITLE = "Notation"
NOTATION_ROWS = (
    (r"B", "batch", "activation"),
    (r"S", "sequence", "activation"),
    (r"h", "heads", "activation"),
    (r"d_{\mathrm{model}}", "hidden", "activation"),
    (r"d_k = d_v", "head dim", "activation"),
    (r"\mathrm{PE}", "RoPE (Q, K)", "compute"),
    (r"@", "matrix product", "compute"),
    (r"W", "learned weights", "weights"),
)

TOTAL_COST = (
    r"\text{total } O(B \cdot S^2 \cdot d_{\mathrm{model}}"
    r" + B \cdot S \cdot d_{\mathrm{model}}^2)"
)
PAPER = "Attention Is All You Need, Vaswani et al. (2017)"

HEAD_FORMULA = (
    r"\mathrm{head}_i = \mathrm{softmax}"
    r"\left(\frac{Q_i \mathbin{@} K_i^\top}{\sqrt{d_k}} + M\right)"
    r" \mathbin{@} V_i"
)

# Causal-mask callout. The mask is added to every head of every batch item,
# so its leading dimensions are 1 and broadcast over ``B`` and ``h``.
MASK_TITLE = "causal mask"
MASK_SYMBOL = r"M"
MASK_BROADCAST = r"\text{broadcast over } B, h"
MASK_SHAPE = r"(1, 1, S, S)"
MASK_FORMULA = (
    r"M_{ij} = \begin{cases}"
    r"0, & j \le i \\"
    r"-\infty, & j > i"
    r"\end{cases}"
)
MASK_SIZE = 5
MASK_LEGEND_KEEP = r"0"
MASK_LEGEND_BLOCK = r"-\infty"
MASK_ROW_AXIS = r"i"
MASK_COL_AXIS = r"j"

CONCAT_NOTE = "heads merged"

INPUT_CARD = BadgeSpec(
    "input x",
    INPUT_SHAPE,
    "activation",
    shape_tex=True,
    title_size="card_title",
)

OUTPUT_CARD = BadgeSpec(
    "output",
    OUTPUT_SHAPE,
    "activation",
    shape_tex=True,
    title_size="card_title",
)

Q_CARD = BadgeSpec(
    "queries Q",
    HEAD_SHAPE,
    "activation",
    shape_tex=True,
    title_size="card_title",
)
K_CARD = BadgeSpec(
    "keys K",
    HEAD_SHAPE,
    "activation",
    shape_tex=True,
    title_size="card_title",
)
V_CARD = BadgeSpec(
    "values V",
    HEAD_SHAPE,
    "activation",
    shape_tex=True,
    title_size="card_title",
)

PROJECTIONS = (
    BadgeSpec(
        r"Q = x \mathbin{@} W_Q",
        INPUT_SHAPE,
        "compute",
        title_tex=True,
        shape_tex=True,
        shape_color="activation",
        title_highlights=((r"W_Q", "weights"),),
    ),
    BadgeSpec(
        r"K = x \mathbin{@} W_K",
        INPUT_SHAPE,
        "compute",
        title_tex=True,
        shape_tex=True,
        shape_color="activation",
        title_highlights=((r"W_K", "weights"),),
    ),
    BadgeSpec(
        r"V = x \mathbin{@} W_V",
        INPUT_SHAPE,
        "compute",
        title_tex=True,
        shape_tex=True,
        shape_color="activation",
        title_highlights=((r"W_V", "weights"),),
    ),
)

ROPE_Q = BadgeSpec(
    r"Q = \mathrm{PE}(Q)",
    HEAD_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)
ROPE_K = BadgeSpec(
    r"K = \mathrm{PE}(K)",
    HEAD_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

SCORES = BadgeSpec(
    r"\mathrm{attn\_weights} = Q \mathbin{@} K^\top / \sqrt{d_k}",
    SCORE_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)
SOFTMAX = BadgeSpec(
    r"\mathrm{attn\_weights} = \mathrm{softmax}"
    r"(\mathrm{attn\_weights} + M, \mathrm{dim}=-1)",
    SCORE_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)
WEIGHTED = BadgeSpec(
    r"\mathrm{attn\_output} = \mathrm{attn\_weights} \mathbin{@} V",
    HEAD_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)
CONCAT = BadgeSpec(
    r"\mathrm{attention\_output} ="
    r" \mathrm{concat\_heads}(\mathrm{attention\_output})",
    INPUT_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)
OUTPUT_PROJ = BadgeSpec(
    r"\mathrm{output} = \mathrm{attention\_output} \mathbin{@} W_O",
    INPUT_SHAPE,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
    title_highlights=((r"W_O", "weights"),),
)

SPLIT_HEADS_LABEL = r"\text{split into } h \text{ heads}"
