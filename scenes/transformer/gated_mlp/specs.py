"""Data for the gated MLP (SwiGLU) scene.

The scene explains the Llama feed-forward block as a computation graph:

    g = x @ W_gate
    u = x @ W_up
    h = act_fn(g) * u
    y = h @ W_down

It then fixes ``act_fn`` to SiLU, which makes the block a SwiGLU block, and
shows element by element that every token is processed independently.

Visual grammar:
    - purple: computation blocks and asymptotic cost;
    - green: activations, tensor shapes, and token rows;
    - yellow: learned weight matrices;
    - white: structural annotations and formulas.
"""

from __future__ import annotations

from schematics.specs import BadgeSpec

# Llama 3 8B values; the diagrams keep the dimensions symbolic.
D_MODEL = 4096
D_FF = 14336

SHAPE_MODEL = r"(B, S, d_{\mathrm{model}})"
SHAPE_FF = r"(B, S, d_{\mathrm{ff}})"
WEIGHT_UP_SHAPE = r"(d_{\mathrm{model}}, d_{\mathrm{ff}})"

TITLE = "Gated MLP"
TITLE_SWIGLU = "SwiGLU"

# Left-hand notation legend. Mathematical symbols stay in LaTeX.
NOTATION_TITLE = "Notation"
NOTATION_ROWS = (
    (r"B,\ S", "batch, sequence", "activation"),
    (r"d_{\mathrm{model}}", "hidden width", "activation"),
    (r"d_{\mathrm{ff}}", "intermediate width", "activation"),
    (r"\sigma", "activation function", "compute"),
    (r"W", "learned weights", "weights"),
    (r"@", "matrix product", "compute"),
    (r"\odot", "element-wise product", "compute"),
)

# --- Graph badges ---------------------------------------------------------

INPUT_CARD = BadgeSpec(
    r"\mathrm{input}\ x",
    SHAPE_MODEL,
    "activation",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

GATE_PROJ = BadgeSpec(
    r"g = x \mathbin{@} W_{\mathrm{gate}}",
    SHAPE_FF,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
    title_highlights=((r"W_{\mathrm{gate}}", "weights"),),
)

UP_PROJ = BadgeSpec(
    r"u = x \mathbin{@} W_{\mathrm{up}}",
    SHAPE_FF,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
    title_highlights=((r"W_{\mathrm{up}}", "weights"),),
)

DOWN_PROJ = BadgeSpec(
    r"y = h \mathbin{@} W_{\mathrm{down}}",
    SHAPE_MODEL,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
    title_highlights=((r"W_{\mathrm{down}}", "weights"),),
)

GATE_CARD = BadgeSpec(
    r"g",
    SHAPE_FF,
    "activation",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

UP_CARD = BadgeSpec(
    r"u",
    SHAPE_FF,
    "activation",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

ACT_FN = BadgeSpec(
    r"\sigma(g) = \mathrm{ACT\_FN}(g)",
    SHAPE_FF,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

MUL = BadgeSpec(
    r"h = \sigma(g) \odot u",
    SHAPE_FF,
    "compute",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

OUTPUT_CARD = BadgeSpec(
    r"\mathrm{output}\ y",
    SHAPE_MODEL,
    "activation",
    title_tex=True,
    shape_tex=True,
    shape_color="activation",
)

# Replacement titles for the SwiGLU step.
SILU_TITLE = r"\sigma(g) = \mathrm{SiLU}(g)"

# --- Side panels ----------------------------------------------------------

SILU_PANEL_TITLE = "SiLU (Swish)"
SILU_FORMULA = r"\mathrm{SiLU}(x) = x \cdot \sigma(x)"
SIGMOID_FORMULA = r"\sigma(x) = \frac{1}{1 + e^{-x}}"
SILU_NOTE = "SwiGLU = Swish-Gated Linear Unit"

TOTAL_COST = (
    r"\text{total } O(B \cdot S \cdot d_{\mathrm{model}}"
    r" \cdot d_{\mathrm{ff}})"
)
CONCRETE_NOTE = (
    r"\text{Llama 3 8B: } d_{\mathrm{model}} = 4096,"
    r"\ d_{\mathrm{ff}} = 14336"
)

# --- Forward pseudocode ---------------------------------------------------

# ``indent`` is the number of 0.32-unit steps from the left code margin.
CODE_LINES = (
    (
        0,
        (
            ("def ", "compute"),
            ("forward", "compute"),
            ("(", "muted"),
            ("x", "activation"),
            ("):", "muted"),
        ),
    ),
    (
        1,
        (
            ("x", "activation"),
            (" = ", "muted"),
            ("self", None),
            (".", None),
            ("down_proj", "compute"),
            ("(", "muted"),
        ),
    ),
    (
        2,
        (
            ("self", None),
            (".", None),
            ("act_fn", "compute"),
            ("(", "muted"),
            ("self", None),
            (".", None),
            ("gate_proj", "compute"),
            ("(", "muted"),
            ("x", "activation"),
            (")", "muted"),
            (")", "muted"),
        ),
    ),
    (
        2,
        (
            ("* ", "muted"),
            ("self", None),
            (".", None),
            ("up_proj", "compute"),
            ("(", "muted"),
            ("x", "activation"),
            (")", "muted"),
        ),
    ),
    (
        1,
        ((")", "muted"),),
    ),
    (
        1,
        (
            ("return ", "muted"),
            ("x", "activation"),
        ),
    ),
)

# --- Position-wise token demonstration -----------------------------------

TOKEN_TITLE = "The MLP is position-wise"

X_LABEL = r"X"
X_SHAPE = r"(B \cdot S, d_{\mathrm{model}})"
W_LABEL = r"W_{\mathrm{gate}}"
G_LABEL = r"G"
G_SHAPE = r"(B \cdot S, d_{\mathrm{ff}})"

# Four example rows; ``i = (b, s)`` names one token.
TOKEN_ROWS = (
    r"(b_0, s_0)",
    r"(b_0, s_1)",
    r"(b_0, s_2)",
    r"(b_1, s_0)",
)

X_CAPTION = "one row = one token"
W_CAPTION = "same weights for every token"
G_CAPTION = "same rows, same order"

ELEMENT_FORMULA_PARTS = (
    r"g_{i,j}",
    "=",
    r"\sum_{k}",
    r"x_{i,k}",
    r"\cdot",
    r"w_{k,j}",
)

NOTE_ELEMENT = r"g_{i,j} \text{ depends only on row } i \text{ of } X"
NOTE_NO_MIXING = (
    r"\text{no other token } i' \neq i \text{ enters this element}"
)
NOTE_POSITION = (
    r"\text{the sequence position } s \text{ only names the row;"
    r" it never enters the math}"
)

# --- Finale ---------------------------------------------------------------

SUMMARY_HEADLINE = "matrix products dominate the graph"

TOKEN_FORMULA = (
    r"y_i = \left(\mathrm{SiLU}(x_i \mathbin{@} W_{\mathrm{gate}})"
    r"\odot(x_i \mathbin{@} W_{\mathrm{up}})\right)"
    r" \mathbin{@} W_{\mathrm{down}}"
)
TOKEN_FORMULA_HIGHLIGHTS = (
    (r"y_i", "activation"),
    (r"x_i", "activation"),
    (r"W_{\mathrm{down}}", "weights"),
    (r"W_{\mathrm{gate}}", "weights"),
    (r"W_{\mathrm{up}}", "weights"),
    (r"\mathrm{SiLU}", "compute"),
)
