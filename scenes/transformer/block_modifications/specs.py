"""Text data for the transformer-block swap-in scene.

The forward pass is written as a short Python function without inference
arguments, so the pseudocode mirrors the visual path through the block.
"""

from __future__ import annotations

from schematics.specs import BadgeSpec

TITLE = "Transformer block"

D_MODEL_TEX = r"(B, T, d_{\mathrm{model}})"

IN_BADGE = BadgeSpec(
    title="x",
    shape=D_MODEL_TEX,
    color="activation",
    title_tex=True,
    shape_tex=True,
    shape_color="dimension",
    title_size="badge_title",
)

OUT_BADGE = BadgeSpec(
    title=r"x^{\prime}",
    shape=D_MODEL_TEX,
    color="activation",
    title_tex=True,
    shape_tex=True,
    shape_color="dimension",
    title_size="badge_title",
)

NORM_LABEL = "Normalization"

# Forward pseudocode. ``None`` marks the blank line between the two sub-blocks.
PSEUDO_LINES: tuple[tuple[tuple[str, str | None], ...] | None, ...] = (
    (
        ("def ", "compute"),
        ("forward", "compute"),
        ("(", "muted"),
        ("x", "activation"),
        ("):", "muted"),
    ),
    (("residual", "activation"), (" = ", "muted"), ("x", "activation")),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("norm", "compute"),
        ("(", "muted"),
        ("x", "activation"),
        (")", "muted"),
    ),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("self_attn", "compute"),
        ("(", "muted"),
        ("x", "activation"),
        (")", "muted"),
    ),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("residual", "activation"),
        (" + ", "muted"),
        ("x", "activation"),
    ),
    None,
    (("residual", "activation"), (" = ", "muted"), ("x", "activation")),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("norm", "compute"),
        ("(", "muted"),
        ("x", "activation"),
        (")", "muted"),
    ),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("mlp", "compute"),
        ("(", "muted"),
        ("x", "activation"),
        (")", "muted"),
    ),
    (
        ("x", "activation"),
        (" = ", "muted"),
        ("residual", "activation"),
        (" + ", "muted"),
        ("x", "activation"),
    ),
    (("return ", "muted"), ("x", "activation")),
)

# Forward demo: (pseudocode line, action, target). Actions name the diagram
# element that flashes while the line is highlighted.
#   "input" / "output" -> a badge, "route" -> a residual path,
#   "node" / "plus" -> the matching diagram element.
FORWARD_STEPS = (
    (0, "input", None),
    (1, "route", 0),
    (2, "node", 0),
    (3, "node", 1),
    (4, "plus", 0),
    (6, "route", 1),
    (7, "node", 2),
    (8, "node", 3),
    (9, "plus", 1),
    (10, "output", None),
)

# Left-hand notation legend, in the style of the attention scene.
NOTATION_TITLE = "Notation"
NOTATION_ROWS = (
    (r"B,\ T", "batch, sequence", "activation"),
    (r"d_{\mathrm{model}}", "hidden width", "activation"),
    (r"\oplus", "residual add", "residual"),
)

# Swap-in captions shown to the right of the block: (title, reference).
CAPTIONS = {
    "mhc": (
        "residual stream \u2192 mHC",
        "multi-stream residual mixing \u00b7 arXiv:2512.24880",
    ),
    "causal": (
        "self-attention \u00b7 causal mask",
        "each query sees keys up to its own position",
    ),
    "windowed": (
        "attention \u2192 windowed",
        "local window of recent keys \u00b7 arXiv:2310.06825 \u00b7 Mistral 7B",
    ),
    "gdn": (
        "attention \u2192 Gated DeltaNet",
        "gated delta rule state \u00b7 arXiv:2412.06464 \u00b7 ICLR 2025",
    ),
    "moe": (
        "MLP \u2192 experts",
        "router picks top-k of N experts \u00b7 arXiv:1701.06538 \u00b7 Shazeer et al.",
    ),
    "norm": (
        "normalization: LayerNorm / RMSNorm",
        "same block, different normalization",
    ),
    "gemma": (
        "extra normalization \u00b7 Gemma 4",
        "pre/post norms and QK-norm \u00b7 arXiv:2607.02770",
    ),
}
