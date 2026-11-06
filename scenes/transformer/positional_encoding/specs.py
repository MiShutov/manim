"""Data for the rotary positional encoding (RoPE) scene.

The scene zooms into one attention head and one query vector ``q_m``. The
head dimension is reduced to ``d_k = 8`` so the vector splits into four
2-D pairs, which is enough to show the fast/slow frequency contrast while
staying readable on screen.
"""

from __future__ import annotations

# Schematic head dimension. Real models use d_k = 64 or more, but the
# frequency ordering is the same; eight coordinates give four clean pairs.
D_K = 8
N_PAIRS = D_K // 2

# Sequence positions used by the timeline sweep and the head block rows.
# Sixteen rows give enough travel for the fast pair to wind several turns.
N_TOKENS = 16
SEQ_LEN = N_TOKENS
BASE = 10000.0

# theta_i = 10000 ** (-2i / d_k). With d_k = 8 this is 1, 0.1, 0.01, 0.001.
THETAS = tuple(BASE ** (-2.0 * index / D_K) for index in range(N_PAIRS))

TITLE = "Rotary positional encoding (RoPE)"

QK_SHAPES = r"q_m, k_n \in \mathbb{R}^{d_k}"
BLOCK_LABEL = "one head from Q/K"
N_TOKENS_LABEL = r"N_{\mathrm{tokens}}"
HEAD_DIM_LABEL = "head dim"
PAIR_LABELS = (r"i=0", r"i=1", r"i=2", r"i=3")

FREQ_FORMULA = r"\theta_i = 10000^{-2i/d_k}"
ANGLE_FORMULA = r"\varphi_i(m) = m\,\theta_i"
ANGLE_FORMULA_K = r"\varphi_i(n) = n\,\theta_i"
PAIR_INDEX = r"i = 0, \dots, d_k/2 - 1"

PAIR_ROTATION = (
    r"\mathrm{RoPE}(q_m)_i = R(m\,\theta_i)\,q_{m,i}"
)
RELATIVE = (
    r"\langle \mathrm{RoPE}(q_m), \mathrm{RoPE}(k_n)\rangle = f(m-n)"
)
ROTATION_MATRIX = (
    r"R(\varphi) = \begin{pmatrix}"
    r"\cos\varphi & -\sin\varphi \\"
    r"\sin\varphi & \cos\varphi"
    r"\end{pmatrix}"
)
RELATIVE_NOTE = (
    r"f\ \text{is a function of the relative offset } m-n"
)
ROTATION_CAPTION = "2-D rotation matrix"

NOTATION_TITLE = "Notation"
NOTATION_ROWS = (
    (r"B, S, h", "batch, seq, heads"),
    (r"d_k", "head dim"),
    (r"i", "pair index"),
    (r"m, n", "positions"),
    (r"\theta_i", "frequency"),
    (r"\varphi_i", "angle"),
    (r"R(\varphi)", "2-D rotation"),
    (r"f(m-n)", "relative offset"),
)

FAST_NOTE = "fast pair: many turns"
SLOW_NOTE = "slow pair: tiny angle"
TIMELINE_CAPTION = "sequence position m"
