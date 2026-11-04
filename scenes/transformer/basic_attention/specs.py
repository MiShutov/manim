"""Data for the classical multi-head attention scene."""

from __future__ import annotations

import math
from collections.abc import Sequence

from schematics.specs import BadgeSpec

# Llama 3 8B dimensions, kept here so the badge strip is self-contained.
D_MODEL = 4096
N_HEADS = 32
HEAD_DIM = 128

# Schematic dimensions used by the single-head view.
N_TOKENS = 5
SCHEMATIC_DIM = 4
QUERY_INDEX = 2

TOKENS = ("t1", "t2", "t3", "t4", "t5")
QUERY_TOKEN = TOKENS[QUERY_INDEX]

# Raw attention logits for the query token.
LOGITS = (2.0, 1.0, 0.5, 0.0, -0.5)


def softmax(values: Sequence[float]) -> tuple[float, ...]:
    """Return a numerically stable softmax over ``values``."""
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


WEIGHTS = softmax(LOGITS)

# Tensor shapes shown in the top strip, from input to output.
ATTENTION_BADGES = (
    BadgeSpec("input", "(B, T, 4096)", "input"),
    BadgeSpec("Q, K, V", "(B, T, 4096) × 3", "query"),
    BadgeSpec("32 heads", "(B, 32, T, 128)", "key"),
    BadgeSpec("scores", "(B, 32, T, T)", "score"),
    BadgeSpec("softmax", "(B, 32, T, T)", "weight"),
    BadgeSpec("output", "(B, T, 4096)", "output"),
)
