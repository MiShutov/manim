"""Data for the multimodal token scene: image patches and text subwords."""

from __future__ import annotations

# A 12 x 12 pixel-art cat face. Each character maps to a theme color key.
CAT_PIXELS = (
    "..F......F..",
    ".FFF....FFF.",
    ".FIF....FIF.",
    ".FIFFFFFFIF.",
    ".FFFFFFFFFF.",
    ".FFEFFFFEFF.",
    ".FFPFFFFPFF.",
    ".FFFFNNFFFF.",
    ".FFFNNNNFFF.",
    ".FFFFMMFFFF.",
    ".FFFFFFFFFF.",
    "..FFFFFFFF..",
)

CAT_COLORS = {
    ".": "panel",
    "F": "residual",
    "I": "score",
    "E": "success",
    "P": "panel",
    "N": "attention",
    "M": "panel",
}

IMAGE_PATCH_ROWS = 3
IMAGE_PATCH_COLS = 3
IMAGE_TOKENS = IMAGE_PATCH_ROWS * IMAGE_PATCH_COLS

TEXT_PROMPT = "The cat is sleeping"
TOKEN_PIECES = ("The", "cat", "is", "sleep", "ing")
TOKEN_IDS = ("464", "3797", "318", "5396", "278")
SUBWORD_INDICES = (3, 4)

# Schematic embedding width; real models use hundreds or thousands of dims.
EMBED_DIM = 5

IMAGE_FORMULA = (
    r"X_{\mathrm{image}} \in \mathbb{R}^{N_{\mathrm{image}} \times d}"
)
TEXT_FORMULA = r"X_{\mathrm{text}} \in \mathbb{R}^{N_{\mathrm{text}} \times d}"
