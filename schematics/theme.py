"""Central theme for diagrams: colors, fonts, sizes, and spacing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = REPO_ROOT / "assets" / "fonts"


def _colors() -> dict[str, str]:
    """Return the default semantic color palette."""
    return {
        "input": "#8B949E",
        "norm": "#58C4DD",
        "attention": "#B07CFF",
        "ffn": "#5CD65C",
        "residual": "#FF9F1C",
        "signal": "#FFE45C",
        "text": "#F0F6FC",
        "muted": "#8B949E",
        "panel": "#151B23",
        "border": "#30363D",
        "success": "#3FB950",
        "query": "#B07CFF",
        "key": "#58C4DD",
        "value": "#5CD65C",
        "score": "#FF9F1C",
        "weight": "#FFE45C",
        "output": "#3FB950",
    }


def _sizes() -> dict[str, float]:
    """Return the default font-size scale."""
    return {
        "title": 40,
        "outro": 22,
        "node": 21,
        "hud_title": 24,
        "hud_sub": 17,
        "hud_op": 20,
        "hud_shape": 27,
        "hud_detail": 16,
        "hud_value": 20,
        "caption": 15,
        "help": 14,
        "small": 13,
        "badge_title": 17,
        "badge_shape": 13,
    }


@dataclass(frozen=True)
class Theme:
    """Semantic theme shared by every diagram.

    Scenes reference colors by meaning (``norm``, ``attention``, ``ffn``)
    instead of hard-coding a particular shade. Override this dataclass, or
    create a new instance, to restyle all scenes at once.

    Attributes:
        bg: Scene background color.
        font: Pango font family used by ``Text``.
        mono_font: Monospace family for tensor shapes and code-like labels.
        colors: Semantic color map, for example ``{"norm": "#58C4DD"}``.
        sizes: Named font sizes, for example ``{"title": 40}``.
        node_width: Default node width.
        node_height: Default node height.
        node_radius: Corner radius of node rectangles.
        node_stroke: Node border width.
        node_fill_opacity: Node fill opacity.
        arrow_color: Color of connectors between nodes.
        arrow_stroke: Connector stroke width.
        skip_stroke: Residual-connection stroke width.
        panel_radius: HUD panel corner radius.
        panel_stroke: HUD panel border width.
        bar_min_log: Lower bound of the HUD log-scale domain.
        bar_max_log: Upper bound of the HUD log-scale domain.
        signal_min_height: Signal size for the smallest activation.
        signal_max_height: Signal size for the largest activation.
    """

    name: str = "dark"
    bg: str = "#0D1117"
    font: str = "Inter"
    mono_font: str = "DejaVu Sans Mono"

    colors: Mapping[str, str] = field(default_factory=_colors)
    sizes: Mapping[str, float] = field(default_factory=_sizes)

    node_width: float = 3.35
    node_height: float = 0.46
    node_radius: float = 0.08
    node_stroke: float = 2.2
    node_fill_opacity: float = 0.12

    arrow_color: str = "#30363D"
    arrow_stroke: float = 2.2
    skip_stroke: float = 2.5

    panel_radius: float = 0.15
    panel_stroke: float = 1.5

    bar_min_log: float = 2.8
    bar_max_log: float = 5.0
    signal_min_height: float = 0.30
    signal_max_height: float = 0.50

    def color(self, key: str) -> str:
        """Return the color registered under ``key``."""
        return self.colors[key]

    def size(self, key: str) -> float:
        """Return the font size registered under ``key``."""
        return self.sizes[key]

    def log_t(self, elems: int) -> float:
        """Map an activation size to a normalized log-scale value in [0, 1]."""
        t = (math.log10(max(elems, 1)) - self.bar_min_log) / (
            self.bar_max_log - self.bar_min_log
        )
        return max(0.05, min(1.0, t))

    def bar_width(self, elems: int, track_width: float) -> float:
        """Return the HUD bar width for ``elems`` inside ``track_width``."""
        return 0.14 + self.log_t(elems) * (track_width - 0.15)

    def signal_height(self, elems: int) -> float:
        """Return the signal-dot height for an activation of ``elems``."""
        return self.signal_min_height + self.log_t(elems) * (
            self.signal_max_height - self.signal_min_height
        )


DARK = Theme()
