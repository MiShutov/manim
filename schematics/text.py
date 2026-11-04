"""Text primitives: token chips for tokenization diagrams."""

from __future__ import annotations

from manim import DOWN, FadeIn, RoundedRectangle, Text, VGroup

from schematics.theme import Theme


class TokenChip(VGroup):
    """Rounded chip that represents one token or subword piece.

    Attributes:
        text: Token text shown inside the chip.
        rect: Background rounded rectangle.
        label: ``Text`` label inside the rectangle.
    """

    def __init__(
        self,
        text: str,
        theme: Theme,
        *,
        color_key: str = "key",
        height: float = 0.38,
        pad: float = 0.14,
        mono: bool = False,
        size: str = "caption",
    ):
        """Build a chip sized around ``text``.

        Args:
            text: Token text or token id.
            theme: Theme supplying colors and font sizes.
            color_key: Semantic color key for the outline and fill.
            height: Chip height in scene units.
            pad: Horizontal padding on each side of the label.
            mono: Use the theme monospace font for token ids.
            size: Named font size from ``Theme.sizes``.
        """
        self.text = text
        color = theme.color(color_key)
        font = theme.mono_font if mono else theme.font

        self.label = Text(
            text,
            font_size=theme.size(size),
            font=font,
            color=theme.color("text"),
        )
        width = self.label.width + 2 * pad

        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=1.4,
            fill_color=color,
            fill_opacity=0.12,
        )
        self.label.move_to(self.rect.get_center())

        super().__init__(self.rect, self.label)

    def reveal(self):
        """Return a small fade-in animation for the chip."""
        return [FadeIn(self, shift=DOWN * 0.06)]
