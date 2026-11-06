"""Text primitives: token chips, code lines, and vocabulary tables."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from manim import (
    DOWN,
    FadeIn,
    Indicate,
    LEFT,
    MarkupText,
    NORMAL,
    RIGHT,
    RoundedRectangle,
    Text,
    VGroup,
)

from schematics.theme import Theme

# Alignment anchors for the key/value pair inside one vocabulary cell.
_KEY_RIGHT = -0.22
_COLON_X = -0.12
_VALUE_LEFT = 0.02


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
        width: float | None = None,
        valign: str = "center",
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
            width: Optional fixed chip width. When omitted the chip hugs the
                label; when given, the label is centered inside and scaled
                down to fit if necessary.
            valign: Vertical alignment of the label inside the chip:
                ``"center"`` (default) or ``"bottom"``. Bottom alignment is
                used for low symbols such as the SentencePiece space marker
                ``\\u2581`` so they sit at the chip baseline instead of the
                middle.
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
        if width is None:
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
        if self.label.width > width - 2 * pad:
            self.label.scale_to_fit_width(max(width - 2 * pad, 0.01))
        if valign == "bottom":
            bottom_y = self.rect.get_bottom()[1] + self.label.height / 2 + 0.05
            self.label.move_to([self.rect.get_center()[0], bottom_y, 0])
        else:
            self.label.move_to(self.rect.get_center())

        super().__init__(self.rect, self.label)

    def reveal(self):
        """Return a small fade-in animation for the chip."""
        return [FadeIn(self, shift=DOWN * 0.06)]


class CodeLine(MarkupText):
    """Monospace line of code with semantic per-segment colors.

    The line is built from ``(text, color_key)`` segments. Use ``None`` as
    the color key for ordinary source text; use ``"compute"`` for function
    calls, ``"activation"`` for string or variable arguments, and
    ``"muted"`` for punctuation.

    Attributes:
        segments: The colored segments used to build the line.
    """

    def __init__(
        self,
        segments: Sequence[tuple[str, str | None]],
        theme: Theme,
        *,
        size: str = "small",
        weight: str = NORMAL,
    ):
        """Build a code line from styled text segments.

        Args:
            segments: Pairs of source text and semantic color key.
            theme: Theme supplying the mono font and colors.
            size: Named font size from ``Theme.sizes``.
            weight: Pango font weight for the whole line.
        """
        self.segments = tuple(segments)
        markup = "".join(
            f'<span foreground="{theme.color(color_key or "text")}">'
            f"{escape(text, quote=False)}</span>"
            for text, color_key in self.segments
        )
        super().__init__(
            markup,
            font=theme.mono_font,
            font_size=theme.size(size),
            weight=weight,
        )


class VocabEntry(VGroup):
    """One JSON-style ``"token_id": "string"`` entry in a vocabulary table.

    Attributes:
        token_id: Numeric token id used as the dictionary key.
        token: Token string stored under that id.
        rect: Rounded cell background toggled by highlighting.
        key_label: Quoted token id.
        value_label: Quoted token string.
    """

    def __init__(
        self,
        token_id: str,
        token: str,
        theme: Theme,
        *,
        width: float,
        height: float,
    ):
        """Build a vocabulary entry centered at the origin.

        Args:
            token_id: Numeric token id.
            token: Token string for that id.
            theme: Theme supplying colors, fonts, and sizes.
            width: Cell width in scene units.
            height: Cell height in scene units.
        """
        self.token_id = token_id
        self.token = token
        self.theme = theme
        muted = theme.color("muted")

        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.06,
            stroke_color=theme.color("border"),
            stroke_width=0.8,
            fill_color=theme.color("panel"),
            fill_opacity=0.0,
        )

        size = theme.size("small")
        self.key_label = Text(
            f'"{token_id}"',
            font_size=size,
            font=theme.mono_font,
            color=muted,
        )
        self.colon_label = Text(
            ":",
            font_size=size,
            font=theme.mono_font,
            color=muted,
        )
        self.value_label = Text(
            f'"{token}"',
            font_size=size,
            font=theme.mono_font,
            color=theme.color("activation"),
        )
        self.comma_label = Text(
            ",",
            font_size=size,
            font=theme.mono_font,
            color=muted,
        )

        self.key_label.move_to([_KEY_RIGHT, 0, 0], aligned_edge=RIGHT)
        self.colon_label.move_to([_COLON_X, 0, 0])
        self.value_label.move_to([_VALUE_LEFT, 0, 0], aligned_edge=LEFT)
        self.comma_label.next_to(self.value_label, RIGHT, buff=0.015)

        super().__init__(
            self.rect,
            self.key_label,
            self.colon_label,
            self.value_label,
            self.comma_label,
        )

    def highlight(self, *, active: bool = True):
        """Return an animation that toggles the cell highlight.

        Args:
            active: ``True`` for the green selected state, ``False`` for the
                neutral table state.
        """
        if active:
            return (
                self.rect.animate.set_stroke(
                    color=self.theme.color("activation"),
                    width=1.6,
                ).set_fill(
                    color=self.theme.color("activation"),
                    opacity=0.16,
                )
            )
        return (
            self.rect.animate.set_stroke(
                color=self.theme.color("border"),
                width=0.8,
            ).set_fill(
                color=self.theme.color("panel"),
                opacity=0.0,
            )
        )


class VocabularyTable(VGroup):
    """Grid of :class:`VocabEntry` cells, laid out row by row.

    The table is centered at the origin so a scene can move it into place.
    Entries keep their JSON order; ``by_id`` gives a direct lookup for
    highlighting.

    Attributes:
        entries: All vocabulary cells in source order.
        by_id: Mapping from token id to its cell.
        columns: Number of grid columns.
    """

    def __init__(
        self,
        entries: Sequence[tuple[str, str]],
        theme: Theme,
        *,
        columns: int = 2,
        cell_width: float = 4.2,
        cell_height: float = 0.36,
        row_gap: float = 0.08,
        col_gap: float = 0.35,
    ):
        """Create and lay out a vocabulary grid.

        Args:
            entries: Pairs of ``(token_id, token)`` in display order.
            theme: Theme supplying colors, fonts, and sizes.
            columns: Number of cells per row.
            cell_width: Width of one cell in scene units.
            cell_height: Height of one cell in scene units.
            row_gap: Vertical gap between rows.
            col_gap: Horizontal gap between columns.
        """
        if columns < 1:
            raise ValueError("columns must be positive")
        self.columns = columns
        self.entries = [
            VocabEntry(
                token_id,
                token,
                theme,
                width=cell_width,
                height=cell_height,
            )
            for token_id, token in entries
        ]
        self.by_id = {entry.token_id: entry for entry in self.entries}

        rows = (len(self.entries) + columns - 1) // columns
        total_width = columns * cell_width + (columns - 1) * col_gap
        total_height = rows * cell_height + (rows - 1) * row_gap
        for index, entry in enumerate(self.entries):
            row, col = divmod(index, columns)
            entry.move_to(
                [
                    -total_width / 2 + cell_width / 2
                    + col * (cell_width + col_gap),
                    total_height / 2 - cell_height / 2
                    - row * (cell_height + row_gap),
                    0,
                ]
            )

        super().__init__(*self.entries)

    def entry(self, token_id: str) -> VocabEntry:
        """Return the cell for ``token_id``.

        Raises:
            KeyError: If the id is not present in the visible window.
        """
        return self.by_id[token_id]

    def highlight(self, token_id: str, *, active: bool = True):
        """Return the highlight animation for one cell."""
        return self.entry(token_id).highlight(active=active)

    def reveal(self):
        """Return staggered fade-in animations for all cells."""
        return [FadeIn(entry, shift=DOWN * 0.05) for entry in self.entries]


class SymbolStrip(VGroup):
    """Horizontal row of symbol chips that can highlight and merge pairs.

    This is the core building block for byte-pair encoding: a word starts as
    one chip per character and is repeatedly compressed by merging two
    neighbouring chips into a single chip. The strip is left-anchored at its
    origin, so a scene can place it with ``move_to(..., aligned_edge=LEFT)``
    and the left edge stays fixed as merges shrink the row.

    Attributes:
        symbols: Current ordered list of symbol strings.
        chips: Current ordered list of ``TokenChip`` mobjects.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        theme: Theme,
        *,
        height: float = 0.34,
        pad: float = 0.09,
        buff: float = 0.06,
        color_key: str = "activation",
        size: str = "small",
        mono: bool = True,
        marker: str | None = None,
    ):
        """Build a chip per symbol, laid out left to right.

        Args:
            symbols: Symbol strings, for example ``("l", "o", "w", "</w>")``.
            theme: Theme supplying colors and font sizes.
            height: Chip height in scene units.
            pad: Horizontal padding inside each chip.
            buff: Gap between neighbouring chips.
            color_key: Semantic color key for the chips.
            size: Named font size from ``Theme.sizes``.
            mono: Render every symbol in the theme monospace font.
            marker: Optional special symbol (for example an end-of-word
                marker) drawn muted instead of the strip color.
        """
        self.symbols = list(symbols)
        self.theme = theme
        self.chip_height = height
        self.pad = pad
        self.buff = buff
        self.color_key = color_key
        self.font_size = size
        self.mono = mono
        self.marker = marker
        self.chips: list[TokenChip] = []
        self._origin: list[float] = [0.0, 0.0, 0.0]

        super().__init__()
        self._rebuild()

    def _make_chip(self, text: str, *, width: float | None = None) -> TokenChip:
        """Create a chip matching the strip's styling."""
        is_marker = self.marker is not None and text == self.marker
        key = "muted" if is_marker else self.color_key
        return TokenChip(
            text,
            self.theme,
            color_key=key,
            height=self.chip_height,
            pad=self.pad,
            mono=self.mono,
            size=self.font_size,
            width=width,
            valign="bottom" if is_marker else "center",
        )

    def _rebuild(self) -> None:
        """Replace every chip from the current ``symbols`` list."""
        for chip in self.chips:
            self.remove(chip)
        self.chips = [self._make_chip(symbol) for symbol in self.symbols]
        self.add(*self.chips)
        self._layout()

    def _layout(self) -> None:
        """Place chips left to right with the left edge at ``self._origin``."""
        origin_x, origin_y = self._origin[0], self._origin[1]
        x = 0.0
        for chip in self.chips:
            chip.move_to([origin_x + x + chip.width / 2, origin_y, 0])
            x += chip.width + self.buff

    def set_left(self, x: float, y: float) -> "SymbolStrip":
        """Anchor the strip's left edge at ``(x, y)`` and relayout."""
        self._origin = [x, y, 0]
        self._layout()
        return self

    def set_center(self, x: float, y: float) -> "SymbolStrip":
        """Center the strip at ``(x, y)`` and relayout."""
        total = sum(chip.width for chip in self.chips) + self.buff * max(
            0, len(self.chips) - 1
        )
        return self.set_left(x - total / 2, y)

    def find_pair(self, left: str, right: str) -> int | None:
        """Return the index of the first ``(left, right)`` pair, or ``None``."""
        for index in range(len(self.symbols) - 1):
            if self.symbols[index] == left and self.symbols[index + 1] == right:
                return index
        return None

    def highlight_pair(
        self,
        index: int,
        *,
        color_key: str = "score",
        scale_factor: float = 1.12,
    ):
        """Return animations that flash the ``index``/``index + 1`` chips."""
        color = self.theme.color(color_key)
        return [
            Indicate(
                self.chips[index],
                color=color,
                scale_factor=scale_factor,
            ),
            Indicate(
                self.chips[index + 1],
                color=color,
                scale_factor=scale_factor,
            ),
        ]

    def merge_pair(self, index: int):
        """Collapse chips ``index`` and ``index + 1`` into one, in place.

        Call this after fading the two old chips out. It removes them from the
        strip, inserts the merged chip, and relayouts the row. The surviving
        chips and the merged chip are reset to their pre-merge positions so
        the caller can animate them sliding into place.

        Returns:
            ``(merged_chip, move_animations)`` where ``move_animations``
            animates every surviving chip and the merged chip into their final
            positions.
        """
        left = self.chips[index]
        right = self.chips[index + 1]
        merged_text = left.text + right.text
        # Keep the chip height, but glue the two widths together so the merged
        # label stays readable instead of shrinking to fit the text.
        merged_chip = self._make_chip(
            merged_text, width=left.width + right.width + self.buff
        )

        survivors = [
            chip
            for i, chip in enumerate(self.chips)
            if i not in (index, index + 1)
        ]
        old_centers = {id(chip): chip.get_center() for chip in survivors}
        midpoint = (left.get_center() + right.get_center()) / 2

        self.symbols[index : index + 2] = [merged_text]
        self.chips[index : index + 2] = [merged_chip]
        self.remove(left, right)
        self.add(merged_chip)
        self._layout()
        targets = {id(chip): chip.get_center() for chip in self.chips}

        moves = []
        for chip in survivors:
            chip.move_to(old_centers[id(chip)])
            moves.append(chip.animate.move_to(targets[id(chip)]))
        merged_chip.move_to(midpoint).set_opacity(0.0)
        moves.append(
            merged_chip.animate.move_to(targets[id(merged_chip)]).set_opacity(1.0)
        )
        return merged_chip, moves
