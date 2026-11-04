"""Reusable tensor visualizations: schematic matrices and weight bars."""

from __future__ import annotations

from collections.abc import Sequence

from manim import (
    FadeIn,
    LaggedStart,
    LEFT,
    Line,
    RIGHT,
    RoundedRectangle,
    Text,
    VGroup,
)

from schematics.theme import Theme


class MatrixGrid(VGroup):
    """Schematic matrix drawn as a grid of rounded cells.

    The grid is intentionally abstract: it shows tensor structure, not real
    values. Use it for Q/K/V blocks, score matrices, or any 2-D tensor.

    Attributes:
        rows: Number of rows.
        cols: Number of columns.
        cells: ``cells[row][col]`` rounded rectangles.
        grid: Group containing only the cells.
        row_labels: Optional labels placed left of each row.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        theme: Theme,
        *,
        cell: float = 0.26,
        gap: float = 0.05,
        color_key: str = "key",
        row_labels: Sequence[str] | None = None,
        highlight_rows: Sequence[int] = (),
        fill_opacity: float = 0.14,
        highlight_opacity: float = 0.42,
    ):
        """Build a ``rows`` by ``cols`` grid centered at the origin.

        Args:
            rows: Number of rows; must be positive.
            cols: Number of columns; must be positive.
            theme: Theme supplying colors and font sizes.
            cell: Side length of one cell.
            gap: Distance between adjacent cells.
            color_key: Semantic color key for the cells.
            row_labels: Optional text labels, one per row.
            highlight_rows: Row indices drawn with a stronger fill.
            fill_opacity: Fill opacity of regular cells.
            highlight_opacity: Fill opacity of highlighted cells.
        """
        if rows < 1 or cols < 1:
            raise ValueError("MatrixGrid needs at least one row and one column")
        if row_labels is not None and len(row_labels) != rows:
            raise ValueError("row_labels must contain exactly one label per row")

        self.rows = rows
        self.cols = cols
        self.theme = theme
        color = theme.color(color_key)
        highlighted = set(highlight_rows)

        pitch = cell + gap
        self.cells: list[list[RoundedRectangle]] = []
        for row in range(rows):
            current_row: list[RoundedRectangle] = []
            for col in range(cols):
                opacity = (
                    highlight_opacity if row in highlighted else fill_opacity
                )
                square = RoundedRectangle(
                    width=cell,
                    height=cell,
                    corner_radius=0.035,
                    stroke_color=color,
                    stroke_width=1.2,
                    fill_color=color,
                    fill_opacity=opacity,
                ).move_to([col * pitch, -row * pitch, 0])
                current_row.append(square)
            self.cells.append(current_row)

        self.grid = VGroup(
            *[square for current_row in self.cells for square in current_row]
        )
        self.grid.move_to([0, 0, 0])

        self.row_labels: list[Text] = []
        label_group = VGroup()
        if row_labels is not None:
            for index, text in enumerate(row_labels):
                label = Text(
                    text,
                    font_size=theme.size("small"),
                    font=theme.mono_font,
                    color=theme.color("muted"),
                )
                label.next_to(self.cells[index][0], LEFT, buff=0.10)
                self.row_labels.append(label)
                label_group.add(label)

        super().__init__(self.grid, label_group)
        self.move_to([0, 0, 0])

        self._units = []
        for index, current_row in enumerate(self.cells):
            unit = VGroup(*current_row)
            if self.row_labels:
                unit.add(self.row_labels[index])
            self._units.append(unit)

    def row(self, index: int) -> VGroup:
        """Return all cells of ``index`` as one group for highlighting."""
        return VGroup(*self.cells[index])

    def reveal(self):
        """Return a row-by-row fade-in animation."""
        return [
            LaggedStart(
                *[FadeIn(unit, shift=RIGHT * 0.06) for unit in self._units],
                lag_ratio=0.10,
            )
        ]


class WeightBars(VGroup):
    """Vertical list of labelled values drawn as horizontal bars.

    Supports two modes:

    - unsigned: bars grow from the left edge, useful for softmax weights;
    - signed: bars diverge from a central baseline, useful for logits.

    Attributes:
        rows: One ``VGroup`` per label, containing label, track, bar, value.
        bars: Bar rectangles in the same order as ``rows``.
        values: Numeric values passed to the constructor.
        signed: Whether the bars use a central baseline.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        theme: Theme,
        *,
        color_key: str = "weight",
        width: float = 1.15,
        bar_height: float = 0.13,
        gap: float = 0.12,
        signed: bool = False,
        max_abs: float | None = None,
        value_fmt: str = "{:.2f}",
    ):
        """Build the bar list centered at the origin.

        Args:
            labels: Row labels, for example token names.
            values: Numeric value for each row.
            theme: Theme supplying colors and font sizes.
            color_key: Semantic color key for the bars.
            width: Width of the full bar track.
            bar_height: Height of one bar.
            gap: Vertical gap between rows.
            signed: If true, draw diverging bars around a central baseline.
            max_abs: Value mapped to the full half-width or width. Defaults
                to the largest absolute value in ``values``.
            value_fmt: Format string used for the numeric label.
        """
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length")
        if not values:
            raise ValueError("WeightBars needs at least one value")

        self.theme = theme
        self.values = tuple(values)
        self.signed = signed
        color = theme.color(color_key)
        track_color = theme.color("border")

        scale = max_abs if max_abs is not None else max(
            abs(value) for value in values
        )
        scale = max(scale, 1e-9)

        pitch = bar_height + gap
        self.rows: list[VGroup] = []
        self.bars: list[RoundedRectangle] = []

        for index, (label_text, value) in enumerate(zip(labels, values)):
            y = -index * pitch

            label = Text(
                label_text,
                font_size=theme.size("small"),
                font=theme.mono_font,
                color=theme.color("muted"),
            )
            label.move_to([-width / 2 - 0.12 - label.width / 2, y, 0])

            track = RoundedRectangle(
                width=width,
                height=bar_height + 0.04,
                corner_radius=0.04,
                stroke_color=track_color,
                stroke_width=0.8,
                fill_color=track_color,
                fill_opacity=0.12,
            ).move_to([0, y, 0])

            ratio = min(abs(value) / scale, 1.0)
            if signed:
                bar_width = max(0.02, ratio * width / 2)
                x = bar_width / 2 if value >= 0 else -bar_width / 2
            else:
                bar_width = max(0.02, ratio * width)
                x = -width / 2 + bar_width / 2

            bar = RoundedRectangle(
                width=bar_width,
                height=bar_height,
                corner_radius=0.03,
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.85,
            ).move_to([x, y, 0])

            value_label = Text(
                value_fmt.format(value),
                font_size=theme.size("small"),
                font=theme.mono_font,
                color=theme.color("text"),
            )
            value_label.next_to(track, RIGHT, buff=0.08)

            self.rows.append(VGroup(label, track, bar, value_label))
            self.bars.append(bar)

        children = [*self.rows]
        self.baseline: Line | None = None
        if signed:
            top = bar_height / 2 + 0.02
            bottom = -(len(values) - 1) * pitch - bar_height / 2 - 0.02
            self.baseline = Line(
                [0, top, 0],
                [0, bottom, 0],
                stroke_width=1.0,
                color=theme.color("muted"),
                stroke_opacity=0.6,
            )
            children.append(self.baseline)

        super().__init__(*children)
        self.move_to([0, 0, 0])

    def reveal(self):
        """Return a row-by-row fade-in animation, including the baseline."""
        fade_ins = [FadeIn(row, shift=RIGHT * 0.06) for row in self.rows]
        if self.baseline is not None:
            fade_ins.insert(0, FadeIn(self.baseline))
        return [LaggedStart(*fade_ins, lag_ratio=0.10)]
