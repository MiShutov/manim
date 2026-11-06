"""Reusable tensor visualizations: schematic matrices and weight bars."""

from __future__ import annotations

from collections.abc import Sequence

from manim import (
    FadeIn,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
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


class CausalMaskGrid(VGroup):
    """Schematic causal mask split into allowed and blocked cells.

    Rows are query positions ``i`` and columns are key positions ``j``. A
    cell is allowed when ``j <= i`` (on or below the diagonal) and blocked
    when ``j > i`` (above the diagonal). Allowed cells use the activation
    color, blocked cells a muted outline, mirroring the ``0`` and
    ``-\\infty`` values of the mask.

    Attributes:
        size: Number of rows and columns.
        cells: ``cells[row][col]`` rounded rectangles.
        allowed: Group of cells on and below the diagonal.
        blocked: Group of cells above the diagonal.
        row_centers: Local y coordinate of each row center.
        col_centers: Local x coordinate of each column center.
        block_size: Width and height of the grid without labels.
    """

    def __init__(
        self,
        size: int,
        theme: Theme,
        *,
        cell: float = 0.17,
        gap: float = 0.03,
        color_key: str = "activation",
        blocked_color_key: str = "muted",
        fill_opacity: float = 0.45,
        blocked_fill_opacity: float = 0.06,
    ):
        """Build a centered ``size`` by ``size`` mask grid.

        Args:
            size: Number of rows and columns; must be positive.
            theme: Theme supplying colors.
            cell: Side length of one cell.
            gap: Distance between adjacent cells.
            color_key: Semantic color key for allowed cells.
            blocked_color_key: Semantic color key for blocked cells.
            fill_opacity: Fill opacity of allowed cells.
            blocked_fill_opacity: Fill opacity of blocked cells.
        """
        if size < 1:
            raise ValueError("CausalMaskGrid needs at least one cell")

        self.size = size
        self.theme = theme
        allowed_color = theme.color(color_key)
        blocked_color = theme.color(blocked_color_key)

        pitch = cell + gap
        total = size * cell + (size - 1) * gap
        x_start = -total / 2 + cell / 2
        y_top = total / 2 - cell / 2

        self.block_size = total
        self.row_centers = [y_top - row * pitch for row in range(size)]
        self.col_centers = [x_start + col * pitch for col in range(size)]

        self.cells: list[list[RoundedRectangle]] = []
        self.allowed = VGroup()
        self.blocked = VGroup()
        self._rows: list[VGroup] = []

        for row in range(size):
            row_cells = VGroup()
            current_row: list[RoundedRectangle] = []
            for col in range(size):
                allowed = col <= row
                square = RoundedRectangle(
                    width=cell,
                    height=cell,
                    corner_radius=0.025,
                    stroke_color=allowed_color if allowed else blocked_color,
                    stroke_width=1.0 if allowed else 0.8,
                    fill_color=allowed_color if allowed else blocked_color,
                    fill_opacity=(
                        fill_opacity if allowed else blocked_fill_opacity
                    ),
                ).move_to(
                    [self.col_centers[col], self.row_centers[row], 0]
                )
                current_row.append(square)
                row_cells.add(square)
                if allowed:
                    self.allowed.add(square)
                else:
                    self.blocked.add(square)
            self.cells.append(current_row)
            self._rows.append(row_cells)

        super().__init__(self.allowed, self.blocked)

    def reveal(self):
        """Return a row-by-row fade-in animation."""
        return [
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.04) for row in self._rows],
                lag_ratio=0.08,
            )
        ]


class PairStrip(VGroup):
    """Horizontal vector split into adjacent 2-D coordinate pairs.

    The strip is a schematic row of ``2 * n_pairs`` cells. Each pair is
    enclosed in its own rounded box, which makes the RoPE grouping explicit:
    coordinates ``(x_0, x_1)`` rotate together, ``(x_2, x_3)`` rotate
    together, and so on.

    Attributes:
        n_pairs: Number of coordinate pairs.
        cells: Flat group of all individual cells, left to right.
        boxes: Group of pair boxes, one per pair.
        labels: Optional pair labels placed above the boxes.
        pairs: One ``VGroup`` per pair containing its box and two cells.
    """

    def __init__(
        self,
        n_pairs: int,
        theme: Theme,
        *,
        cell: float = 0.30,
        gap: float = 0.05,
        pair_gap: float = 0.55,
        color_key: str = "activation",
        fill_opacity: float = 0.14,
        pair_labels: Sequence[str] | None = None,
    ):
        """Build a centered strip of ``n_pairs`` coordinate pairs.

        Args:
            n_pairs: Number of 2-D pairs; must be positive.
            theme: Theme supplying colors and font sizes.
            cell: Side length of one scalar cell.
            gap: Gap between the two cells of a pair and between cells.
            pair_gap: Gap between neighboring pair boxes.
            color_key: Semantic color key for cells and pair boxes.
            fill_opacity: Fill opacity of the scalar cells.
            pair_labels: Optional LaTeX labels placed above each pair.
        """
        if n_pairs < 1:
            raise ValueError("PairStrip needs at least one pair")
        if pair_labels is not None and len(pair_labels) != n_pairs:
            raise ValueError("pair_labels must contain one label per pair")

        self.n_pairs = n_pairs
        self.theme = theme
        color = theme.color(color_key)

        pitch = cell + gap
        pair_width = 2 * cell + gap
        pair_pitch = pair_width + pair_gap
        total_width = n_pairs * pair_pitch - pair_gap
        start_x = -total_width / 2

        self.cells = VGroup()
        self.boxes = VGroup()
        self.labels = VGroup()
        self.pairs: list[VGroup] = []

        for index in range(n_pairs):
            origin_x = start_x + index * pair_pitch
            pair_cells = VGroup()
            for offset in (0.0, pitch):
                square = RoundedRectangle(
                    width=cell,
                    height=cell,
                    corner_radius=0.04,
                    stroke_color=color,
                    stroke_width=1.1,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                ).move_to([origin_x + offset + cell / 2, 0, 0])
                pair_cells.add(square)
                self.cells.add(square)

            box = RoundedRectangle(
                width=pair_width + 0.14,
                height=cell + 0.14,
                corner_radius=0.07,
                stroke_color=color,
                stroke_width=1.4,
                fill_opacity=0,
            ).move_to([origin_x + pair_width / 2, 0, 0])
            self.boxes.add(box)

            pair = VGroup(box, pair_cells)
            if pair_labels is not None:
                label = MathTex(
                    pair_labels[index],
                    font_size=theme.size("small"),
                    color=theme.color("muted"),
                ).move_to([origin_x + pair_width / 2, cell / 2 + 0.20, 0])
                self.labels.add(label)
                pair.add(label)

            self.pairs.append(pair)

        super().__init__(self.boxes, self.cells, self.labels)

    def reveal(self):
        """Return a pair-by-pair fade-in animation."""
        return [
            LaggedStart(
                *[FadeIn(pair, shift=RIGHT * 0.06) for pair in self.pairs],
                lag_ratio=0.10,
            )
        ]


class PairMatrix(VGroup):
    """Two-dimensional block of scalar cells grouped into coordinate pairs.

    Rows usually represent sequence positions and columns represent the
    ``head_dim`` coordinates. The larger gap between pair columns makes the
    RoPE grouping visible without drawing a box around every pair:

    .. text::

        [x0 x1]  [x2 x3]  [x4 x5]  [x6 x7]
        [x0 x1]  [x2 x3]  [x4 x5]  [x6 x7]
        ...

    Attributes:
        rows: Number of rows.
        n_pairs: Number of 2-D pairs per row.
        cells: ``cells[row][pair]`` contains the two cells of that pair.
        row_centers: Local y coordinate of each row center.
        pair_centers: Local x coordinate of each pair center.
        block_width: Width of the cell block without labels.
        block_height: Height of the cell block without labels.
    """

    def __init__(
        self,
        rows: int,
        n_pairs: int,
        theme: Theme,
        *,
        cell: float = 0.15,
        gap: float = 0.025,
        pair_gap: float = 0.17,
        row_gap: float = 0.02,
        color_key: str = "activation",
        fill_opacity: float = 0.12,
        pair_labels: Sequence[str] | None = None,
        label_color_key: str = "muted",
    ):
        """Build a centered pair block.

        Args:
            rows: Number of rows; must be positive.
            n_pairs: Number of 2-D pairs per row; must be positive.
            theme: Theme supplying colors and font sizes.
            cell: Side length of one scalar cell.
            gap: Gap between the two cells of a pair.
            pair_gap: Extra gap between neighboring pairs.
            row_gap: Vertical gap between rows.
            color_key: Semantic color key for the cells.
            fill_opacity: Fill opacity of the scalar cells.
            pair_labels: Optional LaTeX labels placed above the pair columns.
            label_color_key: Color key for the pair labels.
        """
        if rows < 1 or n_pairs < 1:
            raise ValueError("PairMatrix needs at least one row and one pair")
        if pair_labels is not None and len(pair_labels) != n_pairs:
            raise ValueError("pair_labels must contain one label per pair")

        self.rows = rows
        self.n_pairs = n_pairs
        self.theme = theme
        color = theme.color(color_key)

        pitch = cell + gap
        pair_width = 2 * cell + gap
        pair_pitch = pair_width + pair_gap
        total_width = n_pairs * pair_pitch - pair_gap

        row_pitch = cell + row_gap
        total_height = rows * row_pitch - row_gap

        start_x = -total_width / 2
        top_y = total_height / 2 - cell / 2

        self.block_width = total_width
        self.block_height = total_height
        self.cells: list[list[list[RoundedRectangle]]] = []
        self.row_centers: list[float] = []
        self.cells_group = VGroup()

        for row in range(rows):
            y = top_y - row * row_pitch
            row_cells: list[list[RoundedRectangle]] = []
            for pair in range(n_pairs):
                origin_x = start_x + pair * pair_pitch
                pair_cells = []
                for offset in (0.0, pitch):
                    square = RoundedRectangle(
                        width=cell,
                        height=cell,
                        corner_radius=0.025,
                        stroke_color=color,
                        stroke_width=1.0,
                        fill_color=color,
                        fill_opacity=fill_opacity,
                    ).move_to([origin_x + offset + cell / 2, y, 0])
                    pair_cells.append(square)
                    self.cells_group.add(square)
                row_cells.append(pair_cells)
            self.cells.append(row_cells)
            self.row_centers.append(y)

        self.pair_centers = [
            start_x + pair * pair_pitch + pair_width / 2
            for pair in range(n_pairs)
        ]

        self.labels = VGroup()
        if pair_labels is not None:
            for x, label_tex in zip(self.pair_centers, pair_labels):
                label = MathTex(
                    label_tex,
                    font_size=theme.size("small"),
                    color=theme.color(label_color_key),
                ).move_to([x, total_height / 2 + 0.18, 0])
                self.labels.add(label)

        super().__init__(self.cells_group, self.labels)

    def row_group(self, index: int) -> VGroup:
        """Return all cells of ``index`` as one group."""
        return VGroup(
            *[cell for pair in self.cells[index] for cell in pair]
        )

    def reveal(self):
        """Return a row-by-row fade-in animation, including pair labels."""
        animations = []
        if len(self.labels) > 0:
            animations.append(FadeIn(self.labels))
        animations.append(
            LaggedStart(
                *[
                    FadeIn(self.row_group(row), shift=RIGHT * 0.04)
                    for row in range(self.rows)
                ],
                lag_ratio=0.05,
            )
        )
        return animations


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
