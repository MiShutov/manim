"""Rotary positional encoding explained through 2-D pair rotations.

The scene starts from one head of Q or K drawn as a block of green scalar
cells: rows are sequence positions and columns are the ``head_dim``
coordinates, grouped into four 2-D pairs. A frame highlights the selected
row ``m`` and moves down as the position sweep advances. The four pairs of
that row are attached to four rotation dials, which show how the angle
``m * theta_i`` depends on both the pair index and the sequence position.

Color grammar:
    - green: the head block, rotating vectors, and iterating numbers;
    - white: every external label, formula, and the notation legend.

Run with:

    python scripts/render.py positional_encoding -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    RoundedRectangle,
    Text,
    Triangle,
    ValueTracker,
    VGroup,
    linear,
)

from schematics import (
    PairMatrix,
    RotationDial,
    SchematicScene,
)
from scenes.transformer.positional_encoding.specs import (
    ANGLE_FORMULA,
    ANGLE_FORMULA_K,
    BLOCK_LABEL,
    FAST_NOTE,
    FREQ_FORMULA,
    HEAD_DIM_LABEL,
    N_PAIRS,
    N_TOKENS,
    N_TOKENS_LABEL,
    NOTATION_ROWS,
    NOTATION_TITLE,
    PAIR_INDEX,
    PAIR_LABELS,
    PAIR_ROTATION,
    QK_SHAPES,
    RELATIVE,
    RELATIVE_NOTE,
    ROTATION_CAPTION,
    ROTATION_MATRIX,
    SEQ_LEN,
    SLOW_NOTE,
    THETAS,
    TIMELINE_CAPTION,
    TITLE,
)

# --- Layout ---------------------------------------------------------------
TITLE_Y = 3.82

# One head as a two-dimensional block: N_tokens rows by head_dim columns.
BLOCK_X = -3.20
BLOCK_Y = 1.55
BLOCK_CELL = 0.15
BLOCK_GAP = 0.025
BLOCK_PAIR_GAP = 0.17
BLOCK_ROW_GAP = 0.02
BLOCK_PITCH = BLOCK_CELL + BLOCK_ROW_GAP
BLOCK_WIDTH = (
    N_PAIRS * (2 * BLOCK_CELL + BLOCK_GAP)
    + (N_PAIRS - 1) * BLOCK_PAIR_GAP
)
BLOCK_HEIGHT = N_TOKENS * BLOCK_CELL + (N_TOKENS - 1) * BLOCK_ROW_GAP
BLOCK_TOP = BLOCK_Y + BLOCK_HEIGHT / 2
BLOCK_BOTTOM = BLOCK_Y - BLOCK_HEIGHT / 2
BLOCK_LEFT = BLOCK_X - BLOCK_WIDTH / 2
BLOCK_RIGHT = BLOCK_X + BLOCK_WIDTH / 2

# Four dials, one per coordinate pair, to the right of the block.
DIAL_Y = BLOCK_Y
DIAL_RADIUS = 0.42
DIAL_XS = (-1.00, 0.15, 1.30, 2.45)
THETA_Y = 0.72
PHI_Y = 0.42

SPEED_NOTE_Y = DIAL_Y + DIAL_RADIUS + 0.22
TIMELINE_Y = -1.65
TIMELINE_LEFT = -4.10
TIMELINE_RIGHT = 2.90
TIMELINE_TICKS = (0, 4, 8, SEQ_LEN - 1)
M_READOUT_X = 3.70

SUMMARY_Y = -2.55
RELATIVE_Y = -3.08
RELATIVE_NOTE_Y = -3.60

PANEL_X = 4.55
MATRIX_Y = 0.18
MATRIX_CAPTION_Y = -0.45


class RotaryPositionalEncoding(SchematicScene):
    """Animated explanation of RoPE on one attention head."""

    def construct(self):
        """Build the head block, rotation dials, and position sweep."""
        theme = self.theme
        position = ValueTracker(0.0)

        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        legend = self._notation()

        # --- One head from Q/K as a block of scalar cells -----------------
        block = PairMatrix(
            N_TOKENS,
            N_PAIRS,
            theme,
            cell=BLOCK_CELL,
            gap=BLOCK_GAP,
            pair_gap=BLOCK_PAIR_GAP,
            row_gap=BLOCK_ROW_GAP,
            color_key="activation",
            fill_opacity=0.12,
            pair_labels=PAIR_LABELS,
            label_color_key="text",
        ).shift([BLOCK_X, BLOCK_Y, 0])

        block_label = Text(
            BLOCK_LABEL,
            font_size=theme.size("hud_sub"),
            font=theme.font,
            color=theme.color("activation"),
        ).move_to([BLOCK_X, BLOCK_TOP + 0.40, 0])

        n_tokens_x = BLOCK_LEFT - 0.18
        n_tokens_bracket = VGroup(
            Line(
                [n_tokens_x, BLOCK_BOTTOM, 0],
                [n_tokens_x, BLOCK_TOP, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
            Line(
                [n_tokens_x - 0.05, BLOCK_TOP, 0],
                [n_tokens_x + 0.05, BLOCK_TOP, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
            Line(
                [n_tokens_x - 0.05, BLOCK_BOTTOM, 0],
                [n_tokens_x + 0.05, BLOCK_BOTTOM, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
        )
        n_tokens_label = MathTex(
            N_TOKENS_LABEL,
            font_size=theme.size("small") + 1,
            color=theme.color("text"),
        ).rotate(PI / 2)
        n_tokens_label.move_to([n_tokens_x - 0.30, BLOCK_Y, 0])

        head_dim_y = BLOCK_BOTTOM - 0.16
        head_dim_bracket = VGroup(
            Line(
                [BLOCK_LEFT, head_dim_y, 0],
                [BLOCK_RIGHT, head_dim_y, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
            Line(
                [BLOCK_LEFT, head_dim_y - 0.05, 0],
                [BLOCK_LEFT, head_dim_y + 0.05, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
            Line(
                [BLOCK_RIGHT, head_dim_y - 0.05, 0],
                [BLOCK_RIGHT, head_dim_y + 0.05, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.1,
            ),
        )
        head_dim_label = Text(
            HEAD_DIM_LABEL,
            font_size=theme.size("small") + 1,
            font=theme.font,
            color=theme.color("text"),
        ).move_to([BLOCK_X, head_dim_y - 0.20, 0])

        row_top_y = block.row_group(0).get_center()[1]

        def row_y(value: float) -> float:
            """Return the y coordinate of the selected block row."""
            value = max(0.0, min(float(N_TOKENS - 1), value))
            return row_top_y - value * BLOCK_PITCH

        head_frame = RoundedRectangle(
            width=BLOCK_WIDTH + 0.10,
            height=BLOCK_CELL + 0.05,
            corner_radius=0.03,
            stroke_color=theme.color("activation"),
            stroke_width=2.2,
            fill_color=theme.color("activation"),
            fill_opacity=0.06,
        ).move_to([BLOCK_X, row_y(0), 0])

        connector_dot = Dot(
            [BLOCK_RIGHT + 0.06, row_y(0), 0],
            radius=0.035,
            color=theme.color("activation"),
        )

        # --- Rotation dials and the fan from the selected row -------------
        dials = [
            RotationDial(
                theme,
                radius=DIAL_RADIUS,
                color_key="activation",
                tip_length=0.070,
            ).move_to([x, DIAL_Y, 0])
            for x in DIAL_XS
        ]
        trails = []
        for dial in dials:
            trail = dial.make_trail()
            trail.set_stroke(opacity=0.42)
            trails.append(trail)

        connector_lines = VGroup(
            *[
                Line(
                    [BLOCK_RIGHT + 0.06, row_y(0), 0],
                    dial.get_left(),
                    stroke_color=theme.color("muted"),
                    stroke_width=1.0,
                    stroke_opacity=0.55,
                )
                for dial in dials
            ]
        )

        theta_labels = VGroup(
            *[
                MathTex(
                    self._theta_tex(index),
                    font_size=theme.size("small") + 1,
                    color=theme.color("text"),
                ).move_to([x, THETA_Y, 0])
                for index, x in enumerate(DIAL_XS)
            ]
        )
        phi_groups, phi_numbers = self._phi_readouts(DIAL_XS)
        phi_all = VGroup(*phi_groups)

        # --- Formulas -----------------------------------------------------
        qk_shapes = self._formula(
            QK_SHAPES, x=PANEL_X, y=3.18, color_key="text"
        )
        freq_formula = self._formula(
            FREQ_FORMULA, x=PANEL_X, y=2.55, color_key="text"
        )
        angle_formula = self._formula(
            ANGLE_FORMULA, x=PANEL_X, y=1.98, color_key="text"
        )
        angle_formula_k = self._formula(
            ANGLE_FORMULA_K, x=PANEL_X, y=0.88, color_key="text"
        )
        pair_index = self._formula(
            PAIR_INDEX,
            x=PANEL_X,
            y=1.42,
            color_key="text",
            size_key="hud_sub",
        )
        rotation_matrix = self._formula(
            ROTATION_MATRIX,
            x=PANEL_X,
            y=MATRIX_Y,
            color_key="text",
            size_key="hud_sub",
            max_width=3.0,
        )
        rotation_caption = self.caption(
            ROTATION_CAPTION,
            x=PANEL_X,
            y=MATRIX_CAPTION_Y,
            color_key="text",
        )

        # --- Timeline and summary -----------------------------------------
        timeline, ticks = self._timeline()
        marker = Triangle(
            fill_color=theme.color("activation"),
            fill_opacity=1.0,
            stroke_width=0,
        ).scale(0.065).rotate(PI)
        marker.move_to([self._timeline_x(0), TIMELINE_Y + 0.12, 0])

        timeline_caption = self.caption(
            TIMELINE_CAPTION,
            x=TIMELINE_LEFT - 0.15,
            y=TIMELINE_Y,
            color_key="text",
        ).move_to(
            [TIMELINE_LEFT - 0.15, TIMELINE_Y, 0],
            aligned_edge=RIGHT,
        )
        m_group, m_number = self._m_readout()

        summary = self._formula(
            PAIR_ROTATION,
            x=0.0,
            y=SUMMARY_Y,
            color_key="text",
            max_width=12.6,
        )
        relative = self._formula(
            RELATIVE,
            x=0.0,
            y=RELATIVE_Y,
            color_key="text",
            max_width=12.4,
        )
        relative_note = self._formula(
            RELATIVE_NOTE,
            x=0.0,
            y=RELATIVE_NOTE_Y,
            color_key="text",
            size_key="small",
            max_width=11.0,
        )

        fast_note = self.caption(
            FAST_NOTE,
            x=DIAL_XS[0],
            y=SPEED_NOTE_Y,
            color_key="text",
        )
        slow_note = self.caption(
            SLOW_NOTE,
            x=DIAL_XS[-1],
            y=SPEED_NOTE_Y,
            color_key="text",
        )

        # --- Animation ----------------------------------------------------
        self.play(
            FadeIn(title, shift=DOWN * 0.15),
            FadeIn(legend),
            run_time=0.8,
        )

        self.play(*block.reveal(), run_time=1.0)
        self.play(
            FadeIn(block_label),
            FadeIn(n_tokens_bracket),
            FadeIn(n_tokens_label),
            FadeIn(head_dim_bracket),
            FadeIn(head_dim_label),
            FadeIn(qk_shapes),
            run_time=0.6,
        )

        self.play(FadeIn(head_frame), run_time=0.3)
        self.play(
            Indicate(
                head_frame,
                color=theme.color("activation"),
                scale_factor=1.05,
            ),
            run_time=0.5,
        )

        self.play(
            LaggedStart(*[FadeIn(dial) for dial in dials], lag_ratio=0.10),
            FadeIn(theta_labels),
            run_time=0.9,
        )
        self.play(
            FadeIn(connector_dot),
            *[Create(line) for line in connector_lines],
            run_time=0.6,
        )

        self.play(
            FadeIn(freq_formula),
            FadeIn(pair_index),
            run_time=0.5,
        )
        self.play(
            FadeIn(angle_formula),
            FadeIn(angle_formula_k),
            FadeIn(phi_all),
            FadeIn(rotation_matrix),
            FadeIn(rotation_caption),
            run_time=0.6,
        )

        # The frame, connector dot, and fan follow the selected row.
        head_frame.add_updater(
            lambda mob, dt: mob.move_to(
                [BLOCK_X, row_y(position.get_value()), 0]
            )
        )
        connector_dot.add_updater(
            lambda mob, dt: mob.move_to(
                [BLOCK_RIGHT + 0.06, row_y(position.get_value()), 0]
            )
        )
        for line, dial in zip(connector_lines, dials):
            line.add_updater(
                lambda mob, dt, dial=dial: mob.put_start_and_end_on(
                    [
                        BLOCK_RIGHT + 0.06,
                        row_y(position.get_value()),
                        0,
                    ],
                    dial.get_left(),
                )
            )
        for dial, theta in zip(dials, THETAS):
            dial.add_updater(
                lambda mob, dt, theta=theta: mob.set_angle(
                    position.get_value() * theta
                )
            )
        for number, theta in zip(phi_numbers, THETAS):
            number.add_updater(
                lambda mob, dt, theta=theta: mob.set_value(
                    position.get_value() * theta
                )
            )

        self.add(*trails)
        self.bring_to_front(*trails)

        # First rotation: the angle depends on both theta_i and m.
        self.play(
            position.animate.set_value(3.0),
            run_time=1.8,
            rate_func=linear,
        )

        marker.move_to(
            [self._timeline_x(position.get_value()), TIMELINE_Y + 0.12, 0]
        )
        m_number.set_value(position.get_value())

        self.play(
            Create(timeline),
            FadeIn(ticks),
            FadeIn(marker),
            FadeIn(m_group),
            FadeIn(timeline_caption),
            run_time=0.8,
        )

        marker.add_updater(
            lambda mob, dt: mob.move_to(
                [self._timeline_x(position.get_value()), TIMELINE_Y + 0.12, 0]
            )
        )
        m_number.add_updater(
            lambda mob, dt: mob.set_value(position.get_value())
        )

        # Sweep the whole sequence: fast pairs spin, slow pairs creep.
        self.play(
            position.animate.set_value(SEQ_LEN - 1),
            run_time=9.0,
            rate_func=linear,
        )

        # Replace the dynamic trails with deterministic final arcs so the
        # last frame is stable when rendered with ``-s``.
        static_trails = VGroup(
            *[dial.static_trail() for dial in dials]
        )
        self.play(
            *[FadeOut(trail) for trail in trails],
            FadeIn(static_trails),
            run_time=0.4,
        )

        self.play(
            FadeIn(fast_note),
            FadeIn(slow_note),
            run_time=0.9,
        )

        self.play(
            FadeIn(summary),
            FadeIn(relative),
            FadeIn(relative_note),
            run_time=0.9,
        )
        self.wait(2)

        # Freeze all values for the final frame.
        for dial in dials:
            dial.clear_updaters()
        for number in phi_numbers:
            number.clear_updaters()
        marker.clear_updaters()
        m_number.clear_updaters()
        head_frame.clear_updaters()
        connector_dot.clear_updaters()
        for line in connector_lines:
            line.clear_updaters()

    # --- helpers ----------------------------------------------------------
    def _formula(
        self,
        tex: str,
        *,
        x: float,
        y: float,
        color_key: str,
        max_width: float = 2.9,
        size_key: str = "outro",
    ) -> MathTex:
        """Return a LaTeX formula fitted to the available panel width."""
        formula = MathTex(
            tex,
            font_size=self.theme.size(size_key),
            color=self.theme.color(color_key),
        ).move_to([x, y, 0])
        if formula.width > max_width:
            formula.scale_to_fit_width(max_width)
        return formula

    def _theta_tex(self, index: int) -> str:
        """Return the exact frequency label for pair ``index``."""
        if index == 0:
            return r"\theta_0 = 1"
        return rf"\theta_{index} = 10^{{-{index}}}"

    def _phi_readouts(self, xs):
        """Create static angle prefixes with live numeric values."""
        theme = self.theme
        groups = []
        numbers = []
        for index, x in enumerate(xs):
            prefix = MathTex(
                rf"\varphi_{index} =",
                font_size=theme.size("small") + 1,
                color=theme.color("text"),
            )
            number = DecimalNumber(
                0.0,
                num_decimal_places=2,
                font_size=theme.size("small") + 1,
                color=theme.color("activation"),
            )
            group = VGroup(prefix, number).arrange(RIGHT, buff=0.06)
            group.move_to([x, PHI_Y, 0])
            groups.append(group)
            numbers.append(number)
        return groups, numbers

    def _timeline(self):
        """Return the position axis and its sparse tick labels."""
        theme = self.theme
        axis = Line(
            [TIMELINE_LEFT, TIMELINE_Y, 0],
            [TIMELINE_RIGHT, TIMELINE_Y, 0],
            stroke_color=theme.color("border"),
            stroke_width=1.4,
        )
        ticks = VGroup()
        for value in TIMELINE_TICKS:
            x = self._timeline_x(value)
            tick = Line(
                [x, TIMELINE_Y - 0.06, 0],
                [x, TIMELINE_Y + 0.06, 0],
                stroke_color=theme.color("border"),
                stroke_width=1.2,
            )
            label = MathTex(
                str(value),
                font_size=theme.size("small"),
                color=theme.color("text"),
            ).move_to([x, TIMELINE_Y - 0.20, 0])
            ticks.add(tick, label)
        return axis, ticks

    def _m_readout(self):
        """Return a live ``m = ...`` readout next to the timeline."""
        theme = self.theme
        prefix = MathTex(
            r"m =",
            font_size=theme.size("outro"),
            color=theme.color("text"),
        )
        number = DecimalNumber(
            0,
            num_decimal_places=0,
            font_size=theme.size("outro"),
            color=theme.color("activation"),
        )
        group = VGroup(prefix, number).arrange(RIGHT, buff=0.08)
        group.move_to([M_READOUT_X, TIMELINE_Y, 0])
        return group, number

    def _notation(self) -> VGroup:
        """Build the left-hand notation legend."""
        theme = self.theme
        symbol_x = -6.90
        desc_x = -6.18
        title_y = 3.12
        row_start = 2.82
        row_gap = 0.25

        heading = Text(
            NOTATION_TITLE,
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("text"),
        ).move_to([symbol_x, title_y, 0], aligned_edge=LEFT)

        rows = VGroup()
        for index, (symbol_tex, description) in enumerate(NOTATION_ROWS):
            y = row_start - index * row_gap
            symbol = MathTex(
                symbol_tex,
                font_size=theme.size("help"),
                color=theme.color("text"),
            ).move_to([symbol_x, y, 0], aligned_edge=LEFT)
            label = Text(
                description,
                font_size=theme.size("help") - 1,
                font=theme.font,
                color=theme.color("text"),
            ).move_to([desc_x, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(symbol, label))

        return VGroup(heading, rows)

    @staticmethod
    def _timeline_x(value: float) -> float:
        """Map a sequence position to its x coordinate on the timeline."""
        ratio = value / (SEQ_LEN - 1)
        return TIMELINE_LEFT + ratio * (TIMELINE_RIGHT - TIMELINE_LEFT)
