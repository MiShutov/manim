"""Gated MLP (SwiGLU): block diagram, shapes, and position-wise tokens.

The scene has three visual acts:

1. Draw the gated MLP block from a Llama model, together with the forward
   pseudocode and every intermediate shape.
2. Replace ``ACT_FN`` with SiLU and rename the block to SwiGLU.
3. Show element by element that each output token depends only on the same
   input token, then connect the corresponding rows of X and G.

Color grammar:
    - purple: computation blocks and matrix products;
    - green: activations, tensor shapes, and token rows;
    - yellow: learned weight matrices;
    - white: formulas and structural annotations.

Run with:

    python scripts/render.py gated_mlp -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    FadeIn,
    FadeOut,
    Flash,
    Indicate,
    LaggedStart,
    MathTex,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

from schematics import (
    Badge,
    CodeLine,
    MatrixGrid,
    SchematicScene,
)
from scenes.transformer.gated_mlp.specs import (
    ACT_FN,
    CODE_LINES,
    CONCRETE_NOTE,
    DOWN_PROJ,
    ELEMENT_FORMULA_PARTS,
    G_CAPTION,
    G_LABEL,
    G_SHAPE,
    GATE_CARD,
    GATE_PROJ,
    INPUT_CARD,
    MUL,
    NOTE_ELEMENT,
    NOTE_NO_MIXING,
    NOTE_POSITION,
    NOTATION_ROWS,
    NOTATION_TITLE,
    OUTPUT_CARD,
    SILU_FORMULA,
    SILU_NOTE,
    SILU_PANEL_TITLE,
    SILU_TITLE,
    SIGMOID_FORMULA,
    SUMMARY_HEADLINE,
    TITLE,
    TITLE_SWIGLU,
    TOKEN_FORMULA,
    TOKEN_FORMULA_HIGHLIGHTS,
    TOKEN_ROWS,
    TOKEN_TITLE,
    TOTAL_COST,
    UP_CARD,
    UP_PROJ,
    WEIGHT_UP_SHAPE,
    W_CAPTION,
    W_LABEL,
    X_CAPTION,
    X_LABEL,
    X_SHAPE,
)

# --- Graph geometry -------------------------------------------------------
TITLE_Y = 3.84
INPUT_Y = 3.14
PROJ_Y = 2.40
CARD_Y = 1.66
ACT_Y = 0.92
MUL_Y = 0.18
DOWN_Y = -0.56
OUTPUT_Y = -1.30

BRANCH_X = (-0.90, 0.90)
GATE_X, UP_X = BRANCH_X
BLOCK_HEIGHT = 0.50

INPUT_WIDTH = 1.35
PROJ_WIDTH = 1.70
CARD_WIDTH = 1.00
ACT_WIDTH = 1.85
MUL_WIDTH = 1.80
DOWN_WIDTH = 1.90
OUTPUT_WIDTH = 1.55

FRAME_LEFT = -2.05
FRAME_RIGHT = 2.05
FRAME_TOP = 3.55
FRAME_BOTTOM = -1.75

# Right-hand code and formula panels stay outside the block frame.
RIGHT_X = 4.65
RIGHT_PANEL_WIDTH = 4.00

CODE_PANEL_Y = 2.20
CODE_PANEL_HEIGHT = 2.15
CODE_X = RIGHT_X - RIGHT_PANEL_WIDTH / 2 + 0.18
CODE_TOP = 2.95
CODE_STEP = 0.30
CODE_INDENT = 0.32

SILU_PANEL_Y = -0.65
SILU_PANEL_HEIGHT = 2.00

# --- Token-demonstration geometry ----------------------------------------
X_GRID_X = -3.95
W_GRID_X = 0.00
G_GRID_X = 3.95
GRID_Y = 0.45
TOKEN_CELL = 0.42
TOKEN_GAP = 0.07
TOKEN_ROWS_COUNT = len(TOKEN_ROWS)
TOKEN_COLS_MODEL = 5
TOKEN_COLS_FF = 4
SPREAD_STEP = 0.12


class GatedMLP(SchematicScene):
    """Animated tour of a gated MLP and its position-wise computation."""

    def construct(self):
        """Run the block, SwiGLU, token-independence, and summary acts."""
        self._build_stage()
        self._play_stage_intro()
        self._play_swiglu()
        self.wait(0.5)
        self._play_token_demo()
        self._play_summary()

    # --- Construction -----------------------------------------------------
    def _build_stage(self):
        """Create the gated-MLP graph, code panel, and side panels."""
        theme = self.theme

        self.title_m = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        self.legend = self._notation()

        self.frame = RoundedRectangle(
            width=FRAME_RIGHT - FRAME_LEFT,
            height=FRAME_TOP - FRAME_BOTTOM,
            corner_radius=0.12,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=0.30,
        ).move_to(
            [
                (FRAME_LEFT + FRAME_RIGHT) / 2,
                (FRAME_TOP + FRAME_BOTTOM) / 2,
                0,
            ]
        )

        self.input_badge = self._badge(INPUT_CARD, INPUT_WIDTH, 0.0, INPUT_Y)
        self.gate_proj = self._badge(GATE_PROJ, PROJ_WIDTH, GATE_X, PROJ_Y)
        self.up_proj = self._badge(UP_PROJ, PROJ_WIDTH, UP_X, PROJ_Y)
        self.gate_card = self._badge(GATE_CARD, CARD_WIDTH, GATE_X, CARD_Y)
        self.up_card = self._badge(UP_CARD, CARD_WIDTH, UP_X, CARD_Y)
        self.act_badge = self._badge(ACT_FN, ACT_WIDTH, GATE_X, ACT_Y)
        self.mul_badge = self._badge(MUL, MUL_WIDTH, 0.0, MUL_Y)
        self.down_proj = self._badge(DOWN_PROJ, DOWN_WIDTH, 0.0, DOWN_Y)
        self.output_badge = self._badge(
            OUTPUT_CARD, OUTPUT_WIDTH, 0.0, OUTPUT_Y
        )

        self.input_arrows = VGroup(
            self._arrow(
                [-0.50, INPUT_Y - BLOCK_HEIGHT / 2, 0],
                [GATE_X, PROJ_Y + BLOCK_HEIGHT / 2, 0],
            ),
            self._arrow(
                [0.50, INPUT_Y - BLOCK_HEIGHT / 2, 0],
                [UP_X, PROJ_Y + BLOCK_HEIGHT / 2, 0],
            ),
        )
        self.proj_to_cards = VGroup(
            self._arrow(
                [GATE_X, PROJ_Y - BLOCK_HEIGHT / 2, 0],
                [GATE_X, CARD_Y + BLOCK_HEIGHT / 2, 0],
            ),
            self._arrow(
                [UP_X, PROJ_Y - BLOCK_HEIGHT / 2, 0],
                [UP_X, CARD_Y + BLOCK_HEIGHT / 2, 0],
            ),
        )
        self.card_to_act = self._arrow(
            [GATE_X, CARD_Y - BLOCK_HEIGHT / 2, 0],
            [GATE_X, ACT_Y + BLOCK_HEIGHT / 2, 0],
        )
        self.act_to_mul = self._arrow(
            [GATE_X, ACT_Y - BLOCK_HEIGHT / 2, 0],
            [-0.65, MUL_Y + BLOCK_HEIGHT / 2, 0],
        )
        self.up_to_mul = self._arrow(
            [UP_X, CARD_Y - BLOCK_HEIGHT / 2, 0],
            [0.65, MUL_Y + BLOCK_HEIGHT / 2, 0],
        )
        self.mul_to_down = self._arrow(
            [0.0, MUL_Y - BLOCK_HEIGHT / 2, 0],
            [0.0, DOWN_Y + BLOCK_HEIGHT / 2, 0],
        )
        self.down_to_output = self._arrow(
            [0.0, DOWN_Y - BLOCK_HEIGHT / 2, 0],
            [0.0, OUTPUT_Y + BLOCK_HEIGHT / 2, 0],
        )
        self.arrows = VGroup(
            self.input_arrows,
            self.proj_to_cards,
            self.card_to_act,
            self.act_to_mul,
            self.up_to_mul,
            self.mul_to_down,
            self.down_to_output,
        )

        self.block_group = VGroup(
            self.frame,
            self.input_badge,
            self.gate_proj,
            self.up_proj,
            self.gate_card,
            self.up_card,
            self.act_badge,
            self.mul_badge,
            self.down_proj,
            self.output_badge,
            self.arrows,
        )

        self.code_panel = self._code_panel()
        self.silu_panel = self._silu_panel()
        self.footer = self._footer()

        # Everything that belongs to the first two acts. It is faded out
        # before the token demonstration and partially restored at the end.
        self.stage_group = VGroup(
            self.block_group,
            self.title_m,
            self.legend,
            self.code_panel,
            self.silu_panel,
            self.footer,
        )

    def _badge(self, spec, width: float, x: float, y: float) -> Badge:
        """Create a badge with its shape visible immediately."""
        return Badge(
            spec,
            self.theme,
            width=width,
            height=BLOCK_HEIGHT,
        ).move_to([x, y, 0])

    def _arrow(self, start, end) -> Arrow:
        """Return a neutral arrow with the shared stroke and tip size."""
        return Arrow(
            start,
            end,
            buff=0.02,
            stroke_width=self.theme.arrow_stroke,
            color=self.theme.color("muted"),
            tip_length=0.09,
            max_tip_length_to_length_ratio=0.90,
        )

    def _code_panel(self) -> VGroup:
        """Build the forward pseudocode beside the graph."""
        theme = self.theme
        background = RoundedRectangle(
            width=RIGHT_PANEL_WIDTH,
            height=CODE_PANEL_HEIGHT,
            corner_radius=0.12,
            stroke_color=theme.color("border"),
            stroke_width=1.0,
            fill_color=theme.color("panel"),
            fill_opacity=0.55,
        ).move_to([RIGHT_X, CODE_PANEL_Y, 0])

        lines = VGroup()
        for index, (indent, segments) in enumerate(CODE_LINES):
            line = CodeLine(segments, theme, size="small")
            line.move_to(
                [
                    CODE_X + indent * CODE_INDENT,
                    CODE_TOP - index * CODE_STEP,
                    0,
                ],
                aligned_edge=LEFT,
            )
            lines.add(line)

        max_width = RIGHT_PANEL_WIDTH - 0.36
        if lines.width > max_width:
            lines.scale_to_fit_width(max_width)
            lines.shift([CODE_X - lines.get_left()[0], 0.0, 0.0])

        return VGroup(background, lines)

    def _silu_panel(self) -> VGroup:
        """Build the SiLU formula panel used after the SwiGLU rename."""
        theme = self.theme
        background = RoundedRectangle(
            width=RIGHT_PANEL_WIDTH,
            height=SILU_PANEL_HEIGHT,
            corner_radius=0.12,
            stroke_color=theme.color("border"),
            stroke_width=1.0,
            fill_color=theme.color("panel"),
            fill_opacity=0.55,
        ).move_to([RIGHT_X, SILU_PANEL_Y, 0])

        title = Text(
            SILU_PANEL_TITLE,
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("compute"),
        ).move_to([RIGHT_X, -0.15, 0])

        formula = MathTex(
            SILU_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("text"),
        ).move_to([RIGHT_X, -0.56, 0])
        formula.set_color_by_tex("x", theme.color("activation"))

        sigmoid = MathTex(
            SIGMOID_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("text"),
        ).move_to([RIGHT_X, -1.03, 0])
        sigmoid.set_color_by_tex("x", theme.color("activation"))

        note = Text(
            SILU_NOTE,
            font_size=theme.size("micro") + 1,
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([RIGHT_X, -1.40, 0])

        return VGroup(background, title, formula, sigmoid, note)

    def _footer(self) -> VGroup:
        """Build the total-cost footer and the concrete Llama dimensions."""
        theme = self.theme
        cost = MathTex(
            TOTAL_COST,
            font_size=theme.size("small"),
            color=theme.color("complexity"),
        ).move_to([0.0, -2.12, 0])
        concrete = MathTex(
            CONCRETE_NOTE,
            font_size=theme.size("micro") + 1,
            color=theme.color("muted"),
        ).move_to([0.0, -2.40, 0])
        return VGroup(cost, concrete)

    def _notation(self) -> VGroup:
        """Build the left-hand notation legend."""
        theme = self.theme
        symbol_x = -6.85
        desc_x = -6.10
        title_y = 2.95
        row_start = 2.62
        row_gap = 0.29

        heading = Text(
            NOTATION_TITLE,
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([symbol_x, title_y, 0], aligned_edge=LEFT)

        rows = VGroup()
        for index, (symbol_tex, description, color_key) in enumerate(
            NOTATION_ROWS
        ):
            y = row_start - index * row_gap
            symbol = MathTex(
                symbol_tex,
                font_size=theme.size("help"),
                color=theme.color(color_key),
            ).move_to([symbol_x, y, 0], aligned_edge=LEFT)
            label = Text(
                description,
                font_size=theme.size("help") - 1,
                font=theme.font,
                color=theme.color("muted"),
            ).move_to([desc_x, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(symbol, label))

        return VGroup(heading, rows)

    # --- Act 1: block drawing ---------------------------------------------
    def _play_stage_intro(self):
        """Draw the gated MLP block branch by branch with its shapes."""
        theme = self.theme

        self.play(
            FadeIn(self.title_m, shift=DOWN * 0.15),
            FadeIn(self.legend),
            Create(self.frame),
            FadeIn(self.code_panel),
            run_time=0.9,
        )
        self.play(
            FadeIn(self.input_badge, shift=DOWN * 0.08),
            run_time=0.45,
        )
        self.play(
            Indicate(
                self.input_badge,
                color=theme.color("activation"),
                scale_factor=1.03,
            ),
            run_time=0.45,
        )

        self.play(
            *[Create(arrow) for arrow in self.input_arrows],
            run_time=0.55,
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(block, shift=DOWN * 0.06)
                    for block in (self.gate_proj, self.up_proj)
                ],
                lag_ratio=0.14,
            ),
            run_time=0.75,
        )
        self.play(
            *[
                Indicate(
                    block,
                    color=theme.color("compute"),
                    scale_factor=1.03,
                )
                for block in (self.gate_proj, self.up_proj)
            ],
            run_time=0.55,
        )

        self.play(
            *[Create(arrow) for arrow in self.proj_to_cards],
            run_time=0.45,
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(card, shift=DOWN * 0.08)
                    for card in (self.gate_card, self.up_card)
                ],
                lag_ratio=0.14,
            ),
            run_time=0.65,
        )
        self.play(
            *[
                Indicate(
                    card,
                    color=theme.color("activation"),
                    scale_factor=1.03,
                )
                for card in (self.gate_card, self.up_card)
            ],
            run_time=0.5,
        )

        self.play(
            Create(self.card_to_act),
            FadeIn(self.act_badge, shift=DOWN * 0.06),
            run_time=0.6,
        )
        self.play(
            Indicate(
                self.act_badge,
                color=theme.color("compute"),
                scale_factor=1.03,
            ),
            run_time=0.5,
        )

        self.play(
            Create(self.act_to_mul),
            Create(self.up_to_mul),
            FadeIn(self.mul_badge, shift=DOWN * 0.06),
            run_time=0.7,
        )
        self.play(
            Indicate(
                self.mul_badge,
                color=theme.color("compute"),
                scale_factor=1.03,
            ),
            run_time=0.55,
        )

        self.play(
            Create(self.mul_to_down),
            FadeIn(self.down_proj, shift=DOWN * 0.06),
            run_time=0.6,
        )
        self.play(
            Indicate(
                self.down_proj,
                color=theme.color("compute"),
                scale_factor=1.03,
            ),
            run_time=0.55,
        )

        self.play(
            Create(self.down_to_output),
            FadeIn(self.output_badge, shift=DOWN * 0.08),
            Flash(
                self.output_badge,
                color=theme.color("activation"),
                flash_radius=0.50,
            ),
            run_time=0.75,
        )
        self.play(FadeIn(self.footer), run_time=0.7)

    # --- Act 2: ACT_FN -> SiLU --------------------------------------------
    def _play_swiglu(self):
        """Replace the generic activation with SiLU and rename the block."""
        theme = self.theme

        swiglu_title = self.title(
            TITLE_SWIGLU,
            x=0.0,
            y=TITLE_Y,
            size="title_small",
        )
        silu_title = MathTex(
            SILU_TITLE,
            font_size=theme.size("badge_title"),
            color=theme.color("text"),
        )
        silu_title.move_to(self.act_badge.title_label)

        self.play(
            Transform(self.title_m, swiglu_title),
            Transform(self.act_badge.title_label, silu_title),
            run_time=1.0,
        )
        self.play(
            FadeIn(self.silu_panel, shift=LEFT * 0.06),
            run_time=0.75,
        )
        self.play(
            Indicate(
                VGroup(
                    self.silu_panel[1],
                    self.silu_panel[2],
                    self.silu_panel[3],
                ),
                color=theme.color("compute"),
                scale_factor=1.02,
            ),
            run_time=0.8,
        )

    # --- Act 3: tokens are independent ------------------------------------
    def _play_token_demo(self):
        """Show element-wise products and the row correspondence X -> G."""
        theme = self.theme

        self.play(*[FadeOut(mob) for mob in self.stage_group], run_time=0.9)

        title = self.title(TOKEN_TITLE, x=0.0, y=3.72, size="title_small")

        x_grid = MatrixGrid(
            TOKEN_ROWS_COUNT,
            TOKEN_COLS_MODEL,
            theme,
            cell=TOKEN_CELL,
            gap=TOKEN_GAP,
            color_key="activation",
            fill_opacity=0.14,
        ).move_to([X_GRID_X, GRID_Y, 0])
        w_grid = MatrixGrid(
            TOKEN_COLS_MODEL,
            TOKEN_COLS_FF,
            theme,
            cell=TOKEN_CELL,
            gap=TOKEN_GAP,
            color_key="weights",
            fill_opacity=0.14,
        ).move_to([W_GRID_X, GRID_Y, 0])
        # The output grid starts almost empty and is filled element by element.
        g_grid = MatrixGrid(
            TOKEN_ROWS_COUNT,
            TOKEN_COLS_FF,
            theme,
            cell=TOKEN_CELL,
            gap=TOKEN_GAP,
            color_key="activation",
            fill_opacity=0.04,
        ).move_to([G_GRID_X, GRID_Y, 0])

        x_title = self._axis_title(
            X_LABEL,
            X_SHAPE,
            X_GRID_X,
            1.95,
            "activation",
            "activation",
        )
        w_title = self._axis_title(
            W_LABEL,
            WEIGHT_UP_SHAPE,
            W_GRID_X,
            1.95,
            "weights",
            "weights",
        )
        g_title = self._axis_title(
            G_LABEL,
            G_SHAPE,
            G_GRID_X,
            1.95,
            "activation",
            "activation",
        )

        x_row_labels = VGroup(
            *[
                MathTex(
                    TOKEN_ROWS[index],
                    font_size=theme.size("small"),
                    color=theme.color("muted"),
                ).next_to(x_grid.cells[index][0], LEFT, buff=0.16)
                for index in range(TOKEN_ROWS_COUNT)
            ]
        )
        g_row_labels = VGroup(
            *[
                MathTex(
                    TOKEN_ROWS[index],
                    font_size=theme.size("small"),
                    color=theme.color("muted"),
                ).next_to(g_grid.cells[index][-1], RIGHT, buff=0.16)
                for index in range(TOKEN_ROWS_COUNT)
            ]
        )

        x_caption = self.caption(
            X_CAPTION, x=X_GRID_X, y=-0.85, color_key="muted"
        )
        w_caption = self.caption(
            W_CAPTION, x=W_GRID_X, y=-1.00, color_key="muted"
        )
        g_caption = self.caption(
            G_CAPTION, x=G_GRID_X, y=-0.85, color_key="muted"
        )

        at_x = (x_grid.get_right()[0] + w_grid.get_left()[0]) / 2
        at_symbol = MathTex(
            r"@",
            font_size=theme.size("outro"),
            color=theme.color("compute"),
        ).move_to([at_x, GRID_Y, 0])
        eq_x = (w_grid.get_right()[0] + g_grid.get_left()[0]) / 2
        equals = MathTex(
            "=",
            font_size=theme.size("outro"),
            color=theme.color("muted"),
        ).move_to([eq_x, GRID_Y, 0])

        element_formula = MathTex(
            *ELEMENT_FORMULA_PARTS,
            font_size=theme.size("outro"),
        ).move_to([0.0, -1.65, 0])
        element_formula[0].set_color(theme.color("activation"))
        element_formula[3].set_color(theme.color("activation"))
        element_formula[5].set_color(theme.color("weights"))

        note_element = self._note(NOTE_ELEMENT, -2.35)
        note_no_mixing = self._note(NOTE_NO_MIXING, -2.85)
        note_position = self._note(NOTE_POSITION, -3.35)

        # --- Introduction of the three matrices -------------------------
        self.play(
            FadeIn(title, shift=DOWN * 0.15),
            run_time=0.7,
        )
        self.play(
            *x_grid.reveal(),
            FadeIn(x_title),
            FadeIn(x_row_labels),
            FadeIn(x_caption),
            run_time=0.9,
        )
        self.play(
            FadeIn(at_symbol),
            *w_grid.reveal(),
            FadeIn(w_title),
            FadeIn(w_caption),
            run_time=0.9,
        )
        self.play(
            Indicate(
                w_grid.grid,
                color=theme.color("weights"),
                scale_factor=1.02,
            ),
            run_time=0.6,
        )
        self.play(
            FadeIn(equals),
            *g_grid.reveal(),
            FadeIn(g_title),
            FadeIn(g_row_labels),
            FadeIn(g_caption),
            run_time=0.85,
        )
        self.play(Write(element_formula), run_time=0.7)

        # --- Element-by-element products for one row --------------------
        x_rows = [x_grid.row(index) for index in range(TOKEN_ROWS_COUNT)]
        g_rows = [g_grid.row(index) for index in range(TOKEN_ROWS_COUNT)]
        w_cols = [
            VGroup(
                *[
                    w_grid.cells[row][column]
                    for row in range(TOKEN_COLS_MODEL)
                ]
            )
            for column in range(TOKEN_COLS_FF)
        ]

        # Compute one full output row. Once the row is complete, the remaining
        # rows are shown as completed outputs and the scene switches to the
        # row-correspondence view.
        demo_row = 1
        for column_index in range(TOKEN_COLS_FF):
            frame_x = SurroundingRectangle(
                x_rows[demo_row],
                color=theme.color("activation"),
                buff=0.05,
                stroke_width=1.4,
            )
            frame_w = SurroundingRectangle(
                w_cols[column_index],
                color=theme.color("weights"),
                buff=0.05,
                stroke_width=1.4,
            )
            frame_g = SurroundingRectangle(
                g_grid.cells[demo_row][column_index],
                color=theme.color("activation"),
                buff=0.04,
                stroke_width=1.8,
            )
            dim_rows = [
                x_rows[row].animate.set_opacity(
                    1.0 if row == demo_row else 0.25
                )
                for row in range(TOKEN_ROWS_COUNT)
            ]
            dim_columns = [
                w_cols[column].animate.set_opacity(
                    1.0 if column == column_index else 0.25
                )
                for column in range(TOKEN_COLS_FF)
            ]
            self.play(
                Create(frame_x),
                Create(frame_w),
                Create(frame_g),
                *dim_rows,
                *dim_columns,
                run_time=0.55,
            )
            self.play(
                g_grid.cells[demo_row][column_index].animate.set_fill(
                    opacity=0.52
                ),
                Indicate(
                    x_rows[demo_row],
                    color=theme.color("activation"),
                    scale_factor=1.0,
                ),
                Indicate(
                    w_cols[column_index],
                    color=theme.color("weights"),
                    scale_factor=1.0,
                ),
                run_time=0.55,
            )
            self.wait(0.20)
            self.play(
                FadeOut(frame_x),
                FadeOut(frame_w),
                FadeOut(frame_g),
                run_time=0.25,
            )

        self.play(
            *[
                x_rows[row].animate.set_opacity(1.0)
                for row in range(TOKEN_ROWS_COUNT)
            ],
            *[
                w_cols[column].animate.set_opacity(1.0)
                for column in range(TOKEN_COLS_FF)
            ],
            run_time=0.40,
        )
        # The other rows use the same operation; reveal them as a completed
        # matrix rather than repeating the element-wise animation four times.
        self.play(
            *[
                cell.animate.set_fill(opacity=0.42)
                for row in g_grid.cells
                for cell in row
            ],
            run_time=0.60,
        )

        # --- State why this is position-wise ---------------------------
        self.play(Write(note_element), run_time=0.55)
        self.play(Write(note_no_mixing), run_time=0.55)
        self.play(Write(note_position), run_time=0.65)

        # --- Spread the tokens and connect X rows to G rows ------------
        self.play(
            FadeOut(element_formula),
            FadeOut(at_symbol),
            FadeOut(equals),
            FadeOut(w_grid),
            FadeOut(w_title),
            FadeOut(w_caption),
            run_time=0.65,
        )

        spread = [
            (1.5 - index) * SPREAD_STEP
            for index in range(TOKEN_ROWS_COUNT)
        ]
        self.play(
            *[
                x_rows[index].animate.shift(UP * spread[index])
                for index in range(TOKEN_ROWS_COUNT)
            ],
            *[
                g_rows[index].animate.shift(UP * spread[index])
                for index in range(TOKEN_ROWS_COUNT)
            ],
            *[
                x_row_labels[index].animate.shift(UP * spread[index])
                for index in range(TOKEN_ROWS_COUNT)
            ],
            *[
                g_row_labels[index].animate.shift(UP * spread[index])
                for index in range(TOKEN_ROWS_COUNT)
            ],
            x_title.animate.shift(UP * 0.08),
            g_title.animate.shift(UP * 0.08),
            x_caption.animate.shift(DOWN * 0.10),
            g_caption.animate.shift(DOWN * 0.10),
            run_time=0.9,
        )

        token_arrows = VGroup()
        for index in range(TOKEN_ROWS_COUNT):
            y = x_rows[index].get_center()[1]
            token_arrows.add(
                Arrow(
                    [x_grid.get_right()[0] + 0.10, y, 0],
                    [g_grid.get_left()[0] - 0.10, y, 0],
                    buff=0.0,
                    stroke_width=1.2,
                    color=theme.color("muted"),
                    tip_length=0.08,
                    max_tip_length_to_length_ratio=0.08,
                )
            )
        same_token_note = MathTex(
            r"\text{same token } i",
            font_size=theme.size("small"),
            color=theme.color("muted"),
        ).move_to([0.0, 1.95, 0])
        self.play(
            LaggedStart(
                *[Create(arrow) for arrow in token_arrows],
                lag_ratio=0.12,
            ),
            FadeIn(same_token_note),
            run_time=0.9,
        )

        self.token_group = VGroup(
            title,
            x_grid,
            g_grid,
            x_title,
            g_title,
            x_row_labels,
            g_row_labels,
            x_caption,
            g_caption,
            note_element,
            note_no_mixing,
            note_position,
            token_arrows,
            same_token_note,
        )

    # --- Finale -----------------------------------------------------------
    def _play_summary(self):
        """Restore the block and show the per-token formula."""
        theme = self.theme

        self.play(
            *[FadeOut(mob) for mob in self.token_group],
            run_time=0.8,
        )
        self.play(
            FadeIn(self.block_group),
            FadeIn(self.title_m),
            FadeIn(self.legend),
            run_time=1.0,
        )

        headline = self.caption(
            SUMMARY_HEADLINE,
            x=0.0,
            y=-1.98,
            color_key="text",
            size="small",
        )
        self.play(FadeIn(headline), run_time=0.6)
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            self.play(
                Indicate(
                    projection,
                    color=theme.color("weights"),
                    scale_factor=1.04,
                ),
                run_time=0.55,
            )

        token_formula = MathTex(
            TOKEN_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("text"),
            substrings_to_isolate=tuple(
                substring for substring, _ in TOKEN_FORMULA_HIGHLIGHTS
            ),
        )
        for substring, color_key in TOKEN_FORMULA_HIGHLIGHTS:
            token_formula.set_color_by_tex(
                substring,
                theme.color(color_key),
            )
        token_formula.move_to([0.0, -2.80, 0])
        if token_formula.width > 12.4:
            token_formula.scale_to_fit_width(12.4)

        self.play(Write(token_formula), run_time=0.9)
        self.wait(2.0)

    # --- Helpers ----------------------------------------------------------
    def _axis_title(
        self,
        symbol: str,
        shape: str,
        x: float,
        y: float,
        symbol_key: str,
        shape_key: str,
    ) -> VGroup:
        """Return a two-part title: a matrix name and its shape."""
        theme = self.theme
        group = VGroup(
            MathTex(
                symbol,
                font_size=theme.size("outro"),
                color=theme.color(symbol_key),
            ),
            MathTex(
                shape,
                font_size=theme.size("small"),
                color=theme.color(shape_key),
            ),
        ).arrange(RIGHT, buff=0.12)
        group.move_to([x, y, 0])
        return group

    def _note(self, tex: str, y: float) -> MathTex:
        """Return a centered one-line explanatory formula."""
        note = MathTex(
            tex,
            font_size=self.theme.size("outro"),
            color=self.theme.color("text"),
        ).move_to([0.0, y, 0])
        if note.width > 12.4:
            note.scale_to_fit_width(12.4)
        return note
