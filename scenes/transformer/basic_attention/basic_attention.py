"""Causal multi-head attention as an explicit computation graph.

The scene starts from ``(B, S, d_model)``, branches into Q/K/V projections,
applies RoPE to Q and K only, shows the per-head scaled dot-product attention
with a causal mask, and ends with the concatenated heads and the output
projection.

Color grammar:
    - purple outlines: computation blocks;
    - green outlines and shapes: activations and tensor dimensions;
    - yellow symbols: learned weight matrices;
    - purple cost labels: asymptotic complexity;
    - white: the causal-mask callout and structural annotations.

Run with:

    python scripts/render.py basic_attention -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Arrow,
    Create,
    FadeIn,
    Flash,
    Indicate,
    LaggedStart,
    MathTex,
    RoundedRectangle,
    Text,
    VGroup,
    Write,
)

from schematics import (
    Badge,
    CausalMaskGrid,
    SchematicScene,
    route_arrow,
)
from scenes.transformer.basic_attention.specs import (
    CONCAT,
    CONCAT_COST,
    CONCAT_NOTE,
    HEAD_FORMULA,
    INPUT_CARD,
    INPUT_SHAPE,
    K_CARD,
    MASK_BROADCAST,
    MASK_COL_AXIS,
    MASK_FORMULA,
    MASK_LEGEND_BLOCK,
    MASK_LEGEND_KEEP,
    MASK_ROW_AXIS,
    MASK_SHAPE,
    MASK_SIZE,
    MASK_SYMBOL,
    MASK_TITLE,
    NOTATION_ROWS,
    NOTATION_TITLE,
    OUTPUT_CARD,
    OUTPUT_PROJ,
    OUTPUT_COST,
    PAPER,
    PROJECTIONS,
    PROJECTION_COST,
    Q_CARD,
    ROPE_COST,
    ROPE_K,
    ROPE_Q,
    SCORES,
    SCORE_COST,
    SOFTMAX,
    SOFTMAX_COST,
    SPLIT_HEADS_LABEL,
    TITLE,
    TOTAL_COST,
    V_CARD,
    WEIGHTED,
)

# Vertical centers of the graph rows.
TITLE_Y = 3.82
INPUT_Y = 3.16
PROJ_Y = 2.40
CARD_Y = 1.64
ROPE_Y = 0.88
SCORES_Y = 0.14
SOFTMAX_Y = -0.56
WEIGHTED_Y = -1.26
CONCAT_Y = -1.96
OUTPUT_PROJ_Y = -2.66
OUTPUT_Y = -3.32
FOOTER_Y = -3.80

BRANCH_X = (-3.55, 0.0, 3.55)
CORE_X = -1.85
STACK_X = -0.05
BLOCK_HEIGHT = 0.50

INPUT_WIDTH = 2.40
PROJ_WIDTH = 2.10
ROPE_WIDTH = 2.20
CARD_WIDTH = 1.85
SCORES_WIDTH = 3.10
SOFTMAX_WIDTH = 4.55
WEIGHTED_WIDTH = 3.20
CONCAT_WIDTH = 4.35
OUTPUT_PROJ_WIDTH = 3.20
OUTPUT_WIDTH = 2.40

# Cost labels sit in the gaps between neighboring blocks.
PROJ_COST_X = (-1.78, 1.78, 5.32)
ROPE_COST_X = (-1.78, 1.78)

# Thin white frame around the split Q/K/V activations. The arrows stop at
# the frame instead of crossing it.
QKV_FRAME_PAD = 0.06

# Causal-mask callout on the right, beside the attention core. Its left edge
# stays right of the vertical V route at x = 3.55 so the route never crosses
# the callout.
MASK_CENTER = (5.40, -0.05, 0)
MASK_WIDTH = 3.20
MASK_HEIGHT = 2.05
MASK_GRID_X = 0.92
MASK_GRID_Y = -0.05
MASK_FORMULA_X = -0.66
MASK_FORMULA_Y = -0.12


class ClassicalAttention(SchematicScene):
    """Animated block diagram of causal multi-head attention."""

    def construct(self):
        """Build the graph stage by stage, annotating shapes and costs."""
        theme = self.theme

        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        legend = self._notation()

        # --- Tensor entry and Q/K/V projections ---------------------------
        input_card = self._badge(INPUT_CARD, INPUT_WIDTH, 0.0, INPUT_Y)

        projection_blocks = [
            self._badge(spec, PROJ_WIDTH, x, PROJ_Y)
            for spec, x in zip(PROJECTIONS, BRANCH_X)
        ]
        projection_costs = [
            self.math_caption(
                PROJECTION_COST,
                x=cost_x,
                y=PROJ_Y,
                color_key="compute",
            )
            for cost_x in PROJ_COST_X
        ]

        q_card = self._badge(Q_CARD, CARD_WIDTH, BRANCH_X[0], CARD_Y)
        k_card = self._badge(K_CARD, CARD_WIDTH, BRANCH_X[1], CARD_Y)
        v_card = self._badge(V_CARD, CARD_WIDTH, BRANCH_X[2], CARD_Y)
        qkv_cards = (q_card, k_card, v_card)

        # Thin white frame groups the three activations that were split into
        # heads. It sits just outside the cards and stops the arrows cleanly.
        qkv_frame = RoundedRectangle(
            width=qkv_cards[2].get_right()[0]
            - qkv_cards[0].get_left()[0]
            + 2 * QKV_FRAME_PAD,
            height=BLOCK_HEIGHT + 2 * QKV_FRAME_PAD,
            corner_radius=0.10,
            stroke_color=theme.color("text"),
            stroke_width=1.1,
            fill_opacity=0,
        ).move_to([0.0, CARD_Y, 0])

        split_caption = MathTex(
            SPLIT_HEADS_LABEL,
            font_size=theme.size("caption"),
            color=theme.color("text"),
        ).move_to([5.55, CARD_Y, 0])

        # --- RoPE on Q and K only -----------------------------------------
        rope_q = self._badge(ROPE_Q, ROPE_WIDTH, BRANCH_X[0], ROPE_Y)
        rope_k = self._badge(ROPE_K, ROPE_WIDTH, BRANCH_X[1], ROPE_Y)
        rope_blocks = (rope_q, rope_k)
        rope_costs = [
            self.math_caption(
                ROPE_COST,
                x=cost_x,
                y=ROPE_Y,
                color_key="compute",
            )
            for cost_x in ROPE_COST_X
        ]

        # --- Attention core and output -----------------------------------
        scores = self._badge(SCORES, SCORES_WIDTH, CORE_X, SCORES_Y)
        scores_cost = self.math_caption(
            SCORE_COST, x=1.00, y=SCORES_Y, color_key="compute"
        )

        softmax = self._badge(SOFTMAX, SOFTMAX_WIDTH, CORE_X, SOFTMAX_Y)
        softmax_cost = self.math_caption(
            SOFTMAX_COST, x=1.40, y=SOFTMAX_Y, color_key="compute"
        )

        weighted = self._badge(WEIGHTED, WEIGHTED_WIDTH, STACK_X, WEIGHTED_Y)
        concat = self._badge(CONCAT, CONCAT_WIDTH, STACK_X, CONCAT_Y)
        concat_cost = self.math_caption(
            CONCAT_COST, x=4.35, y=CONCAT_Y, color_key="compute"
        )

        # After concatenation the head axis disappears again: the annotation
        # on the right states the merged shape explicitly.
        concat_note = VGroup(
            Text(
                CONCAT_NOTE,
                font_size=theme.size("caption"),
                font=theme.font,
                color=theme.color("text"),
            ),
            MathTex(
                INPUT_SHAPE,
                font_size=theme.size("badge_shape"),
                color=theme.color("activation"),
            ),
        ).arrange(DOWN, buff=0.03)
        concat_note.next_to(concat, RIGHT, buff=0.30)

        # Causal-mask callout: the mask is broadcast over batch and heads, so
        # its shape starts with two singleton dimensions.
        mask_panel, mask_grid = self._causal_mask_panel()

        output_proj = self._badge(
            OUTPUT_PROJ, OUTPUT_PROJ_WIDTH, STACK_X, OUTPUT_PROJ_Y
        )
        output_proj_cost = self.math_caption(
            OUTPUT_COST,
            x=2.40,
            y=OUTPUT_PROJ_Y,
            color_key="compute",
        )

        output_card = self._badge(OUTPUT_CARD, OUTPUT_WIDTH, STACK_X, OUTPUT_Y)

        # --- Edges --------------------------------------------------------
        input_to_projections = VGroup(
            self._arrow(
                [-INPUT_WIDTH / 2, INPUT_Y, 0],
                [BRANCH_X[0], PROJ_Y + BLOCK_HEIGHT / 2, 0],
            ),
            self._arrow(
                [0, INPUT_Y - BLOCK_HEIGHT / 2, 0],
                [0, PROJ_Y + BLOCK_HEIGHT / 2, 0],
            ),
            self._arrow(
                [INPUT_WIDTH / 2, INPUT_Y, 0],
                [BRANCH_X[2], PROJ_Y + BLOCK_HEIGHT / 2, 0],
            ),
        )
        projection_to_cards = VGroup(
            *[
                self._arrow(
                    [x, PROJ_Y - BLOCK_HEIGHT / 2, 0],
                    [x, CARD_Y + BLOCK_HEIGHT / 2 + QKV_FRAME_PAD, 0],
                )
                for x in BRANCH_X
            ]
        )

        cards_to_rope = VGroup(
            *[
                self._arrow(
                    [x, CARD_Y - BLOCK_HEIGHT / 2 - QKV_FRAME_PAD, 0],
                    [x, ROPE_Y + BLOCK_HEIGHT / 2, 0],
                )
                for x in BRANCH_X[:2]
            ]
        )
        rope_to_scores = VGroup(
            self._arrow(
                [BRANCH_X[0], ROPE_Y - BLOCK_HEIGHT / 2, 0],
                [CORE_X - 0.60, SCORES_Y + BLOCK_HEIGHT / 2, 0],
            ),
            self._arrow(
                [BRANCH_X[1], ROPE_Y - BLOCK_HEIGHT / 2, 0],
                [CORE_X + 0.60, SCORES_Y + BLOCK_HEIGHT / 2, 0],
            ),
        )

        scores_to_softmax = self._arrow(
            [CORE_X, SCORES_Y - BLOCK_HEIGHT / 2, 0],
            [CORE_X, SOFTMAX_Y + BLOCK_HEIGHT / 2, 0],
        )
        softmax_to_weighted = self._arrow(
            [CORE_X, SOFTMAX_Y - BLOCK_HEIGHT / 2, 0],
            [STACK_X - 0.45, WEIGHTED_Y + BLOCK_HEIGHT / 2, 0],
        )
        v_to_weighted = route_arrow(
            [
                [BRANCH_X[2], CARD_Y - BLOCK_HEIGHT / 2 - QKV_FRAME_PAD, 0],
                [BRANCH_X[2], WEIGHTED_Y, 0],
                [STACK_X + WEIGHTED_WIDTH / 2, WEIGHTED_Y, 0],
            ],
            theme,
            color_key="muted",
            stroke_width=theme.arrow_stroke,
            tip_length=0.09,
        )
        weighted_to_concat = self._arrow(
            [STACK_X, WEIGHTED_Y - BLOCK_HEIGHT / 2, 0],
            [STACK_X, CONCAT_Y + BLOCK_HEIGHT / 2, 0],
        )
        concat_to_proj = self._arrow(
            [STACK_X, CONCAT_Y - BLOCK_HEIGHT / 2, 0],
            [STACK_X, OUTPUT_PROJ_Y + BLOCK_HEIGHT / 2, 0],
        )
        proj_to_output = self._arrow(
            [STACK_X, OUTPUT_PROJ_Y - BLOCK_HEIGHT / 2, 0],
            [STACK_X, OUTPUT_Y + BLOCK_HEIGHT / 2, 0],
        )

        # --- Per-head formula and summary ---------------------------------
        head_formula = MathTex(
            HEAD_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("text"),
        ).move_to([-4.35, WEIGHTED_Y, 0])
        self._fit_width(head_formula, 3.60)

        footer = VGroup(
            MathTex(
                TOTAL_COST,
                font_size=theme.size("small"),
                color=theme.color("compute"),
            ),
            Text(
                PAPER,
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("muted"),
            ),
        ).arrange(RIGHT, buff=0.70)
        footer.move_to([0, FOOTER_Y, 0])
        self._fit_width(footer, 13.2)

        # --- Animation ----------------------------------------------------
        self.play(
            FadeIn(title, shift=DOWN * 0.15),
            FadeIn(legend),
            run_time=0.8,
        )

        self.play(FadeIn(input_card, shift=DOWN * 0.08), run_time=0.5)
        self.play(
            Indicate(input_card, color=theme.color("activation"), scale_factor=1.03),
            run_time=0.5,
        )

        self.play(*[Create(arrow) for arrow in input_to_projections], run_time=0.7)
        self.play(
            LaggedStart(
                *[FadeIn(block, shift=DOWN * 0.06) for block in projection_blocks],
                lag_ratio=0.12,
            ),
            *[FadeIn(cost) for cost in projection_costs],
            run_time=0.8,
        )
        self.play(
            *[
                Indicate(block, color=theme.color("compute"), scale_factor=1.03)
                for block in projection_blocks
            ],
            run_time=0.7,
        )

        self.play(*[Create(arrow) for arrow in projection_to_cards], run_time=0.5)
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=DOWN * 0.08) for card in qkv_cards],
                lag_ratio=0.12,
            ),
            Create(qkv_frame),
            FadeIn(split_caption, shift=LEFT * 0.05),
            run_time=0.8,
        )
        self.play(
            *[
                Indicate(card, color=theme.color("activation"), scale_factor=1.03)
                for card in qkv_cards
            ],
            run_time=0.6,
        )
        # The per-head formula belongs to the split activations, so it appears
        # right after Q/K/V and no longer points at the output block.
        self.play(Write(head_formula), run_time=0.8)

        self.play(*[Create(arrow) for arrow in cards_to_rope], run_time=0.5)
        self.play(
            LaggedStart(
                *[FadeIn(block, shift=DOWN * 0.06) for block in rope_blocks],
                lag_ratio=0.15,
            ),
            *[FadeIn(cost) for cost in rope_costs],
            run_time=0.8,
        )
        self.play(
            *[
                Indicate(block, color=theme.color("compute"), scale_factor=1.03)
                for block in rope_blocks
            ],
            run_time=0.6,
        )

        self.play(
            Create(rope_to_scores),
            FadeIn(scores, shift=DOWN * 0.06),
            FadeIn(scores_cost),
            run_time=0.8,
        )
        self.play(
            Indicate(scores, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.6,
        )

        self.play(
            Create(scores_to_softmax),
            FadeIn(softmax, shift=DOWN * 0.06),
            FadeIn(softmax_cost),
            run_time=0.6,
        )
        self.play(
            Indicate(softmax, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.6,
        )
        self.play(FadeIn(mask_panel, shift=LEFT * 0.06), run_time=0.7)
        self.play(*mask_grid.reveal(), run_time=0.9)

        self.play(
            Create(softmax_to_weighted),
            Create(v_to_weighted),
            FadeIn(weighted, shift=DOWN * 0.06),
            run_time=1.0,
        )
        self.play(
            Indicate(weighted, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.6,
        )

        self.play(
            Create(weighted_to_concat),
            FadeIn(concat, shift=DOWN * 0.06),
            FadeIn(concat_cost),
            FadeIn(concat_note, shift=LEFT * 0.05),
            run_time=0.7,
        )
        self.play(
            Indicate(concat, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.6,
        )

        self.play(
            Create(concat_to_proj),
            FadeIn(output_proj, shift=DOWN * 0.06),
            FadeIn(output_proj_cost),
            run_time=0.6,
        )
        self.play(
            Indicate(output_proj, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.6,
        )

        self.play(
            Create(proj_to_output),
            FadeIn(output_card, shift=DOWN * 0.08),
            Flash(
                output_card,
                color=theme.color("activation"),
                flash_radius=0.55,
            ),
            run_time=0.8,
        )

        self.play(FadeIn(footer), run_time=0.8)
        self.wait(2)

    def _causal_mask_panel(self) -> tuple[VGroup, CausalMaskGrid]:
        """Build the causal-mask callout beside the attention core.

        The callout explains the mask added inside the softmax block: its
        broadcast shape, its piecewise definition, and a 5x5 visualization.
        The grid is returned separately so it can be revealed row by row.

        Returns:
            A tuple ``(static_parts, grid)``. ``static_parts`` contains the
            background, title, shape, formula, axes, legend, and the
            broadcast note.
        """
        theme = self.theme

        background = RoundedRectangle(
            width=MASK_WIDTH,
            height=MASK_HEIGHT,
            corner_radius=0.12,
            stroke_color=theme.color("border"),
            stroke_width=1.0,
            fill_color=theme.color("panel"),
            fill_opacity=0.55,
        )

        title = VGroup(
            Text(
                MASK_TITLE,
                font_size=theme.size("badge_title"),
                font=theme.font,
                color=theme.color("text"),
            ),
            MathTex(
                MASK_SYMBOL,
                font_size=theme.size("badge_title"),
                color=theme.color("text"),
            ),
        ).arrange(RIGHT, buff=0.12)
        title.move_to([0.0, 0.76, 0])
        shape = MathTex(
            MASK_SHAPE,
            font_size=theme.size("badge_shape"),
            color=theme.color("activation"),
        ).move_to([0.0, 0.50, 0])

        formula = MathTex(
            MASK_FORMULA,
            font_size=theme.size("help"),
            color=theme.color("text"),
        ).move_to([MASK_FORMULA_X, MASK_FORMULA_Y, 0])
        formula.set_color_by_tex("0", theme.color("activation"))
        formula.set_color_by_tex(r"-\infty", theme.color("muted"))

        grid = CausalMaskGrid(
            MASK_SIZE,
            theme,
            cell=0.17,
            gap=0.03,
        ).move_to([MASK_GRID_X, MASK_GRID_Y, 0])

        row_axis = MathTex(
            MASK_ROW_AXIS,
            font_size=theme.size("small"),
            color=theme.color("muted"),
        ).move_to([grid.get_left()[0] - 0.13, grid.get_center()[1], 0])
        col_axis = MathTex(
            MASK_COL_AXIS,
            font_size=theme.size("small"),
            color=theme.color("muted"),
        ).move_to([grid.get_center()[0], grid.get_bottom()[1] - 0.13, 0])

        keep = VGroup(
            RoundedRectangle(
                width=0.13,
                height=0.13,
                corner_radius=0.025,
                stroke_color=theme.color("activation"),
                stroke_width=0.8,
                fill_color=theme.color("activation"),
                fill_opacity=0.45,
            ),
            MathTex(
                MASK_LEGEND_KEEP,
                font_size=theme.size("small"),
                color=theme.color("activation"),
            ),
        ).arrange(RIGHT, buff=0.06)
        blocked = VGroup(
            RoundedRectangle(
                width=0.13,
                height=0.13,
                corner_radius=0.025,
                stroke_color=theme.color("muted"),
                stroke_width=0.8,
                fill_color=theme.color("muted"),
                fill_opacity=0.06,
            ),
            MathTex(
                MASK_LEGEND_BLOCK,
                font_size=theme.size("small"),
                color=theme.color("muted"),
            ),
        ).arrange(RIGHT, buff=0.06)
        legend = VGroup(keep, blocked).arrange(RIGHT, buff=0.30)
        legend.move_to([MASK_FORMULA_X, -0.64, 0])

        broadcast = MathTex(
            MASK_BROADCAST,
            font_size=theme.size("small"),
            color=theme.color("muted"),
        ).move_to([0.0, -0.88, 0])

        static_parts = VGroup(
            background,
            title,
            shape,
            formula,
            legend,
            broadcast,
            row_axis,
            col_axis,
        ).shift(MASK_CENTER)
        grid.shift(MASK_CENTER)

        return static_parts, grid

    def _badge(self, spec, width: float, x: float, y: float) -> Badge:
        """Create a graph badge centered at ``(x, y)``."""
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

    def _notation(self) -> VGroup:
        """Build the left-hand notation legend.

        Keeping the legend on the left frees the full width above the graph
        for the title and the Q/K/V branches.
        """
        theme = self.theme
        symbol_x = -6.80
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

    @staticmethod
    def _fit_width(mobject, max_width: float):
        """Scale ``mobject`` down to ``max_width`` if it is wider."""
        if mobject.width > max_width:
            mobject.scale_to_fit_width(max_width)
        return mobject
