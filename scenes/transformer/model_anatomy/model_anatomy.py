"""Anatomy of a Transformer decoder: embeddings, blocks, and lm_head.

The prompt becomes token chips, an ``Embedding`` lookup turns the token
ids into token embeddings, the embeddings enter one wide transformer
block, and the block unfolds into a stack of N identical ``Block`` units.
The stack outputs hidden states, the ``lm_head`` matmul applies
``W_lm_head``, and the final activations are the next-token logits.

The layout is one horizontal baseline: every main element shares the same
vertical center, captions share two text rows, and all connectors are
straight horizontal or vertical arrows.

Color grammar:
    - purple outlines and labels: computations and the brace label;
    - green outlines and shapes: activations and dimensions;
    - yellow symbols: learned weights (W_lm_head), matching the legend.

Run with:

    python scripts/render.py model_anatomy -q l
"""

from __future__ import annotations

import numpy as np
from manim import (
    Arrow,
    BOLD,
    Brace,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    FadeTransform,
    Indicate,
    LaggedStart,
    LEFT,
    MathTex,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    UR,
    VGroup,
)

from schematics import (
    Badge,
    MatrixGrid,
    Node,
    SchematicScene,
    TokenChip,
)
from scenes.transformer.model_anatomy.specs import (
    ACTIV_SHAPE,
    BLOCK,
    BRACE_LABEL,
    EMBED_NAME,
    HIDDEN,
    HIGHLIGHT_TOKEN,
    LM_HEAD_MATMUL,
    LM_HEAD_NAME,
    LOGITS,
    LOOKUP_FORMULA,
    LOOKUP_NAME,
    NOTATION_ROWS,
    NOTATION_TITLE,
    PRIME_MARK,
    PROMPT,
    SINGLE_BLOCK,
    TITLE,
    TOKEN_IDS,
    TOKEN_PIECES,
)

# Vertical anchors. Main elements sit on MAIN_Y; their captions use the
# same two rows everywhere, which keeps the diagram visually aligned.
TITLE_Y = 3.78
MAIN_Y = 0.0
NAME_Y = -0.60
SHAPE_Y = -0.90
IDX_Y = 0.64

# Prompt flow above the embedding lookup.
PHRASE_XY = (-5.85, 2.15, 0.0)
CHIPS_XY = (-5.85, 1.58, 0.0)
IDS_Y = 1.16

# Left section of the main row.
LOOKUP_X = -5.85
LOOKUP_W = 1.50
LOOKUP_H = 0.72
X_X = -3.83

# Transformer stack: the wide block unfolds into these positions.
SINGLE_XY = (0.55, MAIN_Y, 0.0)
BLOCK_W = 0.98
BLOCK_H = 0.72
TIGHT_X = (-2.11, -1.01, -0.14, 0.76)
SPREAD_X = (-2.11, -0.91, -0.025, 0.86)

# Right section of the main row: hidden states -> lm_head matmul -> logits.
HIDDEN_X = 2.50
MM_X = 4.625
LOGITS_X = 6.38

HIDDEN_W = 1.86
MM_W = 1.95
LOGITS_W = 1.12
BADGE_H = 0.66

# Left-hand legend sits at the bottom-left, keeping the top free for the
# prompt flow and the main row free for the architecture.
LEGEND_TITLE_Y = -1.55
LEGEND_ROW_Y = -1.85


class TransformerLLM(SchematicScene):
    """Animated horizontal pipeline: text -> embeddings -> blocks -> logits."""

    def construct(self):
        """Build the model anatomy stage by stage."""
        theme = self.theme

        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        legend = self._notation()

        # --- Prompt: text becomes tokens ----------------------------------
        phrase = Text(
            PROMPT,
            font_size=theme.size("node"),
            font=theme.font,
            color=theme.color("text"),
        ).move_to(PHRASE_XY)
        if phrase.width > 2.30:
            phrase.scale_to_fit_width(2.30)
        chips = VGroup(
            *[
                TokenChip(
                    piece,
                    theme,
                    color_key="norm",
                    size="small",
                    pad=0.05,
                )
                for piece in TOKEN_PIECES
            ]
        ).arrange(RIGHT, buff=0.08).move_to(CHIPS_XY)

        id_chips = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.28,
                    pad=0.05,
                    mono=True,
                    size="small",
                )
                for token_id in TOKEN_IDS
            ]
        ).arrange(RIGHT, buff=0.06).move_to([CHIPS_XY[0], IDS_Y, 0])

        phrase_arrow = self._arrow(
            [PHRASE_XY[0], phrase.get_bottom()[1]],
            [CHIPS_XY[0], chips.get_top()[1]],
        )

        # --- Embedding lookup, shown by name, not as a weight matrix ------
        lookup_rect = RoundedRectangle(
            width=LOOKUP_W,
            height=LOOKUP_H,
            corner_radius=theme.node_radius,
            stroke_color=theme.color("compute"),
            stroke_width=theme.node_stroke,
            fill_color=theme.color("compute"),
            fill_opacity=theme.node_fill_opacity,
        ).move_to([LOOKUP_X, MAIN_Y, 0])
        lookup_formula = MathTex(
            LOOKUP_FORMULA,
            font_size=theme.size("caption"),
            color=theme.color("text"),
        ).move_to(lookup_rect.get_center())
        if lookup_formula.width > LOOKUP_W - 0.25:
            lookup_formula.scale_to_fit_width(LOOKUP_W - 0.25)
        lookup = VGroup(lookup_rect, lookup_formula)

        lookup_arrow = self._arrow(
            [CHIPS_XY[0], id_chips.get_bottom()[1]],
            [LOOKUP_X, lookup_rect.get_top()[1]],
        )

        activations = MatrixGrid(
            5, 4,
            theme,
            color_key="activation",
            cell=0.16,
            gap=0.03,
            highlight_rows=(HIGHLIGHT_TOKEN,),
        ).move_to([X_X, MAIN_Y, 0])
        embed_arrow = self._arrow(lookup_rect.get_right(), activations.get_left())

        lookup_name = self.caption(LOOKUP_NAME, x=LOOKUP_X, y=NAME_Y)
        embed_name = self.caption(EMBED_NAME, x=X_X, y=NAME_Y)
        activ_shape = self.math_caption(
            ACTIV_SHAPE, x=X_X, y=SHAPE_Y, color_key="activation"
        )

        # The activation matrix becomes this green block once the
        # transformer appears, matching the hidden-states block after the
        # stack.
        hidden_input = Badge(HIDDEN, theme, width=HIDDEN_W, height=BADGE_H)
        hidden_input.move_to([X_X, MAIN_Y, 0])

        # --- Transformer unfolds into blocks -------------------------------
        single = Node(SINGLE_BLOCK, theme, width=3.40, height=BLOCK_H).move_to(
            SINGLE_XY
        )
        block_targets = [
            Node(BLOCK, theme, width=BLOCK_W, height=BLOCK_H).move_to(
                [x, MAIN_Y, 0]
            )
            for x in TIGHT_X
        ]
        block1 = block_targets[0]
        block2 = block_targets[1]
        block_n = block_targets[3]
        omissions = MathTex(
            r"\cdots",
            font_size=theme.size("outro"),
            color=theme.color("compute"),
        ).move_to([TIGHT_X[2], MAIN_Y, 0])

        indices = VGroup(
            *[
                MathTex(
                    label,
                    font_size=theme.size("small"),
                    color=theme.color("activation"),
                ).move_to([x, IDX_Y, 0])
                for label, x in zip(
                    ("1", "2", "N"),
                    (TIGHT_X[0], TIGHT_X[1], TIGHT_X[3]),
                )
            ]
        )

        # --- Output head: hidden states -> lm_head matmul -> logits --------
        hidden_badge = Badge(HIDDEN, theme, width=HIDDEN_W, height=BADGE_H)
        hidden_badge.move_to([HIDDEN_X, MAIN_Y, 0])
        # Prime: the stack output is a changed hidden state, not the input.
        prime_mark = Text(
            PRIME_MARK,
            font_size=theme.size("badge_title"),
            font=theme.font,
            weight=BOLD,
            color=theme.color("text"),
        )
        prime_mark.next_to(hidden_badge.title_label, UR, buff=0.0)
        prime_mark.shift(UP * 0.01)
        hidden_out = VGroup(hidden_badge, prime_mark)
        matmul_badge = Badge(
            LM_HEAD_MATMUL, theme, width=MM_W, height=BADGE_H
        ).move_to([MM_X, MAIN_Y, 0])
        matmul_name = self.math_caption(
            LM_HEAD_NAME, x=MM_X, y=NAME_Y, color_key="muted"
        )
        logits_badge = Badge(LOGITS, theme, width=LOGITS_W, height=BADGE_H)
        logits_badge.move_to([LOGITS_X, MAIN_Y, 0])

        # --- Animation ----------------------------------------------------
        self.play(
            FadeIn(title, shift=DOWN * 0.15),
            FadeIn(legend),
            run_time=0.8,
        )

        self.play(FadeIn(phrase, shift=DOWN * 0.08), run_time=0.5)
        self.play(
            Indicate(phrase, color=theme.color("text"), scale_factor=1.03),
            run_time=0.5,
        )
        self.play(
            Create(phrase_arrow),
            LaggedStart(
                *[FadeIn(chip, shift=DOWN * 0.06) for chip in chips],
                lag_ratio=0.12,
            ),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=DOWN * 0.05) for chip in id_chips],
                lag_ratio=0.10,
            ),
            run_time=0.6,
        )

        # Tokens feed the named lookup operation.
        self.play(
            Create(lookup_arrow),
            FadeIn(lookup, shift=DOWN * 0.08),
            run_time=0.7,
        )
        self.play(FadeIn(lookup_name), run_time=0.4)
        self.play(
            Indicate(lookup, color=theme.color("compute"), scale_factor=1.04),
            run_time=0.6,
        )

        # Lookup returns token embeddings.
        self.play(
            Create(embed_arrow),
            *activations.reveal(),
            run_time=0.8,
        )
        self.play(
            FadeIn(embed_name),
            FadeIn(activ_shape),
            Indicate(
                chips[HIGHLIGHT_TOKEN],
                color=theme.color("norm"),
                scale_factor=1.08,
            ),
            Indicate(
                activations.row(HIGHLIGHT_TOKEN),
                color=theme.color("activation"),
                scale_factor=1.04,
            ),
            run_time=0.7,
        )

        # One wide block. Once the Transformer appears, the embedding matrix
        # turns into a green hidden-states block, which is the same block
        # that the stack outputs later.
        embed_arrow_badge = self._arrow(
            lookup_rect.get_right(), hidden_input.get_left()
        )
        self.play(
            FadeIn(single, shift=RIGHT * 0.25),
            FadeTransform(activations, hidden_input),
            FadeOut(embed_name),
            FadeOut(activ_shape),
            FadeOut(embed_arrow),
            FadeIn(embed_arrow_badge),
            run_time=0.8,
        )
        self.play(
            Indicate(single, color=theme.color("compute"), scale_factor=1.03),
            run_time=0.5,
        )

        # The wide block becomes the first member of the stack.
        self.play(
            single.animate.move_to([TIGHT_X[0], MAIN_Y, 0]).set_opacity(0),
            FadeIn(block1, shift=RIGHT * 0.20),
            run_time=0.7,
        )
        self.play(
            LaggedStart(
                FadeIn(block2, shift=RIGHT * 0.30),
                FadeIn(omissions, shift=RIGHT * 0.30),
                FadeIn(block_n, shift=RIGHT * 0.30),
                FadeIn(indices),
                lag_ratio=0.20,
            ),
            run_time=0.9,
        )

        # Blocks shift slightly apart as the stack settles.
        self.play(
            block2.animate.move_to([SPREAD_X[1], MAIN_Y, 0]),
            omissions.animate.move_to([SPREAD_X[2], MAIN_Y, 0]),
            block_n.animate.move_to([SPREAD_X[3], MAIN_Y, 0]),
            indices[0].animate.move_to([SPREAD_X[0], IDX_Y, 0]),
            indices[1].animate.move_to([SPREAD_X[1], IDX_Y, 0]),
            indices[2].animate.move_to([SPREAD_X[3], IDX_Y, 0]),
            run_time=0.7,
        )

        stack = VGroup(block1, block2, omissions, block_n)
        self.play(
            LaggedStart(
                *[
                    Create(arrow)
                    for arrow in (
                        self._arrow(hidden_input.get_right(), block1.get_left()),
                        self._arrow(block1.get_right(), block2.get_left()),
                        self._arrow(block2.get_right(), omissions.get_left()),
                        self._arrow(omissions.get_right(), block_n.get_left()),
                    )
                ],
                lag_ratio=0.12,
            ),
            run_time=0.8,
        )

        # Stack output: hidden states (primed).
        self.play(
            FadeIn(hidden_out, shift=RIGHT * 0.25),
            Create(self._arrow(block_n.get_right(), hidden_badge.get_left())),
            run_time=0.7,
        )
        self.play(
            Indicate(
                hidden_out,
                color=theme.color("activation"),
                scale_factor=1.04,
            ),
            run_time=0.5,
        )

        # lm_head matmul: hidden states @ W_lm_head -> logits.
        self.play(
            FadeIn(matmul_badge, shift=RIGHT * 0.25),
            FadeIn(matmul_name),
            Create(self._arrow(hidden_badge.get_right(), matmul_badge.get_left())),
            run_time=0.7,
        )
        self.play(
            Indicate(
                matmul_badge,
                color=theme.color("compute"),
                scale_factor=1.04,
            ),
            run_time=0.5,
        )
        self.play(
            FadeIn(logits_badge, shift=RIGHT * 0.25),
            Create(self._arrow(matmul_badge.get_right(), logits_badge.get_left())),
            run_time=0.7,
        )
        self.play(
            Indicate(
                logits_badge,
                color=theme.color("activation"),
                scale_factor=1.04,
            ),
            run_time=0.5,
        )

        # Brace labels the repeated block group.
        brace = Brace(stack, DOWN, buff=0.10, color=theme.color("muted"))
        brace_label = MathTex(
            BRACE_LABEL,
            font_size=theme.size("small"),
            color=theme.color("compute"),
        )
        brace_label.next_to(brace, DOWN, buff=0.12)
        self.play(Create(brace), FadeIn(brace_label), run_time=0.7)

        # Closing summary: embeddings -> blocks -> output head.
        self.play(
            Indicate(
                VGroup(chips, id_chips, lookup, hidden_input),
                color=theme.color("compute"),
                scale_factor=1.02,
            ),
            run_time=0.6,
        )
        self.play(
            Indicate(
                VGroup(stack, indices, brace, brace_label),
                color=theme.color("compute"),
                scale_factor=1.02,
            ),
            run_time=0.6,
        )
        self.play(
            Indicate(
                VGroup(hidden_out, matmul_badge, logits_badge),
                color=theme.color("weights"),
                scale_factor=1.03,
            ),
            run_time=0.6,
        )

        self.wait(2)

    def _arrow(self, start, end) -> Arrow:
        """Return a neutral arrow with the shared stroke and tip size."""
        def as_point(point) -> np.ndarray:
            xy = np.asarray(point, dtype=float).flatten()[:2]
            return np.array([xy[0], xy[1], 0.0])

        return Arrow(
            as_point(start),
            as_point(end),
            buff=0.04,
            stroke_width=self.theme.arrow_stroke,
            color=self.theme.color("muted"),
            tip_length=0.075,
            max_tip_length_to_length_ratio=0.90,
        )

    def _notation(self) -> VGroup:
        """Build the left-hand notation legend.

        The legend sits on the left edge, below the main row, so it never
        competes with the prompt flow or the architecture line.
        """
        theme = self.theme
        symbol_x = -6.80
        desc_x = -6.10
        row_gap = 0.24

        heading = Text(
            NOTATION_TITLE,
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([symbol_x, LEGEND_TITLE_Y, 0], aligned_edge=LEFT)

        rows = VGroup()
        for index, (symbol_tex, description, color_key) in enumerate(
            NOTATION_ROWS
        ):
            y = LEGEND_ROW_Y - index * row_gap
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
