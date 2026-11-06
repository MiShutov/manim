"""Autoregressive generation loop of a transformer decoder.

The scene starts from the final frame of the anatomy diagram: five prompt
tokens with ids, the embedding lookup, N transformer blocks, the lm_head
matmul, and the logits. It then fixes ``B = 1`` and ``S = 5``, walks the
forward pass, shows a 10-token vocabulary with logits, appends the argmax
token to the context, and repeats until the context holds seven tokens.

Color grammar follows the rest of the project: purple for computations,
green for activations and shapes, yellow for learned weights.

Run with:

    python scripts/render.py generation_loop -q l
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
    Line,
    MathTex,
    RIGHT,
    RoundedRectangle,
    Text,
    Transform,
    UP,
    UR,
    VGroup,
    smooth,
)

from schematics import Badge, Node, SchematicScene, TokenChip, route_arrow
from scenes.transformer.generation_loop.specs import (
    GENERATED,
    LOGIT_ROUNDS,
    SEQUENCE_WORDS,
    TITLE,
    VOCAB,
)
from scenes.transformer.model_anatomy.specs import (
    BLOCK,
    BRACE_LABEL,
    HIDDEN,
    LM_HEAD_MATMUL,
    LM_HEAD_NAME,
    LOGITS,
    LOOKUP_FORMULA,
    LOOKUP_NAME,
    NOTATION_ROWS,
    NOTATION_TITLE,
    PRIME_MARK,
    TOKEN_IDS,
    TOKEN_PIECES,
)

# Everything except the title starts higher and slides down smoothly,
# leaving room for the vocabulary table above the pipeline.
DROP = 0.40

# Vertical anchors.
TITLE_Y = 3.78
SETTINGS_Y = 2.45
WORD_Y = 1.95
ID_Y = 1.25
MAIN_Y = -0.15
CAPTION_Y = -0.75
IDX_Y = MAIN_Y + 0.64
ROUTE_Y = 0.78

# Main row, identical to the anatomy scene.
LOOKUP_X = -5.85
LOOKUP_W = 1.50
LOOKUP_H = 0.72
X_X = -3.83
BLOCK_W = 0.98
BLOCK_H = 0.72
BLOCK_X = (-2.11, -0.91, -0.025, 0.86)
HIDDEN_X = 2.50
HIDDEN_W = 1.86
MM_X = 4.625
MM_W = 1.95
LOGITS_X = 6.38
LOGITS_W = 1.12
BADGE_H = 0.66

# Vocabulary table above the pipeline, inside a framed panel.
FRAME_CX = 2.05
FRAME_CY = 1.48
FRAME_W = 7.50
FRAME_H = 1.50
FRAME_RIGHT = FRAME_CX + FRAME_W / 2
CARD_W = 1.35
CARD_H = 0.52
CARD_GAP = 0.07
PANEL_COLS = 5
HEADER_Y = 2.07
DIVIDER_Y = 1.94
GRID_TOP = 1.88
VOCAB_ARROW_Y = 1.05

# Legend in the bottom-left corner, pulled up next to the diagram.
LEGEND_TITLE_Y = -1.45
LEGEND_ROW_Y = -1.75

WORD_GAP = 0.16


class VocabCard(VGroup):
    """One vocabulary entry: token text, token id, and a score slot."""

    def __init__(self, token: str, token_id: str, theme, width: float, height: float):
        """Build a card centered at the origin."""
        self.theme = theme
        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.10,
            stroke_color=theme.color("border"),
            stroke_width=1.1,
            fill_color=theme.color("panel"),
            fill_opacity=0.75,
        )
        self.token_label = Text(
            token,
            font_size=theme.size("small"),
            font=theme.font,
            weight=BOLD,
            color=theme.color("text"),
        ).move_to(self.rect.get_center() + UP * 0.14)
        self.id_label = Text(
            token_id,
            font_size=theme.size("micro"),
            font=theme.mono_font,
            color=theme.color("muted"),
        ).move_to(self.rect.get_center() + DOWN * 0.02)
        super().__init__(self.rect, self.token_label, self.id_label)
        self.score_label = None

    def score_target(self, value: float):
        """Return a score label placed in this card's score slot."""
        label = Text(
            f"{value:.1f}",
            font_size=self.theme.size("micro"),
            font=self.theme.mono_font,
            color=self.theme.color("activation"),
        )
        label.move_to(self.rect.get_center() + DOWN * 0.18)
        return label


class GenerationLoop(SchematicScene):
    """Animated autoregressive loop over the decoder anatomy."""

    def construct(self):
        """Run the prompt, then two generation rounds."""
        theme = self.theme

        # --- Static starting frame (anatomy scene, with token ids) ---------
        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        if title.width > 13.0:
            title.scale_to_fit_width(13.0)
        legend = self._notation()

        word_chips = VGroup(
            *[
                TokenChip(piece, theme, color_key="norm", size="small", pad=0.05)
                for piece in TOKEN_PIECES
            ]
        ).arrange(RIGHT, buff=0.08).move_to([LOOKUP_X, WORD_Y, 0])
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
        ).arrange(RIGHT, buff=0.06).move_to([LOOKUP_X, ID_Y, 0])

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
        lookup_name = self.caption(LOOKUP_NAME, x=LOOKUP_X, y=CAPTION_Y)
        self.lookup = lookup

        token_arrow = self._arrow(
            [LOOKUP_X, id_chips.get_bottom()[1]],
            [LOOKUP_X, lookup_rect.get_top()[1]],
        )

        hidden_in = Badge(HIDDEN, theme, width=HIDDEN_W, height=BADGE_H)
        hidden_in.move_to([X_X, MAIN_Y, 0])
        self.hidden_in = hidden_in

        block1 = Node(BLOCK, theme, width=BLOCK_W, height=BLOCK_H)
        block1.move_to([BLOCK_X[0], MAIN_Y, 0])
        block2 = Node(BLOCK, theme, width=BLOCK_W, height=BLOCK_H)
        block2.move_to([BLOCK_X[1], MAIN_Y, 0])
        block_n = Node(BLOCK, theme, width=BLOCK_W, height=BLOCK_H)
        block_n.move_to([BLOCK_X[3], MAIN_Y, 0])
        omissions = MathTex(
            r"\cdots",
            font_size=theme.size("outro"),
            color=theme.color("compute"),
        ).move_to([BLOCK_X[2], MAIN_Y, 0])
        indices = VGroup(
            *[
                MathTex(
                    label,
                    font_size=theme.size("small"),
                    color=theme.color("activation"),
                ).move_to([x, IDX_Y, 0])
                for label, x in zip(("1", "2", "N"), (BLOCK_X[0], BLOCK_X[1], BLOCK_X[3]))
            ]
        )

        hidden_out_badge = Badge(HIDDEN, theme, width=HIDDEN_W, height=BADGE_H)
        hidden_out_badge.move_to([HIDDEN_X, MAIN_Y, 0])
        self.hidden_out_badge = hidden_out_badge
        prime_mark = Text(
            PRIME_MARK,
            font_size=theme.size("badge_title"),
            font=theme.font,
            weight=BOLD,
            color=theme.color("text"),
        )
        prime_mark.next_to(hidden_out_badge.title_label, UR, buff=0.0)
        prime_mark.shift(UP * 0.01)
        hidden_out = VGroup(hidden_out_badge, prime_mark)

        matmul_badge = Badge(
            LM_HEAD_MATMUL, theme, width=MM_W, height=BADGE_H
        ).move_to([MM_X, MAIN_Y, 0])
        self.matmul_badge = matmul_badge
        matmul_name = self.math_caption(
            LM_HEAD_NAME, x=MM_X, y=CAPTION_Y, color_key="muted"
        )
        logits_badge = Badge(LOGITS, theme, width=LOGITS_W, height=BADGE_H)
        logits_badge.move_to([LOGITS_X, MAIN_Y, 0])
        self.logits_badge = logits_badge

        pipeline_arrows = VGroup(
            self._arrow(lookup_rect.get_right(), hidden_in.get_left()),
            self._arrow(hidden_in.get_right(), block1.get_left()),
            self._arrow(block1.get_right(), block2.get_left()),
            self._arrow(block2.get_right(), omissions.get_left()),
            self._arrow(omissions.get_right(), block_n.get_left()),
            self._arrow(block_n.get_right(), hidden_out_badge.get_left()),
            self._arrow(hidden_out_badge.get_right(), matmul_badge.get_left()),
            self._arrow(matmul_badge.get_right(), logits_badge.get_left()),
        )

        stack = VGroup(block1, block2, omissions, block_n)
        brace = Brace(stack, DOWN, buff=0.10, color=theme.color("muted"))
        brace_label = MathTex(
            BRACE_LABEL,
            font_size=theme.size("small"),
            color=theme.color("compute"),
        )
        brace_label.next_to(brace, DOWN, buff=0.12)

        static = VGroup(
            legend,
            word_chips,
            id_chips,
            token_arrow,
            lookup,
            lookup_name,
            hidden_in,
            *stack,
            indices,
            hidden_out,
            matmul_badge,
            matmul_name,
            logits_badge,
            pipeline_arrows,
            brace,
            brace_label,
        )
        # Start a little higher and settle down smoothly at the beginning.
        static.shift(UP * DROP)
        self.add(title, static)
        self.play(
            static.animate.shift(DOWN * DROP),
            run_time=1.0,
            rate_func=smooth,
        )
        self.wait(0.3)

        # --- Fix B = 1 and S = 5, explaining both numbers ------------------
        settings = self._settings_text(5)
        self.play(FadeIn(settings, shift=DOWN * 0.10), run_time=0.6)
        for word_chip, id_chip in zip(word_chips, id_chips):
            self.play(
                Indicate(
                    VGroup(word_chip, id_chip),
                    color=theme.color("norm"),
                    scale_factor=1.05,
                ),
                run_time=0.25,
            )

        # Symbols become concrete shapes; the S update is highlighted.
        self.play(*self._shape_anims(5), run_time=0.8)
        self.play(
            Indicate(hidden_in, color=theme.color("activation"), scale_factor=1.04),
            run_time=0.5,
        )

        # --- First forward pass, element by element -----------------------
        first_pass = (
            (id_chips, "norm"),
            (lookup, "compute"),
            (hidden_in, "activation"),
            (block1, "compute"),
            (block2, "compute"),
            (block_n, "compute"),
            (hidden_out, "activation"),
            (matmul_badge, "compute"),
            (logits_badge, "activation"),
        )
        for mobject, color_key in first_pass:
            self.play(
                Indicate(
                    mobject,
                    color=theme.color(color_key),
                    scale_factor=1.03,
                ),
                run_time=0.32,
            )

        # --- Vocabulary and logits ----------------------------------------
        cards = self._build_vocab_cards()
        frame = RoundedRectangle(
            width=FRAME_W,
            height=FRAME_H,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=0.92,
        ).move_to([FRAME_CX, FRAME_CY, 0])
        divider = Line(
            [FRAME_CX - FRAME_W / 2 + 0.18, DIVIDER_Y, 0],
            [FRAME_CX + FRAME_W / 2 - 0.18, DIVIDER_Y, 0],
            stroke_width=0.9,
            color=theme.color("border"),
            stroke_opacity=0.55,
        )
        header = self._vocab_header()
        vocab_arrow = route_arrow(
            [
                [LOGITS_X, MAIN_Y + BADGE_H / 2, 0],
                [LOGITS_X, VOCAB_ARROW_Y, 0],
                [FRAME_RIGHT + 0.03, VOCAB_ARROW_Y, 0],
            ],
            theme,
            color_key="activation",
            tip_length=0.08,
        )
        self.play(
            FadeIn(frame),
            FadeIn(divider),
            FadeIn(header),
            Create(vocab_arrow),
            LaggedStart(
                *[FadeIn(card, shift=DOWN * 0.06) for card in cards],
                lag_ratio=0.04,
            ),
            run_time=1.1,
        )

        self._reveal_scores(cards, LOGIT_ROUNDS[0])

        max_index = max(
            range(len(LOGIT_ROUNDS[0])), key=lambda i: LOGIT_ROUNDS[0][i]
        )
        self._highlight_card(cards[max_index], active=True)
        self.play(
            Indicate(
                cards[max_index].id_label,
                color=theme.color("activation"),
                scale_factor=1.25,
            ),
            run_time=0.4,
        )

        # --- Round 1: append the argmax token, grow S to 5 -> 6 ------------
        word_chips_list = list(word_chips)
        id_chips_list = list(id_chips)
        new_word, new_id, down_arrow = self._emit_token(
            *GENERATED[0],
            id_chips_list,
            cards[max_index].id_label.get_center(),
        )
        word_chips_list.append(new_word)
        id_chips_list.append(new_id)
        settings = self._advance_sequence(settings, 6, down_arrow)

        # Second forward pass.
        second_pass = (
            (block1, "compute"),
            (block2, "compute"),
            (block_n, "compute"),
            (hidden_out, "activation"),
            (matmul_badge, "compute"),
            (logits_badge, "activation"),
        )
        for mobject, color_key in second_pass:
            self.play(
                Indicate(
                    mobject,
                    color=theme.color(color_key),
                    scale_factor=1.03,
                ),
                run_time=0.32,
            )

        # --- Round 2: updated logits, append the next token ----------------
        self._update_scores(cards, LOGIT_ROUNDS[1])
        next_index = max(
            range(len(LOGIT_ROUNDS[1])), key=lambda i: LOGIT_ROUNDS[1][i]
        )
        self._highlight_card(cards[max_index], active=False)
        self._highlight_card(cards[next_index], active=True)
        self.play(
            Indicate(
                cards[next_index].id_label,
                color=theme.color("activation"),
                scale_factor=1.25,
            ),
            run_time=0.4,
        )

        new_word2, new_id2, down_arrow2 = self._emit_token(
            *GENERATED[1],
            id_chips_list,
            cards[next_index].id_label.get_center(),
        )
        word_chips_list.append(new_word2)
        id_chips_list.append(new_id2)
        settings = self._advance_sequence(settings, 7, down_arrow2)

        # --- Outro: the context now holds seven tokens ---------------------
        ellipsis = Text(
            "\u2026",
            font_size=theme.size("node"),
            font=theme.font,
            color=theme.color("muted"),
        )
        ellipsis.move_to([word_chips_list[-1].get_right()[0] + 0.20, WORD_Y, 0])
        self.play(FadeIn(ellipsis), run_time=0.4)
        self.play(
            Indicate(
                VGroup(*word_chips_list, *id_chips_list),
                color=theme.color("norm"),
                scale_factor=1.02,
            ),
            run_time=0.8,
        )
        self.wait(2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _settings_text(self, length: int):
        """Return the live ``B = 1 · S = length`` settings line."""
        tex = MathTex(
            rf"B = 1\ (\text{{one sequence}}) \qquad "
            rf"S = {length}\ (\text{{{SEQUENCE_WORDS[length]}}})",
            font_size=self.theme.size("outro"),
            color=self.theme.color("text"),
            substrings_to_isolate=[r"B", r"S"],
        )
        tex.set_color_by_tex(r"B", self.theme.color("activation"))
        tex.set_color_by_tex(r"S", self.theme.color("activation"))
        tex.move_to([0.0, SETTINGS_Y, 0])
        if tex.width > 13.0:
            tex.scale_to_fit_width(13.0)
        return tex

    def _shape_target(self, badge: Badge, tex: str):
        """Return a fitted shape label for ``badge``."""
        label = MathTex(
            tex,
            font_size=self.theme.size("badge_shape"),
            color=self.theme.color("activation"),
        )
        if label.width > badge.width - 0.32:
            label.scale_to_fit_width(badge.width - 0.32)
        label.move_to(badge.shape_label)
        return label

    def _shape_anims(self, length: int):
        """Return shape updates for all activation and logits badges."""
        hidden = rf"(1,\ {length},\ d_{{\mathrm{{model}}}})"
        vocab = rf"(1,\ {length},\ \mathrm{{vocab}})"
        return [
            Transform(
                self.hidden_in.shape_label,
                self._shape_target(self.hidden_in, hidden),
            ),
            Transform(
                self.hidden_out_badge.shape_label,
                self._shape_target(self.hidden_out_badge, hidden),
            ),
            Transform(
                self.matmul_badge.shape_label,
                self._shape_target(self.matmul_badge, vocab),
            ),
            Transform(
                self.logits_badge.shape_label,
                self._shape_target(self.logits_badge, vocab),
            ),
        ]

    def _build_vocab_cards(self):
        """Create and place the 10 vocabulary cards inside the frame."""
        cards = []
        panel_width = PANEL_COLS * CARD_W + (PANEL_COLS - 1) * CARD_GAP
        first_x = FRAME_CX - panel_width / 2 + CARD_W / 2
        first_y = GRID_TOP - CARD_H / 2
        for index, (token, token_id) in enumerate(VOCAB):
            card = VocabCard(token, token_id, self.theme, CARD_W, CARD_H)
            row, col = divmod(index, PANEL_COLS)
            card.move_to([
                first_x + col * (CARD_W + CARD_GAP),
                first_y - row * (CARD_H + CARD_GAP),
                0,
            ])
            cards.append(card)
        return cards

    def _vocab_header(self) -> VGroup:
        """Return the vocabulary panel header."""
        left = Text(
            "vocabulary",
            font_size=self.theme.size("small"),
            font=self.theme.font,
            color=self.theme.color("muted"),
        )
        right = MathTex(
            r"\mathrm{score} = \mathrm{logit}",
            font_size=self.theme.size("small"),
            color=self.theme.color("activation"),
        )
        return VGroup(left, right).arrange(RIGHT, buff=1.0).move_to(
            [FRAME_CX, HEADER_Y, 0]
        )

    def _reveal_scores(self, cards, scores):
        """Attach score labels and fade them in."""
        for card, value in zip(cards, scores):
            card.score_label = card.score_target(value)
        self.play(
            LaggedStart(
                *[FadeIn(card.score_label) for card in cards],
                lag_ratio=0.04,
            ),
            run_time=0.9,
        )

    def _update_scores(self, cards, scores):
        """Crossfade the score labels to the next logit distribution."""
        targets = [card.score_target(value) for card, value in zip(cards, scores)]
        self.play(
            LaggedStart(
                *[
                    FadeTransform(card.score_label, target)
                    for card, target in zip(cards, targets)
                ],
                lag_ratio=0.04,
            ),
            run_time=0.8,
        )
        for card, target in zip(cards, targets):
            card.score_label = target

    def _highlight_card(self, card: VocabCard, *, active: bool):
        """Highlight or unhighlight one vocabulary card."""
        if active:
            self.play(
                card.rect.animate.set_stroke(
                    color=self.theme.color("activation"), width=2.2
                ).set_fill(color=self.theme.color("activation"), opacity=0.20),
                run_time=0.4,
            )
            self.play(
                Indicate(
                    card.token_label,
                    color=self.theme.color("activation"),
                    scale_factor=1.10,
                ),
                run_time=0.35,
            )
        else:
            self.play(
                card.rect.animate.set_stroke(
                    color=self.theme.color("border"), width=1.1
                ).set_fill(color=self.theme.color("panel"), opacity=0.75),
                run_time=0.35,
            )

    def _emit_token(self, word: str, token_id: str, id_list, start_point):
        """Emerge a new id from the vocabulary table and route it inward.

        Returns the new word chip, the new id chip, and the routed arrow.
        """
        theme = self.theme
        new_word = TokenChip(word, theme, color_key="norm", size="small", pad=0.05)
        new_id = TokenChip(
            token_id,
            theme,
            color_key="muted",
            height=0.28,
            pad=0.06,
            mono=True,
            size="small",
        )
        target_x = id_list[-1].get_right()[0] + WORD_GAP + new_id.width / 2
        new_word.move_to([target_x, WORD_Y, 0])
        new_id.move_to(start_point)

        self.play(FadeIn(new_id, scale=0.8), run_time=0.25)
        self.play(
            new_id.animate.move_to([target_x, ID_Y, 0]),
            run_time=1.2,
            rate_func=smooth,
        )

        up_arrow = self._arrow(
            [target_x, new_id.get_top()[1]],
            [target_x, WORD_Y - new_word.height / 2],
        )
        self.play(Create(up_arrow), run_time=0.4)
        self.play(FadeIn(new_word, shift=UP * 0.08), run_time=0.4)
        self.play(FadeOut(up_arrow), run_time=0.25)

        down_arrow = route_arrow(
            [
                [target_x, new_id.get_bottom()[1], 0],
                [target_x, ROUTE_Y, 0],
                [LOOKUP_X, ROUTE_Y, 0],
                [LOOKUP_X, MAIN_Y + LOOKUP_H / 2, 0],
            ],
            theme,
            color_key="muted",
            stroke_width=theme.arrow_stroke,
            tip_length=0.08,
        )
        self.play(Create(down_arrow), run_time=0.7)
        self.play(
            Indicate(
                self.lookup,
                color=theme.color("compute"),
                scale_factor=1.03,
            ),
            run_time=0.35,
        )
        return new_word, new_id, down_arrow

    def _advance_sequence(self, settings, length: int, down_arrow):
        """Update shapes and the settings line, then drop the routed arrow."""
        new_settings = self._settings_text(length)
        self.play(
            *self._shape_anims(length),
            FadeOut(settings),
            FadeIn(new_settings),
            run_time=0.8,
        )
        self.play(
            Indicate(
                self.hidden_in,
                color=self.theme.color("activation"),
                scale_factor=1.05,
            ),
            FadeOut(down_arrow),
            run_time=0.5,
        )
        self.play(
            Indicate(
                self.hidden_in.shape_label,
                color=self.theme.color("activation"),
                scale_factor=1.25,
            ),
            run_time=0.45,
        )
        return new_settings

    def _notation(self) -> VGroup:
        """Build the left-hand notation legend below the main row."""
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
