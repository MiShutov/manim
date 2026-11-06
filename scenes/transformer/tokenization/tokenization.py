"""Tokenization: text to token ids and back through a vocabulary.

The left two thirds are a loop: text enters ``tokenizer.encode``, becomes
subword token chips and token ids, then ``tokenizer.decode`` turns the ids
back into the original text. The right third is the vocabulary itself, a
one-column JSON-style table that maps every token id to a string. The table
is taller than the panel and scrolls to show that the vocabulary is huge.

Run with:

    python scripts/render.py tokenization -q l
"""

from __future__ import annotations

from collections.abc import Sequence

from manim import (
    Arrow,
    Brace,
    Create,
    DOWN,
    FadeIn,
    FadeTransform,
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
    Write,
    smooth,
)

from schematics import (
    CodeLine,
    SchematicScene,
    TokenChip,
    VocabularyTable,
    connect,
)
from scenes.transformer.tokenization.specs import (
    CODE_DECODE,
    CODE_ENCODE,
    DECODE_DIRECTION,
    DECODE_LABEL,
    ENCODE_DIRECTION,
    ENCODE_LABEL,
    SCROLL_VOCAB,
    SUBWORD_SPAN,
    SUBWORD_WORD,
    TEXT,
    TITLE,
    TOKEN_IDS,
    TOKEN_PIECES,
    VOCAB,
)

# Left side: the encode/decode loop. It occupies roughly two thirds of the
# frame, leaving the right third for the vocabulary.
TITLE_Y = 3.78
LOOP_TOP_Y = 1.75
IDS_Y = 1.22
LOOP_BOTTOM_Y = 0.25
ENCODE_CODE_Y = 2.48
DECODE_CODE_Y = 0.51
DECODE_CODE_X = -2.70

CARD_WIDTH = 2.20
CARD_HEIGHT = 0.78
BLOCK_WIDTH = 2.25
BLOCK_HEIGHT = 0.82
TEXT_X = -5.85
ENCODE_X = -3.00
TOKENS_X = 0.45
DECODE_X = 0.45
IDENTITY_Y = -1.75

# Right side: a one-column vocabulary window with a scrollbar. The table is
# taller than the window; opaque masks create a hard editor-style viewport.
PANEL_CX = 4.80
PANEL_WIDTH = 4.40
PANEL_TOP = 3.48
PANEL_BOTTOM = -3.48
PANEL_HEIGHT = PANEL_TOP - PANEL_BOTTOM

PANEL_HEADER_Y = 3.18
PANEL_DIVIDER_Y = 2.95
PANEL_DIRECTION_Y = -2.64
PANEL_FOOTER_Y = -3.10
WINDOW_TOP = 2.78
WINDOW_BOTTOM = -2.38
WINDOW_CENTER_Y = (WINDOW_TOP + WINDOW_BOTTOM) / 2
WINDOW_HEIGHT = WINDOW_TOP - WINDOW_BOTTOM

CELL_WIDTH = 3.72
CELL_HEIGHT = 0.34
ROW_GAP = 0.06
TABLE_HEIGHT = (
    len(SCROLL_VOCAB) * CELL_HEIGHT
    + (len(SCROLL_VOCAB) - 1) * ROW_GAP
)
SCROLL_X = PANEL_CX + PANEL_WIDTH / 2 - 0.18


class Tokenization(SchematicScene):
    """Animate a reversible text <-> token-id round trip and its vocabulary."""

    def construct(self):
        """Build the encode/decode loop, then reveal and use the vocabulary."""
        theme = self.theme
        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        if title.width > 13.0:
            title.scale_to_fit_width(13.0)

        # --- Encode side ---------------------------------------------------
        text_a = self._text_card(TEXT, [TEXT_X, LOOP_TOP_Y, 0])
        input_caption = self.caption(
            "input text", x=TEXT_X, y=LOOP_TOP_Y + 0.53
        )

        encode_block = self._tokenizer_block(
            ENCODE_LABEL, [ENCODE_X, LOOP_TOP_Y, 0]
        )
        encode_caption = self.caption(
            "encode", x=ENCODE_X, y=LOOP_TOP_Y - 0.62, color_key="compute"
        )
        encode_code = CodeLine(CODE_ENCODE, theme, size="small")
        encode_code.move_to([ENCODE_X, ENCODE_CODE_Y, 0])
        if encode_code.width > 3.1:
            encode_code.scale_to_fit_width(3.1)

        token_chips = VGroup(
            *[
                TokenChip(
                    piece,
                    theme,
                    color_key="activation",
                    height=0.32,
                    pad=0.07,
                    size="small",
                )
                for piece in TOKEN_PIECES
            ]
        ).arrange(RIGHT, buff=0.08)
        token_chips.move_to([TOKENS_X, LOOP_TOP_Y, 0])

        id_chips = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.26,
                    pad=0.05,
                    mono=True,
                    size="small",
                )
                for token_id in TOKEN_IDS
            ]
        ).arrange(RIGHT, buff=0.05)
        id_chips.move_to([TOKENS_X, IDS_Y, 0])

        tokens_caption = self.caption(
            "tokens",
            x=0.0,
            y=LOOP_TOP_Y,
            color_key="dimension",
            size="micro",
        )
        tokens_caption.next_to(token_chips, RIGHT, buff=0.16)
        ids_caption = self.caption(
            "ids",
            x=0.0,
            y=IDS_Y,
            size="micro",
        )
        ids_caption.next_to(id_chips, RIGHT, buff=0.16)

        subword = VGroup(
            token_chips[SUBWORD_SPAN[0]],
            token_chips[SUBWORD_SPAN[1]],
        )
        subword_brace = Brace(
            subword,
            UP,
            buff=0.05,
            color=theme.color("activation"),
        )
        subword_label = Text(
            SUBWORD_WORD,
            font_size=theme.size("micro"),
            font=self.theme.font,
            color=theme.color("activation"),
        ).next_to(subword_brace, UP, buff=0.04)

        # --- Decode side ---------------------------------------------------
        text_b = self._text_card(TEXT, [TEXT_X, LOOP_BOTTOM_Y, 0])
        decoded_caption = self.caption(
            "decoded text", x=TEXT_X, y=LOOP_BOTTOM_Y - 0.57
        )

        decode_block = self._tokenizer_block(
            DECODE_LABEL, [DECODE_X, LOOP_BOTTOM_Y, 0]
        )
        decode_caption = self.caption(
            "decode", x=DECODE_X, y=LOOP_BOTTOM_Y - 0.62, color_key="compute"
        )
        decode_code = CodeLine(CODE_DECODE, theme, size="small")
        decode_code.move_to([DECODE_CODE_X, DECODE_CODE_Y, 0])
        if decode_code.width > 3.1:
            decode_code.scale_to_fit_width(3.1)

        text_to_encode = connect(
            text_a, encode_block, theme, buff=0.09, tip_length=0.09
        )
        encode_to_tokens = connect(
            encode_block, token_chips, theme, buff=0.10, tip_length=0.09
        )
        ids_to_decode = Arrow(
            id_chips.get_bottom(),
            decode_block.get_top(),
            buff=0.06,
            stroke_width=theme.arrow_stroke + 0.4,
            color=theme.color("muted"),
            tip_length=0.10,
            max_tip_length_to_length_ratio=0.30,
        )
        decode_to_text = connect(
            decode_block,
            text_b,
            theme,
            buff=0.09,
            tip_length=0.09,
            direction="horizontal",
        )

        check = MathTex(
            r"\checkmark",
            font_size=theme.size("caption"),
            color=theme.color("success"),
        )
        check_label = Text(
            "same text",
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("success"),
        )
        roundtrip = VGroup(check, check_label).arrange(RIGHT, buff=0.06)
        roundtrip.move_to([TEXT_X, (LOOP_TOP_Y + LOOP_BOTTOM_Y) / 2, 0])

        identity = MathTex(
            r"\mathrm{decode}(\mathrm{encode}(x)) = x",
            font_size=theme.size("outro"),
            color=theme.color("activation"),
        ).move_to([DECODE_CODE_X, IDENTITY_Y, 0])
        identity_caption = self.caption(
            "round-trip identity",
            x=DECODE_CODE_X,
            y=IDENTITY_Y - 0.42,
        )

        # --- Vocabulary panel ---------------------------------------------
        panel = RoundedRectangle(
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        ).move_to([PANEL_CX, 0.0, 0])

        panel_title = Text(
            "vocabulary.json",
            font_size=theme.size("small"),
            font=theme.mono_font,
            color=theme.color("muted"),
        ).move_to(
            [PANEL_CX - PANEL_WIDTH / 2 + 0.20, PANEL_HEADER_Y, 0],
            aligned_edge=LEFT,
        )
        panel_count = MathTex(
            r"|V| \approx 100{,}256",
            font_size=theme.size("small"),
            color=theme.color("dimension"),
        ).move_to(
            [PANEL_CX + PANEL_WIDTH / 2 - 0.20, PANEL_HEADER_Y, 0],
            aligned_edge=RIGHT,
        )

        divider = Line(
            [PANEL_CX - PANEL_WIDTH / 2 + 0.20, PANEL_DIVIDER_Y, 0],
            [PANEL_CX + PANEL_WIDTH / 2 - 0.20, PANEL_DIVIDER_Y, 0],
            stroke_width=0.9,
            color=theme.color("border"),
            stroke_opacity=0.55,
        )

        direction_encode = Text(
            ENCODE_DIRECTION,
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("activation"),
        ).move_to([PANEL_CX, PANEL_DIRECTION_Y, 0])
        direction_decode = Text(
            DECODE_DIRECTION,
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("activation"),
        ).move_to([PANEL_CX, PANEL_DIRECTION_Y, 0])

        more = Text(
            "... more entries",
            font_size=theme.size("micro"),
            font=theme.mono_font,
            color=theme.color("muted"),
        ).move_to([PANEL_CX, PANEL_FOOTER_Y, 0])

        # One long column; the panel shows only a window into it.
        table = VocabularyTable(
            SCROLL_VOCAB,
            theme,
            columns=1,
            cell_width=CELL_WIDTH,
            cell_height=CELL_HEIGHT,
            row_gap=ROW_GAP,
        )
        table.move_to([PANEL_CX, 0.0, 0])
        example_ids = {token_id for token_id, _ in VOCAB}
        example_indices = [
            index
            for index, (token_id, _) in enumerate(SCROLL_VOCAB)
            if token_id in example_ids
        ]
        anchor_index = (example_indices[0] + example_indices[-1]) / 2
        anchor_id = SCROLL_VOCAB[int(anchor_index)][0]
        anchor_row_y = table.entry(anchor_id).get_center()[1]
        table.shift(UP * (WINDOW_CENTER_Y - anchor_row_y))
        initial_table_center = table.get_center()

        top_aligned_center_y = WINDOW_TOP - TABLE_HEIGHT / 2
        max_scroll = max(TABLE_HEIGHT - WINDOW_HEIGHT, 1e-6)
        initial_scroll = initial_table_center[1] - top_aligned_center_y
        scroll_delta = max(0.0, min(5.6, max_scroll - initial_scroll))

        thumb_height = max(
            0.55,
            WINDOW_HEIGHT * WINDOW_HEIGHT / TABLE_HEIGHT,
        )
        thumb_travel = WINDOW_HEIGHT - thumb_height
        thumb_initial_y = (
            WINDOW_TOP
            - thumb_height / 2
            - (initial_scroll / max_scroll) * thumb_travel
        )
        thumb_delta = scroll_delta / max_scroll * thumb_travel

        scroll_track = Rectangle(
            width=0.055,
            height=WINDOW_HEIGHT,
            stroke_width=0,
            fill_color=theme.color("border"),
            fill_opacity=0.45,
        ).move_to([SCROLL_X, WINDOW_CENTER_Y, 0])
        scroll_thumb = RoundedRectangle(
            width=0.09,
            height=thumb_height,
            corner_radius=0.045,
            stroke_width=0,
            fill_color=theme.color("muted"),
            fill_opacity=0.75,
        ).move_to([SCROLL_X, thumb_initial_y, 0])

        # Hard editor-style viewport: rows outside the window are covered by
        # opaque masks. The masks outside the panel use the scene background,
        # so no table content can leak past the file frame.
        mask_width = PANEL_WIDTH - 0.08
        mask_top_inside = Rectangle(
            width=mask_width,
            height=PANEL_TOP - WINDOW_TOP,
            stroke_width=0,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        ).move_to([PANEL_CX, (PANEL_TOP + WINDOW_TOP) / 2, 0])
        mask_top_outside = Rectangle(
            width=mask_width,
            height=4.10 - PANEL_TOP,
            stroke_width=0,
            fill_color=theme.bg,
            fill_opacity=1.0,
        ).move_to([PANEL_CX, (4.10 + PANEL_TOP) / 2, 0])
        mask_bottom_inside = Rectangle(
            width=mask_width,
            height=WINDOW_BOTTOM - PANEL_BOTTOM,
            stroke_width=0,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        ).move_to([PANEL_CX, (WINDOW_BOTTOM + PANEL_BOTTOM) / 2, 0])
        mask_bottom_outside = Rectangle(
            width=mask_width,
            height=PANEL_BOTTOM + 4.10,
            stroke_width=0,
            fill_color=theme.bg,
            fill_opacity=1.0,
        ).move_to([PANEL_CX, (-4.10 + PANEL_BOTTOM) / 2, 0])

        panel_outline = RoundedRectangle(
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_opacity=0.0,
        ).move_to([PANEL_CX, 0.0, 0])

        # --- Animation -----------------------------------------------------
        self.play(FadeIn(title, shift=DOWN * 0.12), run_time=0.6)
        self.play(
            FadeIn(text_a, shift=RIGHT * 0.12),
            FadeIn(input_caption),
            run_time=0.6,
        )
        self.play(
            FadeIn(encode_block, shift=UP * 0.10),
            Write(encode_code),
            FadeIn(encode_caption),
            Create(text_to_encode),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=UP * 0.08) for chip in token_chips],
                lag_ratio=0.10,
            ),
            Create(encode_to_tokens),
            FadeIn(tokens_caption),
            run_time=1.2,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=UP * 0.05) for chip in id_chips],
                lag_ratio=0.08,
            ),
            FadeIn(ids_caption),
            run_time=0.8,
        )
        self.play(
            Create(subword_brace),
            FadeIn(subword_label),
            run_time=0.6,
        )

        # The vocabulary is a tall JSON table shown through an editor window.
        self.play(
            FadeIn(panel),
            FadeIn(panel_title),
            FadeIn(panel_count),
            Create(divider),
            FadeIn(direction_encode),
            FadeIn(more),
            FadeIn(scroll_track),
            FadeIn(scroll_thumb),
            run_time=0.8,
        )

        # The table sits behind the masks; the masks clip it to the window.
        self.add(table)
        self.add(
            mask_top_inside,
            mask_top_outside,
            mask_bottom_inside,
            mask_bottom_outside,
        )
        self.add(panel_outline)
        self.add(
            panel_title,
            panel_count,
            divider,
            direction_encode,
            more,
            scroll_track,
            scroll_thumb,
            title,
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(entry, shift=DOWN * 0.03)
                    for entry in table.entries
                ],
                lag_ratio=0.02,
            ),
            run_time=1.0,
        )

        # A quick scroll trip shows that the table is much taller than the
        # window; the thumb and the ids both move.
        self.play(
            Indicate(
                panel_count,
                color=theme.color("dimension"),
                scale_factor=1.12,
            ),
            run_time=0.5,
        )
        self.play(
            table.animate.shift(UP * scroll_delta),
            scroll_thumb.animate.shift(DOWN * thumb_delta),
            run_time=2.0,
            rate_func=smooth,
        )
        self.play(
            table.animate.shift(DOWN * scroll_delta),
            scroll_thumb.animate.shift(UP * thumb_delta),
            run_time=2.0,
            rate_func=smooth,
        )

        # encode: string -> id. Each lookup leaves its row highlighted.
        for chip, id_chip, token_id in zip(
            token_chips, id_chips, TOKEN_IDS
        ):
            self.play(
                Indicate(
                    chip,
                    color=theme.color("activation"),
                    scale_factor=1.08,
                ),
                Indicate(
                    id_chip,
                    color=theme.color("activation"),
                    scale_factor=1.10,
                ),
                table.highlight(token_id, active=True),
                run_time=0.28,
            )

        # decode: ids -> text, using the same vocabulary in reverse.
        self.play(
            Write(decode_code),
            FadeIn(decode_block, shift=UP * 0.10),
            FadeIn(decode_caption),
            Create(ids_to_decode),
            run_time=0.9,
        )
        self.play(
            Create(decode_to_text),
            FadeIn(text_b, shift=LEFT * 0.12),
            FadeIn(decoded_caption),
            run_time=0.9,
        )
        self.play(FadeIn(roundtrip), run_time=0.5)

        # Show that decode reads the same rows in the opposite direction.
        self.play(FadeTransform(direction_encode, direction_decode), run_time=0.5)
        for id_chip, token_id in zip(id_chips, TOKEN_IDS):
            entry = table.entry(token_id)
            self.play(
                Indicate(
                    id_chip,
                    color=theme.color("activation"),
                    scale_factor=1.10,
                ),
                Indicate(
                    entry.value_label,
                    color=theme.color("activation"),
                    scale_factor=1.15,
                ),
                run_time=0.22,
            )

        self.play(
            Write(identity),
            FadeIn(identity_caption),
            run_time=0.8,
        )
        self.play(
            Indicate(
                VGroup(text_a, text_b),
                color=theme.color("success"),
                scale_factor=1.03,
            ),
            Indicate(
                panel_count,
                color=theme.color("dimension"),
                scale_factor=1.10,
            ),
            run_time=1.0,
        )
        self.wait(2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _text_card(self, text: str, center: list[float]) -> VGroup:
        """Return a green text card centered at ``center``."""
        theme = self.theme
        rect = RoundedRectangle(
            width=CARD_WIDTH,
            height=CARD_HEIGHT,
            corner_radius=theme.node_radius,
            stroke_color=theme.color("activation"),
            stroke_width=theme.node_stroke,
            fill_color=theme.color("activation"),
            fill_opacity=theme.node_fill_opacity,
        ).move_to(center)
        label = Text(
            text,
            font_size=theme.size("caption"),
            font=theme.font,
            color=theme.color("text"),
        )
        if label.width > CARD_WIDTH - 0.35:
            label.scale_to_fit_width(CARD_WIDTH - 0.35)
        label.move_to(rect.get_center())
        return VGroup(rect, label)

    def _tokenizer_block(
        self,
        segments: Sequence[tuple[str, str | None]],
        center: list[float],
    ) -> VGroup:
        """Return a purple tokenizer block with a code-style call label."""
        theme = self.theme
        rect = RoundedRectangle(
            width=BLOCK_WIDTH,
            height=BLOCK_HEIGHT,
            corner_radius=theme.node_radius,
            stroke_color=theme.color("compute"),
            stroke_width=theme.node_stroke,
            fill_color=theme.color("compute"),
            fill_opacity=theme.node_fill_opacity,
        ).move_to(center)
        label = CodeLine(segments, theme, size="small")
        if label.width > BLOCK_WIDTH - 0.22:
            label.scale_to_fit_width(BLOCK_WIDTH - 0.22)
        label.move_to(rect.get_center())
        return VGroup(rect, label)
