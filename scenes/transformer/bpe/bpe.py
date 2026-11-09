"""SentencePiece: how a tokenizer learns subwords and encodes fast.

The first act starts from the corpus as running text, folds it into a
word-frequency table, then repeatedly counts adjacent pairs and merges the
most frequent one, growing the token vocabulary and building an ordered list
of merge rules. The second act encodes a word that was never seen in training
by applying those rules greedily left to right.

Run with:

    python scripts/render.py bpe -q l
"""

from __future__ import annotations

from collections.abc import Sequence

from manim import (
    ChangeDecimalToValue,
    DecimalNumber,
    DOWN,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    LEFT,
    MathTex,
    Mobject,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
    Write,
)

from schematics import CodeLine, SchematicScene, SymbolStrip, TokenChip
from scenes.transformer.bpe.specs import (
    ANIMATED_MERGES,
    BASE_SYMBOLS,
    CODE_ENCODE,
    CORPUS,
    MERGES,
    NEW_SYMBOLS,
    NEW_WORD,
    RAW_CORPUS_LINES,
    RESULT_TOKENS,
    SPACE,
    TITLE,
    TOKEN_IDS,
    VISIBLE_MERGES,
    VOCAB_TARGET,
)

# --- Act 1 geometry ---------------------------------------------------------
TITLE_Y = 3.70

CORPUS_PANEL = (-4.75, 0.55, 0)
CORPUS_WIDTH = 4.35
CORPUS_HEIGHT = 4.60
CORPUS_HEADER_Y = 2.32
CORPUS_SUBTITLE_Y = CORPUS_HEADER_Y - 0.30
CORPUS_TEXT_Y = 0.70
CORPUS_ROW_Y = (1.62, 1.00, 0.38, -0.24)
CORPUS_STRIP_X = -6.73
CORPUS_COUNT_X = -2.80
CORPUS_LEGEND_Y = -1.46
CORPUS_HINT_Y = -0.85

VOCAB_PANEL = (0.0, 0.55, 0)
VOCAB_WIDTH = 3.90
VOCAB_HEIGHT = 4.60
VOCAB_HEADER_Y = 2.32
VOCAB_COUNT_Y = 2.02
VOCAB_SYMBOLS_LABEL_Y = 1.66
VOCAB_SYMBOLS_Y = 1.32
VOCAB_MERGES_LABEL_Y = 0.95
VOCAB_MERGES_Y = 0.58
VOCAB_FOOTNOTE_Y = -1.48

RULES_PANEL = (4.75, 0.55, 0)
RULES_WIDTH = 4.35
RULES_HEIGHT = 4.60
RULES_HEADER_Y = 2.32
RULES_STRIP_X = 2.82
RULES_ROW_START_Y = 1.58
RULES_ROW_GAP = 0.40

# Act 2 keeps the vocabulary to the left of the rules and slides both panels
# left as one block, freeing the right side for the encoded word. The gap
# between them is tighter than in act 1.
VOCAB_ACT2_X = -4.93
VOCAB_RULES_GAP = 0.30
RULES_ACT2_X = (
    VOCAB_ACT2_X + VOCAB_WIDTH / 2 + VOCAB_RULES_GAP + RULES_WIDTH / 2
)

ITER_BADGE = (0.0, -2.55, 0)
REPEAT_Y = -2.55

# --- Act 2 geometry ---------------------------------------------------------
WORD_CARD = (3.60, 2.55, 0)
WORD_CAPTION_Y = 3.10
CODE_Y = 1.98
CHAR_Y = 1.30
RESULT_CAPTION_Y = 1.62
ID_Y = 0.72


class BPE(SchematicScene):
    """Train a BPE tokenizer, then encode an unseen word with its rules."""

    def construct(self):
        """Build the panels, run the merge loop, then encode an unseen word."""
        theme = self.theme
        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        if title.width > 13.0:
            title.scale_to_fit_width(13.0)

        corpus_panel, raw_text, mode_label = self._corpus_panel()
        rules_panel = self._rules_panel()
        repeat_note = self._repeat_note()

        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.6)
        self.play(FadeIn(corpus_panel, shift=RIGHT * 0.10), run_time=0.7)
        self.play(FadeIn(raw_text, shift=UP * 0.08), run_time=0.7)

        # The corpus arrives as running text. Counting it is the first step of
        # training and turns it into the (word, frequency) table.
        count_hint = self.caption(
            "count how often each word occurs",
            x=CORPUS_PANEL[0],
            y=CORPUS_HINT_Y,
        )
        self.play(FadeIn(count_hint), run_time=0.4)
        self.play(FadeIn(rules_panel, shift=LEFT * 0.10), run_time=0.7)

        table, counts, strips, legend = self._corpus_table()
        mode_counted = self._corpus_mode("word : count")
        self.play(
            FadeOut(raw_text, shift=UP * 0.08),
            FadeOut(count_hint),
            FadeOut(mode_label),
            FadeIn(mode_counted),
            run_time=0.7,
        )
        # Drop the faded subtitle from the panel group; otherwise fading the
        # whole panel out later would briefly bring it back at full opacity.
        corpus_panel.remove(mode_label)
        corpus_panel.add(mode_counted)

        corpus_panel.add(table)
        self.play(
            LaggedStart(
                *[FadeIn(row, shift=DOWN * 0.05) for row in table],
                lag_ratio=0.12,
            ),
            run_time=0.8,
        )
        corpus_panel.add(counts)
        self.play(
            LaggedStart(*[FadeIn(count) for count in counts], lag_ratio=0.12),
            run_time=0.6,
        )
        corpus_panel.add(legend)
        self.play(FadeIn(legend), run_time=0.4)

        # --- Vocabulary: start from the alphabet --------------------------
        vocab_panel, base_chips, merge_chips, vocab_ellipsis, vocab_size = (
            self._vocab_panel()
        )
        vocab_chips = {
            chip.text: chip for chip in (*base_chips, *merge_chips)
        }
        self.play(FadeIn(vocab_panel, shift=UP * 0.10), run_time=0.6)
        seed_note = self.caption(
            "start: every symbol enters the vocabulary",
            x=VOCAB_PANEL[0],
            y=REPEAT_Y,
        )
        self.play(FadeIn(seed_note), run_time=0.3)
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=UP * 0.05) for chip in base_chips],
                lag_ratio=0.05,
            ),
            ChangeDecimalToValue(vocab_size, len(BASE_SYMBOLS)),
            run_time=0.9,
        )
        self.play(FadeOut(seed_note), run_time=0.3)

        # --- Training: repeated count -> merge ----------------------------
        rule_rows: list[VGroup] = []
        for step_idx in range(ANIMATED_MERGES):
            left, right, merged, count = MERGES[step_idx]
            step = step_idx + 1

            badge = self._iteration_badge(step, left, right, merged, count)
            badge.move_to(ITER_BADGE)
            self.play(FadeIn(badge, shift=UP * 0.06), run_time=0.35)

            occurrences = [
                (strips[word], strips[word].find_pair(left, right))
                for word, _, _ in CORPUS
            ]
            occurrences = [
                (strip, index)
                for strip, index in occurrences
                if index is not None
            ]

            highlights = [
                anim
                for strip, index in occurrences
                for anim in strip.highlight_pair(index)
            ]
            self.play(*highlights, run_time=0.45)

            fades = [
                FadeOut(chip)
                for strip, index in occurrences
                for chip in (strip.chips[index], strip.chips[index + 1])
            ]
            self.play(*fades, run_time=0.25)

            moves = []
            for strip, index in occurrences:
                _, merge_moves = strip.merge_pair(index)
                moves.extend(merge_moves)
            self.play(*moves, run_time=0.45)

            row = self._rule_row(step, left, right, merged, count)
            row.move_to(
                [RULES_STRIP_X, RULES_ROW_START_Y - step_idx * RULES_ROW_GAP, 0],
                aligned_edge=LEFT,
            )
            rules_panel.add(row)
            rule_rows.append(row)
            new_token = merge_chips[step_idx]
            self.play(
                FadeIn(row, shift=RIGHT * 0.08),
                FadeIn(new_token, shift=UP * 0.05),
                ChangeDecimalToValue(
                    vocab_size, len(BASE_SYMBOLS) + step
                ),
                run_time=0.45,
            )
            self.play(
                Indicate(
                    new_token,
                    color=theme.color("activation"),
                    scale_factor=1.12,
                ),
                run_time=0.30,
            )
            self.play(FadeOut(badge), run_time=0.25)

        # The loop keeps going past the animated steps.
        for extra_idx in range(ANIMATED_MERGES, VISIBLE_MERGES):
            left, right, merged, count = MERGES[extra_idx]
            row = self._rule_row(extra_idx + 1, left, right, merged, count)
            row.move_to(
                [RULES_STRIP_X, RULES_ROW_START_Y - extra_idx * RULES_ROW_GAP, 0],
                aligned_edge=LEFT,
            )
            rules_panel.add(row)
            rule_rows.append(row)
            self.play(
                FadeIn(row, shift=RIGHT * 0.08),
                FadeIn(merge_chips[extra_idx], shift=UP * 0.05),
                ChangeDecimalToValue(
                    vocab_size, len(BASE_SYMBOLS) + extra_idx + 1
                ),
                run_time=0.35,
            )

        ellipsis = Text(
            "\u2026",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to(
            [RULES_PANEL[0], RULES_ROW_START_Y - len(rule_rows) * RULES_ROW_GAP, 0]
        )
        rules_panel.add(ellipsis)
        self.play(
            FadeIn(ellipsis),
            FadeIn(vocab_ellipsis),
            FadeIn(repeat_note),
            run_time=0.5,
        )

        # --- Transition: keep the rules, drop the training scaffolding -----
        # The vocabulary and the rules slide left as one block, keeping their
        # order and tightening the gap between them. This frees the right
        # side for the encoded word without a panel jumping across the screen.
        vocab_shift = LEFT * (VOCAB_PANEL[0] - VOCAB_ACT2_X)
        self.play(
            FadeOut(corpus_panel, shift=DOWN * 0.08),
            FadeOut(repeat_note),
            vocab_panel.animate.shift(vocab_shift),
            *[
                chip.animate.shift(vocab_shift)
                for chip in (*base_chips, *merge_chips)
            ],
            vocab_ellipsis.animate.shift(vocab_shift),
            rules_panel.animate.move_to([RULES_ACT2_X, RULES_PANEL[1], 0]),
            run_time=1.0,
        )

        # --- Fast encode of an unseen word --------------------------------
        word_card = self._text_card(f'"{NEW_WORD}"', WORD_CARD)
        word_caption = self.caption(
            "using SentencePiece to tokenize text",
            x=WORD_CARD[0],
            y=WORD_CAPTION_Y,
        )
        code = CodeLine(CODE_ENCODE, theme, size="small")
        code.move_to([WORD_CARD[0], CODE_Y, 0])
        if code.width > 5.6:
            code.scale_to_fit_width(5.6)

        chars = SymbolStrip(
            NEW_SYMBOLS, theme, height=0.30, pad=0.08, buff=0.04, marker=SPACE
        )
        chars.set_center(WORD_CARD[0], CHAR_Y)

        self.play(
            FadeIn(word_card, shift=DOWN * 0.10),
            FadeIn(word_caption),
            Write(code),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=UP * 0.06) for chip in chars.chips],
                lag_ratio=0.08,
            ),
            run_time=0.9,
        )
        split_caption = self.caption(
            "split into symbols", x=WORD_CARD[0], y=CHAR_Y - 0.52, color_key="compute"
        )
        self.play(FadeIn(split_caption), run_time=0.4)
        fast_note = self.caption(
            "greedy, left-to-right (no pair counting)",
            x=WORD_CARD[0],
            y=1.62,
        )
        self.play(FadeIn(fast_note), run_time=0.4)

        # Apply the learned rules in priority order, greedily left to right.
        for idx in range(ANIMATED_MERGES):
            left, right, _, _ = MERGES[idx]
            index = chars.find_pair(left, right)
            self.play(
                Indicate(
                    rule_rows[idx],
                    color=theme.color("compute"),
                    scale_factor=1.06,
                ),
                *chars.highlight_pair(index),
                run_time=0.45,
            )
            self.play(
                FadeOut(chars.chips[index]),
                FadeOut(chars.chips[index + 1]),
                run_time=0.25,
            )
            _, merge_moves = chars.merge_pair(index)
            self.play(*merge_moves, run_time=0.40)
        self.play(FadeOut(split_caption), run_time=0.3)

        # Remaining rules find no match and are skipped.
        skip_caption = self.caption(
            "no match \u2192 skip",
            x=WORD_CARD[0],
            y=CHAR_Y - 0.52,
            color_key="muted",
        )
        self.play(FadeIn(skip_caption), run_time=0.3)
        for idx in range(ANIMATED_MERGES, len(rule_rows)):
            self.play(rule_rows[idx].animate.set_opacity(0.35), run_time=0.18)
        self.play(FadeOut(skip_caption), run_time=0.3)

        # Result: two subword tokens and their ids.
        result_caption = self.caption(
            "2 subword tokens",
            x=WORD_CARD[0],
            y=RESULT_CAPTION_Y,
            color_key="dimension",
        )
        id_chips = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.28,
                    pad=0.07,
                    mono=True,
                    size="small",
                )
                for token_id in TOKEN_IDS
            ]
        )
        for id_chip, chip in zip(id_chips, chars.chips):
            id_chip.move_to([chip.get_center()[0], ID_Y, 0])

        self.play(
            FadeOut(fast_note),
            FadeIn(result_caption),
            LaggedStart(
                *[FadeIn(id_chip, shift=DOWN * 0.05) for id_chip in id_chips],
                lag_ratio=0.10,
            ),
            run_time=0.7,
        )
        vocab_caption = self.caption(
            "both tokens are in the vocabulary",
            x=VOCAB_ACT2_X,
            y=-1.02,
            color_key="dimension",
        )
        self.play(
            *[
                Indicate(chip, color=theme.color("activation"), scale_factor=1.10)
                for chip in chars.chips
            ],
            *[
                Indicate(
                    vocab_chips[token],
                    color=theme.color("dimension"),
                    scale_factor=1.15,
                )
                for token in RESULT_TOKENS
            ],
            FadeIn(vocab_caption),
            run_time=0.7,
        )
        self.wait(2.5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _corpus_panel(self) -> tuple[VGroup, VGroup, Text]:
        """Build the corpus panel with its running-text view.

        Returns the panel frame, the raw text lines, and the mode subtitle
        that is swapped for ``word : count`` once the text is counted.
        """
        theme = self.theme
        rect = RoundedRectangle(
            width=CORPUS_WIDTH,
            height=CORPUS_HEIGHT,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=0.92,
        ).move_to(CORPUS_PANEL)

        header = Text(
            "training corpus",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([CORPUS_PANEL[0], CORPUS_HEADER_Y, 0])

        mode_label = self._corpus_mode("raw text")
        raw_text = self._raw_corpus()

        return VGroup(rect, header, mode_label), raw_text, mode_label

    def _corpus_mode(self, text: str) -> Text:
        """Return the corpus panel subtitle centered at its fixed position."""
        theme = self.theme
        return Text(
            text,
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([CORPUS_PANEL[0], CORPUS_SUBTITLE_Y, 0])

    def _raw_corpus(self) -> VGroup:
        """Build the corpus as running text, before it is counted."""
        theme = self.theme
        lines = VGroup(
            *[
                Text(
                    line,
                    font_size=theme.size("caption"),
                    font=theme.font,
                    color=theme.color("activation"),
                )
                for line in RAW_CORPUS_LINES
            ]
        ).arrange(DOWN, buff=0.26)
        lines.move_to([CORPUS_PANEL[0], CORPUS_TEXT_Y, 0])
        return lines

    def _corpus_table(
        self,
    ) -> tuple[VGroup, VGroup, dict[str, SymbolStrip], VGroup]:
        """Build the counted corpus: symbol strips, count labels, and legend."""
        theme = self.theme
        table = VGroup()
        counts = VGroup()
        strips: dict[str, SymbolStrip] = {}
        for (word, symbols, count), y in zip(CORPUS, CORPUS_ROW_Y):
            strip = SymbolStrip(
                symbols, theme, height=0.30, pad=0.08, buff=0.04, marker=SPACE
            )
            strip.set_left(CORPUS_STRIP_X, y)
            count_label = Text(
                str(count),
                font_size=theme.size("small"),
                font=theme.mono_font,
                color=theme.color("muted"),
            ).move_to([CORPUS_COUNT_X, y, 0], aligned_edge=RIGHT)
            table.add(strip)
            counts.add(count_label)
            strips[word] = strip

        legend = VGroup(
            Text(
                "space",
                font_size=theme.size("micro"),
                font=theme.font,
                color=theme.color("muted"),
            ),
            TokenChip(
                SPACE,
                theme,
                color_key="muted",
                height=0.26,
                pad=0.06,
                mono=True,
                size="micro",
                valign="bottom",
            ),
        ).arrange(RIGHT, buff=0.08)
        legend.move_to([CORPUS_PANEL[0], CORPUS_LEGEND_Y, 0])

        return table, counts, strips, legend

    def _vocab_panel(
        self,
    ) -> tuple[VGroup, list[TokenChip], list[TokenChip], Text, DecimalNumber]:
        """Build the vocabulary panel with every token slot pre-placed.

        The panel starts empty. The caller reveals the base-symbol chips
        first and then one chip from ``merge_chips`` per merge rule. Because
        all chips are positioned up front, revealing one never reflows the
        tokens that are already visible.

        Returns:
            The static panel, the base-symbol chips, the merged-token chips,
            the trailing ellipsis, and the live ``|V|`` number.
        """
        theme = self.theme
        rect = RoundedRectangle(
            width=VOCAB_WIDTH,
            height=VOCAB_HEIGHT,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        ).move_to(VOCAB_PANEL)
        header = Text(
            "token vocabulary",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([VOCAB_PANEL[0], VOCAB_HEADER_Y, 0])
        counter, count = self._vocab_counter()

        content_left = VOCAB_PANEL[0] - VOCAB_WIDTH / 2 + 0.22
        content_width = VOCAB_WIDTH - 0.44
        label_kwargs = {
            "font_size": theme.size("micro"),
            "font": theme.font,
            "color": theme.color("muted"),
        }
        symbols_label = Text("base symbols", **label_kwargs).move_to(
            [content_left, VOCAB_SYMBOLS_LABEL_Y, 0], aligned_edge=LEFT
        )
        merges_label = Text("from merges", **label_kwargs).move_to(
            [content_left, VOCAB_MERGES_LABEL_Y, 0], aligned_edge=LEFT
        )
        footnote = Text(
            "each merge appends one token",
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([VOCAB_PANEL[0], VOCAB_FOOTNOTE_Y, 0])

        base_chips = [
            TokenChip(
                symbol,
                theme,
                color_key="muted" if symbol == SPACE else "activation",
                height=0.30,
                pad=0.07,
                size="small",
                mono=True,
                valign="bottom" if symbol == SPACE else "center",
            )
            for symbol in BASE_SYMBOLS
        ]
        merge_chips = [
            TokenChip(
                merged,
                theme,
                color_key="activation",
                height=0.30,
                pad=0.07,
                size="small",
                mono=True,
            )
            for _, _, merged, _ in MERGES[:VISIBLE_MERGES]
        ]

        self._flow_grid(
            base_chips,
            left_x=content_left,
            top_y=VOCAB_SYMBOLS_Y,
            max_width=content_width,
        )
        merge_rows = self._flow_grid(
            merge_chips,
            left_x=content_left,
            top_y=VOCAB_MERGES_Y,
            max_width=content_width,
        )

        ellipsis = Text(
            "\u2026",
            font_size=theme.size("small"),
            font=theme.mono_font,
            color=theme.color("muted"),
        ).next_to(merge_rows[-1], RIGHT, buff=0.14)

        panel = VGroup(
            rect, header, counter, symbols_label, merges_label, footnote
        )
        return panel, base_chips, merge_chips, ellipsis, count

    def _vocab_counter(self) -> tuple[VGroup, DecimalNumber]:
        """Return the ``|V| = n`` label and the animatable number inside it."""
        theme = self.theme
        prefix = MathTex(
            r"|V| =",
            font_size=theme.size("small"),
            color=theme.color("dimension"),
        )
        number = DecimalNumber(
            0,
            num_decimal_places=0,
            font_size=theme.size("small"),
            color=theme.color("dimension"),
        )
        counter = VGroup(prefix, number).arrange(RIGHT, buff=0.08)
        counter.move_to([VOCAB_PANEL[0], VOCAB_COUNT_Y, 0])
        return counter, number

    def _flow_grid(
        self,
        chips: Sequence[Mobject],
        *,
        left_x: float,
        top_y: float,
        max_width: float,
        gap: float = 0.09,
        row_gap: float = 0.16,
    ) -> list[VGroup]:
        """Place chips in left-aligned rows that wrap at ``max_width``.

        The first pass wraps greedily. If that yields several rows, a second
        pass uses the average row width so the last row does not end with a
        single lone chip.
        """

        def wrap(width: float) -> list[list[Mobject]]:
            """Greedily split the chips into rows of at most ``width``."""
            rows: list[list[Mobject]] = []
            current: list[Mobject] = []
            used = 0.0
            for chip in chips:
                extra = chip.width + (gap if current else 0.0)
                if current and used + extra > width:
                    rows.append(current)
                    current = [chip]
                    used = chip.width
                else:
                    current.append(chip)
                    used += extra
            if current:
                rows.append(current)
            return rows

        rows = wrap(max_width)
        if len(rows) > 1:
            total = sum(chip.width for chip in chips) + gap * (len(chips) - 1)
            balanced = wrap(total / len(rows) + 0.20)
            if len(balanced) == len(rows):
                rows = balanced

        groups = []
        for row_idx, row in enumerate(rows):
            group = VGroup(*row).arrange(RIGHT, buff=gap)
            group.move_to(
                [left_x, top_y - row_idx * (group.height + row_gap), 0],
                aligned_edge=LEFT,
            )
            groups.append(group)
        return groups

    def _rules_panel(self) -> VGroup:
        """Return the (initially empty) merge-rules panel."""
        theme = self.theme
        rect = RoundedRectangle(
            width=RULES_WIDTH,
            height=RULES_HEIGHT,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=0.92,
        ).move_to(RULES_PANEL)
        header = Text(
            "merge rules (priority order)",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([RULES_PANEL[0], RULES_HEADER_Y, 0])
        subtitle = Text(
            "each merge adds one token to the vocabulary",
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([RULES_PANEL[0], RULES_HEADER_Y - 0.30, 0])
        return VGroup(rect, header, subtitle)

    def _repeat_note(self) -> VGroup:
        """Build the 'repeat until target vocab size' note."""
        theme = self.theme
        return VGroup(
            Text(
                "repeat until the vocabulary reaches",
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("muted"),
            ),
            MathTex(
                VOCAB_TARGET,
                font_size=theme.size("small"),
                color=theme.color("dimension"),
            ),
        ).arrange(RIGHT, buff=0.10).move_to([0.0, REPEAT_Y, 0])

    def _iteration_badge(
        self, step: int, left: str, right: str, merged: str, count: int
    ) -> VGroup:
        """Build the per-step popup: 'count pairs -> e s x 9' and the merge."""
        theme = self.theme
        step_label = Text(
            f"step {step}",
            font_size=theme.size("micro"),
            font=theme.font,
            color=theme.color("muted"),
        )
        count_line = VGroup(
            Text(
                "count pairs",
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("compute"),
            ),
            MathTex(
                r"\rightarrow",
                font_size=theme.size("small"),
                color=theme.color("compute"),
            ),
            Text(
                f"{left} {right}",
                font_size=theme.size("small"),
                font=theme.mono_font,
                color=theme.color("activation"),
            ),
            Text(
                f"\u00d7 {count}",
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("muted"),
            ),
        ).arrange(RIGHT, buff=0.09)
        merge_line = VGroup(
            Text(
                "merge",
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("compute"),
            ),
            self._merge_expr(left, right, merged),
        ).arrange(RIGHT, buff=0.09)

        return VGroup(step_label, count_line, merge_line).arrange(DOWN, buff=0.16)

    def _rule_row(
        self, number: int, left: str, right: str, merged: str, count: int
    ) -> VGroup:
        """Build one numbered rule line for the rules panel."""
        theme = self.theme
        number_label = Text(
            f"{number}.",
            font_size=theme.size("small"),
            font=theme.mono_font,
            color=theme.color("muted"),
        )
        expression = self._merge_expr(left, right, merged, count)
        row = VGroup(number_label, expression).arrange(RIGHT, buff=0.12)
        if row.width > RULES_WIDTH - 0.55:
            row.scale_to_fit_width(RULES_WIDTH - 0.55)
        return row

    def _merge_expr(
        self, left: str, right: str, merged: str, count: int | None = None
    ) -> VGroup:
        """Build 'left + right -> merged (count)' with semantic colors."""
        theme = self.theme
        symbol_kwargs = {
            "font": theme.mono_font,
            "font_size": theme.size("small"),
        }
        left_label = Text(
            left, color=theme.color("activation"), **symbol_kwargs
        )
        right_label = Text(
            right, color=theme.color("activation"), **symbol_kwargs
        )
        merged_label = Text(
            merged, color=theme.color("activation"), **symbol_kwargs
        )
        parts = [
            left_label,
            Text(
                " + ",
                font_size=theme.size("small"),
                font=theme.font,
                color=theme.color("muted"),
            ),
            right_label,
            MathTex(
                r"\rightarrow",
                font_size=theme.size("small"),
                color=theme.color("compute"),
            ),
            merged_label,
        ]
        if count is not None:
            parts.append(
                Text(
                    f" ({count})",
                    font_size=theme.size("small"),
                    font=theme.font,
                    color=theme.color("muted"),
                )
            )
        row = VGroup(*parts).arrange(RIGHT, buff=0.07)

        # A lone space-marker symbol is a tiny bottom block, so centering it in
        # the row would make it float mid-height like a minus. Pin its bottom to
        # the row baseline, matching the marker inside the merged symbol.
        baseline = row.get_bottom()[1]
        for label, symbol in ((left_label, left), (right_label, right)):
            if symbol == SPACE:
                label.move_to(
                    [label.get_center()[0], baseline + label.height / 2, 0]
                )
        return row

    def _text_card(self, text: str, center: tuple[float, float, float]) -> VGroup:
        """Return a green text card centered at ``center``."""
        theme = self.theme
        rect = RoundedRectangle(
            width=2.4,
            height=0.62,
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
        if label.width > 2.4 - 0.30:
            label.scale_to_fit_width(2.4 - 0.30)
        label.move_to(rect.get_center())
        return VGroup(rect, label)
