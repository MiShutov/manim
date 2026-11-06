"""SentencePiece: how a tokenizer learns subwords and encodes fast.

The first act trains a tokenizer on a tiny corpus: symbols are repeatedly
counted and the most frequent adjacent pair is merged, building an ordered
list of merge rules. The second act encodes a word that was never seen in
training by applying those rules greedily left to right.

Run with:

    python scripts/render.py bpe -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    LEFT,
    MathTex,
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
    CODE_ENCODE,
    CORPUS,
    MERGES,
    NEW_SYMBOLS,
    NEW_WORD,
    SPACE,
    TITLE,
    TOKEN_IDS,
    VOCAB_TARGET,
)

# --- Act 1 geometry ---------------------------------------------------------
TITLE_Y = 3.70

CORPUS_PANEL = (-4.55, 0.55, 0)
CORPUS_WIDTH = 4.90
CORPUS_HEIGHT = 4.60
CORPUS_HEADER_Y = 2.32
CORPUS_ROW_Y = (1.62, 1.00, 0.38, -0.24)
CORPUS_STRIP_X = -6.72
CORPUS_COUNT_X = -2.42
CORPUS_LEGEND_Y = -1.46

RULES_PANEL = (4.55, 0.55, 0)
RULES_WIDTH = 4.80
RULES_HEIGHT = 4.60
RULES_HEADER_Y = 2.32
RULES_STRIP_X = 2.28
RULES_ROW_START_Y = 1.58
RULES_ROW_GAP = 0.40

ITER_BADGE = (0.0, 1.50, 0)
REPEAT_Y = -2.30

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
        """Build the corpus and rules panels, run the merge loop, then encode."""
        theme = self.theme
        title = self.title(TITLE, x=0.0, y=TITLE_Y, size="title_small")
        if title.width > 13.0:
            title.scale_to_fit_width(13.0)

        corpus_panel, strips = self._corpus_panel()
        rules_panel = self._rules_panel()
        repeat_note = self._repeat_note()

        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.6)
        self.play(FadeIn(corpus_panel, shift=RIGHT * 0.10), run_time=0.7)
        self.play(FadeIn(rules_panel, shift=LEFT * 0.10), run_time=0.7)

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
            self.play(FadeIn(row, shift=RIGHT * 0.08), run_time=0.35)
            self.play(FadeOut(badge), run_time=0.25)

        # The loop keeps going past the animated steps.
        for extra_idx in range(ANIMATED_MERGES, ANIMATED_MERGES + 3):
            left, right, merged, count = MERGES[extra_idx]
            row = self._rule_row(extra_idx + 1, left, right, merged, count)
            row.move_to(
                [RULES_STRIP_X, RULES_ROW_START_Y - extra_idx * RULES_ROW_GAP, 0],
                aligned_edge=LEFT,
            )
            rules_panel.add(row)
            rule_rows.append(row)
            self.play(FadeIn(row, shift=RIGHT * 0.08), run_time=0.30)

        ellipsis = Text(
            "\u2026",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to(
            [RULES_PANEL[0], RULES_ROW_START_Y - len(rule_rows) * RULES_ROW_GAP, 0]
        )
        rules_panel.add(ellipsis)
        self.play(FadeIn(ellipsis), FadeIn(repeat_note), run_time=0.5)

        # --- Transition: keep the rules, drop the training scaffolding -----
        self.play(
            FadeOut(corpus_panel, shift=DOWN * 0.08),
            FadeOut(repeat_note),
            rules_panel.animate.move_to([-4.55, 0.55, 0]),
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
        self.play(
            *[
                Indicate(chip, color=theme.color("activation"), scale_factor=1.10)
                for chip in chars.chips
            ],
            run_time=0.6,
        )
        self.wait(2.5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _corpus_panel(self) -> tuple[VGroup, dict[str, SymbolStrip]]:
        """Build the corpus panel and return it with a word -> strip lookup."""
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
            "corpus (word : count)",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([CORPUS_PANEL[0], CORPUS_HEADER_Y, 0])

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

        panel = VGroup(rect, header, legend)
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
            ).move_to([CORPUS_COUNT_X, y, 0], aligned_edge=LEFT)
            panel.add(strip, count_label)
            strips[word] = strip

        return panel, strips

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
