"""YouTube thumbnail for the BPE scene.

A single 16:9 frame, rendered with ``-s``, that reuses the diagram palette:
green chips for activations and tokens, purple for the headline accents. The
left column carries the headline, the right panel shows the whole story of the
video in one glance: one word and the two subword tokens the learned merge
rules produce.

Run with:

    python scripts/render.py bpe_thumbnail --still -q h
"""

from __future__ import annotations

from manim import (
    BOLD,
    DOWN,
    FadeIn,
    LaggedStart,
    LEFT,
    Line,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
    Arrow,
    CapStyleType,
)

from schematics import DARK, SchematicScene, Theme, TokenChip
from scenes.transformer.bpe.specs import (
    NEW_WORD,
    RESULT_TOKENS,
    SPACE,
    TOKEN_IDS,
)

# --- Sizes ------------------------------------------------------------------
# The diagram sizes are tuned for a full 14x8 frame of small blocks. A
# thumbnail is read at 20% zoom, so it needs its own, much larger scale.
THUMBNAIL_SIZES = {
    **DARK.sizes,
    "thumb_eyebrow": 30,
    "thumb_headline": 210,
    "thumb_subtitle": 46,
    "thumb_tag": 32,
    "thumb_word": 52,
    "thumb_token": 52,
    "thumb_id": 30,
}
THUMBNAIL_THEME = Theme(name="bpe-thumbnail", sizes=THUMBNAIL_SIZES)

# --- Geometry ---------------------------------------------------------------
COLUMN_LEFT = -6.65
EYEBROW_Y = 3.12
HEADLINE_Y = 1.52
UNDERLINE_Y = 0.28
SUBTITLE_Y = -0.58
TAG_Y = -2.36

PANEL_CENTER = (3.62, 0.05, 0)
PANEL_PAD = 0.55
PANEL_RADIUS = 0.22

WORD_Y = 2.00
ARROW_START_Y = 1.45
ARROW_END_Y = 0.85
TOKENS_Y = 0.26
TOKEN_HEIGHT = 0.95
TOKEN_GAP = 0.16
IDS_Y = -0.70


class BPEThumbnail(SchematicScene):
    """Static cover image for the BPE video."""

    theme = THUMBNAIL_THEME
    # The cover is saved next to the video it belongs to:
    # media/videos/bpe/thumbnail.png
    video_key = "bpe"

    def construct(self):
        """Compose the headline column and the token panel."""
        eyebrow, headline, underline, subtitle, tag = self._headline()
        panel, word, arrow, tokens, ids = self._token_panel()

        self.play(FadeIn(panel), run_time=0.4)
        self.play(
            LaggedStart(
                FadeIn(eyebrow, shift=DOWN * 0.08),
                FadeIn(headline, shift=RIGHT * 0.12),
                FadeIn(underline, shift=RIGHT * 0.12),
                FadeIn(subtitle, shift=UP * 0.06),
                FadeIn(tag, shift=UP * 0.06),
                lag_ratio=0.12,
            ),
            run_time=1.0,
        )
        self.play(FadeIn(word, shift=DOWN * 0.06), run_time=0.3)
        self.play(FadeIn(arrow), run_time=0.3)
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=UP * 0.06) for chip in tokens],
                lag_ratio=0.12,
            ),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[FadeIn(chip) for chip in ids], lag_ratio=0.10),
            run_time=0.4,
        )
        self.wait(1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headline(self) -> tuple[Text, Text, Line, VGroup, VGroup]:
        """Build the left column: eyebrow, name, rule, subtitle, and tag."""
        theme = self.theme
        eyebrow = Text(
            "SUBWORD TOKENIZER",
            font_size=theme.size("thumb_eyebrow"),
            font=theme.font,
            weight=BOLD,
            color=theme.color("compute"),
        ).move_to([COLUMN_LEFT, EYEBROW_Y, 0], aligned_edge=LEFT)

        headline = Text(
            "BPE",
            font_size=theme.size("thumb_headline"),
            font=theme.font,
            weight=BOLD,
            color=theme.color("text"),
        ).move_to([COLUMN_LEFT, HEADLINE_Y, 0], aligned_edge=LEFT)

        underline = Line(
            [COLUMN_LEFT, UNDERLINE_Y, 0],
            [COLUMN_LEFT + headline.width, UNDERLINE_Y, 0],
            stroke_width=9,
            color=theme.color("compute"),
            cap_style=CapStyleType.ROUND,
        )

        subtitle = VGroup(
            Text(
                "how a tokenizer",
                font_size=theme.size("thumb_subtitle"),
                font=theme.font,
                color=theme.color("text"),
            ),
            Text(
                "learns subwords",
                font_size=theme.size("thumb_subtitle"),
                font=theme.font,
                color=theme.color("muted"),
            ),
        ).arrange(DOWN, buff=0.10, aligned_edge=LEFT)
        subtitle.move_to([COLUMN_LEFT, SUBTITLE_Y, 0], aligned_edge=LEFT)

        tag_label = Text(
            "SentencePiece",
            font_size=theme.size("thumb_tag"),
            font=theme.font,
            color=theme.color("compute"),
        )
        tag_rect = RoundedRectangle(
            width=tag_label.width + 0.52,
            height=tag_label.height + 0.36,
            corner_radius=0.10,
            stroke_color=theme.color("compute"),
            stroke_width=2.2,
            fill_color=theme.color("compute"),
            fill_opacity=0.14,
        )
        tag = VGroup(tag_rect, tag_label)
        tag_label.move_to(tag_rect.get_center())
        tag.move_to([COLUMN_LEFT, TAG_Y, 0], aligned_edge=LEFT)

        return eyebrow, headline, underline, subtitle, tag

    def _token_panel(self) -> tuple[VGroup, Text, Arrow, VGroup, VGroup]:
        """Build the right panel: word, arrow, the two tokens, and their ids."""
        theme = self.theme
        center_x = PANEL_CENTER[0]

        word = Text(
            NEW_WORD,
            font_size=theme.size("thumb_word"),
            font=theme.mono_font,
            color=theme.color("activation"),
        ).move_to([center_x, WORD_Y, 0])

        arrow = Arrow(
            [center_x, ARROW_START_Y, 0],
            [center_x, ARROW_END_Y, 0],
            buff=0.0,
            stroke_width=16,
            max_tip_length_to_length_ratio=0.50,
        )
        arrow.set_color(theme.color("activation"))

        # The cover hides the SentencePiece space marker, so the two pieces
        # read as plain "low" and "est" cut out of the word above.
        tokens = VGroup(
            *[
                TokenChip(
                    token.removeprefix(SPACE),
                    theme,
                    color_key="activation",
                    height=TOKEN_HEIGHT,
                    pad=0.18,
                    size="thumb_token",
                    mono=True,
                )
                for token in RESULT_TOKENS
            ]
        ).arrange(RIGHT, buff=TOKEN_GAP)
        tokens.move_to([center_x, TOKENS_Y, 0])

        ids = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.46,
                    pad=0.14,
                    size="thumb_id",
                    mono=True,
                )
                for token_id in TOKEN_IDS
            ]
        )
        for id_chip, token in zip(ids, tokens):
            id_chip.move_to([token.get_center()[0], IDS_Y, 0])

        # Center the remaining stack optically inside the panel instead of
        # hand-tuning every y so the top and bottom padding match.
        content = VGroup(word, arrow, tokens, ids)
        content.move_to([center_x, PANEL_CENTER[1], 0])

        # The card hugs its content: one padding constant keeps the frame
        # tight around the text at every font size.
        rect = RoundedRectangle(
            width=content.width + 2 * PANEL_PAD,
            height=content.height + 2 * PANEL_PAD,
            corner_radius=PANEL_RADIUS,
            stroke_color=theme.color("border"),
            stroke_width=2.0,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        ).move_to(content.get_center())

        return rect, word, arrow, tokens, ids
