"""Educational scene: tokens are sequence elements, not only text.

The upper flow turns a sentence into subword tokens, token ids, and then into
embedding vectors. The lower flow turns an image into patches and then into
embedding vectors. Both streams converge on the same Transformer block.

Run with:

    python scripts/render.py token_anatomy -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    RIGHT,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    Text,
    VGroup,
    Write,
    smooth,
)

from schematics import (
    MatrixGrid,
    Node,
    NodeSpec,
    PatchGrid,
    SchematicScene,
    TokenChip,
    connect,
)
from scenes.transformer.tokens.specs import (
    CAT_COLORS,
    CAT_PIXELS,
    EMBED_DIM,
    IMAGE_FORMULA,
    IMAGE_PATCH_COLS,
    IMAGE_PATCH_ROWS,
    IMAGE_TOKENS,
    SUBWORD_INDICES,
    TEXT_FORMULA,
    TEXT_PROMPT,
    TOKEN_IDS,
    TOKEN_PIECES,
)


class TokenAnatomy(SchematicScene):
    """Show text subwords and image patches becoming a shared token sequence."""

    def construct(self):
        """Build both modality flows and converge them on one Transformer."""
        theme = self.theme
        title = self.title(
            "From text and pixels to sequence tokens",
            x=0.0,
        )
        title.scale(0.75)
        divider = Line(
            [-6.55, 0.90, 0],
            [6.55, 0.90, 0],
            stroke_width=0.8,
            color=theme.color("border"),
            stroke_opacity=0.35,
        )

        # --- Text flow (upper) --------------------------------------------
        phrase = Text(
            TEXT_PROMPT,
            font_size=22,
            font=theme.font,
            color=theme.color("text"),
        ).move_to([-5.60, 2.15, 0])
        text_stage = self.caption(
            "TEXT", x=-5.60, y=3.00, color_key="key", size="caption"
        )

        chips = VGroup(
            *[
                TokenChip(
                    piece,
                    theme,
                    color_key="score" if index in SUBWORD_INDICES else "key",
                )
                for index, piece in enumerate(TOKEN_PIECES)
            ]
        ).arrange(RIGHT, buff=0.08).move_to([-2.45, 2.15, 0])
        tokenize_stage = self.caption(
            "TOKENIZATION", x=-2.45, y=3.00, color_key="key", size="caption"
        )
        subword_caption = self.caption(
            "sleeping → sleep + ing", x=-2.45, y=1.72, color_key="score"
        )

        id_chips = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.30,
                    pad=0.10,
                    mono=True,
                    size="small",
                )
                for token_id in TOKEN_IDS
            ]
        ).arrange(RIGHT, buff=0.06).move_to([0.85, 2.15, 0])
        ids_stage = self.caption(
            "TOKEN IDS", x=0.85, y=3.00, color_key="muted", size="caption"
        )

        lookup = MatrixGrid(
            len(TOKEN_PIECES),
            EMBED_DIM,
            theme,
            cell=0.16,
            gap=0.03,
            color_key="key",
            row_labels=TOKEN_IDS,
        ).move_to([3.30, 2.15, 0])
        lookup_stage = self.caption(
            "EMBEDDING LOOKUP", x=3.30, y=3.00, color_key="key", size="caption"
        )

        x_text = MatrixGrid(
            len(TOKEN_PIECES),
            EMBED_DIM,
            theme,
            cell=0.16,
            gap=0.03,
            color_key="value",
        ).move_to([5.45, 2.15, 0])
        text_emb_stage = self.caption(
            "EMBEDDINGS", x=5.45, y=3.00, color_key="value", size="caption"
        )

        text_formula = MathTex(
            TEXT_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("value"),
        ).move_to([4.45, 1.30, 0])
        if text_formula.width > 3.8:
            text_formula.scale_to_fit_width(3.8)

        text_arrow1 = connect(phrase, chips, theme, buff=0.14)
        text_arrow2 = connect(chips, id_chips, theme, buff=0.14)
        text_arrow3 = connect(id_chips, lookup, theme, buff=0.14)
        text_arrow4 = connect(lookup, x_text, theme, buff=0.14)

        # --- Image flow (lower) -------------------------------------------
        image = PatchGrid(
            CAT_PIXELS,
            theme,
            patch_rows=IMAGE_PATCH_ROWS,
            patch_cols=IMAGE_PATCH_COLS,
            size=1.35,
            color_map=CAT_COLORS,
        ).move_to([-5.55, -0.35, 0])
        image.grid_lines.set_opacity(0.0)

        image_stage = self.caption(
            "IMAGE", x=-5.55, y=0.45, color_key="residual", size="caption"
        )
        patches_stage = self.caption(
            "PATCHES", x=-2.70, y=0.45, color_key="key", size="caption"
        )
        image_emb_stage = self.caption(
            "EMBEDDINGS", x=2.05, y=0.45, color_key="key", size="caption"
        )

        patch_target = 0.24
        patch_gap = 0.06
        patch_copies = []
        for patch in image.patches:
            copy = patch.copy().scale(patch_target / image.patch_size)
            copy.move_to(patch.get_center())
            patch_copies.append(copy)
        patch_row = VGroup(*patch_copies)

        row_width = IMAGE_TOKENS * patch_target + (IMAGE_TOKENS - 1) * patch_gap
        first_x = -2.70 - row_width / 2 + patch_target / 2
        target_centers = [
            [first_x + index * (patch_target + patch_gap), -0.35, 0]
            for index in range(IMAGE_TOKENS)
        ]

        projection = Node(
            NodeSpec("Linear Projection", "", "", "attention", 1),
            theme,
            width=1.75,
            height=0.62,
        ).move_to([-0.15, -0.35, 0])

        x_image = MatrixGrid(
            IMAGE_TOKENS,
            EMBED_DIM,
            theme,
            cell=0.11,
            gap=0.02,
            color_key="key",
        ).move_to([2.05, -0.35, 0])

        image_formula = MathTex(
            IMAGE_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("key"),
        ).move_to([2.05, -1.15, 0])
        if image_formula.width > 4.6:
            image_formula.scale_to_fit_width(4.6)

        patch_caption = self.caption(
            "one patch → one token vector", x=-2.70, y=-0.98
        )

        image_arrow1 = connect(image, patch_row, theme, buff=0.14)
        image_arrow2 = connect(patch_row, projection, theme, buff=0.14)
        image_arrow3 = connect(projection, x_image, theme, buff=0.14)

        # --- Convergence --------------------------------------------------
        text_caption = self.caption(
            "text tokens · N_text × d", x=-1.60, y=-1.15, color_key="value"
        )
        image_caption = self.caption(
            "image tokens · N_image × d", x=1.60, y=-1.15, color_key="key"
        )
        same_d = self.caption(
            "same embedding size d", x=0.0, y=-1.85
        )

        transformer = Node(
            NodeSpec("Transformer", "", "", "attention", 1),
            theme,
            width=4.00,
            height=0.50,
        ).move_to([0.0, -2.95, 0])

        # --- Animation ----------------------------------------------------
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(divider), run_time=0.8)

        # Text flow first (now on top)
        self.play(
            FadeIn(phrase, shift=RIGHT * 0.15),
            FadeIn(text_stage),
            run_time=0.7,
        )
        self.play(
            Indicate(phrase, color=theme.color("score"), scale_factor=1.02),
            run_time=0.6,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=DOWN * 0.08) for chip in chips],
                lag_ratio=0.12,
            ),
            Create(text_arrow1),
            FadeIn(tokenize_stage),
            run_time=1.0,
        )
        self.play(
            Indicate(
                chips[SUBWORD_INDICES[0]],
                color=theme.color("score"),
                scale_factor=1.08,
            ),
            Indicate(
                chips[SUBWORD_INDICES[1]],
                color=theme.color("score"),
                scale_factor=1.08,
            ),
            FadeIn(subword_caption),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                *[FadeIn(chip, shift=DOWN * 0.06) for chip in id_chips],
                lag_ratio=0.10,
            ),
            Create(text_arrow2),
            FadeIn(ids_stage),
            run_time=0.8,
        )
        self.play(
            FadeIn(lookup, shift=RIGHT * 0.1),
            Create(text_arrow3),
            FadeIn(lookup_stage),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                *[
                    Indicate(
                        lookup.row(index),
                        color=theme.color("key"),
                        scale_factor=1.04,
                    )
                    for index in range(len(TOKEN_PIECES))
                ],
                lag_ratio=0.15,
            ),
            run_time=1.2,
        )
        self.play(
            FadeIn(x_text, shift=RIGHT * 0.1),
            Create(text_arrow4),
            FadeIn(text_emb_stage),
            run_time=0.8,
        )
        self.play(Write(text_formula), run_time=0.7)

        # Image flow second (now on bottom)
        self.add(image)
        self.play(
            *image.reveal(),
            FadeIn(image_stage),
            run_time=0.9,
        )
        self.play(image.grid_lines.animate.set_opacity(1.0), run_time=0.6)
        self.add(patch_row)
        self.play(
            *[
                copy.animate.move_to(target)
                for copy, target in zip(patch_copies, target_centers)
            ],
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(
            Create(image_arrow1),
            FadeIn(patches_stage),
            FadeIn(patch_caption),
            run_time=0.6,
        )
        self.play(
            FadeIn(projection, shift=RIGHT * 0.1),
            Create(image_arrow2),
            run_time=0.7,
        )
        self.play(
            FadeIn(x_image, shift=RIGHT * 0.1),
            Create(image_arrow3),
            FadeIn(image_emb_stage),
            run_time=0.8,
        )
        self.play(Write(image_formula), run_time=0.7)
        self.play(
            Indicate(
                image.patches[0],
                color=theme.color("key"),
                scale_factor=1.08,
            ),
            Indicate(
                patch_row[0],
                color=theme.color("key"),
                scale_factor=1.08,
            ),
            Indicate(
                x_image.row(0),
                color=theme.color("key"),
                scale_factor=1.08,
            ),
            run_time=0.9,
        )

        # Clear the two flows, keeping only the embedding matrices.
        cleanup = [
            image,
            image_stage,
            patch_row,
            patches_stage,
            patch_caption,
            image_arrow1,
            projection,
            image_arrow2,
            image_arrow3,
            image_emb_stage,
            image_formula,
            phrase,
            text_stage,
            chips,
            tokenize_stage,
            subword_caption,
            text_arrow1,
            id_chips,
            ids_stage,
            text_arrow2,
            lookup,
            lookup_stage,
            text_arrow3,
            text_emb_stage,
            text_arrow4,
            text_formula,
        ]
        self.play(
            *[FadeOut(mobject, shift=DOWN * 0.08) for mobject in cleanup],
            x_image.animate.move_to([1.60, -1.85, 0]),
            x_text.animate.move_to([-1.60, -1.85, 0]),
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(
            FadeIn(image_caption),
            FadeIn(text_caption),
            FadeIn(same_d),
            run_time=0.6,
        )

        conv_arrow_image = connect(
            x_image,
            transformer,
            theme,
            color_key="key",
            buff=0.04,
            direction="vertical",
        )
        conv_arrow_text = connect(
            x_text,
            transformer,
            theme,
            color_key="value",
            buff=0.04,
            direction="vertical",
        )
        self.play(
            FadeIn(transformer, shift=DOWN * 0.1),
            Create(conv_arrow_image),
            Create(conv_arrow_text),
            run_time=0.9,
        )
        self.wait(2.5)
        