"""Educational scene: tokens are sequence elements, not only text.

The upper flow turns a sentence into subword tokens, token ids, and then into
embedding vectors. The lower flow turns an image into patches and then into
embedding vectors. Both streams converge on the same Transformer block.

Run with:

    python scripts/render.py token_anatomy -q l
"""

from __future__ import annotations

import numpy as np
from manim import (
    Arrow,
    DOWN,
    RIGHT,
    Create,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    MathTex,
    RoundedRectangle,
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
        # The row is laid out from the measured widths of its stages, so the
        # elements never overlap: phrase -> tokens -> ids -> lookup -> matrix.
        text_row_y = 2.15
        text_gap = 0.44

        phrase = Text(
            TEXT_PROMPT,
            font_size=15,
            font=theme.font,
            color=theme.color("text"),
        )

        chips = VGroup(
            *[
                TokenChip(
                    piece,
                    theme,
                    color_key="score" if index in SUBWORD_INDICES else "key",
                    pad=0.12,
                )
                for index, piece in enumerate(TOKEN_PIECES)
            ]
        ).arrange(RIGHT, buff=0.08)

        id_chips = VGroup(
            *[
                TokenChip(
                    token_id,
                    theme,
                    color_key="muted",
                    height=0.30,
                    pad=0.07,
                    mono=True,
                    size="small",
                )
                for token_id in TOKEN_IDS
            ]
        ).arrange(RIGHT, buff=0.06)

        # Lookup as a function of the token id: a purple node whose label is
        # the LaTeX formula x_i = Embedding(id_i).
        lookup_rect = RoundedRectangle(
            width=1.7,
            height=0.62,
            corner_radius=theme.node_radius,
            stroke_color=theme.color("attention"),
            stroke_width=theme.node_stroke,
            fill_color=theme.color("attention"),
            fill_opacity=theme.node_fill_opacity,
        )
        lookup_formula = MathTex(
            r"x_i = \mathrm{Embedding}(\mathrm{id}_i)",
            font_size=theme.size("caption"),
            color=theme.color("text"),
        ).move_to(lookup_rect.get_center())
        lookup = VGroup(lookup_rect, lookup_formula)

        x_text = MatrixGrid(
            len(TOKEN_PIECES),
            EMBED_DIM,
            theme,
            cell=0.16,
            gap=0.03,
            color_key="value",
        )

        # Distribute the stages over the full width with equal gaps.
        text_row = (phrase, chips, id_chips, lookup, x_text)
        row_width = sum(item.width for item in text_row) + text_gap * (
            len(text_row) - 1
        )
        cursor = -row_width / 2.0
        for item in text_row:
            item.move_to([cursor + item.width / 2.0, text_row_y, 0])
            cursor += item.width + text_gap

        text_stage = self.caption(
            "TEXT",
            x=phrase.get_center()[0],
            y=3.00,
            color_key="key",
            size="caption",
        )
        tokenization_stage = self.caption(
            "tokenization",
            x=chips.get_center()[0],
            y=1.72,
            color_key="key",
            size="caption",
        )
        ids_stage = self.caption(
            "TOKEN IDS",
            x=id_chips.get_center()[0],
            y=3.00,
            color_key="muted",
            size="caption",
        )
        embedding_lookup_stage = self.caption(
            "embedding lookup",
            x=lookup.get_center()[0],
            y=1.48,
            color_key="attention",
            size="caption",
        )
        text_emb_stage = self.caption(
            "EMBEDDINGS",
            x=x_text.get_center()[0],
            y=3.00,
            color_key="value",
            size="caption",
        )

        text_formula = MathTex(
            TEXT_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("value"),
        )
        # Keep the formula narrow enough to clear the "embedding lookup"
        # caption standing to its left.
        if text_formula.width > 1.4:
            text_formula.scale_to_fit_width(1.4)
        text_formula.move_to([x_text.get_center()[0], 1.47, 0])
        # d — embedding dimension, kept clear of the divider line below.
        d_caption = self.caption(
            "d — embedding dimension",
            x=x_text.get_center()[0],
            y=1.16,
            color_key="muted",
        )

        text_arrow1 = connect(phrase, chips, theme, buff=0.08, tip_length=0.10)
        text_arrow2 = connect(chips, id_chips, theme, buff=0.08, tip_length=0.10)
        text_arrow3 = connect(id_chips, lookup, theme, buff=0.08, tip_length=0.10)
        text_arrow4 = connect(lookup, x_text, theme, buff=0.08, tip_length=0.10)

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
            "EMBEDDINGS", x=2.60, y=0.45, color_key="key", size="caption"
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

        # Vectorization as a function of one patch: a purple node whose label
        # is the LaTeX formula x_i = vectorize(patch_i).
        vectorization_rect = RoundedRectangle(
            width=2.0,
            height=0.62,
            corner_radius=theme.node_radius,
            stroke_color=theme.color("attention"),
            stroke_width=theme.node_stroke,
            fill_color=theme.color("attention"),
            fill_opacity=theme.node_fill_opacity,
        ).move_to([0.60, -0.35, 0])
        vectorization_formula = MathTex(
            r"x_i = \mathrm{vectorize}(\mathrm{patch}_i)",
            font_size=theme.size("caption"),
            color=theme.color("text"),
        ).move_to(vectorization_rect.get_center())
        vectorization = VGroup(vectorization_rect, vectorization_formula)
        vectorization_stage = self.caption(
            "vectorization", x=0.60, y=-1.05, color_key="attention", size="caption"
        )

        x_image = MatrixGrid(
            IMAGE_TOKENS,
            EMBED_DIM,
            theme,
            cell=0.11,
            gap=0.02,
            color_key="key",
        ).move_to([2.60, -0.35, 0])

        image_formula = MathTex(
            IMAGE_FORMULA,
            font_size=theme.size("outro"),
            color=theme.color("key"),
        ).move_to([2.60, -1.15, 0])
        if image_formula.width > 4.6:
            image_formula.scale_to_fit_width(4.6)

        patchification_stage = self.caption(
            "patchification", x=-2.70, y=-0.98, color_key="key", size="caption"
        )

        # Arrow from the image to the patch row ("to the patches"). It runs
        # through the gap between the image and the row so it never crosses
        # the patches themselves.
        patch_row_left = first_x - patch_target / 2
        patch_row_right = (
            first_x + (IMAGE_TOKENS - 1) * (patch_target + patch_gap)
            + patch_target / 2
        )
        image_arrow1 = Arrow(
            image.get_right(),
            np.array([patch_row_left - 0.05, -0.35, 0]),
            buff=0.10,
            stroke_width=theme.arrow_stroke,
            color=theme.color("muted"),
            tip_length=0.10,
            max_tip_length_to_length_ratio=0.25,
        )
        # Arrow from the patch row to vectorization ("from the patches").
        image_arrow2 = Arrow(
            np.array([patch_row_right + 0.05, -0.35, 0]),
            np.array([vectorization.get_left()[0] - 0.05, -0.35, 0]),
            buff=0.10,
            stroke_width=theme.arrow_stroke,
            color=theme.color("muted"),
            tip_length=0.10,
            max_tip_length_to_length_ratio=0.25,
        )
        image_arrow3 = connect(vectorization, x_image, theme, buff=0.14, tip_length=0.11)

        # --- Convergence --------------------------------------------------
        # Final composition is centered on the screen: labels on top, matrices
        # below them, then the two token streams entering the Transformer.
        text_caption = self.caption(
            "text tokens · N_text × d", x=-1.60, y=1.25, color_key="value"
        )
        image_caption = self.caption(
            "image tokens · N_image × d", x=1.60, y=1.25, color_key="key"
        )

        transformer = Node(
            NodeSpec("Transformer", "", "", "attention", 1),
            theme,
            width=4.00,
            height=0.50,
        ).move_to([0.0, -1.15, 0])

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
            FadeIn(tokenization_stage),
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
            FadeIn(embedding_lookup_stage),
            run_time=0.8,
        )
        self.play(
            Indicate(
                lookup,
                color=theme.color("attention"),
                scale_factor=1.08,
            ),
            run_time=1.2,
        )
        self.play(
            FadeIn(x_text, shift=RIGHT * 0.1),
            Create(text_arrow4),
            FadeIn(text_emb_stage),
            # The dimension d is explained the moment the green text
            # embeddings first appear.
            FadeIn(d_caption),
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
            FadeIn(patchification_stage),
            run_time=0.6,
        )
        self.play(
            FadeIn(vectorization, shift=RIGHT * 0.1),
            Create(image_arrow2),
            FadeIn(vectorization_stage),
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

        # Clear the two flows, keeping only the embedding matrices. The divider
        # goes away too: it only separated the two modality rows.
        cleanup = [
            divider,
            image,
            image_stage,
            patch_row,
            patches_stage,
            patchification_stage,
            image_arrow1,
            vectorization,
            vectorization_stage,
            image_arrow2,
            image_arrow3,
            image_emb_stage,
            embedding_lookup_stage,
            image_formula,
            phrase,
            text_stage,
            chips,
            tokenization_stage,
            text_arrow1,
            id_chips,
            ids_stage,
            text_arrow2,
            lookup,
            text_arrow3,
            text_emb_stage,
            text_arrow4,
            text_formula,
        ]
        self.play(
            *[FadeOut(mobject, shift=DOWN * 0.08) for mobject in cleanup],
            x_image.animate.move_to([1.60, 0.40, 0]),
            x_text.animate.move_to([-1.60, 0.40, 0]),
            d_caption.animate.move_to([0.0, -0.45, 0]),
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(
            FadeIn(image_caption),
            FadeIn(text_caption),
            run_time=0.6,
        )

        conv_arrow_image = connect(
            x_image,
            transformer,
            theme,
            color_key="key",
            buff=0.10,
            direction="vertical",
            tip_length=0.12,
        )
        conv_arrow_text = connect(
            x_text,
            transformer,
            theme,
            color_key="value",
            buff=0.10,
            direction="vertical",
            tip_length=0.12,
        )
        self.play(
            FadeIn(transformer, shift=DOWN * 0.1),
            Create(conv_arrow_image),
            Create(conv_arrow_text),
            run_time=0.9,
        )
        self.wait(2.5)
        