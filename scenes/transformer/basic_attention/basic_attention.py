"""Classical multi-head attention, explained one head at a time.

The top strip shows the full tensor shapes. The detailed view follows a
single head: queries meet keys, logits become softmax weights, and the
weights mix the values into one output vector.

Run with:

    python scripts/render.py attention -q l
"""

from __future__ import annotations

from manim import (
    DOWN,
    RIGHT,
    Create,
    FadeIn,
    Flash,
    Indicate,
    LaggedStart,
    MathTex,
    Text,
    VGroup,
    Write,
)

from schematics import (
    Badge,
    MatrixGrid,
    SchematicScene,
    WeightBars,
    connect,
)
from scenes.transformer.basic_attention.specs import (
    ATTENTION_BADGES,
    HEAD_DIM,
    LOGITS,
    QUERY_TOKEN,
    SCHEMATIC_DIM,
    TOKENS,
    WEIGHTS,
)


class ClassicalAttention(SchematicScene):
    """Animated explainer for classical multi-head attention without GQA."""

    def construct(self):
        """Build the tensor-shape strip and the single-head detail view."""
        theme = self.theme
        title = self.title("Classical multi-head attention", x=0.0)

        badges = VGroup(
            *[Badge(spec, theme, width=2.0) for spec in ATTENTION_BADGES]
        ).arrange(RIGHT, buff=0.14).move_to([0, 2.98, 0])

        # One head shown schematically: 5 tokens and 4 of 128 dimensions.
        query = MatrixGrid(SCHEMATIC_DIM, 1, theme, color_key="query")
        keys = MatrixGrid(
            len(TOKENS),
            SCHEMATIC_DIM,
            theme,
            color_key="key",
            row_labels=TOKENS,
        )
        scores = WeightBars(
            TOKENS,
            LOGITS,
            theme,
            color_key="score",
            signed=True,
            value_fmt="{:.2f}",
        )
        weights = WeightBars(
            TOKENS,
            WEIGHTS,
            theme,
            color_key="weight",
            value_fmt="{:.3f}",
        )
        values = MatrixGrid(
            len(TOKENS),
            SCHEMATIC_DIM,
            theme,
            color_key="value",
            row_labels=TOKENS,
            highlight_rows=(0,),
        )
        output = MatrixGrid(SCHEMATIC_DIM, 1, theme, color_key="output")

        query.move_to([-5.95, 1.20, 0])
        keys.move_to([-3.85, 1.20, 0])
        scores.move_to([-0.85, 1.20, 0])
        weights.move_to([-0.85, -1.30, 0])
        values.move_to([1.75, -1.30, 0])
        output.move_to([3.95, -1.30, 0])

        q_label = self.caption(
            f"query q ({QUERY_TOKEN})", x=-5.95, y=2.16, color_key="query"
        )
        k_label = self.caption(
            "keys K  ·  5 tokens", x=-3.85, y=2.16, color_key="key"
        )
        s_label = self.caption(
            "scores  q · Kᵀ / √d", x=-0.85, y=2.16, color_key="score"
        )
        w_label = self.caption(
            "softmax", x=-2.65, y=-1.30, color_key="weight"
        )
        v_label = self.caption(
            "values V", x=1.75, y=-0.28, color_key="value"
        )
        o_label = self.caption(
            "output", x=3.95, y=-0.28, color_key="output"
        )

        arrow_qk = connect(query, keys, theme, buff=0.18)
        arrow_ks = connect(keys, scores, theme, buff=0.18)
        arrow_sw = connect(scores, weights, theme, buff=0.16)
        arrow_wv = connect(weights, values, theme, buff=0.18)
        arrow_vo = connect(values, output, theme, buff=0.20)

        dot_label = self.caption(
            "q · Kᵀ", x=-2.40, y=1.50, color_key="muted"
        )
        mix_label = self.caption(
            "w · V", x=0.55, y=-1.05, color_key="muted"
        )

        formula = MathTex(
            r"\mathrm{output}(t_3) = \sum_j w_j\, v_j",
            font_size=theme.size("outro"),
            color=theme.color("output"),
        ).move_to([1.10, -2.55, 0])
        dim_caption = self.caption(
            f"{SCHEMATIC_DIM} of {HEAD_DIM} dims shown",
            x=4.85,
            y=-2.55,
        )

        note = Text(
            "All 32 heads have separate K and V projections — classical MHA, no GQA.",
            font_size=theme.size("outro"),
            font=theme.font,
            color=theme.color("residual"),
        ).move_to([0, -3.35, 0])
        if note.width > 13.4:
            note.scale_to_fit_width(13.4)

        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)
        self.play(
            LaggedStart(
                *[FadeIn(badge, shift=DOWN * 0.10) for badge in badges],
                lag_ratio=0.08,
            ),
            run_time=1.2,
        )

        self.play(FadeIn(q_label), FadeIn(k_label), FadeIn(s_label), run_time=0.5)
        self.play(*query.reveal(), *keys.reveal(), run_time=1.1)
        self.play(*scores.reveal(), run_time=0.8)
        self.play(
            Create(arrow_qk),
            Create(arrow_ks),
            FadeIn(dot_label),
            run_time=0.7,
        )

        self.play(Create(arrow_sw), run_time=0.3)
        self.play(*weights.reveal(), FadeIn(w_label), run_time=0.8)

        self.play(*values.reveal(), FadeIn(v_label), run_time=0.9)
        self.play(Create(arrow_wv), FadeIn(mix_label), run_time=0.5)

        self.play(
            Create(arrow_vo),
            *output.reveal(),
            FadeIn(o_label),
            run_time=0.8,
        )

        self.play(
            Indicate(
                weights.rows[0],
                color=theme.color("weight"),
                scale_factor=1.05,
            ),
            Indicate(
                values.row(0),
                color=theme.color("value"),
                scale_factor=1.05,
            ),
            run_time=0.9,
        )

        self.play(
            Write(formula),
            FadeIn(dim_caption),
            Flash(output, color=theme.color("success"), flash_radius=0.55),
            run_time=1.1,
        )

        self.play(Write(note), run_time=1.0)
        self.wait(2)
