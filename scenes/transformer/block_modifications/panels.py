"""Compact schematics used by the transformer-block swap-ins.

Every panel is built around the origin so the scene can place it inside an
expanded block. The mHC diagram is the only absolute-coordinate assembly: it
replaces the whole block, so it is laid out on the scene grid directly.

Color grammar:
    - purple: computations (normalization, attention, MLP, router, experts);
    - green: activations (cards, shapes, mask cells);
    - yellow: learned mappings (``pre`` / ``post``) and routing;
    - orange: residual streams, add nodes, and bypass arrows.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Arrow,
    Circle,
    FadeIn,
    LaggedStart,
    MathTex,
    RoundedRectangle,
    Text,
    VGroup,
)

from schematics import route_arrow
from schematics.theme import Theme

# Expanded-block sizes used by the scene.
ATT_PANEL = (2.8, 1.1)
MLP_PANEL = (2.8, 1.6)


def _text(
    value: str,
    theme: Theme,
    *,
    size: float = 12,
    color_key: str = "text",
) -> Text:
    """Return a plain text label in the theme font."""
    return Text(
        value,
        font_size=size,
        font=theme.font,
        color=theme.color(color_key),
    )


def _box(
    width: float,
    height: float,
    theme: Theme,
    *,
    color_key: str,
    stroke_width: float = 1.3,
    fill_opacity: float = 0.10,
    corner: float = 0.06,
) -> RoundedRectangle:
    """Return a small rounded block outline."""
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=corner,
        stroke_color=theme.color(color_key),
        stroke_width=stroke_width,
        fill_color=theme.color(color_key),
        fill_opacity=fill_opacity,
    )


def norm_chip(theme: Theme, label: str = "norm", *, width: float = 0.9) -> VGroup:
    """Return a small normalization block used inside expanded modules.

    The fill is opaque so the underlying connector is hidden behind the chip.
    """
    rect = RoundedRectangle(
        width=width,
        height=0.20,
        corner_radius=0.06,
        stroke_color=theme.color("compute"),
        stroke_width=1.3,
        fill_color=theme.color("panel"),
        fill_opacity=1.0,
    )
    text = _text(label, theme, size=10).move_to(rect)
    return VGroup(rect, text)


class MaskGrid(VGroup):
    """Causal or sliding-window mask drawn as a small cell grid."""

    def __init__(
        self,
        size: int,
        theme: Theme,
        *,
        window: int | None = None,
        cell: float = 0.082,
        gap: float = 0.015,
    ):
        """Build a centered ``size`` by ``size`` mask grid."""
        pitch = cell + gap
        total = size * cell + (size - 1) * gap
        half = total / 2

        self.allowed = VGroup()
        self.blocked = VGroup()
        self.rows = VGroup()

        for row in range(size):
            row_cells = VGroup()
            for col in range(size):
                open_cell = (
                    col <= row
                    if window is None
                    else 0 <= row - col < window
                )
                color = theme.color("activation" if open_cell else "muted")
                square = RoundedRectangle(
                    width=cell,
                    height=cell,
                    corner_radius=0.015,
                    stroke_color=color,
                    stroke_width=0.8 if open_cell else 0.6,
                    fill_color=color,
                    fill_opacity=0.45 if open_cell else 0.06,
                ).move_to(
                    [
                        -half + cell / 2 + col * pitch,
                        half - cell / 2 - row * pitch,
                        0,
                    ]
                )
                row_cells.add(square)
                (self.allowed if open_cell else self.blocked).add(square)
            self.rows.add(row_cells)

        super().__init__(*self.rows)

    def reveal(self):
        """Return a row-by-row fade-in animation."""
        return [
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.03) for row in self.rows],
                lag_ratio=0.08,
            )
        ]


def _mask_legend(theme: Theme, *, windowed: bool) -> VGroup:
    """Return the two-line legend of the mask panel."""
    if windowed:
        rule = MathTex(
            r"0 \le i - j < w",
            font_size=11,
            color=theme.color("activation"),
        )
        note = _text("local window", theme, size=10, color_key="muted")
        return VGroup(rule, note).arrange(DOWN, buff=0.05, aligned_edge=LEFT)

    keep = VGroup(
        _box(0.08, 0.08, theme, color_key="activation", fill_opacity=0.45),
        MathTex(r"0", font_size=11, color=theme.color("activation")),
        _text("keep", theme, size=10, color_key="muted"),
    ).arrange(RIGHT, buff=0.05)
    drop = VGroup(
        _box(0.08, 0.08, theme, color_key="muted", fill_opacity=0.05),
        MathTex(r"-\infty", font_size=11, color=theme.color("muted")),
        _text("mask", theme, size=10, color_key="muted"),
    ).arrange(RIGHT, buff=0.05)
    return VGroup(keep, drop).arrange(DOWN, buff=0.05, aligned_edge=LEFT)


def causal_mask_panel(theme: Theme) -> VGroup:
    """Return the causal-mask content of the expanded attention block."""
    title = _text("causal mask", theme, size=12)
    title.move_to([-1.25, 0.36, 0], aligned_edge=LEFT)
    grid = MaskGrid(5, theme).move_to([-0.85, -0.12, 0])
    legend = _mask_legend(theme, windowed=False).move_to(
        [0.10, -0.12, 0], aligned_edge=LEFT
    )
    return VGroup(title, grid, legend)


def windowed_mask_panel(theme: Theme) -> VGroup:
    """Return the sliding-window mask content."""
    title = _text("windowed attention", theme, size=12)
    title.move_to([-1.25, 0.36, 0], aligned_edge=LEFT)
    grid = MaskGrid(5, theme, window=3).move_to([-0.85, -0.12, 0])
    legend = _mask_legend(theme, windowed=True).move_to(
        [0.10, -0.12, 0], aligned_edge=LEFT
    )
    return VGroup(title, grid, legend)


def gdn_panel(theme: Theme) -> VGroup:
    """Return a miniature Gated DeltaNet recurrent-state schematic."""
    title = _text("Gated DeltaNet", theme, size=12)
    title.move_to([-1.25, 0.36, 0], aligned_edge=LEFT)

    state_in = Circle(
        radius=0.13,
        stroke_color=theme.color("activation"),
        stroke_width=1.3,
        fill_color=theme.color("activation"),
        fill_opacity=0.12,
    ).move_to([-1.05, -0.12, 0])
    state_in_label = MathTex(
        r"S_{t-1}", font_size=11, color=theme.color("activation")
    ).move_to(state_in)

    gate_box = _box(0.90, 0.28, theme, color_key="compute")
    gate_box.move_to([0.0, -0.12, 0])
    gate_label = MathTex(
        r"g_t", font_size=12, color=theme.color("text")
    ).move_to(gate_box)

    state_out = Circle(
        radius=0.13,
        stroke_color=theme.color("activation"),
        stroke_width=1.3,
        fill_color=theme.color("activation"),
        fill_opacity=0.12,
    ).move_to([1.05, -0.12, 0])
    state_out_label = MathTex(
        r"S_t", font_size=11, color=theme.color("activation")
    ).move_to(state_out)

    arrow_in = Arrow(
        state_in.get_right(),
        gate_box.get_left(),
        buff=0.03,
        stroke_width=theme.arrow_stroke,
        color=theme.color("muted"),
        tip_length=0.06,
    )
    arrow_out = Arrow(
        gate_box.get_right(),
        state_out.get_left(),
        buff=0.03,
        stroke_width=theme.arrow_stroke,
        color=theme.color("muted"),
        tip_length=0.06,
    )
    loop = route_arrow(
        [
            [1.05, -0.25, 0],
            [1.05, -0.40, 0],
            [-1.05, -0.40, 0],
            [-1.05, -0.25, 0],
        ],
        theme,
        color_key="muted",
        stroke_width=theme.arrow_stroke,
        tip_length=0.06,
    )

    return VGroup(
        title,
        state_in,
        state_in_label,
        gate_box,
        gate_label,
        state_out,
        state_out_label,
        arrow_in,
        arrow_out,
        loop,
    )


class MoePanel(VGroup):
    """Vertical router -> experts -> plus panel.

    The three router arrows touch the router body, the centre expert feeds the
    plus through a thick vertical arrow, and the side experts enter the plus
    from its sides.

    Attributes:
        router: Router block (used for the routing pulse).
        arrows: The three router-to-expert arrows.
        experts: The three expert groups.
        plus: The output add node.
    """

    def __init__(self, theme: Theme):
        """Build the panel centered at the origin."""
        router = _box(1.75, 0.30, theme, color_key="compute")
        router.move_to([0.0, 0.55, 0])
        router_label = _text("router", theme, size=11).move_to(router)
        self.router = router

        self.arrows = VGroup()
        self.experts = VGroup()
        for index, x in enumerate((-0.85, 0.0, 0.85)):
            expert = _box(0.75, 0.28, theme, color_key="compute")
            expert.move_to([x, -0.10, 0])
            label = _text(f"Expert {index + 1}", theme, size=10).move_to(expert)
            self.experts.add(VGroup(expert, label))
            self.arrows.add(
                Arrow(
                    [x, 0.40, 0],
                    [x, 0.04, 0],
                    buff=0.01,
                    stroke_width=theme.arrow_stroke,
                    color=theme.color("muted"),
                    tip_length=0.05,
                )
            )

        self.plus = VGroup(
            Circle(
                radius=0.11,
                stroke_color=theme.color("residual"),
                stroke_width=1.4,
                fill_color=theme.color("residual"),
                fill_opacity=0.12,
            ),
            MathTex("+", font_size=14, color=theme.color("residual")),
        ).move_to([0.0, -0.52, 0])

        # The centre expert feeds the add node from above; the side experts
        # enter it from the sides.
        self.center_out = Arrow(
            [0.0, -0.24, 0],
            [0.0, -0.41, 0],
            buff=0.01,
            stroke_width=3.2,
            color=theme.color("muted"),
            tip_length=0.06,
        )
        self.side_out = VGroup(
            Arrow(
                [-0.85, -0.24, 0],
                [-0.11, -0.52, 0],
                buff=0.01,
                stroke_width=theme.arrow_stroke,
                color=theme.color("muted"),
                tip_length=0.05,
            ),
            Arrow(
                [0.85, -0.24, 0],
                [0.11, -0.52, 0],
                buff=0.01,
                stroke_width=theme.arrow_stroke,
                color=theme.color("muted"),
                tip_length=0.05,
            ),
        )
        self.out_arrow = Arrow(
            [0.0, -0.63, 0],
            [0.0, -0.80, 0],
            buff=0.01,
            stroke_width=theme.arrow_stroke,
            color=theme.color("muted"),
            tip_length=0.05,
        )
        in_arrow = Arrow(
            [0.0, 0.80, 0],
            [0.0, 0.70, 0],
            buff=0.01,
            stroke_width=theme.arrow_stroke,
            color=theme.color("muted"),
            tip_length=0.05,
        )

        super().__init__(
            in_arrow,
            router,
            router_label,
            self.arrows,
            self.experts,
            self.center_out,
            self.side_out,
            self.plus,
            self.out_arrow,
        )
