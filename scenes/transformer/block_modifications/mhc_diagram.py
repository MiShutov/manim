"""Four-stream mHC diagram kept for a future, dedicated scene.

This module is not imported by the current block scene: the scene only marks
the residual paths with mHC blocks. The full schematic is preserved here so a
later scene can show the four residual streams, the bypass and compute
pre-mappings, the shared chain, and the post mapping in detail.

The diagram is built in scene coordinates and returned as one VGroup.
"""

from __future__ import annotations

from manim import (
    DOWN,
    Arrow,
    Circle,
    Line,
    MathTex,
    RoundedRectangle,
    Text,
    VGroup,
)

from schematics.theme import Theme


def _text(value: str, theme: Theme, *, size: float = 11, color_key: str = "text") -> Text:
    """Return a plain text label in the theme font."""
    return Text(value, font_size=size, font=theme.font, color=theme.color(color_key))


def _box(width, height, theme, *, color_key, stroke_width=1.3, fill_opacity=0.10, corner=0.06):
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


# --- mHC: four residual streams ---------------------------------------------
MHC_ROWS = (2.90, 2.10, 1.30, 0.50)
MHC_CARD_W = 0.95
MHC_CARD_H = 0.38
PRE_H = 2.60
CHAIN_W = 1.10
CHAIN_H = 0.30


def _mhc_card(theme: Theme, label: str, x: float, y: float) -> VGroup:
    """Return one axis-aligned activation card."""
    rect = RoundedRectangle(
        width=MHC_CARD_W,
        height=MHC_CARD_H,
        corner_radius=0.05,
        stroke_color=theme.color("activation"),
        stroke_width=1.3,
        fill_color=theme.color("activation"),
        fill_opacity=0.12,
    )
    text = MathTex(
        label, font_size=12, color=theme.color("activation")
    ).move_to(rect)
    return VGroup(rect, text).move_to([x, y, 0])


def _mhc_plus(theme: Theme, x: float, y: float) -> VGroup:
    """Return one circled plus add node."""
    return VGroup(
        Circle(
            radius=0.13,
            stroke_color=theme.color("residual"),
            stroke_width=1.4,
            fill_color=theme.color("residual"),
            fill_opacity=0.12,
        ),
        MathTex("+", font_size=15, color=theme.color("residual")),
    ).move_to([x, y, 0])


def _mhc_arrow(theme: Theme, start, end, *, color_key: str = "muted"):
    """Return a thin connector between two points."""
    return Arrow(
        start,
        end,
        buff=0.02,
        stroke_width=theme.arrow_stroke,
        color=theme.color(color_key),
        tip_length=0.06,
    )


def _mhc_node(theme: Theme, label: str, x: float, y: float) -> VGroup:
    """Return one chain node."""
    rect = _box(CHAIN_W, CHAIN_H, theme, color_key="compute")
    text = _text(label, theme, size=11).move_to(rect)
    return VGroup(rect, text).move_to([x, y, 0])


def _mhc_stage(
    theme: Theme,
    *,
    x_in: float,
    x_bus: float,
    x_pre: float,
    x_chain: float,
    x_plus: float,
    x_out: float,
    in_tex: tuple[str, ...],
    out_tex: tuple[str, ...],
    mid_label: str,
    draw_in_cards: bool,
) -> tuple[VGroup, VGroup]:
    """Build one mHC stage; return the stage group and the input cards."""
    group = VGroup()
    in_cards = VGroup()
    if draw_in_cards:
        in_cards = VGroup(
            *[
                _mhc_card(theme, tex, x_in, y)
                for tex, y in zip(in_tex, MHC_ROWS)
            ]
        )
        group.add(in_cards)

    # Collector bus and the four card arrows.
    group.add(
        Line(
            [x_bus, MHC_ROWS[0], 0],
            [x_bus, MHC_ROWS[-1] - 0.18, 0],
            stroke_width=theme.arrow_stroke,
            color=theme.color("muted"),
        )
    )
    for y in MHC_ROWS:
        group.add(
            _mhc_arrow(
                theme,
                [x_in + MHC_CARD_W / 2, y, 0],
                [x_bus, y, 0],
            )
        )

    # Bypass pre block: four inputs, four outputs.
    bypass = _box(0.70, PRE_H, theme, color_key="weights")
    bypass.move_to([x_pre, 1.70, 0])
    bypass_label = _text("pre", theme, size=11, color_key="weights")
    group.add(bypass, bypass_label.move_to(bypass))
    for y in MHC_ROWS:
        group.add(
            _mhc_arrow(
                theme, [x_bus, y, 0], [x_pre - 0.35, y, 0]
            )
        )
        group.add(
            _mhc_arrow(
                theme,
                [x_pre + 0.35, y, 0],
                [x_plus - 0.13, y, 0],
                color_key="residual",
            )
        )

    # Compute pre block below the band: four inputs, one output.
    compute = _box(1.40, 0.36, theme, color_key="weights")
    compute.move_to([x_pre, -0.50, 0])
    compute_label = _text("pre", theme, size=11, color_key="weights")
    group.add(compute, compute_label.move_to(compute))
    group.add(
        _mhc_arrow(
            theme, [x_bus, MHC_ROWS[-1] - 0.18, 0], [x_bus, -0.50, 0]
        )
    )
    group.add(
        _mhc_arrow(
            theme,
            [x_pre + 0.70, -0.50, 0],
            [x_chain - CHAIN_W / 2, -0.50, 0],
        )
    )

    # Normalization -> attention/MLP -> post mapping chain.
    chain_ys = (-0.50, -1.15, -1.80)
    nodes = [
        _mhc_node(theme, "Normalization", x_chain, chain_ys[0]),
        _mhc_node(theme, mid_label, x_chain, chain_ys[1]),
        _mhc_node(theme, "Post", x_chain, chain_ys[2]),
    ]
    group.add(*nodes)
    for upper, lower in zip(nodes[:-1], nodes[1:]):
        group.add(
            _mhc_arrow(
                theme,
                [x_chain, upper.get_bottom()[1], 0],
                [x_chain, lower.get_top()[1], 0],
            )
        )

    # The post mapping turns the single arrow back into four outputs.
    for index, y in enumerate(MHC_ROWS):
        origin_y = chain_ys[2] + 0.10 - 0.066 * index
        group.add(
            _mhc_arrow(
                theme,
                [x_chain - CHAIN_W / 2, origin_y, 0],
                [x_plus, y - 0.13, 0],
            )
        )

    # Add nodes and the output card column.
    for y in MHC_ROWS:
        group.add(_mhc_plus(theme, x_plus, y))
        group.add(
            _mhc_arrow(
                theme,
                [x_plus + 0.13, y, 0],
                [x_out - MHC_CARD_W / 2, y, 0],
            )
        )
    out_cards = VGroup(
        *[
            _mhc_card(theme, tex, x_out, y)
            for tex, y in zip(out_tex, MHC_ROWS)
        ]
    )
    group.add(out_cards)
    return group, in_cards


def build_mhc(theme: Theme) -> VGroup:
    """Build the four-stream mHC diagram in scene coordinates."""
    in_labels = tuple(rf"x_{{{i}}}" for i in range(1, 5))
    prime_labels = tuple(rf"x^{{\prime}}_{{{i}}}" for i in range(1, 5))
    double_labels = tuple(rf"x^{{\prime\prime}}_{{{i}}}" for i in range(1, 5))

    stage_a, _ = _mhc_stage(
        theme,
        x_in=-6.35,
        x_bus=-5.55,
        x_pre=-4.90,
        x_chain=-0.90,
        x_plus=-2.00,
        x_out=0.60,
        in_tex=in_labels,
        out_tex=prime_labels,
        mid_label="Attention",
        draw_in_cards=True,
    )
    stage_b, _ = _mhc_stage(
        theme,
        x_in=0.60,
        x_bus=1.35,
        x_pre=2.00,
        x_chain=5.70,
        x_plus=4.80,
        x_out=6.55,
        in_tex=prime_labels,
        out_tex=double_labels,
        mid_label="MLP",
        draw_in_cards=False,
    )
    return VGroup(stage_a, stage_b).shift(DOWN * 0.4)
