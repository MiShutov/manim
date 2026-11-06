"""Low-level visual primitives: graph nodes and residual connections."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import (
    Arrow,
    BOLD,
    Create,
    DOWN,
    FadeIn,
    LEFT,
    MathTex,
    Mobject,
    PI,
    RoundedRectangle,
    Text,
    Triangle,
    UP,
    VGroup,
    VMobject,
)

from schematics.specs import BadgeSpec, NodeSpec
from schematics.theme import Theme


class Node(VGroup):
    """Rounded rectangle with a centered label.

    Attributes:
        spec: The ``NodeSpec`` used to build this node.
        rect: Background ``RoundedRectangle``.
        label: ``Text`` or ``MathTex`` label inside the rectangle.
    """

    def __init__(
        self,
        spec: NodeSpec,
        theme: Theme,
        *,
        width: float,
        height: float,
    ):
        """Build a node from ``spec`` using theme geometry and colors."""
        self.spec = spec
        color = theme.color(spec.color)

        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=theme.node_radius,
            stroke_color=color,
            stroke_width=theme.node_stroke,
            fill_color=color,
            fill_opacity=theme.node_fill_opacity,
        )

        self.label = (
            MathTex(
                spec.label,
                font_size=theme.size(spec.label_size),
                color=theme.color("text"),
            )
            if spec.label_tex
            else Text(
                spec.label,
                font_size=theme.size(spec.label_size),
                font=theme.font,
            )
        )
        if self.label.width > width - 0.3:
            self.label.scale_to_fit_width(width - 0.3)
        self.label.move_to(self.rect.get_center())

        super().__init__(self.rect, self.label)


class Badge(VGroup):
    """Compact two-line block for a tensor name and its shape.

    Badges are used both for graph blocks and for strips of tensor shapes.
    A badge can mix plain text and LaTeX and can recolor selected title
    substrings, which keeps learned weights visually distinct.

    Attributes:
        spec: Source ``BadgeSpec``.
        rect: Rounded background rectangle.
        title_label: First line with the short name or formula.
        shape_label: Second line with the tensor shape.
    """

    def __init__(
        self,
        spec: BadgeSpec,
        theme: Theme,
        *,
        width: float | None = None,
        height: float = 0.64,
    ):
        """Build a badge from ``spec`` using theme colors and font sizes."""
        self.spec = spec
        color = theme.color(spec.color)
        shape_color = theme.color(spec.shape_color or spec.color)

        if spec.title_tex:
            highlight_keys = [substring for substring, _ in spec.title_highlights]
            math_kwargs = {
                "font_size": theme.size(spec.title_size),
                "color": theme.color("text"),
            }
            if highlight_keys:
                math_kwargs["substrings_to_isolate"] = highlight_keys
            self.title_label = MathTex(spec.title, **math_kwargs)
            for substring, color_key in spec.title_highlights:
                self.title_label.set_color_by_tex(
                    substring, theme.color(color_key)
                )
        else:
            self.title_label = Text(
                spec.title,
                font_size=theme.size(spec.title_size),
                font=theme.font,
                weight=BOLD,
                color=theme.color("text"),
            )

        if spec.shape_tex:
            self.shape_label = MathTex(
                spec.shape,
                font_size=theme.size("badge_shape"),
                color=shape_color,
            )
        else:
            self.shape_label = Text(
                spec.shape,
                font_size=theme.size("badge_shape"),
                font=theme.mono_font,
                color=shape_color,
            )

        pad = 0.16
        if width is None:
            width = max(self.title_label.width, self.shape_label.width) + 2 * pad
        max_text_width = width - 2 * pad
        for label in (self.title_label, self.shape_label):
            if label.width > max_text_width:
                label.scale_to_fit_width(max_text_width)

        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.10,
            stroke_color=color,
            stroke_width=1.6,
            fill_color=color,
            fill_opacity=0.10,
        )
        self.title_label.move_to(self.rect.get_center() + UP * 0.11)
        self.shape_label.move_to(self.rect.get_center() + DOWN * 0.12)

        super().__init__(self.rect, self.title_label, self.shape_label)

    def reveal(self):
        """Return a fade-in animation for the badge."""
        return [FadeIn(self, shift=DOWN * 0.08)]


class SkipConnection(VGroup):
    """Orthogonal residual path with an arrow head.

    Attributes:
        start: Index of the source node.
        end: Index of the destination node.
        points: Polyline vertices in scene coordinates.
        path: ``VMobject`` used for drawing and ``MoveAlongPath``.
        tip: Arrow head at the destination.
    """

    def __init__(
        self,
        points: Sequence[np.ndarray],
        theme: Theme,
        *,
        start: int,
        end: int,
        color_key: str = "residual",
    ):
        """Create a residual connection from the given polyline ``points``."""
        self.start = start
        self.end = end
        color = theme.color(color_key)
        self.points = [np.asarray(p, dtype=float) for p in points]

        self.path = VMobject(color=color, stroke_width=theme.skip_stroke)
        self.path.set_points_as_corners(self.points)

        self.tip = Triangle(
            fill_color=color,
            fill_opacity=1.0,
            stroke_width=0,
        ).scale(0.055)
        self.tip.rotate(-PI / 2)
        self.tip.move_to(self.points[-1] + LEFT * 0.02)

        super().__init__(self.path, self.tip)

    def reveal(self):
        """Return animations that draw the path and its arrow head."""
        return [Create(self.path), FadeIn(self.tip)]


def route_arrow(
    points: Sequence[np.ndarray],
    theme: Theme,
    *,
    color_key: str = "muted",
    stroke_width: float | None = None,
    tip_length: float = 0.11,
) -> VGroup:
    """Return an orthogonal polyline arrow through ``points``.

    Unlike :func:`connect`, this helper accepts an explicit route. Use it
    when a straight arrow would cross another block, for example when a
    value tensor has to travel around the attention core.

    Args:
        points: Polyline vertices from source to destination.
        theme: Theme supplying arrow color and stroke width.
        color_key: Semantic color key for the path.
        stroke_width: Optional override for the line width.
        tip_length: Length of the arrow head.
    """
    if len(points) < 2:
        raise ValueError("route_arrow needs at least two points")

    color = theme.color(color_key)
    vertices = [np.asarray(point, dtype=float) for point in points]
    path = VMobject(
        color=color,
        stroke_width=theme.arrow_stroke if stroke_width is None else stroke_width,
    )
    path.set_points_as_corners(vertices)

    tip = Triangle(
        fill_color=color,
        fill_opacity=1.0,
        stroke_width=0,
    ).scale_to_fit_height(tip_length)
    direction = vertices[-1] - vertices[-2]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError("the last two route points must differ")
    unit = direction / length
    tip.rotate(np.arctan2(unit[1], unit[0]) - PI / 2)
    extent = (
        abs(unit[0]) * tip.width / 2 + abs(unit[1]) * tip.height / 2
    )
    tip.move_to(vertices[-1] - unit * extent)

    return VGroup(path, tip)


def connect(
    start: Mobject,
    end: Mobject,
    theme: Theme,
    *,
    color_key: str = "muted",
    buff: float = 0.12,
    tip_length: float = 0.12,
    direction: str | None = None,
) -> Arrow:
    """Return a styled arrow from ``start`` to ``end``.

    The endpoints are selected from the dominant direction, so the arrow
    leaves and enters the mobjects at natural edges. Use ``direction`` to
    force ``"vertical"`` or ``"horizontal"`` routing. The forced endpoint is
    clamped to the target edge, so the arrow stays straight.

    Args:
        start: Source mobject.
        end: Target mobject.
        theme: Theme supplying arrow color and stroke width.
        color_key: Semantic color key for the arrow.
        buff: Gap between the arrow and the mobjects.
        tip_length: Arrow head length.
        direction: ``None`` for automatic routing, or ``"vertical"`` /
            ``"horizontal"`` to force one axis.
    """
    if direction not in {None, "vertical", "horizontal"}:
        raise ValueError("direction must be None, 'vertical', or 'horizontal'")

    delta = end.get_center() - start.get_center()
    if direction is None:
        direction = "horizontal" if abs(delta[0]) >= abs(delta[1]) else "vertical"

    if direction == "horizontal":
        if delta[0] >= 0:
            start_point, end_point = start.get_right(), end.get_left().copy()
        else:
            start_point, end_point = start.get_left(), end.get_right().copy()
        end_point[1] = np.clip(
            start.get_center()[1], end.get_bottom()[1], end.get_top()[1]
        )
    else:
        if delta[1] >= 0:
            start_point, end_point = start.get_top(), end.get_bottom().copy()
        else:
            start_point, end_point = start.get_bottom(), end.get_top().copy()
        end_point[0] = np.clip(
            start.get_center()[0], end.get_left()[0], end.get_right()[0]
        )

    return Arrow(
        start_point,
        end_point,
        buff=buff,
        stroke_width=theme.arrow_stroke,
        color=theme.color(color_key),
        tip_length=tip_length,
        max_tip_length_to_length_ratio=0.25,
    )
