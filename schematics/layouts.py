"""Graph layouts: vertical pipelines and residual routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from manim import (
    Arrow,
    Create,
    Dot,
    FadeIn,
    LaggedStart,
    MoveAlongPath,
    RIGHT,
    VGroup,
)

from schematics.primitives import Node, SkipConnection
from schematics.specs import NodeSpec, SkipSpec
from schematics.theme import Theme


@dataclass
class Pipeline:
    """Result of laying out a vertical graph.

    Attributes:
        nodes: Node mobjects in top-to-bottom order.
        arrows: Connectors between consecutive nodes.
        skips: Residual connections routed on the left side.
        x: Horizontal center of the node column.
        ys: Vertical center of each node.
        width: Node width.
        height: Node height.
        spacing: Distance between consecutive node centers.
    """

    nodes: list[Node]
    arrows: VGroup
    skips: list[SkipConnection]
    x: float
    ys: list[float]
    width: float
    height: float
    spacing: float

    @property
    def all(self) -> VGroup:
        """Return every pipeline mobject in one group."""
        return VGroup(*self.nodes, self.arrows, *self.skips)

    def reveal_nodes(self):
        """Return the intro animation for all nodes."""
        return [
            LaggedStart(
                *[FadeIn(node, shift=RIGHT * 0.15) for node in self.nodes],
                lag_ratio=0.05,
            )
        ]

    def reveal_edges(self):
        """Return intro animations for arrows and residual paths."""
        animations = [
            LaggedStart(*[Create(arrow) for arrow in self.arrows], lag_ratio=0.03)
        ]
        for skip in self.skips:
            animations.extend(skip.reveal())
        return animations

    def residual_merge(self, index: int, theme: Theme):
        """Return ``(skip, dot, animation)`` for a residual entering ``index``.

        Returns ``None`` when no residual ends at that node. The caller is
        responsible for adding ``dot`` to the scene and fading it out.
        """
        for skip in self.skips:
            if skip.end == index:
                dot = Dot(radius=0.07, color=theme.color("residual"))
                dot.move_to(skip.path.points[0])
                return skip, dot, MoveAlongPath(dot, skip.path.copy())
        return None


def vertical_pipeline(
    specs: Sequence[NodeSpec],
    theme: Theme,
    *,
    x: float,
    top_y: float,
    width: float | None = None,
    height: float | None = None,
    spacing: float = 0.66,
    skips: Sequence[SkipSpec] = (),
) -> Pipeline:
    """Lay out ``specs`` as a top-to-bottom computation graph.

    Args:
        specs: Node descriptions in display order.
        theme: Theme supplying colors, sizes, and stroke widths.
        x: Horizontal center of the node column.
        top_y: Vertical center of the first node.
        width: Optional node width; defaults to ``theme.node_width``.
        height: Optional node height; defaults to ``theme.node_height``.
        spacing: Distance between consecutive node centers.
        skips: Residual connections to route on the left side.

    Returns:
        A ``Pipeline`` containing all mobjects and geometry.

    Each residual is offset further to the left than the previous one, so
    multiple skip connections do not overlap.
    """
    node_w = width if width is not None else theme.node_width
    node_h = height if height is not None else theme.node_height

    ys = [top_y - i * spacing for i in range(len(specs))]
    nodes = [
        Node(spec, theme, width=node_w, height=node_h).move_to([x, y, 0])
        for spec, y in zip(specs, ys)
    ]

    arrows = VGroup()
    for i in range(len(nodes) - 1):
        start = np.array([x, ys[i] - node_h / 2, 0.0])
        end = np.array([x, ys[i + 1] + node_h / 2, 0.0])
        arrows.add(
            Arrow(
                start,
                end,
                buff=0,
                stroke_width=theme.arrow_stroke,
                color=theme.arrow_color,
                tip_length=0.08,
                max_tip_length_to_length_ratio=0.5,
            )
        )

    skip_objects: list[SkipConnection] = []
    for idx, skip in enumerate(skips):
        route_x = x - node_w / 2 - skip.route_offset * (idx + 1)
        points = [
            np.array([x - node_w / 2, ys[skip.start] + skip.start_dy, 0.0]),
            np.array([route_x, ys[skip.start] + skip.start_dy, 0.0]),
            np.array([route_x, ys[skip.end] + skip.end_dy, 0.0]),
            np.array([x - node_w / 2, ys[skip.end] + skip.end_dy, 0.0]),
        ]
        skip_objects.append(
            SkipConnection(points, theme, start=skip.start, end=skip.end)
        )

    return Pipeline(
        nodes=nodes,
        arrows=arrows,
        skips=skip_objects,
        x=x,
        ys=ys,
        width=node_w,
        height=node_h,
        spacing=spacing,
    )
