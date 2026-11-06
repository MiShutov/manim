"""Circular rotation dials for RoPE and other rotation-based diagrams."""

from __future__ import annotations

import numpy as np
from manim import (
    PI,
    Arc,
    Arrow,
    Circle,
    DashedLine,
    FadeIn,
    Line,
    Rotate,
    TracedPath,
    Triangle,
    VGroup,
    smooth,
)

from schematics.theme import Theme


class RotationDial(VGroup):
    """Circle with one or two rotating vectors and an optional trail.

    The dial contains a unit-circle style background, a solid primary vector,
    and an optional faded secondary vector. Both vectors rotate by the same
    angle, so the secondary vector keeps a fixed angular offset from the
    primary one. This is useful for comparing two directions on the same
    plane.

    Attributes:
        radius: Radius of the dial.
        primary_arrow: Solid vector from the center to the circle edge.
        secondary_arrow: Optional faded dashed vector.
        vectors: Group containing both arrows, rotated as one unit.
        trail: Optional ``TracedPath`` following the primary arrow tip.
        angle: Current rotation offset in radians.
    """

    def __init__(
        self,
        theme: Theme,
        *,
        radius: float = 0.62,
        color_key: str = "activation",
        secondary_color_key: str | None = None,
        secondary_radius_scale: float = 0.78,
        initial_angle: float = 0.0,
        secondary_initial_angle: float = 0.0,
        tip_length: float = 0.08,
        trail: bool = True,
    ):
        """Create a dial centered at the origin.

        Args:
            theme: Theme supplying colors and stroke widths.
            radius: Dial radius.
            color_key: Semantic color for the primary vector and trail.
            secondary_color_key: Optional color for the secondary vector.
            secondary_radius_scale: Length of the secondary vector relative
                to the primary vector.
            initial_angle: Starting angle of the primary vector in radians.
            secondary_initial_angle: Starting angle of the secondary vector.
            tip_length: Arrow head length for the primary vector.
            trail: Whether ``make_trail()`` should create a traced path.
        """
        self.radius = radius
        self.angle = initial_angle
        self.trail = None

        color = theme.color(color_key)
        border = theme.color("border")
        axis = theme.color("muted")

        self.circle = Circle(
            radius=radius,
            stroke_color=border,
            stroke_width=1.2,
            stroke_opacity=0.9,
        )
        self.axes = VGroup(
            Line(
                [-radius, 0, 0],
                [radius, 0, 0],
                stroke_color=axis,
                stroke_width=0.6,
                stroke_opacity=0.35,
            ),
            Line(
                [0, -radius, 0],
                [0, radius, 0],
                stroke_color=axis,
                stroke_width=0.6,
                stroke_opacity=0.35,
            ),
        )

        self.primary_arrow = Arrow(
            np.zeros(3),
            radius * self._unit(initial_angle),
            buff=0.0,
            stroke_width=theme.arrow_stroke,
            color=color,
            tip_length=tip_length,
            max_tip_length_to_length_ratio=0.30,
        )

        self.secondary_arrow = None
        if secondary_color_key is not None:
            secondary_color = theme.color(secondary_color_key)
            secondary_end = (
                radius * secondary_radius_scale
                * self._unit(secondary_initial_angle)
            )
            self.secondary_arrow = self._dashed_arrow(
                np.zeros(3),
                secondary_end,
                secondary_color,
                tip_length * 0.82,
            )

        self.vectors = VGroup(self.primary_arrow)
        if self.secondary_arrow is not None:
            self.vectors.add(self.secondary_arrow)

        super().__init__(self.circle, self.axes, self.vectors)

        if trail:
            self.trail = TracedPath(
                self._traced_point,
                stroke_color=color,
                stroke_width=1.2,
                stroke_opacity=0.55,
            )

    def reveal(self):
        """Return fade-in animations for the dial parts."""
        return [FadeIn(self.circle), FadeIn(self.axes), FadeIn(self.vectors)]

    def rotate_to(self, angle: float, *, rate_func=smooth):
        """Return an animation rotating the vectors to ``angle`` radians."""
        delta = angle - self.angle
        self.angle = angle
        return Rotate(
            self.vectors,
            angle=delta,
            about_point=self.get_center(),
            rate_func=rate_func,
        )

    def set_angle(self, angle: float):
        """Rotate the vectors immediately to ``angle`` radians."""
        delta = angle - self.angle
        self.angle = angle
        self.vectors.rotate(delta, about_point=self.get_center())
        return self

    def make_trail(self) -> TracedPath:
        """Return the trail mobject.

        Call this after positioning the dial. The trail lives in scene
        coordinates, so add it to the scene separately and behind the dial.
        """
        if self.trail is None:
            self.trail = TracedPath(
                self._traced_point,
                stroke_color=self.primary_arrow.get_color(),
                stroke_width=1.2,
                stroke_opacity=0.55,
            )
        return self.trail

    def _traced_point(self) -> np.ndarray:
        """Return a point slightly inside the circle for the visible trail.

        The arrow tip lies exactly on the circle stroke, so tracing it would
        only recolor the border. Pulling the trail inward by ten percent
        keeps the trail clearly visible as a separate ring.
        """
        center = self.get_center()
        end = self.primary_arrow.get_end()
        return center + (end - center) * 0.90

    def static_trail(self, angle: float | None = None, *, radius_scale: float = 0.90):
        """Return a static arc or full circle matching the current rotation.

        A dynamic ``TracedPath`` loses its accumulated points when Manim
        renders only the last frame (``-s``). This helper builds the final
        trail deterministically: a full circle after one or more turns, or
        an arc for a smaller angle.
        """
        angle = self.angle if angle is None else angle
        radius = self.radius * radius_scale
        color = self.primary_arrow.get_color()
        if abs(angle) >= 2 * PI:
            shape = Circle(
                radius=radius,
                stroke_color=color,
                stroke_width=1.2,
                stroke_opacity=0.55,
            )
        else:
            shape = Arc(
                radius=radius,
                start_angle=0.0,
                angle=angle,
                stroke_color=color,
                stroke_width=1.2,
                stroke_opacity=0.55,
            )
        return shape.shift(self.get_center())

    @staticmethod
    def _unit(angle: float) -> np.ndarray:
        """Return a 3-D unit vector for ``angle`` in the xy-plane."""
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    @staticmethod
    def _dashed_arrow(
        start: np.ndarray,
        end: np.ndarray,
        color: str,
        tip_length: float,
    ) -> VGroup:
        """Return a dashed line with a triangular arrow head."""
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            raise ValueError("dashed arrow needs a non-zero length")
        unit = direction / length

        line = DashedLine(
            start,
            end - unit * tip_length * 0.75,
            dash_length=0.06,
            dashed_ratio=0.55,
            stroke_width=1.6,
            stroke_color=color,
            stroke_opacity=0.55,
        )
        tip = Triangle(
            fill_color=color,
            fill_opacity=0.55,
            stroke_width=0,
        ).scale_to_fit_height(tip_length)
        tip.rotate(np.arctan2(unit[1], unit[0]) - PI / 2)
        tip.move_to(end - unit * tip_length * 0.35)
        return VGroup(line, tip)
