"""Moving signal dot: core, halo, and glowing trail."""

from __future__ import annotations

from collections.abc import Sequence

from manim import Circle, Create, Dot, FadeIn, Line, VGroup

from schematics.theme import Theme


class SignalDot:
    """Signal marker that travels through a ``Pipeline``.

    Attributes:
        group: The halo and core dot, moved together.
        trail: Line segments left behind by the signal.
    """

    def __init__(self, theme: Theme, start: Sequence[float]):
        """Create a signal at ``start`` using the theme signal color."""
        self.theme = theme
        color = theme.color("signal")

        self.halo = Circle(
            radius=0.16,
            color=color,
            fill_color=color,
            fill_opacity=0.15,
            stroke_width=0,
        )
        self.core = Dot(radius=0.075, color=color)
        self.group = VGroup(self.halo, self.core).move_to(start)
        self.trail = VGroup()

    def add_to(self, scene):
        """Add the signal and its trail to ``scene``."""
        scene.add(self.group, self.trail)

    def reveal(self):
        """Return the intro animation for the signal."""
        return [FadeIn(self.group, scale=0.3)]

    def advance_to(self, point: Sequence[float]):
        """Return animations that move the signal to ``point``.

        A new trail segment is appended immediately and returned as a
        ``Create`` animation.
        """
        prev = self.group.get_center()
        segment = Line(
            prev,
            point,
            color=self.theme.color("signal"),
            stroke_width=3.5,
            stroke_opacity=0.85,
        )
        self.trail.add(segment)
        return [self.group.animate.move_to(point), Create(segment)]

    def resize(self, elems: int):
        """Return an animation that resizes the signal for ``elems``."""
        return self.group.animate.scale_to_fit_height(
            self.theme.signal_height(elems)
        )
