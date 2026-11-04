"""Right-hand HUD: activation shape, log-scale bar, and legend."""

from __future__ import annotations

from collections.abc import Sequence

from manim import (
    BOLD,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    NORMAL,
    RIGHT,
    RoundedRectangle,
    Text,
    UP,
    VGroup,
)

from schematics.specs import NodeSpec
from schematics.theme import Theme


class TensorPanel:
    """HUD that updates as the signal moves between graph nodes.

    The panel is intentionally not a single ``Mobject``. Static elements and
    the current dynamic labels are added to the scene separately, so
    ``FadeOut``/``FadeIn`` does not leave stale children inside a ``VGroup``.

    Attributes:
        static: Panel background, captions, bar, and legend.
        dynamic: Current operation, shape, detail, and value labels.
    """

    def __init__(
        self,
        theme: Theme,
        specs: Sequence[NodeSpec],
        *,
        center: Sequence[float] = (4.62, 0.42, 0),
        width: float = 3.35,
        height: float = 5.95,
        title: str = "Activation shapes",
        subtitle: str = "",
        bar_caption: str = "elements per token",
        help_text: str | None = None,
        legend_items: Sequence[tuple[str, str]] | None = None,
    ):
        """Build the panel and initialize it from ``specs[0]``."""
        self.theme = theme
        self.width = width
        self.height = height
        self.cx, self.cy = float(center[0]), float(center[1])
        self.top = self.cy + height / 2

        self.track_w = width - 0.60
        self.track_left = self.cx - self.track_w / 2

        self.y_op = self.top - 1.485
        self.y_shape = self.top - 1.995
        self.y_detail = self.top - 2.445
        self.y_bar = self.top - 3.195
        self.y_value = self.top - 3.575

        first = specs[0]

        panel = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=theme.panel_radius,
            stroke_color=theme.color("border"),
            stroke_width=theme.panel_stroke,
            fill_color=theme.color("panel"),
            fill_opacity=0.92,
        ).move_to([self.cx, self.cy, 0])

        hud_title = Text(
            title,
            font_size=theme.size("hud_title"),
            font=theme.font,
            weight=BOLD,
        ).move_to([self.cx, self.top - 0.375, 0])

        hud_sub = Text(
            subtitle,
            font_size=theme.size("hud_sub"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([self.cx, self.top - 0.695, 0])

        divider = Line(
            [self.cx - width / 2 + 0.17, self.top - 0.925, 0],
            [self.cx + width / 2 - 0.17, self.top - 0.925, 0],
            stroke_width=1,
            color=theme.color("border"),
        )

        caption = Text(
            "current operation",
            font_size=theme.size("caption"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([self.cx, self.top - 1.235, 0])

        bar_caption_obj = Text(
            bar_caption,
            font_size=theme.size("caption"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([self.cx, self.top - 2.835, 0])

        track = RoundedRectangle(
            width=self.track_w,
            height=0.24,
            corner_radius=0.07,
            stroke_color=theme.color("border"),
            stroke_width=1,
            fill_color=theme.color("border"),
            fill_opacity=0.25,
        ).move_to([self.cx, self.y_bar, 0])

        initial_w = theme.bar_width(first.elems, self.track_w)
        self.bar = RoundedRectangle(
            width=initial_w,
            height=0.18,
            corner_radius=0.05,
            stroke_width=0,
            fill_color=theme.color(first.color),
            fill_opacity=0.9,
        ).move_to([self.track_left + initial_w / 2, self.y_bar, 0])

        scale_note = Text(
            "log scale",
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([self.cx + 1.0, self.top - 3.855, 0])

        if help_text is None:
            help_text = (
                "Dot = signal · circle size and bar grow\n"
                "logarithmically with activation size."
            )
        help_obj = Text(
            help_text,
            font_size=theme.size("help"),
            font=theme.font,
            color=theme.color("muted"),
            line_spacing=0.8,
        ).move_to([self.cx, self.top - 4.55, 0])
        if help_obj.width > width - 0.35:
            help_obj.scale_to_fit_width(width - 0.35)

        if legend_items is None:
            legend_items = (
                ("RMSNorm", "norm"),
                ("Attention", "attention"),
                ("FFN", "ffn"),
                ("Residual", "residual"),
            )
        legend = self._build_legend(legend_items)

        self.static = VGroup(
            panel,
            hud_title,
            hud_sub,
            divider,
            caption,
            bar_caption_obj,
            track,
            self.bar,
            scale_note,
            help_obj,
            legend,
        )

        self.op = self._text(
            first.label,
            theme.size("hud_op"),
            self.y_op,
            color=theme.color("text"),
        )
        self.shape = self._text(
            first.shape,
            theme.size("hud_shape"),
            self.y_shape,
            color=theme.color(first.color),
            weight=BOLD,
        )
        self.detail = self._text(
            first.detail,
            theme.size("hud_detail"),
            self.y_detail,
            color=theme.color("muted"),
        )
        self.value = self._text(
            f"{first.elems:,}",
            theme.size("hud_value"),
            self.y_value,
            weight=BOLD,
        )
        self.dynamic = [self.op, self.shape, self.detail, self.value]

    def _text(
        self,
        text: str,
        size: float,
        y: float,
        *,
        color=None,
        weight=NORMAL,
    ):
        """Create a ``Text`` fitted to the panel width and centered at ``y``."""
        obj = Text(
            text,
            font_size=size,
            font=self.theme.font,
            color=color,
            weight=weight,
        )
        if obj.width > self.width - 0.35:
            obj.scale_to_fit_width(self.width - 0.35)
        obj.move_to([self.cx, y, 0])
        return obj

    def _build_legend(self, items: Sequence[tuple[str, str]]) -> VGroup:
        """Build a two-column legend from ``(label, color_key)`` pairs."""
        group = VGroup()
        x_left = self.cx - 1.07
        y_top = self.top - 5.20
        for idx, (label, color_key) in enumerate(items):
            col = idx % 2
            row = idx // 2
            dot = Dot(radius=0.06, color=self.theme.color(color_key))
            text = Text(
                label,
                font_size=self.theme.size("small"),
                font=self.theme.font,
                color=self.theme.color("muted"),
            )
            text.next_to(dot, RIGHT, buff=0.10)
            entry = VGroup(dot, text)
            entry.move_to([x_left + col * 1.75, y_top - row * 0.34, 0])
            group.add(entry)
        return group

    def reveal(self):
        """Return fade-in animations for the static and dynamic parts."""
        return [FadeIn(self.static), *[FadeIn(obj) for obj in self.dynamic]]

    def update(self, spec: NodeSpec):
        """Return animations that update the HUD for ``spec``.

        The internal references are replaced immediately, so the caller can
        play the returned animations and then call ``update`` again.
        """
        color = self.theme.color(spec.color)

        new_op = self._text(
            spec.label,
            self.theme.size("hud_op"),
            self.y_op,
            color=self.theme.color("text"),
        )
        new_shape = self._text(
            spec.shape,
            self.theme.size("hud_shape"),
            self.y_shape,
            color=color,
            weight=BOLD,
        )
        new_detail = self._text(
            spec.detail,
            self.theme.size("hud_detail"),
            self.y_detail,
            color=self.theme.color("muted"),
        )
        new_value = self._text(
            f"{spec.elems:,}",
            self.theme.size("hud_value"),
            self.y_value,
            weight=BOLD,
        )

        new_w = self.theme.bar_width(spec.elems, self.track_w)
        animations = [
            FadeOut(self.op, shift=UP * 0.06),
            FadeOut(self.shape, shift=UP * 0.06),
            FadeOut(self.detail, shift=UP * 0.06),
            FadeOut(self.value),
            FadeIn(new_op, shift=UP * 0.06),
            FadeIn(new_shape, shift=UP * 0.06),
            FadeIn(new_detail, shift=UP * 0.06),
            FadeIn(new_value),
            self.bar.animate.stretch_to_fit_width(new_w)
            .move_to([self.track_left + new_w / 2, self.y_bar, 0])
            .set_fill(color, opacity=0.9),
        ]

        self.op = new_op
        self.shape = new_shape
        self.detail = new_detail
        self.value = new_value
        self.dynamic = [new_op, new_shape, new_detail, new_value]
        return animations
