"""Base scene: background, fonts, and bundled-font registration."""

from __future__ import annotations

from contextlib import ExitStack

from manim import BOLD, Scene, Text, register_font

from schematics.theme import DARK, FONT_DIR, Theme


class SchematicScene(Scene):
    """Scene preconfigured with the shared theme and font defaults.

    Subclasses usually implement only ``construct``. Override the ``theme``
    class attribute to restyle the whole scene.

    Attributes:
        theme: Theme used for background, fonts, colors, and sizes.
    """

    theme: Theme = DARK

    def setup(self):
        """Apply the theme before ``construct`` runs."""
        self.camera.background_color = self.theme.bg
        Text.set_default(font=self.theme.font)

        # Register bundled fonts from assets/fonts for the scene lifetime.
        self._font_stack = ExitStack()
        if FONT_DIR.exists():
            for path in sorted(FONT_DIR.iterdir()):
                if path.suffix.lower() in {".ttf", ".otf"}:
                    self._font_stack.enter_context(register_font(path))

    def tear_down(self):
        """Release fonts registered for this scene."""
        self._font_stack.close()
        super().tear_down()

    def title(self, text: str, *, x: float = -0.75, y: float = 3.68):
        """Create the standard diagram title centered at ``(x, y)``."""
        return Text(
            text,
            font_size=self.theme.size("title"),
            font=self.theme.font,
            weight=BOLD,
        ).move_to([x, y, 0])

    def caption(
        self,
        text: str,
        *,
        x: float,
        y: float,
        color_key: str = "muted",
        size: str = "small",
    ):
        """Create a short caption centered at ``(x, y)``.

        Use captions for stage names, tensor hints, and footnotes. They are
        intentionally smaller and muted by default.
        """
        return Text(
            text,
            font_size=self.theme.size(size),
            font=self.theme.font,
            color=self.theme.color(color_key),
        ).move_to([x, y, 0])
