"""Vision primitives: pixel-art images divided into regular patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from manim import FadeIn, Line, Square, VGroup

from schematics.theme import Theme


class PatchGrid(VGroup):
    """Square pixel-art image divided into a regular grid of patches.

    The image is drawn as a mosaic of colored squares. Patches are square
    groups of pixels, ordered row-major. Use the patches to show how a vision
    transformer turns an image into a sequence of tokens.

    Attributes:
        pixels: Source color characters, row by row.
        pixel_rows: Number of pixel rows.
        pixel_cols: Number of pixel columns.
        patch_rows: Number of patch rows.
        patch_cols: Number of patch columns.
        patch_size: Side length of one patch in scene units.
        mosaic: Group containing all pixel squares.
        border: Outline around the whole image.
        grid_lines: Internal patch boundaries.
        patches: Patch ``VGroup`` objects in row-major order.
        patch_centers: Center of each patch in scene coordinates.
    """

    def __init__(
        self,
        pixels: Sequence[Sequence[str]],
        theme: Theme,
        *,
        patch_rows: int,
        patch_cols: int,
        size: float = 1.4,
        color_map: Mapping[str, str] | None = None,
        pixel_gap: float = 0.006,
        border_width: float = 1.6,
        grid_width: float = 1.8,
    ):
        """Build a square patch grid centered at the origin.

        Args:
            pixels: Rectangular matrix of color characters.
            theme: Theme supplying colors and stroke widths.
            patch_rows: Number of patch rows; must divide ``pixel_rows``.
            patch_cols: Number of patch columns; must divide ``pixel_cols``.
            size: Side length of the square image in scene units.
            color_map: Maps a character to a theme color key or hex string.
                Characters missing from the map are treated as theme keys.
            pixel_gap: Gap between adjacent pixel squares.
            border_width: Stroke width of the outer image border.
            grid_width: Stroke width of the internal patch boundaries.

        Raises:
            ValueError: If the image is empty, non-rectangular, or cannot be
                divided into the requested number of patches.
        """
        if not pixels or not pixels[0]:
            raise ValueError("PatchGrid needs a non-empty pixel matrix")
        if patch_rows < 1 or patch_cols < 1:
            raise ValueError("PatchGrid needs at least one patch")
        if len(pixels) % patch_rows != 0:
            raise ValueError("patch_rows must divide the pixel rows")
        if len(pixels[0]) % patch_cols != 0:
            raise ValueError("patch_cols must divide the pixel columns")
        if any(len(row) != len(pixels[0]) for row in pixels):
            raise ValueError("PatchGrid requires a rectangular pixel matrix")

        self.pixels = tuple(tuple(row) for row in pixels)
        self.pixel_rows = len(pixels)
        self.pixel_cols = len(pixels[0])
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.patch_size = size / patch_cols

        pixel_step = size / max(self.pixel_rows, self.pixel_cols)
        square_side = pixel_step - pixel_gap
        color_source = color_map or {}

        def resolve(char: str) -> str:
            value = color_source.get(char, char)
            return value if value.startswith("#") else theme.color(value)

        self.pixel_squares: list[list[Square]] = []
        for row in range(self.pixel_rows):
            current_row: list[Square] = []
            for col in range(self.pixel_cols):
                square = Square(
                    side_length=square_side,
                    fill_color=resolve(self.pixels[row][col]),
                    fill_opacity=1.0,
                    stroke_width=0,
                ).move_to(
                    [
                        (col - (self.pixel_cols - 1) / 2) * pixel_step,
                        ((self.pixel_rows - 1) / 2 - row) * pixel_step,
                        0,
                    ]
                )
                current_row.append(square)
            self.pixel_squares.append(current_row)

        self.mosaic = VGroup(
            *[square for row in self.pixel_squares for square in row]
        )
        self.border = Square(
            side_length=size,
            stroke_color=theme.color("border"),
            stroke_width=border_width,
            fill_opacity=0,
        )

        rows_per_patch = self.pixel_rows // patch_rows
        cols_per_patch = self.pixel_cols // patch_cols
        self.patches: list[VGroup] = []
        for patch_row in range(patch_rows):
            for patch_col in range(patch_cols):
                patch = VGroup(
                    *[
                        self.pixel_squares[row][col]
                        for row in range(
                            patch_row * rows_per_patch,
                            (patch_row + 1) * rows_per_patch,
                        )
                        for col in range(
                            patch_col * cols_per_patch,
                            (patch_col + 1) * cols_per_patch,
                        )
                    ]
                )
                self.patches.append(patch)

        self.patch_centers = [patch.get_center() for patch in self.patches]

        self.grid_lines = VGroup()
        for col in range(1, patch_cols):
            x = -size / 2 + col * self.patch_size
            self.grid_lines.add(
                Line(
                    [x, -size / 2, 0],
                    [x, size / 2, 0],
                    stroke_color=theme.color("border"),
                    stroke_width=grid_width,
                )
            )
        for row in range(1, patch_rows):
            y = size / 2 - row * self.patch_size
            self.grid_lines.add(
                Line(
                    [-size / 2, y, 0],
                    [size / 2, y, 0],
                    stroke_color=theme.color("border"),
                    stroke_width=grid_width,
                )
            )

        super().__init__(self.mosaic, self.border, self.grid_lines)

    def patch_copy(self, index: int) -> VGroup:
        """Return a detached copy of one patch for extraction animation."""
        return self.patches[index].copy()

    def reveal(self):
        """Return fade-in animations for the mosaic and border."""
        return [FadeIn(self.mosaic, scale=0.98), FadeIn(self.border)]
