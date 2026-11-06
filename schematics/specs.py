"""Data models shared by scenes and layout helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    """Description of one node in a computation graph.

    Attributes:
        label: Short node title shown inside the rectangle.
        shape: Activation shape shown in the HUD, for example ``(B, T, 4096)``.
        detail: Secondary HUD line with implementation details.
        color: Semantic color key from ``Theme.colors``.
        elems: Elements per token, used for the log-scale HUD bar.
        label_tex: Render ``label`` as LaTeX instead of plain text.
        label_size: Named font size from ``Theme.sizes`` for the label.
    """

    label: str
    shape: str
    detail: str = ""
    color: str = "norm"
    elems: int = 1
    label_tex: bool = False
    label_size: str = "node"


@dataclass(frozen=True)
class BadgeSpec:
    """Compact two-line label for a tensor shape.

    Attributes:
        title: Short name shown on the first line.
        shape: Tensor shape shown on the second line.
        color: Semantic color key from ``Theme.colors``.
        title_tex: Render ``title`` as LaTeX instead of plain text.
        shape_tex: Render ``shape`` as LaTeX instead of monospace text.
        shape_color: Optional color key for the shape line. Defaults to
            ``color``, which keeps simple badges monochrome.
        title_highlights: Optional ``(substring, color_key)`` pairs that
            recolor parts of a LaTeX title, for example ``W_Q`` in yellow.
        title_size: Named font size from ``Theme.sizes`` for the title line.
    """

    title: str
    shape: str
    color: str = "input"
    title_tex: bool = False
    shape_tex: bool = False
    shape_color: str | None = None
    title_highlights: tuple[tuple[str, str], ...] = ()
    title_size: str = "badge_title"


@dataclass(frozen=True)
class SkipSpec:
    """Residual connection between two node indices.

    Attributes:
        start: Index of the source node.
        end: Index of the destination node.
        route_offset: Base horizontal offset of the route from the node edge.
        start_dy: Vertical offset of the source point.
        end_dy: Vertical offset of the destination point.
    """

    start: int
    end: int
    route_offset: float = 0.55
    start_dy: float = 0.0
    end_dy: float = 0.0


@dataclass(frozen=True)
class SceneSpec:
    """Metadata used by the scene registry and batch renderer.

    Attributes:
        key: Stable CLI key, for example ``llama_block``.
        path: Repository-relative path to the scene file.
        cls: Scene class defined in that file.
        title: Human-readable title for galleries.
        tags: Optional tags for filtering.
    """

    key: str
    path: str
    cls: type
    title: str = ""
    tags: tuple[str, ...] = ()
