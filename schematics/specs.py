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
    """

    label: str
    shape: str
    detail: str = ""
    color: str = "norm"
    elems: int = 1


@dataclass(frozen=True)
class BadgeSpec:
    """Compact two-line label for a tensor shape.

    Attributes:
        title: Short name shown on the first line.
        shape: Tensor shape shown on the second line.
        color: Semantic color key from ``Theme.colors``.
    """

    title: str
    shape: str
    color: str = "input"


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
