"""Reusable Manim building blocks for technical diagrams."""

from schematics.base import SchematicScene
from schematics.hud import TensorPanel
from schematics.layouts import Pipeline, vertical_pipeline
from schematics.primitives import (
    Badge,
    Node,
    SkipConnection,
    connect,
    route_arrow,
)
from schematics.rotation import RotationDial
from schematics.signal import SignalDot
from schematics.specs import BadgeSpec, NodeSpec, SceneSpec, SkipSpec
from schematics.tensors import (
    CausalMaskGrid,
    MatrixGrid,
    PairMatrix,
    PairStrip,
    WeightBars,
)
from schematics.text import TokenChip
from schematics.theme import DARK, Theme
from schematics.vision import PatchGrid

__all__ = [
    "DARK",
    "Badge",
    "BadgeSpec",
    "CausalMaskGrid",
    "MatrixGrid",
    "Node",
    "NodeSpec",
    "PatchGrid",
    "PairMatrix",
    "PairStrip",
    "Pipeline",
    "RotationDial",
    "SceneSpec",
    "SchematicScene",
    "SignalDot",
    "SkipConnection",
    "SkipSpec",
    "TensorPanel",
    "Theme",
    "TokenChip",
    "WeightBars",
    "connect",
    "route_arrow",
    "vertical_pipeline",
]
