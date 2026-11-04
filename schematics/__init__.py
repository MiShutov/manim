"""Reusable Manim building blocks for technical diagrams."""

from schematics.base import SchematicScene
from schematics.hud import TensorPanel
from schematics.layouts import Pipeline, vertical_pipeline
from schematics.primitives import Badge, Node, SkipConnection, connect
from schematics.signal import SignalDot
from schematics.specs import BadgeSpec, NodeSpec, SceneSpec, SkipSpec
from schematics.tensors import MatrixGrid, WeightBars
from schematics.text import TokenChip
from schematics.theme import DARK, Theme
from schematics.vision import PatchGrid

__all__ = [
    "DARK",
    "Badge",
    "BadgeSpec",
    "MatrixGrid",
    "Node",
    "NodeSpec",
    "PatchGrid",
    "Pipeline",
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
    "vertical_pipeline",
]
