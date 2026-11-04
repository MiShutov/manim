# Schematics Library: Agent Quickstart

This repository builds technical diagrams with Manim. The reusable layer lives
in `schematics/`; scene files under `scenes/` should stay thin.

## Read order for a fresh agent

1. `AGENTS.md` - project rules.
2. `docs/SCHEMATICS.md` - this guide.
3. `schematics/__init__.py` - public API surface.
4. `scenes/llm/llama_block.py` - reference pipeline scene.
5. `scenes/llm/attention.py` - reference tensor-detail scene.
6. `scenes/llm/token_anatomy.py` - reference multimodal/tokenization scene.
7. `scenes/llm/specs.py` - reference node data.

## Environment and commands

Manim is installed in the conda environment `manim`:

```bash
~/Utils/miniconda3/envs/manim/bin/python -m manim --version
```

The project is installed editable, so `import schematics` works from anywhere
inside this environment. If imports fail, reinstall:

```bash
~/Utils/miniconda3/envs/manim/bin/python -m pip install -e . --no-deps
```

Common commands:

```bash
python scripts/render.py --list                 # registered scenes
python scripts/render.py llama_block -q l       # one scene
python scripts/render.py all -q h               # all scenes, 1080p
python scripts/render.py llama_block --still    # last frame only
pytest -q                                       # smoke tests
```

Direct Manim invocation also works:

```bash
~/Utils/miniconda3/envs/manim/bin/manim -pql \
  scenes/llm/llama_block.py LlamaBlock
```

## Module map

| Module | Responsibility |
| --- | --- |
| `schematics/theme.py` | Colors, fonts, sizes, spacing, log-scale helpers. |
| `schematics/specs.py` | `NodeSpec`, `BadgeSpec`, `SkipSpec`, `SceneSpec` dataclasses. |
| `schematics/base.py` | `SchematicScene`: background, fonts, `title()`, `caption()`. |
| `schematics/primitives.py` | `Node`, `Badge`, `SkipConnection`, `connect()`. |
| `schematics/layouts.py` | `vertical_pipeline()`, `Pipeline`. |
| `schematics/hud.py` | `TensorPanel`: shape badge, log bar, legend. |
| `schematics/signal.py` | `SignalDot`: moving marker and trail. |
| `schematics/tensors.py` | `MatrixGrid`, `WeightBars` for tensor schematics. |
| `schematics/vision.py` | `PatchGrid`: pixel-art image split into patches. |
| `schematics/text.py` | `TokenChip`: token and token-id chips. |
| `scenes/registry.py` | Maps CLI keys to scene classes and files. |

## Core API

### `NodeSpec`

Immutable description of one graph node:

```python
NodeSpec(
    label="RMSNorm",              # text inside the rectangle
    shape="(B, T, 4096)",         # shown in the HUD
    detail="pre-attention norm",  # secondary HUD line
    color="norm",                 # key from Theme.colors
    elems=4096,                   # elements per token for the log bar
)
```

### `SkipSpec`

Residual connection between node indices:

```python
SkipSpec(
    start=0,          # source node index
    end=5,            # destination node index
    route_offset=0.55,
    start_dy=0.0,     # optional vertical offset at the source
    end_dy=0.0,       # optional vertical offset at the destination
)
```

### `vertical_pipeline`

```python
pipeline = vertical_pipeline(
    specs,                    # Sequence[NodeSpec]
    theme,                    # Theme
    x=-0.75,                  # horizontal center
    top_y=3.12,               # center of the first node
    spacing=0.66,             # distance between node centers
    skips=[SkipSpec(...)],    # optional residual connections
)
```

Returns a `Pipeline` with:

- `nodes` - `Node` mobjects top to bottom;
- `arrows` - connectors between nodes;
- `skips` - `SkipConnection` mobjects;
- `ys` - vertical center of each node;
- `reveal_nodes()` / `reveal_edges()` - intro animation lists;
- `residual_merge(index, theme)` - `(skip, dot, animation)` or `None`.

### `TensorPanel`

Right-hand HUD that shows the current activation shape.

```python
hud = TensorPanel(
    theme,
    specs,
    center=[4.62, 0.42, 0],
    subtitle="Llama 3 8B  ·  T = 2048",
)
self.play(*hud.reveal())
self.play(*hud.update(spec))
```

`TensorPanel` is intentionally not a single `Mobject`: static parts and the
current dynamic labels are added to the scene separately. Do not wrap it in a
`VGroup`; use `hud.reveal()` and `hud.update()`.

### `SignalDot`

```python
signal = SignalDot(theme, pipeline.nodes[0].get_center())
signal.add_to(self)
self.play(*signal.reveal())

self.play(*signal.advance_to(pipeline.nodes[i].get_center()))
self.play(signal.resize(spec.elems))
```

`advance_to()` appends a trail segment immediately and returns animations.
`resize()` uses the same log-scale as the HUD bar.

### `Badge` and `BadgeSpec`

Compact two-line labels for tensor shapes, useful as a top strip:

```python
from schematics import Badge, BadgeSpec

BADGES = [
    BadgeSpec("input", "(B, T, 4096)", "input"),
    BadgeSpec("scores", "(B, 32, T, T)", "score"),
]

badges = VGroup(
    *[Badge(spec, theme, width=2.0) for spec in BADGES]
).arrange(RIGHT, buff=0.14).move_to([0, 2.98, 0])
self.play(LaggedStart(*[FadeIn(b) for b in badges], lag_ratio=0.08))
```

### `MatrixGrid`

Schematic matrix of rounded cells, for Q/K/V blocks or score matrices:

```python
from schematics import MatrixGrid

keys = MatrixGrid(
    5, 4,                          # rows, columns
    theme,
    color_key="key",               # semantic color
    row_labels=("t1", "t2", "t3", "t4", "t5"),
    highlight_rows=(0,),           # stronger fill for selected rows
)
keys.move_to([-3.85, 1.20, 0])
self.play(*keys.reveal())          # row-by-row fade-in
self.play(Indicate(keys.row(0)))   # one row as a VGroup
```

### `WeightBars`

Labelled horizontal bars for logits or softmax weights:

```python
from schematics import WeightBars

logits = WeightBars(
    ("t1", "t2", "t3"),
    (2.0, 1.0, 0.5),
    theme,
    color_key="score",
    signed=True,           # diverging bars around a central baseline
    value_fmt="{:.2f}",
)

weights = WeightBars(
    ("t1", "t2", "t3"),
    (0.55, 0.20, 0.12),
    theme,
    color_key="weight",
    value_fmt="{:.3f}",
)
self.play(*weights.reveal())
```

Use `signed=True` for raw logits and the default unsigned mode for
non-negative weights.

### `connect`

Styled arrow between two mobjects, with automatic edge selection:

```python
from schematics import connect

arrow = connect(keys, scores, theme, buff=0.18)
self.play(Create(arrow))
```

### `SchematicScene` and `Theme`

```python
class MyScene(SchematicScene):
    theme = Theme(font="Inter", bg="#0D1117")
```

`SchematicScene.setup()` applies the background, sets the default `Text` font,
and registers any `.ttf`/`.otf` files from `assets/fonts/`.

Helper methods:

```python
title = self.title("My diagram", x=0.0)
stage = self.caption("softmax", x=-2.65, y=-1.30, color_key="weight")
```

`caption()` creates a small muted label; override `color_key` or `size` when
the label belongs to a specific stage.

## Minimal new scene

```python
from manim import FadeIn, Indicate

from schematics import (
    NodeSpec,
    SchematicScene,
    SignalDot,
    TensorPanel,
    vertical_pipeline,
)

NODES = [
    NodeSpec("Input", "(B, T, 4096)", "hidden states", "input", 4096),
    NodeSpec("Layer", "(B, T, 4096)", "computation", "norm", 4096),
    NodeSpec("Output", "(B, T, 4096)", "result", "residual", 4096),
]


class MyDiagram(SchematicScene):
    def construct(self):
        title = self.title("My diagram")
        pipeline = vertical_pipeline(NODES, self.theme, x=-0.75, top_y=3.12)
        hud = TensorPanel(self.theme, NODES, subtitle="T = 2048")
        signal = SignalDot(self.theme, pipeline.nodes[0].get_center())

        self.play(FadeIn(title))
        self.play(*pipeline.reveal_nodes())
        self.play(*pipeline.reveal_edges())
        self.play(*hud.reveal())

        signal.add_to(self)
        self.play(*signal.reveal())

        for i, spec in enumerate(NODES):
            if i > 0:
                self.play(*signal.advance_to(pipeline.nodes[i].get_center()))
            self.play(Indicate(pipeline.nodes[i]), signal.resize(spec.elems))
            self.play(*hud.update(spec))

        self.wait(2)
```

Then register it:

```python
# scenes/registry.py
"my_diagram": SceneSpec(
    key="my_diagram",
    path="scenes/<topic>/my_diagram.py",
    cls=MyDiagram,
    title="My diagram",
    tags=("demo",),
),
```

## Conventions and pitfalls

- Keep scene files thin. Put geometry in `theme.py` and layout helpers.
- Use semantic color keys (`norm`, `attention`, `ffn`, `residual`, `input`),
  never raw Manim colors inside a scene.
- `TensorPanel.update()` and `SignalDot.advance_to()` return animation lists;
  pass them to `self.play(*...)`.
- `Pipeline.residual_merge()` creates a dot at the path start. Add it to the
  scene before playing `MoveAlongPath`, then fade it out.
- `Text.set_default(font=...)` is process-global. Set it through the theme.
- `MathTex` uses LaTeX, not Pango. For diagrams prefer `Text`; use `MathTex`
  only for real formulas.
- All comments and docstrings must be in English.

## Verification checklist

1. `python scripts/render.py <key> --still` - catches layout errors quickly.
2. `pytest -q` - renders every registered scene's last frame.
3. Full render for animation timing: `python scripts/render.py <key> -q l`.
