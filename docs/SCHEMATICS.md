# Schematics Library: Agent Quickstart

This repository builds technical diagrams with Manim. The reusable layer lives
in `schematics/`; scene files under `scenes/` should stay thin.

## Read order for a fresh agent

1. `AGENTS.md` - project rules.
2. `docs/SCHEMATICS.md` - this guide.
3. `schematics/__init__.py` - public API surface.
4. `scenes/transformer/transformer_block/transformer_block.py` - reference pipeline scene.
5. `scenes/transformer/basic_attention/basic_attention.py` - reference computation-graph scene.
6. `scenes/transformer/tokens/tokens.py` - reference multimodal/tokenization scene.
7. `scenes/transformer/basic_attention/specs.py` - reference node data.
8. `scenes/transformer/positional_encoding/positional_encoding.py` -
   reference rotation scene with `RotationDial` and `PairStrip`.

## Visual and notation rules

These rules apply to every diagram, not only to the attention scene.

### Color grammar

- Purple (`compute`, `complexity`): computation blocks, operations, and
  asymptotic cost labels.
- Green (`activation`, `dimension`): activations, inputs, outputs, tensor
  shapes, and dimension symbols.
- Yellow (`weights`): learned parameter matrices such as `W_Q`, `W_K`, `W_V`,
  `W_O`.
- The softmax output is an activation, not a learned parameter. Name it
  `attn_weights` (or `A` in compact diagrams) and color it green.
- Prefer the semantic keys above. The older per-tensor keys (`query`, `key`,
  `value`, `score`, `weight`) remain for existing scenes, but new diagrams
  should use `compute`, `activation`, `dimension`, `complexity`, and
  `weights`.

### Math notation

- Write matrix multiplication as `@` in prose and `\mathbin{@}` in LaTeX.
- Write element-wise multiplication with a dot, `\cdot`.
- Every index and subscript must be LaTeX. Never draw `d_k`, `d_model`,
  `W_Q`, or similar as plain text.
- Use `MathTex`, `SchematicScene.math_caption()`, `BadgeSpec(title_tex=True,
  shape_tex=True)`, and `NodeSpec(label_tex=True)` for math labels.
- Always state the axis of a `concat`, `reshape`, or `transpose`, for example
  `concat(heads, dim=-1)`. Never write a bare `concat`.
- Keep the notation legend on the left edge. This frees vertical space for
  the graph and keeps the reading direction consistent.

### Scene storyboard

Every scene file has a sibling `.txt` storyboard. It describes the purpose,
layout, animation sequence, and data without requiring the reader to parse
the Python file. Update the storyboard whenever the visual story changes.

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
| `schematics/base.py` | `SchematicScene`: background, fonts, `title()`, `caption()`, `math_caption()`. |
| `schematics/primitives.py` | `Node`, `Badge`, `SkipConnection`, `connect()`. |
| `schematics/layouts.py` | `vertical_pipeline()`, `Pipeline`. |
| `schematics/hud.py` | `TensorPanel`: shape badge, log bar, legend. |
| `schematics/signal.py` | `SignalDot`: moving marker and trail. |
| `schematics/tensors.py` | `MatrixGrid`, `CausalMaskGrid`, `PairMatrix`, `PairStrip`, `WeightBars` for tensor schematics. |
| `schematics/rotation.py` | `RotationDial`: 2-D vectors rotating inside a circle. |
| `schematics/vision.py` | `PatchGrid`: pixel-art image split into patches. |
| `schematics/text.py` | `TokenChip`, `CodeLine`, `SymbolStrip`, `VocabularyTable`: token chips, syntax-colored code, mergeable symbol strips, and JSON-style vocabulary grids. |
| `scenes/registry.py` | Maps CLI keys to scene classes and files. |

## Core API

### `NodeSpec`

Immutable description of one graph node:

```python
NodeSpec(
    label="RMSNorm",              # text inside the rectangle
    shape="(B, T, 4096)",         # shown in the HUD
    detail="pre-attention norm",  # secondary HUD line
    color="compute",              # key from Theme.colors
    elems=4096,                   # elements per token for the log bar
    label_tex=False,              # True for MathTex labels with indices
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

Compact two-line labels for tensor shapes. Use the math flags for any label
with subscripts or operators:

```python
from schematics import Badge, BadgeSpec

BADGES = [
    BadgeSpec("input", "(B, T, 4096)", "activation"),
    BadgeSpec(
        r"Q = x \mathbin{@} W_Q",
        r"(B, T, d_{\mathrm{model}})",
        "compute",
        title_tex=True,
        shape_tex=True,
        shape_color="activation",
        title_highlights=((r"W_Q", "weights"),),
    ),
]

badges = VGroup(
    *[Badge(spec, theme, width=2.4) for spec in BADGES]
).arrange(RIGHT, buff=0.14).move_to([0, 2.98, 0])
self.play(LaggedStart(*[FadeIn(b) for b in badges], lag_ratio=0.08))
```

`title_highlights` recolors selected LaTeX substrings, which is how learned
weights such as `W_Q` stay yellow inside a purple computation block.
`title_size` selects a named size from `Theme.sizes`; use it to make
activation cards quieter than computation blocks.

### `MatrixGrid`

Schematic matrix of rounded cells, for Q/K/V blocks or score matrices:

```python
from schematics import MatrixGrid

keys = MatrixGrid(
    5, 4,                          # rows, columns
    theme,
    color_key="activation",        # semantic color
    row_labels=("t1", "t2", "t3", "t4", "t5"),
    highlight_rows=(0,),           # stronger fill for selected rows
)
keys.move_to([-3.85, 1.20, 0])
self.play(*keys.reveal())          # row-by-row fade-in
self.play(Indicate(keys.row(0)))   # one row as a VGroup
```

### `CausalMaskGrid`

Schematic causal mask: cells on and below the diagonal are allowed, cells
above the diagonal are blocked. Rows are query positions `i`, columns are
key positions `j`:

```python
from schematics import CausalMaskGrid

mask = CausalMaskGrid(5, theme, cell=0.17, gap=0.03)
mask.move_to([5.40, -0.05, 0])
self.play(*mask.reveal())          # row by row, top to bottom
```

`mask.allowed` and `mask.blocked` group the cells by triangular region.
`mask.row_centers` / `mask.col_centers` give the local coordinates for
axis labels. Pair it with a piecewise formula and a two-entry legend.

### `WeightBars`

Labelled horizontal bars for logits or softmax weights:

```python
from schematics import WeightBars

logits = WeightBars(
    ("t1", "t2", "t3"),
    (2.0, 1.0, 0.5),
    theme,
    color_key="activation",
    signed=True,           # diverging bars around a central baseline
    value_fmt="{:.2f}",
)

weights = WeightBars(
    ("t1", "t2", "t3"),
    (0.55, 0.20, 0.12),
    theme,
    color_key="activation",
    value_fmt="{:.3f}",
)
self.play(*weights.reveal())
```

Use `signed=True` for raw logits and the default unsigned mode for
non-negative weights.

### `CodeLine`

Monospace code with semantic per-segment colors, built from Pango markup:

```python
from schematics import CodeLine

line = CodeLine(
    (
        ("ids", "muted"),
        (" = ", None),
        ("tokenizer.encode", "compute"),
        ("(", "muted"),
        ("text", "activation"),
        (")", "muted"),
    ),
    theme,
    size="small",
)
line.move_to([0.0, 3.10, 0])
self.play(Write(line))
```

Use `None` for ordinary source text. Function names are purple
(`compute`), string or variable arguments are green (`activation`), and
punctuation is muted.

### `VocabularyTable`

JSON-style grid of `"token_id": "string"` cells. The table is centered at
the origin, so move it into place after construction:

```python
from schematics import VocabularyTable

table = VocabularyTable(
    (
        ("13", "."),
        ("262", "the"),
        ("278", "ing"),
        ("318", "is"),
        ("464", "The"),
        ("3797", "cat"),
    ),
    theme,
    columns=2,
    cell_width=4.2,
).move_to([0.0, -1.86, 0])

self.play(LaggedStart(*table.reveal(), lag_ratio=0.05))
self.play(table.highlight("278", active=True))
self.play(Indicate(table.entry("278").value_label))
```

`entry(token_id)` returns one cell, and `highlight(token_id)` returns an
animation that toggles its green selected state. Token ids are muted and
strings are green, matching the tokenization diagrams. Use `columns=1` for a
single-column vocabulary window; a taller table can be scrolled while rows
near the panel edges are faded out.

### `SymbolStrip`

Horizontal row of symbol chips that can highlight and merge adjacent pairs,
the core operation of byte-pair encoding. It is left-anchored, so a scene
places it with `set_left(x, y)` and the left edge stays fixed as merges shrink
the row:

```python
from schematics import SymbolStrip

word = SymbolStrip(("\u2581", "l", "o", "w"), theme, marker="\u2581")
word.set_left(-3.0, 2.0)
self.play(FadeIn(word))

index = word.find_pair("l", "o")
self.play(*word.highlight_pair(index))            # flash the two chips
self.play(FadeOut(word.chips[index]), FadeOut(word.chips[index + 1]))
_, moves = word.merge_pair(index)                 # collapse into one chip
self.play(*moves)
```

`merge_pair(index)` removes the two chips, inserts the merged chip, relayouts,
and returns `(merged_chip, move_animations)` so the caller can slide the
surviving chips into place. The optional `marker` (for example the SentencePiece
space symbol ``\u2581``) is drawn muted instead of the strip color.

### `PairStrip`

Horizontal vector split into 2-D coordinate pairs, used by RoPE diagrams:

```python
from schematics import PairStrip

pairs = PairStrip(
    4,                              # d_k / 2 pairs
    theme,
    cell=0.30,
    pair_gap=0.55,
    color_key="activation",
).shift([-2.30, 2.15, 0])

self.play(FadeIn(pairs.cells))
self.play(LaggedStart(*[Create(box) for box in pairs.boxes]))
```

The strip has `n_pairs` boxes. `pairs.cells` contains all scalar cells,
`pairs.boxes` contains the pair outlines, and `pairs.pairs[i]` groups one
box with its two cells. Build it at the origin and use `shift()` rather than
`move_to()` if you want the cells to stay on a fixed row while optional pair
labels sit above them.

### `PairMatrix`

Two-dimensional block of scalar cells grouped into coordinate pairs, used
when rows are sequence positions and columns are `head_dim` coordinates:

```python
from manim import RoundedRectangle

from schematics import PairMatrix

head = PairMatrix(
    16,                 # N_tokens rows
    4,                  # d_k / 2 pairs per row
    theme,
    cell=0.15,
    pair_gap=0.17,
    color_key="activation",
    pair_labels=(r"i=0", r"i=1", r"i=2", r"i=3"),
).shift([-3.20, 1.55, 0])

self.play(*head.reveal())
frame = RoundedRectangle(
    width=head.block_width + 0.10,
    height=0.20,
).move_to([-3.20, head.row_group(0).get_center()[1], 0])
```

`head.cells[row][pair]` stores the two cells of a pair, `head.row_group(i)`
returns all cells of one row, and `head.row_centers` / `head.pair_centers`
give the local coordinates of rows and pair columns. The block itself is
centered at the origin; shift it into place. Use `head.block_width` and
`head.block_height` for selection frames and brackets.

### `RotationDial`

Circle with a rotating vector, optional trail, and an optional dashed
secondary vector:

```python
from schematics import RotationDial

dial = RotationDial(
    theme,
    radius=0.55,
    color_key="activation",
).move_to([-4.10, 0.35, 0])

self.play(FadeIn(dial))
trail = dial.make_trail()
self.add(trail)
self.bring_to_back(trail)
self.play(dial.rotate_to(1.5))
```

`rotate_to(angle)` returns a `Rotate` animation and updates the stored angle.
`set_angle(angle)` rotates immediately, which is useful inside an updater
driven by a `ValueTracker`. Add the dial itself to the scene (not only its
parts) before attaching updaters, otherwise the updater will not run.
`make_trail()` returns a `TracedPath` in scene coordinates; add it
separately. The trace is drawn slightly inside the circle so it stays
visible against the dial border. `static_trail()` returns a deterministic
arc or full circle for the current angle, which keeps the final frame
correct when Manim renders with `--still`.

### `connect`

Styled arrow between two mobjects, with automatic edge selection:

```python
from schematics import connect

arrow = connect(keys, scores, theme, buff=0.18)
self.play(Create(arrow))
```

### `route_arrow`

Orthogonal polyline arrow for an explicit route, useful when a straight
connector would cross another block:

```python
from schematics import route_arrow

arrow = route_arrow(
    [[1.0, 0.0, 0], [1.0, -1.0, 0], [0.0, -1.0, 0]],
    theme,
    color_key="activation",
)
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
title = self.title("My diagram", x=0.0, size="title_small")
stage = self.caption("softmax", x=-2.65, y=-1.30, color_key="compute")
cost = self.math_caption(
    r"O(B \cdot S^2 \cdot d_{\mathrm{model}})",
    x=0.55,
    y=-0.30,
    color_key="complexity",
)
```

`caption()` creates a small muted label; override `color_key` or `size` when
the label belongs to a specific stage. `math_caption()` is the LaTeX variant
for shapes, dimensions, and cost formulas.

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
    NodeSpec("Input", "(B, T, 4096)", "hidden states", "activation", 4096),
    NodeSpec("Layer", "(B, T, 4096)", "computation", "compute", 4096),
    NodeSpec("Output", "(B, T, 4096)", "result", "activation", 4096),
]


class MyDiagram(SchematicScene):
    def construct(self):
        title = self.title("My diagram", size="title_small")
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
- Use semantic color keys (`compute`, `activation`, `dimension`,
  `complexity`, `weights`), never raw Manim colors inside a scene.
- Keep the same color meaning across scenes: purple computes, green
  activations and shapes, yellow learned weights.
- Use `@` (`\mathbin{@}`) for matrix multiplication and `\cdot` for
  element-wise multiplication. Never draw indices as plain text.
- Keep the notation legend on the left edge.
- Every scene has a sibling `.txt` storyboard; update it with the scene.
- `TensorPanel.update()` and `SignalDot.advance_to()` return animation lists;
  pass them to `self.play(*...)`.
- `Pipeline.residual_merge()` creates a dot at the path start. Add it to the
  scene before playing `MoveAlongPath`, then fade it out.
- `Text.set_default(font=...)` is process-global. Set it through the theme.
- `MathTex` uses LaTeX, not Pango. Use `Text` for plain prose and `MathTex`
  for any label with indices, operators, or dimensions.
- All comments and docstrings must be in English.

## Verification checklist

1. `python scripts/render.py <key> --still` - catches layout errors quickly.
2. `pytest -q` - renders every registered scene's last frame.
3. Full render for animation timing: `python scripts/render.py <key> -q l`.
