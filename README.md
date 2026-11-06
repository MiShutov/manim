# Manim schematics

Technical diagrams for ML/LLM topics: classical multi-head attention, RoPE
positional encoding, a Llama transformer block, the anatomy of a whole LLM
(embeddings, blocks, lm_head), linear algebra, and future scenes.

> For agents and models: see `docs/SCHEMATICS.md` for the full library guide.
> It is loaded automatically through `opencode.jsonc`.

## Structure

```text
AGENTS.md           project rules (auto-loaded)
docs/SCHEMATICS.md  agent guide for the schematics library
opencode.jsonc      loads docs/SCHEMATICS.md as instructions
schematics/         reusable library
  theme.py          colors, fonts, sizes, log scale
  base.py           SchematicScene: background, fonts, title(), math_caption()
  specs.py          NodeSpec / BadgeSpec / SkipSpec / SceneSpec
  primitives.py     Node, Badge, SkipConnection, connect()
  layouts.py        vertical_pipeline(), Pipeline
  hud.py            TensorPanel
  signal.py         SignalDot
  tensors.py        MatrixGrid, PairMatrix, PairStrip, WeightBars
  rotation.py       RotationDial for 2-D rotations
scenes/             thin scenes: data plus composition only
  transformer/basic_attention/basic_attention.py
  transformer/positional_encoding/positional_encoding.py
  transformer/transformer_block/transformer_block.py
  transformer/model_anatomy/model_anatomy.py
  transformer/tokens/tokens.py
  math/matrix_multiplication.py
  registry.py       registry for batch rendering and tests
scripts/render.py   single rendering command
tests/              scene smoke tests
assets/fonts/       optional bundled fonts
```

## Installation

Manim is available in the conda environment `manim`. Install the project in
editable mode so `import schematics` works when running
`manim scenes/...` directly:

```bash
~/Utils/miniconda3/envs/manim/bin/python -m pip install -e . --no-deps
```

Alternatively, use `scripts/render.py`, which sets `PYTHONPATH` itself.

## Rendering

```bash
# list registered scenes
python scripts/render.py --list

# one scene, low quality
python scripts/render.py basic_attention -q l
python scripts/render.py positional_encoding -q l
python scripts/render.py llama_block -q l

# all scenes, 1080p
python scripts/render.py all -q h

# last frame only (fast layout check)
python scripts/render.py llama_block --still
```

Direct invocation also works after the editable install:

```bash
~/Utils/miniconda3/envs/manim/bin/manim -pql \
  scenes/transformer/transformer_block/transformer_block.py LlamaBlock
```

## Adding a new scene

1. Create `scenes/<topic>/<name>.py` and, for shared data, a matching
   `<name>_specs.py` (or `specs.py` in the scene directory).
2. Add a sibling `<name>.txt` storyboard with the purpose, layout, animation
   sequence, and data.
3. Describe graph nodes with `NodeSpec` and tensor-shape blocks with
   `BadgeSpec`.
4. Build pipelines with `vertical_pipeline(...)`; build matrix and bar
   schematics with `MatrixGrid` and `WeightBars`.
5. Add `TensorPanel` if you need an activation-shape HUD.
6. Register the scene in `scenes/registry.py`.
7. Run `python scripts/render.py <key> --still` and `pytest -q`.

## Style

- Use semantic color keys: purple `compute`/`complexity`, green
  `activation`/`dimension`, yellow `weights`, and orange `residual`.
- Never hard-code Manim colors in a scene.
- Use `@` (`\mathbin{@}`) for matrix multiplication and `\cdot` for
  element-wise multiplication.
- Draw every index and subscript in LaTeX; never use plain `d_k`, `d_model`,
  or `W_Q` text.
- Always state the axis of a `concat`, `reshape`, or `transpose`, for example
  `concat(heads, dim=-1)`. Never write a bare `concat`.
- Keep the notation legend on the left edge.
- Change fonts and sizes in `schematics/theme.py`.
- Scenes must not contain magic coordinates or raw colors; use node data and
  layout components instead.
- Put new visual elements in `schematics/` first, then use them in scenes.
- Comments and docstrings are written in English only.
