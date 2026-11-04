# Manim schematics

Technical diagrams for ML/LLM topics: classical multi-head attention, a
Llama transformer block, linear algebra, and future scenes.

> For agents and models: see `docs/SCHEMATICS.md` for the full library guide.
> It is loaded automatically through `opencode.jsonc`.

## Structure

```text
AGENTS.md           project rules (auto-loaded)
docs/SCHEMATICS.md  agent guide for the schematics library
opencode.jsonc      loads docs/SCHEMATICS.md as instructions
schematics/         reusable library
  theme.py          colors, fonts, sizes, log scale
  base.py           SchematicScene: background, fonts, title()
  specs.py          NodeSpec / BadgeSpec / SkipSpec / SceneSpec
  primitives.py     Node, Badge, SkipConnection, connect()
  layouts.py        vertical_pipeline(), Pipeline
  hud.py            TensorPanel
  signal.py         SignalDot
  tensors.py        MatrixGrid, WeightBars
scenes/             thin scenes: data plus composition only
  llm/attention.py
  llm/attention_specs.py
  llm/llama_block.py
  linear_algebra/matrix_multiplication.py
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
python scripts/render.py attention -q l
python scripts/render.py llama_block -q l

# all scenes, 1080p
python scripts/render.py all -q h

# last frame only (fast layout check)
python scripts/render.py llama_block --still
```

Direct invocation also works after the editable install:

```bash
~/Utils/miniconda3/envs/manim/bin/manim -pql \
  scenes/llm/llama_block.py LlamaBlock
```

## Adding a new scene

1. Create `scenes/<topic>/<name>.py` and, for shared data, a matching
   `<name>_specs.py`.
2. Describe graph nodes with `NodeSpec` and tensor-shape strips with
   `BadgeSpec`.
3. Build pipelines with `vertical_pipeline(...)`; build matrix and bar
   schematics with `MatrixGrid` and `WeightBars`.
4. Add `TensorPanel` if you need an activation-shape HUD.
5. Register the scene in `scenes/registry.py`.
6. Run `python scripts/render.py <key> --still` and `pytest -q`.

## Style

- Use semantic color keys: `norm`, `attention`, `ffn`, `residual`, `input`,
  `query`, `key`, `value`, `score`, `weight`, `output`.
- Change fonts and sizes in `schematics/theme.py`.
- Scenes must not contain magic coordinates or raw colors; use node data and
  layout components instead.
- Put new visual elements in `schematics/` first, then use them in scenes.
- Comments and docstrings are written in English only.
