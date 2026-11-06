# Project Instructions

## General
- Analyze the existing architecture before editing.
- Follow existing coding conventions.
- Prefer simple and maintainable solutions.
- Avoid unnecessary dependencies.

## Workflow
- Inspect relevant files before making changes.
- Explain the implementation plan.
- Make changes incrementally.
- Run tests after modifications.
- Fix errors introduced by your changes.

## Code Quality
- Use strict typing where possible.
- Handle errors explicitly.
- Avoid duplicated code.
- Write comments and docstrings in English.

## Safety
- Never delete files without confirmation.
- Never expose secrets or credentials.
- Do not modify unrelated files.

## Diagrams
- Read `docs/SCHEMATICS.md` before creating or changing a diagram.
- Use `scenes/transformer/transformer_block/transformer_block.py` as the pipeline reference.
- Use `scenes/transformer/basic_attention/basic_attention.py` as the computation-graph reference.
- Use `scenes/transformer/tokens/tokens.py` as the multimodal reference.
- Reuse the shared layer in `schematics/` before adding scene-specific code.
- Keep scene files thin: node data, layout calls, animation orchestration.
- Put colors, fonts, sizes, and spacing in `schematics/theme.py`.
- Register every new scene in `scenes/registry.py`.
- Add a sibling `.txt` storyboard next to every new scene file.
- Verify with `python scripts/render.py <key> --still` and `pytest -q`.

## Diagram color grammar
- Purple (`compute`, `complexity`): computation blocks, operations, and
  asymptotic cost labels.
- Green (`activation`, `dimension`): activations, inputs, outputs, tensor
  shapes, and dimension symbols.
- Yellow (`weights`): learned parameter matrices such as `W_Q`, `W_K`, `W_V`,
  `W_O`.
- Attention weights produced by softmax are activations, not parameters:
  show them in green and name them `attn_weights` (or `A` in compact
  diagrams), not `W`.
- Use semantic color keys only; never hard-code a Manim color in a scene.

## Diagram math notation
- Matrix multiplication: `@` in the text, and `\mathbin{@}` inside `MathTex`.
- Element-wise multiplication: `\cdot` (a dot).
- Every index and subscript must be LaTeX. Use `MathTex`,
  `SchematicScene.math_caption()`, and `BadgeSpec(title_tex=True,
  shape_tex=True)` instead of plain `d_k`, `W_Q`, or `d_model` text.
- Always state the axis of a `concat`, `reshape`, or `transpose`, for example
  `concat(heads, dim=-1)`. Never write a bare `concat`.
- Keep the notation legend on the left edge so the vertical space stays free
  for the computation graph.

## Commands
- Manim environment: `~/Utils/miniconda3/envs/manim/bin/python`.
- List scenes: `python scripts/render.py --list`.
- Render one scene: `python scripts/render.py <key> -q l`.
- Smoke tests: `pytest -q`.
