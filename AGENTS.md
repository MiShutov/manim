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
- Use `scenes/llm/llama_block.py` as the pipeline reference.
- Use `scenes/llm/attention.py` as the tensor-detail reference.
- Reuse the shared layer in `schematics/` before adding scene-specific code.
- Keep scene files thin: node data, layout calls, animation orchestration.
- Put colors, fonts, sizes, and spacing in `schematics/theme.py`.
- Register every new scene in `scenes/registry.py`.
- Verify with `python scripts/render.py <key> --still` and `pytest -q`.

## Commands
- Manim environment: `~/Utils/miniconda3/envs/manim/bin/python`.
- List scenes: `python scripts/render.py --list`.
- Render one scene: `python scripts/render.py <key> -q l`.
- Smoke tests: `pytest -q`.
