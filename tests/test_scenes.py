"""Smoke tests: every scene must build and render its last frame."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenes import SCENES  # noqa: E402

# Covers are generated only on explicit request (``scripts/thumbnails.py``),
# never as a side effect of the regular smoke run.
DIAGRAM_KEYS = sorted(
    key for key, spec in SCENES.items() if "thumbnail" not in spec.tags
)


@pytest.mark.parametrize("key", DIAGRAM_KEYS)
def test_scene_renders_last_frame(key: str, tmp_path: Path) -> None:
    """Render a scene with ``-s -ql`` and fail on a non-zero exit code."""
    spec = SCENES[key]
    env = os.environ.copy()
    prefix = str(ROOT)
    env["PYTHONPATH"] = (
        prefix + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else prefix
    )

    cmd = [
        sys.executable,
        "-m",
        "manim",
        "-s",
        "-ql",
        "--media_dir",
        str(tmp_path),
        str(ROOT / spec.path),
        spec.cls.__name__,
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=300)
