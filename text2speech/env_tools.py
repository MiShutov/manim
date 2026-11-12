"""Helpers for running project scripts inside the conda environment.

The console scripts in this folder are plain Python with a ``python3``
shebang. They may therefore start in the system interpreter, which does not
have torch or the TTS dependencies. The helpers below find the project
environment and restart the script with it when needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_ENV_NAME = "f5tts"


def conda_python() -> str:
    """Return the interpreter of the project conda environment."""
    env_name = os.environ.get("CONDA_ENV", DEFAULT_ENV_NAME)
    candidate = Path.home() / "Utils" / "miniconda3" / "envs" / env_name / "bin" / "python"
    if candidate.is_file():
        return str(candidate)

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def ensure_module(module: str) -> None:
    """Restart the current script in the project environment if needed."""
    try:
        __import__(module)
    except ModuleNotFoundError:
        python = conda_python()
        if Path(python).resolve() != Path(sys.executable).resolve():
            os.execv(python, [python, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
        sys.exit(
            f"ERROR: module '{module}' is missing. "
            "Install dependencies with pip install -r requirements.txt"
        )
