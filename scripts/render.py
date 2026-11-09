#!/usr/bin/env python3
"""Render one or all registered Manim scenes.

Examples:
    python scripts/render.py --list
    python scripts/render.py llama_block -q l
    python scripts/render.py all -q h
    python scripts/render.py llama_block --still
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenes import SCENES  # noqa: E402

QUALITY = {"l": "-ql", "m": "-qm", "h": "-qh", "p": "-qp", "k": "-qk"}


def main() -> None:
    """Parse CLI arguments and invoke Manim for each selected scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", default="all", help="scene key or 'all'")
    parser.add_argument("-q", "--quality", choices=QUALITY, default="l")
    parser.add_argument("--still", action="store_true", help="render only the last frame")
    parser.add_argument("--media-dir", default="media")
    parser.add_argument("--list", action="store_true", help="list registered scenes")
    args = parser.parse_args()

    if args.list:
        for key, spec in sorted(SCENES.items()):
            tags = ", ".join(spec.tags)
            print(f"{key:24} {spec.path}::{spec.cls.__name__}  [{tags}]")
        return

    # Covers are exported on demand by scripts/thumbnails.py; they stay out of
    # batch renders so a normal "render everything" run does not touch them.
    if args.scene == "all":
        keys = sorted(
            key
            for key, spec in SCENES.items()
            if "thumbnail" not in spec.tags
        )
    else:
        keys = [args.scene]
    if keys[0] not in SCENES:
        parser.error(f"unknown scene {keys[0]!r}; use --list")
    if "thumbnail" in SCENES[keys[0]].tags:
        print(
            f"note: {keys[0]} is a cover; "
            "scripts/thumbnails.py exports it to media/videos/"
        )

    env = os.environ.copy()
    prefix = str(ROOT)
    env["PYTHONPATH"] = (
        prefix + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else prefix
    )

    for key in keys:
        spec = SCENES[key]
        cmd = [
            sys.executable,
            "-m",
            "manim",
            QUALITY[args.quality],
            "--media_dir",
            args.media_dir,
        ]
        if args.still:
            cmd.append("-s")
        cmd.extend([str(ROOT / spec.path), spec.cls.__name__])

        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
