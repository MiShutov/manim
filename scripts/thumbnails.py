#!/usr/bin/env python3
"""Render every video cover and save it next to its video.

A cover is the last frame of a scene tagged ``thumbnail`` in the registry.
The full-size master stays in ``media/images/<module>/``; this script also
writes YouTube's 1280x720 copy to::

    media/videos/<video_key>/thumbnail.png

Examples:
    python scripts/thumbnails.py          # every registered cover
    python scripts/thumbnails.py bpe      # just the bpe cover
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schematics import SceneSpec  # noqa: E402
from scenes import SCENES  # noqa: E402

COVER_SIZE = (1280, 720)
THUMBNAIL_TAG = "thumbnail"


def cover_scenes() -> dict[str, SceneSpec]:
    """Map video key to its registered cover scene.

    Raises:
        ValueError: If a cover scene does not declare ``video_key``.
    """
    covers: dict[str, SceneSpec] = {}
    for spec in SCENES.values():
        if THUMBNAIL_TAG not in spec.tags:
            continue
        video_key = getattr(spec.cls, "video_key", None)
        if video_key is None:
            raise ValueError(
                f"{spec.cls.__name__} is tagged '{THUMBNAIL_TAG}' but has no "
                "video_key class attribute"
            )
        covers[video_key] = spec
    return covers


def render_still(spec: SceneSpec, media_dir: Path) -> Path:
    """Render the cover scene as one 1920x1080 frame and return its file."""
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
        "-qh",
        "--media_dir",
        str(media_dir),
        str(ROOT / spec.path),
        spec.cls.__name__,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    module = Path(spec.path).stem
    matches = sorted(
        (media_dir / "images" / module).glob(f"{spec.cls.__name__}_*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"no still for {spec.cls.__name__} under "
            f"{media_dir / 'images' / module}"
        )
    return matches[-1]


def export_cover(master: Path, video_key: str, media_dir: Path) -> Path:
    """Downscale ``master`` to 1280x720 into the video's folder."""
    target_dir = media_dir / "videos" / video_key
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "thumbnail.png"

    with Image.open(master) as image:
        image.convert("RGB").resize(COVER_SIZE, Image.LANCZOS).save(
            target, optimize=True
        )
    return target


def main() -> None:
    """Render the selected covers and write their 1280x720 copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        default="all",
        help="video key whose cover to render, or 'all'",
    )
    parser.add_argument("--media-dir", default="media")
    args = parser.parse_args()

    covers = cover_scenes()
    keys = sorted(covers) if args.video == "all" else [args.video]
    if keys[0] not in covers:
        parser.error(
            f"unknown video key {keys[0]!r}; known keys: "
            f"{', '.join(sorted(covers))}"
        )

    media_dir = (ROOT / args.media_dir).resolve()
    for key in keys:
        spec = covers[key]
        master = render_still(spec, media_dir)
        target = export_cover(master, key, media_dir)
        print(f"cover  {key:24} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
