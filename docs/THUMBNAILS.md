# Video Covers (Thumbnails)

Covers are Manim scenes, not hand-drawn images. They reuse the diagram palette
and typography, so a cover and its video read as one product, and they rebuild
deterministically from source.

Reference implementation: `scenes/transformer/bpe/bpe_thumbnail.py`
(cover for the BPE video). Copy it when starting a new cover.

## What a cover must do

- **Promise, don't process.** A cover shows the punchline of the video in one
  glance (one word turned into two tokens), not the pipeline that produced it.
  Rows of intermediate states, legends, vocabularies, and captions stay in the
  video.
- **Read at 320x180.** Every element is checked at YouTube's smallest preview
  size. If a label needs to be read there, it must be large; if it is decoration,
  it must be quiet.
- **One idea per cover.** One headline plus one story card. Never two stories.
- **Match the video.** Same background, palette, fonts, and chip language as the
  scene it belongs to.

## Canvas and output

- 16:9, built on the default Manim frame (`14.222 x 8.0` units).
- Render the last frame at 1920x1080: `manim -s -qh`.
- Master stays in `media/images/<module>/`.
- The upload copy is 1280x720 and goes next to its video:

  ```
  media/videos/<video_key>/thumbnail.png
  ```

Export both with the project script:

```bash
python scripts/thumbnails.py bpe     # one cover
python scripts/thumbnails.py         # every registered cover
```

## Generation policy

Covers are produced **only on explicit request**. They are not part of the
regular render or test loop:

- `pytest -q` skips every scene tagged `thumbnail`;
- `python scripts/render.py all` skips them as well;
- `python scripts/thumbnails.py <video_key>` is the only way to build or
  rebuild a cover.

Do not regenerate a cover while working on a video scene, the palette, the
registry, or the docs. Touch a cover only when the user asks for one.

## Composition template

```text
+--------------------------------------------------------------------+
|  SUBWORD TOKENIZER                    +------------------------+  |
|                                       |        lowest          |  |
|  BPE                                  |           |            |  |
|  _______________                      |        low   est       |  |
|  how a tokenizer                      |       1024   2048      |  |
|  learns subwords                      +------------------------+  |
|  [ SentencePiece ]                                                 |
+--------------------------------------------------------------------+
```

Two blocks: a typographic column on the left, one story card on the right.
The left block stays identical between covers of the same series; only the
name, subtitle, and tag change.

### Left column (headline)

Left-aligned at `x = -6.65`, four stacked elements:

| Element | Style | Value |
| --- | --- | --- |
| Eyebrow | Inter Bold, purple | 30 pt, y = 3.12 |
| Name | Inter Bold, white | 210 pt, y = 1.52 |
| Rule | purple line, round caps | stroke 9, width = name width, y = 0.28 |
| Subtitle | Inter, two lines | 46 pt, y = -0.58, first line white, second muted |
| Tag | purple pill | 32 pt inside a 0.52-padded pill, y = -2.36 |

The name is the only element at 210 pt. Do not let the subtitle or the card
compete with it.

### Right card (the story)

Build the story first, center it, then hug it with a frame:

1. **Input** — the word, mono, green, at the same size as the tokens below.
2. **Arrow** — green, stroke 16, head ratio 0.5, short and vertical.
3. **Result chips** — `TokenChip`s, green, same type size as the input, so the
   cover says "this is the same word, cut in two".
4. **Ids** — muted mono chips, centered under their token.

Then wrap the stack:

```python
content = VGroup(word, arrow, tokens, ids)
content.move_to(PANEL_CENTER)

rect = RoundedRectangle(
    width=content.width + 2 * PANEL_PAD,     # PANEL_PAD = 0.55
    height=content.height + 2 * PANEL_PAD,
    corner_radius=0.22,
    stroke_color=theme.color("border"),
    stroke_width=2.0,
    fill_color=theme.color("panel"),
    fill_opacity=1.0,
).move_to(content.get_center())
```

Never hard-code card width or height: the card must keep hugging the text when
fonts change.

## Typography scale

Covers need their own scale. Extend the shared sizes instead of littering the
scene with numbers:

```python
THUMBNAIL_SIZES = {
    **DARK.sizes,
    "thumb_eyebrow": 30,
    "thumb_headline": 210,
    "thumb_subtitle": 46,
    "thumb_tag": 32,
    "thumb_word": 52,
    "thumb_token": 52,
    "thumb_id": 30,
}
THUMBNAIL_THEME = Theme(name="bpe-thumbnail", sizes=THUMBNAIL_SIZES)


class BPEThumbnail(SchematicScene):
    theme = THUMBNAIL_THEME
    video_key = "bpe"    # media/videos/bpe/thumbnail.png
```

Rules:

- Input word and result tokens share one size (`thumb_word == thumb_token`).
- Ids are the only small mono text; they are secondary by design.
- No captions, no legends, no footnotes: at 320x180 they turn into noise.

## Color grammar

| Color | Key | On a cover |
| --- | --- | --- |
| Green | `activation` | input word, result tokens, flow arrow |
| Purple | `compute` | eyebrow, rule, tag — typographic accents only |
| White | `text` | name and first subtitle line |
| Grey | `muted` | second subtitle line, token ids |
| Yellow | `weights` | unused |

The flow arrow is the one deliberate deviation from the diagram grammar: on a
diagram purple means "computation", but a cover needs one continuous green
chain from the word to its tokens. Keep purple for typography.

## Adding a cover for a new video

1. Create `scenes/<topic>/<video>_thumbnail.py` with class
   `<Name>Thumbnail(SchematicScene)` and `video_key = "<video>"`.
2. Give it a sibling storyboard `<video>_thumbnail.txt` (purpose, composition,
   color grammar, frame list, constraints).
3. Register it in `scenes/registry.py`:

   ```python
   "bpe_thumbnail": SceneSpec(
       key="bpe",
       path="scenes/transformer/bpe/bpe_thumbnail.py",
       cls=BPEThumbnail,
       title="BPE video thumbnail",
       tags=("thumbnail", "bpe", "youtube"),
   ),
   ```

   The `thumbnail` tag is what `scripts/thumbnails.py` looks for and what
   keeps the cover out of batch renders and tests.
4. Export: `python scripts/thumbnails.py <video>`; the 1280x720 file lands in
   `media/videos/<video>/thumbnail.png`.
5. Inspect the result, including a 320x180 downscale. The cover is not part of
   `pytest`, so this visual check is the only verification it gets.

## Skeleton

```python
from manim import (
    BOLD, DOWN, FadeIn, LEFT, LaggedStart, Line, RIGHT, RoundedRectangle,
    Text, UP, VGroup, Arrow, CapStyleType, Create,
)

from schematics import DARK, SchematicScene, Theme, TokenChip

THUMBNAIL_SIZES = {**DARK.sizes, "thumb_headline": 210, "thumb_word": 52,
                   "thumb_token": 52, "thumb_id": 30}
THUMBNAIL_THEME = Theme(name="cover", sizes=THUMBNAIL_SIZES)


class MyThumbnail(SchematicScene):
    theme = THUMBNAIL_THEME
    video_key = "my_video"

    def construct(self):
        theme = self.theme
        headline = Text("NAME", font_size=theme.size("thumb_headline"),
                        font=theme.font, weight=BOLD,
                        color=theme.color("text")).move_to(
            [-6.65, 1.52, 0], aligned_edge=LEFT)
        rule = Line([-6.65, 0.28, 0], [-6.65 + headline.width, 0.28, 0],
                    stroke_width=9, color=theme.color("compute"),
                    cap_style=CapStyleType.ROUND)

        word = Text("input", font_size=theme.size("thumb_word"),
                    font=theme.mono_font,
                    color=theme.color("activation"))
        tokens = VGroup(*[
            TokenChip(text, theme, color_key="activation",
                      height=0.95, pad=0.18, size="thumb_token", mono=True)
            for text in ("first", "second")
        ]).arrange(RIGHT, buff=0.16)
        arrow = Arrow([0, 0.45, 0], [0, -0.05, 0], buff=0.0,
                      stroke_width=16, max_tip_length_to_length_ratio=0.5)
        arrow.set_color(theme.color("activation"))

        content = VGroup(word, arrow, tokens).arrange(DOWN, buff=0.30)
        content.move_to([3.62, 0.05, 0])
        panel = RoundedRectangle(
            width=content.width + 2 * 0.55,
            height=content.height + 2 * 0.55, corner_radius=0.22,
            stroke_color=theme.color("border"), stroke_width=2.0,
            fill_color=theme.color("panel"), fill_opacity=1.0,
        ).move_to(content.get_center())

        self.play(FadeIn(panel), FadeIn(headline), Create(rule))
        self.play(FadeIn(content))
        self.wait(1.0)
```

## Checklist

1. The cover was requested explicitly; do not rebuild it "just in case".
2. `python scripts/thumbnails.py <video>` renders and exports without errors.
3. Downscale the PNG to 320x180: the name and the result chips are readable.
4. Text is spelled exactly as intended; no placeholder strings.
5. Nothing important within 0.45 units of the frame edge; bottom-right corner
   stays free for YouTube's duration badge.
6. One idea: no process rows, no captions, no vocabulary dumps.
7. The storyboard reflects the final composition.

## Pitfalls

- `VGroup.arrange(LEFT, ...)` reverses the order; use `RIGHT` for left-to-right.
- Do not hide a tokenizer marker (such as `▁`) on a cover if the result stops
  looking like plain text. Explain the marker in the video instead.
- Do not shrink fonts to fill a fixed-size block; size the block after the text.
- Do not reuse diagram sizes (`caption`, `small`) for cover text — they vanish at
  preview size.
- Do not add a second arrow or a second card to "tell the whole story".
