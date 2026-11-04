# Fonts

Put `.ttf` or `.otf` files here if you want the font to travel with the
repository.

`SchematicScene.setup()` registers every file in this folder through
`manim.register_font`, so the theme only needs the family name:

```python
Theme(font="JetBrains Mono")
```

System fonts (Inter, DejaVu Sans, Noto Sans, Lato, etc.) do not need to be
registered; Pango finds them automatically.
