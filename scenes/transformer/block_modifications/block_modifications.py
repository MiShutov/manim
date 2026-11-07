"""Transformer block: forward pass, then one architecture swap-in at a time.

The block is drawn without a frame as a compact computation graph:
input x -> Normalization -> Self-attention -> circled plus (residual joins
from the left) -> Normalization -> MLP -> circled plus -> output x'. A short
Python-style ``forward`` function is highlighted line by line while the
matching block pulses. Swap-ins are shown in this order: normalization,
attention (causal mask, windowed mask, Gated DeltaNet), MLP (router plus
experts), and finally mHC blocks inserted into both residual paths. When a
block grows, every element below it shifts down by the same amount.

Run with:

    python scripts/render.py block_modifications -q l
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Arrow,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    FadeTransform,
    Flash,
    Indicate,
    LaggedStart,
    MathTex,
    RoundedRectangle,
    Text,
    Transform,
    VGroup,
)

from schematics import Badge, CodeLine, Node, SchematicScene, route_arrow
from schematics.specs import NodeSpec
from scenes.transformer.block_modifications.panels import (
    ATT_PANEL,
    MLP_PANEL,
    MoePanel,
    causal_mask_panel,
    gdn_panel,
    norm_chip,
    windowed_mask_panel,
)
from scenes.transformer.block_modifications.specs import (
    CAPTIONS,
    D_MODEL_TEX,
    FORWARD_STEPS,
    IN_BADGE,
    NOTATION_ROWS,
    NOTATION_TITLE,
    OUT_BADGE,
    PSEUDO_LINES,
    TITLE,
)

# --- Geometry ---------------------------------------------------------------
X0 = -2.2
NODE_W = 1.55
NODE_H = 0.34
BADGE_W = 1.55
BADGE_H = 0.44
PLUS_R = 0.14
GAP = 0.18
INPUT_GAP = 0.36
CHIP_H = 0.20
BUS_X = X0 - 1.62
TOP_Y = 3.50

ROWS = ("input", "n1", "attn", "p1", "n2", "mlp", "p2", "out")
HEIGHTS = {
    "input": BADGE_H,
    "n1": NODE_H,
    "attn": NODE_H,
    "p1": 2 * PLUS_R,
    "n2": NODE_H,
    "mlp": NODE_H,
    "p2": 2 * PLUS_R,
    "out": BADGE_H,
}

# Pseudocode panel and notation legend.
CODE_X = 1.00
CODE_TOP = 2.92
CODE_STEP = 0.30
LEGEND_X = -6.90
LEGEND_TOP = 3.46
LEGEND_STEP = 0.28


class BlockModifications(SchematicScene):
    """Forward pass through one block, then its architecture swap-ins."""

    def construct(self):
        """Build the block, walk the forward pass, then show the swaps."""
        self._build()
        self._play_intro()
        self._play_forward()
        self._play_swapins()
        self._finish()

    # --- Layout --------------------------------------------------------------
    def _layout(self, attn_h=NODE_H, mlp_h=NODE_H, inserts=(), hidden=()):
        """Return centered row centers, chip centers, and visible rows."""
        heights = dict(HEIGHTS, attn=attn_h, mlp=mlp_h)
        ys: dict[str, float] = {}
        chips: dict[str, float] = {}
        visible = [name for name in ROWS if name not in hidden]
        top = TOP_Y
        for name in visible:
            height = heights[name]
            ys[name] = top - height / 2
            top -= height
            top -= INPUT_GAP if name == "input" else GAP
            if name in inserts:
                chips[name] = top - CHIP_H / 2
                top -= CHIP_H + GAP

        # Keep the block centered vertically on the screen.
        bottom = ys[visible[-1]] - heights[visible[-1]] / 2
        offset = -(TOP_Y + bottom) / 2
        ys = {name: y + offset for name, y in ys.items()}
        chips = {name: y + offset for name, y in chips.items()}
        return ys, chips, visible

    def _arrow(self, start, end, *, color_key: str = "muted"):
        """Return a connector with the shared stroke and arrow head."""
        def point(value):
            coords = np.asarray(value, dtype=float).reshape(-1)
            if coords.size == 2:
                coords = np.append(coords, 0.0)
            return coords

        return Arrow(
            point(start),
            point(end),
            buff=0.02,
            stroke_width=self.theme.arrow_stroke,
            color=self.theme.color(color_key),
            tip_length=0.08,
            max_tip_length_to_length_ratio=0.9,
        )

    def _plus(self, y: float) -> VGroup:
        """Return a circled plus: one residual add node."""
        theme = self.theme
        return VGroup(
            Circle(
                radius=PLUS_R,
                stroke_color=theme.color("residual"),
                stroke_width=1.6,
                fill_color=theme.color("residual"),
                fill_opacity=0.12,
            ),
            MathTex("+", font_size=15, color=theme.color("residual")),
        ).move_to([X0, y, 0])

    def _connectors(
        self,
        ys: dict[str, float],
        visible,
        *,
        attn_h=NODE_H,
        mlp_h=NODE_H,
    ) -> dict:
        """Build every arrow and residual route for one layout."""
        theme = self.theme
        heights = dict(HEIGHTS, attn=attn_h, mlp=mlp_h)

        def bottom(name):
            return ys[name] - heights[name] / 2

        def top(name):
            return ys[name] + heights[name] / 2

        connectors = {}
        for upper, lower in zip(visible[:-1], visible[1:]):
            connectors[f"{upper}_{lower}"] = self._arrow(
                [X0, bottom(upper)], [X0, top(lower)]
            )

        first_upper, first_lower = visible[0], visible[1]
        fork1 = (bottom(first_upper) + top(first_lower)) / 2
        fork2 = ys["p1"] - PLUS_R - GAP / 2
        connectors["route1"] = route_arrow(
            [
                [X0, fork1, 0],
                [BUS_X, fork1, 0],
                [BUS_X, ys["p1"], 0],
                [X0 - PLUS_R, ys["p1"], 0],
            ],
            theme,
            color_key="residual",
            stroke_width=theme.arrow_stroke,
            tip_length=0.08,
        )
        connectors["route2"] = route_arrow(
            [
                [X0, fork2, 0],
                [BUS_X, fork2, 0],
                [BUS_X, ys["p2"], 0],
                [X0 - PLUS_R, ys["p2"], 0],
            ],
            theme,
            color_key="residual",
            stroke_width=theme.arrow_stroke,
            tip_length=0.08,
        )
        return connectors

    def _morph(
        self,
        *,
        attn=(NODE_W, NODE_H),
        mlp=(NODE_W, NODE_H),
        inserts=(),
        hidden=(),
        attn_label_dx: float = 0.0,
        run_time: float = 0.9,
    ):
        """Relayout the block: grow panels, shift rows, keep it centered."""
        attn_w, attn_h = attn
        mlp_w, mlp_h = mlp
        ys, chips, visible = self._layout(attn_h, mlp_h, inserts, hidden)

        animations = []
        for name in visible:
            mobject = self.row_mobs[name]
            if name == "attn":
                animations.append(
                    self.attn.rect.animate.stretch_to_fit_width(attn_w)
                    .stretch_to_fit_height(attn_h)
                    .move_to([self._panel_x(attn_w), ys[name], 0])
                )
                if self.attn_label_visible:
                    animations.append(
                        self.attn.label.animate.move_to(
                            [X0 + attn_label_dx, ys[name], 0]
                        )
                    )
            elif name == "mlp":
                animations.append(
                    self.mlp.rect.animate.stretch_to_fit_width(mlp_w)
                    .stretch_to_fit_height(mlp_h)
                    .move_to([self._panel_x(mlp_w), ys[name], 0])
                )
                if self.mlp_label_visible:
                    animations.append(
                        self.mlp.label.animate.move_to(
                            [self._panel_x(mlp_w), ys[name], 0]
                        )
                    )
            else:
                animations.append(mobject.animate.move_to([X0, ys[name], 0]))

        for name in ROWS:
            if name in hidden and name not in self.hidden:
                animations.append(FadeOut(self.row_mobs[name]))
            elif name not in hidden and name in self.hidden:
                self.row_mobs[name].move_to([X0, ys[name], 0])
                animations.append(FadeIn(self.row_mobs[name]))

        new_connectors = self._connectors(
            ys, visible, attn_h=attn_h, mlp_h=mlp_h
        )
        if visible == self.visible:
            for key, old in self.connectors.items():
                animations.append(Transform(old, new_connectors[key]))
        else:
            animations.append(FadeOut(self.connector_group))
            self.connectors = new_connectors
            self.connector_group = VGroup(*new_connectors.values())
            animations.append(FadeIn(self.connector_group))

        self.ys = ys
        self.chip_ys = chips
        self.visible = visible
        self.hidden = set(hidden)
        self.play(*animations, run_time=run_time)

    @staticmethod
    def _panel_x(width: float) -> float:
        """Grow a node symmetrically around the vertical flow."""
        return X0

    # --- Construction --------------------------------------------------------
    def _build(self):
        """Create every mobject of the standard block."""
        theme = self.theme

        self.title_m = self.title(TITLE, x=0.0, y=3.82, size="title_small")

        self.in_badge = Badge(IN_BADGE, theme, width=BADGE_W, height=BADGE_H)
        self.out_badge = Badge(OUT_BADGE, theme, width=BADGE_W, height=BADGE_H)

        self.norm1 = self._node("Normalization")
        self.attn = self._node("Self-attention")
        self.norm2 = self._node("Normalization")
        self.mlp = self._node("MLP")
        self.plus1 = self._plus(0.0)
        self.plus2 = self._plus(0.0)

        self.row_mobs = {
            "input": self.in_badge,
            "n1": self.norm1,
            "attn": self.attn,
            "p1": self.plus1,
            "n2": self.norm2,
            "mlp": self.mlp,
            "p2": self.plus2,
            "out": self.out_badge,
        }

        self.ys, self.chip_ys, self.visible = self._layout()
        self.hidden: set[str] = set()
        for name, mobject in self.row_mobs.items():
            mobject.move_to([X0, self.ys[name], 0])
        self.connectors = self._connectors(self.ys, self.visible)
        self.connector_group = VGroup(*self.connectors.values())

        self.attn_label_visible = True
        self.mlp_label_visible = True

        self._build_code()
        self.legend = self._notation()

    def _node(self, label: str) -> Node:
        """Return one block node with the compact label size."""
        return Node(
            NodeSpec(label, D_MODEL_TEX, "", "compute", 1, label_size="help"),
            self.theme,
            width=NODE_W,
            height=NODE_H,
        )

    def _build_code(self):
        """Build the Python-style pseudocode panel."""
        theme = self.theme
        self.code_lines: list[CodeLine | None] = []
        y = CODE_TOP
        for index, segments in enumerate(PSEUDO_LINES):
            if segments is None:
                self.code_lines.append(None)
                y -= CODE_STEP
                continue
            line = CodeLine(segments, theme, size="help")
            line.move_to(
                [CODE_X + (0 if index == 0 else 0.30), y, 0],
                aligned_edge=LEFT,
            )
            self.code_lines.append(line)
            y -= CODE_STEP
        self.code_group = VGroup(
            *[line for line in self.code_lines if line is not None]
        )

    def _notation(self) -> VGroup:
        """Build the notation legend in the top-left corner."""
        theme = self.theme
        heading = Text(
            NOTATION_TITLE,
            font_size=theme.size("small"),
            font=theme.font,
            color=theme.color("muted"),
        ).move_to([LEGEND_X, LEGEND_TOP, 0], aligned_edge=LEFT)

        rows = VGroup()
        for index, (symbol_tex, description, color_key) in enumerate(
            NOTATION_ROWS
        ):
            y = LEGEND_TOP - LEGEND_STEP - index * LEGEND_STEP
            symbol = MathTex(
                symbol_tex,
                font_size=12,
                color=theme.color(color_key),
            ).move_to([LEGEND_X, y, 0], aligned_edge=LEFT)
            label = Text(
                description,
                font_size=10,
                font=theme.font,
                color=theme.color("muted"),
            ).move_to([LEGEND_X + 0.80, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(symbol, label))
        return VGroup(heading, rows)

    # --- Intro and forward pass ---------------------------------------------
    def _play_intro(self):
        """Reveal the block and the forward pseudocode."""
        self.play(
            FadeIn(self.title_m, shift=DOWN * 0.15),
            FadeIn(self.legend),
            run_time=0.7,
        )
        self.play(*self.in_badge.reveal(), run_time=0.4)
        self.play(
            LaggedStart(
                *[
                    FadeIn(self.row_mobs[name], shift=RIGHT * 0.06)
                    for name in ("n1", "attn", "p1", "n2", "mlp", "p2")
                ],
                lag_ratio=0.08,
            ),
            run_time=1.0,
        )
        self.play(
            Create(self.connectors["input_n1"]),
            Create(self.connectors["n1_attn"]),
            Create(self.connectors["attn_p1"]),
            Create(self.connectors["p1_n2"]),
            Create(self.connectors["n2_mlp"]),
            Create(self.connectors["mlp_p2"]),
            Create(self.connectors["p2_out"]),
            run_time=0.8,
        )
        self.play(
            Create(self.connectors["route1"]),
            Create(self.connectors["route2"]),
            *self.out_badge.reveal(),
            run_time=0.6,
        )
        self.play(
            LaggedStart(
                *[FadeIn(line, shift=RIGHT * 0.08) for line in self.code_group],
                lag_ratio=0.06,
            ),
            run_time=0.8,
        )

    def _hl(self, line: CodeLine) -> Indicate:
        """Highlight a pseudocode line with the signal color."""
        return Indicate(
            line, color=self.theme.color("signal"), scale_factor=1.06
        )

    def _play_forward(self):
        """Pulse each block element while its code line highlights."""
        theme = self.theme
        for line_index, kind, payload in FORWARD_STEPS:
            animations = [self._hl(self.code_lines[line_index])]
            if kind == "input":
                animations.append(
                    Indicate(
                        self.in_badge,
                        color=theme.color("activation"),
                        scale_factor=1.05,
                    )
                )
            elif kind == "output":
                animations.append(
                    Indicate(
                        self.out_badge,
                        color=theme.color("activation"),
                        scale_factor=1.05,
                    )
                )
            elif kind == "route":
                animations.append(
                    Indicate(
                        self.connectors[f"route{payload + 1}"],
                        color=theme.color("residual"),
                        scale_factor=1.0,
                    )
                )
            elif kind == "node":
                animations.append(
                    Indicate(
                        [self.norm1, self.attn, self.norm2, self.mlp][payload],
                        color=theme.color("compute"),
                        scale_factor=1.05,
                    )
                )
            else:  # "plus"
                animations.append(
                    Indicate(
                        [self.plus1, self.plus2][payload],
                        color=theme.color("residual"),
                        scale_factor=1.15,
                    )
                )
            self.play(*animations, run_time=0.55)
            self.wait(0.05)

        self.play(
            Flash(
                self.out_badge,
                color=theme.color("success"),
                flash_radius=0.4,
            ),
            run_time=0.5,
        )
        self.wait(0.3)
        self.play(FadeOut(self.code_group), run_time=0.5)

    # --- Swap-ins ------------------------------------------------------------
    def _caption(self, key: str) -> VGroup:
        """Return the swap-in caption: title plus reference line."""
        theme = self.theme
        title_text, ref_text = CAPTIONS[key]
        title = Text(
            title_text,
            font_size=theme.size("caption"),
            font=theme.font,
            color=theme.color("text"),
        ).move_to([1.70, 0.58, 0], aligned_edge=LEFT)
        ref = Text(
            ref_text,
            font_size=10,
            font=theme.mono_font,
            color=theme.color("muted"),
        )
        if ref.width > 5.1:
            ref.scale_to_fit_width(5.1)
        ref.move_to([1.70, 0.34, 0], aligned_edge=LEFT)
        return VGroup(title, ref)

    def _node_label(self, node: Node, text: str) -> Text:
        """Return a replacement label for a narrow node."""
        theme = self.theme
        label = Text(
            text,
            font_size=theme.size("caption"),
            font=theme.font,
            color=theme.color("text"),
        )
        if label.width > NODE_W - 0.24:
            label.scale_to_fit_width(NODE_W - 0.24)
        return label.move_to(node.rect.get_center())

    def _play_swapins(self):
        """Run every swap-in in order."""
        self._play_norm()
        self._play_attention()
        self._play_moe()
        self._play_mhc()

    # --- Normalization -------------------------------------------------------
    def _play_norm(self):
        """Show LayerNorm/RMSNorm names, then the extra Gemma 4 norms."""
        theme = self.theme
        caption = self._caption("norm")
        self.play(FadeIn(caption), run_time=0.4)

        original = self.norm1.label
        previous = (self.norm1.label, self.norm2.label)
        for name in ("LayerNorm", "RMSNorm"):
            first = self._node_label(self.norm1, name)
            second = self._node_label(self.norm2, name)
            self.play(
                FadeTransform(previous[0], first),
                FadeTransform(previous[1], second),
                run_time=0.6,
            )
            self.wait(1.2)
            previous = (first, second)
        self.play(
            FadeTransform(previous[0], self.norm1.label),
            FadeTransform(previous[1], self.norm2.label),
            FadeOut(caption),
            run_time=0.6,
        )
        self.wait(0.2)

        # Gemma 4: extra norms before and after each residual add, plus
        # QK-norm inside the attention module.
        self._morph(
            attn=ATT_PANEL,
            inserts=("attn", "p1", "mlp", "p2"),
            hidden=("n1", "n2"),
            attn_label_dx=-0.55,
        )
        gemma_caption = self._caption("gemma")
        chips = VGroup(
            *[
                norm_chip(theme).move_to([X0, self.chip_ys[name], 0])
                for name in ("attn", "p1", "mlp", "p2")
            ]
        )
        panel_x = self._panel_x(ATT_PANEL[0])
        panel_y = self.ys["attn"]
        qk_norm = VGroup(
            norm_chip(theme, "Q norm").move_to([panel_x + 0.75, panel_y + 0.20, 0]),
            norm_chip(theme, "K norm").move_to([panel_x + 0.75, panel_y - 0.20, 0]),
        )
        self.play(FadeIn(gemma_caption), run_time=0.3)
        self.play(
            LaggedStart(
                *[
                    FadeIn(chip, shift=DOWN * 0.04)
                    for chip in [*chips, *qk_norm]
                ],
                lag_ratio=0.10,
            ),
            run_time=1.0,
        )
        self.wait(2.6)
        self.play(
            FadeOut(chips),
            FadeOut(qk_norm),
            FadeOut(gemma_caption),
            run_time=0.5,
        )
        self._morph(run_time=0.8)
        self.wait(0.2)

    # --- Attention -----------------------------------------------------------
    def _attention_center(self):
        """Return the center of the expanded attention panel."""
        return [self._panel_x(ATT_PANEL[0]), self.ys["attn"], 0]

    def _play_attention(self):
        """Expand attention into a mask panel, then windowed and GDN."""
        theme = self.theme

        self.attn_label_visible = False
        self._morph(attn=ATT_PANEL)
        causal = causal_mask_panel(theme).move_to(self._attention_center())
        causal_caption = self._caption("causal")
        self.play(FadeOut(self.attn.label), FadeIn(causal_caption), run_time=0.4)
        self.play(FadeIn(causal), run_time=0.6)
        self.wait(1.6)

        windowed = windowed_mask_panel(theme).move_to(self._attention_center())
        windowed_caption = self._caption("windowed")
        self.play(FadeOut(causal_caption), run_time=0.2)
        self.play(
            FadeTransform(causal, windowed),
            FadeIn(windowed_caption),
            run_time=0.6,
        )
        self.wait(2.2)

        gdn = gdn_panel(theme).move_to(self._attention_center())
        gdn_caption = self._caption("gdn")
        self.play(FadeOut(windowed_caption), run_time=0.2)
        self.play(
            FadeTransform(windowed, gdn),
            FadeIn(gdn_caption),
            run_time=0.6,
        )
        self.wait(2.4)

        self.play(
            FadeOut(gdn),
            FadeOut(gdn_caption),
            FadeIn(self.attn.label),
            run_time=0.4,
        )
        self.attn_label_visible = True
        self._morph(run_time=0.8)

    # --- MLP / MoE -----------------------------------------------------------
    def _play_moe(self):
        """Grow the MLP into a vertical router/experts panel and animate it."""
        theme = self.theme
        caption = self._caption("moe")
        self.mlp_label_visible = False
        self._morph(mlp=MLP_PANEL)
        panel = MoePanel(theme).move_to(
            [self._panel_x(MLP_PANEL[0]), self.ys["mlp"], 0]
        )
        self.play(
            FadeOut(self.mlp.label),
            FadeIn(caption),
            run_time=0.4,
        )
        self.play(FadeIn(panel), run_time=0.6)

        muted = theme.color("muted")
        signal = theme.color("signal")
        out_for = {
            0: panel.side_out[0],
            1: panel.center_out,
            2: panel.side_out[1],
        }
        base_width = {
            0: theme.arrow_stroke,
            1: 3.2,
            2: theme.arrow_stroke,
        }
        for active in ((0, 2), (1, 2), (0, 1)):
            self.play(
                Indicate(panel.router, color=theme.color("compute"), scale_factor=1.06),
                run_time=0.4,
            )
            lit = [
                panel.arrows[index].animate.set_stroke(color=signal, width=3.2)
                for index in active
            ] + [
                out_for[index].animate.set_stroke(
                    color=signal, width=base_width[index] + 1.0
                )
                for index in active
            ]
            experts = []
            for index in active:
                rect, label = panel.experts[index]
                experts += [
                    rect.animate.set_stroke(color=theme.color("weights"), width=2.4),
                    label.animate.set_color(theme.color("weights")),
                ]
            self.play(*lit, *experts, run_time=0.6)
            self.wait(0.5)
            self.play(
                *[
                    panel.arrows[index].animate.set_stroke(
                        color=muted, width=theme.arrow_stroke
                    )
                    for index in active
                ],
                *[
                    out_for[index].animate.set_stroke(
                        color=muted, width=base_width[index]
                    )
                    for index in active
                ],
                *[
                    panel.experts[index][0].animate.set_stroke(
                        color=theme.color("compute"), width=1.3
                    )
                    for index in active
                ],
                *[
                    panel.experts[index][1].animate.set_color(
                        theme.color("text")
                    )
                    for index in active
                ],
                run_time=0.3,
            )
        self.wait(0.3)
        self.play(FadeOut(panel), FadeOut(caption), run_time=0.5)
        self.play(FadeIn(self.mlp.label), run_time=0.3)
        self.mlp_label_visible = True
        self._morph(run_time=0.8)

    # --- mHC -----------------------------------------------------------------
    def _mhc_badge(self, y: float) -> VGroup:
        """Return one mHC block placed on the residual bus."""
        theme = self.theme
        rect = RoundedRectangle(
            width=0.72,
            height=0.34,
            corner_radius=0.07,
            stroke_color=theme.color("compute"),
            stroke_width=1.5,
            fill_color=theme.color("panel"),
            fill_opacity=1.0,
        )
        label = Text(
            "mHC",
            font_size=12,
            font=theme.font,
            color=theme.color("text"),
        ).move_to(rect)
        return VGroup(rect, label).move_to([BUS_X, y, 0])

    def _play_mhc(self):
        """Mark both residual paths with mHC blocks."""
        theme = self.theme
        caption = self._caption("mhc")
        fork1 = self.ys["input"] - BADGE_H / 2 - INPUT_GAP / 2
        fork2 = self.ys["p1"] - PLUS_R - GAP / 2
        blocks = VGroup(
            self._mhc_badge((fork1 + self.ys["p1"]) / 2),
            self._mhc_badge((fork2 + self.ys["p2"]) / 2),
        )
        residual = theme.color("residual")
        self.play(FadeIn(caption), run_time=0.4)
        self.play(
            FadeIn(blocks, scale=0.9),
            Indicate(self.connectors["route1"], color=residual, scale_factor=1.0),
            Indicate(self.connectors["route2"], color=residual, scale_factor=1.0),
            run_time=0.9,
        )
        self.wait(2.8)
        self.play(FadeOut(blocks), FadeOut(caption), run_time=0.6)

    # --- Finish --------------------------------------------------------------
    def _finish(self):
        """Center the plain block horizontally and end the scene."""
        group = VGroup(
            *[self.row_mobs[name] for name in ROWS],
            self.connector_group,
        )
        self.play(group.animate.shift(RIGHT * (-X0)), run_time=1.0)
        self.wait(1.0)
