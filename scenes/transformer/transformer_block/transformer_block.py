"""One Llama transformer block: a signal walks through the computation graph.

The scene shows a forward pass through a Llama 3 8B block and updates the
activation shape at every step. Run with:

    manim -pql scenes/llm/llama_block.py LlamaBlock

Notation: B is batch size, T is sequence length (2048 by default).
"""

from manim import DOWN, FadeIn, FadeOut, Flash, Indicate, Text, Write, smooth

from schematics.base import SchematicScene
from schematics.hud import TensorPanel
from schematics.layouts import vertical_pipeline
from schematics.signal import SignalDot
from scenes.transformer.transformer_block.specs import LLAMA_NODES, LLAMA_SKIPS, LLAMA_SUBTITLE


class LlamaBlock(SchematicScene):
    """Animated diagram of a single Llama transformer block."""

    def construct(self):
        """Build the graph, HUD, and signal animation."""
        title = self.title("A Llama transformer block")

        pipeline = vertical_pipeline(
            LLAMA_NODES,
            self.theme,
            x=-0.75,
            top_y=3.12,
            spacing=0.66,
            skips=LLAMA_SKIPS,
        )
        hud = TensorPanel(
            self.theme,
            LLAMA_NODES,
            center=[4.62, 0.42, 0],
            subtitle=LLAMA_SUBTITLE,
        )
        signal = SignalDot(self.theme, pipeline.nodes[0].get_center())

        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)
        self.play(*pipeline.reveal_nodes(), run_time=1.4)
        self.play(*pipeline.reveal_edges(), run_time=1.2)
        self.play(*hud.reveal(), run_time=0.7)

        signal.add_to(self)
        self.play(*signal.reveal(), run_time=0.5)

        for i, spec in enumerate(LLAMA_NODES):
            if i > 0:
                self.play(
                    *signal.advance_to(pipeline.nodes[i].get_center()),
                    run_time=0.45,
                    rate_func=smooth,
                )

            self.play(
                Indicate(
                    pipeline.nodes[i],
                    color=self.theme.color(spec.color),
                    scale_factor=1.04,
                ),
                signal.resize(spec.elems),
                run_time=0.45,
            )
            self.play(*hud.update(spec), run_time=0.32)

            merge = pipeline.residual_merge(i, self.theme)
            if merge is not None:
                skip, dot, move = merge
                self.add(dot)
                self.play(
                    Indicate(
                        skip,
                        color=self.theme.color("residual"),
                        scale_factor=1.0,
                    ),
                    move,
                    run_time=0.55,
                )
                self.play(FadeOut(dot), run_time=0.2)

            self.wait(0.18)

        outro = Text(
            "Output shape = input shape  ·  residual stream preserved",
            font_size=self.theme.size("outro"),
            font=self.theme.font,
            color=self.theme.color("residual"),
        ).move_to([-0.75, -3.55, 0])

        self.play(
            Write(outro),
            Flash(
                signal.group,
                color=self.theme.color("success"),
                flash_radius=0.5,
            ),
            run_time=0.9,
        )
        self.wait(2)
