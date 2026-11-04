"""Step-by-step visualization of matrix multiplication."""

import numpy as np
from manim import *


class MatrixMultiplication(Scene):
    """Show how each output element is a dot product of a row and a column."""

    def construct(self):
        """Build the matrices and animate the dot products one by one."""
        # Input matrices.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        B = np.array([
            [7, 8],
            [9, 10],
            [11, 12]
        ])

        C = A @ B

        # Title.
        title = Text(
            "Matrix multiplication",
            font_size=34
        ).to_edge(UP)

        self.play(Write(title))

        # Create the visual matrices.
        mat_A = Matrix(A.tolist()).scale(0.75)
        mat_B = Matrix(B.tolist()).scale(0.75)
        mat_C = Matrix(
            [["?", "?"], ["?", "?"]]
        ).scale(0.75)

        # Layout.
        mat_A.move_to(LEFT * 4.2 + UP * 0.5)
        mat_B.move_to(UP * 0.5)
        mat_C.move_to(RIGHT * 4.2 + UP * 0.5)

        label_A = MathTex("A").next_to(mat_A, UP)
        label_B = MathTex("B").next_to(mat_B, UP)
        label_C = MathTex("C").next_to(mat_C, UP)

        multiply = MathTex(r"\times").move_to(
            LEFT * 2.1 + UP * 0.5
        )
        equals = MathTex("=").move_to(
            RIGHT * 2.1 + UP * 0.5
        )

        self.play(
            FadeIn(mat_A),
            FadeIn(mat_B),
            FadeIn(mat_C),
            Write(label_A),
            Write(label_B),
            Write(label_C),
            Write(multiply),
            Write(equals)
        )

        # Access individual matrix entries.
        a_entries = mat_A.get_entries()
        b_entries = mat_B.get_entries()
        c_entries = mat_C.get_entries()

        # Explanation caption.
        explanation = Text(
            "Each element is a dot product",
            font_size=25
        ).to_edge(DOWN)

        self.play(Write(explanation))

        # Compute every element of C.
        for i in range(2):
            for j in range(2):

                # Row and column indices for this output element.
                row = [i * 3 + k for k in range(3)]
                col = [k * 2 + j for k in range(3)]

                # Highlight the row of A.
                row_box = SurroundingRectangle(
                    VGroup(*[a_entries[x] for x in row]),
                    color=YELLOW,
                    buff=0.12
                )

                # Highlight the column of B.
                col_box = SurroundingRectangle(
                    VGroup(*[b_entries[x] for x in col]),
                    color=BLUE,
                    buff=0.12
                )

                self.play(
                    Create(row_box),
                    Create(col_box),
                    run_time=0.7
                )

                # Element-wise multiplication.
                products = []

                for k in range(3):
                    a = A[i, k]
                    b = B[k, j]
                    products.append(f"{a}\\cdot {b}")

                    self.play(
                        Indicate(a_entries[row[k]],
                                 color=YELLOW),
                        Indicate(b_entries[col[k]],
                                 color=BLUE),
                        run_time=0.5
                    )

                # Dot-product formula.
                formula = MathTex(
                    " + ".join(products),
                    "=",
                    str(C[i, j]),
                    font_size=34
                )

                formula.move_to(DOWN * 2.1)

                self.play(Write(formula))
                self.wait(0.5)

                # Write the result into C.
                result = MathTex(
                    str(C[i, j])
                ).move_to(c_entries[i * 2 + j])

                self.play(
                    Transform(
                        c_entries[i * 2 + j],
                        result
                    ),
                    Flash(
                        c_entries[i * 2 + j],
                        color=GREEN
                    )
                )

                self.play(
                    FadeOut(row_box),
                    FadeOut(col_box),
                    FadeOut(formula)
                )

        # Final result.
        final_text = Text(
            "Matrix multiplication completed!",
            font_size=26,
            color=GREEN
        ).to_edge(DOWN)

        self.play(
            FadeOut(explanation),
            Write(final_text),
            Indicate(mat_C, color=GREEN)
        )

        self.wait(2)
