"""Scene registry used by batch rendering, tests, and galleries."""

from schematics.specs import SceneSpec
from scenes.math.matrix_multiplication import MatrixMultiplication
from scenes.transformer.basic_attention.basic_attention import ClassicalAttention
from scenes.transformer.transformer_block.transformer_block import LlamaBlock
from scenes.transformer.tokens.tokens import TokenAnatomy

# Public registry: CLI key -> scene metadata.
SCENES: dict[str, SceneSpec] = {
    "basic_attention": SceneSpec(
        key="attention",
        path="scenes/transformer/basic_attention/basic_attention.py",
        cls=ClassicalAttention,
        title="Classical multi-head attention",
        tags=("llm", "transformer", "attention"),
    ),
    "llama_block": SceneSpec(
        key="transformer_block",
        path="scenes/transformer/transformer_block/transformer_block.py",
        cls=LlamaBlock,
        title="Llama transformer block",
        tags=("llm", "transformer"),
    ),
    "tokens": SceneSpec(
        key="tokens",
        path="scenes/transformer/tokens/tokens.py",
        cls=TokenAnatomy,
        title="Tokens across image and text modalities",
        tags=("llm", "transformer", "tokenization", "multimodal"),
    ),
    "matrix_multiplication": SceneSpec(
        key="matrix_multiplication",
        path="scenes/math/matrix_multiplication.py",
        cls=MatrixMultiplication,
        title="Matrix multiplication",
        tags=("linear-algebra",),
    ),
}
