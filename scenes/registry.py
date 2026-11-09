"""Scene registry used by batch rendering, tests, and galleries."""

from schematics.specs import SceneSpec
from scenes.math.matrix_multiplication import MatrixMultiplication
from scenes.transformer.basic_attention.basic_attention import ClassicalAttention
from scenes.transformer.block_modifications.block_modifications import (
    BlockModifications,
)
from scenes.transformer.bpe.bpe import BPE
from scenes.transformer.bpe.bpe_thumbnail import BPEThumbnail
from scenes.transformer.generation_loop.generation_loop import GenerationLoop
from scenes.transformer.gated_mlp.gated_mlp import GatedMLP
from scenes.transformer.model_anatomy.model_anatomy import TransformerLLM
from scenes.transformer.positional_encoding.positional_encoding import (
    RotaryPositionalEncoding,
)
from scenes.transformer.tokenization.tokenization import Tokenization
from scenes.transformer.tokens.tokens import TokenAnatomy

# Public registry: CLI key -> scene metadata.
SCENES: dict[str, SceneSpec] = {
    "basic_attention": SceneSpec(
        key="attention",
        path="scenes/transformer/basic_attention/basic_attention.py",
        cls=ClassicalAttention,
        title="Original multi-head attention",
        tags=("transformer", "attention", "multi-head"),
    ),
    "positional_encoding": SceneSpec(
        key="positional_encoding",
        path=(
            "scenes/transformer/positional_encoding/"
            "positional_encoding.py"
        ),
        cls=RotaryPositionalEncoding,
        title="Rotary positional encoding (RoPE)",
        tags=("transformer", "attention", "positional-encoding", "rope"),
    ),
    "model_anatomy": SceneSpec(
        key="model_anatomy",
        path="scenes/transformer/model_anatomy/model_anatomy.py",
        cls=TransformerLLM,
        title="Inside a transformer LLM",
        tags=("llm", "transformer", "architecture", "embeddings"),
    ),
    "generation_loop": SceneSpec(
        key="generation_loop",
        path="scenes/transformer/generation_loop/generation_loop.py",
        cls=GenerationLoop,
        title="Transformer-decoder autoregressive generation",
        tags=("llm", "transformer", "generation", "logits"),
    ),
    "block_modifications": SceneSpec(
        key="block_modifications",
        path=(
            "scenes/transformer/block_modifications/"
            "block_modifications.py"
        ),
        cls=BlockModifications,
        title="Transformer block and swap-in modifications",
        tags=("llm", "transformer", "architecture", "moe", "attention"),
    ),
    "gated_mlp": SceneSpec(
        key="gated_mlp",
        path="scenes/transformer/gated_mlp/gated_mlp.py",
        cls=GatedMLP,
        title="Gated MLP (SwiGLU)",
        tags=("llm", "transformer", "mlp", "swiglu", "ffn"),
    ),
    "tokens": SceneSpec(
        key="tokens",
        path="scenes/transformer/tokens/tokens.py",
        cls=TokenAnatomy,
        title="Tokens across image and text modalities",
        tags=("llm", "transformer", "tokenization", "multimodal"),
    ),
    "tokenization": SceneSpec(
        key="tokenization",
        path="scenes/transformer/tokenization/tokenization.py",
        cls=Tokenization,
        title="Tokenization: text to token ids and back",
        tags=("llm", "transformer", "tokenization", "vocabulary", "encode"),
    ),
    "bpe": SceneSpec(
        key="bpe",
        path="scenes/transformer/bpe/bpe.py",
        cls=BPE,
        title="BPE: how a tokenizer learns subwords (SentencePiece)",
        tags=("llm", "transformer", "tokenization", "bpe", "sentencepiece"),
    ),
    "bpe_thumbnail": SceneSpec(
        key="bpe",
        path="scenes/transformer/bpe/bpe_thumbnail.py",
        cls=BPEThumbnail,
        title="BPE video thumbnail",
        tags=("thumbnail", "bpe", "youtube"),
    ),
    "matrix_multiplication": SceneSpec(
        key="matrix_multiplication",
        path="scenes/math/matrix_multiplication.py",
        cls=MatrixMultiplication,
        title="Matrix multiplication",
        tags=("linear-algebra",),
    ),
}
