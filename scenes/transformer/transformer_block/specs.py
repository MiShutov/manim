"""Data for a single Llama 3 8B transformer block."""

from schematics.specs import NodeSpec, SkipSpec

# Llama 3 8B hyperparameters.
D_MODEL = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
D_FF = 14336
SEQ_LEN = 2048

LLAMA_SUBTITLE = "Llama 3 8B  ·  T = 2048"

# Nodes are rendered top-to-bottom in this order.
LLAMA_NODES = [
    NodeSpec(
        label="Input  x",
        shape="(B, T, 4096)",
        detail="hidden states",
        color="input",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="RMSNorm",
        shape="(B, T, 4096)",
        detail="pre-attention norm",
        color="norm",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="Q / K / V  +  RoPE",
        shape="(B, T, 4096) / (B, T, 1024)",
        detail=f"Q: {N_HEADS} × {HEAD_DIM}  ·  K/V: {N_KV_HEADS} × {HEAD_DIM} (GQA)",
        color="attention",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="Scaled dot-product attention",
        shape="(B, 32, T, T)",
        detail="softmax over keys",
        color="attention",
        elems=N_HEADS * SEQ_LEN,
    ),
    NodeSpec(
        label="o_proj",
        shape="(B, T, 4096)",
        detail="merge heads",
        color="attention",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="Add  (+ residual)",
        shape="(B, T, 4096)",
        detail="residual stream",
        color="residual",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="RMSNorm",
        shape="(B, T, 4096)",
        detail="pre-FFN norm",
        color="norm",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="SwiGLU  (gate · up)",
        shape="(B, T, 14336)",
        detail="SiLU(gate) × up",
        color="ffn",
        elems=D_FF,
    ),
    NodeSpec(
        label="down_proj",
        shape="(B, T, 4096)",
        detail="back to d_model",
        color="ffn",
        elems=D_MODEL,
    ),
    NodeSpec(
        label="Add  (+ residual)",
        shape="(B, T, 4096)",
        detail="block output",
        color="residual",
        elems=D_MODEL,
    ),
]

# Residual connections: around attention and around the FFN.
LLAMA_SKIPS = [
    SkipSpec(start=0, end=5, route_offset=0.55),
    SkipSpec(start=5, end=9, route_offset=0.55, start_dy=-0.10, end_dy=0.10),
]
