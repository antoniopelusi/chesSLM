import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash-attention kernel support.

    Projects the input into queries, keys, and values with a single fused
    linear layer, then applies :func:`torch.nn.functional.scaled_dot_product_attention`
    with ``is_causal=True`` so each position can only attend to earlier
    positions (including itself). The output is projected back to *d_model*.

    Weight tying and bias-free projections follow the GPT-2 / nanoGPT design.

    Args:
        d_model (int): Model embedding dimension. Must be divisible by *n_heads*.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention weights
            during training; set to 0 during inference automatically.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

    def forward(self, x):
        """Compute causal multi-head self-attention for a sequence.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, C)`` where *B* is
                batch size, *T* is sequence length, and *C* is *d_model*.

        Returns:
            torch.Tensor: Output tensor of the same shape ``(B, T, C)``.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    """Single transformer block: pre-norm attention followed by pre-norm feed-forward.

    Applies layer normalisation *before* each sub-layer (Pre-LN) and adds the
    sub-layer output as a residual, which improves gradient flow compared to
    the original Post-LN design from "Attention Is All You Need".

    The feed-forward network expands the dimension by a factor of
    ``d_ff / d_model`` (typically 4×), applies GELU activation, and projects
    back to *d_model*.

    Args:
        d_model (int): Embedding/hidden dimension.
        n_heads (int): Number of attention heads passed to
            :class:`CausalSelfAttention`.
        d_ff (int): Inner dimension of the feed-forward network.
        dropout (float): Dropout probability used in both the attention layer
            and the feed-forward network.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the attention and feed-forward sub-layers with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

        Returns:
            torch.Tensor: Output tensor of the same shape ``(B, T, C)``.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ChesSLM(nn.Module):
    """Decoder-only transformer language model for chess move sequence prediction.

    The architecture follows the GPT family: token and positional embeddings are
    summed and passed through a stack of :class:`Block` layers, a final layer
    norm, and an unembedding head that shares weights with the token embedding
    matrix (weight tying).

    The model is trained to predict the next move token in a game sequence;
    at inference time the logits at the last real token position are used to
    rank legal moves.

    Args:
        vocab_size (int): Number of tokens in the vocabulary (size of the
            embedding and output projection).
        d_model (int): Embedding / hidden dimension.
        n_heads (int): Number of attention heads per block.
        n_layers (int): Number of transformer blocks.
        d_ff (int): Inner dimension of the feed-forward sub-layer.
        ctx_len (int): Maximum sequence length (context window).
        dropout (float): Dropout probability used throughout the network.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, ctx_len, dropout):
        super().__init__()
        self.ctx = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """Initialise all linear and embedding weights with a standard normal (σ=0.02).

        The output projection head is excluded because its weights are shared
        with the token embedding and are therefore already initialised. Biases,
        where present, are zeroed.
        """
        for m in self.modules():
            if m is self.head:
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        """Run a forward pass and return logits over the vocabulary for every position.

        Args:
            x (torch.Tensor): Integer tensor of token IDs with shape ``(B, T)``
                where *T* ≤ ``ctx_len``.

        Returns:
            torch.Tensor: Logit tensor of shape ``(B, T, vocab_size)``.
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


def new_model(vocab_size):
    """Instantiate a :class:`ChesSLM` model from the current config and move it to the target device.

    This factory is the single source of truth for model construction and is
    used identically by both the training pipeline and the inference entry
    point, ensuring both always build the same architecture.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.

    Returns:
        ChesSLM: Newly created model on ``config.DEVICE``.
    """
    return ChesSLM(
        vocab_size,
        config.D_MODEL,
        config.N_HEADS,
        config.N_LAYERS,
        config.D_FF,
        config.CONTEXT_LEN,
        config.DROPOUT,
    ).to(config.DEVICE)
