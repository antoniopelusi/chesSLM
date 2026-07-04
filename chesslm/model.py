import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class RotaryEmbedding(nn.Module):
    """Precomputed rotary positional embedding (RoPE) cache.

    Replaces the learned absolute positional embedding table used in the
    original architecture. Rotary embeddings inject position by rotating
    query/key vectors in 2-D subspaces as a function of their sequence
    position, rather than adding a learned absolute positional vector to
    the token embedding. This generalises better to positions less densely
    represented in training (e.g. very long endgames) and encodes relative
    position directly into the attention dot product, which is a better
    inductive bias for a move sequence than absolute ply index.

    The cos/sin tables are precomputed for the maximum context length and
    cached as non-persistent buffers (deterministic and cheap to
    regenerate, so not saved in checkpoints).

    Args:
        d_head (int): Per-head dimension. Must be even.
        max_seq_len (int): Maximum sequence length to precompute rotations
            for; must be >= any sequence length seen at train or inference
            time.
        base (float): Base of the geometric progression of rotation
            frequencies, following the original RoPE paper's default.
    """

    def __init__(self, d_head, max_seq_len, base=10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, d_head / 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, d_head)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        """Return the cos/sin rotation tables for the first *seq_len* positions.

        Args:
            seq_len (int): Number of sequence positions needed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)``, each of shape
                ``(seq_len, d_head)``.
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x):
    """Rotate the last dimension of *x* by swapping and negating its two halves.

    Args:
        x (torch.Tensor): Tensor of shape ``(..., D)`` with even ``D``.

    Returns:
        torch.Tensor: Tensor of the same shape as *x*.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """Apply rotary positional embedding to a query or key tensor.

    Args:
        x (torch.Tensor): Query or key tensor of shape ``(B, n_heads, T, d_head)``.
        cos (torch.Tensor): Cosine table of shape ``(T, d_head)`` from
            :class:`RotaryEmbedding`.
        sin (torch.Tensor): Sine table of shape ``(T, d_head)`` from
            :class:`RotaryEmbedding`.

    Returns:
        torch.Tensor: Rotated tensor of the same shape as *x*.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and flash-attention kernel support.

    Projects the input into queries, keys, and values with a single fused
    linear layer, rotates queries and keys with rotary positional
    embeddings (see :class:`RotaryEmbedding`), then applies
    :func:`torch.nn.functional.scaled_dot_product_attention` with
    ``is_causal=True`` so each position can only attend to earlier
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
        self.proj.RESIDUAL_SCALE_INIT = True
        self.attn_drop = dropout

    def forward(self, x, cos, sin):
        """Compute causal multi-head self-attention for a sequence.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, C)`` where *B* is
                batch size, *T* is sequence length, and *C* is *d_model*.
            cos (torch.Tensor): RoPE cosine table of shape ``(T, d_head)``.
            sin (torch.Tensor): RoPE sine table of shape ``(T, d_head)``.

        Returns:
            torch.Tensor: Output tensor of the same shape ``(B, T, C)``.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
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
    ``d_ff / d_model`` (4x, the standard GPT ratio), applies GELU
    activation, and projects back to *d_model*.

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
        ff_out = nn.Linear(d_ff, d_model)
        ff_out.RESIDUAL_SCALE_INIT = True
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            ff_out,
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        """Apply the attention and feed-forward sub-layers with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, C)``.
            cos (torch.Tensor): RoPE cosine table, forwarded to the attention
                sub-layer.
            sin (torch.Tensor): RoPE sine table, forwarded to the attention
                sub-layer.

        Returns:
            torch.Tensor: Output tensor of the same shape ``(B, T, C)``.
        """
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


class ChesSLM(nn.Module):
    """Decoder-only transformer language model for chess move sequence prediction.

    The architecture follows the GPT family with rotary positional
    embeddings in place of learned absolute position embeddings: token
    embeddings are passed through a stack of :class:`Block` layers (each
    injecting position via RoPE inside its attention sub-layer), a final
    layer norm, and an unembedding head that shares weights with the token
    embedding matrix (weight tying).

    Sizing note: parameter count is dominated by the token embedding/output
    matrix (tied) and the ``N_LAYERS`` transformer blocks — see
    :func:`new_model` for the current configuration. Dropout, weight decay,
    and validation-based early stopping (see config.py and train.py) are the
    backstop against the memorisation risk that the epoch schedule's
    repetition carries.

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
        ctx_len (int): Maximum sequence length (context window); also the
            number of positions the RoPE cache is precomputed for.
        dropout (float): Dropout probability used throughout the network.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, ctx_len, dropout):
        super().__init__()
        self.ctx = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(d_model // n_heads, ctx_len)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """Initialise linear and embedding weights with a standard normal (σ=0.02).

        The output projection head is skipped because its weights are shared
        with the token embedding and are therefore already initialised.
        Layers that write directly onto the residual stream (attention
        output projection, FFN output projection — tagged with
        ``RESIDUAL_SCALE_INIT`` at construction time) get their std scaled
        by ``1 / sqrt(2 * n_layers)``, following GPT-2/nanoGPT practice: with
        ``n_layers`` unscaled residual additions per stream (attention +
        FFN), the residual stream's variance would otherwise grow with
        depth, which this compensates for at initialisation. Biases, where
        present, are zeroed.
        """
        for m in self.modules():
            if m is self.head:
                continue
            if isinstance(m, nn.Linear):
                std = 0.02
                if getattr(m, "RESIDUAL_SCALE_INIT", False):
                    std *= (2 * len(self.blocks)) ** -0.5
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        """Run a forward pass and return logits over the vocabulary for every position.

        Args:
            x (torch.Tensor): Integer tensor of token IDs with shape ``(B, T)``
                where *T* <= ``ctx_len``.

        Returns:
            torch.Tensor: Logit tensor of shape ``(B, T, vocab_size)``.
        """
        B, T = x.shape
        cos, sin = self.rotary(T)
        h = self.drop(self.tok_emb(x))
        for blk in self.blocks:
            h = blk(h, cos, sin)
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
