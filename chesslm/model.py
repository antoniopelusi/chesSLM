import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, HEAD_DIM

FF_MULT = 4
ROPE_BASE = 10000.0


def rope_tables(max_len):
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
    angles = torch.cat([freqs, freqs], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n_heads = d_model // HEAD_DIM
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def project(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = (
            t.view(B, T, self.n_heads, HEAD_DIM).transpose(1, 2)
            for t in self.qkv(x).split(C, dim=-1)
        )
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def merge(self, out):
        B, H, T, D = out.shape
        return self.proj(out.transpose(1, 2).reshape(B, T, H * D))

    def forward(self, x, cos, sin):
        q, k, v = self.project(x, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.merge(out)

    def step(self, x, cos, sin, past):
        q, k, v = self.project(x, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        T, S = q.size(2), k.size(2)
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.merge(out), (k, v)


class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, FF_MULT * d_model, bias=False)
        self.ff_out = nn.Linear(FF_MULT * d_model, d_model, bias=False)

    def ff(self, x):
        return self.ff_out(F.gelu(self.ff_in(x)))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ff(self.ln2(x))

    def step(self, x, cos, sin, past):
        out, kv = self.attn.step(self.ln1(x), cos, sin, past)
        x = x + out
        return x + self.ff(self.ln2(x)), kv


class ChesSLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, context_len):
        super().__init__()
        self.arch = dict(d_model=d_model, n_layers=n_layers, context_len=context_len)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        cos, sin = rope_tables(context_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        std = 0.02 / (2 * len(self.blocks)) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=std)
            nn.init.normal_(block.ff_out.weight, std=std)

    def forward(self, x):
        T = x.size(1)
        cos, sin = self.cos[:T], self.sin[:T]
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h, cos, sin)
        return F.linear(self.ln_f(h), self.tok_emb.weight)

    def step(self, x, past=None):
        offset = 0 if past is None else past[0][0].size(2)
        cos, sin = (
            self.cos[offset : offset + x.size(1)],
            self.sin[offset : offset + x.size(1)],
        )
        h = self.tok_emb(x)
        new_past = []
        for i, block in enumerate(self.blocks):
            h, kv = block.step(h, cos, sin, None if past is None else past[i])
            new_past.append(kv)
        return F.linear(self.ln_f(h), self.tok_emb.weight), new_past


def save_checkpoint(model, path):
    torch.save({"arch": model.arch, "state": model.state_dict()}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    vocab_size = checkpoint["state"]["tok_emb.weight"].size(0)
    model = ChesSLM(vocab_size, **checkpoint["arch"]).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    return model
